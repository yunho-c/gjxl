#!/usr/bin/env python3
"""Freeze and qualify the selected arithmetic policy without editing production."""
import argparse
import difflib
import json
from pathlib import Path
import shutil
import subprocess
from precision import ROOT, OUT, run, save, sha

def location(variant):
    return OUT if variant == 'ba_relaxed_precise' else OUT/'integrations'/variant

def label(variant, name):
    return name if variant == 'ba_relaxed_precise' else variant+'-'+name

def prepare(variant='ba_relaxed_precise'):
    base=location(variant)
    source=base/"candidate-src"
    if source.exists():raise RuntimeError("Candidate source already exists")
    baseline=json.loads((OUT/"identity.json").read_text())
    for name, expected in baseline["sources"].items():
        if name.startswith(("src/","include/","cmake/")) and sha(ROOT/name)!=expected:
            raise RuntimeError("Production source changed during study: "+name)
    shutil.copytree(OUT/"src",source,symlinks=True)
    # Include the retained benchmark targets in the integration build.
    original=(ROOT/"CMakeLists.txt").read_text()
    code=original
    if variant == 'ba_relaxed_precise':
        start=original.index('add_custom_command(\n  OUTPUT\n    "${GJXL_BUTTERAUGLI_METAL_IR}"')
        end=original.index('# .ir -> .metallib',start)
        block=original[start:end]
        assert block.count('-fmetal-math-mode=safe')==1
        assert block.count('    -ffp-contract=off\n')==1
        block=block.replace('-fmetal-math-mode=safe','-fmetal-math-mode=relaxed')
        block=block.replace('    -ffp-contract=off\n','')
        block=block.replace('Compiling strict-math Metal Butteraugli shaders','Compiling Metal Butteraugli shaders')
        code=original[:start]+block+original[end:]
    (source/'CMakeLists.txt').write_text(code)
    for name in ('metal_precision_probe.cpp','metal_precision_metric_probe.cpp'):
        shutil.copy2(ROOT/'benchmarks'/name,source/'benchmarks'/name)
    shader='src/gpu/metal/kernels/butteraugli.metal'
    before=(source/shader).read_text()
    after=(OUT/'variants'/variant/'kernels/butteraugli.metal').read_text()
    (source/shader).write_text(after)
    patch=''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),
        fromfile='a/CMakeLists.txt',tofile='b/CMakeLists.txt'))
    patch+=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
        fromfile='a/'+shader,tofile='b/'+shader))
    (base/'candidate.patch').write_text(patch)
    save(base/'candidate-source.json',dict(policy=variant,
        original_cmake_sha256=sha(ROOT/'CMakeLists.txt'),
        sources={str(p.relative_to(source)):sha(p) for base in ('src','include','cmake','benchmarks')
                 for p in (source/base).rglob('*') if p.is_file()},cmake_sha256=sha(source/'CMakeLists.txt')))
    run(label(variant,'candidate-configure'),['cmake','-S',source,'-B',base/'candidate-release','-G','Ninja',
        '-DCMAKE_BUILD_TYPE=Release','-DGJXL_BUILD_TESTS=ON','-DGJXL_BUILD_BENCHMARKS=ON',
        '-DGJXL_ENABLE_LIBJXL_REFERENCE=OFF','-DGJXL_ENABLE_METAL_PROFILING=OFF'])

def build(variant='ba_relaxed_precise'):
    base=location(variant)
    release=base/'candidate-release'
    run(label(variant,'candidate-build'),['cmake','--build',release,'-j8'])
    save(base/'candidate-artifacts.json',{str(p):sha(p) for p in
        [release/'gjxl_encode',release/'gjxl_encoding_benchmark',release/'metal/gjxl.metallib',
         release/'gjxl_metal_precision_probe',release/'gjxl_metal_precision_metric_probe']})
    shader_results={}
    ir_folder=base/'ir-comparison'
    ir_folder.mkdir(exist_ok=True)
    for p in (release/'metal').glob('*.ir'):
        expected=OUT/'variants'/variant/'metal'/p.name
        normalized=[]
        for side,input_path in [('screen',expected),('candidate',p)]:
            output=ir_folder/(p.stem+'-'+side+'.ll')
            run(label(variant,'ir-'+p.stem+'-'+side),['xcrun','metal-opt','-S',input_path,'-o',output])
            text=output.read_text()
            # AIR embeds source filenames even without profiling. Ignore only
            # the module identifier and these exact known source path strings.
            text=''.join(line for line in text.splitlines(True) if not line.startswith('; ModuleID = '))
            for source_folder in [OUT/'src/src/gpu/metal/kernels',base/'candidate-src/src/gpu/metal/kernels',
                                  OUT/'variants'/variant/'kernels']:
                text=text.replace(str(source_folder/(p.stem+'.metal')),'<source/'+p.stem+'.metal>')
            normalized_path=ir_folder/(p.stem+'-'+side+'.normalized.ll')
            normalized_path.write_text(text)
            normalized.append(sha(normalized_path))
        shader_results[p.name]=dict(candidate=sha(p),screened=sha(expected),
            binary_equal=sha(p)==sha(expected),normalized_ir=normalized,equal=normalized[0]==normalized[1])
    save(base/'candidate-shader-equivalence.json',shader_results)
    if not all(v['equal'] for v in shader_results.values()):
        raise RuntimeError('Candidate shader differs from qualified screen')
    for name, library in [('confirm-baseline',OUT/'baseline/metal/gjxl.metallib'),
                          ('confirm-candidate',release/'metal/gjxl.metallib')]:
        folder=base/name
        (folder/'metal').mkdir(parents=True,exist_ok=True)
        shutil.copy2(library,folder/'metal/gjxl.metallib')
        shutil.copy2(release/'gjxl_encoding_benchmark',folder/'gjxl_encoding_benchmark')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','build'))
    parser.add_argument('--variant',default='ba_relaxed_precise',
        choices=('ba_relaxed_precise','ba_malta_reassociate','ba_selective_reassociate',
                 'ba_malta_fastdivide','ba_malta_reciprocal'))
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.variant)
    else:build(args.variant)
