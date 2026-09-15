#!/usr/bin/env python3
"""Extend the retained e8ff0976 precomputed-state bridge for the actual DC grid."""
from pathlib import Path
import shlex
import shutil
import subprocess

import study


def main():
    root=Path('build/e3-e4-tail-probe-20260914').resolve();root.mkdir(exist_ok=True)
    assert not (root/'build.json').exists()
    source=Path('/Users/yunhocho/GitHub/gjxl-runtime-main-e1-4-20260914')
    base=source/'build/quality-study'
    old=Path('/Users/yunhocho/GitHub/gjxl-e4-coefficient-rca-20260913')
    tail=Path('/Users/yunhocho/GitHub/gjxl-libjxl-tail')
    libsrc=tail/'third_party/libjxl';libbuild=tail/'build/libjxl-tail-main-refresh'
    frame=study.BASE/'e4-low-quality-rca-20260913/libjxl_enc_frame.cpp'
    build_record=study.BASE/'e4-low-quality-rca-20260913/build-commands.json'
    old_link_record=Path('/Users/yunhocho/GitHub/gjxl-writer-rate-attribution/build/writer-diagnostics/build-diagnostic.json')
    precision=Path('build/e3-e4-rate-20260914/precision-probe').resolve()
    inputs=[frame,build_record,old_link_record,old/'tools/e4_rca/tail.cpp',
        old/'src/codestream/libjxl_tail_internal.h',old/'src/codestream/writer_diagnostics_internal.h',
        libsrc/'lib/jxl/enc_precomputed_vardct.h',source/'src/codestream/encoder.cpp',
        Path('tools/low_effort_rate/tail_capture.cpp').resolve(),Path(__file__).resolve()]
    files={str(p):study.sha(p) for p in inputs}
    files.update(study.read(precision/'build.json')['files'])
    for p,h in files.items():assert study.sha(p)==h,p
    include=root/'include'
    for name in ['libjxl_tail_internal.h','writer_diagnostics_internal.h']:
        p=include/'codestream'/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(old/'src/codestream'/name,p)
    p=include/'lib/jxl/enc_precomputed_vardct.h';p.parent.mkdir(parents=True,exist_ok=True)
    text=(libsrc/'lib/jxl/enc_precomputed_vardct.h').read_text()
    assert text.count('  uint32_t quant_dc = 0;')==1
    p.write_text(text.replace('  uint32_t quant_dc = 0;','  uint32_t quant_dc = 0;\n  uint8_t extra_dc_precision = 0;'))
    text=(old/'tools/e4_rca/tail.cpp').read_text()
    assert text.count('    .quant_dc = quantizer.quant_dc,')==1
    text=text.replace('    .quant_dc = quantizer.quant_dc,','    .quant_dc = quantizer.quant_dc,\n    .extra_dc_precision = profile.extra_dc_precision,')
    (root/'tail.cpp').write_text(text)
    text=frame.read_text()
    marker='        &state->enc_state));\n  }\n  // qDC above is authoritative'
    assert text.count(marker)==1
    text=text.replace(marker,'        &state->enc_state));\n    state->modular->extra_dc_precision[group] = frame.extra_dc_precision;\n  }\n  // qDC above is authoritative')
    text=text.replace('  quantizer.AddU32(frame.quant_dc);','  quantizer.AddU32(frame.quant_dc);\n  quantizer.AddByte(frame.extra_dc_precision);')
    text=text.replace('  quantizer.AddU32(quantizer_params.quant_dc);','  quantizer.AddU32(quantizer_params.quant_dc);\n  quantizer.AddByte(state.modular->extra_dc_precision[0]);')
    (root/'libjxl_enc_frame.cpp').write_text(text)
    text=(source/'src/codestream/encoder.cpp').read_text()
    marker='namespace gjxl {\n'
    assert text.count(marker)==1
    text=text.replace(marker,marker+'Status CaptureRcaTail(const vardct_frame_internal::VarDctFrameView&);\n')
    marker='    *output = std::move(candidate_output);'
    assert text.count(marker)==1
    text=text.replace(marker,'    const auto tail_status = CaptureRcaTail(frame);\n    if (!tail_status.ok()) return tail_status;\n'+marker)
    (root/'encoder.cpp').write_text(text)
    shutil.copy2(Path('tools/low_effort_rate/tail_capture.cpp'),root/'capture.cpp')
    commands=[]
    libcmd=study.read(build_record)[0]['argv'][:]
    libcmd.insert(1,'-I'+str(include))
    for flag,suffix in [('-MT','.o'),('-MF','.o.d'),('-o','.o'),('-c','.cpp')]:
        libcmd[libcmd.index(flag)+1]=str(root/('libjxl_enc_frame'+suffix))
    commands.append(libcmd)
    for name in ['encoder','tail','capture']:
        commands.append(['/usr/bin/c++','-O3','-DNDEBUG','-std=gnu++20','-arch','arm64',
            '-I'+str(include),'-I'+str(source/'src'),'-I'+str(libsrc),
            '-I'+str(libbuild/'third_party/libjxl-tail/lib/include'),
            '-c',str(root/f'{name}.cpp'),'-o',str(root/f'{name}.o')])
    link=shlex.split((base/'CMakeFiles/gjxl_quality_benchmark.dir/link.txt').read_text())
    for i,arg in enumerate(link):
        if arg.endswith(('.a','.o')):link[i]=str(base/arg)
    link[link.index('-o')+1]=str(root/'gjxl_tail_probe')
    at=link.index(str(base/'libgjxl_codestream.a'))
    link[at:at]=[str(precision/'workflow.cpp.o')]+[str(root/f'{n}.o') for n in ['encoder','tail','capture','libjxl_enc_frame']]
    for arg in study.read(old_link_record)['link']:
        if arg.startswith('third_party/') and arg.endswith('.a'):
            p=libbuild/arg;link.append(str(p));files[str(p)]=study.sha(p)
    commands.append(link)
    for argv in commands:
        result=subprocess.run(argv,cwd=libbuild,capture_output=True,text=True)
        study.append(root/'build-commands.jsonl',{'argv':argv,'cwd':str(libbuild),
            'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        result.check_returncode()
    for p in root.rglob('*'):
        if p.is_file() and p.name!='build-commands.jsonl':files[str(p)]=study.sha(p)
    for p in libsrc.rglob('*.h'):files[str(p)]=study.sha(p)
    study.save(root/'build.json',{'files':files,'gjxl_revision':study.read(precision/'build.json')['base_revision'],
        'libjxl_base_revision':'e8ff09762481785938d8e4e01333ed3917571161',
        'libjxl_bridge_revision':'466ee1be396d3c850d6813ceb04f68701914b121',
        'extension':'Carry extra DC precision independently, set emitted group precision, and include precision in source/copied quantizer digest. Original qDC and reconstructed DC remain authoritative.',
        'qualification':'Pending native byte identity and exact external decoded-PFM identity on every case; bridge failure invalidates any rate attribution.'})
    print(root/'gjxl_tail_probe')


if __name__=='__main__':main()
