#!/usr/bin/env python3
"""Build a diagnostic workflow object against the plotted, immutable libraries.

The only intervention independently sets extra_dc_precision (0..3). With
the variable absent, the original workflow policy and shader are preserved.
"""
import json
from pathlib import Path
import shlex
import subprocess

import study


def main():
    source = Path('/Users/yunhocho/GitHub/gjxl-runtime-main-e1-4-20260914')
    build = source / 'build/quality-study'
    output = Path('build/e3-e4-rate-20260914/precision-probe').resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not subprocess.check_output(['git', '-C', source, 'status', '--porcelain'], text=True).strip()
    revision = subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip()
    manifest = study.read(output.parent / 'manifest.json')
    assert revision == manifest['revision']
    assert study.sha(build / 'gjxl_quality_benchmark') == study.sha(manifest['binary'])
    text = (source / 'src/codestream/workflow.cpp').read_text()
    target = '''  pipeline_options.adaptive_quantization.profile.extra_dc_precision =
    ResolveDcQuantization(options) == DcQuantizationMode::kPredictionAware ? 1 : 0;
'''
    replacement = target + '''  // Diagnostic only: independently vary the DC grid precision.
  if (const char* value = std::getenv("GJXL_RCA_DC_PRECISION")) {
    if (value[0] < '0' || value[0] > '3' || value[1] != '\\0') {
      return Status::InvalidArgument("GJXL_RCA_DC_PRECISION must be 0..3");
    }
    pipeline_options.adaptive_quantization.profile.extra_dc_precision =
      static_cast<uint8_t>(value[0] - '0');
  }
'''
    assert text.count(target) == 1
    text = text.replace('#include <cstddef>', '#include <cstddef>\n#include <cstdlib>')
    text = text.replace(target, replacement)
    overlay = output / 'workflow.cpp'
    overlay.write_text(text)
    obj = output / 'workflow.cpp.o'
    commands = [['/usr/bin/c++', '-O3', '-DNDEBUG', '-std=gnu++20', '-arch', 'arm64',
                 '-I' + str(source / 'src'), '-c', str(overlay), '-o', str(obj)]]
    original_link = shlex.split((build / 'CMakeFiles/gjxl_quality_benchmark.dir/link.txt').read_text())
    link = list(original_link)
    for n, arg in enumerate(link):
        if arg.endswith(('.o', '.a')):
            link[n] = str(build / arg)
    link[link.index('-o') + 1] = str(output / 'gjxl_precision_probe')
    # Supply every workflow symbol before archive scanning. The untouched
    # archive's workflow object is consequently not linked.
    link.insert(link.index(str(build / 'libgjxl_codestream.a')), str(obj))
    commands.append(link)
    for argv in commands:
        result = subprocess.run(argv, capture_output=True, text=True)
        study.append(output / 'build-commands.jsonl', {'argv': argv,
                     'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        result.check_returncode()
    files = {Path(p) for p in link if p.endswith(('.o', '.a'))}
    files |= {p for p in (source / 'src').rglob('*') if p.is_file()}
    files |= {overlay, output / 'gjxl_precision_probe', Path(__file__).resolve(),
              build / 'CMakeFiles/gjxl_codestream.dir/flags.make',
              build / 'CMakeFiles/gjxl_quality_benchmark.dir/link.txt'}
    study.save(output / 'build.json', {'kind': 'diagnostic-workflow-object-overlay',
        'base_revision': revision, 'base_binary_sha256': study.sha(manifest['binary']),
        'variable': 'GJXL_RCA_DC_PRECISION', 'allowed_values': [0,1,2,3],
        'qualification': 'pending: disabled byte parity and intervention controls required',
        'files': {str(p): study.sha(p) for p in sorted(files)}})
    print(output / 'gjxl_precision_probe')


if __name__ == '__main__':
    main()
