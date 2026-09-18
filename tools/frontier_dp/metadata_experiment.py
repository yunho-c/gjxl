#!/usr/bin/env python3
"""Freeze and validate the device AQ metadata builder, then time its submission."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

from run_experiment import ROOT, run, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--pipeline', action='store_true', help='Qualify the combined production handoff')
    args = parser.parse_args()
    if args.pipeline and not args.validate_only:
        parser.error('--pipeline requires --validate-only')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary, library = output / ('pipeline-test' if args.pipeline else 'metadata-test'), output / 'gjxl.metallib'
    test_name = 'gjxl_quantization_gpu_pipeline_test' if args.pipeline else 'gjxl_metal_aq_strategy_metadata_test'
    shutil.copy2(ROOT / 'build/frontier' / test_name, binary)
    shutil.copy2(ROOT / 'build/frontier/metal/gjxl.metallib', library)
    tracked = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    sources = sorted(set(tracked + untracked + ['tools/frontier_dp/metadata_experiment.py']))
    for name in sources:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    manifest = dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    files={str(p): sha(p) for p in (binary, library)},
                    sources={p: sha(ROOT / p) for p in sources}, commands=[], completed=False,
                    protocol='Normal and Metal-validation CPU-builder parity; 3 independent timing processes, '
                             'rotated size order, 3 warmups and 21 GPU timestamp samples per size; '
                             'same deterministic mixed cover; device metadata only, excludes selection and AQ')
    (output / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT))
    def save():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GJXL_', 'MTL_'))}
    base = [binary, '--metallib', library]
    for name, selected_env in [('normal', env), ('validation', {**env, 'MTL_DEBUG_LAYER': '1', 'MTL_SHADER_VALIDATION': '1'})]:
        manifest['commands'].append(run(base, output / f'{name}.log', selected_env))
        save()
    if args.validate_only:
        manifest['protocol'] = ('Normal and Metal-validation combined ACS/AQ pipeline parity, reuse and failure atomicity; no timing'
                                if args.pipeline else 'Normal and Metal-validation CPU-builder and indirect AQ consumer parity; no timing')
        manifest['checks'] = {name: (output / f'{name}.log').read_text()
                              for name in ('normal', 'validation')}
        manifest['completed'] = True
        save()
        print(json.dumps(manifest['checks'], indent=2))
        return
    rows = []
    for repetition in range(3):
        log = output / f'timing-{repetition}.jsonl'
        manifest['commands'].append(run([*base, '--benchmark', '--rotate', repetition], log, env))
        rows += [dict(json.loads(line), repetition=repetition) for line in log.read_text().splitlines() if line.startswith('{')]
        save()
    assert len(rows) == 9 and all(len(row['samples_ms']) == 21 for row in rows)
    summary = []
    for pixels in sorted({row['pixels'] for row in rows}):
        observations = [row for row in rows if row['pixels'] == pixels]
        medians = [statistics.median(row['samples_ms']) for row in observations]
        summary.append(dict(pixels=pixels, process_medians_ms=medians,
                            median_ms=statistics.median(medians), scratch_bytes=observations[0]['scratch_bytes']))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest['completed'] = True
    save()
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
