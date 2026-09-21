#!/usr/bin/env python3
"""Repeat selector kernels on already captured, hashed candidate costs."""
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
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.captures.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    build = ROOT / 'build/frontier'
    probe = build / 'gjxl_frontier_probe'
    files = [ROOT / 'tools/frontier_dp' / f for f in
             ('graph.h', 'probe.cpp', 'frontier.metal', 'compare_selectors.py', 'run_experiment.py')]
    files += [probe, build / 'frontier.metallib', source / 'manifest.json']
    cases = sorted(source.glob('*/capture-0.gfc'))
    files += cases
    files += [p.with_suffix('.json') for p in cases]
    manifest = dict(files={str(p): sha(p) for p in files},
                    revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    commands=[], completed=False,
                    boundary='Resident captured inputs; preparation, selection and required wide fallback; '
                             'excludes scoring, host upload, graph setup, AQ metadata and encoder. '
                             'CPU timings are the earlier capture controls, not concurrent paired samples.',
                    repetitions=3, samples=7, warmups=3)
    frozen = output / 'frozen'
    frozen.mkdir()
    for p in files[:7]:
        shutil.copy2(p, frozen / p.name)
    (frozen / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff'], cwd=ROOT))
    def save():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    def execute(argv, log, env=None):
        manifest['commands'].append(run(argv, log, env))
        save()
    save()
    execute([probe, '--self-test', '--out', output / 'selftest.json'], output / 'selftest.log')
    execute([probe, '--self-test', '--out', output / 'selftest-validation.json'], output / 'selftest-validation.log',
            {**os.environ, 'MTL_DEBUG_LAYER': '1', 'MTL_SHADER_VALIDATION': '1'})
    execute([probe, '--self-test-discount', '--out', output / 'discount-selftest.json'], output / 'discount-selftest.log')
    execute([probe, '--self-test-discount', '--out', output / 'discount-selftest-validation.json'],
            output / 'discount-selftest-validation.log',
            {**os.environ, 'MTL_DEBUG_LAYER': '1', 'MTL_SHADER_VALIDATION': '1'})
    for repetition in range(3):
        for capture in cases[repetition:] + cases[:repetition]:
            name = capture.parent.name
            execute([probe, '--capture', capture, '--samples', 7, '--warmups', 3,
                     '--out', output / f'{name}-{repetition}.json'], output / f'{name}-{repetition}.log')
            print('Measured', name, repetition, flush=True)
    summary = []
    for capture in cases:
        rows = [json.loads((output / f'{capture.parent.name}-{i}.json').read_text()) for i in range(3)]
        controls = json.loads(capture.with_suffix('.json').read_text())
        item = {k: v for k, v in rows[0].items() if k not in ('samples', 'host_oracle_and_setup_ms', 'capture')}
        item.update(name=capture.parent.name, cpu_greedy_ms=statistics.median(controls['cpu_greedy_ms']), timings={})
        for mode in ('complete', 'rectangle', 'greedy', 'rectangle_greedy'):
            for threads in (64, 128, 256, 512):
                medians = [statistics.median(x['gpu_ms'] for x in row['samples']
                                            if x['mode'] == mode and x['threads'] == threads) for row in rows]
                walls = [statistics.median(x['wall_ms'] for x in row['samples']
                                          if x['mode'] == mode and x['threads'] == threads) for row in rows]
                item['timings'][f'{mode}_{threads}'] = dict(gpu_ms=statistics.median(medians),
                    process_gpu_medians=medians, wall_ms=statistics.median(walls), process_wall_medians=walls)
        summary.append(item)
    for name, digest in manifest['files'].items():
        if sha(name) != digest:
            raise RuntimeError(f'Input changed during measurement: {name}')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest['completed'] = True
    save()


if __name__ == '__main__':
    main()
