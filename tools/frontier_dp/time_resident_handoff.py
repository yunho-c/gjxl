#!/usr/bin/env python3
"""Compare CPU selection, synchronous GPU handoff, and combined resident ACS/AQ."""
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
    parser.add_argument('--cpu-control-binary', type=Path, required=True)
    parser.add_argument('--gpu-control-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binaries = {name: output / name for name in ('cpu', 'handoff', 'combined')}
    shutil.copy2(args.cpu_control_binary, binaries['cpu'])
    shutil.copy2(args.gpu_control_binary, binaries['handoff'])
    shutil.copy2(ROOT / 'build/frontier/gjxl_quality_benchmark', binaries['combined'])
    corpus = ROOT.parent / 'libjxl-runtime-study-2026-09-03/corpus/pfm/unsplash'
    paths = {'alpine_3mp': ROOT / 'build/frontier/results/kernel-probe-20260917/alpine-3mp.pfm',
             'alpine_24mp': corpus / 'alpine_lake/24mp.pfm',
             'forest_24mp': corpus / 'forest_stream/24mp.pfm'}
    sources = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines()
    sources += ['tools/frontier_dp/time_resident_handoff.py']
    for name in sources:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    manifest = dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    files={str(p): sha(p) for p in [*binaries.values(), *paths.values()]},
                    sources={p: sha(ROOT / p) for p in sources}, commands=[], completed=False,
                    cpu_control_binary=str(args.cpu_control_binary.resolve()),
                    gpu_control_binary=str(args.gpu_control_binary.resolve()),
                    protocol='Complete-call wall time; efforts 5/8 at distance 1.2; 8 threads; '
                             '3 processes, 3 warmups, 7 samples per arm/case; rotated case/arm order; '
                             'CPU control, GPU selector with host handoff and fused initialization, combined resident ACS/AQ; '
                             'byte identity required; no profiling')
    (output / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT))
    def save():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    env = {k: v for k, v in os.environ.items() if not k.startswith('GJXL_')}
    arms = ['cpu', 'handoff', 'combined']
    checks = {}
    cases = [(name, effort) for name in paths for effort in (5, 8)]
    for repetition in range(3):
        for index, (name, effort) in enumerate(cases[repetition:] + cases[:repetition]):
            shift = (repetition + index) % 3
            for arm in arms[shift:] + arms[:shift]:
                folder = output / f'{name}-e{effort}-{arm}-{repetition}'
                folder.mkdir()
                selected_env = {**env, 'GJXL_AC_SEARCH_EXPERIMENT': 'greedy'} if arm == 'cpu' else env
                binary = binaries[arm]
                argv = [binary, '--input', paths[name], '--output', folder / 'output.jxl',
                        '--raw-samples', folder / 'raw.json', '--effort', effort, '--distance', 1.2,
                        '--num-threads', 8, '--warmups', 3, '--samples', 7]
                record = run(argv, folder / 'encode.log', selected_env)
                record.update(arm=arm, image=name, effort=effort, repetition=repetition)
                manifest['commands'].append(record)
                digest = sha(folder / 'output.jxl')
                key = (name, effort)
                if key in checks and checks[key] != digest:
                    raise RuntimeError(f'Output identity changed: {key}')
                checks[key] = digest
                save()
                print('Measured', name, effort, arm, repetition, flush=True)
    summary = []
    for name, effort in cases:
        item = dict(image=name, effort=effort, output_sha256=checks[(name, effort)])
        for arm in arms:
            medians = []
            for repetition in range(3):
                data = json.loads((output / f'{name}-e{effort}-{arm}-{repetition}/raw.json').read_text())
                assert data['timing_semantics'] == 'complete-encode-wall-time'
                assert not data['stage_profile_enabled'] and data['sample_count'] == 7
                medians.append(statistics.median(x['elapsed_nanoseconds'] / 1e6 for x in data['samples']))
            item[arm] = dict(ms=statistics.median(medians), process_medians=medians)
        item['combined_speedup'] = item['cpu']['ms'] / item['combined']['ms']
        item['handoff_speedup'] = item['handoff']['ms'] / item['combined']['ms']
        summary.append(item)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest['completed'] = True
    save()


if __name__ == '__main__':
    main()
