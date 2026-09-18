#!/usr/bin/env python3
"""Bounded, sequential captured-cost experiment. Never changes encoder defaults."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(command, log, env=None):
    started = time.time()
    with log.open('w') as f:
        subprocess.run([str(x) for x in command], cwd=ROOT, env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    return {'command': [str(x) for x in command], 'log': str(log),
            'started_unix': started, 'finished_unix': time.time()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/frontier')
    parser.add_argument('--corpus', type=Path, default=ROOT.parent / 'libjxl-runtime-study-2026-09-03/corpus/pfm')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    build = args.build.resolve()
    sources = ['CMakeLists.txt', 'src/gpu/ops/ac_strategy_search.cpp',
               'src/gpu/ops/ac_strategy_capture_internal.h',
               'src/codec/ac_strategy_search_policy.h', 'src/core/ac_strategy.h',
               'tools/frontier_dp/capture.cpp', 'tools/frontier_dp/graph.h',
               'tools/frontier_dp/frontier.metal', 'tools/frontier_dp/probe.cpp',
               'tools/frontier_dp/run_experiment.py']
    manifest = {
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sources': {p: sha(ROOT / p) for p in sources},
        'binaries': {p: sha(build / p) for p in ['gjxl_frontier_probe', 'gjxl_encoding_benchmark', 'frontier.metallib', 'metal/gjxl.metallib']},
        'protocol': {'process_repetitions': 3, 'samples_per_process': 7,
                     'warmups': 3, 'thread_counts': [64, 128, 256, 512],
                     'effort': 7, 'distance': 1.2,
                     'boundary': 'preparation, frontier DP, traceback and checked wide fallback dispatch; resident captured inputs; excludes scoring, upload, graph setup, AQ metadata and encoder'},
        'machine': {}, 'commands': [], 'cases': [],
    }
    for key, command in {
        'os': ['sw_vers'], 'cpu': ['sysctl', '-n', 'machdep.cpu.brand_string'],
        'memory': ['sysctl', '-n', 'hw.memsize'],
        'gpu': ['system_profiler', 'SPDisplaysDataType'],
        'compiler': ['clang++', '--version'],
        'initial_processes': ['ps', '-axo', 'pid,pcpu,comm'],
    }.items():
        manifest['machine'][key] = subprocess.check_output(command, text=True)
    (output / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT))
    source_dir = output / 'source'
    for p in sources:
        target = source_dir / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / p).read_bytes())

    def save():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    probe = build / 'gjxl_frontier_probe'
    capture = build / 'gjxl_encoding_benchmark'
    manifest['commands'].append(run([probe, '--self-test', '--out', output / 'selftest.json'], output / 'selftest.log'))
    env = os.environ.copy()
    env.update(MTL_DEBUG_LAYER='1', MTL_SHADER_VALIDATION='1')
    manifest['commands'].append(run([probe, '--self-test', '--out', output / 'selftest-metal-validation.json'], output / 'selftest-metal-validation.log', env))
    save()

    alpine = args.corpus / 'unsplash/alpine_lake/24mp.pfm'
    small = output / 'alpine-3mp.pfm'
    manifest['commands'].append(run(['magick', alpine, '-filter', 'Lanczos', '-resize',
        '2121x1414!', '-define', 'quantum:format=floating-point', '-depth', '32', small], output / 'resize.log'))
    cases = [('alpine_3mp', ['--input', small]),
             ('synthetic_4k', ['--workload', 'padded_4k']),
             ('alpine_24mp', ['--input', alpine]),
             ('forest_24mp', ['--input', args.corpus / 'unsplash/forest_stream/24mp.pfm']),
             ('campus_24mp', ['--input', args.corpus / 'unsplash/campus_interior/24mp.pfm'])]
    for name, selection in cases:
        directory = output / name
        directory.mkdir()
        env = os.environ.copy()
        env['GJXL_FRONTIER_CAPTURE_DIR'] = str(directory)
        command = [capture, *selection, '--scope', 'metal-public-workflow', '--implementation', 'simd',
                   '--ac-residual-inverse', 'fused-tuned', '--gpu-aq', 'fully-resident',
                   '--validation', 'metal-only', '--distance', '1.2', '--effort', '7',
                   '--warmups', '0', '--samples', '1']
        print('Capture', name, flush=True)
        manifest['commands'].append(run(command, directory / 'capture.log', env))
        captures = sorted(directory.glob('capture-*.gfc'))
        assert captures
        hashes = {p.name: sha(p) for p in captures}
        # The benchmark performs one validation and one measured encode. Costs
        # must remain identical, despite capture overhead invalidating encode times.
        assert len(set(hashes.values())) == 1, 'repeated captured costs changed'
        item = {'name': name, 'capture': str(captures[-1]), 'capture_sha256': hashes,
                'input': str(selection[1]), 'input_sha256': sha(selection[1]) if selection[0] == '--input' else None}
        manifest['cases'].append(item)
        save()
    # Independent processes, rotating image order between rounds; one GPU job
    # at a time. Compilation, reference checking and transfers precede timers.
    for repetition in range(3):
        order = manifest['cases'][repetition:] + manifest['cases'][:repetition]
        for item in order:
            directory = output / item['name']
            print('Measure', repetition, item['name'], flush=True)
            manifest['commands'].append(run([probe, '--capture', item['capture'], '--samples', '7', '--warmups', '3',
                '--out', directory / f'probe-{repetition}.json'], directory / f'probe-{repetition}.log'))
            save()
    summary = []
    for item in manifest['cases']:
        directory = output / item['name']
        rows = [json.loads((directory / f'probe-{i}.json').read_text()) for i in range(3)]
        assert all(x['validation'].startswith('passed') for x in rows)
        control = json.loads(Path(item['capture']).with_suffix('.json').read_text())
        timings = {}
        for mode, threads in sorted({(x['mode'], x['threads']) for x in rows[0]['samples']}):
            medians = [statistics.median(x['gpu_ms'] for x in row['samples'] if x['mode'] == mode and x['threads'] == threads) for row in rows]
            walls = [statistics.median(x['wall_ms'] for x in row['samples'] if x['mode'] == mode and x['threads'] == threads) for row in rows]
            timings[f'{mode}_{threads}'] = {'gpu_ms': statistics.median(medians), 'process_gpu_medians': medians,
                                          'wall_ms': statistics.median(walls), 'process_wall_medians': walls}
        summary.append({'name': item['name'], **{k: v for k, v in rows[0].items() if k not in ('samples', 'capture', 'host_oracle_and_setup_ms')},
                        'cpu_greedy_ms': statistics.median(control['cpu_greedy_ms']), 'timings': timings})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest['completed'] = True
    save()
    print(output / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
