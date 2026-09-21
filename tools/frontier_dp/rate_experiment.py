#!/usr/bin/env python3
"""Resumable diagnostic rate-quality pilot; timings do not qualify GPU speed."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
CORPUS = BASE / 'libjxl-runtime-study-2026-09-03/corpus/pfm'
TOOLS = {
    'djxl': BASE / 'gjxl-libjxl-comparison/build/libjxl-comparison/libjxl/tools/djxl',
    'butteraugli': BASE / 'gjxl-libjxl-comparison/build/libjxl-comparison/libjxl/tools/butteraugli_main',
    'scorer': BASE / 'libjxl/tools/scripts/quality_metric/target/release/cjxl-quality-metric',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / 'frozen'
    frozen.mkdir()
    build = ROOT / 'build/frontier'
    executable = frozen / 'gjxl_quality_benchmark'
    shutil.copy2(build / executable.name, executable)
    source_files = ['CMakeLists.txt', 'src/gpu/ops/ac_strategy_search.cpp',
                    'src/gpu/ops/ac_strategy_capture_internal.h',
                    'src/codec/ac_strategy_search_policy.h',
                    'tools/frontier_dp/capture.cpp', 'tools/frontier_dp/graph.h',
                    'tools/frontier_dp/rate_experiment.py']
    for source in source_files:
        target = frozen / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    (frozen / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff'], cwd=ROOT))
    clic = sorted((CORPUS / 'clic2024_test').glob('*.pfm'))
    images = [CORPUS / f'kodak/{i:02}.pfm' for i in (1, 12, 24)]
    images += [clic[i] for i in (0, len(clic) // 2, len(clic) - 1)]
    files = [executable, build / 'metal/gjxl.metallib', *TOOLS.values(), *images]
    manifest = {
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'binary': str(executable), 'tools': {k: str(v) for k, v in TOOLS.items()},
        'files': {str(p): sha(p) for p in files},
        'sources': {p: sha(frozen / p) for p in source_files},
        'images': [{'id': str(p.relative_to(CORPUS).with_suffix('')), 'path': str(p)} for p in images],
        'selection': 'Kodak 1,12,24; first, middle, last lexically sorted CLIC PFM; chosen before observations',
        'distances': [.7, 1.2, 2., 4., 7.], 'efforts': [5, 8],
        'arms': ['greedy', 'frontier', 'rectangle'],
        'policy': 'Frozen FP32 additive costs; exact integer sums; retain greedy on ties or rectangle losses',
        'timing': 'Diagnostic CPU override and one sample; not a speed qualification',
        'expected': 180, 'created': time.time(), 'completed': False,
    }
    save(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = json.loads((output / 'manifest.json').read_text()) if output.exists() else freeze(output)
    for name, digest in manifest['files'].items():
        if sha(name) != digest:
            raise RuntimeError(f'Frozen input changed: {name}')
    ledger = output / 'observations.jsonl'
    rows = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    done = {r['id']: r for r in rows}
    for row in rows:
        if sha(row['output']) != row['output_sha256']:
            raise RuntimeError('Retained output changed')
    env = {k: v for k, v in os.environ.items() if not k.startswith('GJXL_')}

    def command(argv, folder, name, command_env=None):
        argv = [str(x) for x in argv]
        with (folder / 'commands.jsonl').open('a') as f:
            f.write(json.dumps({'argv': argv, 'time': time.time(),
                                'selector': (command_env or {}).get('GJXL_AC_SEARCH_EXPERIMENT')}) + '\n')
        result = subprocess.run(argv, cwd=ROOT, env=command_env or env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        (folder / (name + '.log')).write_text(result.stdout)
        result.check_returncode()
        return result.stdout

    for image in manifest['images']:
        with (output / 'scorer.stderr').open('a') as errors:
            scorer = subprocess.Popen([manifest['tools']['scorer'], image['path'], 'auto'],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=errors, text=True, bufsize=1, env=env)
            def response():
                if not select.select([scorer.stdout], [], [], 600)[0]:
                    raise RuntimeError('Scorer timeout')
                line = scorer.stdout.readline()
                if not line:
                    raise RuntimeError('Scorer exited')
                return json.loads(line)
            try:
                ready = response()
                for effort in manifest['efforts']:
                    for distance in manifest['distances']:
                        for arm in manifest['arms']:
                            key = f"{image['id']}/e{effort}/d{distance}/{arm}"
                            if key in done:
                                continue
                            if shutil.disk_usage(output).free < 8 * 1024**3:
                                raise RuntimeError('Free disk below 8 GiB floor')
                            folder = output / 'cases' / key
                            folder.mkdir(parents=True, exist_ok=True)
                            codestream, raw, decoded = [folder / n for n in ('output.jxl', 'raw.json', 'decoded.pfm')]
                            arm_env = {**env, 'GJXL_AC_SEARCH_EXPERIMENT': arm}
                            command([manifest['binary'], '--input', image['path'], '--output', codestream,
                                     '--raw-samples', raw, '--effort', effort, '--distance', distance,
                                     '--num-threads', 8, '--warmups', 0, '--samples', 1], folder, 'encode', arm_env)
                            timing = json.loads(raw.read_text())
                            assert timing['effort'] == effort and timing['thread_count'] == 8
                            assert timing['samples'][0]['encoded_bytes'] == codestream.stat().st_size
                            command([manifest['tools']['djxl'], codestream, decoded,
                                     '--color_space=RGB_D65_SRG_Rel_Lin', '--num_threads=1'], folder, 'decode')
                            scorer.stdin.write(json.dumps({'path': str(decoded)}) + '\n')
                            scorer.stdin.flush()
                            score = response()['score']
                            ba_text = command([manifest['tools']['butteraugli'], image['path'], decoded,
                                               '--colorspace', 'RGB_D65_SRG_Rel_Lin', '--intensity_target', 80],
                                              folder, 'butteraugli')
                            ba = float(ba_text.splitlines()[0])
                            assert math.isfinite(score) and math.isfinite(ba) and ba >= 0
                            row = {'id': key, 'image': image['id'], 'effort': effort, 'distance': distance,
                                   'arm': arm, 'ssimulacra2': score, 'butteraugli': ba,
                                   'bytes': codestream.stat().st_size, 'output': str(codestream),
                                   'output_sha256': sha(codestream), 'decoded_sha256': sha(decoded),
                                   'raw_sha256': sha(raw), 'metric_info': ready, 'time': time.time()}
                            with ledger.open('a') as f:
                                f.write(json.dumps(row) + '\n')
                                f.flush()
                                os.fsync(f.fileno())
                            decoded.unlink()  # Retain codestream and decoded hash, bound disk use.
                            done[key] = row
                            save(output / 'progress.json', {'completed': len(done), 'expected': manifest['expected'],
                                                             'pid': os.getpid(), 'time': time.time()})
                            print(len(done), '/', manifest['expected'], key, score, ba, row['bytes'], flush=True)
            finally:
                scorer.stdin.close()
                scorer.wait(timeout=30)
    assert len(done) == manifest['expected']
    manifest['completed'] = True
    manifest['finished'] = time.time()
    save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
