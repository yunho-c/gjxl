#!/usr/bin/env python3
"""Qualify the production e8 writer against frozen baseline/prototype outputs."""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-tools', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--prototype', type=Path, required=True)
    p.add_argument('--libjxl', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--budget-seconds', type=int, default=1800)
    args = p.parse_args()
    sys.path.insert(0, str(args.study_tools.resolve()))
    import cjxl_quality_characterization as study
    import cjxl_bd_rate as bd
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = study.load_config(args.baseline)
        libconfig = study.load_config(args.libjxl)
        prototype = json.loads((args.prototype / 'manifest.json').read_text())
        assert config['metric_version'] == libconfig['metric_version'] == prototype['metric_version']
        baseline = {(r['image_id'], r['distance']): r for r in
                    study.ledger(args.baseline / 'timings.jsonl') if r['repetition'] == 0}
        controls = study.ledger(args.prototype / 'results.jsonl')
        expected_count = len(prototype['images']) * len(prototype['distances'])
        assert len(controls) == expected_count
        baseline_scores = {(r['image_id'], r['distance']): r for r in
                           study.read_records(args.baseline, 'scores', config) if r['effort'] == 8}
        files = [Path(__file__), args.binary, args.binary.parent / 'build.json',
                 args.baseline / 'metadata.json', args.baseline / 'timings.jsonl',
                 args.baseline / 'scores.jsonl', args.prototype / 'manifest.json',
                 args.prototype / 'results.jsonl', args.libjxl / 'metadata.json',
                 args.libjxl / 'scores.jsonl', args.study_tools / 'cjxl_bd_rate.py',
                 args.study_tools / 'cjxl_quality_characterization.py',
                 args.study_tools / 'cjxl_sweep_common.py']
        hashes = {str(f.resolve()): sha(f) for f in files}
        hashes.update(config['tool_hashes'])
        manifest = {'candidate': 'production-effort8-rate-optimized-with-balanced-fallback',
                    'images': prototype['images'], 'distances': prototype['distances'],
                    'hashes': hashes, 'metric_version': config['metric_version'],
                    'timing_scope': 'diagnostic only; rate qualification, no latency claim'}
        mp = run / 'manifest.json'
        if mp.exists() and json.loads(mp.read_text()) != manifest:
            raise ValueError('Frozen integration study changed')
        for path, digest in hashes.items():
            study.verify_file(path, digest)
        write(mp, manifest)
        done = {r['key']: r for r in study.ledger(run / 'results.jsonl')}
        for row in done.values():
            study.verify_file(row['output_path'], row['output_sha256'])
        budget = study.Budget(args.budget_seconds)
        start = time.monotonic()
        for im in prototype['images']:
            study.verify_file(im['pfm_path'], im['pfm_sha256'])
            with study.Scorer(config, im, run, budget) as scorer:
                for control in (r for r in controls if r['image_id'] == im['image_id']):
                    distance = control['distance']
                    key = f"{im['image_id']}|d={distance}"
                    old = baseline[im['image_id'], distance]
                    old_score = baseline_scores[im['image_id'], distance]
                    chosen = old if old['encoded_bytes'] <= control['encoded_bytes'] else control
                    study.verify_file(control['output_path'], control['output_sha256'])
                    study.verify_file(old['output_path'], old['output_sha256'])
                    if key in done:
                        continue
                    budget.check()
                    case = run / 'cases' / im['image_id'] / str(distance)
                    case.mkdir(parents=True, exist_ok=True)
                    output, raw = case / 'output.jxl', case / 'raw.json'
                    command = list(map(str, [args.binary.resolve(), '--input', im['pfm_path'],
                        '--output', output, '--raw-samples', raw, '--distance', distance,
                        '--effort', 8, '--num-threads', 8, '--warmups', 0, '--samples', 1]))
                    study.append(run / 'commands.jsonl', {'key': key, 'command': command})
                    env = {k: v for k, v in os.environ.items() if not k.startswith('GJXL_E8_')}
                    with (case / 'stderr.txt').open('w+') as err:
                        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err,
                                                 text=True, start_new_session=True, env=env)
                        try:
                            stdout, _ = child.communicate(timeout=budget.timeout(1800))
                        except BaseException:
                            study.stop_process(child)
                            raise
                        (case / 'stdout.txt').write_text(stdout)
                        err.seek(0)
                        if child.returncode:
                            raise RuntimeError(err.read()[-4000:])
                    digest = sha(output)
                    if digest != chosen['output_sha256']:
                        raise ValueError('Integrated output did not replay the smaller frozen encoding: ' + key)
                    score = scorer.score(output)
                    if score['decoded_sha256'] != old_score['decoded_sha256']:
                        raise ValueError('Decoded pixels changed: ' + key)
                    row = {'key': key, 'image_id': im['image_id'], 'distance': distance,
                           'encoded_bytes': output.stat().st_size, 'output_path': str(output),
                           'output_sha256': digest, 'raw_sha256': sha(raw),
                           'baseline_bytes': old['encoded_bytes'], 'prototype_bytes': control['encoded_bytes'],
                           'selected_balanced': chosen is old, 'exact_pixels': True, **score}
                    study.append(run / 'results.jsonl', row)
                    done[key] = row
                    print(key, row['encoded_bytes'], 'balanced' if chosen is old else 'expanded', flush=True)
                    write(run / 'status.json', {'state': 'running', 'completed': len(done),
                                               'expected': expected_count})
        grouped = defaultdict(list)
        for row in done.values():
            grouped[row['image_id'], 'integrated'].append(row)
        for row in baseline_scores.values():
            grouped[row['image_id'], 'baseline'].append(row)
        images = {im['image_id']: im for im in prototype['images']}
        for row in study.read_records(args.libjxl, 'scores', libconfig):
            if row['effort'] == 8 and row['resampling'] == 1 and row['image_id'] in images:
                assert row['reference_sha256'] == images[row['image_id']]['pfm_sha256']
                grouped[row['image_id'], 'libjxl'].append(row)
        def curve(name, variant):
            rows = sorted(grouped[name, variant], key=lambda r: -r['distance'])
            return [r['score'] for r in rows], [r['encoded_bytes'] for r in rows]
        rates = []
        for name in images:
            for anchor in ('baseline', 'libjxl'):
                rates.append({'image_id': name, 'anchor': anchor, **{
                    method: bd.bd_rate(*curve(name, anchor), *curve(name, 'integrated'), method=method)
                    for method in ('pchip', 'akima')}})
        summary = {anchor: {method: statistics.mean(r[method] for r in rates if r['anchor'] == anchor)
                           for method in ('pchip', 'akima')} for anchor in ('baseline', 'libjxl')}
        report = {'quality_range': [75, 85], 'image_count': len(images), 'points': len(done),
                  'exact_pixels': sum(r['exact_pixels'] for r in done.values()),
                  'selected_balanced': sum(r['selected_balanced'] for r in done.values()),
                  'summary': summary, 'rates': rates, 'manifest_sha256': sha(mp),
                  'results_sha256': sha(run / 'results.jsonl')}
        write(run / 'analysis.json', report)
        write(run / 'status.json', {'state': 'complete', 'completed': len(done),
              'expected': expected_count, 'elapsed_seconds': time.monotonic() - start})
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
