#!/usr/bin/env python3
"""Explicit, resumable e5 refinement quality qualification; no timing claims.

Freeze two separately built source trees, then run rate, cross-metric, and
saved-data analysis phases. Calibration uses persistent, bounded probe budgets.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import statistics
import struct
import subprocess
import time

import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator

DISTANCES = [4.6, 2.8, 1.9, 1.0, 0.55]
SELECTED = [
    'kodak/01', 'kodak/08', 'kodak/13', 'kodak/23',
    'clic2024_test/097cb426910ba8ce2525dd8bb7fb1777',
    'clic2024_test/100a02c269c5948392f283b2aa3bb4da',
    'clic2024_test/28d24b9c83de066597ff96a68769884f',
    'clic2024_test/d1a9be98d1936065967adac50a6fb750',
    'unsplash/alpine_lake/24mp', 'unsplash/campus_interior/12mp',
    'unsplash/campus_interior/48mp', 'unsplash/forest_stream/48mp',
]
PROTOCOL = {
    'distances': DISTANCES, 'effort': 5, 'threads': 8,
    'interval': [75, 85], 'target': 80, 'tolerance': 0.025,
    'pair_tolerance': 0.05, 'max_calibration_probes': 14,
    'bounds': [0.05, 9.5], 'free_disk_floor': 5 * 1024**3,
    'timing_status': 'deferred until AC power is available',
    'collection': 'Two deterministic complete encodes per probe. Incidental wall '
                  'samples are diagnostic only and excluded from analysis.',
    'selection': 'All 65 original photographic inputs for rate curves; '
                 '12 diagnostic images for matched SSIMULACRA2 and Butteraugli, '
                 'including the historical worst mixed-transform size regression. '
                 'Selected before new measurements; not an unseen population sample.',
    'aggregation': 'Arithmetic mean of per-image BD percentages, PCHIP primary; '
                   'Akima sensitivity; no extrapolation or monotonicity repair. '
                   'Cross-metric percentages also use equal-image arithmetic means.',
    'adoption': 'Correctness is mandatory. Rate and perceptual results describe '
                'the tradeoff; no final adoption recommendation without fresh '
                'controlled timing qualification. No paper throughput refresh.',
}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def append(path, data):
    with Path(path).open('a') as f:
        f.write(json.dumps(data, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def key(*values):
    return hashlib.sha256('|'.join(map(str, values)).encode()).hexdigest()[:24]


def freeze(root, metadata):
    check(not (root / 'manifest.json').exists(), 'Manifest already exists')
    old = read(metadata)
    check(len(old['images']) == 65, 'Expected the original 65-image corpus')
    check(set(SELECTED) <= {i['image_id'] for i in old['images']}, 'Missing selected input')
    files = {str(metadata): sha(metadata), str(Path(__file__).resolve()): sha(__file__)}
    for image in old['images']:
        check(sha(image['pfm_path']) == image['pfm_sha256'], image['pfm_path'])
        files[image['pfm_path']] = image['pfm_sha256']
    for path, digest in old['tool_hashes'].items():
        if path != old['benchmark']:
            check(sha(path) == digest, f'Pinned external tool changed: {path}')
            files[path] = digest
    butteraugli = str(Path(old['djxl']).with_name('butteraugli_main'))
    files[butteraugli] = sha(butteraugli)
    binaries = {}
    for arm in ('baseline', 'candidate'):
        binary = root / f'{arm}-build/gjxl_quality_benchmark'
        binaries[arm] = str(binary)
        for p in [binary, binary.with_name('gjxl_encode'), *binary.parent.rglob('*.metallib')]:
            files[str(p)] = sha(p)
        for p in (root / f'{arm}-source').rglob('*'):
            if p.is_file():
                files[str(p)] = sha(p)
    for p in [root / 'source.patch', root / 'source-identity.json']:
        files[str(p)] = sha(p)
    save(root / 'manifest.json', {
        'protocol': PROTOCOL, 'images': old['images'], 'selected': SELECTED,
        'files': files, **binaries, 'djxl': old['djxl'], 'scorer': old['scorer'],
        'metric_version': old['metric_version'], 'butteraugli': butteraugli,
        'source': read(root / 'source-identity.json'),
        'system': subprocess.check_output(['uname', '-a'], text=True),
        'hardware': subprocess.check_output(['sysctl', '-n', 'hw.model', 'hw.memsize', 'hw.ncpu', 'machdep.cpu.brand_string'], text=True),
        'power': subprocess.check_output(['pmset', '-g', 'batt'], text=True),
        'created': time.time(), 'numpy': np.__version__,
    })


class Study:
    def __init__(self, root):
        self.root, self.m = root, read(root / 'manifest.json')
        check(self.m['protocol'] == PROTOCOL, 'Protocol changed; use a new study')
        for path, digest in self.m['files'].items():
            check(sha(path) == digest, f'Frozen file changed: {path}')
        self.data = {r['id']: r for r in rows(root / 'observations.jsonl')}
        check(len(self.data) == len(rows(root / 'observations.jsonl')), 'Duplicate observation IDs')
        for row in self.data.values():
            check(sha(row['output_path']) == row['output_sha256'], row['output_path'])
            check(sha(row['raw_path']) == row['raw_sha256'], row['raw_path'])
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('GJXL_', 'RCA_'))}
        self.env['RAYON_NUM_THREADS'] = '8'
        self.scorer = self.image = self.err = None

    def command(self, argv):
        check(shutil.disk_usage(self.root).free > PROTOCOL['free_disk_floor'], 'Free disk floor reached')
        argv = list(map(str, argv))
        start = time.time()
        r = subprocess.run(argv, env=self.env, capture_output=True, text=True, timeout=600)
        append(self.root / 'commands.jsonl', dict(argv=argv, started=start,
               seconds=time.time()-start, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr))
        r.check_returncode()
        return r.stdout

    def response(self):
        check(bool(select.select([self.scorer.stdout], [], [], 600)[0]), 'Scorer timeout')
        value = json.loads(self.scorer.stdout.readline())
        check('error' not in value, str(value))
        return value

    def close(self):
        if self.scorer:
            self.scorer.stdin.close()
            self.scorer.wait(timeout=30)
            self.err.close()
            self.scorer = self.image = self.err = None

    def score(self, image, decoded):
        if self.image != image['image_id']:
            self.close()
            self.err = (self.root / 'scorer.stderr').open('a')
            self.scorer = subprocess.Popen([self.m['scorer'], image['pfm_path'], 'auto'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                text=True, bufsize=1, env=self.env)
            ready = self.response()
            check(ready['ready'] and ready['fast_ssim2_revision'] == self.m['metric_version']['fast_ssim2_revision'], 'Wrong scorer')
            check((ready['width'], ready['height']) == (image['width'], image['height']), 'Scorer extent mismatch')
            self.image = image['image_id']
        self.scorer.stdin.write(json.dumps({'path': str(decoded)}) + '\n')
        self.scorer.stdin.flush()
        score = self.response()['score']
        check(math.isfinite(score), 'Nonfinite score')
        return score

    def decode(self, output, folder):
        decoded = folder / 'decoded.pfm'
        self.command([self.m['djxl'], output, decoded,
                      '--color_space=RGB_D65_SRG_Rel_Lin', '--num_threads=1'])
        return decoded

    def probe(self, image, arm, distance, kind):
        distance = f32(distance)
        identity = key(image['image_id'], arm, distance.hex())
        if identity in self.data:
            return self.data[identity]
        folder = self.root / 'cases' / identity
        folder.mkdir(parents=True, exist_ok=True)
        output, raw = folder / 'out.jxl', folder / 'raw.json'
        self.command([self.m[arm], '--input', image['pfm_path'], '--output', output,
                      '--raw-samples', raw, '--effort', 5, '--distance', format(distance, '.9g'),
                      '--num-threads', 8, '--warmups', 0, '--samples', 1])
        record = read(raw)
        check(record['backend'] == 'metal' and record['gpu_aq_mode'] == 'fully-resident' and
              record['effort'] == 5 and record['thread_count'] == 8 and
              record['density'] == 'default' and record['compression'] == 'automatic' and
              not record['collect_final_score'] and not record['stage_profile_enabled'] and
              record['dc_quantization'] == 'prediction-aware' and record['adaptive_dc_smoothing'] and
              record['resampling'] == 1 and record['validation_encodes'] == record['sample_count'] == 1 and
              record['samples'][0]['encoded_bytes'] == output.stat().st_size and
              (record['input_width'], record['input_height']) == (image['width'], image['height']),
              'Unexpected encoder configuration')
        decoded = self.decode(output, folder)
        score = self.score(image, decoded)  # Scorer rejects every nonfinite input/output sample.
        row = dict(id=identity, image_id=image['image_id'], arm=arm, distance=distance,
                   score=score, encoded_bytes=output.stat().st_size, output_path=str(output),
                   output_sha256=sha(output), decoded_sha256=sha(decoded),
                   raw_path=str(raw), raw_sha256=sha(raw), kind=kind, time=time.time())
        append(self.root / 'observations.jsonl', row)
        self.data[identity] = row
        decoded.unlink()  # Disposable decode; retained codestream and hash reproduce it.
        return row

    def rate(self):
        for image in self.m['images']:
            for distance in DISTANCES:
                for arm in ('baseline', 'candidate'):
                    self.probe(image, arm, distance, 'rate')
            print('rate', len(self.data), '/', len(self.m['images'])*10, image['image_id'], flush=True)
            save(self.root / 'progress.json', dict(phase='rate', observations=len(self.data), time=time.time()))

    def calibrate(self, image, arm):
        path = self.root / 'targets' / (key(image['image_id'], arm) + '.json')
        if path.exists():
            return read(path)
        budget = path.with_suffix('.attempts.json')
        attempts = read(budget)['attempts'] if budget.exists() else 0
        target = PROTOCOL['target']
        while True:
            trials = sorted((r for r in self.data.values() if r['image_id'] == image['image_id'] and r['arm'] == arm), key=lambda r: r['distance'])
            check(bool(trials), 'Run rate collection first')
            best = min(trials, key=lambda r: abs(r['score'] - target))
            if abs(best['score']-target) <= PROTOCOL['tolerance'] or attempts >= PROTOCOL['max_calibration_probes']:
                break
            brackets = [(a, b) for a, b in zip(trials, trials[1:]) if (a['score']-target)*(b['score']-target) < 0]
            if not brackets:
                break
            a, b = min(brackets, key=lambda ab: ab[1]['distance']-ab[0]['distance'])
            weight = max(.1, min(.9, (target-a['score'])/(b['score']-a['score'])))
            distance = f32(math.exp((1-weight)*math.log(a['distance'])+weight*math.log(b['distance'])))
            if not PROTOCOL['bounds'][0] <= distance <= PROTOCOL['bounds'][1] or distance in {r['distance'] for r in trials}:
                break
            attempts += 1
            save(budget, dict(attempts=attempts))
            self.probe(image, arm, distance, 'calibration')
        result = dict(image_id=image['image_id'], arm=arm, observation=best['id'],
                      attempts=attempts, score=best['score'],
                      matched=abs(best['score']-target) <= PROTOCOL['tolerance'])
        save(path, result)
        return result

    def cross_metric(self):
        done = {(r['image_id'], r['arm']) for r in rows(self.root / 'matches.jsonl')}
        images = {i['image_id']: i for i in self.m['images']}
        for name in SELECTED:
            for arm in ('baseline', 'candidate'):
                if (name, arm) in done:
                    continue
                match = self.calibrate(images[name], arm)
                row = self.data[match['observation']]
                decoded = self.decode(row['output_path'], Path(row['output_path']).parent)
                check(sha(decoded) == row['decoded_sha256'], 'Decoded output changed')
                value = float(self.command([self.m['butteraugli'], images[name]['pfm_path'], decoded,
                              '--colorspace', 'RGB_D65_SRG_Rel_Lin', '--intensity_target', 80]).splitlines()[0])
                check(math.isfinite(value) and value >= 0, 'Invalid Butteraugli score')
                append(self.root / 'matches.jsonl', {**row, **match, 'butteraugli': value})
                decoded.unlink()
                print('matched', name, arm, match['matched'], row['score'], value, flush=True)


def bd_rate(anchor, candidate, method):
    integrals = []
    for rows_ in (anchor, candidate):
        ordered = sorted(rows_, key=lambda r: -r['distance'])
        x = np.array([r['score'] for r in ordered])
        y = np.array([r['encoded_bytes'] for r in ordered])
        check(len(x) >= 4 and np.isfinite(x).all() and np.isfinite(y).all() and (y > 0).all(), 'Insufficient finite curve points')
        check((np.diff(x) > 0).all() and (np.diff(y) > 0).all(), 'Nonmonotone curve')
        check(x[0] <= 75 and x[-1] >= 85, 'Unbracketed 75-85 interval')
        cls = PchipInterpolator if method == 'pchip' else Akima1DInterpolator
        integrals.append(float(cls(x, np.log(y), extrapolate=False).integrate(75, 85)))
    return 100 * math.expm1((integrals[1]-integrals[0])/10)


def analyze(study):
    groups = defaultdict(list)
    for row in study.data.values():
        if row['kind'] == 'rate':
            groups[row['image_id'], row['arm']].append(row)
    curves = []
    for image in study.m['images']:
        name = image['image_id']
        result = {'image_id': name, 'corpus': image['corpus'], 'status': 'ready'}
        try:
            for method in ('pchip', 'akima'):
                result[method] = bd_rate(groups[name, 'baseline'], groups[name, 'candidate'], method)
        except RuntimeError as error:
            result.update(status='unresolved', reason=str(error))
        curves.append(result)
    pairs = []
    matches = {(r['image_id'], r['arm']): r for r in rows(study.root / 'matches.jsonl')}
    for name in SELECTED:
        a, b = matches.get((name, 'baseline')), matches.get((name, 'candidate'))
        result = {'image_id': name, 'status': 'unresolved'}
        if a and b and a['matched'] and b['matched'] and abs(a['score']-b['score']) <= PROTOCOL['pair_tolerance']:
            result.update(status='matched', bytes_percent=100*(b['encoded_bytes']/a['encoded_bytes']-1),
                          butteraugli_percent=100*(b['butteraugli']/a['butteraugli']-1),
                          score_difference=b['score']-a['score'])
        pairs.append(result)
    ready = [r for r in curves if r['status'] == 'ready']
    matched = [r for r in pairs if r['status'] == 'matched']
    summary = dict(rate_complete=len(ready) == 65, curves=len(ready), expected_curves=65,
                   matched_pairs=len(matched), expected_pairs=12,
                   observations=len(study.data), timing_status=PROTOCOL['timing_status'])
    for metric in ('pchip', 'akima'):
        if ready:
            summary[metric] = dict(mean=statistics.mean(r[metric] for r in ready),
                                  minimum=min(r[metric] for r in ready), maximum=max(r[metric] for r in ready))
    for metric in ('bytes_percent', 'butteraugli_percent'):
        if matched:
            summary[metric] = dict(mean=statistics.mean(r[metric] for r in matched),
                                  minimum=min(r[metric] for r in matched), maximum=max(r[metric] for r in matched))
    save(study.root / 'analysis.json', dict(summary=summary, curves=curves, matched=pairs))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'rate', 'cross-metric', 'analyze'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'freeze':
            check(args.metadata is not None, '--metadata is required for freeze')
            freeze(root, args.metadata.resolve())
            return
        study = Study(root)
        try:
            if args.action == 'rate':
                study.rate()
            elif args.action == 'cross-metric':
                study.cross_metric()
            else:
                analyze(study)
        finally:
            study.close()


if __name__ == '__main__':
    main()
