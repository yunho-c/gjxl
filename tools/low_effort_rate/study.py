#!/usr/bin/env python3
"""Pinned e3/e4 rate investigation; no production changes or timing claims.

Use the plot's original binary and ledgers. All collection is explicit,
resumable, serial, locked, and bounded by the frozen factorial manifest.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import select
import shutil
import statistics
import subprocess
import sys
import time

BASE = Path('/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03')
GJXL = BASE / 'fixed-gjxl-e1-4-20260914'
LIBJXL = BASE / 'quality-full-20260908'
ANALYSIS = BASE / 'gjxl-e1-4-frankenstein-20260914/analysis-code'
sys.path.insert(0, str(ANALYSIS))
import cjxl_bd_rate as bd

PILOT = [
    'kodak/01', 'kodak/17',
    'clic2024_test/097cb426910ba8ce2525dd8bb7fb1777',
    'clic2024_test/d1a9be98d1936065967adac50a6fb750',
    'clic2024_test/b51d5fb537246482ef7a7ab63f093cab',
    'unsplash/campus_interior/12mp',
    'unsplash/campus_interior/24mp',
    'unsplash/campus_interior/48mp',
    'unsplash/alpine_lake/24mp',
    'unsplash/forest_stream/48mp',
]
QUALITIES = [50, 70, 80, 90, 95]


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def arm(effort, quantization, smoothing):
    return f'e{effort}-{quantization}-smooth{int(smoothing)}'


def native(effort):
    return arm(effort, 'round' if effort == 3 else 'prediction-aware', effort == 4)


def freeze(root):
    if (root / 'manifest.json').exists():
        raise RuntimeError('Manifest exists; use collect/analyze to resume')
    config = read(GJXL / 'metadata.json')
    studies = bd.load_studies([LIBJXL, GJXL])
    for e in (3, 4):
        result = bd.analyze(studies, baseline_effort=e, efforts=[e])
        save(root / f'direct-e{e}.json', result)
        assert all(p['status'] == 'ready' for p in result['points'])
    files = dict(config['tool_hashes'])
    for directory in (GJXL, LIBJXL):
        for name in ('metadata.json', 'scores.jsonl'):
            files[str(directory / name)] = sha(directory / name)
    for name in ('cjxl_bd_rate.py', 'cjxl_quality_characterization.py'):
        files[str(ANALYSIS / name)] = sha(ANALYSIS / name)
    files[str(Path(__file__).resolve())] = sha(__file__)
    images = {i['image_id']: i for i in config['images'] if i['image_id'] in PILOT}
    assert len(images) == len(PILOT)
    for image in images.values():
        assert sha(image['pfm_path']) == image['pfm_sha256']
        files[image['pfm_path']] = image['pfm_sha256']
    verified = 0
    for study in studies:
        for r in study['scores']:
            if r['effort'] in (3, 4):
                assert sha(r['output_path']) == r['output_sha256'], r['output_path']
                verified += 1
    for path, digest in files.items():
        assert sha(path) == digest, path
    arms = {arm(e, q, s): {'effort': e, 'quantization': q, 'smoothing': s}
            for e, q, s in itertools.product((3, 4), ('round', 'prediction-aware'), (False, True))}
    save(root / 'manifest.json', {
        'revision': config['encoder_revision'], 'binary': config['benchmark'],
        'djxl': config['djxl'], 'scorer': config['scorer'],
        'metric_version': config['metric_version'], 'images': images,
        'image_order': PILOT, 'qualities': QUALITIES, 'arms': arms, 'files': files,
        'interval': [75, 85], 'verified_source_outputs': verified,
        'selection': 'Diagnostic: e3/e4 outliers, Kodak controls, repeated-scene resolution controls; not representative or held out',
        'precision_factor': 'Prediction-aware DC mode also adds one DC precision bit; not a rounding-only intervention',
        'timing': 'Diagnostic only; concurrent workloads allowed; no latency conclusions',
        'created': time.time(), 'free_disk_floor': 6 * 1024**3,
    })
    print('Frozen', verified, 'source outputs;', len(arms) * len(PILOT) * len(QUALITIES), 'factorial cells', flush=True)


class Collector:
    def __init__(self, root):
        self.root = root
        self.m = read(root / 'manifest.json')
        for path, digest in self.m['files'].items():
            assert sha(path) == digest, path
        self.data = {r['id']: r for r in rows(root / 'observations.jsonl')}
        assert len(self.data) == len(rows(root / 'observations.jsonl'))
        for r in self.data.values():
            assert sha(r['output_path']) == r['output_sha256']
        self.saved = {(r['image_id'], r['effort'], r['requested_quality']): r
                      for r in rows(GJXL / 'scores.jsonl') if r['effort'] in (3, 4)}
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('GJXL_', 'RCA_'))}
        self.env['RAYON_NUM_THREADS'] = '8'
        self.scorer = None
        self.err = None

    def command(self, argv):
        argv = list(map(str, argv))
        started = time.time()
        result = subprocess.run(argv, capture_output=True, text=True, env=self.env)
        append(self.root / 'commands.jsonl', {'argv': argv, 'started': started,
               'seconds': time.time() - started, 'returncode': result.returncode,
               'stdout': result.stdout, 'stderr': result.stderr})
        result.check_returncode()

    def response(self):
        if not select.select([self.scorer.stdout], [], [], 600)[0]:
            raise RuntimeError('Scorer response timeout; retain current process evidence')
        line = self.scorer.stdout.readline()
        if not line:
            raise RuntimeError('Scorer terminated')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result)
        return result

    def close(self):
        if self.scorer is not None:
            self.scorer.stdin.close()
            self.scorer.wait(timeout=30)
            self.scorer = None
            self.err.close()

    def collect(self):
        for name in self.m['image_order']:
            image = self.m['images'][name]
            self.close()
            self.err = (self.root / 'scorer.stderr').open('a')
            self.scorer = subprocess.Popen([self.m['scorer'], image['pfm_path'], 'auto'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                text=True, bufsize=1, env=self.env)
            info = self.response()
            assert info['ready'] and info['fast_ssim2_revision'] == self.m['metric_version']['fast_ssim2_revision']
            assert (info['width'], info['height']) == (image['width'], image['height'])
            for key, policy in self.m['arms'].items():
                for quality in self.m['qualities']:
                    identifier = f'{name}|{key}|q{quality}'
                    if identifier in self.data:
                        continue
                    if shutil.disk_usage(self.root).free < self.m['free_disk_floor']:
                        raise RuntimeError('Free disk floor reached')
                    source = self.saved[name, policy['effort'], quality]
                    if key == native(policy['effort']) and quality != 80:
                        result = {**source, 'id': identifier, 'arm': key, 'source': 'verified-saved-native'}
                    else:
                        folder = self.root / 'outputs' / name / key / f'q{quality}'
                        folder.mkdir(parents=True, exist_ok=True)
                        output, raw, decoded = [folder / n for n in ('output.jxl', 'raw.json', 'decoded.pfm')]
                        self.command([self.m['binary'], '--input', image['pfm_path'],
                            '--output', output, '--raw-samples', raw, '--effort', policy['effort'],
                            '--distance', format(source['distance'], '.9g'), '--num-threads', 8,
                            '--warmups', 0, '--samples', 1, '--dc-quantization', policy['quantization'],
                            '--adaptive-dc-smoothing' if policy['smoothing'] else '--no-adaptive-dc-smoothing'])
                        record = read(raw)
                        assert record['revision'] == self.m['revision']
                        assert record['dc_quantization'] == policy['quantization']
                        assert record['adaptive_dc_smoothing'] == policy['smoothing']
                        assert record['dc_prediction'] == 'weighted'
                        assert record['effort'] == policy['effort'] and record['resampling'] == 1
                        assert record['thread_count'] == 8 and record['metal_aq_mode'] == 'fully-resident'
                        assert record['validation_encodes'] == record['sample_count'] == 1
                        assert record['samples'][0]['encoded_bytes'] == output.stat().st_size
                        digest = sha(output)
                        if key == native(policy['effort']):
                            assert digest == source['output_sha256'], identifier
                        self.command([self.m['djxl'], output, decoded,
                            '--color_space=RGB_D65_SRG_Rel_Lin', '--num_threads=1'])
                        self.scorer.stdin.write(json.dumps({'path': str(decoded)}) + '\n')
                        self.scorer.stdin.flush()
                        score = self.response()['score']
                        assert math.isfinite(score)
                        decoded_sha = sha(decoded)
                        if key == native(policy['effort']):
                            assert decoded_sha == source['decoded_sha256']
                            assert abs(score - source['score']) < 1e-9
                        result = {'id': identifier, 'arm': key, 'image_id': name,
                            'effort': policy['effort'], 'requested_quality': quality,
                            'distance': source['distance'], 'score': score,
                            'encoded_bytes': output.stat().st_size, 'output_path': str(output),
                            'output_sha256': digest, 'decoded_sha256': decoded_sha,
                            'source': 'fresh-native-control' if key == native(policy['effort']) else 'dc-intervention',
                            'raw_sha256': sha(raw), 'scorer_mode': info.get('mode'),
                            'recorded_at': time.time()}
                        decoded.unlink()
                    append(self.root / 'observations.jsonl', result)
                    self.data[identifier] = result
                    if len(self.data) % 10 == 0:
                        print('observations', len(self.data), '/', 400, name, key, flush=True)
                        save(self.root / 'progress.json', {'status': 'running', 'pid': os.getpid(),
                             'observations': len(self.data), 'time': time.time()})
        self.close()
        assert len(self.data) == 400
        save(self.root / 'progress.json', {'status': 'completed', 'observations': len(self.data), 'time': time.time()})


def curve(rows_):
    sorted_rows = sorted(rows_, key=lambda r: -r['distance'])
    return [r['score'] for r in sorted_rows], [r['encoded_bytes'] for r in sorted_rows]


def analyze(root):
    m = read(root / 'manifest.json')
    data = defaultdict(list)
    for r in rows(root / 'observations.jsonl'):
        data[r['image_id'], r['arm']].append(r)
    stock = defaultdict(list)
    for r in rows(LIBJXL / 'scores.jsonl'):
        if r['effort'] in (3, 4) and r['resampling'] == 1:
            stock[r['image_id'], r['effort']].append(r)
    comparisons = []
    for name in m['image_order']:
        for key, p in m['arms'].items():
            for reference in dict.fromkeys(('stock', native(p['effort']), 'e3-round-smooth0',
                                            'e3-prediction-aware-smooth1')):
                test = data[name, key]
                anchor = stock[name, p['effort']] if reference == 'stock' else data[name, reference]
                row = {'image_id': name, 'arm': key, 'reference': reference,
                       'effort': p['effort'], 'status': 'ready'}
                try:
                    if len(test) != len(m['qualities']):
                        raise ValueError('Incomplete candidate curve')
                    for method in ('pchip', 'akima'):
                        row[method] = bd.bd_rate(*curve(anchor), *curve(test), m['interval'], method)
                except ValueError as error:
                    row['status'] = str(error)
                comparisons.append(row)
    groups = defaultdict(list)
    for r in comparisons:
        groups[r['arm'], r['reference']].append(r)
    summary = []
    for (key, reference), group in groups.items():
        ready = [r for r in group if r['status'] == 'ready']
        row = {'arm': key, 'reference': reference, 'ready': len(ready), 'planned': len(m['image_order'])}
        # No complete-cohort mean is published for incomplete curves.
        if len(ready) == len(m['image_order']):
            row.update({method: statistics.mean(r[method] for r in ready) for method in ('pchip', 'akima')})
            row.update(wins=sum(r['pchip'] < 0 for r in ready),
                       worst=max(r['pchip'] for r in ready), best=min(r['pchip'] for r in ready))
        summary.append(row)
    save(root / 'dc-factorial-analysis.json', {'summary': summary, 'images': comparisons,
          'observations': sum(map(len, data.values())), 'scope': m['selection']})
    for row in summary:
        if row['reference'] == 'stock':
            print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'collect', 'analyze'))
    parser.add_argument('--root', type=Path, default=Path('build/e3-e4-rate-20260914'))
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'freeze':
            freeze(root)
        elif args.action == 'analyze':
            analyze(root)
        else:
            collector = Collector(root)
            save(root / 'progress.json', {'status': 'running', 'pid': os.getpid(), 'time': time.time()})
            try:
                collector.collect()
            finally:
                collector.close()


if __name__ == '__main__':
    main()
