#!/usr/bin/env python3
"""Opt-in qualification against the retained September 14 e4 corpus.

Run from the repository root. Never collect data during report generation.
Every candidate must decode to the previously verified linear RGB float hash;
every freshly built baseline must reproduce the pinned native codestream.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'low_effort_rate'))
import cross_metric
import study


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True) + '\n')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def freeze(root, baseline, candidate):
    path = root / 'manifest.json'
    if not path.exists():
        reference = Path('build/e4-native-tail-full-20260914/manifest.json').resolve()
        previous = read(reference)
        files = {str(reference): sha(reference)}
        for binary in [baseline, candidate]:
            files[str(binary)] = sha(binary)
            for library in binary.parent.rglob('*.metallib'):
                files[str(library)] = sha(library)
        for file in [Path(__file__).resolve(), Path(study.__file__),
                     Path(cross_metric.__file__), Path(previous['djxl'])]:
            files[str(file)] = sha(file)
        sources = subprocess.check_output(
            ['git', 'ls-files', 'src', 'include', 'CMakeLists.txt'], text=True).splitlines()
        for source in sources:
            file = Path(source).resolve()
            files[str(file)] = sha(file)
        for im in previous['images'].values():
            assert sha(im['pfm_path']) == im['pfm_sha256']
            files[im['pfm_path']] = im['pfm_sha256']
        for row in previous['rows']:
            assert sha(row['output_path']) == row['output_sha256']
        (root / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD']))
        files[str(root / 'source.patch')] = sha(root / 'source.patch')
        save(path, {
            'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'baseline': str(baseline), 'candidate': str(candidate), 'files': files,
            'reference': str(reference), 'images': previous['images'],
            'rows': previous['rows'], 'djxl': previous['djxl'], 'expected': 455,
            'timing_images': cross_metric.IMAGES,
            'protocol': 'All 65 images x seven distances. Fresh baseline byte equality; '
                        'candidate exact decoded PFM equality. Saved scores remain valid '
                        'only after exact reconstruction checks. Rate collection is not timing '
                        'qualification. Timing uses four alternating process pairs, two warmups '
                        'and five complete-call samples per process, at original Q80 distances. '
                        'No extrapolation; arithmetic mean per-image BD percentages at 75-85.',
        })
    manifest = read(path)
    assert manifest['baseline'] == str(baseline) and manifest['candidate'] == str(candidate)
    for path, digest in manifest['files'].items():
        assert sha(path) == digest, path
    return manifest


def command(root, argv):
    assert shutil.disk_usage(root).free > 5 * 1024**3, 'Less than 5 GiB free'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GJXL_', 'RCA_'))}
    start = time.time()
    result = subprocess.run(list(map(str, argv)), env=env, capture_output=True,
                            text=True, timeout=180)
    append(root / 'commands.jsonl', {
        'argv': list(map(str, argv)), 'started': start, 'seconds': time.time() - start,
        'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
    })
    result.check_returncode()


def encode(root, m, ref, arm, folder, warmups=0, samples=1):
    folder.mkdir(parents=True, exist_ok=True)
    command(root, [m[arm], '--input', m['images'][ref['image_id']]['pfm_path'],
                   '--output', folder / 'out.jxl', '--raw-samples', folder / 'raw.json',
                   '--effort', 4, '--distance', format(ref['distance'], '.9g'),
                   '--num-threads', 8, '--warmups', warmups, '--samples', samples])
    raw = read(folder / 'raw.json')
    assert raw['timing_semantics'] == 'complete-encode-wall-time'
    assert raw['validation_encodes'] == 1 and raw['sample_count'] == samples
    assert raw['dc_quantization'] == 'prediction-aware' and raw['adaptive_dc_smoothing']
    size = (folder / 'out.jxl').stat().st_size
    assert all(r['encoded_bytes'] == size for r in raw['samples'])
    return raw


def rate(root, m, limit):
    path = root / 'observations.jsonl'
    done = {row['id'] for row in rows(path)}
    selected = sorted(m['rows'], key=lambda row: (
        row['image_id'] not in m['timing_images'], row['image_id'],
        row['requested_quality'] != 80, row['requested_quality']))
    assert len(selected) == m['expected']
    for ref in selected[:limit]:
        key = hashlib.sha256(ref['id'].encode()).hexdigest()[:24]
        if key in done:
            continue
        folder = root / 'rate' / key
        for arm in ['baseline', 'candidate']:
            encode(root, m, ref, arm, folder / arm)
        assert sha(folder / 'baseline/out.jxl') == ref['output_sha256'], ref['id']
        decoded = folder / 'decoded.pfm'
        command(root, [m['djxl'], folder / 'candidate/out.jxl', decoded,
                       '--color_space=RGB_D65_SRG_Rel_Lin', '--num_threads=1'])
        decoded_hash = sha(decoded)
        assert decoded_hash == ref['decoded_sha256'], ref['id']
        decoded.unlink()
        result = {
            'id': key, 'image_id': ref['image_id'], 'distance': ref['distance'],
            'requested_quality': ref['requested_quality'], 'score': ref['score'],
            'decoded_sha256': decoded_hash, 'folder': str(folder),
            'exact_decoded_identity': True,
        }
        for arm in ['baseline', 'candidate']:
            result[arm + '_bytes'] = (folder / arm / 'out.jxl').stat().st_size
            result[arm + '_sha256'] = sha(folder / arm / 'out.jxl')
            result[arm + '_raw_sha256'] = sha(folder / arm / 'raw.json')
        append(path, result)
        done.add(key)
        save(root / 'progress.json', {'completed': len(done), 'expected': m['expected']})
        print(len(done), '/', m['expected'], ref['image_id'], ref['requested_quality'],
              round(100 * (result['candidate_bytes'] / result['baseline_bytes'] - 1), 4),
              flush=True)


def timing(root, m):
    path = root / 'timing.jsonl'
    done = {row['id'] for row in rows(path)}
    rates = {(row['image_id'], row['requested_quality']): row
             for row in rows(root / 'observations.jsonl')}
    selected = {row['image_id']: row for row in m['rows'] if row['requested_quality'] == 80}
    for pair in range(4):
        for image in m['timing_images']:
            ref = selected[image]
            key = hashlib.sha256(image.encode()).hexdigest()[:16]
            order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
            for arm in order:
                identifier = f'{key}-{pair}-{arm}'
                if identifier in done:
                    continue
                folder = root / 'timing' / identifier
                raw = encode(root, m, ref, arm, folder, 2, 5)
                assert sha(folder / 'out.jxl') == rates[image, 80][arm + '_sha256']
                values = [sample['elapsed_nanoseconds'] for sample in raw['samples']]
                append(path, {'id': identifier, 'image_id': image, 'pair': pair,
                              'arm': arm, 'samples_ns': values,
                              'median_ns': statistics.median(values),
                              'raw_path': str(folder / 'raw.json'),
                              'raw_sha256': sha(folder / 'raw.json')})
                done.add(identifier)
                print('timing', len(done), '/96', image, arm, flush=True)


def report(root, m):
    observations = rows(root / 'observations.jsonl')
    assert len(observations) == len({r['id'] for r in observations}) == m['expected']
    grouped = defaultdict(list)
    for row in observations:
        for arm in ['baseline', 'candidate']:
            folder = Path(row['folder']) / arm
            assert sha(folder / 'out.jxl') == row[arm + '_sha256']
            assert sha(folder / 'raw.json') == row[arm + '_raw_sha256']
        grouped[row['image_id']].append(row)
    stock = defaultdict(list)
    for row in study.rows(study.LIBJXL / 'scores.jsonl'):
        if row['effort'] == 4 and row['resampling'] == 1:
            stock[row['image_id']].append(row)
    results = []
    for image, points in grouped.items():
        assert len(points) == 7
        baseline = [{**p, 'encoded_bytes': p['baseline_bytes']} for p in points]
        candidate = [{**p, 'encoded_bytes': p['candidate_bytes']} for p in points]
        row = {'image_id': image}
        for name, reference in [('vs_native', baseline), ('vs_stock_e4', stock[image])]:
            for method in ['pchip', 'akima']:
                row[name + '_' + method] = study.bd.bd_rate(
                    *study.curve(reference), *study.curve(candidate), (75, 85), method)
        results.append(row)
    summary = {name: statistics.mean(r[name] for r in results)
               for name in results[0] if name != 'image_id'}
    summary.update(images=len(results), native_wins=sum(r['vs_native_pchip'] < 0 for r in results),
                   stock_wins=sum(r['vs_stock_e4_pchip'] < 0 for r in results),
                   smaller_files=sum(r['candidate_bytes'] < r['baseline_bytes'] for r in observations),
                   larger_files=sum(r['candidate_bytes'] > r['baseline_bytes'] for r in observations))
    timing_rows = rows(root / 'timing.jsonl')
    timing_summary = []
    if timing_rows:
        assert len(timing_rows) == len({r['id'] for r in timing_rows}) == 96
        for r in timing_rows:
            assert sha(r['raw_path']) == r['raw_sha256']
        for image in m['timing_images']:
            selected = [r for r in timing_rows if r['image_id'] == image]
            ratios = []
            for pair in range(4):
                times = {r['arm']: r['median_ns'] for r in selected if r['pair'] == pair}
                ratios.append(times['candidate'] / times['baseline'])
            medians = {arm: statistics.median(v for r in selected if r['arm'] == arm
                                             for v in r['samples_ns']) / 1e6
                       for arm in ['baseline', 'candidate']}
            timing_summary.append({'image_id': image, 'median_ms': medians,
                                   'paired_ratios': ratios,
                                   'median_paired_ratio': statistics.median(ratios)})
    save(root / 'analysis.json', {'summary': summary, 'images': results, 'timing': timing_summary,
                                 'range': [75, 85], 'timing_boundary': 'complete-encode-wall-time'})
    save(root / 'audit.json', {'status': 'passed', 'fresh_baseline_byte_identity': 455,
                              'exact_decoded_float_identity': 455,
                              'timing_processes': len(timing_rows), 'source_files': len(m['files'])})
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['rate', 'timing', 'report'])
    parser.add_argument('--root', type=Path, default=Path('build/dc-uint-qualification-20260914'))
    parser.add_argument('--baseline', type=Path, default=Path('build/dc-uint-baseline/gjxl_quality_benchmark'))
    parser.add_argument('--candidate', type=Path, default=Path('build/dc-uint-candidate/gjxl_quality_benchmark'))
    parser.add_argument('--limit', type=int, default=455)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = freeze(root, args.baseline.resolve(), args.candidate.resolve())
        if args.action == 'rate':
            rate(root, manifest, args.limit)
        elif args.action == 'timing':
            timing(root, manifest)
        else:
            report(root, manifest)


if __name__ == '__main__':
    main()
