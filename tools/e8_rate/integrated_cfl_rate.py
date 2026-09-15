#!/usr/bin/env python3
"""Qualify integrated eight-step CfL plus the e8 writer against frozen controls."""
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
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study-tools', 'baseline', 'writer-run', 'cfl-run', 'libjxl',
                 'prototype', 'binary', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--images', help='Comma-separated subset; default full writer corpus')
    p.add_argument('--distances', help='Comma-separated subset; default writer distances')
    p.add_argument('--budget-seconds', type=int, default=2400)
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
        writer_manifest = json.loads((args.writer_run / 'manifest.json').read_text())
        cfl_manifest = json.loads((args.cfl_run / 'manifest.json').read_text())
        assert config['metric_version'] == libconfig['metric_version'] == writer_manifest['metric_version'] == cfl_manifest['metric_version']
        images = writer_manifest['images']
        if args.images:
            selected = set(args.images.split(','))
            images = [im for im in images if im['image_id'] in selected]
            assert {im['image_id'] for im in images} == selected
        distances = [float(d) for d in args.distances.split(',')] if args.distances else writer_manifest['distances']
        assert set(distances) <= set(writer_manifest['distances'])
        assert len(distances) == len(set(distances)) and distances
        original_rows = [r for r in study.read_records(args.baseline, 'scores', config) if r['effort'] == 8]
        writer_rows = study.ledger(args.writer_run / 'results.jsonl')
        cfl_rows = study.ledger(args.cfl_run / 'results.jsonl')
        writer = {(r['image_id'], r['distance']): r for r in writer_rows}
        cfl = {(r['image_id'], r['distance']): r for r in cfl_rows}
        for binary in (args.binary, args.prototype):
            build = json.loads((binary.parent / 'build.json').read_text())
            assert sha(binary) == build['binary_sha256']
        files = [Path(__file__), args.binary, args.binary.parent / 'build.json',
                 args.prototype, args.prototype.parent / 'build.json',
                 args.study_tools / 'cjxl_bd_rate.py',
                 args.study_tools / 'cjxl_quality_characterization.py',
                 args.study_tools / 'cjxl_sweep_common.py']
        for directory, names in ((args.baseline, ('metadata.json', 'scores.jsonl')),
                                 (args.libjxl, ('metadata.json', 'scores.jsonl')),
                                 (args.writer_run, ('manifest.json', 'results.jsonl')),
                                 (args.cfl_run, ('manifest.json', 'results.jsonl'))):
            files += [directory / name for name in names]
        hashes = {str(f.resolve()): sha(f) for f in files}
        hashes.update(config['tool_hashes'])
        manifest = {'candidate': 'production-e8-eight-step-final-cfl-with-integrated-writer',
                    'images': images, 'distances': distances, 'hashes': hashes,
                    'metric_version': config['metric_version'],
                    'scope': 'full corpus' if not args.images and not args.distances else 'diagnostic subset',
                    'timing_scope': 'diagnostic rate collection; no latency qualification'}
        mp = run / 'manifest.json'
        if mp.exists() and json.loads(mp.read_text()) != manifest:
            raise ValueError('Frozen integration study changed')
        for path, digest in hashes.items():
            study.verify_file(path, digest)
        write(mp, manifest)
        done = {r['key']: r for r in study.ledger(run / 'results.jsonl')}
        for row in done.values():
            study.verify_file(row['output_path'], row['output_sha256'])
            study.verify_file(row['prototype_path'], row['prototype_sha256'])
        budget = study.Budget(args.budget_seconds)
        start = time.monotonic()
        expected = len(images) * len(distances)
        for im in images:
            study.verify_file(im['pfm_path'], im['pfm_sha256'])
            with study.Scorer(config, im, run, budget) as scorer:
                for distance in distances:
                    key = f"{im['image_id']}|d={distance}"
                    if key in done:
                        continue
                    old = writer[im['image_id'], distance]
                    standalone = cfl[im['image_id'], distance]
                    study.verify_file(old['output_path'], old['output_sha256'])
                    study.verify_file(standalone['output_path'], standalone['output_sha256'])
                    case = run / 'cases' / im['image_id'] / str(distance)
                    case.mkdir(parents=True, exist_ok=True)
                    outputs = {}
                    for label, binary in (('prototype', args.prototype), ('integrated', args.binary)):
                        budget.check()
                        output, raw = case / (label + '.jxl'), case / (label + '.json')
                        command = list(map(str, [binary.resolve(), '--input', im['pfm_path'],
                            '--output', output, '--raw-samples', raw, '--distance', distance,
                            '--effort', 8, '--num-threads', 8, '--warmups', 0, '--samples', 1]))
                        study.append(run / 'commands.jsonl', {'key': key, 'label': label, 'command': command})
                        env = {k: v for k, v in os.environ.items() if not k.startswith('GJXL_E8_')}
                        with (case / (label + '.stderr')).open('w+') as err:
                            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err,
                                                     text=True, start_new_session=True, env=env)
                            try:
                                stdout, _ = child.communicate(timeout=budget.timeout(1800))
                            except BaseException:
                                study.stop_process(child)
                                raise
                            (case / (label + '.stdout')).write_text(stdout)
                            err.seek(0)
                            if child.returncode:
                                raise RuntimeError(err.read()[-4000:])
                        outputs[label] = output
                    output = outputs['integrated']
                    digest = sha(output)
                    if digest != sha(outputs['prototype']):
                        raise ValueError('Integrated output differs from independently linked combined prototype: ' + key)
                    score = scorer.score(output)
                    if score['decoded_sha256'] != standalone['decoded_sha256']:
                        raise ValueError('Integrated pixels differ from the eight-step CfL control: ' + key)
                    row = {'key': key, 'image_id': im['image_id'], 'distance': distance,
                           'encoded_bytes': output.stat().st_size, 'output_path': str(output),
                           'output_sha256': digest, 'raw_sha256': sha(case / 'integrated.json'),
                           'prototype_path': str(outputs['prototype']), 'prototype_sha256': digest,
                           'writer_bytes': old['encoded_bytes'], 'writer_score': old['score'],
                           'pixels_equal_cfl_control': True, 'bytes_equal_combined_prototype': True, **score}
                    study.append(run / 'results.jsonl', row)
                    done[key] = row
                    print(key, row['encoded_bytes'], f"{row['score']:.6f}", flush=True)
                    write(run / 'status.json', {'state': 'running', 'completed': len(done), 'expected': expected})
        grouped = defaultdict(list)
        for variant, values in (('integrated', done.values()), ('writer', writer_rows), ('original', original_rows)):
            for row in values:
                grouped[row['image_id'], variant].append(row)
        image_map = {im['image_id']: im for im in images}
        for row in study.read_records(args.libjxl, 'scores', libconfig):
            if row['effort'] == 8 and row['resampling'] == 1 and row['image_id'] in image_map:
                assert row['reference_sha256'] == image_map[row['image_id']]['pfm_sha256']
                grouped[row['image_id'], 'libjxl'].append(row)
        def curve(name, variant):
            rows = sorted(grouped[name, variant], key=lambda r: -r['distance'])
            return [r['score'] for r in rows], [r['encoded_bytes'] for r in rows]
        rates = []
        for name in image_map:
            for anchor in ('writer', 'original', 'libjxl'):
                row = {'image_id': name, 'anchor': anchor}
                try:
                    for method in ('pchip', 'akima'):
                        row[method] = bd.bd_rate(*curve(name, anchor), *curve(name, 'integrated'), method=method)
                    row['status'] = 'supported'
                except ValueError as error:
                    row['status'] = str(error)
                rates.append(row)
        summary = {}
        for anchor in ('writer', 'original', 'libjxl'):
            values = [r for r in rates if r['anchor'] == anchor and r['status'] == 'supported']
            summary[anchor] = {'ready_images': len(values), 'images': len(images)}
            if len(values) == len(images):
                summary[anchor].update({method: statistics.mean(r[method] for r in values)
                                        for method in ('pchip', 'akima')})
                summary[anchor]['improved_images'] = sum(r['pchip'] < 0 for r in values)
        write(run / 'analysis.json', {'quality_range': [75, 85], 'image_count': len(images),
            'points': len(done), 'exact_control_pixels': len(done), 'exact_combined_bytes': len(done),
            'summary': summary, 'rates': rates, 'manifest_sha256': sha(mp),
            'results_sha256': sha(run / 'results.jsonl')})
        write(run / 'status.json', {'state': 'complete', 'completed': len(done),
              'expected': expected, 'elapsed_seconds': time.monotonic() - start})
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
