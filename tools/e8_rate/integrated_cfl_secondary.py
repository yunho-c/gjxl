#!/usr/bin/env python3
"""Reuse verified identical-pixel Butteraugli scores with integrated byte sizes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study-tools', 'secondary-run', 'writer-run', 'candidate-run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.study_tools.resolve()))
    import cjxl_quality_characterization as study
    from cjxl_bd_rate import bd_rate
    manifest = json.loads((args.secondary_run / 'manifest.json').read_text())
    assert json.loads((args.secondary_run / 'status.json').read_text())['state'] == 'complete'
    rows = study.ledger(args.secondary_run / 'results.jsonl')
    integrated = {}
    for variant, run in (('writer', args.writer_run), ('combined', args.candidate_run)):
        assert json.loads((run / 'status.json').read_text())['state'] == 'complete'
        integrated[variant] = {(r['image_id'], r['distance']): r
                              for r in study.ledger(run / 'results.jsonl')}
    points = []
    for old in rows:
        study.verify_file(old['output_path'], old['output_sha256'])
        points.append(old)
        replacement = {'resident': 'writer', 'nonlinear-metal': 'combined'}.get(old['variant'])
        if replacement is None:
            continue
        new = integrated[replacement][old['image_id'], old['distance']]
        study.verify_file(new['output_path'], new['output_sha256'])
        assert old['decoded_sha256'] == new['decoded_sha256']
        points.append({**new, 'variant': replacement, 'score': old['score'],
                       'fast_ssim2': new['score'], 'metric_reused_from': old['key'],
                       'metric_source_output_sha256': old['output_sha256']})
    comparisons = []
    for image in manifest['images']:
        name = image['image_id']
        curves = {v: sorted((r for r in points if r['image_id'] == name and r['variant'] == v),
                            key=lambda r: -r['distance'])
                  for v in ('resident', 'writer', 'combined', 'libjxl-e8')}
        low = max(min(r['score'] for r in c) for c in curves.values())
        high = min(max(r['score'] for r in c) for c in curves.values())
        for anchor in ('writer', 'resident', 'libjxl-e8'):
            a, b = curves[anchor], curves['combined']
            result = {'image_id': name, 'test': 'combined', 'anchor': anchor,
                      'butteraugli_interval': [low, high]}
            try:
                for method in ('pchip', 'akima'):
                    result[method] = bd_rate(
                        [-r['score'] for r in a], [r['encoded_bytes'] for r in a],
                        [-r['score'] for r in b], [r['encoded_bytes'] for r in b],
                        quality_range=(-high, -low), method=method)
                result['status'] = 'supported'
            except ValueError as error:
                result['status'] = str(error)
            comparisons.append(result)
    files = [Path(__file__), args.study_tools / 'cjxl_bd_rate.py']
    for run in (args.secondary_run, args.writer_run, args.candidate_run):
        files += [run / name for name in ('manifest.json', 'results.jsonl', 'status.json')]
    args.output.write_text(json.dumps({
        'metric': 'Butteraugli conventional distance, lower is better',
        'method': 'Negate distance; integrate log bytes on per-image shared support',
        'scope': 'six diagnostic images; differing intervals; no aggregate estimate',
        'score_reuse': 'Exact decoded hashes checked against previously scored pixels; new encoded byte sizes',
        'comparisons': comparisons, 'points': points,
        'hashes': {str(f.resolve()): sha(f) for f in files},
    }, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
