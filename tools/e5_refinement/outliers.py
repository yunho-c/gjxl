#!/usr/bin/env python3
"""Supplementary matched-quality checks selected by observed BD-rate loss.

These are deliberately post-selection diagnostics, separate from the twelve
preselected images. The selection and input analysis hash are frozen first.
"""
import argparse
import fcntl
import math
from pathlib import Path

from qualify import Study, append, check, read, rows, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = Study(root)
        path = root / 'outlier-selection.json'
        if not path.exists():
            analysis = read(root / 'analysis.json')
            check(analysis['summary']['rate_complete'], 'Complete the rate sweep first')
            selected = sorted(analysis['curves'], key=lambda r: -r['pchip'])[:3]
            save(path, dict(selection='Three largest observed PCHIP BD-rate increases',
                            analysis_sha256=sha(root / 'analysis.json'), images=selected,
                            runner_sha256=sha(__file__)))
        selection = read(path)
        check(selection['runner_sha256'] == sha(__file__), 'Outlier runner changed')
        images = {i['image_id']: i for i in study.m['images']}
        done = {(r['image_id'], r['arm']) for r in rows(root / 'outlier-matches.jsonl')}
        try:
            for selected in selection['images']:
                name = selected['image_id']
                for arm in ('baseline', 'candidate'):
                    if (name, arm) in done:
                        continue
                    match = study.calibrate(images[name], arm)
                    row = study.data[match['observation']]
                    decoded = study.decode(row['output_path'], Path(row['output_path']).parent)
                    check(sha(decoded) == row['decoded_sha256'], 'Decoded hash mismatch')
                    value = float(study.command([study.m['butteraugli'], images[name]['pfm_path'], decoded,
                                  '--colorspace', 'RGB_D65_SRG_Rel_Lin', '--intensity_target', 80]).splitlines()[0])
                    check(math.isfinite(value) and value >= 0, 'Nonfinite Butteraugli')
                    append(root / 'outlier-matches.jsonl', {**row, **match, 'butteraugli': value})
                    decoded.unlink()
                    print('outlier', name, arm, match['matched'], row['score'], value, flush=True)
        finally:
            study.close()


if __name__ == '__main__':
    main()
