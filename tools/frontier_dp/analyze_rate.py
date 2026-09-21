#!/usr/bin/env python3
"""Exploratory BD-rate for the bounded selector pilot; never extrapolate."""
import argparse
import json
import math
from pathlib import Path
import statistics

from scipy.interpolate import Akima1DInterpolator, PchipInterpolator


def analyze(root):
    rows = [json.loads(x) for x in (root / 'observations.jsonl').read_text().splitlines()]
    intervals = [('ssimulacra2', (60, 80)), ('ssimulacra2', (75, 85)),
                 ('butteraugli', (-math.log(4), -math.log(1.5)))]
    outcomes = []
    for image in sorted({x['image'] for x in rows}):
        for effort in sorted({x['effort'] for x in rows}):
            for arm in ('frontier', 'rectangle'):
                for metric, interval in intervals:
                    row = dict(image=image, effort=effort, arm=arm, metric=metric, interval=interval)
                    for method, cls in [('pchip', PchipInterpolator), ('akima', Akima1DInterpolator)]:
                        try:
                            totals = []
                            for a in ('greedy', arm):
                                curve = sorted((x for x in rows if x['image'] == image and
                                                x['effort'] == effort and x['arm'] == a),
                                               key=lambda x: -x['distance'])
                                assert len(curve) >= 4, 'insufficient points'
                                q = [x[metric] if metric == 'ssimulacra2' else -math.log(x[metric]) for x in curve]
                                r = [math.log(x['bytes']) for x in curve]
                                assert all(y > x for x, y in zip(q, q[1:])), 'nonmonotonic quality'
                                assert all(y > x for x, y in zip(r, r[1:])), 'nonmonotonic rate'
                                assert q[0] <= interval[0] and q[-1] >= interval[1], 'unbracketed'
                                totals.append(float(cls(q, r, extrapolate=False).integrate(*interval)))
                            row[method] = 100 * math.expm1((totals[1] - totals[0]) / (interval[1] - interval[0]))
                        except (AssertionError, ValueError) as e:
                            row[method + '_error'] = str(e)
                    outcomes.append(row)
    summary = []
    for arm in ('frontier', 'rectangle'):
        for metric, interval in intervals:
            values = [r for r in outcomes if r['arm'] == arm and r['metric'] == metric and r['interval'] == interval]
            item = dict(arm=arm, metric=metric, interval=interval,
                        coverage=sum('pchip' in r for r in values), expected=len(values))
            for method in ('pchip', 'akima'):
                a = [r[method] for r in values if method in r]
                item[method] = (dict(mean=statistics.mean(a), min=min(a), max=max(a),
                                    median=statistics.median(a), wins=sum(x < 0 for x in a)) if a else None)
            summary.append(item)
    result = dict(boundary='Exploratory six-image pilot; five-point curves; no extrapolation; '
                          'equal image-effort weights; lower BD-rate is better; no timing claim. '
                          'Butteraugli uses negative log(distance), equivalent to decibel scaling.',
                  summary=summary, curves=outcomes)
    (root / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
