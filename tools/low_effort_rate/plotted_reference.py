#!/usr/bin/env python3
"""Recompute candidate/native curves against the attachment's libjxl e7 reference."""
from collections import defaultdict
from pathlib import Path
import statistics

import study


def main():
    root=Path('build/e3-e4-dc-grid-confirm-20260914').resolve()
    assert study.read(root/'audit.json')['status']=='passed'
    manifest=study.read(root/'manifest.json');curves=defaultdict(list)
    for row in study.rows(root/'observations.jsonl'):
        curves[row['image_id'],row['arm']].append(row)
    for row in study.rows(study.LIBJXL/'scores.jsonl'):
        if row['effort']==7 and row['resampling']==1:
            curves[row['image_id'],'libjxl-e7'].append(row)
    rows=[]
    for image in manifest['image_order']:
        for arm in manifest['arms']:
            row={'image_id':image,'arm':arm,'reference':'libjxl-e7',
                 'scope':manifest['images'][image]['resolution_class']}
            for method in ['pchip','akima']:
                row[method]=study.bd.bd_rate(*study.curve(curves[image,'libjxl-e7']),
                    *study.curve(curves[image,arm]),(75,85),method)
            rows.append(row)
    summary=[]
    for scope in ['all','kodak_0_4mp','clic_1_8_to_3_4mp','12mp','24mp','48mp']:
        for arm in manifest['arms']:
            selected=[r for r in rows if r['arm']==arm and (scope=='all' or r['scope']==scope)]
            summary.append({'scope':scope,'arm':arm,'n':len(selected),
                **{k:statistics.mean(r[k] for r in selected) for k in ['pchip','akima']}})
    files=[root/'manifest.json',root/'observations.jsonl',root/'audit.json',
           study.LIBJXL/'scores.jsonl',Path(__file__).resolve(),Path(study.__file__).resolve()]
    study.save(root/'comparison-to-plotted-e7-reference.json',{'quality_range':[75,85],
        'aggregation':'arithmetic mean per-image BD percentages, all curves supported',
        'inputs':{str(p):study.sha(p) for p in files},'summary':summary,'images':rows})
    for row in summary:
        if row['scope']=='all':print(row)


if __name__=='__main__':main()
