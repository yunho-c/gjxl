#!/usr/bin/env python3
"""Analyze byte-qualified current-frame writer swaps from saved data only."""
from collections import Counter, defaultdict
import argparse
from pathlib import Path
import statistics

import study
import tail_study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=tail_study.ROOT)
    args=parser.parse_args();root=args.root.resolve();m=study.read(root/'manifest.json')
    expected=len(m['image_order'])
    points=len(m['rows'])//(expected*len(m['arms']))
    assert study.read(root/'audit.json')['status']=='passed'
    observations=study.rows(root/'observations.jsonl');curves=defaultdict(list)
    for r in observations:
        curves[r['image_id'],r['arm'],'native'].append({**r,'encoded_bytes':r['native_bytes']})
        curves[r['image_id'],r['arm'],'tail'].append({**r,'encoded_bytes':r['tail_bytes']})
    stock=defaultdict(list)
    for r in study.rows(study.LIBJXL/'scores.jsonl'):
        if r['effort'] in [3,4] and r['resampling']==1:stock[r['image_id'],r['effort']].append(r)
    results=[]
    for image in m['image_order']:
        for arm,policy in m['arms'].items():
            assert len(curves[image,arm,'native'])==len(curves[image,arm,'tail'])==points
            pairs=[('tail-vs-native',curves[image,arm,'native'],curves[image,arm,'tail']),
                   ('native-vs-stock',stock[image,policy['effort']],curves[image,arm,'native']),
                   ('tail-vs-stock',stock[image,policy['effort']],curves[image,arm,'tail'])]
            for label,reference,candidate in pairs:
                row={'image_id':image,'arm':arm,'comparison':label,'status':'ready'}
                try:
                    for method in ['pchip','akima']:
                        row[method]=study.bd.bd_rate(*study.curve(reference),*study.curve(candidate),(75,85),method)
                except ValueError as error:row['status']=str(error)
                results.append(row)
    summary=[]
    for arm in m['arms']:
        for label in ['tail-vs-native','native-vs-stock','tail-vs-stock']:
            rows=[r for r in results if r['arm']==arm and r['comparison']==label]
            ready=[r for r in rows if r['status']=='ready']
            row={'arm':arm,'comparison':label,'ready':len(ready),'expected':expected,
                 'failures':dict(Counter(r['status'] for r in rows if r['status']!='ready'))}
            if len(ready)==expected:
                row.update({k:statistics.mean(r[k] for r in ready) for k in ['pchip','akima']})
                row.update(wins=sum(r['pchip']<0 for r in ready),worst=max(r['pchip'] for r in ready),best=min(r['pchip'] for r in ready))
            summary.append(row);print(row)
    study.save(root/'analysis.json',{'range':[75,85],
        'aggregation':f'arithmetic mean per-image BD percentages over all {expected} declared inputs; no extrapolation or partial means',
        'boundary':'Same completed GJXL frame; native and tail have exact equal decoded PFM hashes. Tail policy matches the effort of the GJXL arm. Frontend, filters and DC precision are held fixed. Stock reference is the original independent full libjxl encoder.',
        'limits':f'{expected}-input declared cohort. Arithmetic mean percentages are not additive. Tail implementation overhead and timing are not production implementation costs.',
        'inputs':{str(p):study.sha(p) for p in [root/'audit.json',root/'observations.jsonl',study.LIBJXL/'scores.jsonl']},
        'summary':summary,'images':results})


if __name__=='__main__':main()
