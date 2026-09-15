#!/usr/bin/env python3
"""Verify terminal calibration coverage and expose score discontinuity brackets."""
from collections import Counter, defaultdict
import math
from pathlib import Path

import cross_metric
import study


def main():
    root=cross_metric.ROOT;m=study.read(root/'manifest.json')
    assert study.read(root/'progress.json')['status']=='completed'
    for path,digest in m['files'].items():assert study.sha(path)==digest,path
    fresh=study.rows(root/'observations.jsonl');data={r['id']:r for r in m['seeds']+fresh}
    assert len(data)==len(m['seeds'])+len(fresh)
    for r in data.values():
        assert study.sha(r['output_path'])==r['output_sha256']
        assert Path(r['output_path']).stat().st_size==r['encoded_bytes']
    for row in fresh:
        path=Path(row['output_path']).parent/'raw.json';assert study.sha(path)==row['raw_sha256']
        raw=study.read(path);policy=m['arms'][row['arm']]
        assert raw['effort']==policy['effort'] and raw['revision']==policy['revision']
        assert raw['validation_encodes']==raw['sample_count']==1 and raw['thread_count']==8
        assert raw['samples'][0]['encoded_bytes']==row['encoded_bytes']
    matches=study.rows(root/'matches.jsonl')
    expected={(image,arm) for image in m['image_order'] for arm in m['arms']}
    assert len(matches)==len(expected)==72
    assert {(r['image_id'],r['arm']) for r in matches}==expected
    per_arm=defaultdict(Counter);brackets=[]
    for match in matches:
        key=cross_metric.identifier(match['image_id'],match['arm'])
        target=study.read(root/'targets'/f'{key}.json')
        for k,v in target.items():assert match[k]==v,(match['image_id'],k)
        row=data[match['observation']]
        for k in ['score','encoded_bytes','output_path','output_sha256','decoded_sha256','distance']:assert match[k]==row[k]
        assert math.isfinite(match['butteraugli']) and match['butteraugli']>=0
        assert match['matched']==(abs(match['score']-m['target'])<=m['tolerance'])
        trials=sorted([r for r in data.values() if r['image_id']==match['image_id'] and r['arm']==match['arm']],key=lambda r:r['distance'])
        new=[r for r in fresh if r['image_id']==match['image_id'] and r['arm']==match['arm']]
        assert len(new)==match['attempts']<=m['max_probes_per_target']
        assert abs(match['error'])<=min(abs(r['score']-m['target']) for r in trials)+1e-12
        per_arm[match['arm']]['matched' if match['matched'] else 'unresolved']+=1
        if not match['matched']:
            candidates=[(a,b) for a,b in zip(trials,trials[1:]) if (a['score']-m['target'])*(b['score']-m['target'])<0]
            result={'image_id':match['image_id'],'arm':match['arm'],'best_score':match['score'],
                    'error':match['error'],'probes':match['attempts']}
            if candidates:
                a,b=min(candidates,key=lambda ab:ab[1]['distance']-ab[0]['distance'])
                result.update(distance_bracket=[a['distance'],b['distance']],score_bracket=[a['score'],b['score']],
                              relative_distance_span=(b['distance']-a['distance'])/a['distance'],
                              score_jump=abs(b['score']-a['score']))
            brackets.append(result)
    controls=study.rows(root/'stock-controls.jsonl')
    assert len(controls)==24 and {(r['image_id'],r['arm']) for r in controls}=={
        (name,f'libjxl-e{e}') for name in m['image_order'] for e in [3,4]}
    for row in controls:
        p=root/'controls'/row['id']/'output.jxl';assert study.sha(p)==row['output_sha256']
        seed=next(r for r in m['seeds'] if r['image_id']==row['image_id'] and r['arm']==row['arm'] and r['requested_quality']==80)
        assert row['output_sha256']==seed['output_sha256']
    commands=study.rows(root/'commands.jsonl');assert all(r['returncode']==0 for r in commands)
    study.save(root/'audit.json',{'status':'passed','targets':72,'matched':sum(r['matched'] for r in matches),
        'unresolved':len(brackets),'per_arm':dict(per_arm),'new_scored_probes':len(fresh),
        'stock_byte_decode_score_controls':24,'verified_outputs':len(data),
        'verified_frozen_files':len(m['files']),'successful_external_commands':len(commands),
        'protocol_amendment':'01-stock-schema: stock record omits GJXL-only stage_profile_enabled; no targets were collected before correction',
        'claim':'All targets terminal; only accepted pairs are matched-quality evidence. No matched-Butteraugli or BD-rate inference from this point check.'})
    study.save(root/'unresolved-brackets.json',{'targets':brackets,
        'interpretation':'A tiny final distance bracket with a large score jump is evidence of a discontinuity at this target. A wider bracket only establishes failure under the persistent probe budget; do not assume the target is unattainable.'})
    print(study.read(root/'audit.json'))


if __name__=='__main__':main()
