#!/usr/bin/env python3
"""Secondary matching for the four unresolved strict score-80 targets.

No encoding or extra calibration budget: select the closest existing score
pair within 80 +/- 0.1, then independently score its retained decodes. The
original strict-target results stay unchanged.
"""
import argparse
import fcntl
import math
from pathlib import Path

from qualify import Study, append, check, read, rows, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = Study(root)
        path = root / 'near-target-selection.json'
        strict = {(r['image_id'], r['arm']): r for r in rows(root / 'matches.jsonl')}
        if not path.exists():
            selected = []
            for name in study.m['selected']:
                a,b = strict[name,'baseline'],strict[name,'candidate']
                if a['matched'] and b['matched'] and abs(a['score']-b['score']) <= .05:
                    selected.append(dict(image_id=name, kind='strict-target', baseline=a['id'], candidate=b['id']))
                    continue
                options = {arm:[r for r in study.data.values() if r['image_id']==name and r['arm']==arm and abs(r['score']-80)<=.1]
                           for arm in ('baseline','candidate')}
                pairs = [(abs(a['score']-b['score']), abs((a['score']+b['score'])/2-80),a,b)
                         for a in options['baseline'] for b in options['candidate']]
                check(bool(pairs), f'No near-target pair: {name}')
                difference,offset,a,b = min(pairs,key=lambda pair:pair[:2])
                check(difference <= .01, f'Near-target pair differs by more than 0.01: {name}')
                selected.append(dict(image_id=name,kind='secondary-saved-probe-match',baseline=a['id'],candidate=b['id']))
            save(path,dict(protocol='Keep strict accepted pairs; for unresolved pairs minimize score difference, then target offset, among existing probes with each score in [79.9,80.1]. Require secondary pair difference <=0.01. No additional encodes.',
                           original_matches_sha256=sha(root/'matches.jsonl'),observations_sha256=sha(root/'observations.jsonl'),
                           runner_sha256=sha(__file__),selection=selected))
        selection = read(path)
        check(selection['runner_sha256']==sha(__file__), 'Secondary matcher changed')
        check(selection['original_matches_sha256']==sha(root/'matches.jsonl'), 'Original targets changed')
        done={(r['image_id'],r['arm']) for r in rows(root/'near-target-matches.jsonl')}
        images={i['image_id']:i for i in study.m['images']}
        try:
            for case in selection['selection']:
                for arm in ('baseline','candidate'):
                    if (case['image_id'],arm) in done:
                        continue
                    row=study.data[case[arm]]
                    if case['kind']=='strict-target':
                        value=strict[case['image_id'],arm]['butteraugli']
                    else:
                        decoded=study.decode(row['output_path'],Path(row['output_path']).parent)
                        check(sha(decoded)==row['decoded_sha256'],'Decode changed')
                        value=float(study.command([study.m['butteraugli'],images[case['image_id']]['pfm_path'],decoded,
                                                  '--colorspace','RGB_D65_SRG_Rel_Lin','--intensity_target',80]).splitlines()[0])
                        decoded.unlink()
                    check(math.isfinite(value) and value>=0,'Invalid Butteraugli')
                    append(root/'near-target-matches.jsonl',{**row,'matching_kind':case['kind'],'butteraugli':value})
                    print(case['image_id'],arm,row['score'],value,flush=True)
        finally:
            study.close()


if __name__=='__main__':
    main()
