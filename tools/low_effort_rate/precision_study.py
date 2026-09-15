#!/usr/bin/env python3
"""Split DC rounding and extra precision after the original factorial finishes."""
from collections import defaultdict
import fcntl
import itertools
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import study


def main():
    base = Path('build/e3-e4-rate-20260914').resolve()
    root = base / 'precision-factorial'
    root.mkdir(exist_ok=True)
    # Wait on the actual collection lock, then require completed coverage.
    # A stale status JSON is never used to decide whether collection is live.
    with (base / 'run.lock').open('a') as parent_lock:
        fcntl.flock(parent_lock, fcntl.LOCK_EX)
        assert len(study.rows(base / 'observations.jsonl')) == 400
        c = study.Collector(base)
    c.root = root
    c.data = {r['id']: r for r in study.rows(root / 'observations.jsonl')}
    original = study.rows(base / 'observations.jsonl')
    probe = base / 'precision-probe'
    build = study.read(probe / 'build.json')
    assert study.read(probe / 'qualification.json')['status'] == 'passed'
    files = dict(build['files'])
    files[str(Path(__file__).resolve())] = study.sha(__file__)
    files[str(base / 'observations.jsonl')] = study.sha(base / 'observations.jsonl')
    image_ids = [*c.m['image_order'][:6], 'unsplash/campus_interior/48mp']
    manifest = {'files': files, 'image_order': image_ids, 'qualities': c.m['qualities'],
                'interventions': 'round + extra bit 1; prediction-aware + extra bit 0; smoothing off; efforts 3 and 4',
                'expected': 140, 'base_manifest_sha256': study.sha(base / 'manifest.json')}
    if (root / 'manifest.json').exists():
        assert study.read(root / 'manifest.json') == manifest
    else:
        study.save(root / 'manifest.json', manifest)
    for path, digest in files.items():
        assert study.sha(path) == digest, path
    for r in c.data.values():
        assert study.sha(r['output_path']) == r['output_sha256']
    expected_native = {(r['image_id'],r['effort']):r for r in original if r['source']=='fresh-native-control'}
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for name in image_ids:
                image = c.m['images'][name]
                c.close()
                c.err = (root / 'scorer.stderr').open('a')
                c.scorer = subprocess.Popen([c.m['scorer'], image['pfm_path'], 'auto'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=c.err,
                    text=True, bufsize=1, env=c.env)
                info = c.response()
                assert info['ready'] and info['fast_ssim2_revision'] == c.m['metric_version']['fast_ssim2_revision']
                # Neutral overlay controls extend qualification to large inputs.
                for e in (3, 4):
                    folder = root / 'controls' / name / f'e{e}'
                    folder.mkdir(parents=True, exist_ok=True)
                    c.env.pop('GJXL_RCA_DC_PRECISION', None)
                    c.command([probe/'gjxl_precision_probe','--input',image['pfm_path'],
                        '--output',folder/'output.jxl','--raw-samples',folder/'raw.json',
                        '--effort',e,'--distance',1.9,'--num-threads',8,'--warmups',0,'--samples',1])
                    assert study.sha(folder/'output.jxl') == expected_native[name,e]['output_sha256']
                for e, mode, quality in itertools.product((3,4), ('round','prediction-aware'), manifest['qualities']):
                    precision = 1 if mode == 'round' else 0
                    key = f'e{e}-{mode}-precision{precision}'
                    identifier = f'{name}|{key}|q{quality}'
                    if identifier in c.data:
                        continue
                    if study.shutil.disk_usage(root).free < c.m['free_disk_floor']:
                        raise RuntimeError('Free disk floor reached')
                    source = c.saved[name,e,quality]
                    folder = root / 'outputs' / name / key / f'q{quality}'
                    folder.mkdir(parents=True, exist_ok=True)
                    output, raw, decoded = [folder / n for n in ('output.jxl','raw.json','decoded.pfm')]
                    c.env['GJXL_RCA_DC_PRECISION'] = str(precision)
                    c.command([probe/'gjxl_precision_probe','--input',image['pfm_path'],
                        '--output',output,'--raw-samples',raw,'--effort',e,
                        '--distance',format(source['distance'],'.9g'),'--num-threads',8,
                        '--warmups',0,'--samples',1,'--dc-quantization',mode,'--no-adaptive-dc-smoothing'])
                    record = study.read(raw)
                    assert record['dc_quantization'] == mode and not record['adaptive_dc_smoothing']
                    assert record['revision'] == c.m['revision'] and record['effort'] == e
                    c.command([c.m['djxl'],output,decoded,'--color_space=RGB_D65_SRG_Rel_Lin','--num_threads=1'])
                    c.scorer.stdin.write(study.json.dumps({'path':str(decoded)})+'\n')
                    c.scorer.stdin.flush()
                    score = c.response()['score']
                    assert math.isfinite(score)
                    row = {'id':identifier,'image_id':name,'arm':key,'effort':e,
                           'dc_mode':mode,'extra_dc_precision':precision,'smoothing':False,
                           'distance':source['distance'],'requested_quality':quality,'score':score,
                           'encoded_bytes':output.stat().st_size,'output_path':str(output),
                           'output_sha256':study.sha(output),'decoded_sha256':study.sha(decoded),
                           'raw_sha256':study.sha(raw),'recorded_at':time.time()}
                    decoded.unlink()
                    study.append(root/'observations.jsonl',row)
                    c.data[identifier] = row
                    study.save(root/'progress.json',{'status':'running','pid':os.getpid(),
                               'observations':len(c.data),'expected':manifest['expected'],'time':time.time()})
                    if len(c.data)%10==0:
                        print('precision observations',len(c.data),'/',manifest['expected'],name,flush=True)
            c.close()
            assert len(c.data) == manifest['expected']
            for path, digest in files.items():
                assert study.sha(path) == digest, path
            for row in c.data.values():
                assert study.sha(row['output_path']) == row['output_sha256']
            groups = defaultdict(list)
            for row in original:
                if row['image_id'] in image_ids and row['arm'].endswith('smooth0'):
                    mode = 'prediction-aware' if 'prediction-aware' in row['arm'] else 'round'
                    groups[row['image_id'],row['effort'],mode,int(mode=='prediction-aware')].append(row)
            for row in c.data.values():
                groups[row['image_id'],row['effort'],row['dc_mode'],row['extra_dc_precision']].append(row)
            comparisons = []
            for name,e,mode,precision in itertools.product(image_ids,(3,4),('round','prediction-aware'),(0,1)):
                row = {'image_id':name,'effort':e,'mode':mode,'precision':precision,'status':'ready'}
                try:
                    for method in ('pchip','akima'):
                        row[method] = study.bd.bd_rate(*study.curve(groups[name,e,'round',0]),
                            *study.curve(groups[name,e,mode,precision]), (75,85), method)
                except ValueError as error:
                    row['status'] = str(error)
                comparisons.append(row)
            summary = []
            for e,mode,precision in itertools.product((3,4),('round','prediction-aware'),(0,1)):
                values = [r for r in comparisons if (r['effort'],r['mode'],r['precision'])==(e,mode,precision)]
                row = {'effort':e,'mode':mode,'precision':precision,'ready':sum(r['status']=='ready' for r in values),'expected':len(image_ids)}
                if row['ready'] == row['expected']:
                    row.update({method:statistics.mean(r[method] for r in values) for method in ('pchip','akima')})
                summary.append(row)
            study.save(root/'analysis.json',{'reference':'same effort, ordinary rounding, zero extra bits, smoothing off',
                       'summary':summary,'images':comparisons})
            study.save(root/'progress.json',{'status':'completed','observations':len(c.data),'time':time.time()})
            print(study.json.dumps(summary,indent=2),flush=True)
        finally:
            c.close()


if __name__ == '__main__':
    main()
