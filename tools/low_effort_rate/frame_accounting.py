#!/usr/bin/env python3
"""Byte-qualified native section accounting at fixed distance and matched score."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import study
import cross_metric

ROOT = Path('build/e3-e4-frame-probe-20260914').resolve()
FULL = cross_metric.FULL
BINARY = ROOT/'gjxl_frame_probe'
COMPONENTS = ['dc_global_bits','dc_header_bits','dc_token_bits',
              'metadata_header_bits','metadata_token_bits','ac_section_bits',
              'frame_toc_padding_bits']


def collect(mode):
    dest=ROOT/mode;dest.mkdir(exist_ok=True)
    full=study.read(FULL/'manifest.json');build=study.read(ROOT/'build.json')
    path=dest/'manifest.json'
    if not path.exists():
        source=FULL/'observations.jsonl' if mode=='fixed' else cross_metric.ROOT/'matches.jsonl'
        if mode=='matched':
            assert study.read(cross_metric.ROOT/'progress.json')['status']=='completed'
        selected=[]
        for row in study.rows(source):
            if row['image_id'] not in cross_metric.IMAGES or row['arm'] not in full['arms']:continue
            if mode=='fixed' and row['requested_quality']!=80:continue
            selected.append(row)
        assert len(selected)==48
        files={**build['files'],str(source):study.sha(source)}
        for p in (ROOT/'build.json',FULL/'manifest.json',Path(__file__).resolve(),Path(study.__file__).resolve()):
            files[str(p)]=study.sha(p)
        for row in selected:
            files[row['output_path']]=row['output_sha256']
            image=full['images'][row['image_id']]
            files[image['pfm_path']]=image['pfm_sha256']
        study.save(path,{'mode':mode,'files':files,'rows':selected,'arms':full['arms'],
            'images':full['images'],'revision':full['revision'],
            'policy':'Capture must preserve exact prior codestream bytes. First image repeated for every arm, including diagnostic equality. Section counts are additive; model and quantizer estimates overlap larger sections.'})
    manifest=study.read(path)
    for p,digest in manifest['files'].items():assert study.sha(p)==digest,p
    rows=study.rows(dest/'observations.jsonl');done={r['id'] for r in rows}
    env={k:v for k,v in os.environ.items() if not k.startswith(('GJXL_','RCA_'))}
    selected=sorted(manifest['rows'],key=lambda r:(cross_metric.IMAGES.index(r['image_id']),r['arm']))
    for ref in selected:
        key=cross_metric.identifier(mode,ref['image_id'],ref['arm'])
        if key in done:continue
        folder=dest/'cases'/key;folder.mkdir(parents=True,exist_ok=True)
        policy=manifest['arms'][ref['arm']];image=manifest['images'][ref['image_id']]
        if 'precision_override' in policy:env['GJXL_RCA_DC_PRECISION']=str(policy['precision_override'])
        else:env.pop('GJXL_RCA_DC_PRECISION',None)
        captures=[]
        repetitions=2 if ref['image_id']=='kodak/01' else 1
        for repeat in range(repetitions):
            output=folder/f'output-{repeat}.jxl';raw=folder/f'raw-{repeat}.json';capture=folder/f'capture-{repeat}.json'
            env['GJXL_FINAL_FIELD_RATE_DUMP']=str(capture)
            argv=list(map(str,[BINARY,'--input',image['pfm_path'],'--output',output,'--raw-samples',raw,
                '--effort',policy['effort'],'--distance',format(ref['distance'],'.9g'),
                '--num-threads',8,'--warmups',0,'--samples',1,
                '--dc-quantization',policy['quantization'],
                '--adaptive-dc-smoothing' if policy['smoothing'] else '--no-adaptive-dc-smoothing']))
            start=time.time();result=subprocess.run(argv,env=env,capture_output=True,text=True)
            study.append(dest/'commands.jsonl',{'argv':argv,'precision_override':env.get('GJXL_RCA_DC_PRECISION'),
                'capture':str(capture),'seconds':time.time()-start,'returncode':result.returncode,
                'stdout':result.stdout,'stderr':result.stderr})
            result.check_returncode()
            assert study.sha(output)==ref['output_sha256'],(ref['image_id'],ref['arm'])
            data=study.read(capture);raw_data=study.read(raw)
            assert raw_data['revision']==manifest['revision']
            assert data['total_bytes']==ref['encoded_bytes']==output.stat().st_size
            assert data['reemit_byte_identity']
            assert sum(data[k] for k in COMPONENTS)==8*data['total_bytes']
            expected=policy.get('precision_override',0 if policy['quantization']=='round' else 1)
            assert data['extra_dc_precision']==expected and data['dc_smoothing']==policy['smoothing']
            captures.append(data)
            if repeat:assert captures[0]==data
        row={'id':key,'image_id':ref['image_id'],'arm':ref['arm'],'distance':ref['distance'],
             'score':ref['score'],'output_sha256':ref['output_sha256'],'output_path':str(folder/'output-0.jxl'),
             'capture_path':str(folder/'capture-0.json'),'capture_sha256':study.sha(folder/'capture-0.json'),
             'repetitions':repetitions,'capture':captures[0]}
        study.append(dest/'observations.jsonl',row);rows.append(row);done.add(key)
        study.save(dest/'progress.json',{'status':'running','pid':os.getpid(),'completed':len(done),'expected':48})
        print(mode,ref['image_id'],ref['arm'],len(done),'/48',flush=True)
    for p,digest in manifest['files'].items():assert study.sha(p)==digest,p
    for row in rows:
        assert study.sha(row['output_path'])==row['output_sha256']
        assert study.sha(row['capture_path'])==row['capture_sha256']
    assert len(rows)==len(done)==48
    study.save(dest/'audit.json',{'status':'passed','cases':48,'commands':len(study.rows(dest/'commands.jsonl')),
        'exact_output_identity':48,'repeat_diagnostic_identity':4,'additive_section_checks':48,
        'source_hashes_checked':len(manifest['files'])})
    study.save(dest/'progress.json',{'status':'completed','completed':48,'expected':48})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['fixed','matched'])
    args=parser.parse_args()
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        collect(args.mode)


if __name__=='__main__':main()
