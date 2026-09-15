#!/usr/bin/env python3
"""Current-frame native/libjxl serialization comparison with exact decoded-float checks."""
from collections import defaultdict
import fcntl
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import cross_metric
import study

ROOT=Path('build/e3-e4-tail-confirm-20260914').resolve()
PROBE=Path('build/e3-e4-tail-probe-20260914').resolve()


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        collect()


def collect():
    path=ROOT/'manifest.json'
    if not path.exists():
        m=study.read(cross_metric.FULL/'manifest.json');files=study.read(PROBE/'build.json')['files']
        selected=[r for r in study.rows(cross_metric.FULL/'observations.jsonl')
                  if r['image_id'] in cross_metric.IMAGES and r['requested_quality'] in [50,70,80,90,95]]
        assert len(selected)==240
        for p in [PROBE/'build.json',cross_metric.FULL/'manifest.json',cross_metric.FULL/'observations.jsonl',
                  Path(__file__).resolve(),Path(study.__file__).resolve()]:files[str(p)]=study.sha(p)
        for name in cross_metric.IMAGES:
            im=m['images'][name];files[im['pfm_path']]=im['pfm_sha256']
        for row in selected:files[row['output_path']]=row['output_sha256']
        study.save(path,{'files':files,'images':m['images'],'image_order':cross_metric.IMAGES,
            'arms':m['arms'],'djxl':m['djxl'],'rows':selected,'expected':240,
            'protocol':'Twelve selected inputs, four native/candidate GJXL arms, five original distances. Every native output must reproduce the prior SHA256; every same-frame tail must reproduce the prior decoded PFM SHA256. Native and tail repeat bytes and bridge state-copy/immutability checked. No new lossy decisions, no speed claim. Same-effort libjxl writer policy on the GJXL frame; stock libjxl frontend is separate.'})
    manifest=study.read(path)
    for p,digest in manifest['files'].items():assert study.sha(p)==digest,p
    observations=study.rows(ROOT/'observations.jsonl');done={r['id'] for r in observations}
    env={k:v for k,v in os.environ.items() if not k.startswith(('GJXL_','RCA_'))}
    selected=sorted(manifest['rows'],key=lambda r:(manifest['image_order'].index(r['image_id']),r['requested_quality'],r['arm']))
    for ref in selected:
        key=cross_metric.identifier(ref['image_id'],ref['arm'],ref['requested_quality'])
        if key in done:continue
        folder=ROOT/'cases'/key;folder.mkdir(parents=True,exist_ok=True)
        policy=manifest['arms'][ref['arm']];im=manifest['images'][ref['image_id']]
        env.update(GJXL_RCA_TAIL_OUTPUT=str(folder),GJXL_RCA_TAIL_EFFORT=str(policy['effort']),
                   GJXL_RCA_TAIL_DISTANCE=format(ref['distance'],'.9g'))
        if 'precision_override' in policy:env['GJXL_RCA_DC_PRECISION']=str(policy['precision_override'])
        else:env.pop('GJXL_RCA_DC_PRECISION',None)
        commands=[list(map(str,[PROBE/'gjxl_tail_probe','--input',im['pfm_path'],
            '--output',folder/'native.jxl','--raw-samples',folder/'raw.json','--effort',policy['effort'],
            '--distance',format(ref['distance'],'.9g'),'--num-threads',8,'--warmups',0,'--samples',1,
            '--dc-quantization',policy['quantization'],
            '--adaptive-dc-smoothing' if policy['smoothing'] else '--no-adaptive-dc-smoothing'])),
            [manifest['djxl'],str(folder/'tail.jxl'),str(folder/'decoded.pfm'),
             '--color_space=RGB_D65_SRG_Rel_Lin','--num_threads=1']]
        for number,argv in enumerate(commands):
            start=time.time();result=subprocess.run(argv,env=env,capture_output=True,text=True)
            study.append(ROOT/'commands.jsonl',{'argv':argv,'precision_override':env.get('GJXL_RCA_DC_PRECISION'),
                'seconds':time.time()-start,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            result.check_returncode()
            if number==0:assert study.sha(folder/'native.jxl')==ref['output_sha256'],ref['id']
        decoded_hash=study.sha(folder/'decoded.pfm');assert decoded_hash==ref['decoded_sha256'],ref['id']
        (folder/'decoded.pfm').unlink();stats=study.read(folder/'tail.json')
        assert stats['source_copy_equal'] and stats['repeat_bytes_equal'] and stats['frame_immutable']
        assert stats['extra_dc_precision']==policy.get('precision_override',0 if policy['quantization']=='round' else 1)
        row={'id':key,'image_id':ref['image_id'],'arm':ref['arm'],'requested_quality':ref['requested_quality'],
             'distance':ref['distance'],'score':ref['score'],'native_bytes':ref['encoded_bytes'],
             'tail_bytes':(folder/'tail.jxl').stat().st_size,'native_sha256':ref['output_sha256'],
             'tail_sha256':study.sha(folder/'tail.jxl'),'decoded_sha256':decoded_hash,
             'tail_stats_sha256':study.sha(folder/'tail.json'),'raw_sha256':study.sha(folder/'raw.json'),
             'folder':str(folder),'exact_decoded_identity':True}
        study.append(ROOT/'observations.jsonl',row);observations.append(row);done.add(key)
        study.save(ROOT/'progress.json',{'status':'running','pid':os.getpid(),'completed':len(done),'expected':240})
        print(ref['image_id'],ref['arm'],ref['requested_quality'],len(done),'/240',flush=True)
    for p,h in manifest['files'].items():assert study.sha(p)==h,p
    assert len(observations)==len(done)==240
    for r in observations:
        folder=Path(r['folder'])
        for file,key in [('native.jxl','native_sha256'),('tail.jxl','tail_sha256'),('tail.json','tail_stats_sha256'),('raw.json','raw_sha256')]:
            assert study.sha(folder/file)==r[key]
    commands=study.rows(ROOT/'commands.jsonl');assert len(commands)==480 and all(r['returncode']==0 for r in commands)
    study.save(ROOT/'audit.json',{'status':'passed','cases':240,'exact_native_byte_identity':240,
        'exact_decoded_float_identity':240,'successful_commands':480,'verified_source_files':len(manifest['files'])})
    study.save(ROOT/'progress.json',{'status':'completed','completed':240,'expected':240})


if __name__=='__main__':main()
