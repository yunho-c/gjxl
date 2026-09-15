#!/usr/bin/env python3
"""Freeze and ablate modular integer-mapping search with neutral byte controls."""
import fcntl
import os
from pathlib import Path
import statistics
import subprocess
import time

import cross_metric
import study
import tail_study

ROOT=Path('build/e3-e4-tail-uint-confirm-20260914').resolve()
PROBE=Path('build/e3-e4-tail-uint-probe-20260914').resolve()


def collect():
    path=ROOT/'manifest.json'
    if not path.exists():
        assert study.read(tail_study.ROOT/'audit.json')['status']=='passed'
        parent=study.read(tail_study.ROOT/'manifest.json')
        rows=[r for r in study.rows(tail_study.ROOT/'observations.jsonl') if r['requested_quality']==80]
        assert len(rows)==48
        files=study.read(PROBE/'build.json')['files']
        for p in [PROBE/'build.json',tail_study.ROOT/'observations.jsonl',tail_study.ROOT/'manifest.json',
                  Path(__file__).resolve(),Path(study.__file__).resolve()]:files[str(p)]=study.sha(p)
        for row in rows:
            im=parent['images'][row['image_id']];files[im['pfm_path']]=im['pfm_sha256']
            files[str(Path(row['folder'])/'tail.jxl')]=row['tail_sha256']
        study.save(path,{'files':files,'rows':rows,'arms':parent['arms'],'images':parent['images'],'djxl':parent['djxl'],
            'protocol':'All twelve selected inputs, four recipes, original Q80 distance. Neutral and no-modular-Uint-search variants. Neutral must reproduce both native and old tail bytes. Both variants must reproduce the same decoded-float hash. No quality changes or timing claim.'})
    m=study.read(path)
    for p,h in m['files'].items():assert study.sha(p)==h,p
    observations=study.rows(ROOT/'observations.jsonl');done={r['id'] for r in observations}
    env={k:v for k,v in os.environ.items() if not k.startswith(('GJXL_','RCA_'))}
    for ref in m['rows']:
        policy=m['arms'][ref['arm']]
        for variant in ['neutral','no-uint-search']:
            key=cross_metric.identifier(ref['image_id'],ref['arm'],variant)
            if key in done:continue
            folder=ROOT/'cases'/key;folder.mkdir(parents=True,exist_ok=True)
            env.update(GJXL_RCA_TAIL_OUTPUT=str(folder),GJXL_RCA_TAIL_EFFORT=str(policy['effort']),
                       GJXL_RCA_TAIL_DISTANCE=format(ref['distance'],'.9g'))
            if 'precision_override' in policy:env['GJXL_RCA_DC_PRECISION']=str(policy['precision_override'])
            else:env.pop('GJXL_RCA_DC_PRECISION',None)
            if variant=='no-uint-search':env['GJXL_RCA_MODULAR_UINT_NONE']='1'
            else:env.pop('GJXL_RCA_MODULAR_UINT_NONE',None)
            argv=list(map(str,[PROBE/'gjxl_tail_uint_probe','--input',m['images'][ref['image_id']]['pfm_path'],
                '--output',folder/'native.jxl','--raw-samples',folder/'raw.json','--effort',policy['effort'],
                '--distance',format(ref['distance'],'.9g'),'--num-threads',8,'--warmups',0,'--samples',1,
                '--dc-quantization',policy['quantization'],
                '--adaptive-dc-smoothing' if policy['smoothing'] else '--no-adaptive-dc-smoothing']))
            commands=[argv,[m['djxl'],str(folder/'tail.jxl'),str(folder/'decoded.pfm'),
                            '--color_space=RGB_D65_SRG_Rel_Lin','--num_threads=1']]
            for i,cmd in enumerate(commands):
                start=time.time();result=subprocess.run(cmd,env=env,capture_output=True,text=True)
                study.append(ROOT/'commands.jsonl',{'argv':cmd,'variant':variant,'returncode':result.returncode,
                    'seconds':time.time()-start,'stdout':result.stdout,'stderr':result.stderr});result.check_returncode()
                if i==0:
                    assert study.sha(folder/'native.jxl')==ref['native_sha256']
                    if variant=='neutral':assert study.sha(folder/'tail.jxl')==ref['tail_sha256']
            assert study.sha(folder/'decoded.pfm')==ref['decoded_sha256']
            (folder/'decoded.pfm').unlink();stats=study.read(folder/'tail.json')
            assert stats['source_copy_equal'] and stats['repeat_bytes_equal'] and stats['frame_immutable']
            row={'id':key,'image_id':ref['image_id'],'arm':ref['arm'],'variant':variant,
                 'native_bytes':ref['native_bytes'],'tail_bytes':stats['bytes'],
                 'tail_sha256':study.sha(folder/'tail.jxl'),'native_sha256':ref['native_sha256'],
                 'decoded_sha256':ref['decoded_sha256'],'folder':str(folder),
                 'stats_sha256':study.sha(folder/'tail.json'),'raw_sha256':study.sha(folder/'raw.json')}
            study.append(ROOT/'observations.jsonl',row);observations.append(row);done.add(key)
            study.save(ROOT/'progress.json',{'status':'running','pid':os.getpid(),'completed':len(done),'expected':96})
            print(ref['image_id'],ref['arm'],variant,len(done),'/96',flush=True)
    for p,h in m['files'].items():assert study.sha(p)==h,p
    assert len(done)==len(observations)==96
    for r in observations:
        for name,key in [('tail.jxl','tail_sha256'),('native.jxl','native_sha256'),('tail.json','stats_sha256'),('raw.json','raw_sha256')]:
            assert study.sha(Path(r['folder'])/name)==r[key]
    commands=study.rows(ROOT/'commands.jsonl');assert len(commands)==192 and all(r['returncode']==0 for r in commands)
    study.save(ROOT/'audit.json',{'status':'passed','cases':96,'exact_neutral_tail_controls':48,
        'exact_native_byte_controls':96,'exact_decoded_float_checks':96,'successful_commands':192})
    study.save(ROOT/'progress.json',{'status':'completed','completed':96,'expected':96})
    data={(r['image_id'],r['arm'],r['variant']):r for r in observations};rows=[]
    for ref in m['rows']:
        name=ref['image_id'];arm=ref['arm'];a=data[name,arm,'neutral'];b=data[name,arm,'no-uint-search'];native=ref['native_bytes']
        rows.append({'image_id':name,'arm':arm,
            'tail_with_search_vs_native_pp':100*(a['tail_bytes']/native-1),
            'tail_without_search_vs_native_pp':100*(b['tail_bytes']/native-1),
            'saving_from_search_pp':100*(b['tail_bytes']-a['tail_bytes'])/native,
            'search_changes_bytes':a['tail_sha256']!=b['tail_sha256']})
    summary=[]
    for arm in m['arms']:
        selected=[r for r in rows if r['arm']==arm]
        row={'arm':arm,'n':len(selected),'search_changes_bytes':sum(r['search_changes_bytes'] for r in selected)}
        for k in ['tail_with_search_vs_native_pp','tail_without_search_vs_native_pp','saving_from_search_pp']:
            row[k]=statistics.mean(r[k] for r in selected)
        summary.append(row);print(row)
    study.save(ROOT/'analysis.json',{'aggregation':'Arithmetic mean of file-size differences divided by each original native file size. Additive percentage-point decomposition at original Q80, not BD-rate.',
        'summary':summary,'images':rows})


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);collect()


if __name__=='__main__':main()
