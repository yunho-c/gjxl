#!/usr/bin/env python3
"""Bounded score-80 calibration and independent Butteraugli checks.

Never infer matched quality from distance. Failed targets remain terminal
under their original persistent probe budgets. No timing claims are made.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import statistics
import struct
import subprocess
import time

import study

FULL = Path('build/e3-e4-dc-grid-confirm-20260914').resolve()
ROOT = Path('build/e3-e4-cross-metric-20260914').resolve()
IMAGES = [
    'kodak/01','kodak/17',
    'clic2024_test/097cb426910ba8ce2525dd8bb7fb1777',
    'clic2024_test/d1a9be98d1936065967adac50a6fb750',
    'clic2024_test/b51d5fb537246482ef7a7ab63f093cab',
    'kodak/08','kodak/13','clic2024_test/28d24b9c83de066597ff96a68769884f',
    'unsplash/alpine_lake/24mp','unsplash/campus_interior/12mp',
    'unsplash/campus_interior/48mp','unsplash/forest_stream/48mp',
]


def f32(value):
    return struct.unpack('f',struct.pack('f',value))[0]


def identifier(*values):
    return hashlib.sha256('|'.join(map(str,values)).encode()).hexdigest()[:24]


def freeze():
    assert not (ROOT/'manifest.json').exists()
    assert study.read(FULL/'audit.json')['status']=='passed'
    base=study.read(FULL/'manifest.json')
    stock=study.read(study.LIBJXL/'metadata.json')
    old=study.read(study.BASE/'e4-frontend-aq-20260914/manifest.json')
    arms={key:{**value,'encoder':'gjxl','binary':base['binary'],'revision':base['revision']}
          for key,value in base['arms'].items()}
    for e in (3,4):
        arms[f'libjxl-e{e}']={'encoder':'libjxl','effort':e,'binary':stock['benchmark'],
                              'revision':stock['libjxl_revision']}
    files=dict(base['files'])
    for path,digest in stock['tool_hashes'].items():
        if path in files:assert files[path]==digest,path
    files.update(stock['tool_hashes'])
    files[old['butteraugli']]=old['files'][old['butteraugli']]
    for p in (FULL/'manifest.json',FULL/'observations.jsonl',FULL/'audit.json',
              study.LIBJXL/'scores.jsonl',Path(__file__).resolve(),
              Path(study.__file__).resolve(),ROOT/'PROTOCOL.md'):
        files[str(p)]=study.sha(p)
    seeds=[]
    for row in study.rows(FULL/'observations.jsonl'):
        if row['image_id'] in IMAGES:
            seeds.append({**row,'seed':True})
    for row in study.rows(study.LIBJXL/'scores.jsonl'):
        if row['image_id'] in IMAGES and row['effort'] in (3,4) and row['resampling']==1:
            seeds.append({**row,'arm':f'libjxl-e{row["effort"]}',
                          'id':identifier('seed',row['image_id'],row['effort'],row['distance']), 'seed':True})
    for row in seeds:
        assert study.sha(row['output_path'])==row['output_sha256']
    for path,digest in files.items():
        assert study.sha(path)==digest,path
    study.save(ROOT/'manifest.json',{'files':files,'arms':arms,'images':base['images'],
        'image_order':IMAGES,'djxl':base['djxl'],'scorer':base['scorer'],
        'butteraugli':old['butteraugli'],'metric_version':base['metric_version'],
        'target':80,'tolerance':0.025,'pair_tolerance':0.05,'max_probes_per_target':14,
        'distance_bounds':[0.05,9.5],'free_disk_floor':6*1024**3,
        'selection':'Five diagnostic inputs, three independently selected small-image controls, four large scene/resolution controls; not a representative population estimate',
        'seeds':seeds})
    print('Frozen',len(seeds),'verified seed observations and 72 score targets',flush=True)


class CrossMetric:
    def __init__(self):
        self.m=study.read(ROOT/'manifest.json')
        for path,digest in self.m['files'].items():
            assert study.sha(path)==digest,path
        self.data={r['id']:r for r in [*self.m['seeds'],*study.rows(ROOT/'observations.jsonl')]}
        self.scorer=None;self.image=None;self.err=None
        self.env={k:v for k,v in os.environ.items() if not k.startswith(('GJXL_','RCA_'))}
        self.env['RAYON_NUM_THREADS']='8'
        for r in self.data.values():
            assert study.sha(r['output_path'])==r['output_sha256']

    def command(self,argv):
        argv=list(map(str,argv));start=time.time()
        result=subprocess.run(argv,env=self.env,capture_output=True,text=True)
        study.append(ROOT/'commands.jsonl',{'argv':argv,'precision_override':self.env.get('GJXL_RCA_DC_PRECISION'),
            'started':start,'seconds':time.time()-start,'returncode':result.returncode,
            'stdout':result.stdout,'stderr':result.stderr})
        result.check_returncode();return result.stdout

    def close(self):
        if self.scorer is not None:
            self.scorer.stdin.close();self.scorer.wait(timeout=30)
            self.scorer=None;self.image=None;self.err.close()

    def response(self):
        if not select.select([self.scorer.stdout],[],[],600)[0]:
            raise RuntimeError('Scorer observation timed out')
        line=self.scorer.stdout.readline()
        if not line:raise RuntimeError('Scorer exited')
        response=json.loads(line)
        if 'error' in response:raise RuntimeError(response)
        return response

    def score(self,image,decoded):
        if self.image!=image['image_id']:
            self.close();self.err=(ROOT/'scorer.stderr').open('a')
            self.scorer=subprocess.Popen([self.m['scorer'],image['pfm_path'],'auto'],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,
                text=True,bufsize=1,env=self.env)
            info=self.response()
            assert info['ready'] and info['fast_ssim2_revision']==self.m['metric_version']['fast_ssim2_revision']
            assert (info['width'],info['height'])==(image['width'],image['height'])
            self.image=image['image_id']
        self.scorer.stdin.write(json.dumps({'path':str(decoded)})+'\n');self.scorer.stdin.flush()
        value=self.response()['score'];assert math.isfinite(value);return value

    def encode(self,image,arm,distance,folder):
        if shutil.disk_usage(ROOT).free<self.m['free_disk_floor']:
            raise RuntimeError('Free disk floor reached')
        folder.mkdir(parents=True,exist_ok=True)
        policy=self.m['arms'][arm];output=folder/'output.jxl';raw=folder/'raw.json'
        if 'precision_override' in policy:self.env['GJXL_RCA_DC_PRECISION']=str(policy['precision_override'])
        else:self.env.pop('GJXL_RCA_DC_PRECISION',None)
        argv=[policy['binary'],'--input',image['pfm_path'],'--output',output,'--raw-samples',raw,
              '--effort',policy['effort'],'--distance',format(distance,'.9g'),
              '--num-threads',8,'--warmups',0,'--samples',1]
        if policy['encoder']=='gjxl':
            argv+=['--dc-quantization',policy['quantization'],
                   '--adaptive-dc-smoothing' if policy['smoothing'] else '--no-adaptive-dc-smoothing']
        self.command(argv);record=study.read(raw)
        assert record['revision']==policy['revision'] and record['effort']==policy['effort']
        assert record['thread_count']==8
        assert record['sample_count']==1 and record['samples'][0]['encoded_bytes']==output.stat().st_size
        if policy['encoder']=='gjxl':
            assert not record['stage_profile_enabled']
            assert record['dc_quantization']==policy['quantization']
            assert record['adaptive_dc_smoothing']==policy['smoothing']
        return output

    def decode(self,output,folder):
        folder.mkdir(parents=True,exist_ok=True);decoded=folder/'decoded.pfm'
        self.command([self.m['djxl'],output,decoded,'--color_space=RGB_D65_SRG_Rel_Lin','--num_threads=1'])
        return decoded

    def probe(self,image,arm,distance):
        distance=f32(distance);key=identifier(image['image_id'],arm,distance.hex())
        if key in self.data:return self.data[key]
        folder=ROOT/'cases'/key;output=self.encode(image,arm,distance,folder)
        decoded=self.decode(output,folder);score=self.score(image,decoded)
        row={'id':key,'image_id':image['image_id'],'arm':arm,'distance':distance,
             'score':score,'encoded_bytes':output.stat().st_size,'output_path':str(output),
             'output_sha256':study.sha(output),'decoded_sha256':study.sha(decoded),
             'raw_sha256':study.sha(folder/'raw.json'),'seed':False,'time':time.time()}
        decoded.unlink();study.append(ROOT/'observations.jsonl',row);self.data[key]=row
        return row

    def calibrate(self,image,arm):
        key=identifier(image['image_id'],arm);path=ROOT/'targets'/f'{key}.json';path.parent.mkdir(exist_ok=True)
        if path.exists():return study.read(path)
        attempt_path=path.with_suffix('.attempts.json')
        attempts=study.read(attempt_path)['attempts'] if attempt_path.exists() else 0
        target=self.m['target'];tolerance=self.m['tolerance']
        while True:
            trials=sorted([r for r in self.data.values() if r['image_id']==image['image_id'] and r['arm']==arm],
                          key=lambda r:r['distance'])
            best=min(trials,key=lambda r:abs(r['score']-target))
            if abs(best['score']-target)<=tolerance or attempts>=self.m['max_probes_per_target']:break
            brackets=[(a,b) for a,b in zip(trials,trials[1:]) if (a['score']-target)*(b['score']-target)<0]
            if not brackets:break
            a,b=min(brackets,key=lambda ab:ab[1]['distance']-ab[0]['distance'])
            weight=max(.1,min(.9,(target-a['score'])/(b['score']-a['score'])))
            distance=f32(math.exp(math.log(a['distance'])*(1-weight)+math.log(b['distance'])*weight))
            distance=f32(max(self.m['distance_bounds'][0],min(self.m['distance_bounds'][1],distance)))
            if distance in {f32(r['distance']) for r in trials}:break
            attempts+=1;study.save(attempt_path,{'attempts':attempts})
            self.probe(image,arm,distance)
        result={'image_id':image['image_id'],'arm':arm,'target':target,'attempts':attempts,
                'observation':best['id'],'score':best['score'],'error':best['score']-target,
                'matched':abs(best['score']-target)<=tolerance}
        study.save(path,result);return result

    def run(self):
        matches=study.rows(ROOT/'matches.jsonl');done={(r['image_id'],r['arm']) for r in matches}
        controls={r['id']:r for r in study.rows(ROOT/'stock-controls.jsonl')}
        try:
            for name in self.m['image_order']:
                image=self.m['images'][name]
                for e in (3,4):
                    arm=f'libjxl-e{e}';key=identifier(name,arm,'control')
                    if key in controls:continue
                    ref=next(r for r in self.m['seeds'] if r['image_id']==name and r['arm']==arm and r['requested_quality']==80)
                    folder=ROOT/'controls'/key;output=self.encode(image,arm,ref['distance'],folder)
                    assert study.sha(output)==ref['output_sha256']
                    decoded=self.decode(output,folder);score=self.score(image,decoded)
                    assert study.sha(decoded)==ref['decoded_sha256'] and abs(score-ref['score'])<1e-9
                    decoded.unlink();row={'id':key,'image_id':name,'arm':arm,'output_sha256':study.sha(output)}
                    study.append(ROOT/'stock-controls.jsonl',row);controls[key]=row
                for arm in self.m['arms']:
                    if (name,arm) in done:continue
                    match=self.calibrate(image,arm);row=self.data[match['observation']]
                    folder=ROOT/'selected'/identifier(name,arm)
                    decoded=self.decode(row['output_path'],folder)
                    assert study.sha(decoded)==row['decoded_sha256']
                    text=self.command([self.m['butteraugli'],image['pfm_path'],decoded,
                        '--colorspace','RGB_D65_SRG_Rel_Lin','--intensity_target',80])
                    value=float(text.splitlines()[0]);assert math.isfinite(value) and value>=0
                    result={**match,'output_path':row['output_path'],'output_sha256':row['output_sha256'],
                            'decoded_sha256':row['decoded_sha256'],'encoded_bytes':row['encoded_bytes'],
                            'distance':row['distance'],'butteraugli':value}
                    decoded.unlink();study.append(ROOT/'matches.jsonl',result);matches.append(result);done.add((name,arm))
                    study.save(ROOT/'progress.json',{'status':'running','pid':os.getpid(),'matches':len(done),
                               'matched':sum(r['matched'] for r in matches),'expected':72,'time':time.time()})
                    print(name,arm,'matched' if result['matched'] else 'UNRESOLVED',
                          f"score={result['score']:.5f}",f'BA={value:.5f}',len(done),'/72',flush=True)
            self.close();assert len(done)==72
            for path,digest in self.m['files'].items():assert study.sha(path)==digest,path
            for row in matches:assert study.sha(row['output_path'])==row['output_sha256']
            study.save(ROOT/'progress.json',{'status':'completed','targets':72,
                       'matched':sum(r['matched'] for r in matches),'time':time.time()})
        finally:self.close()


def analyze():
    manifest=study.read(ROOT/'manifest.json');data={(r['image_id'],r['arm']):r for r in study.rows(ROOT/'matches.jsonl')}
    comparisons=[]
    pairs=[]
    for e in (3,4):
        candidate=f'e{e}-round-smooth0-precision1'
        pairs.extend([(candidate,study.native(e)),(candidate,f'libjxl-e{e}'),(study.native(e),f'libjxl-e{e}')])
    pairs.append(('e3-round-smooth0-precision1','libjxl-e4'))
    for name in manifest['image_order']:
        for candidate,reference in pairs:
            a=data.get((name,reference));b=data.get((name,candidate))
            row={'image_id':name,'candidate':candidate,'reference':reference,'status':'incomplete'}
            if a and b:
                row.update(score_difference=b['score']-a['score'],native_score=a['score'],candidate_score=b['score'])
                row['status']='matched' if a['matched'] and b['matched'] and abs(row['score_difference'])<=manifest['pair_tolerance'] else 'unresolved'
                if row['status']=='matched':
                    row.update(bytes_percent=100*(b['encoded_bytes']/a['encoded_bytes']-1),
                               butteraugli_percent=100*(b['butteraugli']/a['butteraugli']-1))
            comparisons.append(row)
    summary=[]
    for candidate,reference in pairs:
        values=[r for r in comparisons if r['candidate']==candidate and r['reference']==reference and r['status']=='matched']
        row={'candidate':candidate,'reference':reference,'matched':len(values),'expected':len(manifest['image_order'])}
        if values:
            for key in ('bytes_percent','butteraugli_percent'):
                row[key]=100*math.expm1(statistics.mean(math.log1p(r[key]/100) for r in values))
            row['maximum_score_difference']=max(abs(r['score_difference']) for r in values)
            row['aggregation_scope']='all selected images' if len(values)==len(manifest['image_order']) else 'matched subset only'
        summary.append(row)
    study.save(ROOT/'analysis.json',{'metric':'actual matched SSIMULACRA2 point near 80; not BD-rate',
               'summary':summary,'images':comparisons})
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','run','analyze'])
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='freeze':freeze()
        elif args.action=='analyze':analyze()
        else:
            task=CrossMetric()
            try:task.run()
            except BaseException as error:
                study.save(ROOT/'progress.json',{'status':'stopped','error':repr(error),'time':time.time()})
                raise


if __name__=='__main__':main()
