#!/usr/bin/env python3
"""Resumable complete-call B1/B4 comparison with a shared CPU limit."""
import argparse,csv,hashlib,json,os,shutil,statistics,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/Users/yunhocho/GitHub/gjxl-tokenization-base-20260922')
STUDY=Path('/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/gjxl-stages-q80-20260921')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--images',default='1,4');p.add_argument('--efforts',default='1,7');p.add_argument('--batches',default='1,4');p.add_argument('--rounds',type=int,default=2);p.add_argument('--samples',type=int,default=5);p.add_argument('--modes',default='base,cpu,release,joined');p.add_argument('--cpu',type=int,default=8);p.add_argument('--memory-gib',type=int,default=24);p.add_argument('--variants');a=p.parse_args()
 run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=True)
 variants=json.loads(Path(a.variants).read_text()) if a.variants else {}
 config=dict(vars(a),runner_sha256=sha(Path(__file__)),variant_overrides=variants)
 if (run/'run-config.json').exists():assert json.loads((run/'run-config.json').read_text())==config
 else:(run/'run-config.json').write_text(json.dumps(config,indent=2))
 for name,root in [('base',BASE),('candidate',ROOT)]:
  binary=run/name;stamp=run/(name+'-sha256.txt')
  if not binary.exists():shutil.copy2(root/'build/release/tokenization_batch',binary);stamp.write_text(sha(binary)+'\n')
  assert sha(binary)==stamp.read_text().strip()
 if not (run/'source.diff').exists():
  (run/'source-commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT))
  (run/'source.diff').write_bytes(subprocess.check_output(['git','diff'],cwd=ROOT))
  for name in subprocess.check_output(['git','ls-files','--others','--exclude-standard','src','tests','tools'],cwd=ROOT,text=True).splitlines():
   dest=run/'new-sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
 images=[json.loads((STUDY/'config.json').read_text())['images'][int(i)] for i in a.images.split(',')]
 for im in images:assert sha(Path(im['input_path']))==im['input_sha256']
 results=[];expected={}
 for repeat in range(a.rounds):
  for im in images:
   for effort in map(int,a.efforts.split(',')):
    for batch in map(int,a.batches.split(',')):
     modes=a.modes.split(',');shift=(repeat//2)%len(modes);modes=modes[shift:]+modes[:shift]
     if repeat%2:modes.reverse()
     for mode in modes:
      case=run/f"r{repeat}-{im['image_id']}-e{effort}-B{batch}-{mode}";case.mkdir(parents=True,exist_ok=True);done=case/'complete.json'
      if done.exists():record=json.loads(done.read_text())
      else:
       env={k:v for k,v in os.environ.items() if not k.startswith('GJXL_EXPERIMENT_') and k != 'GJXL_GPU_TOKENIZATION'};env['GJXL_EXPERIMENT_DC_WORKERS']='8'
       env.update(GJXL_EXPERIMENT_TOKEN_COMPACT='0',GJXL_EXPERIMENT_TOKEN_OVERLAP='0',GJXL_EXPERIMENT_TOKEN_SHARDS='4',GJXL_EXPERIMENT_TOKEN_SCALAR_DCT8='0')
       if mode=='release':env['GJXL_EXPERIMENT_PARALLEL_TOKEN_RELEASE']='1'
       if mode in ('gpu','joined'):env.update(GJXL_EXPERIMENT_GPU_TOKENS='1',GJXL_EXPERIMENT_TOKEN_SHARDS='1',GJXL_EXPERIMENT_TOKEN_COMPACT='2')
       if mode=='joined':env.update(GJXL_EXPERIMENT_TOKEN_OVERLAP='1',GJXL_EXPERIMENT_TOKEN_SCALAR_DCT8='2')
       env.update(variants.get(mode,{}))
       env.setdefault('GJXL_GPU_TOKENIZATION',env.get('GJXL_EXPERIMENT_GPU_TOKENS','0'))
       attempt=0
       while (case/f'attempt{attempt}').exists():attempt+=1
       data=case/f'attempt{attempt}'
       command=[str(run/('base' if mode=='base' else 'candidate')),im['input_path'],str(effort),str(a.cpu),str(batch),str(a.samples),str(data),str(a.memory_gib*2**30)]
       (case/f'command{attempt}.json').write_text(json.dumps({'argv':command,'environment':{k:v for k,v in env.items() if k.startswith('GJXL_EXPERIMENT_') or k == 'GJXL_GPU_TOKENIZATION'},'input_sha256':im['input_sha256']},indent=2))
       (case/f'vm-before{attempt}.txt').write_bytes(subprocess.check_output(['vm_stat']))
       start=time.monotonic()
       with (case/f'stdout{attempt}.txt').open('w') as stdout,(case/f'stderr{attempt}.txt').open('w') as stderr:subprocess.run(command,env=env,stdout=stdout,stderr=stderr,check=True,timeout=600)
       (case/f'vm-after{attempt}.txt').write_bytes(subprocess.check_output(['vm_stat']))
       rows=list(csv.DictReader((data/'samples.csv').open()));timed=[r for r in rows if int(r['sample'])>=0];assert len(timed)==a.samples
       walls=[int(r['wall_ns'])/1e6 for r in timed]
       record={'round':repeat,'image':im['image_id'],'effort':effort,'batch':batch,'mode':mode,'wall_ms':statistics.median(walls),'wall_samples_ms':walls,'fps':1000*batch/statistics.median(walls),'peak_backing_bytes':max(int(r['peak_backing_bytes']) for r in rows),'peak_cpu_slots':max(int(r['peak_cpu_slots']) for r in rows),'output_sha256':sha(data/'reference.jxl'),'artifacts':str(data.relative_to(run)),'elapsed_s':time.monotonic()-start}
       done.write_text(json.dumps(record,indent=2)+'\n')
       with (run/'ledger.jsonl').open('a') as ledger:ledger.write(json.dumps(record)+'\n')
      assert sha(run/record['artifacts']/'reference.jxl')==record['output_sha256']
      key=(im['image_id'],effort)
      if key in expected:assert expected[key]==record['output_sha256'],(key,batch,mode,'byte mismatch')
      expected[key]=record['output_sha256'];results.append(record)
      (run/'results.json').write_text(json.dumps(results,indent=2)+'\n')
      print(repeat,im['resolution_class'],effort,'B'+str(batch),mode,round(record['wall_ms'],3),round(record['fps'],2),'peak_GiB',round(record['peak_backing_bytes']/2**30,2),flush=True)
 print('Complete; all bytes identical across variants, batches and rounds.',flush=True)
if __name__=='__main__':main()
