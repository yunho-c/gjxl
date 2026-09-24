#!/usr/bin/env python3
"""Small resumable DC-worker screen, alternating independent processes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/tokenization-20260922/dc-screen'
STUDY = Path('/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/gjxl-stages-q80-20260921')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import argparse, shutil
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--images',default='4')
    parser.add_argument('--efforts',default='1,7')
    parser.add_argument('--modes',default='cpu,gpu1,gpu4,gpu8,overlap1,overlap4,overlap8')
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--samples',type=int,default=2)
    parser.add_argument('--variants')
    parser.add_argument('--baseline')
    args=parser.parse_args()
    variants=json.loads(Path(args.variants).read_text()) if args.variants else {}
    run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True)
    binary=run/'capture'
    if not binary.exists():
        shutil.copy2(ROOT/'build/release/tokenization_capture',binary)
        (run/'source-commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT))
        (run/'source.diff').write_bytes(subprocess.check_output(['git','diff'],cwd=ROOT))
        files=subprocess.check_output(['git','ls-files','--others','--exclude-standard','src','tests','tools'],cwd=ROOT,text=True).splitlines()
        for name in files:
            dst=run/'new-sources'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
        (run/'binary-sha256.txt').write_text(sha(binary)+'\n')
    assert sha(binary)==(run/'binary-sha256.txt').read_text().strip()
    baseline=run/'baseline'
    if args.baseline:
        if not baseline.exists():
            shutil.copy2(args.baseline,baseline)
            (run/'baseline-sha256.txt').write_text(sha(baseline)+'\n')
        assert sha(baseline)==(run/'baseline-sha256.txt').read_text().strip()
    run_config={'runner_sha256':sha(Path(__file__)),'images':args.images,'efforts':args.efforts,'modes':args.modes,'rounds':args.rounds,'samples':args.samples,'variants':variants,'diagnostic':args.diagnostic,'baseline':args.baseline}
    if (run/'run-config.json').exists():assert json.loads((run/'run-config.json').read_text())==run_config
    else:(run/'run-config.json').write_text(json.dumps(run_config,indent=2))
    config=json.loads((STUDY/'config.json').read_text())
    images=[config['images'][int(i)] for i in args.images.split(',')]
    for im in images:assert sha(Path(im['input_path']))==im['input_sha256']
    records=[];expected={}
    for repeat in range(args.rounds):
      for im in images:
       for effort in map(int,args.efforts.split(',')):
        modes=args.modes.split(',')
        shift=(repeat//2)%len(modes);modes=modes[shift:]+modes[:shift]
        if repeat%2:modes.reverse()
        for mode in modes:
         out=run/f"r{repeat}-{im['image_id']}-e{effort}-{mode}";out.mkdir(parents=True,exist_ok=True)
         done=out/'complete.json'
         if done.exists():record=json.loads(done.read_text())
         else:
          env={k:v for k,v in os.environ.items() if not k.startswith('GJXL_EXPERIMENT_') and k != 'GJXL_GPU_TOKENIZATION'}
          env['GJXL_EXPERIMENT_DC_WORKERS']='8'
          env.update(GJXL_EXPERIMENT_TOKEN_COMPACT='0',GJXL_EXPERIMENT_TOKEN_OVERLAP='0',GJXL_EXPERIMENT_TOKEN_SHARDS='4',GJXL_EXPERIMENT_TOKEN_SCALAR_DCT8='0')
          if mode.startswith(('gpu','overlap','fresh','compact','tight','smart','joined')):
           env['GJXL_EXPERIMENT_GPU_TOKENS']='1'
           env['GJXL_EXPERIMENT_TOKEN_SHARDS']=mode.removeprefix('gpu').removeprefix('overlap').removeprefix('fresh').removeprefix('compact').removeprefix('tight').removeprefix('smart').removeprefix('joined')
          if mode.startswith(('overlap','tight','joined')):env['GJXL_EXPERIMENT_TOKEN_OVERLAP']='1'
          if mode.startswith(('compact','tight')):env['GJXL_EXPERIMENT_TOKEN_COMPACT']='1'
          if mode.startswith('fresh'):env['GJXL_EXPERIMENT_TOKEN_CACHE']='0'
          if mode.startswith(('smart','joined')):env['GJXL_EXPERIMENT_TOKEN_COMPACT']='2'
          if mode.startswith('joined'):env['GJXL_EXPERIMENT_TOKEN_SCALAR_DCT8']='2'
          env.update(variants.get(mode,{}))
          env.setdefault('GJXL_GPU_TOKENIZATION',env.get('GJXL_EXPERIMENT_GPU_TOKENS','0'))
          if args.diagnostic:env['GJXL_EXPERIMENT_TOKEN_DIAGNOSTIC']='1'
          command=[str(baseline if mode=='base' else binary),'--input',im['input_path'],'--scope','metal-public-workflow','--validation','metal-only','--gpu-aq','fully-resident','--effort',str(effort),'--distance','1.9','--cpu-threads','8','--warmups','1','--samples',str(args.samples),'--gpu-profile','stage','--gpu-profile-output',str(out/'gpu.json')]
          (out/'command.json').write_text(json.dumps({'argv':command,'environment':{k:v for k,v in env.items() if k.startswith('GJXL_EXPERIMENT_') or k == 'GJXL_GPU_TOKENIZATION'},'input_sha256':im['input_sha256']},indent=2))
          with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
           subprocess.run(command,env=env,stdout=stdout,stderr=stderr,check=True,timeout=240)
          rows=[r for r in json.loads((out/'paired.json').read_text())['rows'] if r['sample_index']>=0]
          assert len(rows)==2*args.samples and all(r['byte_equal'] and r['summary_equal'] for r in rows)
          ordinary=[r['complete_call_nanoseconds']/1e6 for r in rows if r['mode']=='ordinary']
          profiles=[r['phase_nanoseconds'] for r in rows if r['mode']=='profiled']
          record={'round':repeat,'image':im['image_id'],'effort':effort,'mode':mode,'ordinary_ms':statistics.median(ordinary),'ordinary_samples_ms':ordinary,'dc_ms':statistics.mean(r['codestream_dc_tokenization'] for r in profiles)/1e6,'ac_ms':statistics.mean(r['codestream_ac_tokenization'] for r in profiles)/1e6,'output_sha256':sha(out/'reference.jxl')}
          done.write_text(json.dumps(record,indent=2)+'\n')
         assert sha(out/'reference.jxl')==record['output_sha256']
         key=(im['image_id'],effort)
         if key in expected:assert expected[key]==record['output_sha256'],(key,mode,'byte mismatch')
         expected[key]=record['output_sha256'];records.append(record)
         print(repeat,im['resolution_class'],effort,mode,round(record['ordinary_ms'],3),round(record['dc_ms'],3),round(record['ac_ms'],3),flush=True)
         (run/'results.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Complete; all bytes identical across variants.',flush=True)

if __name__=='__main__':main()
