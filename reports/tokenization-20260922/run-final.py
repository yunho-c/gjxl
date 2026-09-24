#!/usr/bin/env python3
import json,os,subprocess,time,traceback,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
BASE=Path('/Users/yunhocho/GitHub/gjxl-tokenization-base-20260922')
def status(**kw):
 (RUN/'final-status.json').write_text(json.dumps(dict(time=time.time(),pid=os.getpid(),**kw),indent=2))
def stage(name,argv):
 status(state='running',stage=name,argv=argv)
 with (RUN/(name+'.log')).open('w') as log:
  subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name,'complete',flush=True)
try:
 if '--measure' not in sys.argv: stage('final-default-tests',['ctest','--test-dir','build/release','--output-on-failure','-j','4'])
 stage('final-profile',['python3','tools/gpu_tokenization_screen.py','--run',str(RUN/'final-profile'),'--images','0,1,2,3,4,5','--efforts','1,7,8','--modes','base,cpu,release,joined1','--rounds','5','--samples','4','--variants',str(RUN/'final-variants.json'),'--baseline',str(BASE/'build/release/tokenization_capture')])
 stage('final-batch',['python3','tools/tokenization_batch_screen.py','--run',str(RUN/'final-batch'),'--images','1,4,5','--efforts','1,7','--batches','1,4','--modes','base,release,joined','--rounds','3','--samples','5','--cpu','8','--memory-gib','36'])
 stage('final-analysis',['python3',str(RUN/'analyze.py')])
 stage('final-decode',['python3',str(RUN/'decode.py')])
 status(state='complete')
except Exception:
 status(state='failed',error=traceback.format_exc());raise
