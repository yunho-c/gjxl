import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
def status(**kw):(RUN/'group-final-status.json').write_text(json.dumps(dict(time=time.time(),pid=os.getpid(),**kw),indent=2))
def stage(name,argv,env=None,allow_failure=False):
 status(state='running',stage=name,argv=argv)
 with (RUN/(name+'.log')).open('w') as f:r=subprocess.run(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode and not allow_failure:raise RuntimeError((name,r.returncode))
 print(name,r.returncode,flush=True)
try:
 stage('group-final-default-tests',['ctest','--test-dir','build/release','--output-on-failure','-j','4'],allow_failure=True)
 env=dict(os.environ,GJXL_EXPERIMENT_GPU_TOKENS='1',GJXL_EXPERIMENT_TOKEN_COMPACT='2',GJXL_EXPERIMENT_TOKEN_OVERLAP='1',GJXL_EXPERIMENT_TOKEN_SHARDS='1',GJXL_EXPERIMENT_TOKEN_SCALAR_DCT8='2',GJXL_EXPERIMENT_TOKEN_GROUP_DCT8='1',GJXL_EXPERIMENT_TOKEN_GROUP_THREADS='256',GJXL_EXPERIMENT_PARALLEL_TOKEN_RELEASE='1')
 stage('group-final-focused-tests',['ctest','--test-dir','build/release','--output-on-failure','-R','^(resident_workflow_storage_plan|workflow_storage_plan|workflow_admission|metal_ac_tokenization|metal_ac_tokenization_failure|metal_cache_admission|metal_completed_frame|worker_launch_failure_metal_sections|dc_parallel_tokenization)$'],env)
 for n in ['tokenization_capture','tokenization_batch']:
  stage('group-final-build-'+n,json.loads((ROOT/'build/release'/ (n+'-command.json')).read_text()),dict(os.environ,DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer'))
 stage('group-final-profile',['python3','tools/gpu_tokenization_screen.py','--run',str(RUN/'group-final-profile'),'--images','0,1,2,3,4,5','--efforts','1','--modes','release,joined1,group256','--rounds','5','--samples','4','--variants',str(RUN/'group-dct8-qualification-variants.json')])
 stage('group-final-batch',['python3','tools/tokenization_batch_screen.py','--run',str(RUN/'group-final-batch'),'--images','1,4,5','--efforts','1','--batches','1,4','--modes','release,joined,group256','--rounds','3','--samples','5','--cpu','8','--memory-gib','36','--variants',str(RUN/'group-dct8-qualification-variants.json')])
 status(state='complete')
except Exception:
 status(state='failed',error=traceback.format_exc());raise
