#!/usr/bin/env python3
from pathlib import Path
import datetime,json,subprocess,sys,time,xml.etree.ElementTree as ET,struct,math,hashlib
P=Path(__file__).resolve().parent
W=Path('/Users/yunhocho/GitHub/gjxl-butteraugli-production')
B=Path('/Users/yunhocho/GitHub/gjxl-butteraugli-production-base')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,data):
 f=P/name;t=f.with_suffix(f.suffix+'.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(f)
def run(name,cmd):
 save('qualification-state.json',{'status':'running','step':name,'at':stamp()})
 save(name+'-command.json',cmd)
 with (P/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'{name} failed with {r.returncode}; see log')
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
try:
 while not (P/'full-suite-final-exit.json').exists():time.sleep(2)
 root=ET.parse(P/'full-suite-final.xml').getroot()
 failed=[c.attrib['name'] for c in root.findall('testcase') if c.find('failure') is not None]
 assert len(root.findall('testcase')) == 156
 if failed:
  if failed!=['metal_aq_strategy_metadata']:raise RuntimeError('New full-suite failure: '+str(failed))
  cases=ET.parse(P/'baseline-failure-control.xml').getroot().findall('testcase')
  baseline_failed=[c.attrib['name'] for c in cases if c.find('failure') is not None]
  assert baseline_failed==failed
  save('known-failure.json',{'test':'metal_aq_strategy_metadata','baseline_reproduced':True,'candidate_log':'full-suite-final.log','baseline_log':'baseline-failure-control.log'})
 run('canonical-parity',['python3',str(P/'parity.py'),'--baseline-build',str(P/'baseline'),'--candidate-build',str(P/'candidate'),'--corpus','/Users/yunhocho/GitHub/gjxl-libjxl-comparison/build/libjxl-comparison/corpus-phase1-pilot-pinned/canonical','--decoder','/Users/yunhocho/GitHub/gjxl/build/pinned-libjxl/tools/djxl','--output',str(P/'canonical-parity-v2')])
 summary=json.loads((P/'canonical-parity-v2/summary.json').read_text());assert summary['cases']==56
 checks=[]
 for row in summary['decodes']:
  files=[Path(f['file']) for f in row['outputs']];assert sha(files[0])==sha(files[1])
  with files[0].open('rb') as f:
   magic=f.readline().strip();assert magic==b'PF'
   width,height=map(int,f.readline().split());scale=float(f.readline());data=f.read()
  assert len(data)==width*height*3*4
  count=sum(not math.isfinite(x[0]) for x in struct.iter_unpack('<f' if scale<0 else '>f',data));assert count==0
  checks.append({'name':row['name'],'width':width,'height':height,'nonfinite':count,'decoded_sha256':sha(files[0]),'both_decodes_byte_equal':True})
 save('canonical-parity/finite-pixels.json',{'at':stamp(),'checks':checks})
 for cohort in json.loads((P/'timing-plan.json').read_text())['cohorts']:
  run(cohort['name'],['python3',str(P/'run_paired.py'),*cohort['args']])
 save('qualification-state.json',{'status':'complete','at':stamp()})
 print('Qualification complete',flush=True)
except Exception as e:
 save('qualification-state.json',{'status':'failed','error':repr(e),'at':stamp()})
 raise
