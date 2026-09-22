#!/usr/bin/env python3
"""Frozen-binary paired complete-call or stage comparisons, resumable per call."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,os,statistics,subprocess,time
HERE=Path(__file__).resolve().parent
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True)
parser.add_argument('--baseline',default='baseline-v2');parser.add_argument('--output',required=True);parser.add_argument('--profile',action='store_true')
parser.add_argument('--pairs',type=int,default=3);parser.add_argument('--warmups',type=int,default=2)
parser.add_argument('--samples',type=int,default=3);parser.add_argument('--job',action='append',default=[])
args=parser.parse_args();out=HERE/args.output;out.mkdir(parents=True,exist_ok=True)
corpus=Path('/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/corpus/pfm')
jobs={'alpine24-e7':('unsplash/alpine_lake/24mp.pfm',7,1.9),
      'forest48-e10':('unsplash/forest_stream/48mp.pfm',10,1.9),
      'campus12-e7':('unsplash/campus_interior/12mp.pfm',7,1.9),
      'alpine12-e10-d08':('unsplash/alpine_lake/12mp.pfm',10,0.8),
      'forest24-e7-d08':('unsplash/forest_stream/24mp.pfm',7,0.8),
      'campus48-e10-d08':('unsplash/campus_interior/48mp.pfm',10,0.8),
      'kodak01-e7':('kodak/01.pfm',7,1.9),
      'kodak17-e10-d08':('kodak/17.pfm',10,0.8)}
selected=args.job or ['alpine24-e7','forest48-e10']
binaries={'baseline':HERE/args.baseline/'encode_probe','candidate':HERE/args.candidate/'encode_probe'}
identity={'protocol_sha256':sha(Path(__file__)),'binary_sha256':{k:sha(v) for k,v in binaries.items()},
          'binaries':{k:str(v) for k,v in binaries.items()},'jobs':{k:jobs[k] for k in selected},
          'input_sha256':{k:sha(corpus/jobs[k][0]) for k in selected},'pairs':args.pairs,
          'warmups':args.warmups,'samples':args.samples,'profile':args.profile,'cpu_threads':8}
identity=json.loads(json.dumps(identity))
if (out/'identity.json').exists():assert json.loads((out/'identity.json').read_text())==identity
else:save(out/'identity.json',identity)
(out/'protocol.py').write_text(Path(__file__).read_text())
def competitors():
 bad=[]
 for line in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines():
  fields=line.strip().split(None,2)
  if len(fields)!=3:continue
  pid,state,command=fields;name=Path(command).name
  if int(pid)==os.getpid() or any(c in state for c in 'TZ'):continue
  if name.startswith('gjxl_') or name in ['encode_probe','capture','ctest','ninja','clang','clang++','metal','xctrace','filter_screen','rolling_screen','adjacent_screen','norm_screen','interior_screen','short_screen','malta_screen','malta_adj_screen','vertical_screen','mask_filter_screen','mask_reduction_screen']:
   bad.append(line)
 return bad
def validate_case(case):
 d=json.loads((case/'samples.json').read_text())
 assert len(d['samples'])==args.samples and all(t>0 for t in d['samples'])
 assert d['profiled']==args.profile and d['byte_equal'] and d['summary_equal']
 metrics={'complete_call_ms':statistics.median(d['samples'])/1e6}
 if args.profile:
  g=json.loads((case/'gpu.json').read_text());s=g['workloads'][0]['samples'];assert len(s)==args.samples
  vals=collections.defaultdict(list)
  for sample in s:
   byid=collections.defaultdict(int);intervals=[]
   for sub in sample['submissions']:
    for stage in sub['stages']:
     assert stage['timestamp_valid'] and stage['end_timestamp']>=stage['begin_timestamp']
     assert stage['gpu_nanoseconds']==stage['end_timestamp']-stage['begin_timestamp']
     byid[stage['stage_id']]+=stage['gpu_nanoseconds'];intervals.append((stage['begin_timestamp'],stage['end_timestamp']))
   intervals.sort();assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
   total=sum(byid.values());low=sum(v for k,v in byid.items() if k.endswith('.low_medium'))
   vals['gpu_total_ms'].append(total/1e6);vals['low_medium_ms'].append(low/1e6)
   vals['other_gpu_ms'].append((total-low)/1e6)
   reduction=byid['butteraugli.resident_reduction']
   vals['resident_reduction_ms'].append(reduction/1e6)
   maskmain=byid['butteraugli.mask.main']
   vals['mask_main_ms'].append(maskmain/1e6)
   vals['mask_main_and_reduction_ms'].append((maskmain+reduction)/1e6)
   vals['affected_filter_reduction_ms'].append((low+reduction)/1e6)
   subfinal=byid['butteraugli.mask_final.sub']
   malta=sum(v for k,v in byid.items() if k.startswith('butteraugli.malta.'))
   vals['mask_final_sub_ms'].append(subfinal/1e6)
   vals['filter_reduction_subfinal_ms'].append((low+reduction+subfinal)/1e6)
   vals['malta_ms'].append(malta/1e6)
   vals['ultra_ms'].append(sum(v for k,v in byid.items() if k.endswith('.ultra_x') or k.endswith('.ultra_y'))/1e6)
   vals['opsin_ms'].append(sum(v for k,v in byid.items() if k.endswith('.opsin'))/1e6)
   vals['high_ms'].append(sum(v for k,v in byid.items() if k.endswith('.high_x') or k.endswith('.high_y'))/1e6)
   vals['medium_b_ms'].append(sum(v for k,v in byid.items() if k.endswith('.medium_b'))/1e6)
   vals['mask_all_ms'].append(sum(v for k,v in byid.items() if k.endswith('.mask') or k in ['butteraugli.mask.main','butteraugli.mask_final.sub'])/1e6)
   vals['filter_reduction_subfinal_malta_ms'].append((low+reduction+subfinal+malta)/1e6)
   vals['butteraugli_all_ms'].append(sum(v for k,v in byid.items() if k.startswith('butteraugli.') or k.startswith('frontend.prepare_aq.reference.'))/1e6)
  metrics.update({k:statistics.median(v) for k,v in vals.items()})
 return {'metrics':metrics,'codestream_sha256':sha(case/'reference.jxl'),'encoded_bytes':d['encoded_bytes'],
         'submissions':d['submissions'],'files':{f.name:sha(f) for f in case.iterdir() if f.is_file() and f.name!='complete.json'}}
results=[]
for job in selected:
 for pair in range(args.pairs):
  sides={}
  for side in (['baseline','candidate'] if pair%2==0 else ['candidate','baseline']):
   case=out/job/f'{pair:02d}-{side}';case.mkdir(parents=True,exist_ok=True)
   if (case/'complete.json').exists():
    r=json.loads((case/'complete.json').read_text());assert validate_case(case)==r;sides[side]=r;continue
   bad=competitors()
   if bad:raise RuntimeError('Competing work: '+str(bad))
   rel,e,d=jobs[job]
   cmd=[str(binaries[side]),'--input',str(corpus/rel),'--scope','metal-public-workflow',
        '--validation','metal-only','--gpu-aq','fully-resident','--effort',str(e),'--distance',str(d),
        '--cpu-threads','8','--warmups',str(args.warmups),'--samples',str(args.samples)]
   if args.profile:cmd+=['--gpu-profile','stage','--gpu-profile-output',str(case/'gpu.json')]
   else:cmd+=['--raw-samples',str(case/'samples.json')]
   save(case/'command.json',cmd)
   save(out/'state.json',{'status':'running','pid':os.getpid(),'case':str(case),'updated_at':stamp()})
   with (case/'stdout.txt').open('w') as f,(case/'stderr.txt').open('w') as err:
    p=subprocess.Popen(cmd,stdout=f,stderr=err)
    interference=[]
    while p.poll() is None:
     time.sleep(1)
     for item in competitors():
      if int(item.strip().split()[0])!=p.pid:interference.append(item)
   save(case/'execution.json',{'returncode':p.returncode,'interference':sorted(set(interference)),'finished_at':stamp()})
   if p.returncode or interference:raise RuntimeError(f'Failed/contaminated {case}: {p.returncode} {interference}')
   r=validate_case(case);save(case/'complete.json',r);sides[side]=r
  assert sides['baseline']['codestream_sha256']==sides['candidate']['codestream_sha256'],'Codestream differs'
  assert sides['baseline']['submissions']==sides['candidate']['submissions']
  changes={k:100*(sides['candidate']['metrics'][k]/v-1) for k,v in sides['baseline']['metrics'].items() if v>0}
  result={'job':job,'pair':pair,'sides':sides,'changes_percent':changes};results.append(result)
  save(out/'pairs.json',results)
  print(job,pair,{k:round(v,3) for k,v in changes.items()},flush=True)
summary={}
for job in selected:
 rows=[r for r in results if r['job']==job]
 summary[job]={k:{'median_change_percent':statistics.median(r['changes_percent'][k] for r in rows),
                  'wins':sum(r['changes_percent'][k]<0 for r in rows),'pairs':len(rows),
                  'baseline_ms':statistics.median(r['sides']['baseline']['metrics'][k] for r in rows),
                  'candidate_ms':statistics.median(r['sides']['candidate']['metrics'][k] for r in rows)}
               for k in rows[0]['changes_percent']}
save(out/'summary.json',summary);save(out/'state.json',{'status':'complete','finished_at':stamp(),'pairs':len(results)})
with (HERE/'experiments.jsonl').open('a') as f:f.write(json.dumps({'output':str(out),'at':stamp(),'summary_sha256':sha(out/'summary.json')})+'\n')
