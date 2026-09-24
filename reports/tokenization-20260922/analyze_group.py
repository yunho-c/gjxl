import csv,json,statistics as st,re,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
base={(x['image'],x['effort']):x['output_sha256'] for x in json.loads((R/'final-profile/results.json').read_text())}
P=json.loads((R/'group-final-profile/results.json').read_text());B=json.loads((R/'group-final-batch/results.json').read_text())
assert len(P)==90 and len(B)==54
out={'profile':[],'batch':[],'validation':{}}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,rows):
 with (R/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
unnamed=0
for x in P:
 assert x['output_sha256']==base[x['image'],x['effort']]
 p=R/'group-final-profile'/f"r{x['round']}-{x['image']}-e{x['effort']}-{x['mode']}"
 assert sha(p/'reference.jxl')==x['output_sha256']
 pairs=json.loads((p/'paired.json').read_text())['rows'];assert all(v['byte_equal'] and v['summary_equal'] for v in pairs)
 assert len({v['committed_submissions'] for v in pairs})==1
 g=json.loads((p/'gpu.json').read_text())
 unnamed+=sum(not d['kernel_id'].startswith('gjxl_') for w in g['workloads'] for s in w['samples'] for sub in s['submissions'] for stage in sub['stages'] for d in stage['dispatches'])
for image in dict.fromkeys(x['image'] for x in P):
 d={(x['round'],x['mode']):x for x in P if x['image']==image};assert len(d)==15
 r={'image':image,'effort':1}
 for m in ['release','joined1','group256']:
  for k in ['ordinary_ms','ac_ms','dc_ms']:r[m+'_'+k]=st.median(d[i,m][k] for i in range(5))
 for control in ['release','joined1']:
  a=[100*(1-d[i,'group256']['ordinary_ms']/d[i,control]['ordinary_ms']) for i in range(5)]
  r[control+'_saving_median']=st.median(a);r[control+'_saving_min']=min(a);r[control+'_saving_max']=max(a)
 out['profile'].append(r)
pageouts=swapouts=0
for x in B:
 assert x['output_sha256']==base[x['image'],x['effort']]
 p=R/'group-final-batch'/x['artifacts'];assert sha(p/'reference.jxl')==x['output_sha256']
 plan=json.loads((p/'plan.json').read_text());x['in_flight']=plan['in_flight']
 rows=list(csv.DictReader((p/'samples.csv').open()));assert max(int(q['peak_committed_bytes']) for q in rows)<=plan['memory_limit'];assert x['peak_cpu_slots']<=8
 x['first_encode_ms']=int(rows[0]['wall_ns'])/1e6
 vm=[];attempt=p.name.removeprefix('attempt')
 for when in ['before','after']:vm.append({k:int(v) for k,v in re.findall(r'^([^:\n]+):\s+(\d+)\.',(p.parent/f'vm-{when}{attempt}.txt').read_text(),re.M)})
 pageouts+=vm[1]['Pageouts']>vm[0]['Pageouts'];swapouts+=vm[1]['Swapouts']>vm[0]['Swapouts']
for image in dict.fromkeys(x['image'] for x in B):
 for batch in [1,4]:
  d={(x['round'],x['mode']):x for x in B if x['image']==image and x['batch']==batch};assert len(d)==9
  r={'image':image,'effort':1,'batch':batch}
  for m in ['release','joined','group256']:
   for k in ['wall_ms','fps','first_encode_ms']:r[m+'_'+k]=st.median(d[i,m][k] for i in range(3))
   r[m+'_in_flight']=d[0,m]['in_flight'];r[m+'_peak_backing_gib']=max(d[i,m]['peak_backing_bytes'] for i in range(3))/2**30
  for control in ['release','joined']:
   a=[d[i,control]['wall_ms']/d[i,'group256']['wall_ms'] for i in range(3)]
   r[control+'_ratio_median']=st.median(a);r[control+'_ratio_min']=min(a);r[control+'_ratio_max']=max(a)
  out['batch'].append(r)
out['validation']={'profile_processes':len(P),'ordinary_samples':4*len(P),'batch_processes':len(B),'batch_samples':5*len(B),'outputs_match_independently_decoded_v7':True,'unnamed_dispatches':unnamed,'pageout_processes':pageouts,'swapout_processes':swapouts}
save('group-profile-summary.csv',out['profile']);save('group-batch-summary.csv',out['batch']);(R/'group-summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['validation'],indent=2))
