#!/usr/bin/env python3
"""Independent, saved-data-only integrity and aggregation audit after collection."""
from pathlib import Path
import json,hashlib,statistics,datetime
P=Path(__file__).resolve().parent
for n in [15,16,17,18,19,20]:
 assert json.loads((P/f'campaign{n}-state.json').read_text())['status']=='complete',f'Campaign {n} is not complete; avoid audit I/O during timing'
cache={}
def sha(path):
 path=path.resolve()
 if path not in cache:
  with path.open('rb') as f:cache[path]=hashlib.file_digest(f,'sha256').hexdigest()
 return cache[path]
labels=[f'{v}-bundle-increment-{m}' for v in ['ultra-reuse','high-reuse','mask-reuse','medium-reuse'] for m in ['stage','wall']]
labels+=['short-bundle-increment-stage','short-bundle-increment-wall','short-bundle-broad-wall','short-bundle-wall-confirm','control-wall-short-final','short-bundle-original-stage','short-bundle-high-increment-wall-confirm']
labels += ['short-final-increment-stage','short-final-increment-wall-confirm','control-wall-short-increment','short-final-increment-broad-wall']
labels += [f'{v}-increment-{m}' for v in ['short-eight','short-wide-ultra'] for m in ['stage','wall']]
results=[]
corpus=Path('/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/corpus/pfm')
for label in labels:
 out=P/label;identity=json.loads((out/'identity.json').read_text())
 assert json.loads((out/'state.json').read_text())['status']=='complete'
 assert sha(out/'protocol.py')==identity['protocol_sha256']
 for side,binary in identity['binaries'].items():assert sha(Path(binary))==identity['binary_sha256'][side]
 for job,params in identity['jobs'].items():assert sha(corpus/params[0])==identity['input_sha256'][job]
 pairs=json.loads((out/'pairs.json').read_text());summary=json.loads((out/'summary.json').read_text())
 assert len(pairs)==identity['pairs']*len(identity['jobs'])
 assert len({(r['job'],r['pair']) for r in pairs})==len(pairs)
 for row in pairs:
  assert 0<=row['pair']<identity['pairs'] and row['job'] in identity['jobs']
  sides=row['sides']
  assert sides['baseline']['codestream_sha256']==sides['candidate']['codestream_sha256']
  assert sides['baseline']['submissions']==sides['candidate']['submissions']
  for side,record in sides.items():
   case=out/row['job']/f"{row['pair']:02d}-{side}"
   assert json.loads((case/'complete.json').read_text())==record
   for filename,expected in record['files'].items():assert sha(case/filename)==expected,(label,str(case),filename)
   assert sha(case/'reference.jxl')==record['codestream_sha256']
   samples=json.loads((case/'samples.json').read_text())
   assert samples['byte_equal'] and samples['summary_equal'] and samples['profiled']==identity['profile']
   assert len(samples['samples'])==identity['samples']
   assert record['metrics']['complete_call_ms']==statistics.median(samples['samples'])/1e6
   execution=json.loads((case/'execution.json').read_text());assert execution['returncode']==0 and not execution['interference']
  for metric,value in row['changes_percent'].items():
   assert value==100*(sides['candidate']['metrics'][metric]/sides['baseline']['metrics'][metric]-1)
 for job,metrics in summary.items():
  rows=[r for r in pairs if r['job']==job];assert len(rows)==identity['pairs']
  for metric,record in metrics.items():
   assert record=={'median_change_percent':statistics.median(r['changes_percent'][metric] for r in rows),
    'wins':sum(r['changes_percent'][metric]<0 for r in rows),'pairs':len(rows),
    'baseline_ms':statistics.median(r['sides']['baseline']['metrics'][metric] for r in rows),
    'candidate_ms':statistics.median(r['sides']['candidate']['metrics'][metric] for r in rows)}
 results.append({'cohort':label,'pairs':len(pairs),'profiled':identity['profile'],'summary_sha256':sha(out/'summary.json')})
for label in ['ultra-reuse','high-reuse','mask-reuse','medium-reuse','short-bundle','short-eight','short-wide-ultra','short-final']:
 out=P/'integrated'/label;identity=json.loads((out/'identity.json').read_text())
 assert identity['test_returncode']==0
 for name,expected in identity['files'].items():assert sha(out/name)==expected,(label,name)
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cohorts':results,'files_hashed':len(cache),
 'scope':'Frozen binaries/inputs/source identities, case file hashes, byte/submission equality, ordinary medians, paired changes and summaries. The collection drivers separately validate GPU timestamps; this audit is not a remeasurement.'}
(P/'short-artifact-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS',len(results),'cohorts,',sum(r['pairs'] for r in results),'pairs,',len(cache),'file hashes')
