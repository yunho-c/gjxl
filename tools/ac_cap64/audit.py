"""Audit saved qualification artifacts; never builds or encodes."""
from pathlib import Path
import json
import qualify as q

b=q.load(q.OUT/'build.json')
assert b['base']=='06857bedd0467b4cb8b034d6456b0209b1db99c5', 'Qualification requires the recorded pre-policy baseline'
for p,d in b['files'].items():assert q.sha(p)==d,p
for mode,d in b['binaries'].items():assert q.sha(q.OUT/mode)==d
summary=q.load(q.OUT/'summary.json')
assert summary['complete'] and summary['cases']==112 and summary['encodes']==672 and summary['decoder_comparisons']==112
checks=q.load(q.OUT/'test-evidence.json')
for p,d in checks['logs'].items():assert q.sha(q.OUT/p)==d
for p,d in checks['test_sources'].items():assert q.sha(q.ROOT/p)==d
assert '99% tests passed, 1 tests failed out of 129' in (q.OUT/'release-ctest.log').read_text()
assert '100% tests passed, 0 tests failed out of 4' in (q.OUT/'asan-ctest.log').read_text()
large=[];small=[];deltas=[]
for r in summary['results']:
    for p,d in r['files'].items():assert q.sha(p)==d,p
    c=r['case'];item=c['input'];rows=r['rows']
    assert q.sha(item['path'])==item['sha256']
    is_large=item['width']*item['height']>=3840*2160
    assert rows['policy']['sha256']==rows['forced64' if is_large else 'baseline']['sha256']
    (large if is_large else small).append(r)
    if is_large:
        key=f"{item['name']}-{c['frontend']}-{c['distance']:g}"
        old=q.load(q.PRIOR/c['stage']/key/'complete.json')['rows']
        old={s['variant']:s for s in old}
        previous=old['baseline']['bytes']-old['ac-clusters64']['bytes']
        current=rows['baseline']['bytes']-rows['policy']['bytes']
        deltas.append({'case':key,'previous_saved_bytes':previous,'current_saved_bytes':current,'equal':previous==current})
result={'qualification_cases':len(summary['results']),'large_cases':len(large),'small_cases':len(small),
        'release_passed':128,'release_total':129,'release_known_failure':'quantization_pipeline',
        'asan_ubsan_passed':4,'asan_ubsan_total':4,
        'large_models_over_32':sum(r['rows']['policy']['ac_clusters']>32 for r in large),
        'prior_savings_equal':sum(r['equal'] for r in deltas),'size_savings':deltas}
q.save(q.OUT/'audit.json',result)
print(json.dumps(result,indent=2))
