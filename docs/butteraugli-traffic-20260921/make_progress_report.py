#!/usr/bin/env python3
from pathlib import Path
import json,datetime
P=Path(__file__).resolve().parent
text=['# Butteraugli traffic investigation — complete','',
 'The investigation is **complete at the user-approved practical stopping point**. Substantial exact-output gains are established; the final refinements are small and workload dependent. This is not a proof of a theoretical hardware optimum. All changes are isolated from the primary checkout.', '',
 'Baseline: `4f3e414`, current main plus production-aligned profiling. Existing numerical tolerances are unchanged. See [opportunity inventory](OPPORTUNITIES.md), [traffic model](TRAFFIC-NOTES.md), and [incidents](INCIDENTS.md).','',
 '## Kernel screens','',
 'GPU command-buffer timing of repeated isolated filter dispatches. Each extent/variant uses five alternating pairs, three warmup submissions and five retained submissions per side, three dispatches per submission. Inputs, padding, output planes and guards are compared bitwise. These are not whole-encoder speedups.','',
 '| Family / variant | 4K change | 24 MP change | Status |','|---|---:|---:|---|']
for family in ['geometry','rolling','adjacent','norm','interior','short','malta','vertical','malta-adj','vertical-wide','reuse-geometry','reuse-closure','short-reuse','short-interactions','short-horizontal']:
 p=P/f'{family}-timing/summary.json'
 if not p.exists():continue
 rows=json.loads(p.read_text())
 groups={}
 for r in rows:groups.setdefault(r['variant']['name'],{})[r['extent']]=r
 for name,g in groups.items():
  if name=='baseline_control':label=family+' control'
  else:label=name.removeprefix('gjxl_ba_')
  cells=[]
  for extent in ['3839x2159','6000x4000']:
   r=g.get(extent);cells.append(f"{r['median_change_percent']:+.2f}% ({r['wins']}/5)" if r else 'pending')
  text.append('| '+label+' | '+' | '.join(cells)+' | screened |')
text+=['','## Complete encoding and attribution','',
 'Initial exploratory cohorts: three alternating independent-process pairs, two warmups and three measured complete calls per process. Profiling and ordinary timing are separate. Inputs are Alpine Lake 24 MP/e7 and Forest Stream 48 MP/e10, distance 1.9, eight CPU participants. Successful paired records require identical baseline/candidate codestream hashes, deterministic per-call bytes/summaries, equal submission counts, and no detected competing build/benchmark. Three pairs do not qualify sub-percent improvements. Cohorts labeled confirm instead use seven pairs, three warmups and seven samples. The dc-sub-increment cohort compares against main-only DC fusion, and ac-increment compares against main+sub DC fusion; older cohorts otherwise use the original baseline; newer cohort baselines are given in the table and identity.json. Displayed percentages are medians of per-pair ratios, not ratios of separately aggregated medians.','',
 '| Cohort / input / baseline | Complete-call change | Low/medium stage change |','|---|---:|---:|']
labels=['integrated-control','geometry-integrated-wall','integrated-control-stage-v2','geometry-integrated-stage-v2','rolling-integrated-wall-v2','rolling-integrated-stage-v2','dc-integrated-stage','dc-integrated-wall-confirm','dc-sub-increment-stage','interior-integrated-stage','control-wall-confirm','ac-increment-stage','dc-interior-stage','interior-wall-confirm','dc-interior-wall-confirm','ac-interior-stage','ac-interior-broad-wall','ultra-stage','malta-geometry-stage','materialized-increment-stage','vertical-increment-stage','ultra-bundle-increment-stage','malta-bundle-increment-stage','address-increment-stage','fusion-bundle-broad-wall','mask-compose-increment-stage','mask-compose-transpose-increment-stage','reuse-h64-bundle-increment-stage','reuse-h96-bundle-increment-stage']
labels+=['fusion-bundle-original-stage','fusion-bundle-wall-confirm','control-wall-final']
labels += [f'{v}-bundle-increment-{mode}' for v in ['reuse-h64','reuse-h96','malta','opsin32','mask16','ultra-reuse','high-reuse','mask-reuse','medium-reuse'] for mode in ['stage','wall'] if f'{v}-bundle-increment-{mode}' not in labels]
labels += ['short-bundle-increment-stage','short-bundle-increment-wall','short-bundle-broad-wall','short-bundle-wall-confirm','control-wall-short-final','short-bundle-original-stage','short-bundle-high-increment-wall-confirm']
labels += [f'{v}-increment-{mode}' for v in ['short-eight','short-wide-ultra'] for mode in ['stage','wall']]
labels += ['short-final-increment-stage','short-final-increment-wall-confirm','control-wall-short-increment','short-final-increment-broad-wall']
for label in labels:
 f=P/label/'summary.json'
 if not f.exists():continue
 identity=json.loads((f.parent/'identity.json').read_text())
 baseline=Path(identity.get('binaries',{}).get('baseline','unknown')).parent.name
 mode='profiled' if identity.get('profile') else 'ordinary'
 for job,metrics in json.loads(f.read_text()).items():
  a=metrics['complete_call_ms'];b=metrics.get('low_medium_ms')
  text.append(f"| {label} / {job} / {baseline} ({mode}) | {a['median_change_percent']:+.3f}% ({a['wins']}/{a['pairs']}) | "+(f"{b['median_change_percent']:+.3f}% ({b['wins']}/{b['pairs']})" if b else 'not instrumented')+' |')
text+=['','Negative changes mean less time. The stage-mode complete-call number is instrumented and does not replace ordinary timing. Changes in untouched stages show meaningful run-to-run variability; preliminary whole-call ratios cannot be attributed entirely to the filter.','',
 '## Current interpretation','',
 'See [REPORT.md](REPORT.md), [PHASE10-11.md](PHASE10-11.md), and [HYPOTHESIS-AUDIT.md](HYPOTHESIS-AUDIT.md) for current interpretation and qualification limits. The tables above retain historical cohorts; their baseline and mode are explicit. Do not compare candidates by dividing medians from different cohorts, add their effects, or infer ordinary gains from profiled calls.', '',
 'The earlier fusion bundle passes seven Metal/resident tests, 56 exact encoder comparisons and three identical finite decoded pairs. Seven-pair ordinary confirmation improves 24 MP/e7 by 7.659% and 48 MP/e10 by 6.883%, with 7/7 favorable pairs each. Independent broader ordinary coverage also improves every covered large-image setting. Fresh same-binary controls are -0.161% and +0.322%.', '',
 'Four-output/Malta and geometry-only refinements do not show consistent ordinary increments. New shared-load short filters reveal further headroom: the combined short-bundle independently passes seven focused tests, 56 exact encoder comparisons and three identical finite decoded pairs. Its broad three-pair ordinary cohort improves all six large-image settings by 9.3-12.4% versus original baseline, with all 18 pairs favorable. Seven-pair confirmation improves 24 MP/e7 by 12.697% and 48 MP/e10 by 10.715%, with 7/7 favorable pairs each. High-only ablation confirms the joint medium-B/mask addition. All geometry/reuse integrations and controls are complete; see PHASE15-17.md. The independently validated short-final refinement saves another 1.220%/0.296% in seven-pair direct-parent confirmation, with smaller/mixed broad results and unchanged-binary controls of +0.284%/+0.209%. The study closes at the practical stopping point; possible future small or workload-specific gains are not ruled out.', '',
 'Updated: '+datetime.datetime.now(datetime.timezone.utc).isoformat()]
(P/'PROGRESS.md').write_text('\n'.join(text)+'\n')
print('Wrote',P/'PROGRESS.md')
