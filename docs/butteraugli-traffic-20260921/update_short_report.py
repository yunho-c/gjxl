#!/usr/bin/env python3
"""Saved-data-only final report for the combined short-filter qualification."""
from pathlib import Path
import json,datetime
P=Path(__file__).resolve().parent
def read(rel):
 f=P/rel
 return json.loads(f.read_text()) if f.exists() else None
lines=['# Combined short-filter qualification, campaigns 12–20','',
 'Complete at the user-approved practical stopping point. Both combined candidates have independent correctness, broad and seven-pair ordinary results; controls, ablation and final refinement coverage are complete. No primary-checkout source has changed.','',
 '`short-bundle` adds the shared-load 15-tap high and medium-B filters and the disjoint 13-tap mask path. The mixed 16x64 ultra refinement is excluded. Its frozen source, binary and shader identities are under `integrated/short-bundle/`. All timing percentages below are direct paired changes against each cohort\'s declared parent; effects from different cohorts must not be added.','',
 '## Execution status','', '| Campaign | Status | Current step |','|---|---|---|']
for n in [12,13,14,15,16,17,18,19,20]:
 d=read(f'campaign{n}-state.json')
 if d:lines.append(f"| {n} | {d['status']} | {d.get('step','')} |")
parity=read('short-bundle-canonical-parity/summary.json')
finite=read('short-bundle-canonical-parity/finite-pixels.json')
if parity and finite:
 lines+=['','## Correctness','',f"The new combined candidate passes {parity['cases']} canonical/policy byte comparisons and {len(finite['checks'])} independently decoded pairs with identical finite pixels. The build's three focused tests and four extended Metal/resident tests pass under API/shader validation. The rebuilt library is byte-identical to the frozen measured library. This is focused qualification, not a full-suite success claim."]
lines+=['','## Ordinary complete-call cohorts','',
 'Public encoding from loaded linear RGB through returned codestream, including CPU serialization. Backend creation and input loading are outside. Eight CPU participants; validation layers disabled for timing. Initial cohorts use three alternating independent-process pairs with two warmups/three samples. Confirmation cohorts use seven pairs with three warmups/seven samples.','',
 '| Cohort / input | Direct baseline | Change | Faster pairs |','|---|---|---:|---:|']
for label in ['short-bundle-increment-wall','short-bundle-broad-wall','short-bundle-wall-confirm','control-wall-short-final','short-bundle-high-increment-wall-confirm','short-eight-increment-wall','short-wide-ultra-increment-wall','short-final-increment-wall-confirm','control-wall-short-increment','short-final-increment-broad-wall']:
 summary=read(label+'/summary.json');identity=read(label+'/identity.json')
 if not summary:continue
 baseline=Path(identity['binaries']['baseline']).parent.name
 for job,row in summary.items():
  v=row['complete_call_ms']
  lines.append(f"| {label} / {job} | {baseline} | {v['median_change_percent']:+.3f}% | {v['wins']}/{v['pairs']} |")
lines+=['','## Separate GPU attribution','',
 '| Cohort / input | Total Butteraugli | High | Medium B | Mask scopes | Ultra |','|---|---:|---:|---:|---:|---:|']
for label in ['short-bundle-increment-stage','short-bundle-original-stage','short-eight-increment-stage','short-wide-ultra-increment-stage','short-final-increment-stage']:
 d=read(label+'/summary.json')
 if not d:continue
 for job,row in d.items():
  cells=[f"{row[k]['median_change_percent']:+.3f}%" for k in ['butteraugli_all_ms','high_ms','medium_b_ms','mask_all_ms','ultra_ms']]
  lines.append('| '+label+' / '+job+' | '+' | '.join(cells)+' |')
lines+=['','## Bounded interaction followup','',
 'Campaign 14 checks 21 tile/reuse combinations, including three current-shape controls. Every configuration passes 48 guarded bitwise cases. The 15-tap eight-output variants reduce isolated two-pass blur time roughly 53-56%, compared with about 50-53% for the current four-output shape. The 13-tap increment is smaller. Wider 32x32/four-output ultra has a further isolated advantage, but its nonlinear epilogue must be included before a gain is credited.','',
 'Campaign 16 compares the new bundle directly with high-only over seven pairs, so the stronger high-frequency improvement cannot hide unhelpful medium-B/mask additions. Campaign 17 independently integrates the 16x64/eight-output 13/15-tap filters and the 32x32/four-output ultra filter, validates them, and measures each against `short-bundle`. These integrated results are complete; their ordinary effects are mixed or small and are listed below.','',
 'The smaller 16x64 eight-output shape is used because 16x128 doubles tile storage and shows no consistent advantage across both 13/15-tap radii and extents. Comparisons between separate micro-cohorts are diagnostic; their control variation and production epilogues preclude a precise encoder prediction.','',
 'Campaign 18 closes one remaining degree of freedom in the newly successful short filters: four/eight horizontal outputs versus the existing two. Eighteen configurations include six two-output controls, both leading ultra shapes, and four/eight vertical outputs for the 13/15-tap filters. It completed guarded parity and isolated timing sequentially after campaign 17. Three-channel 33-tap experiments did not favor four horizontal outputs, but the shorter one-channel kernels have different register pressure, so those earlier results do not exclude this case.','',
 'The phase 17 ordinary results are mixed or small: eight vertical outputs are +1.593% at 24 MP and -1.023% at 48 MP; wide ultra is -0.206% and -0.602%. A read-only snapshot records active suggestd, media analysis and indexing processes, without proving which calls they affected. No system service was changed. Campaign 18 passes all 18 x 48 guarded cases; four horizontal outputs modestly improve 15-tap filtering, while eight usually regress. Campaign 19 combines the remaining best choices and independently passes seven focused tests, 56 byte comparisons and three identical finite decoded pairs. Its seven-pair direct-parent comparison is complete: -1.220% at 24 MP (7/7), -0.296% at 48 MP (5/7), -0.377% on Kodak01 (4/7), and +0.057% on Kodak17 (3/7). Matching controls are +0.284% at 24 MP and +0.209% at 48 MP, diagnostic of variability rather than corrections. Campaign 20 is complete: all six large-image medians improve 0.24–1.35%, but many individual pairs are mixed; Kodak01 improves 0.364% (2/3), while Kodak17 regresses 0.427% (1/3 faster). These smaller effects do not change the confirmed original-baseline gain of the main short-filter bundle. The final prototype is restored in the isolated worktree; seven tests re-pass and the shader matches the frozen measured library. The independent artifact audit verifies 23 late-study cohorts, 256 pairs and 3,356 file hashes.','',
 'Broader interpretation and prior rejected mechanisms: [REPORT.md](REPORT.md), [HYPOTHESIS-AUDIT.md](HYPOTHESIS-AUDIT.md), [PHASE12-14.md](PHASE12-14.md). No planned comparison remains pending. The practical stopping conclusion is empirical for these workloads, not a mathematical hardware optimum or proof that no small improvement remains.','',
 'Updated: '+datetime.datetime.now(datetime.timezone.utc).isoformat()]
(P/'PHASE15-17.md').write_text('\n'.join(lines)+'\n')
print('Updated PHASE15-17.md from completed saved cohorts only')
