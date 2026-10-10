#!/usr/bin/env python3
"""Render the audited timing result without starting any encodes."""
import argparse
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from qualify import check, read, rows, save, sha


def label(name):
    if name.startswith('clic2024_test/'):
        return 'CLIC '+name.split('/')[1][:8]
    return name.replace('kodak/','Kodak ').replace('unsplash/','')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args()
    root,dest=args.root.resolve(),args.destination.resolve()
    a=read(root/'analysis.json');m=read(root/'manifest.json')
    check(a['accepted_pairs']==120 and len(a['images'])==12,'Incomplete timing audit')
    for path,digest in a['provenance'].items():
        check(sha(path)==digest,f'Audited evidence changed: {path}')
    for path,digest in m['files'].items():
        check(sha(path)==digest,f'Frozen identity changed: {path}')
    service=rows(root/'service-control.jsonl')
    paused=[r for r in service if r['action']=='paused']
    for r in paused:
        check(any(s.get('media_pid',s.get('service_pid'))==r.get('media_pid',r.get('service_pid')) and s['collector_pid']==r['collector_pid'] and
                  s['time']>=r['time'] and s['action'] in ('resumed','original process no longer exists')
                  for s in service),'Media-analysis pause has not been restored')
    confirmation=read(root/'confirmation-plan.json') if (root/'confirmation-plan.json').exists() else None
    decision=None
    if confirmation:
        first=Path(confirmation['first_root'])/'analysis.json'
        check(sha(first)==confirmation['first_analysis_sha256'],'Original timing analysis changed')
        start=min(r['started'] for r in rows(root/'attempts.jsonl'))
        check(confirmation['created']<start and paused and paused[0]['time']<start,
              'Confirmation plan or media-analysis pause was not in place before measurement')
        difference=abs(a['aggregate_ratio']-confirmation['first_aggregate_ratio'])
        decision=dict(all_timing_gates_passed=a['status']=='passed',
                      aggregate_ratio_difference=difference,
                      replication_agrees=difference<=confirmation['maximum_aggregate_ratio_difference'],
                      protocol_limit=confirmation['maximum_aggregate_ratio_difference'],
                      original_status=read(first)['status'],
                      original_analysis_sha256=sha(first))
        decision['passed']=decision['all_timing_gates_passed'] and decision['replication_agrees']
    dest.mkdir(parents=True,exist_ok=True)
    q=read(Path(m['reference'])/'analysis.json')['summary']
    passed=a['status']=='passed' and (not decision or decision['passed'])
    change=a['time_change_percent'];lo,hi=a['conditional_95_time_change_percent']
    lines=['# Effort 5: AC-powered timing '+('confirmation' if confirmation else 'qualification'),'',
           f"**Timing gates: {'passed' if passed else 'not all passed'}.** Zero refinement changes equal-image warm complete-call time by **{change:+.2f}%** "
           f"(conditional 95% interval **{lo:+.2f}% to {hi:+.2f}%**), equivalent to **{a['throughput_factor']:.3f}×** throughput on this fixed twelve-image cohort.",'',
           ('The measured tradeoff supports adopting zero refinement as the faster e5 tier on the tested Metal backend, '
            'while retaining e6’s one-update policy. This is a speed/quality tradeoff, not a compression-preserving optimization.'
            if passed else 'This run does not satisfy every predeclared timing gate. Keep the policy as a candidate until the failed gates are resolved; do not present this as an unconditional timing qualification.'),'',
           f"The preceding 65-image rate study measured **{q['pchip']['mean']:+.3f}% mean BD-rate**, with a **{q['pchip']['maximum']:+.3f}%** worst image over SSIMULACRA2 75–85. "
           'Independent near-80 checks included image-specific Butteraugli regressions, reaching +20.46% in the worst rate outlier. '
           'Those three post-selection quality outliers are not part of this preselected timing cohort. '
           'See the [quality report](../report.md) for per-image results, matching limits, and existing debug-validation failures.','',
           '## Protocol and scope','',
           '- Apple M4 Pro, 14 CPU cores, 48 GB; AC power, ordinary fully resident Metal, e5, eight participating CPU threads, default density and automatic compression.',
           '- Exact retained baseline and candidate executables, shaders, source snapshots, inputs, and output hashes verified before collection and audit. No rebuild or new quality calibration during collection.',
           '- Twelve measured near-80 quality pairs: eight strict target pairs plus four secondary saved-probe pairs. The original four strict calibration failures remain unresolved; timing does not convert them into strict successes.',
           '- Eight A/B process pairs per image, divided into two blocks; rotating image order and balanced baseline/candidate process order. Two same-baseline A/A process pairs per image are interleaved across the blocks.',
           '- Each process performs one validation encode, two warmups, and five timed calls. Every repeated encode must reproduce its expected codestream, and each process output must match the corresponding retained quality-study hash.',
           '- Boundary: synchronous public encode-call wall time, including its complete normal work and synchronization. Input loading, process/Metal initialization, output hashing and file writing are outside timing. These are warm single-image calls, not batch or cold-start results.',
           '- One-second environment monitoring rejects entire pairs for declared CPU pressure, competing encoders/builds, loss of AC power, thermal warnings, swap growth, or monitoring errors. Exclusion decisions never inspect latency. All attempts are retained.',
           f"- **96 accepted A/B pairs + 24 A/A pairs**, **240 processes**, **1,200 timed calls**, **1,920 total accepted encodes**. **{a['excluded_pairs']} excluded pairs**. Accepted pairs had no swap growth.",'',
           'The primary estimate is the equal-image geometric mean of each image’s median paired candidate/baseline ratio. '
           'Each process value is the median of five calls. The confidence interval resamples process pairs within each fixed image '
           '(10,000 draws, seed 20261007); it measures conditional repeat uncertainty, not generalization to unseen images. '
           'Open desktop applications remain part of the environment, subject to the declared sampled admission thresholds.','',
           '### Collection history','',
           ('Exactly one fresh complete confirmation was declared after the first run failed its Kodak 08 A/A control. '
            'This run began with the authorized media-analysis pause already active. It uses the same binaries, quality pairs, '
            'warmups, samples, load thresholds and timing gates, with the previously amended twelve-attempt retry limit. '
            'The decision also requires the aggregate candidate/baseline ratio to agree with the first run within 0.05. '
            'The [first failed run](../timing/report.md) remains separate and is never pooled, replaced, or relabeled as passing.'
            if confirmation else
           'After 33 accepted and seven environment-excluded pairs, one case exhausted its original three-attempt budget. '
           'A recorded amendment increased the budget uniformly to twelve attempts per pair; the original manifest and attempt ledger are preserved. '
           'The decision addressed background-load exclusions before aggregate timing analysis. All timing gates, load thresholds, '
           'sample counts, quality settings and ordering stayed unchanged. Excluded attempts never enter the timing estimate.'),'',
           'With explicit user approval, `mediaanalysisd` was temporarily suspended under a watchdog during collection. '
           'The watchdog resumed it on collector exit, with a thirty-minute fallback. Every recorded pause has a corresponding restoration event; '
           'the service-control ledger and watchdog sources are included in this package. Other desktop applications remained running.','',
           ('A separately authorized pause of `spotlightknowledged.updater` was needed before the confirmation admitted its first pair. '
            'After the user requested autonomous continuation, this pause was extended to the remaining `spotlightknowledged` indexer. '
            'Both used the same exit/timeout restoration safeguards and are recorded in the service-control ledger. '
            'This is an environment-management change from the first run; the numerical admission thresholds stayed unchanged.'
           if any(r.get('service')=='spotlightknowledged.updater' for r in paused) else ''),'',
           ('A single separately logged foreground encoding burst tested the suggestion that background work would quiet under load. '
            'It comprised 32 samples plus two warmups and one validation encode per arm on forest_stream/48mp; these 70 encodes are excluded '
            'from the scheduled experiment. Spotlight remained active, and the monitored quiet admission criteria stayed unchanged.'
            if (root/'conditioning/plan.json').exists() else ''),'',
           '## Results','',
           '| Image | Matching | Baseline ms | Candidate ms | Paired time change | A/A change |',
           '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in a['images']:
        kind='strict target' if r['matching_kind']=='strict-target' else 'secondary pair'
        lines.append(f"| {label(r['image_id'])} | {kind} | {r['baseline_ms']:.3f} | {r['candidate_ms']:.3f} | {100*(r['median_ratio']-1):+.2f}% | {100*(r['aa_median_ratio']-1):+.2f}% |")
    lines+=['','Milliseconds are medians of process medians; the paired-change column is computed from paired ratios, so it need not equal the ratio of the displayed time columns.','',
            f"Unchanged-binary A/A aggregate change: **{a['aa_time_change_percent']:+.2f}%**. "
            f"Block 1 / block 2 A/B changes: **{100*(a['block_ratios'][0]-1):+.2f}% / {100*(a['block_ratios'][1]-1):+.2f}%**.",'',
            '| Matching stratum | Images | Paired time change |','| --- | ---: | ---: |']
    for kind,r in a['strata'].items():
        lines.append(f"| {kind} | {r['images']} | {100*(r['ratio']-1):+.2f}% |")
    lines+=['','![Per-image paired timing changes](timing.png)','','## Predeclared timing gates','',
            '| Gate | Result |','| --- | --- |']
    descriptions={'aggregate_gain':'At least 15% aggregate time reduction',
                  'conditional_interval':'Conditional 95% interval entirely below no change',
                  'no_material_image_regression':'No per-image median slowdown above 5%',
                  'aa_aggregate':'A/A aggregate within ±3%', 'aa_images':'Every A/A image median within ±10%'}
    for name,ok in a['checks'].items():
        lines.append(f"| {descriptions[name]} | {'Pass' if ok else '**Fail**'} |")
    if decision:
        lines.append(f"| Confirmation aggregate ratio within 0.05 of first run | {'Pass' if decision['replication_agrees'] else '**Fail**'} ({decision['aggregate_ratio_difference']:.5f}) |")
        lines+=['',f"The first run measured **{100*(confirmation['first_aggregate_ratio']-1):+.2f}%** aggregate time change; "
                f"this confirmation measures **{change:+.2f}%**. The first run’s Kodak 08 A/A median was **+13.657%**, "
                'outside the original ±10% image-control limit. That failed result remains reported as failed; '
                'the confirmation is a separate experiment with the same acceptance gates.']
    failed=[r for r in a['images'] if not .9<=r['aa_median_ratio']<=1.1]
    if failed:
        lines+=['','Failed unchanged-binary image controls:']
        for r in failed:
            values=', '.join(f'{100*(x-1):+.3f}%' for x in r['aa_ratios'])
            lines.append(f"- {label(r['image_id'])}: individual A/A pairs {values}; median {100*(r['aa_median_ratio']-1):+.3f}%. These accepted measurements remain in the analysis.")
    lines+=['','## Evidence and remaining boundaries','',
            f"Full raw timing artifacts are retained at `{root}`. `package.json` hashes the exported ledgers and report; "
            '`analysis.json` additionally hashes the complete environment and command logs in the raw directory. '
            'The original quality evidence and all debug-validation failures are preserved. The timeout and baseline-reproduced Metal '
            'validation assertion remain separate unresolved issues; this timing run is uninstrumented and does not certify debug validation.','',
            'This experiment does not refresh the paper’s historical e5 MP/s row or its comparison against libjxl. '
            'It qualifies this exact candidate versus this exact baseline at the selected measured near-80 settings. '
            'CUDA and batch throughput require their own measurements. No merge, commit, or push is implied.','']
    (dest/'report.md').write_text('\n'.join(lines))
    fig,ax=plt.subplots(figsize=(10,6.4),constrained_layout=True)
    plt.rcParams.update({'font.size':10})
    for i,r in enumerate(a['images']):
        y=len(a['images'])-1-i
        xs=100*(np.array(r['paired_ratios'])-1)
        ax.scatter(xs,y+np.linspace(-.14,.14,len(xs)),color='#277b8e',s=20,alpha=.55)
        ax.scatter([100*(r['median_ratio']-1)],[y],color='#165464',s=65,marker='D',zorder=4)
        ax.scatter([100*(r['aa_median_ratio']-1)],[y],color='#b24b25',s=55,marker='x',zorder=4)
    ax.axvline(0,color='#7b858c',linewidth=1)
    ax.set(yticks=list(range(12)),yticklabels=[label(r['image_id']) for r in reversed(a['images'])],
           xlabel='Warm complete-call time change (%) · negative is faster',
           title=f"Effort 5: zero versus one refinement update\nEqual-image aggregate {change:+.1f}% · blue: A/B pairs and medians · orange: A/A controls")
    ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(dest/'timing.png',dpi=180);plt.close(fig)
    for name in ('analysis.json','manifest.json','manifest-original.json','retry-amendment.json',
                 'attempts-before-amendment.jsonl','attempts.jsonl','commands.jsonl','service-control.jsonl',
                 'watch_media.py','watch_media_continuation.py','watch_spotlight.py','watch_spotlight_main.py','confirmation-plan.json'):
        if (root/name).exists():
            shutil.copy2(root/name,dest/name)
    if decision:
        save(dest/'confirmation-decision.json',decision)
    if (root/'conditioning/plan.json').exists():
        save(dest/'conditioning.json',dict(plan=read(root/'conditioning/plan.json'),
             commands=rows(root/'conditioning/commands.jsonl'),
             files={str(p.relative_to(root)):sha(p) for p in sorted((root/'conditioning').rglob('*')) if p.is_file()}))
    env=rows(root/'environment.jsonl')
    save(dest/'environment-summary.json',[{k:v for k,v in r.items() if k!='processes'} for r in env])
    save(dest/'package.json',dict(raw_root=str(root),status='passed' if passed else 'complete-with-failed-timing-gates',generator_sha256=sha(__file__),
                                 files={p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='package.json'}))
    print(dest/'report.md')


if __name__=='__main__':
    main()
