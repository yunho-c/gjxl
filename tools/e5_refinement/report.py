#!/usr/bin/env python3
"""Generate a reviewable quality report from retained evidence only."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import statistics
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from qualify import Study, check, read, rows, save, sha


def label(name):
    if name.startswith('clic2024_test/'):
        return 'CLIC ' + name.split('/')[1][:8]
    return name.replace('unsplash/', '').replace('kodak/', 'Kodak ')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    root, dest = args.root.resolve(), args.destination.resolve()
    timing = None
    timing_dir = 'timing-confirmation' if (dest/'timing-confirmation/package.json').exists() else 'timing'
    if (dest/timing_dir/'package.json').exists():
        package = read(dest/timing_dir/'package.json')
        for name in ('analysis.json', 'report.md', 'manifest.json'):
            check(sha(dest/timing_dir/name) == package['files'][name], 'Timing export changed')
        check(read(dest/timing_dir/'manifest.json')['reference'] == str(root), 'Timing uses a different quality study')
        timing = read(dest/timing_dir/'analysis.json')
        timing['status'] = package['status']
    study = Study(root)
    analysis = read(root / 'analysis.json')
    controls = read(root / 'controls.json')
    tests = ET.parse(root / 'candidate-tests.xml').getroot()
    cases = tests.findall('.//testcase')
    check(analysis['summary']['rate_complete'], 'Incomplete rate coverage')
    check(controls['complete'] and all(r['identical'] for r in controls['results']), 'Incomplete controls')
    check(len(cases) == 170 and not tests.findall('.//failure') and not tests.findall('.//error'), 'Native tests incomplete')
    near = defaultdict(dict)
    for r in rows(root / 'near-target-matches.jsonl'):
        near[r['image_id']][r['arm']] = r
    near_pairs = []
    for name, arms in near.items():
        a,b = arms['baseline'],arms['candidate']
        check(abs(a['score']-80)<=.1 and abs(b['score']-80)<=.1 and abs(a['score']-b['score'])<=.05, 'Invalid secondary matched pair')
        near_pairs.append(dict(image_id=name,bytes_percent=100*(b['encoded_bytes']/a['encoded_bytes']-1),
                               butteraugli_percent=100*(b['butteraugli']/a['butteraugli']-1),
                               score_difference=b['score']-a['score'],matching_kind=a['matching_kind']))
    check(len(near_pairs)==12,'Incomplete secondary matching')
    batch=read(root/'batch-checks.json')
    check(len(batch['checks'])==2 and all(r['exit_code']==0 for r in batch['checks']),'Batch checks failed')
    curves = analysis['curves']
    groups = defaultdict(list)
    for r in curves:
        groups[r['corpus']].append(r)
    dest.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    names = [('kodak', 'Kodak'), ('clic2024_test', 'CLIC 2024'), ('unsplash_controlled', 'Large photographs')]
    for index, (key, display) in enumerate(names):
        values = sorted(r['pchip'] for r in groups[key])
        rng = np.random.default_rng(5 + index)
        ax.scatter(values, index + rng.uniform(-.13, .13, len(values)), s=32, c='#287d8e', alpha=.8)
        ax.scatter([statistics.mean(values)], [index], marker='D', s=85, c='#b24b25', edgecolor='white', zorder=5)
    ax.axvline(0, color='#869298', linewidth=1)
    ax.set(yticks=range(3), yticklabels=[f'{d} (n={len(groups[k])})' for k,d in names],
           xlabel='File-size change at matched SSIMULACRA2 75–85 (BD-rate, %)',
           title='Effort 5: zero refinement versus one update\nEach dot is one image; diamonds are arithmetic means.')
    ax.grid(axis='x', alpha=.2)
    fig.savefig(dest / 'rate-quality.png', dpi=180)
    plt.close(fig)
    outliers = defaultdict(dict)
    for r in rows(root / 'outlier-matches.jsonl'):
        outliers[r['image_id']][r['arm']] = r
    supplements = []
    for name, arms in outliers.items():
        a,b = arms['baseline'],arms['candidate']
        check(a['matched'] and b['matched'] and abs(a['score']-b['score']) <= .05, 'Unmatched outlier')
        supplements.append(dict(image_id=name, bytes_percent=100*(b['encoded_bytes']/a['encoded_bytes']-1),
                                butteraugli_percent=100*(b['butteraugli']/a['butteraugli']-1),
                                score_difference=b['score']-a['score']))
    check(len(supplements) == 3, 'Missing supplementary outlier checks')
    s = analysis['summary']
    status = ('**The full rate sweep and ordinary native checks pass; adoption remains pending AC-powered timing and review of the limits below.** '
              'The user explicitly deferred timing while the Mac was on battery. ')
    if timing:
        status = (f"**Controlled AC timing is complete: {timing['time_change_percent']:+.2f}% aggregate warm call time; "
                  f"timing gates {'passed' if timing['status']=='passed' else 'not all passed'}.** "
                  f'See the [timing qualification and adoption decision]({timing_dir}/report.md), including the retained first-run control failure and collection history. ')
    lines = [
        '# Effort 5: zero-refinement quality qualification', '',
        'The branch implements zero Butteraugli-guided quantization-field refinement updates at ordinary effort 5. '
        + status + 'No speedup is inferred from incidental quality-probe timings, '
        'and the existing paper throughput row is unchanged.', '',
        '| Policy | e4 | e5 candidate | e6 |', '| --- | --- | --- | --- |',
        '| Transform selection | Fixed DCT8 | Mixed search | Mixed search |',
        '| Initial field | Uniform | Spatial | Spatial |',
        '| Gaborish | Off | On | On |',
        '| Refinement updates | 0 | **0** | 1 |', '',
        'High-density and maximum-error overrides retain their existing behavior. Target-byte/BPP retries and explicit '
        'maximum compression follow the new ordinary e5 update count. CPU and exact-coefficient paths still evaluate '
        'their final field; ordinary fully resident Metal e5 omits perceptual evaluation unless final scoring is requested.', '',
        '## Rate–quality results', '',
        'Fresh baseline and candidate encodes cover all **65 images × 5 distances × 2 arms = 650 curve points**. '
        'Each point reproduced identical bytes across two complete encodes, decoded successfully through pinned libjxl, '
        'and produced finite linear-sRGB samples and a finite fast-ssim2 score. The distance grid is 4.6, 2.8, 1.9, 1.0, 0.55. '
        'All 65 curves are strictly monotone and bracket 75–85; there is no extrapolation or curve repair.', '',
        '| Corpus | Images | Mean BD-rate | Minimum | Maximum |', '| --- | ---: | ---: | ---: | ---: |',
    ]
    for k,d in names:
        v=[r['pchip'] for r in groups[k]]
        lines.append(f'| {d} | {len(v)} | {statistics.mean(v):+.3f}% | {min(v):+.3f}% | {max(v):+.3f}% |')
    lines += [f"| **All images** | **65** | **{s['pchip']['mean']:+.3f}%** | {s['pchip']['minimum']:+.3f}% | {s['pchip']['maximum']:+.3f}% |", '',
              f"PCHIP is primary; Akima sensitivity gives **{s['akima']['mean']:+.3f}%**. These are arithmetic means of per-image BD percentages, not a pooled curve or a ratio of total bytes. Positive values mean larger files.", '',
              '![Per-image BD-rate changes](rate-quality.png)', '', '## Independent perceptual checks', '',
              'Twelve images were selected before measurement: four Kodak, four CLIC (including the earlier experiment’s worst case), '
              'and four large photographs covering 12, 24, and 48 MP. Each arm was independently calibrated to SSIMULACRA2 80 ±0.025 '
              'with at most fourteen new probes. **Only 8/12 pairs met the original strict target**: four Kodak targets exhausted their budgets. '
              'Those original failures remain unchanged in `matches.jsonl`; this is not complete strict score-80 calibration. '
              'A separate, explicitly post hoc phase selected the closest existing score pairs for those four images, '
              'requiring each score within 80 ±0.1 and pair difference ≤0.01. Their actual differences are below 0.004. '
              'This secondary phase used no new encodes and did not reset calibration budgets. '
              'The following twelve-image results combine the eight strict pairs with those four separately labeled near-target pairs. '
              'Independent pinned libjxl Butteraugli scoring uses linear sRGB and an 80-nit intensity target. '
              'These are Butteraugli comparisons at matched SSIMULACRA2, not matched-Butteraugli rate measurements.', '',
              '| At matched SSIMULACRA2 ≈80 | Mean change | Minimum | Maximum |', '| --- | ---: | ---: | ---: |']
    for metric,title in [('bytes_percent','File size'),('butteraugli_percent','Butteraugli error (lower is better)')]:
        values=[r[metric] for r in near_pairs]
        lines.append(f"| {title} | {statistics.mean(values):+.3f}% | {min(values):+.3f}% | {max(values):+.3f}% |")
    lines += ['', 'Per-image results make the metric disagreement visible. A higher Butteraugli error is a perceptual-metric '
              'regression even when the SSIMULACRA2 score is matched; these percentages are not human severity ratings.', '',
              '| Image | Matching | Size near score 80 | Butteraugli error near score 80 |', '| --- | --- | ---: | ---: |']
    for r in near_pairs:
        kind='strict target' if r['matching_kind']=='strict-target' else 'secondary pair'
        lines.append(f"| {label(r['image_id'])} | {kind} | {r['bytes_percent']:+.3f}% | {r['butteraugli_percent']:+.3f}% |")
    lines += ['', 'The three largest observed BD-rate regressions were also checked after selection. These diagnostic cases '
              'are separate from the preselected twelve-image means:', '',
              '| Image | BD-rate 75–85 | Size at score 80 | Butteraugli error at score 80 |', '| --- | ---: | ---: | ---: |']
    by_name={r['image_id']:r for r in curves}
    for r in sorted(supplements,key=lambda r:-by_name[r['image_id']]['pchip']):
        lines.append(f"| {label(r['image_id'])} | {by_name[r['image_id']]['pchip']:+.3f}% | {r['bytes_percent']:+.3f}% | {r['butteraugli_percent']:+.3f}% |")
    lines += ['', '## Correctness and evidence', '',
              f'- Release native suite: **170/170 passed**. Focused tests cover the e5/e6 CPU/Metal score schedule, storage admission, initial-field/Gaborish boundary, and final-score byte parity.',
              f"- **{controls['checks']} additional byte controls passed**: unchanged e1–4/e6–10 and explicit overrides, CPU/exact e6, selected larger images, e5 scored/unscored paths, and CPU/Metal target-byte/BPP retries.",
              '- E5/e6 finite-budget batch checks passed with Metal API and shader validation: two concurrent callers, four images per batch, two odd-sized inputs, three rounds, and a shared eight-CPU limit. The driver checks reference bytes and memory/CPU admission bounds.',
              '- Focused workflow and effort-policy tests passed with Metal API and shader validation. Two broader storage tests aborted on `gjxl_aq_dc_quantize`: threadgroup memory 49,152 bytes exceeds 32,768. **Both failures reproduced in an independently built unchanged baseline.** This is an existing validation limitation, not evidence of a new e5 regression; full debug validation is not qualified.',
              '- With API validation alone, compatibility storage passed, but resident storage timed out after 605.57 seconds, after reporting 223 completed bounded cases. This additional timeout remains unresolved; the original ordinary run passed that test. All failed and successful logs are retained.',
              '- Baseline and candidate Metal shader SHA-256 are identical. The implementation change is in the shared host policy resolver; the candidate has no shader changes.',
              '- CUDA is not physically qualified on this Mac. This is a photographic-corpus quality study, not a universal visual-quality guarantee.', '',
              '## Reproduction and remaining decision', '',
              f"Baseline: `{study.m['source']['baseline_revision']}`. Candidate: that revision plus the retained `source.patch`. "
              'Apple M4 Pro, 14 CPU cores (10 performance, 4 efficiency), 48 GB. Separate Release builds use ordinary fully resident Metal, '
              'eight participating CPU threads, default density, automatic compression, weighted/prediction-aware DC with smoothing, '
              'native resolution, and final diagnostic scoring disabled. All inherited GJXL/RCA overrides are removed.', '',
              f'Full immutable source snapshots, native binaries, shaders, inputs/tool hashes, commands, raw samples, and codestreams are retained in `{root}`. '
              'The accompanying ledgers retain output and decoded-pixel hashes. `package.json` pins this exported evidence. '
              'The collectors and timing protocol are in `tools/e5_refinement/README.md`.', '',
              (f'The [completed AC-powered timing qualification]({timing_dir}/report.md) weighs these size/perceptual tradeoffs against controlled complete-call timing. '
               'It uses the eight strict and four secondary near-80 pairs, retaining the original strict target failures. '
               if timing else
               'The final adoption decision must weigh these measured size/perceptual tradeoffs against fresh, controlled complete-call timing. '
               'Freeze a near-80 protocol using the four separately labeled secondary pairs, or collect a new calibration cohort. ')
              +
              'Review the baseline-reproduced debug assertion and unresolved API-validation timeout separately. '
              'Do not replace the paper’s historical e5 MP/s or same-effort libjxl BD-rate with this GJXL-versus-GJXL study.', '']
    (dest / 'report.md').write_text('\n'.join(lines))
    names_to_copy=['analysis.json','controls.json','observations.jsonl','matches.jsonl','outlier-matches.jsonl',
                   'outlier-selection.json','near-target-selection.json','near-target-matches.jsonl',
                   'source.patch','source-identity.json','candidate-tests.xml','manifest.json','native-checks.json',
                   'batch-checks.json','metal-validation.log','baseline-validation.log','metal-api-validation.log',
                   'metal-validation.xml','baseline-validation.xml','metal-api-validation.xml']
    for name in names_to_copy:
        shutil.copy2(root/name,dest/name)
    save(dest/'supplementary-analysis.json',supplements)
    save(dest/'near-target-analysis.json',near_pairs)
    save(dest/'package.json', {'raw_root':str(root), 'files':{p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='package.json'},
                              'generator_sha256':sha(__file__),
                              'timing':timing['status'] if timing else 'deferred',
                              'adoption':(f'see {timing_dir}/report.md' if timing else 'pending controlled timing'),
                              'timing_package_sha256':sha(dest/timing_dir/'package.json') if timing else None})
    print(dest/'report.md')


if __name__ == '__main__':
    main()
