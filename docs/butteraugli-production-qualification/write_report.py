#!/usr/bin/env python3
"""Render the final report from the completed saved records."""
from pathlib import Path
import json

P = Path(__file__).resolve().parent
D = Path('/Users/yunhocho/GitHub/gjxl-butteraugli-production/docs/butteraugli-production-qualification')
read = lambda name: json.loads((P / name).read_text())


def table(cohort, metric='complete_call_ms'):
    lines = ['| Workload | Main (ms) | Prepared (ms) | Paired change | Faster pairs |',
             '|---|---:|---:|---:|---:|']
    for name, row in read(cohort + '/summary.json').items():
        r = row[metric]
        lines.append(f'| `{name}` | {r["baseline_ms"]:.3f} | {r["candidate_ms"]:.3f} | '
                     f'{r["median_change_percent"]:+.3f}% | {r["wins"]}/{r["pairs"]} |')
    return '\n'.join(lines)


def main():
    assert read('qualification-state.json')['status'] == 'complete'
    assert read('production-validation-state.json')['status'] == 'complete'
    provenance = json.loads((D / 'git-provenance.json').read_text())
    stage_lines = ['| Workload | Butteraugli GPU change | All staged GPU change |',
                   '|---|---:|---:|']
    for name, row in read('stage-attribution/summary.json').items():
        stage_lines.append(f'| `{name}` | {row["butteraugli_all_ms"]["median_change_percent"]:+.3f}% | '
                           f'{row["gpu_total_ms"]["median_change_percent"]:+.3f}% |')
    control_lines = []
    controls = read('unchanged-control/pairs.json')
    for name, row in read('unchanged-control/summary.json').items():
        vals = [r['changes_percent']['complete_call_ms'] for r in controls if r['job'] == name]
        control_lines.append(f'- `{name}`: median {row["complete_call_ms"]["median_change_percent"]:+.3f}%; '
                             f'individual pairs {min(vals):+.3f}% to {max(vals):+.3f}%.')
    report = f'''# Butteraugli production preparation on current main

The main measured Butteraugli traffic bundle is prepared for integration on **Apple M4 Pro**, with fresh current-main output and performance qualification. The separate small refinement remains excluded. The final full suite passes **156/156** after correcting an inherited stale test expectation. One intermittent AC-test failure from an earlier run remains unexplained and is disclosed below. This branch is prepared for review and has not been merged into `main`.

## Revision and production boundary

| Role | Revision |
|---|---|
| Clean current-main baseline | `{provenance['base']}` |
| Production source and tests | `{provenance['source_commit']}` |
| Preserved and pushed study | `{provenance['study_head']}` on `perf/butteraugli-traffic` |
| Source study bundle | `{provenance['study_bundle']}` |
| Excluded incremental refinement | `{provenance['excluded_refinement']}` |

The integration branch is `perf/butteraugli-traffic-production`, created from the current-main baseline in a separate worktree. The primary checkout's unrelated staged and unstaged work was preserved. The [original study](https://github.com/yunho-c/gjxl/blob/c7549eaed63e3a84d0fd86a4b113d36e64764aeb/docs/butteraugli-traffic-20260921/REPORT.md) remains a separate historical measurement campaign; its percentages are not substituted for the new measurements below.

Production changes give the generated kernels descriptive names, remove unused low/medium experiments, use named host parameter layouts, and install ten optional pipelines as one complete bundle. Selection requires the device name `Apple M4 Pro`, Apple GPU family 9, sufficient pipeline thread counts and combined static/dynamic threadgroup memory, and SIMD width 32 for the small reduction. Other devices and optional-pipeline failures retain the original kernels and layout. Performance was measured on the **20 GPU-core M4 Pro with 48 GB RAM**; other devices and M4 Pro core-count variants were not measured.

The bundle uses shared filter loads, direct short filters, packed DC consumers, a fused Malta L2 path and a small reduction with the original addition order. Raw X/Y medium values temporarily use the future ultra slots; raw B uses the sixth borrowed image plane. Direct high and ultra filters have separate halo inputs and outputs. The packed mask path uses `kDc + 2` and never overwrites the live reference mask in `kWork + 3`. The legacy reference layout remains available. No public tuning flag, changed precision, relaxed tolerance, new scratch-storage requirement or codec policy is introduced. Backend-independent profiling plans remain conservative bounds for both paths.

## Fresh ordinary complete-call measurements

Each row reports the median of paired percentage changes. The two millisecond columns are independent medians of the per-process sample medians, so their quotient need not exactly equal the paired percentage. Negative changes mean lower time.

Seven alternating independent-process pairs, three warmups and seven retained calls per side:

{table('ordinary-confirmation')}

The predeclared broader cohort uses three pairs, two warmups and three retained calls per side:

{table('ordinary-broad')}

Names encode the image family, megapixel size and effort. Suffix `d08` means distance 0.8; the other cases use distance 1.9. Kodak cases are 0.39 MP. These fixed-image results are not a universal speedup guarantee.

Both sides use freshly built current-main sources, Release/Ninja, eight CPU threads and fully resident Metal. Timing begins with loaded linear RGB and ends with the returned codestream: reference preparation and CPU serialization are inside, while input loading and cached backend construction are outside. Final-score diagnostics and runtime GPU profiling are disabled in these ordinary cohorts. Backend initialization, cold-start latency, multi-image concurrency and other hardware remain outside this qualification.

Each process first makes one untimed reference call, then validates every warmup and timed call against that codestream, summary and submission count. Every baseline/candidate pair has the same codestream hash and submission count. Recognized competing encoder, compiler, test and profiling processes were checked before and during each case; this does not establish that all OS activity was absent.

## Unchanged control and separate stage attribution

The same frozen main binary was compared against itself using the seven-pair confirmation protocol:

{chr(10).join(control_lines)}

These controls expose measurement variability. They are not subtracted from the measured optimization, and no sub-percent refinement is selected from these results.

The separate three-pair stage-profile cohort reports the sum of nonoverlapping valid GPU stage intervals. Butteraugli includes reference preparation and all `butteraugli.*` stages:

{chr(10).join(stage_lines)}

The ordinary call results above establish the user-visible timing effect; these instrumented runs attribute it. Their profiled wall times are not mixed with ordinary results. No new Instruments trace was required for this integration check.

## Correctness and integration validation

- Both clean-baseline and prepared Release builds succeeded, including the Metal library, CLI, tests and benchmarks.
- Thirteen focused candidate checks passed with `MTL_DEBUG_LAYER=1` and `MTL_SHADER_VALIDATION=1`: the nine [initial focused checks](focused.log), followed by three [updated dispatch/storage checks](integration-tests.log) and the [corrected metadata test](metadata-candidate.log). The corrected metadata fixture also passes against untouched main with API/shader validation. [Baseline result](metadata-corrected-baseline-v3.log)
- Optimized-versus-legacy tests compare maps and scores bit for bit, including independent strides, offsets, option changes, cache reuse, small extents and added 127x131 / 257x259 cases. Forced-legacy Butteraugli, resident AQ and host storage-plan tests also pass on the same GPU.
- **56/56 exact-byte comparisons** pass across the canonical corpus and policy cases: efforts 1-10, resident/exact-coefficient/throughput/maximum-throughput modes, final-score collection, high density, maximum compression, target bytes and CPU maximum-error control. [Summary](canonical-parity-v2/summary.json)
- Three pinned-decoder pairs have identical bytes, valid dimensions and zero nonfinite pixels. [Pixel checks](canonical-parity-v2/finite-pixels.json)
- The final complete CTest run is **156/156 passing**. [Final suite log](full-suite-release.log) The earlier 155/156 result contained an inherited `metal_aq_strategy_metadata` failure, reproduced on untouched main. An external diagnostic located its obsolete `Unavailable` assertion: current main supports indirect/resident-metadata stage profiling. The separate test-only correction now asserts identical quantization and score history, one submission, valid ordered intervals, and resolved zero-work arguments for untimed empty indirect stages. The encoder and shader sources are unchanged by this correction. [Earlier candidate log](full-suite-final.log), [baseline control](baseline-failure-control.log), [resolution](metadata-resolution.json)

An intervening full-suite run failed `metal_ac_strategy` once. That unchanged test had passed the two earlier full-suite runs; subsequent original binaries passed 20/20 times on both baseline and candidate, and instrumented copies passed 10/10 on each. The cause remains unresolved; these repeats do not prove the optimization is unrelated or eliminate the possibility of a test flake. No AC source, expectation or tolerance was changed. The final suite then passed. [Incident log](full-suite-production.log), [investigation](ac-investigation.json), [original-binary repeat records](ac-original-results.json)

The first suite run also exposed two stale expectations for the now-fused dispatch counts. Those tests were updated to assert the exact selected-path counts while keeping allocation bounds conservative; the updated dispatch expectations and forced-legacy storage-plan checks now pass. The original failure log is retained. A first parity launch stopped before encoding because copied CLI files lacked executable permission; permissions were restored without changing binary hashes and the successful run uses a fresh output directory. These setup incidents are retained in the archive, not hidden as passing runs.

## Evidence and reproduction

Run `python3 docs/butteraugli-production-qualification/verify_evidence.py` from a checkout containing the source commit and its ancestors. It checks package hashes, source identities, original per-call JSON hashes, all 72 paired aggregates and 768 retained call samples, recorded codestream/decoder equality and both the inherited-failure control and final 156/156 suite result. It does not launch encoders or claim to re-decode absent pixel files. The [artifact audit](artifact-audit.json) was also run against the original full case files and recomputed the profiled GPU/Butteraugli totals from valid stage intervals.

Build/configure commands, the probe source, frozen binary/library hashes, input hashes, parity protocol, timing plan and per-cohort protocols are archived here. Protocol scripts preserve original absolute paths and require adaptation to new worktrees and fresh run directories. The original full artifacts, binaries, images, codestreams and raw GPU profiles remain at `/Users/yunhocho/GitHub/gjxl/reports/butteraugli-production-20260921`. The compact Git archive excludes those large payloads and retains their identities; hashes are not a substitute for rerunning correctness on a new platform.

For deployment, review and merge this prepared branch into the validated main revision, including the separate correction to the stale profiling test and reviewing the retained intermittent AC-test incident. Broadening the device gate or changing the kernels requires its own qualification. The practical optimization study remains closed; this preparation does not pursue the excluded small refinement.
'''
    (D / 'REPORT.md').write_text(report)
    (D / 'README.md').write_text('''# Butteraugli production qualification

[REPORT.md](REPORT.md) describes the prepared M4 Pro implementation, fresh current-main timings, exact-output validation, fallback coverage and the resolved stale profiling test.

Verify the compact saved evidence without running measurements:

```sh
python3 docs/butteraugli-production-qualification/verify_evidence.py
```

See [git-provenance.json](git-provenance.json) for the exact source commits and [hardware.json](hardware.json) for the measured device/toolchain. Large original artifacts remain outside Git; archived protocols retain their original absolute paths.
''')


if __name__ == '__main__':
    main()
