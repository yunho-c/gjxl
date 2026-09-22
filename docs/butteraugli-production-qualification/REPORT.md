# Butteraugli production preparation on current main

The main measured Butteraugli traffic bundle is prepared for integration on **Apple M4 Pro**, with fresh current-main output and performance qualification. The separate small refinement remains excluded. The final full suite passes **156/156** after correcting an inherited stale test expectation. One intermittent AC-test failure from an earlier run remains unexplained and is disclosed below. This branch is prepared for review and has not been merged into `main`.

## Revision and production boundary

| Role | Revision |
|---|---|
| Clean current-main baseline | `3b0de6d9d47846f6aa13f33ad6cca7e24e52ada6` |
| Production source and tests | `b58c236fe408b5a8e1265f913e5038b1368257ce` |
| Preserved and pushed study | `c7549eaed63e3a84d0fd86a4b113d36e64764aeb` on `perf/butteraugli-traffic` |
| Source study bundle | `db8a4d67d554bcb04b651243c3660c42ec245bb8` |
| Excluded incremental refinement | `ef5f29003c52ec788d339f00b6f2e50a6f1818e7` |

The integration branch is `perf/butteraugli-traffic-production`, created from the current-main baseline in a separate worktree. The primary checkout's unrelated staged and unstaged work was preserved. The [original study](https://github.com/yunho-c/gjxl/blob/c7549eaed63e3a84d0fd86a4b113d36e64764aeb/docs/butteraugli-traffic-20260921/REPORT.md) remains a separate historical measurement campaign; its percentages are not substituted for the new measurements below.

Production changes give the generated kernels descriptive names, remove unused low/medium experiments, use named host parameter layouts, and install ten optional pipelines as one complete bundle. Selection requires the device name `Apple M4 Pro`, Apple GPU family 9, sufficient pipeline thread counts and combined static/dynamic threadgroup memory, and SIMD width 32 for the small reduction. Other devices and optional-pipeline failures retain the original kernels and layout. Performance was measured on the **20 GPU-core M4 Pro with 48 GB RAM**; other devices and M4 Pro core-count variants were not measured.

The bundle uses shared filter loads, direct short filters, packed DC consumers, a fused Malta L2 path and a small reduction with the original addition order. Raw X/Y medium values temporarily use the future ultra slots; raw B uses the sixth borrowed image plane. Direct high and ultra filters have separate halo inputs and outputs. The packed mask path uses `kDc + 2` and never overwrites the live reference mask in `kWork + 3`. The legacy reference layout remains available. No public tuning flag, changed precision, relaxed tolerance, new scratch-storage requirement or codec policy is introduced. Backend-independent profiling plans remain conservative bounds for both paths.

## Fresh ordinary complete-call measurements

Each row reports the median of paired percentage changes. The two millisecond columns are independent medians of the per-process sample medians, so their quotient need not exactly equal the paired percentage. Negative changes mean lower time.

Seven alternating independent-process pairs, three warmups and seven retained calls per side:

| Workload | Main (ms) | Prepared (ms) | Paired change | Faster pairs |
|---|---:|---:|---:|---:|
| `alpine24-e7` | 469.296 | 410.737 | -12.211% | 7/7 |
| `forest48-e10` | 1969.861 | 1712.974 | -12.122% | 7/7 |
| `kodak01-e7` | 17.040 | 16.349 | -3.497% | 7/7 |
| `kodak17-e10-d08` | 32.878 | 31.757 | -3.707% | 7/7 |

The predeclared broader cohort uses three pairs, two warmups and three retained calls per side:

| Workload | Main (ms) | Prepared (ms) | Paired change | Faster pairs |
|---|---:|---:|---:|---:|
| `alpine24-e7` | 464.776 | 408.195 | -12.203% | 3/3 |
| `forest48-e10` | 1892.033 | 1682.185 | -11.091% | 3/3 |
| `campus12-e7` | 261.249 | 231.211 | -11.185% | 3/3 |
| `alpine12-e10-d08` | 514.682 | 466.811 | -9.613% | 3/3 |
| `forest24-e7-d08` | 528.578 | 464.355 | -12.150% | 3/3 |
| `campus48-e10-d08` | 2077.844 | 1881.267 | -9.185% | 3/3 |
| `kodak01-e7` | 16.465 | 16.227 | -0.576% | 2/3 |
| `kodak17-e10-d08` | 32.677 | 32.068 | -2.580% | 3/3 |

Names encode the image family, megapixel size and effort. Suffix `d08` means distance 0.8; the other cases use distance 1.9. Kodak cases are 0.39 MP. These fixed-image results are not a universal speedup guarantee.

Both sides use freshly built current-main sources, Release/Ninja, eight CPU threads and fully resident Metal. Timing begins with loaded linear RGB and ends with the returned codestream: reference preparation and CPU serialization are inside, while input loading and cached backend construction are outside. Final-score diagnostics and runtime GPU profiling are disabled in these ordinary cohorts. Backend initialization, cold-start latency, multi-image concurrency and other hardware remain outside this qualification.

Each process first makes one untimed reference call, then validates every warmup and timed call against that codestream, summary and submission count. Every baseline/candidate pair has the same codestream hash and submission count. Recognized competing encoder, compiler, test and profiling processes were checked before and during each case; this does not establish that all OS activity was absent.

## Unchanged control and separate stage attribution

The same frozen main binary was compared against itself using the seven-pair confirmation protocol:

- `alpine24-e7`: median +0.179%; individual pairs -0.353% to +1.229%.
- `forest48-e10`: median +0.086%; individual pairs -0.993% to +2.926%.

These controls expose measurement variability. They are not subtracted from the measured optimization, and no sub-percent refinement is selected from these results.

The separate three-pair stage-profile cohort reports the sum of nonoverlapping valid GPU stage intervals. Butteraugli includes reference preparation and all `butteraugli.*` stages:

| Workload | Butteraugli GPU change | All staged GPU change |
|---|---:|---:|
| `alpine24-e7` | -29.750% | -17.353% |
| `forest48-e10` | -27.628% | -16.328% |

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
