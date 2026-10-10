# Fused entropy-value collection

Direct ANS preparation collects exact sorted raw-value populations per cluster.
For at least 4096 input values, `CollectClusteredEntropyValues` counts values
below 1024 directly in a cluster-major table and sends larger values through the
existing aggregation path. It then prepends the common values in sorted order.
Small inputs retain collection followed by aggregation.

The table is bounded by 64 clusters times 1024 `uint64_t` counters (512 KiB).
The entropy storage plan charges this table in addition to its conservative
raw-value and aggregation bounds, including balanced DC HybridUint search.
Ordinary balanced AC coding does not need the raw-value collection path.
The collector supports interleaved and split token views, validates partition
and context indices, and publishes output only after successful completion.
Counts, HybridUint choices, model selection and emitted bytes remain unchanged.

The collector is enabled in ordinary builds. Diagnostic builds configured with
`-DGJXL_BUILD_TOKENIZATION_EXPERIMENT=ON` accept
`GJXL_EXPERIMENT_FUSED_ENTROPY_VALUES=0` to retain the unfused collector for
comparison. Ordinary builds ignore that variable. Configure process-wide
settings before encoding, not while calls are active.

## Recorded qualification

The September 28, 2026 study compared the final ordinary build against
`1fb0ff1c491555c4c0a5458a32aba37b1811ff1e`, which already enabled DC/AC
section-writing overlap. The measured source snapshot also contained separate
GPU preparation and representation-measurement experiments, all disabled in
ordinary builds. Those experiments are not part of this change.

Complete encode-call latency, excluding input loading and file I/O, was measured
on a 14-core Apple M4 Pro with 48 GB, AC power, Apple Clang 21.0.0 / SDK 27.0,
Release `-O3`, and distance 1.9. The initial comparison used ABBA/BAAB process
ordering across 60 settings, 240 processes, 960 retained samples and 1680 public
calls. Values below are geometric means of per-image time ratios; negative
changes mean lower latency.

| Effort | Images | Initial time change | Longer repeat |
|---|---:|---:|---:|
| 1 | 8 | +1.05% | -0.65% |
| 4 | 8 | -4.71% | -2.62% |
| 7 | 8 | -0.002% | — |
| 9 | 16 | -4.13% | — |
| 10 | 16 | -4.12% | — |

The 12 photographic cases at efforts 9/10 improved by 3.52%/3.36%; synthetic
noise improved about 20%. All 16 effort-10 cases improved. The only positive
high-effort result was effort-9 flat imagery at +0.16% (about 0.05 ms).

Noisy initial effort-1 results prompted the longer repeat: all eight effort-1/4
inputs, opposite-order rounds, 1536 retained calls and 1920 public calls. It
supported no broad low-effort penalty, while individual regressions still
reached +1.20% at effort 1 and +1.63% at effort 4. Sub-percent differences should
not be treated as established gains above background noise.

Effort-9 CPU-1 controls reduced time by 5.73% (CLIC) and 2.64% (Kodak); batches
of three with eight CPU participants improved by 1.52% and 3.43%. Managed peak
backing on Kodak effort 9 decreased from 109.10 to 105.02 MiB, while large-image
cases dominated by GPU/AQ storage were unchanged. This is not a whole-process
RSS result.

The final ordinary study binary passed all 525 image/effort comparisons across
105 images at efforts 1/4/7/9/10 with identical bytes and summaries. All 516
distinct streams had successful pinned-`djxl` validation: 198 new decodes and
318 previously validated decodes reused by exact compressed hash. The ordinary
full suite passed 171/171 before final policy/accounting cleanup; all nine
affected ordinary tests passed after cleanup.

These historical measurements apply to their frozen binaries and corpus. They
are not a new benchmark of the isolated commit or of later main revisions, and
do not qualify concurrent independent single-image calls.

## Evidence and focused validation

The retained study directory is
`/Users/yunhocho/GitHub/gjxl/reports/tokenization-entropy-20260928`.
It contains `REPORT.md`, `production-final-analysis.json`,
`production-recheck-analysis.json`, `correctness-final.json`,
`decode/validation.json`, `freeze-final.json`, `final.patch` and `source-final/`.
The collector implementation, internal declaration, storage constants and
entropy test extracted here match the four corresponding September 28 source
fingerprints exactly.

The collector regression test compares fused and unfused populations for
1/7/64 clusters, input lengths around the 4096-value threshold, dense and sparse
distributions, boundary values through `UINT32_MAX`, and mixed split/interleaved
views. Invalid contexts and cluster maps must preserve the previous output.
Existing entropy and serializer storage tests cover admitted memory,
allocation failures, resource cleanup and encoder behavior.

On October 10, the isolated four-file source/test change was exported over
`1fb0ff1` and built in Release mode with Metal, CUDA, benchmarks and diagnostic
controls disabled. With inherited `GJXL_*` environment overrides cleared, all
six focused CTests passed: `entropy`, `entropy_storage_plan`,
`serializer_storage_plan`, `codestream_encoder`, `section_writing` and
`workflow_admission_cpu`. This checks the extracted change independently of the
remaining prototypes; no new performance measurements were taken.
