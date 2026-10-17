# Parallel GPU node setup and property summarization

Parallelizing node setup and property summarization reduces median complete prepared
fitting by **11.55%** and the sum of complete-encode medians by
**6.05%** in the matched comparisons below. Trees,
codestream bytes and decoded pixels remain exact. GPU managed backing is unchanged.
The GPU route still takes **13.19% longer** than the
hybrid CPU-fitting continuous control; CPU fitting remains the default.

The baseline is commit `12f396979f393137c2b87ceed1f8f49eec6244fa`, the qualified
[fused small-job implementation](gpu-frontier-fused-small-jobs.md). The two changes
are measured separately and together. GPU comparisons use 256 active nodes,
64 MiB total tiled scratch and 16 frontier rounds per submission. The private
128-node/64-MiB defaults, compression policy and opt-in GPU fitting status remain
unchanged.

## Implementation

Node setup previously used one lane to walk every active slot. Each lane now owns
a contiguous range of at most eight slots. An integer prefix ranks free slots in
their original order, then assigns the available queue entries in that order.
A second prefix computes disjoint row-task and statistics-tile ranges. Pending
slots retain their state. Shared integer counters retain the per-fit population
diagnostics, and one lane publishes the aggregate queue, live-node and dispatch
counts. Idle and failed rounds retain valid empty-work dispatches.

Property summarization previously used one lane per node to walk all properties
and bins. Four SIMD groups now own separate properties. Integer prefixes compute
occupancy totals and the original-to-compact-bin map, including empty bins and
partial final groups. Constant-property masks and histogram-byte demands are
combined after all properties finish. Parent entropy still executes on one lane
with exactly the previous floating-point arithmetic. Split and predictor ties,
partitioning, child construction and final tree canonicalization are unchanged.

The new static threadgroup reservations are **288 bytes for setup** and **32 bytes
for summarization**, previously zero. Both pipelines report 32-lane execution
widths and a 1,024-thread maximum on the M4 Pro. No host buffer, storage-layout
field, dispatch, encoder boundary or command submission is added. The scratch
budget and failure-atomic publication contract remain intact; no CPU retry is
introduced after GPU execution starts. These resource observations do not identify
a register or occupancy-manager limit.

## Complete prepared fitting

The input is the first archived prepared cohort from photograph 1029: **42 fits,
5,285,712 distinct rows and 35,878 nodes**. Timing uses the M4 Pro on battery with
14 CPU participants. The same frozen replay executable loads each shader variant.
Each configuration has three process blocks, each with one warmup and three
measured CPU/GPU pairs. Configuration order rotates/reverses; CPU/GPU order
alternates within each process. Diagnostic capture and validation layers are off.

| Configuration | Median complete fitting (s) | Measured range (s) |
|---|---:|---:|
| Previous tiled GPU | 0.458234 | 0.448779–0.463465 |
| Parallel node setup | 0.425806 | 0.419122–0.433204 |
| Parallel setup + summarization | 0.405325 | 0.400161–0.413297 |
| CPU14, pooled controls | 0.408643 | 0.396190–0.437153 |

Each GPU row has nine measured calls; the CPU row pools 27 controls. Setup alone
reduces the median by 7.08%; summarization adds 4.81% relative
to setup alone. All **72 calls**, including warmups, match native CPU trees.
The combined GPU and CPU medians are close on this cohort; this does not establish
superiority for complete encoding or other inputs.

Complete fitting includes validation, allocation/upload, histogram work,
partitioning, node/queue management, canonicalization and publication. It excludes
sampling, CPU input restoration, backend/domain creation and worker-pool creation.
GPU command-span medians are 0.429382 s
before and 0.373190 s
after; these spans include scheduling and stalls, not just active arithmetic.
Managed backing is exactly 304,702,152 bytes for every GPU variant.

## Complete image qualification

The inputs are the same **4000 × 3000 RGB8 photographs 1029, 1039 and 1049**.
Every configuration uses learned `auto/auto/best/fit-off`, Metal unique sample
rows and 14 participants for the configured CPU stages. GPU cohorts use an
8 Mi-sample feature window; the CPU continuous control uses 16 Mi samples.
The CPU-fitting control retains the hybrid pipeline's other Metal stages and
is not a pure-CPU encoder.

Each image/configuration has one warmup and three measured calls in rotating
order. The complete private encode boundary excludes backend/domain creation,
file I/O, independent decoding and hashing. All **36 encodes** match the previous
output hashes, work counts and exact pinned-libjxl decode. Every GPU encode
performs all 192 local fits, with no CPU-node or admission fallback.

Times are seconds per image. **Sum** means the sum of per-image medians over all
three photographs, not time for one photograph.

| Hybrid configuration | 1029 | 1039 | 1049 | Sum |
|---|---:|---:|---:|---:|
| Previous GPU, 256/64 cohorts | 3.884336 | 2.577863 | 2.407324 | 8.869523 |
| Parallel GPU, 256/64 cohorts | 3.613625 | 2.444893 | 2.274173 | 8.332691 |
| CPU fitting, continuous | 3.037649 | 2.233814 | 2.090277 | 7.361740 |

Per-image median reductions are 6.97% / 5.16% / 5.53%.
The summed reduction is 0.536832 s (6.05%).
Optimized ranges are 3.595057–3.650277 / 2.430490–2.499343 / 2.257352–2.295144 s. All individual observations,
including warmups and outliers, are retained in the structured results.

GPU backing peaks are unchanged at 1044.40 / 996.72 / 969.27 MiB.
CPU-continuous median backing peaks are 781.59 / 744.77 / 716.64 MiB.
These are managed allocator peaks, not process RSS; committed-byte peaks also
include outstanding reservations. Fitting and GPU command intervals are contained
within encoding and must not be added to complete-call wall time. Overlapping
worker elapsed intervals are not CPU-seconds.

Prepared fitting used Battery power; full-image timing used
AC power. Each comparison has fresh controls under
its own condition; these conditions are not pooled. The first battery full-image
attempt stopped before any encode because the machine had switched to AC during
validation. Its failed manifest is retained. Guards observed the requested power
condition throughout every accepted run and no competing codec, build or profiler
jobs. Ordinary desktop activity remained present.
These are matched session measurements, not peak-machine results. The fresh
controls should be used instead of substituting previous-session absolute times.
This study does not establish additive CPU/GPU throughput.

## Diagnostic attribution

Separate AC-power captures insert encoder boundaries
between phases, perturbing scheduling.
They explain the mechanism and are not substitutes for the uninstrumented timing
above. All variants replay the same prepared samples and retain exact trees.

| GPU phase | Previous (ms) | Parallel setup (ms) | Setup + summary (ms) |
|---|---:|---:|---:|
| Initialization | 0.176 | 0.178 | 0.179 |
| Begin nodes | 31.561 | 2.107 | 1.446 |
| Clear statistics | 2.048 | 1.569 | 1.561 |
| Parent statistics | 58.067 | 57.463 | 55.838 |
| Summarize properties | 24.011 | 24.080 | 3.884 |
| Schedule | 12.050 | 12.706 | 10.891 |
| Clear histograms | 20.362 | 20.266 | 20.674 |
| Collection | 132.713 | 136.058 | 131.880 |
| Normal scoring | 88.506 | 87.870 | 88.372 |
| Fused small jobs | 44.109 | 43.845 | 42.950 |
| Finish/partition | 27.499 | 24.116 | 27.080 |
| All recorded phases | 441.103 | 410.257 | 384.754 |

Node setup falls from 31.561 ms
to 1.446 ms, and property
summarization from 24.011 ms
to 3.884 ms.
These are single diagnostic captures, so small changes in untouched phases should
not be attributed to the implementation. Collection remains the largest recorded
phase. Reducing repeated sample reads during statistics and collection is the
next optimization to investigate, with complete-fitting and complete-encode
measurements retained as the decision criteria.

A separate one-round capture visits all 35,878 unique nodes in
147 rounds and records 482,729 property jobs. It independently
checks each node's compact-histogram demand, local-job eligibility, per-round job
counts and collection grid. The unchanged fused route handles
239,456 property jobs, leaving 3,478,356 regular
collection groups.

## Validation and artifacts

All six release suites pass: the combined reference suite, tiled cohort and
continuous pipeline suites, and resident, frontier and tiled kernel suites.
The expanded kernel test checks **111 complete trees and exact row permutations**,
19 atomic-failure cases and concurrent fitting. New cases cover uneven property
counts; bin counts around SIMD scan boundaries and the 256-bin limit; constant
properties; early stopping; and a full 64-fit cohort filling more than 128 active
slots. Existing narrow, 129-slot and 1,024-slot frontiers, histogram-pool deferral,
weighted counts, extra bits and local-only histogram paths remain covered.

Metal API/shader validation passes the expanded kernel test and five pipeline
failure/recovery cases. ASan/UBSan passes the tiled kernel, cohort and complete
continuous-pipeline suites plus those five failures. Leak detection is disabled;
the existing narrow metal-cpp nil-message suppression remains, and pinned libjxl
is not sanitizer-instrumented.

Raw runs, frozen binaries/libraries, source snapshots, patches, build/test logs,
process/power records and reproduction scripts are under
`build/frontier-parallel-bookkeeping-20261017/`. The
[structured results](gpu-frontier-parallel-bookkeeping-results.json) preserve
individual timing/memory observations, source/artifact hashes and exactness
evidence. The four pre-existing untracked repeat/pipeline reports remain unchanged.

Reproduction uses `run_frontier_shader_study.py` with `qualified-config.json`
(three blocks, three repetitions), then `run_frontier_tiled_qualification.py`
with `full-config.json`. Both runners are under `tools/modular_reference_learning/`.
Artifact-local `validate.py`, `profile-phases.py`, `profile-workload.py`,
`summarize.py` and `write-report.py` retain the validation and reporting workflow.
Exact commands, inputs and hashes are recorded in the manifests.
