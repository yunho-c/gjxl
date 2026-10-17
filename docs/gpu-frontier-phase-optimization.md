# Tiled frontier phase optimization and continuous pipeline

This follows `b66cfbed`, the first tiled GPU-resident histogram fitter. The work
covers phase attribution, measured kernel changes, admission experiments,
continuous-pipeline integration, and matched complete-encode qualification.
The public CPU-fitting default and the learned compression policy are unchanged.

The implementation retains exact floating-point scoring order, predictor
preferences, split ties, sample partitions, final tree numbering, and codestream
bytes. The resident per-fit limit remains 131,072 distinct rows. There is no
implicit CPU retry after GPU execution begins.

## Attribution

The prepared input is the first real fitting cohort from photograph 1029:
42 fits, 5,285,712 rows and 35,878 nodes. Early, middle and final command buffers
were captured separately. Complete-cohort phase attribution uses a diagnostic
build with one compute encoder per phase, because direct dispatch timestamp
sampling is unavailable on this device. Encoder boundaries perturb scheduling;
these are diagnostic measurements, separate from ordinary benchmark timings.

With the original tiled shader, the phase durations sum to 717.12 ms:

| Phase | Diagnostic GPU time | Share |
|---|---:|---:|
| Score candidate splits | 196.13 ms | 27.35% |
| Collect histograms | 180.10 ms | 25.11% |
| Schedule histogram work | 136.19 ms | 18.99% |
| Parent statistics | 61.51 ms | 8.58% |
| Summarize properties | 45.31 ms | 6.32% |
| Select split and partition | 42.44 ms | 5.92% |
| Begin nodes | 30.46 ms | 4.25% |
| Clear histograms | 21.57 ms | 3.01% |
| Clear statistics and initialize | 3.41 ms | 0.48% |

Separate shader sampling shifts from collection in the early capture (58.09%)
to scoring and scheduling in the tail (37.25% and 24.05%). Sampled cost is not
additive elapsed time. Compiler reports show 86 temporary registers and no spills
for score, 32/no spills for collect, and 31/no spills for the original scheduler.
These counts alone do not identify the occupancy manager's limiting resource.
Counter averages restricted to shader execution windows were explored but are
excluded from conclusions: separate replay passes can execute different nodes.

A one-round-per-submission diagnostic independently confirms the phase ordering
and records each processed node exactly once:

| Rows in node | Nodes | Row visits | Property jobs |
|---|---:|---:|---:|
| 1–32 | 2,690 | 64,325 | 34,242 |
| 33–128 | 11,247 | 810,667 | 150,566 |
| 129–512 | 10,979 | 2,958,989 | 148,718 |
| 513–2,048 | 7,232 | 7,448,674 | 98,297 |
| 2,049–8,192 | 2,661 | 9,895,379 | 36,496 |
| 8,193–131,072 | 1,069 | 40,854,377 | 14,410 |

Nodes with at most 128 rows are 38.85% of nodes but only 1.41% of row visits.
This exposes the difference between row-processing work and fixed work per
node/property. Mixed rounds classified by median node size do not provide
separate timing attribution for individual small nodes.

## Implemented changes

The scheduler now builds counts and task lists across lanes when every pending
node fits in the histogram pool. One lane performs a short prefix over counts;
the lanes then write disjoint task ranges in the original slot/property order.
If the pool cannot admit every pending node, the original serial first-fit
policy runs. The byte prefix saturates safely instead of overflowing on wide
frontiers. Its bounded shared prefix array uses 12 KiB. Property-job and
selected-node counts share a word: the contract permits at most 16,384 property
jobs, so the low 16 bits cannot carry into the selected count.

Integer histogram prefixes now use SIMD scans across bins, while preserving the
subsequent floating-point entropy calculation exactly. Zero extra-bit counts
skip redundant atomic additions. Symbol maxima reduce within each SIMD group
before updating device memory.

Three other experiments were exact but did not improve the complete-fit pilot:
four-lane entropy evaluation, an adaptive shared hash-table size, and direct
device-atomic histogram collection. They were removed; their artifacts remain.

The accepted shader comparison uses the same frozen host executable, external
shader libraries, three order-balanced process blocks, one warmup and two
measured repetitions per process. Each process also measures CPU fitting with
14 participants on identical prepared samples. This study ran on AC power.

| Shader | Complete GPU fit median | Range |
|---|---:|---:|
| Original tiled | 0.708139 s | 0.700285–0.711267 s |
| Parallel scheduler | 0.580233 s | 0.578260–0.592015 s |
| Scheduler + SIMD prefixes | 0.542864 s | 0.539034–0.545399 s |
| All retained changes | 0.526205 s | 0.522893–0.536320 s |

The reduction is 25.69% for this complete prepared-cohort fitting boundary,
including input upload, partitioning, node management and publication. It is
not a full-image speedup. All 72 calls and 3,024 tree comparisons are exact.
Matched CPU medians are 0.410188–0.411096 s across the four shader variants;
the optimized GPU median is still 28.12% slower than its 0.410712 s CPU control.

Those initial AC measurements used a 16 KiB prefix array. Metal shader
validation subsequently exposed 32,800 bytes of instrumented shared memory,
above the 32 KiB device limit. The final packed representation reduces the
array to 12 KiB and passes validation, including 1024-slot cases. This was a
validation-instrumentation limit; it does not identify the original runtime
occupancy limiter.

A fresh matched battery study repeats the original/final comparison after that
change: original **0.716423 s** (0.709595–0.744804), final **0.532316 s**
(0.525007–0.547623), a **25.70%** reduction. All 36 calls and 1,512 tree
comparisons are exact. Its CPU controls are 0.406036 and 0.421497 s, respectively;
the final GPU route remains slower than CPU fitting. AC and battery observations
are not pooled.

A separate paired battery diagnostic confirms the mechanism: scheduler encoder
time falls from 131.85 to 6.42 ms, and score from 189.85 to 147.85 ms. The final
diagnostic totals 512.27 ms, with collection and scoring now the largest phases.
These instrumented timings describe the changed work; the uninstrumented study
above supplies the performance claim.

The final phase shares are collection 31.77%, scoring 28.86%, parent statistics
11.68%, property summarization 8.68%, split/partition 7.59%, node setup 5.89%,
histogram clearing 3.69%, scheduling 1.25%, and initialization/statistic clearing
0.59%. Scheduler serialization is substantially reduced; collection/scoring
remain the next kernel targets. This does not establish a specific register,
bandwidth or occupancy-manager limit for those kernels.

## Admission and scratch

A separate matched AC sweep uses the optimized library and the same repetition
protocol. Scratch is the total tiled scratch allowance, including per-slot
metadata and the shared histogram pool, rather than just histogram storage.

| Active capacity / scratch | Complete GPU fit median | Maximum deferred nodes, observed range |
|---|---:|---:|
| 128 / 64 MiB | 0.549608 s | 0 |
| 128 / 48 MiB | 0.550501 s | 6–14 |
| 256 / 64 MiB | 0.522994 s | 57–60 |
| 256 / 96 MiB | 0.513946 s | 0 |
| 512 / 64 MiB | 0.622951 s | 354–368 |
| 512 / 96 MiB | 0.497287 s | 180–201 |

Increasing capacity without enough scratch can regress. The 256/64 configuration
is a useful balanced candidate; 512/96 trades more memory for a better result on
this cohort. All 108 calls and 4,536 tree comparisons are exact. The two studies
were separate timing blocks: compare configurations within each study rather
than interpreting their different absolute times as a regression.

## Continuous pipeline

The private continuous schedule now supports complete resident DFS, frontier,
and tiled-frontier fitting. CPU workers prepare samples into bounded, managed
owners. The coordinator submits ready groups of at most 64 fits to the GPU while
workers can prepare later groups. GPU node setup, statistics, split selection,
partitioning and next-node state remain resident.

A feature batch remains alive until its own trees have been published and its
tokens produced. Under admission pressure a prepared group prefix can be fitted
to retire owners; samples are not recomputed. Admission retries occur only
before execution starts. Error paths join workers before releasing their
feature borrows, and failed operations preserve the previously published frame.

The existing cohort route and the continuous route share the same complete
resident fitting helper and resource plan. Private profiling options expose
`--resident-active=N` and `--resident-scratch=MiB`; their defaults remain
128 and 64. Continuous feature windows remain bounded to 8 or 16 Mi samples.
The legacy CPU-driven incremental GPU fitter is still unsupported in this
pipeline branch.

## Complete encodes

All 84 accepted calls are byte-identical to the prior oracle and decode exactly.
Every GPU configuration fits all 192 nonempty trees per image on the GPU, with
zero CPU fitting nodes or admission fallbacks. The initial AC run stopped when the machine
switched to battery. Its affected encode and partial matrix are excluded from
the accepted comparison; the raw artifacts remain in `full-study/`.

The matched replacement uses battery power throughout, 14 CPU participants,
4000 × 3000 RGB8 photographs 1029, 1039 and 1049, one warmup and three measured
encodes per configuration. CPU-fitting controls retain the hybrid pipeline's
Metal stages and are not pure-CPU encoders. The boundary is the complete private
encode, excluding backend/domain creation, file I/O, independent decoding and
hashing. No diagnostic capture or validation layer is enabled during timing.
The machine remained in ordinary desktop use: snapshots include Spotlight
indexing (`mds_stores`, often near one CPU core) and WindowServer activity.
The runner checks for competing codec/build jobs and power changes throughout;
timing ranges are retained, including outliers. These are matched session
measurements, not a claim of peak machine throughput.

Times below are seconds per image, followed by the **sum of per-image medians
across all three photographs**. Each median uses three measured encodes.

| Hybrid configuration | 1029 | 1039 | 1049 | Sum |
|---|---:|---:|---:|---:|
| Previous tiled GPU, 128/64, cohorts | 5.421051 | 3.387712 | 3.048960 | 11.857724 |
| CPU fitting, cohorts | 3.406257 | 2.475797 | 2.183220 | 8.065274 |
| CPU fitting, continuous | 2.953483 | 2.176648 | 2.042444 | 7.172576 |
| Optimized tiled GPU, 128/64, cohorts | 4.248848 | 2.755514 | 2.477749 | 9.482112 |
| Optimized tiled GPU, 256/64, cohorts | 4.085994 | 2.645615 | 2.369886 | 9.101494 |
| Optimized tiled GPU, 256/64, continuous | 4.051059 | 2.818113 | 2.452003 | 9.321175 |
| Optimized tiled GPU, 512/96, continuous | 4.118033 | 2.811661 | 2.412343 | 9.342036 |

The best GPU configuration here is **256 active nodes, 64 MiB scratch,
cohort scheduling**. It is 23.24% faster than the previous tiled route, but
26.89% slower than the CPU-fitting continuous control. Its per-image timing
ranges are 4.084996–4.090174, 2.634621–2.650854 and 2.368497–2.385514 s.
The structured results retain every configuration's observations and ranges;
for example, the CPU-cohort 1029 runs range from 3.336791 to 4.358200 s.

Continuous GPU scheduling is 2.41% slower in the summed medians than matched
256/64 GPU cohorts. It helps 1029 by 0.035 s but costs 0.172 s on 1039 and
0.082 s on 1049. This experiment therefore does not support promoting that
schedule. The prepared-cohort advantage of 512/96 also does not carry through
to the complete images.

Complete GPU-fitting medians for 256/64 cohorts are 3.002450, 1.656727 and
1.471298 s; their GPU command spans are 2.883299, 1.559785 and 1.377530 s.
Continuous 256/64 fitting is nearly identical at 3.001946, 1.658221 and
1.473321 s. Pipeline scheduling did not materially shorten the fitter itself.
Command spans include GPU scheduling/stalls and are not pure ALU-active time.
CPU preparation, fitting, tokenization and owner lifetimes can overlap; summing
worker elapsed intervals would not produce thread CPU-seconds or a wall-time
breakdown.

| Median managed backing peak (MiB) | 1029 | 1039 | 1049 |
|---|---:|---:|---:|
| CPU fitting, continuous | 791.92 | 744.14 | 716.58 |
| Tiled GPU, 256/64, cohorts | 1,044.40 | 996.72 | 969.27 |
| Tiled GPU, 256/64, continuous | 1,280.72 | 1,242.26 | 1,203.04 |
| Tiled GPU, 512/96, continuous | 1,312.72 | 1,274.26 | 1,235.04 |

Continuous 256/64 retains roughly 234–246 MiB more backing than GPU cohorts
without a summed complete-call win. Increasing scratch to 96 MiB adds another
32 MiB. These are managed allocator peaks, not process RSS; committed-byte
peaks also include outstanding reservation capacity. All observed storage stays
within the admission plans.

## Validation and reproduction

The release tests `metal_modular_tiled_frontier_histogram`,
`modular_reference_tiled_frontier` and `modular_reference_tiled_pipeline` pass.
The expanded kernel test includes active widths 129 and 1024. Continuous tests
compare native CPU trees, ANS/prefix bytes and exact pinned-libjxl decoding,
participant limits, feature reuse, multiple live batches and owner release.
Injected GPU submission/completion errors, partial worker launch failure,
admission rejection and unsupported shared-fit size preserve previous output.
An additional nine-group smoke comparison covers continuous CPU, resident DFS
and monolithic frontier fitting: all three routes produce identical bytes and
exact decodes, and both GPU routes fit all nine trees without CPU fallback.

Metal API/shader validation passes for the final kernel and resident pipeline
failure/recovery cases. ASan/UBSan passes the focused kernel/cohort tests and the
full continuous-pipeline test, with halt-on-error. After the device-only prefix
packing change, the focused sanitizer tests and pipeline failure/recovery cases
were repeated and passed. Leak detection is disabled; the existing narrow
metal-cpp `NSObject.hpp` nil-message suppression remains. The pinned libjxl
decoder is not sanitizer-instrumented.

The combined legacy fitter test originally asserted that no cached backing could
remain after frame release. The original executable reproduced that failure:
the retained bytes were intentionally idle cache capacity. The test now calls
the explicit cache-trim API before asserting zero ownership; production caching
is unchanged. An initial rejection fixture also turned out to be eligible; the
replacement explicitly exercises a shared fit above the existing row limit.

Build the resident probe, hybrid profiler and the two focused test executables.
Prepared shader/admission comparisons use
`tools/modular_reference_learning/run_frontier_shader_study.py`, with a JSON
configuration list containing name, metallib, active capacity, scratch MiB and
rounds per submission. Full encodes use
`tools/modular_reference_learning/run_frontier_tiled_qualification.py`; its
optional `--variants` JSON accepts fitter, schedule, feature window, extra
arguments and a frozen binary override for the previous implementation.

Diagnostic builds enable `GJXL_BUILD_MODULAR_FRONTIER_DIAGNOSTICS` and use
`GJXL_FRONTIER_DIAGNOSTIC_DIR` plus `GJXL_FRONTIER_PROFILE_PHASES=1`.
One-round submissions additionally emit per-node workload records. These hooks
are disabled in ordinary builds.

Raw captures, replay exports, frozen binaries/libraries, manifests, source
patches, logs and summary scripts are under
`build/frontier-phase-study-20261017/`. The
[structured results](gpu-frontier-phase-optimization-results.json) retain individual
timings, memory bounds and artifact hashes. The four pre-existing untracked
repeat/pipeline reports are preserved unchanged.

## Follow-up boundary

Keep CPU fitting as the default. The useful next kernel experiment is to reduce
the remaining collection/scoring work, including a specialized path for small
nodes, while retaining exact arithmetic and checking complete fitting time.
Larger frontiers alone are not a substitute for that work reduction. The new
continuous route provides a qualified integration point for future experiments;
its additional feature/sample lifetimes must earn their memory cost in complete
encodes. For further GPU experiments, use the explicitly selected 256/64 cohort
configuration as the measured comparison point; the 128/64 private defaults
remain unchanged. No additive CPU-plus-GPU throughput or 2× encoder speedup is established
by these experiments.
