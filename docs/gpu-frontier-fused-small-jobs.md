# Fused collection and scoring for small GPU fitting jobs

The small-job path reduces median complete prepared fitting by **4.28%** and
the sum of complete-encode medians by **1.68%** in this matched battery session.
Trees, codestream bytes and decoded pixels remain exact. Managed backing is
unchanged. The GPU route still takes **19.05% longer** than the hybrid encoder
using CPU fitting with continuous scheduling, so CPU fitting remains the default.

The baseline is `9bffcff95ff7add1378e01dc04b9588c84e86fe8`, the qualified
[dense collection and paired scoring implementation](gpu-frontier-small-histograms.md).
Both GPU variants use 256 active nodes, a 64 MiB scratch budget, 16 frontier
rounds per submission and cohort scheduling. This change applies within the
private tiled GPU executor; it does not promote GPU fitting or change the
private 128-node/64-MiB defaults.

## Implementation and storage

A property job qualifies when its node has at most **512 distinct rows** and
`compact_bins * (symbol_span + 1) <= 1024`. The extra column stores weighted
extra-bit costs. A dedicated 128-thread kernel collects, prefixes and scores
each qualifying job entirely in threadgroup memory. Predictors are processed
in their original order. Larger jobs retain the existing collection and scoring
kernels and their resource requirements.

The final kernel uses **4,896 bytes of static threadgroup storage**. Its histogram
and final shared winners occupy the same scratch union at different times, with
a barrier between them. At least four symbol slots plus one extra-bit column
bound the job to 204 compact bins. Each thread retains at most two threshold
winners while iterating predictors. Valid thresholds are assigned densely to
lanes, avoiding gaps between occupied original bins; the final scan restores
original bin order for deterministic ties and split preferences.

All histogram sums and prefixes remain integer operations. Each child's entropy
uses the same four float accumulators, explicit logarithm, clamp, pairwise sum,
extra-bit expression and strict predictor comparison as before. Metal compilation
retains `-fno-fast-math -ffp-contract=off`. Partitioning, final tree canonicalization,
failure-atomic publication and the no-hidden-CPU-retry rule remain intact.

The existing histogram-offset field uses a reserved, unaligned sentinel for
local jobs. Such jobs contribute no device-histogram bytes and no collection
threadgroups. Both the parallel all-admitted scheduler and its bounded pool
fallback handle this case, including rounds where every job is local. The
existing clear and normal-score dispatches return immediately for local jobs;
the new kernel returns immediately for other jobs. This adds one dispatch and
resource barrier per round, without a new command submission or CPU handoff.

The host still reserves the existing bounded scratch arena. Avoiding a job's
histogram traffic therefore does **not** imply a smaller host allocation or
lower managed backing peak. No storage-layout field or struct size changes.

## Work coverage and kernel resources

A fresh one-round diagnostic capture of the archived first prepared cohort of
1029 contains 42 fits, 5,285,712 distinct rows, 35,878 visited nodes and 482,729
property jobs. The new path handles **239,456 jobs (49.60%)**, across 24,774 nodes.
It eliminates **478,912 collection threadgroups (12.10%)**, leaving 3,478,356.
The larger nodes still account for most collection tiles. These are work counts,
not fractions of elapsed time.

The capture checks the routing rule, each node's histogram-byte demand, every
round's property-job count and its collection grid. All 35,878 node references
are unique. Exact native trees are checked independently by the replay.

Read-only pipeline inspection on the M4 Pro reports:

| Kernel | Static threadgroup bytes |
|---|---:|
| Existing collection | 9,216 |
| Existing scoring | 6,144 |
| New fused small jobs | 4,896 |
| Scheduler | 12,304 |

All four report an execution width of 32 and a maximum of 1,024 threads, against
a device limit of 32 KiB of threadgroup storage. These measurements establish
static resource reservations; they do not identify the occupancy manager's
limiting resource, register pressure or a bandwidth bottleneck.

## Complete prepared fitting

The Apple M4 Pro remained on **battery**, with 14 participants for CPU fitting.
Each configuration has three process blocks, each containing one warmup and
three measured CPU/GPU pairs. Configuration order rotates/reverses, and CPU/GPU
order alternates within each process. The previous executable and shader library
are frozen separately, preserving the baseline's original dispatch sequence.
The study runner now supports a per-configuration executable override for this
purpose. Validation layers and diagnostic captures are disabled during timing.

| Configuration | Median complete fitting (s) | Measured range (s) |
|---|---:|---:|
| Previous tiled GPU | 0.481851 | 0.462124–0.527149 |
| Fused small jobs | 0.461240 | 0.453271–0.468883 |
| CPU14, pooled controls | 0.422563 | 0.395791–0.503176 |

Each GPU row has nine measured calls; the CPU row pools 18 controls. The new
version's median is lower in all three blocks, by 1.05%, 5.74% and 6.36%.
The CPU controls also vary, so 4.28% is the observed session result, not a general
speedup guarantee. An earlier three-block pilot of the same final kernel showed
a 2.54% reduction. Initial versions with larger scratch and sparse threshold
assignment produced smaller or inconsistent block gains; those observations
and intermediate shader libraries are retained.

Every one of the **48 CPU/GPU calls**, including warmups, matches the native
trees. The boundary includes validation, allocation/upload, histogram work,
partitioning, node/queue management, canonicalization and publication. It
excludes sampling, CPU input restoration, backend/domain creation and worker-pool
creation. GPU command-span medians are 0.445891 s before and 0.427595 s after;
these include scheduling and stalls, and are not pure ALU-active time. GPU peak
managed backing is exactly 304,702,152 bytes for both variants.

## Complete image qualification

The inputs are the same **4000 × 3000 RGB8 photographs 1029, 1039 and 1049**.
All configurations use learned `auto/auto/best/fit-off`, Metal unique sample
rows and 14 participants for the configured CPU stages. GPU cohorts use an
8 Mi-sample feature window; the CPU continuous control uses 16 Mi samples.
The CPU-fitting control retains the hybrid pipeline's other Metal stages and
is not a pure-CPU encoder.

Each image/configuration has one warmup and three measured calls in rotating
order. The complete private encode boundary excludes backend/domain creation,
file I/O, independent decoding and hashing. All **36 encodes** match the prior
codestream hashes, work counts and exact pinned-libjxl decode. GPU coverage
requires all 192 local fits per image, without CPU-node or admission fallback.

Times are seconds per image. **Sum** means the sum of per-image medians across
all three photographs, not time for one photograph.

| Hybrid configuration | 1029 | 1039 | 1049 | Sum |
|---|---:|---:|---:|---:|
| Previous GPU, 256/64 cohorts | 3.968660 | 2.577584 | 2.368435 | 8.914680 |
| Fused GPU, 256/64 cohorts | 3.864809 | 2.552251 | 2.347797 | 8.764857 |
| CPU fitting, continuous | 3.039360 | 2.278474 | 2.044212 | 7.362046 |

The observed reductions are 2.62%, 0.98% and 0.87%, totaling 0.149823 s (1.68%).
The smaller changes on 1039 and 1049 are within run-to-run variation; individual
observations overlap and do not all favor fusion. Fused ranges are
3.839204–3.970535 / 2.547139–2.557917 / 2.320776–2.400510 s. All observations,
including warmups and outliers, remain in the structured results.

Complete GPU-fitting medians within those encodes fall from
2.871052 / 1.574469 / 1.423453 s to 2.802475 / 1.524339 / 1.409454 s.
Their summed medians decrease from 5.868974 to 5.736268 s. GPU command-span sums
decrease from 5.509239 to 5.382991 s. These fitting and command intervals are
contained in complete encoding; they must not be added to its wall time.

GPU managed backing peaks are unchanged at **1,044.40 / 996.72 / 969.27 MiB**.
CPU-continuous median backing peaks are 773.46 / 731.61 / 719.33 MiB in this
session. Managed backing is not process RSS; committed-byte peaks additionally
include outstanding reservations. Per-call values are retained in the results.

Power/process guards observed battery throughout and no competing codec, build
or profiler jobs. Ordinary desktop activity remained present. These are matched
session observations, not peak-machine measurements. Earlier-session absolute
times should not replace the fresh controls. Overlapping worker elapsed intervals
are not CPU-seconds, and this study does not establish additive CPU/GPU throughput.

## Diagnostic attribution

A separate capture inserts phase encoder boundaries. Each variant processes
35,878 nodes in 147 rounds and 10 submissions. The extra boundaries perturb
scheduling, so these numbers explain work movement rather than replace the
complete-call timings above.

| GPU phase | Previous (ms) | Fused (ms) |
|---|---:|---:|
| Collection | 146.645 | 133.454 |
| Normal scoring | 121.702 | 88.112 |
| Fused small jobs | — | 43.063 |
| Clear histograms | 22.278 | 19.663 |
| Parent statistics | 59.876 | 56.658 |
| Begin nodes | 31.187 | 31.883 |
| Summarize | 23.775 | 24.032 |
| Schedule | 10.443 | 11.131 |
| Finish/partition | 28.343 | 24.335 |

Collection, both scoring paths and histogram clearing together fall from
290.624 to 284.293 ms: **6.332 ms less**, after counting the new fused kernel.
The apparently large reduction in normal scoring mostly moves work into fusion.
Total recorded phase time, including initialization and statistics clearing,
falls from 446.188 to 434.126 ms. This remains a modest optimization: collection
and scoring still account for about 61% of the new diagnostic phase time.

## Validation, artifacts and follow-up

Six release suites pass: the combined reference suite, tiled cohort and continuous
pipeline suites, and resident, frontier and tiled kernel suites. The expanded
tiled test covers 511/512/513 rows; histogram cell boundaries with 4, 16 and 256
symbol spans; weighted counts and zero/63 extra bits; exact row permutations;
predictor ties; mixed local/device jobs; and a nontrivial tree requiring zero
device-histogram bytes. The kernel test passes **42 complete tree/permutation
cases, 19 atomic-failure checks and concurrent fits**.

Metal API/shader validation passes the kernel test and five pipeline
failure/recovery cases. ASan/UBSan passes the tiled kernel, cohort and full
continuous-pipeline suites, plus those five failure cases. Leak detection is
disabled and the existing narrow metal-cpp nil-message suppression is retained;
pinned libjxl is not sanitizer-instrumented.

Raw runs, frozen executables/libraries, source snapshots, implementation patch,
pilots, logs, process/power records and manifests are under
`build/frontier-fused-small-jobs-20261017/`. The
[structured results](gpu-frontier-fused-small-jobs-results.json) contain individual
timings, memory observations, source/artifact hashes and exactness evidence.
The four pre-existing untracked repeat/pipeline reports remain unchanged.

Reproduction uses `run_frontier_shader_study.py` with the archived
`qualified-config.json`, three blocks and three repetitions, followed by
`run_frontier_tiled_qualification.py` with `full-config.json`. Both runners live
under `tools/modular_reference_learning/`. Artifact-local
`profile-phases-final.py`, `profile-workload-final.py` and `summarize.py` reproduce
the diagnostics and structured report. Exact commands and frozen inputs are
recorded in their manifests.

Keep the fused path within the opt-in tiled GPU executor and retain CPU fitting
as the default. Larger-node collection and scoring remain the main measured
targets. Applying dense valid-threshold assignment to the remaining scoring
kernel is a bounded follow-up worth testing; its benefit is not established by
this small-job result. Further admission increases or continuous GPU scheduling
are not justified by this study alone.
