# Small GPU histograms and paired child scoring

The next collection/scoring slice reduces complete prepared fitting by **5.70%**
and complete encoding by **3.60%** in the matched battery study below. It adds no
scratch or managed backing. GPU fitting remains slower than CPU fitting and
stays opt-in.

The baseline is commit `9a7902a` (`Optimize tiled GPU fitting and qualify
continuous scheduling`). Its frozen hybrid executable is byte-identical to the
optimized executable used in the preceding
[phase optimization study](gpu-frontier-phase-optimization.md). Both GPU versions
here use **256 active nodes, 64 MiB scratch, cohort scheduling**. CPU fitting with
continuous scheduling remains the full-encoder comparison target. Private
128/64 defaults and the public CPU-fitting default are unchanged.

## Implementation

The collection kernel previously built a 1024-entry shared hash table for every
property/predictor/512-row tile. It now uses that same count storage as a dense
histogram whenever `compact_bins * (symbol_span + 1) <= 1024`. The additional
column holds weighted extra bits. These groups clear only their actual cells,
add counts directly, and flush nonzero cells to the existing device histogram.
Larger histograms retain the bounded hash path. Both paths use exact integer
sums, including weights and extra bits; the branch is uniform within a group.

This eligibility rule uses histogram size, so it also benefits small histograms
belonging to large nodes. In the existing first-cohort workload capture for 1029,
it covers 325,745 of 482,729 property jobs (67.48%) and 2,190,050 of 3,957,268
collection threadgroups (55.34%). That cohort contains 42 fits, 5,285,712 distinct
rows and 35,878 visited nodes. The original capture and its hash are retained in
the structured results; this is workload coverage, not a timing estimate.

Scoring now evaluates the two children's entropy together, reusing reads of the
lower histogram. Each child retains the original integer totals, four float
accumulators, probability clamp, explicit logarithm operations and final
pairwise sum. There is no floating-point reassociation. The extra-bit cost
expressions, predictor iteration, strict tie comparisons, split selection and
partitioning are unchanged. Metal compilation still disables fast math and
implicit floating-point contraction.

The change adds no shared arrays, host buffers, dispatches, storage-layout fields
or admission rules. It preserves the existing GPU-resident node lifecycle and
failure-atomic publication. It does not establish a new occupancy limit or
identify a particular register/shared-memory constraint.

## Complete prepared fitting

All timing uses the Apple M4 Pro on battery with 14 CPU participants. The replay
uses the identical archived first cohort of 1029, with three process blocks per
configuration, one warmup and three measured repetitions in each process. The
configuration order rotates/reverses across blocks; CPU14 and GPU calls are
interleaved within each process. Validation layers and diagnostic captures are
disabled for these measurements.

| Configuration | Median complete fitting (s) | Measured range (s) |
|---|---:|---:|
| Previous tiled GPU | 0.503193 | 0.497192–0.539915 |
| Dense collection only | 0.483780 | 0.478375–0.493907 |
| Dense collection + paired scoring | 0.474533 | 0.464090–0.518308 |
| CPU14 control, pooled | 0.424461 | 0.398263–0.538046 |

Each GPU median has nine measured calls. The CPU row pools the 27 interleaved
controls. Dense collection improves the median by 3.86%; paired scoring adds
1.91% relative to dense collection alone. The combined improvement is 5.70%,
and the combined version beats the previous version in all three block medians.
Every one of the 72 CPU/GPU calls, including warmups, matches the native trees.

The boundary includes validation, allocation/upload, histogram work, partitioning,
node/queue management, canonicalization and publication. It excludes input
sampling, CPU input restoration, backend/domain creation and worker-pool creation.
The GPU command-span median falls from 0.475286 to 0.440572 s; that span includes
scheduling/stalls and is not pure ALU-active time. Peak managed backing is exactly
304,702,152 bytes for all GPU variants.

## Complete image qualification

The inputs are the same **4000 × 3000 RGB8 photographs 1029, 1039 and 1049**.
All configurations use the same learned `auto/auto/best/fit-off` policy, Metal
unique sample rows and 14 participants for the configured CPU stages. CPU-fitting
controls retain the hybrid pipeline's other Metal stages; they are not pure-CPU
encoders. GPU cohorts retain an 8 Mi-sample feature window; the established CPU
continuous control uses 16 Mi samples.

There is one warmup and three measured encodes per image/configuration, with
rotating order. The complete private encode boundary excludes backend/domain
creation, file I/O, independent decoding and hashing. Every one of the **36
encodes** matches the prior output hash, work counts and exact pinned-libjxl
decode. GPU coverage checks require all local fits and no CPU/admission fallback.

Times are seconds per image. The final column is the **sum of per-image medians
over all three photographs**, not time for one photograph.

| Hybrid configuration | 1029 | 1039 | 1049 | Sum |
|---|---:|---:|---:|---:|
| Previous GPU, 256/64 cohorts | 4.143998 | 2.745924 | 2.451022 | 9.340944 |
| Optimized GPU, 256/64 cohorts | 3.992830 | 2.639100 | 2.372599 | 9.004529 |
| CPU fitting, continuous | 3.103617 | 2.417735 | 2.038226 | 7.559577 |

The optimized GPU route improves all three images, reducing the sum by 0.336415 s
(3.60%). It remains **19.11% slower** than the matched CPU-fitting continuous
control. The optimized per-image ranges are 3.946089–4.037803,
2.620472–2.666412 and 2.328481–2.420058 s. All observations, including warmups and
outliers, are retained in the structured results.

Complete GPU-fitting medians fall from 3.040263 / 1.684127 / 1.511772 s to
2.881317 / 1.583322 / 1.431744 s. Managed backing peaks are unchanged at
1,044.40 / 996.72 / 969.27 MiB. The CPU-continuous medians are
786.39 / 744.84 / 720.83 MiB. These are managed allocator peaks, not process RSS;
committed-byte peaks also include outstanding reservations.

The machine remained in ordinary desktop use, including Spotlight and
WindowServer activity. Guards observed battery power throughout and no competing
codec/build/profiler jobs. This is a matched session comparison, not a peak-machine
claim. Absolute times from the earlier study should not be substituted for these
fresh controls. Overlapping worker elapsed intervals are not CPU-seconds, and
these measurements do not establish additive CPU/GPU throughput.

## Diagnostic attribution

A separate phase-encoder capture replays the same archive at 256/64. Both runs
visit 35,878 nodes in 147 frontier rounds and 10 submissions. It changes encoder
boundaries and is diagnostic evidence, not the benchmark above.

| GPU phase | Previous (ms) | Optimized (ms) |
|---|---:|---:|
| Collection | 162.927 | 145.000 |
| Scoring | 135.739 | 121.358 |
| Parent statistics | 58.737 | 60.847 |
| Summarize | 23.239 | 23.700 |
| Begin nodes | 31.216 | 31.202 |
| Schedule | 9.669 | 9.715 |
| Clear histograms | 22.030 | 22.243 |
| Finish/partition | 26.476 | 27.403 |

Collection decreases by 11.00% and scoring by 10.59% in this capture. The remaining
phases are similar; collection/scoring still account for about 60% of the
optimized diagnostic phase time. Full phase totals, including initialization and
statistics clearing, are retained in the results file.

## Validation and artifacts

Release tests pass for `metal_modular_tiled_frontier_histogram`,
`modular_reference_tiled_frontier` and `modular_reference_tiled_pipeline`.
The expanded kernel test covers histograms immediately around the dense/hash
boundary: 60/61 bins with a 16-symbol span and 3/4 bins with a 256-symbol span.
It includes weighted counts, zero/63 extra bits, partial tiles, predictor ties,
exact row permutations and native CPU tree comparisons. The complete kernel test
passes 34 tree/permutation cases, 19 atomic-failure checks and concurrent fits.

Metal API/shader validation passes that test and five continuous-pipeline
failure/recovery cases. ASan/UBSan passes the focused tiled kernel/cohort tests and
the five pipeline failure/recovery checks. Leak detection is disabled and the
existing narrow metal-cpp nil-message suppression is retained; pinned libjxl is
not sanitizer-instrumented. The full sanitizer pipeline suite was not repeated
for this device-only implementation change.

Raw artifacts, frozen executables/libraries, build logs, source snapshots, patches,
pilots, process/power records and manifests are in
`build/frontier-small-histograms-20261017/`. The
[structured results](gpu-frontier-small-histograms-results.json) retain individual
timings, memory observations, hashes and exactness evidence. The four pre-existing
untracked repeat/pipeline reports remain unchanged.

Reproduction uses the existing `run_frontier_shader_study.py` with three blocks
and three measured repetitions, and `run_frontier_tiled_qualification.py` with
the frozen previous executable override. Both scripts are under
`tools/modular_reference_learning/`; their exact configurations and commands are
archived. The artifact-local `profile-phases.py` and `summarize.py` reproduce the
diagnostic comparison and structured report.

Keep CPU fitting as the default and 256/64 cohorts as the explicit GPU comparison
point. A further collection/scoring design should aim to remove more work or
histogram traffic, especially for nodes that fit in a single row tile, and must
earn its complexity in complete fitting and complete encoding. Merely enlarging
the admitted frontier or enabling continuous GPU scheduling is not supported by
these results.
