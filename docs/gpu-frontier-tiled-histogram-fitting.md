# Tiled GPU-resident frontier fitting

This prototype separates parent summaries, joint histogram collection, scoring,
and final selection/partition into GPU phases. It retains the existing monolithic
frontier executor for comparison. The public encoder still uses CPU fitting;
the private qualification route selects `fitter-tiled` explicitly.

It follows the [large-admission and descriptor study](gpu-frontier-high-admission.md).
That study showed that removing immutable descriptor copies sharply reduced
spills and occupancy-target depression, but did little to raise actual compute
occupancy. The descriptor-reference change is now applied to both existing
resident executors; arithmetic and scheduling are unchanged by that change.

## Operation and storage

The input remains identical prepared weighted samples. Eligibility, native
float arithmetic, strict predictor/property/threshold ties, tree-capacity errors,
canonical publication, and failure-atomic output match the existing resident
operation. No sample truncation, CPU node decisions, or implicit CPU retry is
introduced. Publication still renumbers nodes to native DFS order.

Each command encodes a bounded number of GPU cycles:

1. Assign ready nodes to free active slots and build a compact row-task list.
2. Clear new slots and collect parent histograms, extra-bit sums and property
   occupancy in independent 512-row tiles.
3. Compute parent costs, inherited constants and original-to-dense property-bin
   maps. Keep original symbol indices, including zero-count symbols, so native
   four-lane floating-point accumulation is unchanged.
4. Admit whole nodes' joint histograms into a shared fixed-size pool. Nodes that
   do not fit retain their summaries in active slots for a later cycle.
5. Clear admitted histograms and collect independent property/predictor/row
   tiles. Each threadgroup uses a 1,024-entry shared hash table for at most 512
   row keys, plus 256 extra-bit counters, then adds integer sums to the pool.
   This avoids rescanning a node once for each histogram-bin tile.
6. Prefix compact histograms and score thresholds. Predictors are considered in
   native order. Candidate thresholds still use original bin numbers and
   unweighted row occupancy; histogram increments use row weights.
7. Select final splits in native property/category order, perform the exact
   native two-pointer partition permutation, append children, and free slots.

All cross-threadgroup dependencies are dispatch barriers. No group spins while
waiting for another group to finish. CPU interaction remains at coarse command
completion, error handling and final publication. The host checks both the ready
queue and retained live slots; queue exhaustion alone does not mean completion.

The storage planner reserves immutable rows, nodes/queue/permutations, active
summary slots, bounded task lists, the histogram pool and publication storage.
It guarantees room for at least one worst-case legal node histogram before
submission. This makes progress possible even when other nodes must wait.
Scratch limits can reduce active capacity or reject admission; they never cause
hidden allocation growth. The pool compacts occupied **property bins**, while
retaining the full rounded symbol span needed by exact entropy arithmetic.

More active slots consume more retained summary storage, leaving less histogram
pool space under the same cap. Consequently, a requested capacity need not be
fully occupied, and increasing it need not improve throughput.

## Prepared-sample measurements

Measured on the M4 Pro (20 GPU cores, 14 CPU participants, 48 GiB), AC power,
using the same archived first cohort from photo 1029: **42 fits, 5,285,712 rows,
35,878 final tree nodes**. This is one complete fitting cohort, not a whole-image
encode. Timing includes validation/upload, scratch allocation, summaries,
histograms, selection, partition/queue management, canonicalization and
publication. Sampling, CPU sample restoration, backend/domain/pool creation
and oracle comparison are outside timing.

Three process blocks vary configuration order; each process includes one warmup
and two measured calls for GPU fitting and CPU14. All 90 complete-fit calls
(3,780 trees) are exact. The table contains six measured GPU calls per row and
30 matched CPU observations. No competing codec/build work, power change or
thermal warning was recorded.

| Executor / active capacity | Median fitting time (range), s | Scratch MiB | Median managed peak MiB |
|---|---:|---:|---:|
| CPU14 | 0.408 (0.389–0.492) | — | 162.45 |
| Monolithic, 128 | 1.173 (1.133–1.199) | 33.31 | 259.90 |
| Monolithic, 512 | 0.761 (0.760–0.770) | 133.25 | 359.84 |
| Tiled, 128 | **0.694 (0.686–0.695)** | **64.00** | **290.59** |
| Tiled, 512 | 0.701 (0.695–0.705) | 64.00 | 290.59 |
| Tiled, 1024 | 0.782 (0.776–0.788) | 64.00 | 290.59 |

At 128 slots, tiling reduces complete fitting time by **40.8%**. It also takes
**8.9% less time than the 512-slot monolithic control**, with 52.0% less scratch
and 19.2% less managed peak backing. It still takes **1.70× CPU14's time**.
The descriptor-reference fix is active in both GPU executors in this table.

The slot itself shrinks from 266.5 KiB to 36.75 KiB, but the new shared histogram
pool is additional storage. Consequently, the 128-slot tiled route uses
**30.69 MiB more total scratch/managed peak than the 128-slot monolithic route**.
The memory improvement is relative to the wider 512-slot control, not the
old 128-slot default. Managed backing excludes the common archive, driver,
profiler and process-wide resources; it is not RSS.

The tiled 128-slot route has no pool deferrals on this cohort and uses up to
47.21 MiB of histogram payload. At 512 slots, retained summaries leave only
45.49 MiB for histograms and the maximum deferred-node count is 349–370.
At a 1024-slot capacity, only 26.98 MiB remains for histograms; active slots
peak at 664–722 and deferred nodes at 447–502. Both capacities are admitted
exactly, but the 1024 limit is never filled. Its increased waiting raises GPU
cycles to 170–173, versus 104–106 at 512. The 128-slot route needs 285 cycles,
but executes each much more cheaply. Cycle count alone does not predict time.

Median command-buffer GPU spans are 0.666 s for tiled128 and 0.734 s for
monolithic512; complete calls add about 28 ms in each case. These spans include
the command's phases and barriers, rather than summing CPU worker intervals.
This experiment neither establishes additive CPU/GPU throughput nor attributes
the remaining GPU time to individual phases.

An initial sweep stopped because its runner incorrectly required all admitted
slots to become occupied. Those partial timings are retained separately; the
table uses only the subsequent complete three-block sweep. The corrected runner
checks the bound and reports whether it was filled.

## Descriptor-reference full-image qualification

Before introducing tiled execution, paired frozen binaries compared copied
descriptors with device references on photos 1029/1039/1049 (4000×3000 RGB8 each).
Each configuration had one warmup and three measured fresh-process calls per
photo, with alternating order, AC power and 14 CPU participants. All 24 calls
produced identical bytes and exact decoded pixels.

| Photo | Copied descriptors, s | Device references, s |
|---|---:|---:|
| 1029 | 8.245 | 8.019 |
| 1039 | 5.214 | 5.066 |
| 1049 | 4.431 | 4.298 |
| Sum of per-image medians | **17.890** | **17.383** |

The complete-call reduction is **2.8%**. These are coarse-cohort, 128-slot
monolithic-frontier encodes, not CPU-fitting controls or tiled measurements.

## Full-image tiled qualification

The final matched comparison uses one frozen executable, AC power and 14 CPU
participants on the same three 4000×3000 RGB8 photos. Each route has one warmup
and three measured fresh-process calls per photo, with rotated order. All
**48 encodes** match the earlier output hashes and decode exactly. GPU routes
complete all 192 nonempty local fits per image, with no CPU node work or
admission fallback. The combined descriptor and tiled studies total **72 exact
full-image calls**. All six focused Release tests pass.

Both CPU-fitting controls retain the hybrid encoder's Metal stages. The CPU
continuous pipeline uses its existing 16 Mi sample feature window; CPU cohorts
and both GPU routes use the same 8 Mi window. GPU fitting remains integrated
through coarse cohorts. Continuous scheduling's existing CPU-fitter requirement
is unchanged.

| Photo | CPU fitting, cohorts | CPU fitting, continuous | Monolithic GPU, 128 | Tiled GPU, 128 |
|---|---:|---:|---:|---:|
| 1029 | 3.395 s | **3.022 s** | 8.077 s | 5.461 s |
| 1039 | 2.488 s | **2.150 s** | 5.172 s | 3.437 s |
| 1049 | 2.185 s | **2.033 s** | 4.356 s | 3.089 s |
| **Sum of per-image medians** | **8.067 s** | **7.206 s** | **17.605 s** | **11.987 s** |

These are complete encode-call times, excluding input/output file I/O, setup
and validation. The sums cover **all three images**, not each individual image.
Tiling reduces time by **31.9%** relative to the matched monolithic GPU route.
It still takes **1.49× the CPU-cohort time** and **1.66× the continuous CPU-fitting
pipeline time**. The new path remains opt-in.

Per-image medians for complete GPU fitting alone fall from
6.991 / 4.140 / 3.433 s to 4.383 / 2.413 / 2.170 s. GPU fitting submission
counts are 121 / 62 / 56 for monolithic and 121 / 63 / 56 for tiled: the
improvement does not come from fewer CPU waits. Most of the remaining fitting
wall time is still inside GPU command-buffer spans, rather than the roughly
0.11–0.15 s per image outside those spans. These figures do not identify which
individual GPU phase dominates.

Median managed peaks across the three images are 969–1044 MiB for tiled,
939–1014 MiB for monolithic128, 640–743 MiB for CPU cohorts and 724–790 MiB for
the CPU continuous pipeline. Tiled adds 30.69 MiB over monolithic128, reflecting
the larger total scratch allowance discussed above. There is no whole-encode
memory reduction relative to that 128-slot control.

The next useful investigation is phase-level GPU profiling of the new executor,
sampling both large early nodes and small-node tails. Separate collection,
scoring and partition kernels now make that attribution practical. The present
results support keeping the 128-slot tiled prototype for that work; they do not
support promoting GPU fitting over CPU fitting or increasing its active limit.

## Correctness qualification

The focused tiled Metal test compares 30 complete trees and exact row
permutations against native, resumable and scalar-frontier CPU oracles. It covers
1/2/14 predictors, predictor/property reordering, constant and sparse properties,
all symbol spans, large weights/extra-bit sums, partial row tiles, and three
dense fits that force at least two nodes to retain summaries while waiting for
the histogram pool. Widths 1/3/17, one/three-round submissions and concurrent
operations preserve exact results. Nineteen injected/preflight/capacity failures
preserve previous output, including failure after earlier completed GPU work
and during CPU publication.

Metal API and shader validation pass. ASan/UBSan focused fitting and hybrid
integration tests pass with halt-on-error; leak detection is disabled and the
existing narrow metal-cpp `NSObject.hpp` null-message suppression is retained.
The pinned libjxl decoder is not sanitizer-instrumented. The initial expanded
test fixture incorrectly iterated unused column slots; its host crash was fixed
before timing. The first small-budget test also requested less space than the
documented worst-case single-node guarantee; the corrected test uses 8 MiB.

## Reproduction

Build `gjxl_modular_resident_fitting_probe`, `gjxl_modular_hybrid_profile`,
`gjxl_metal_modular_resident_histogram_test` and
`gjxl_modular_reference_hybrid_test`. The focused tests are
`metal_modular_tiled_frontier_histogram` and `modular_reference_tiled_frontier`.

Prepared-sample comparisons use:

```sh
python3 tools/modular_reference_learning/run_frontier_tiled_study.py \
  PROBE ARCHIVE.gjfs OUTPUT_DIR --power AC
```

The probe accepts `cpu-tiled` in place of `cpu-frontier`; active capacity,
rounds per submission and total scratch MiB follow that argument. Full-image
qualification uses:

```sh
python3 tools/modular_reference_learning/run_frontier_tiled_qualification.py \
  HYBRID_PROFILE PRIOR_FULL_STUDY_RESULTS.json OUTPUT_DIR --power AC
```

Both runners freeze the executable, record source/input hashes and power/activity
snapshots, refuse existing output directories, retain warmups, and fail on
interfering codec/build work or power changes. Full-image qualification checks
work counts, output hashes, exact decode and complete GPU fit coverage. No
diagnostic capture or validation layer is enabled during timing.

The [machine-readable results](gpu-frontier-tiled-histogram-fitting-results.json)
retain individual observations, ranges, operation bounds, source hashes and
artifact hashes. Raw manifests, frozen executables, source patch/new sources,
exact output files, build/test/sanitizer logs and the summary script are under
`build/frontier-tiled-20261017/`. The accepted fitting sweep is
`prepared-study-complete/`; the earlier partial sweep remains in
`prepared-study/` and is excluded. All measured source hashes and accepted
output/binary hashes were checked against the retained artifacts.
