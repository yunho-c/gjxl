# GPU-resident frontier histogram fitting

This continues the [resident prototype](gpu-resident-histogram-fitting.md),
committed as `11bd897`, on `feat/gpu-resident-fitting`. The independent pipeline
branch and capacity-reuse commits remain intact. The referenced “Assess GPU
implementation readiness (2.2)” conversation distinguishes ready-node
parallelism from residency and recommends a narrow scalar frontier reference,
exact canonical numbering, and complete-fit measurements before integration.

The subsequent [tiled-frontier prototype](gpu-frontier-tiled-histogram-fitting.md)
adds a separate executor with GPU row tasks, compact histogram storage and
separate scoring/partition phases. This report describes the original
monolithic frontier implementation and its historical measurements.

## Execution and exactness

The resident operation now offers `kDepthFirst` and `kFrontier` executors.
Frontier mode maintains an append-only GPU queue of `(fit, node)` references.
A dispatch removes a bounded set of ready nodes; a second, indirectly sized
dispatch processes them concurrently. This can include siblings and more
distant relatives from the same tree, but never an ancestor and descendant.
Children become eligible only after the producing dispatch has completed and
its writes are visible. There is no global spin barrier between workgroups.

Each active node owns its row range, inherited constants, predictor and depth.
The GPU builds its parent summary, evaluates candidates, applies the exact
native preferences, partitions its rows, atomically allocates two children and
appends their references. Allocation order may vary; split decisions do not.
Partitioning preserves the native two-pointer swap permutation, so operations
on disjoint ranges commute exactly.

The initial executor uses one workgroup per ready node. Within a node it visits
properties and predictors in sequence. Histogram collection uses bounded
threadgroup-memory tiles, reusing parent-summary scratch after copying the
summary. This removes the depth-first prototype's device-atomic collection.
Both executors share the native entropy arithmetic and partition helper.
This milestone does not yet distribute one node's properties across several
workgroups or subtract a child histogram from its parent.

Native child IDs encode depth-first allocation order. Final CPU publication
validates the GPU tree, then performs a bounded traversal that copies it into
that canonical order. This occurs once after all GPU fitting completes and is
included in measured fitting time. It makes no split or predictor decisions.
The entire output cohort remains private until every tree passes validation.

The scalar FIFO reference reuses the native node preparation, evaluation,
selection and partition functions. It deliberately changes traversal order and
then canonicalizes. It is a differential oracle, not a production CPU route.

## Scheduling and storage

Default execution admits up to 128 active nodes and encodes 16 ready batches
per command buffer. The GPU writes the indirect dispatch count. The CPU reads
only coarse progress/error counters between command buffers, then submits
another bounded chunk if work remains. It never reads candidate results or
constructs the next frontier. Empty remaining rounds are safe no-ops.

The queue capacity is the sum of admitted tree-node capacities. Each node is
enqueued once, so it cannot grow beyond this bound on a successful fit. Node
arrays, immutable columns and row/partition permutations stay resident for the
cohort. Scratch is reserved per active slot and reused between batches. Its
64 MiB default cap can reduce the admitted width; inability to fit one slot
returns `Unsupported`. Invalid width/round controls fail before submission.
Tree/depth overflow fails the operation rather than truncating a tree.

The admission plan includes GPU queue/control/scratch, completed CPU trees,
and temporary canonicalization backing. Caller-owned samples and old output
remain excluded. Errors after an earlier completed GPU batch still preserve
the old output and release the private candidate. Pending commands drain
before their buffers are released.

The private encoder selects this mode with `fitter-frontier`; automatic/public
routing still uses its established policy. Coarse preparation cohorts are
retained. Combining this executor with the separate continuous-pipeline work
and reducing the inherited conservative job reservations remain later tasks.

## Correctness and failure qualification

The targeted release integration set passed all 10 tests, covering the native
fitter, fitting plans and queue, both resident GPU executors, private encoder
integration and resource budgeting. The final frontier-focused set passed
3/3 in release and 3/3 under ASan/UBSan. The scalar FIFO reference also matches
native trees and physically reordered samples on the existing fitter cases,
including forced multiplier splits that are outside this GPU operation's
eligibility.

Metal API and shader validation passed 27 complete-tree and exact-row-order
cases, 19 failure checks, and concurrent operations on one backend. Widths
1, 3 and 17 and one/three-round command chunks preserve exact results. A small
scratch budget reduces width; insufficient scratch rejects admission. Tests
also exercise failure after earlier completed GPU work, publication-allocation
failure, and recovery while preserving the previous output.

The first shader-validation attempt exceeded the instrumented kernel's
resource limit. Reusing the parent-summary counters for histogram tile scratch
reduced static threadgroup storage; the final instrumented run passed. Both
logs are retained. Sanitizer runs disable leak detection and narrowly suppress
the existing metal-cpp null-instance message-dispatch diagnostic at
`third_party/metal-cpp/*NSObject.hpp`. The pinned libjxl decode oracle is not
sanitizer-instrumented; GPU accesses are covered separately by shader validation.

## Prepared-sample fitting measurements

Measured on the M4 Pro, 14 CPU participants, **AC power**, with one excluded
warmup and six measured repetitions per variant. Six variants rotate through
all six order positions. Every image supplies 16 deterministic 128×128
single-channel tiles, yielding 171,424 prepared rows. These are replay batches
derived from the photographs, not their complete encoding fits.

The replay compares CPU1, CPU14, resident depth-first, frontier, and the same
eight-frontier/eight-CPU assignment run sequentially and concurrently. Setup,
upload, partition, node/queue management, canonicalization, validation and
publication are inside the timer. Shared immutable input preparation and
worker-pool creation are outside; CPU input restoration time and backing are
recorded separately. Process CPU time is measured independently of wall time.

Complete-fitting median milliseconds:

| Source photo | CPU1 | CPU14 | GPU depth-first | GPU frontier | Split sequential | Split concurrent |
|---|---:|---:|---:|---:|---:|---:|
| 1029 | 267.80 | 35.34 | 858.88 | 157.40 | 115.14 | 85.97 |
| 1039 | 86.49 | 16.37 | 384.99 | 47.64 | 42.83 | 31.06 |
| 1049 | 108.59 | 17.83 | 407.51 | 56.80 | 52.66 | 41.83 |

All **126 observations**, including warmups, produced exact native trees.
Frontier is 5.46–8.08× faster than this depth-first GPU prototype, but still
2.91–4.45× slower than CPU14 on these batches. This improvement combines ready
node parallelism with threadgroup histogram collection; it does not isolate
the causal contribution of either change.

Concurrent execution reduces the same split assignment's wall time by
20.6–27.5%, but remains 1.90–2.43× slower than putting all fits on CPU14. This
is measured overlap, not evidence for additive peak throughput or a 2× win.
Frontier's process CPU medians are only 2.73–3.35 ms per batch. GPU command
spans account for approximately 93–98% of its complete fitting interval.

Maximum managed backing is 56.35–56.39 MiB for frontier, 25.32–25.35 MiB for
depth-first, and 15.64–16.05 MiB for CPU14. The split-concurrent peak is
53.22–53.26 MiB. These include the route's input-copy/pool backing but exclude
the shared immutable prepared samples and backend creation; they are not RSS.
The plan's conservative frontier bound is about 61.3 MiB. Every run releases
all managed ownership after its result is retired.

## Complete encoding measurements

The matched full-photo comparison uses 4000×3000 RGB8 inputs 1029/1039/1049,
one warmup plus four measured repetitions, alternating
CPU14 and frontier order. Both routes use Metal sample-row generation and
the same automatic feature/token paths; the CPU-fitting control is not an
entirely CPU encoder. Both use the existing cohort schedule. The separate
continuous-pipeline branch is not included in this comparison.

Complete private encode seconds, median [minimum–maximum]:

| Photo | CPU fitting, 14 participants | GPU frontier fitting | Frontier / CPU |
|---|---:|---:|---:|
| 1029 | 7.186 [6.817–7.440] | 11.179 [11.027–11.498] | 1.56× |
| 1039 | 5.161 [4.854–5.450] | 6.920 [6.664–7.159] | 1.34× |
| 1049 | 4.640 [4.342–4.937] | 6.055 [5.887–6.332] | 1.30× |
| Sum of medians | **16.987** | **24.154** | **1.42×** |

Timing includes complete encoder work, including fitting setup, partitioning,
queue management and publication. File I/O, backend/domain creation, independent
decode and hashing are outside the timer. All **30 encodes** have identical
logical learning work and output bytes across variants and match the previous
resident prototype's hashes. Independent libjxl decoding is exact throughout.
Output sizes remain 10,507,251 / 7,138,537 / 7,015,756 bytes.

Each frontier encode fits all 192 nonempty trees on the GPU, with **no CPU
subtrees or admission fallbacks**. The one empty global tree is still handled
by coarse preparation. GPU node totals are 240,142 / 120,546 / 105,820. All runs
reach 128 concurrently active nodes; peak same-tree concurrency ranges from
56 to 112 across these photos. Atomic append order can change the number of
ready rounds slightly without changing any tree or codestream bytes.

| Photo | Complete resident fit median | GPU command-span median | GPU submissions | Nonempty ready rounds |
|---|---:|---:|---:|---:|
| 1029 | 9.975 s | 9.817 s | 121 | 1904–1905 |
| 1039 | 5.801 s | 5.672 s | 62–63 | 962–963 |
| 1049 | 5.055 s | 4.932 s | 56 | 846 |

GPU command spans account for approximately 97.6–98.4% of the fitting interval.
This places the next performance work inside GPU execution; reducing the
remaining CPU command boundaries alone cannot close the observed gap. Command
spans include device scheduling/barriers and are not a per-kernel attribution.
The CPU control's summed worker fitting intervals overlap and are deliberately
not presented as comparable wall-time stages or thread CPU-seconds.

Maximum managed backing in the complete encode is 1,013.72 / 966.03 / 938.58 MiB
for frontier versus 743.34 / 665.98 / 644.37 MiB for CPU fitting. Frontier
therefore retains a material memory cost even with bounded shared scratch.
Complete-encode process CPU medians fall from 60.39 / 41.35 / 36.77 CPU-seconds
to 10.53 / 8.97 / 8.29, while wall time increases. This is useful evidence of
offload, not evidence of a throughput gain.

## Evidence and interpretation

The [machine-readable summary](gpu-frontier-histogram-fitting-results.json)
contains medians/ranges, memory, work counts and hashes of the raw manifests.
Artifacts are under `build/frontier-prototype/`: frozen binaries, tracked
source patches, copies of new source files, source/dependency/input hashes,
all raw observations, validation logs, power samples and before/after process
snapshots. Known competing codec/build jobs were monitored. System services
including Spotlight and media analysis were active; their snapshots are
retained. These are matched session measurements, not an isolated peak-system
claim. No runs were rejected for competing codec/build jobs or power changes.
The 1029 CPU control's excluded warmup was 3.360 s, substantially below its
measured repetitions; it remains in the raw record. No causal explanation for
that drift is established. The prior prototype report's battery numbers and
the separate continuous-pipeline measurements are separate experiments.

The frozen capture commands and wrappers are in `prepared-study/manifest.json`,
`full-study-binary/command.json` and their copied Python runners. For a single
replay or complete encode after building the two private tools:

```sh
build/release/gjxl_modular_resident_fitting_probe INPUT.pam replay.json 16 6 128
build/release/gjxl_modular_hybrid_profile INPUT.pam OUT auto auto best fit-off 14 14 14 14 14 fitter-frontier 16384 64 --sample-rows=metal-unique
```

Use `fitter-cpu` in the second command for the matching CPU-fitting control.

Public/default routing is unchanged. Remaining optimization candidates are
within-node histogram/scoring parallelism, reducing histogram work (including
child subtraction), active-width/chunk tuning and eventual pipeline/lifetime
integration. This implementation establishes exact GPU frontier growth; a
complete-encode speedup remains an independent requirement.
