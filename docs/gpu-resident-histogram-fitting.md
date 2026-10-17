# GPU-resident histogram fitting prototype

The first milestone now moves complete fitting onto the GPU, with exact tree and
byte parity in the checks below. Its first depth-first executor is substantially
slower than the parallel CPU fitter. Frontier expansion is the next milestone.
The working branch is
`feat/gpu-resident-fitting`, based on `7f66a10`; it preserves capacity reuse
(`6343e50`) and its qualification. The independent pipeline branch at `efc0335`
is left untouched. Its [report](../../gjxl-modular-cohort-pipeline/docs/modular-pipeline-gpu.md)
and hashed timing/diagnostic manifests were read before implementation.

## Operation and first executor

`FitHistogramTrees` accepts immutable prepared host sample sets and per-fit native
learning options. It validates and uploads once, then submits one persistent
128-thread workgroup per fit. Within each workgroup, GPU state drives depth-first
growth: node initialization, symbol ranges, weighted parent histograms, property
histograms and costs, final category/predictor selection, partitioning, and the
pending-node stack. All subtrees remain on the GPU, including small tails.
CPU work is coarse validation/admission, immutable configuration, submission,
failure handling and final tree validation/publication.

Node records retain their own row ranges, depth and inherited constants. The
DFS stack stores indices into those records; a subsequent frontier executor can
schedule the same node state without changing its ownership model. This first
executor does not claim efficient within-tree frontier parallelism.

The first operation supports up to 64 independent fits, 131,072 distinct rows
per fit, at most 256 bins per property, and native predictor/property subsets.
Multiplier ranges, empty sample uploads and CPU per-node execution statistics
are explicitly unsupported in this version. Inputs outside the prototype
contract are rejected before submission. Tree capacity and depth limits fail
the entire operation instead of truncating decisions. The private workflow
handles empty sample sets before calling the operation.

The private `fitter-resident` route prepares ordinary coarse cohorts and calls
the operation in batches of at most 64 nonempty fits. It skips CPU fitter
initialization and retains no CPU subtrees. It reports complete resident-call
time and GPU execution separately; the old per-node fitting profiles and
evaluator controls are rejected on this route. Its diagnostics explicitly
identify the fresh arena and single encoder. Public routing and the existing capacity-reuse defaults
are unchanged. The continuous pipeline integration remains on its separate
branch and was not merged as a prerequisite.

## Exactness and storage

Histogram integer sums, native entropy arithmetic, predictor order and strict
comparison ties are preserved. The operation uses the existing Metal entropy
functions and strict floating-point build flags. A small predictor-penalty table
is prepared once on the CPU to retain the native double-literal rounding; GPU
node state selects the applicable row. No parent summary or node decision is
computed on the CPU.

Partitioning collects ascending misplaced row indices on either side of the
chosen cut, then swaps corresponding pairs in parallel. This preserves the
native two-pointer permutation, not only the resulting sets. Immutable sample
columns remain unchanged. GPU node IDs use the native sibling allocation and
high-side-first DFS order, so this milestone needs no renumbering.

The admission plan includes immutable GPU rows, per-fit device nodes, DFS stack,
row order and partition indices, parent/property scratch, thresholds, policy
tables and all CPU result backing. Scratch is reused within a fit. Caller sample
columns and old output are excluded and remain caller-owned. No unbounded device
allocation occurs during growth. All trees publish together only after command
completion, device-status checks and native host tree validation. Optional row
order readback is a diagnostic cost and is disabled for normal publication.
The work arena is fresh per operation; within-fit scratch is reused, and there
are no allocations per node. The first workflow integration still reserves the
old conservative CPU preparation/fitting job envelope alongside the separate
resident operation bound. Removing unused reservation headroom and adopting a
bounded resident arena cache belong to later admission/lifetime integration.

## Qualification and measurement boundary

The Release checks cover 24 complete trees and their exact final row
permutations against both ordinary and resumable native CPU fitters. Cases
include 1, 2 and 14 predictors, reordered predictor/property preferences,
constant properties, 256 static bins, zero split threshold, and the maximum
weighted population with 63 extra bits. Eleven failure checks cover allocation,
submission, completion, malformed/unsupported inputs, tree/depth capacity and
admission; previous output remains intact. Recovery, concurrent calls on one
backend, immutable inputs and exact-bound backing cleanup also pass.

Three private workflow fixtures cover gray8, RGB8 and RGBA16, both predictor
policies, Metal features, and Metal unique sample rows. They compare native
trees and complete bytes and independently decode through pinned libjxl.
The focused eight-test Release regression set passes, including the old GPU
evaluator, fitter queue and resource-budget checks. The new operation also
passes Metal API validation. Both new tests pass ASan/UBSan with leak detection
disabled and a narrow `null:*third_party/metal-cpp/*NSObject.hpp` suppression for
the vendor's nil-receiver wrapper. The unsuppressed failure is retained. GJXL
host code remains instrumented; this does not instrument shader execution or
the pinned libjxl binary.

Full-resolution correctness smokes on all three 4000×3000 photographs also
decode exactly and match the previously qualified codestream hashes. Each
completes all 192 nonempty fits on the GPU, in five resident submissions, with
zero CPU fitter nodes or admission fallbacks. That covers **240,142 / 120,546 /
105,820 nodes** for 1029 / 1039 / 1049. Each image also has one empty coarse fit
handled before dispatch. These are correctness runs, not a repeated or matched
end-to-end performance comparison.

The measurement replays **identical prepared samples**, rather than timing
just the split-cost kernel. For each 4000×3000 RGB8 photograph (1029, 1039 and
1049), it prepares 16 deterministic single-channel 128×128 tiles using the
default Best policy. Each batch has 171,424 sample rows; resulting total node
counts are 4,426 / 1,372 / 1,432 respectively. These are tile-derived fitting
inputs, not all image pixels or the transform-selected production fit set.

The measured boundary includes validation, workspace setup, parent summaries,
partitioning, node management, complete tree construction and publication.
GPU upload, command submission and wait are included. Sample preparation,
backend/domain creation and persistent CPU worker-pool creation are excluded.
The CPU learner mutates its input, so fresh sample restoration occurs before
each observation; its time is recorded separately and its backing is counted.
Shared immutable source samples and expected oracle trees are outside both
timing and reported operation backing. Output verification follows the timer.

Measurements use the M4 Pro, 48 GB, on **battery**, with the recorded battery
setting `powermode 1`. They are separate from the earlier AC pipeline study.
Each mode has one excluded warmup and five measured repeats, with cyclic order
balanced across the five measured rounds. All 90 batch observations passed
tree equality. No overlapping codec/build jobs or power-source changes were
detected; there were no rejected runs. Sources, binary, dependencies and inputs
were hashed, with the frozen executable and source patch retained.

| Prepared batch | CPU, 1 worker | CPU, 14 workers | Resident GPU | GPU / CPU14 |
|---|---:|---:|---:|---:|
| 1029 | 377.20 ms | 53.85 ms | 690.03 ms | 12.81× |
| 1039 | 129.32 ms | 24.24 ms | 331.83 ms | 13.69× |
| 1049 | 157.63 ms | 25.53 ms | 346.15 ms | 13.56× |

Entries are complete-fit wall medians. CPU14 ranges were 49.65–57.54,
23.15–26.35 and 24.95–27.19 ms; resident ranges were 686.76–692.26,
331.69–332.78 and 345.67–347.07 ms. This executor establishes residency and
exactness; it demonstrates no throughput win. GPU command execution accounts
for 99.4–99.6% of the resident full-call medians. Reducing CPU handoffs further
cannot explain away this gap.

The executor exposes one workgroup per fit and loops over properties/predictors
within that workgroup. In these batches that permits only 16 workgroups, with
one active node per tree. That is a source-level parallelism limit; this study
does not separately attribute GPU time to atomics, scans, occupancy or cost
arithmetic. Frontier execution should expose many node/property tasks together
and reuse the existing tiled collection/evaluation design.

For concurrency, the exact same 16 fits are assigned as eight GPU fits plus
eight CPU fits. The CPU uses 13 participants plus one GPU caller within a shared
14-participant execution domain. The assignment is identical between the
sequential and concurrent controls, and thread launch/join is included.

| Prepared batch | Split sequential | Split concurrent | Wall reduction |
|---|---:|---:|---:|
| 1029 | 728.68 ms | 685.11 ms | 6.0% |
| 1039 | 351.17 ms | 331.21 ms | 5.7% |
| 1049 | 360.04 ms | 344.51 ms | 4.3% |

CPU work overlaps, but the GPU determines completion. The split route remains
much slower than CPU14; these results establish no additive throughput benefit
or 2× speedup. Process CPU time is recorded using `std::clock`, independently of
wall time. Summed worker elapsed intervals are not presented as CPU-seconds.

Measured managed-backing peaks for CPU14 were 16.05 / 15.27 / 15.65 MiB, versus
25.35 / 25.32 / 25.32 MiB for resident fitting. Concurrent split peaks were
21.09 / 21.05 / 21.06 MiB, higher than the roughly 15 MiB sequential split peaks
because the workspaces overlap. These are operation-accounted backing peaks,
including CPU input restoration where applicable, not process RSS or whole
encoder memory. Raw records include baseline, incremental peak and the
conservative resident storage plan separately.

## Reproduction and evidence

The tool target is `gjxl_modular_resident_fitting_probe`. Its CLI is:

```sh
build/release/gjxl_modular_resident_fitting_probe INPUT.pam OUTPUT.json 16 5 128
```

It refuses to overwrite results, prepares a fixed oracle outside timing, runs
the five controls, and verifies every completed tree. The private full-encode
route is:

```sh
build/release/gjxl_modular_hybrid_profile INPUT.pam OUTPUT_DIR \
  auto auto best fit-off 14 14 14 14 14 fitter-resident 16384 64 \
  --sample-rows=metal-unique
```

The [results](gpu-resident-histogram-fitting-results.json) retain every fitting
observation, dimensions, sample locations, ranges, memory, commands and artifact
hashes. Local raw evidence is under `build/resident-prototype/`: Release and
sanitizer logs, Metal API validation, `prepared-study/manifest.json`, frozen
binaries, the source patch and new source files. The prepared-study manifest
also records power/activity snapshots and dependency hashes.
Full-image byte/decode checks are retained in
`full-photo-smokes/manifest.json`. The final profile-tool diagnostic cleanup
does not change the fitter, shader or prepared-sample probe; the measurement
source hashes and frozen tools remain available in the manifests.

## Next milestone: frontier fitting

1. Replace the DFS stack executor with bounded GPU current/next node queues.
   Keep node-owned ranges, inherited constants and predictors. Compute node
   summaries and enqueue children on device, without per-node CPU decisions.
2. Dispatch collection and evaluation over many active nodes and properties,
   including tiled work within large nodes. Reuse the exact entropy arithmetic
   and integer reductions, then select winners in the native property/bin and
   predictor order on GPU. Use bounded scratch waves and device-generated
   dispatch counts; do not make large frontiers require unbounded storage.
3. Preserve exact partition membership/permutation, tree limits and all-or-none
   publication. Canonicalize completed nodes into the native high-side-first
   sibling allocation order before publishing, since frontier creation order
   will differ. Compare complete trees and bytes with this prototype and CPU.
4. Repeat the identical-sample CPU1/CPU14/resident/concurrent comparison,
   including all frontier and canonicalization costs and memory. Then qualify
   complete encodes under matched power and input/policy conditions before
   changing any automatic routing.

The existing pipeline's readiness/CPU-tail trace remains a bounded later
integration aid. It does not gate this prototype. Input readiness, coarse
admission, feature/token lifetimes and any future CPU fallback still need their
own integration qualification after the resident operation is established.
