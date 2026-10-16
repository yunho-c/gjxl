# Batched GPU histogram fitting

Status: the operation/storage contract and resumable CPU reference are
implemented and tested. CPU
consolidation is committed as `3aca01c`; its paired performance qualification
is recorded in [the baseline report](gpu-histogram-fitting-baseline.md).
The Metal evaluator and its connection to the public learner are subsequent
steps. The public encoder continues to use the CPU histogram fitter.

This effort targets the native quantized, weighted histogram learner. The older
`GpuModularTreeStatistics` operation evaluates sorted rows using integer Q12
costs; it is not the scoring contract used here. The CPU oracle is
`EvaluateProperty` in `histogram_tree_learning.cpp`, including its four-lane
floating-point entropy estimate and ordered candidate comparisons.

## Consolidated starting point

The starting implementation combines continuous local fitting queues and
bounded feature cohorts (`51dbbe0`) with the committed `feat/modular` changes
through `9d8a011`:

- Fused occupancy/histogram scans and inherited constant-property masks.
- Independent CPU property evaluation inside shared/global fits, with ordered
  reduction of property results and no nested property teams in local fits.
- Bounded ANS histogram-search storage reuse.
- Reuse of the two training-row hash positions between lookup and insertion.

Sampling, quantization, predictor choices, split preferences and codestream
policy remain unchanged. Consolidation must retain local queue traces, shared
property diagnostics, memory-pressure behavior, partial-launch cleanup and
exact output bytes. Shared-fit property timers overlap and cannot be added as
exclusive subdivisions of elapsed fitting time.

The fresh consolidated capture finds histogram construction, occupancy and
candidate evaluation responsible for 78–83% of summed diagnostic fitting time
on 1029/1039/1049. Training including preparation occupies 65.6% of complete CPU
encoder elapsed time and 70.3% with explicit Metal. These are distinct timing
boundaries: summed fitting work is not caller elapsed time. Small nodes remain
substantial: nodes with at most 1,024 rows account for approximately 25–43%
of diagnostic node time. Batched evaluation and a measured CPU crossover are
required; accelerating only the largest roots is insufficient.

## Operation boundary

The private operation has two synchronous calls:

1. Prepare immutable quantized sample columns for a bounded cohort of fits.
2. Evaluate a batch containing one ready node from each participating fit.

Each source retains its canonical stream ID, predictor order, property order,
bin counts, positive uint16 weights, residual symbol/extra-bit columns and
static/dynamic property columns. Static columns retain uint32 bins; dynamic
columns retain uint8 bins. Every native predictor and property is represented.
Static bins may exceed 256; eligibility depends on the admitted workspace,
not on silently changing quantization.

Preparation copies the finalized CPU sample columns once. The CPU keeps its
existing mutable columns and tree state. A separately accounted uint32 row-order
vector initially contains the identity permutation; every CPU partition swaps
its entries alongside the corresponding CPU rows. A node dispatch uploads only
the selected original-row indices and compact node/parent metadata. It does not
re-upload all prediction/property columns after each split.

The prepared GPU columns are immutable even as CPU columns are partitioned.
Borrowed row lists and parent summaries remain stable through the synchronous
evaluation. No CPU partition may mutate them until that call has completed.
An implementation may later use a shared resident index buffer, but that needs
an explicit ownership and synchronization extension rather than an implicit
alias of mutable CPU storage.

The first integration exposes one ready node per fit. This retains each tree's
existing depth-first growth and node numbering while batching independent trees.
Requests use ascending prepared-fit indices. Forced multiplier splits, nodes
already known not to split, tree capacity/depth checks and final publication
remain CPU decisions. A node not eligible for the GPU uses the CPU evaluator
before any GPU execution starts.

## Exact evaluation semantics

Inputs include the node's original-row membership, padded symbol span, parent
histograms/extra-bit sums, parent predictor, exact predictor-change penalty and
inherited constant-property mask. Parent summaries come from the existing CPU
node setup. The host packer evaluates the native predictor-bias expressions and
uploads their final float bits. In particular, the CPU's `1e-8` double literals
must not silently become GPU float additions/subtractions. The operation returns
one result per node/property in canonical
order, with a constant flag and four category winners:

1. Static property with a zero-entropy side.
2. Other static-property candidate.
3. Nonstatic candidate using property 15 or a weighted side predictor.
4. Nonstatic candidate without weighted prediction.

Candidate side counts are counts of distinct row entries, not weighted sample
populations. Histograms and extra-bit sums use row weights. Candidate thresholds
are occupied property bins excluding the last occupied bin. Inherited constants
and single-bin properties skip the histogram/candidate work; newly discovered
constants are reported so their descendants can skip it too.

The following must reproduce the native evaluator exactly:

- Integer histogram counts, distinct-row occupancy and weighted extra bits.
- Padded symbol span and four-lane accumulation/reduction order.
- The native `FastLog2` polynomial, its explicit fused operations, division,
  probability clamp and separate multiply/subtract operations in `EstimateBits`.
- Side-cost expression order, including converting and adding the parent extra
  bits before subtracting the lower-side extra bits. Computing the integer
  difference first can round differently.
- Predictor order, separate raw/penalty costs, weighted/zero predictor biases,
  strict comparisons, bin order and four-category classification.
- Canonical property reduction followed by the CPU's existing category
  preferences, split threshold, unquantization and partition operation.

Penalties affect each side's predictor choice. Exported side costs and the
subsequent category/split comparisons use raw costs, as in the CPU evaluator.

Generic GPU `log2`, unordered floating-point reductions and relaxed fast-math
are not interchangeable with this contract. Exact candidate-cost bits and
winners must be checked against the CPU oracle before integrating tree growth.
If Metal cannot reproduce the arithmetic for a supported case, that is an
unresolved implementation issue, not permission to change the learned policy.

## Storage, ownership and eligibility

The operation uses explicit limits and allocation-free, failure-atomic plans.
Initial cohort admission is at most 256 nonempty fits and 8 Mi distinct rows.
Each fit retains the native population bound, up to 14 predictors, 16 properties,
two leading static properties, 256 dynamic bins and 65,536 static bins. These
are representation/admission bounds, not a truncation of learning samples.

Preparation has a retained device arena for descriptors and aligned columns,
retained host shape metadata, and a conservative fresh host staging arena.
The planner also reports the separate host row-order bound required by the
future CPU integration. CPU sample columns, trees, feature owners and any old
output remain separately charged.

Evaluation uses a dispatch arena for node descriptors, row indices, parent
summaries and exact CPU-computed predictor penalties; a completed result arena;
bounded reusable property scratch; and a
conservative dispatch staging copy. Property scratch includes histogram shards,
reduced/prefix histograms, occupancy, extra-bit sums, candidate side costs and
ordered side winners. The number of concurrently resident property tasks is
limited by both an explicit task cap and the scratch-byte allowance. Oversized
nodes are routed to CPU or a separate admissible batch before execution.

All device-region offsets, buffer limits and arithmetic are checked. Planners
describe new backing; the workflow composes the live prepared owner, row-order
storage, CPU fitter state, feature owners and old outputs exactly once. A
completed result exposes immutable host-readable views without an extra copy
or a second accounting charge.

Malformed shapes, summaries and ownership return `InvalidArgument`. Valid work
outside device/cohort/storage limits returns `Unsupported`. Runtime device
status checks cover row indices, weighted population, parent-summary mismatch
and arithmetic invariants. Both calls consume ambient admission, publish only
after success, drain submitted work before releasing any input/scratch on
failure, and do not silently retry on CPU after execution has started.

## Implemented contract

The implementation is private to the repository:

- [`modular_histogram_fitting.h`](../src/gpu/ops/modular_histogram_fitting.h)
  defines the optional backend capability, prepared-row/result owners, borrowed
  request views and storage/eligibility helpers.
- [`modular_histogram_fitting_layout.h`](../src/gpu/ops/modular_histogram_fitting_layout.h)
  defines fixed-width C++/Metal descriptors and results. A fit descriptor is
  336 bytes, a node descriptor 48 bytes and a property result 144 bytes.
- [`modular_histogram_fitting.cpp`](../src/gpu/ops/modular_histogram_fitting.cpp)
  validates native sources and parent summaries, computes exact predictor
  penalties, and plans all operation backing without allocating or accessing
  a GPU. Regions start at 256-byte boundaries; device offsets and buffers must
  fit uint32 and the backend's reported limits.
- [`modular_histogram_fitting_plan_test.cpp`](../tests/modular_histogram_fitting_plan_test.cpp)
  checks manually derived layouts/lifetime peaks, native weighted samples,
  cohort/device limits, all-constant and wide-static cases, malformed metadata,
  failure-atomic outputs and exact penalty bits.

The default storage options request four histogram shards, at most 64 active
property tasks and 64 MiB scratch. These are explicit workspace choices, not
automatic GPU crossover thresholds. Every requested property still gets an
output when reduced scratch permits only one active task. A property too wide
for one scratch slot returns `Unsupported` before submission. The shape tests
exercise all 65,536 static bins under a larger explicit allowance without
allocating that arena.

`ValidateHistogramFittingInput` checks host metadata and parent summaries; it
deliberately does not scan membership indices. The backend must verify its
native owner type and allocation bounds, then check row indices, population
and agreement with parent summaries on device before publishing a result.
This is a requirement for the next implementation, not device validation
already supplied by this slice. Inherited constant masks are trusted CPU-parent
state after their bit ranges have been checked.

`ComputeHistogramFittingWorkingSetBound` composes the prepared device/metadata
owner, separate CPU row-order vector and evaluation output with the larger of
the preparation and evaluation peaks. The caller must still add borrowed CPU
samples, trees, retained features and any prior output. Neither a second host
copy of completed results nor a second charge for the same result backing is
required.

The new test passes in Release and a CPU-only C++23 AddressSanitizer/UBSan
build (`ASAN_OPTIONS=detect_leaks=0`, because macOS LeakSanitizer is unsupported).
Contract validation records and source/binary hashes are archived with the
baseline's focused integration checks. No encoder route queries the capability
yet. Actual Metal allocation/error paths and GPU candidate/tree parity remain
later gates.

## Resumable CPU reference

`HistogramTreeFitter` in `histogram_tree_fitter.h` owns one depth-first tree,
preallocated CPU scratch and an original-row permutation. `Advance` performs
native node setup and forced splits until the next independent evaluation;
`EvaluateCpu` exports the four category winners; `Apply` validates all results
before native selection/partitioning; `Publish` validates and moves a complete
tree. Ordinary and resumable fitting share the same setup, property evaluation
and split-application functions. The private GPU bridge exports/imports the
fixed-width results without allocating.

Initialization allocates every tree, workspace and row-order backing before
partitioning starts. Failed replacement initialization preserves the old fit;
malformed evaluation results preserve the ready state. An execution/capacity
failure prevents publication. Prepared samples and multiplier views must stay
at stable addresses until the fit is published or destroyed. CPU evaluation
uses one participant, leaving independent-fit parallelism to its caller.

The resumable storage bound includes the row-order vector. A workflow composing
it with the GPU operation must not charge that same vector again. Each active
fit needs a separate diagnostic profile; fit/node envelopes include time spent
paused, whereas the named CPU phases record work inside those phases. They
must not be presented as exclusive subdivisions of batch elapsed time.

Step 2 validation on 2026-10-16 passed five focused Release tests and the two
new contract/fitter tests under ASan/UBSan. The pinned-libjxl harness passed
257 generated differential cases through both ordinary and resumable fitting.
Additional coverage includes exact candidate cost bits, complete trees and
partition order, original-row associations, interleaved fits, empty/forced
nodes, inherited constants, wide static properties, the 2^24 population
ceiling, allocation-failure positions and failure-atomic publication. No
managed allocations occur after successful initialization.

The CPU-overhead replay used seven genuine captured fits: Macan Best/All,
alpha16-ramp All, gray-patches Best, flower reference local fits 0/48, and
noise16 All. With one warmup per mode and five alternating measured pairs on
M4 Pro/AC, the sum of per-fit medians changes by **-0.11%**; individual changes
range from **-1.60% to +0.56%**. All trees match the captured reference, and
measured managed peaks match the resumable bounds exactly. This compares
ordinary and resumable fitting in the same refactored CPU-only Release binary;
it does not measure the refactor's end-to-end effect against the frozen encoder
baseline. Timings include validation, initialization, evaluation, partitioning
and publication, and exclude native sample conversion and reference-tree
comparison. The first timing pass overlapped another qualification and was
discarded; the replacement monitored competing encoder/compiler processes
throughout each measured invocation.

Raw timings, fixture/source/binary hashes and test evidence are in
[`gpu-histogram-fitting-cpu-reference-results.json`](gpu-histogram-fitting-cpu-reference-results.json).
Local build/log archives are under
`build/gpu-histogram-fitting-implementation-20261016/`; the research harness's
`resumable-paired-replay FIXTURE REPEATS` mode reproduces the comparison.

## Implementation sequence and gates

1. **Consolidation and contract.** Merge the latest CPU work, resolve diagnostic
   semantics, run focused correctness/sanitizer checks and capture paired public
   CPU/Metal measurements. Define the private types, fixed-width layouts,
   eligibility validation, storage plans and CPU-only contract tests.
2. **CPU evaluator and resumable tree state.** Extract the existing property
   evaluator behind this contract, compare exact candidates and complete trees,
   and expose node-ready state across independent fits. Preserve CPU sample
   order and admit row-order/cohort state for its full lifetime. Qualify overhead
   with GPU dispatch disabled before adding kernels.
3. **Metal evaluator.** Implement exact integer histogram/occupancy/extra-bit
   construction, prefix work and candidate scoring using bounded property
   scratch. Batch ready nodes across fits. Compare every result with the CPU
   oracle on weighted rows, sparse bins, constants, ties, Best/All predictors,
   static properties and extreme supported populations.
4. **Workflow and performance qualification.** Connect the optional evaluator,
   test failure atomicity and concurrent resource use, and establish measured
   CPU fallback thresholds. Require exact trees, complete compressed bytes and
   independent decoding. Measure complete calls against the updated
   14-participant CPU baseline, including preparation, upload, dispatch, waits,
   CPU partitioning, fallback and teardown. Include shared/small/16-bit cases
   alongside the three 12 MP photographs. No public default change follows
   merely from a kernel-only speedup.
