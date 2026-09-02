# Amdahl leakage after entropy-behavior alignment

- Status: analysis and implementation roadmap
- GJXL revision: `af3a9e6`
- Reference libjxl revision: `e8ff09762481785938d8e4e01333ed3917571161`
- Profile date: 2026-09-02
- Related analysis: [entropy behavior alignment](entropy-behavior-alignment.md)

## Executive finding

After GJXL stopped running its former exhaustive entropy/codestream tournament
at ordinary efforts, the remaining effort-7 gap to libjxl was no longer caused
by one dominant serializer search. It was distributed across host preparation,
full-image validation and copying, chromaticity statistics, GPU-to-host frame
handoff, coefficient-token preparation, and ANS histogram construction.

The main Metal algorithms are not the evidence-backed problem:

- AC-strategy candidate arithmetic runs efficiently on Metal. The host still
  performs the exact non-overlap merge and deterministic tie-breaking, but the
  measured combined GJXL path was not slower than libjxl's CPU AC search.
- Initial adaptive quantization is fast on Metal.
- GJXL's resident Butteraugli-controlled AQ is much faster than libjxl's
  effort-8-and-higher CPU quantizer search when those iterative policies are
  compared directly.
- Final rANS/model emission and final codestream framing are already faster
  than the corresponding instrumented libjxl phases.

The problem is Amdahl leakage around those kernels: work that remains serial or
host-resident becomes the end-to-end limit even when the central image kernels
are accelerated. The current effort-7 profile also performs two AQ updates,
unlike libjxl effort 7. Changing effort 7 to a zero-update policy is priority 1,
but that work is intentionally outside this document's implementation scope.

The remaining priorities are:

| Priority | Area | Current 4K signal | First implementation target |
| ---: | --- | ---: | --- |
| 2 | Preparation and storage | 61.5 ms complete input preparation; heavy allocation and clearing | Do not allocate outputs unused by the selected path |
| 3 | Color conversion, validation, and copies | about 10.2 ms GJXL versus 2.2 ms libjxl color-transform attribution | Validate once and transform directly into final storage |
| 4 | Quantization-matrix chromaticity statistics | about 16.8 ms GJXL versus 5.9 ms libjxl attribution | Apply libjxl's effort gate and vectorize the retained pass |
| 5 | Metal readback and frame assembly | about 10.2 ms readback plus 12.9 ms assembly | Remove intermediate copies and redundant full-frame scans |
| 6 | Residual serializer | 62.6 ms GJXL versus 42.3 ms libjxl | Give one-representation encoding a one-pass token path |

These signals are not additive estimates. The sampled function values include
parallel aggregate CPU, several operations are nested, and allocator/memory
traffic appears beneath multiple stages. Each retained change must therefore be
qualified with an alternating end-to-end wall-clock comparison.

## Measurement snapshot and interpretation

The current matched-quality comparison used canonical PFM inputs, production
thread policies, three alternating independent process pairs, two warmups, and
five measured samples per process. The table reports the median of the process
medians:

| Input | Encoder | Complete encode | Serializer | Coefficient tokenization | Entropy construction | Model/token emission |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1080p photo | GJXL | 141.610 ms | 27.209 ms | 7.207 ms | 14.314 ms | 4.674 ms |
| 1080p photo | libjxl effort 7 | 103.004 ms | 13.763 ms | 0.612 ms | 6.548 ms | 6.469 ms |
| 4K photo | GJXL | 494.201 ms | 62.562 ms | 21.931 ms | 26.910 ms | 8.856 ms |
| 4K photo | libjxl effort 7 | 410.754 ms | 42.318 ms | 2.197 ms | 18.001 ms | 21.275 ms |

At 4K, GJXL spent approximately 61.5 ms in input preparation, 369.6 ms in the
quantization pipeline, and 62.6 ms in codestream serialization. The total gap
was 83.4 ms. The serializer accounted for 20.2 ms of it; the remaining 63.2 ms
was outside the serializer.

Three timing distinctions are essential:

1. Complete encode and explicitly instrumented stage durations are wall time.
2. Samply thread-CPU deltas and GJXL `_work` counters are aggregate worker time.
   They can exceed the containing wall stage and cannot be summed into an
   end-to-end latency claim.
3. The GJXL and libjxl coefficient-tokenization labels are not identical.
   GJXL's phase includes coefficient-order derivation, while libjxl computes
   its orders immediately before entering its tokenization timer. The raw
   9.98x 4K ratio is therefore not a pure token-loop ratio.

The current sampled capture nevertheless supplies useful causal evidence. Per
4K encode, it attributes about 42 ms of aggregate CPU to `__bzero` and 29 ms to
`memmove`. GJXL's own work counters attribute about 14.5 ms to coefficient-order
derivation, 20.9 ms to AC template construction, 4.1 ms to context
materialization, and 19.3 ms to direct-ANS histogram construction. These values
overlap in wall time, but they identify actual repeated memory passes.

## Priority 2: prepare only the selected execution path

### Current structure

[`PreparedWorkflow`](../src/codestream/workflow.cpp) eagerly owns:

- padded linear RGB;
- padded Opsin;
- a [`PipelineStorage`](../src/codestream/workflow.cpp) containing initial
  quantization, strategy and pixel masks, final quantization, block distances,
  reconstructed RGB, and the encoder frame;
- a [`PreparedQuantizationPipeline`](../src/codec/quantization_pipeline.cpp);
  and
- prepared GPU adaptive-quantization state.

`PrepareQuantizationPipeline()` then creates another owned coding-Opsin image,
copies the workflow's Opsin into it, and allocates another collection of initial
and final quantization outputs, masks, block distances, and reconstructed RGB.

The encoding-only fully-resident call explicitly disables materialization of
initial quantization, final quantization, block distance, and reconstructed RGB
through `QuantizationPipelineMaterialization`. By then, however, much of the
corresponding host storage has already been allocated and value-initialized.

The production Metal backend is not the missing cache: it is already initialized
once and reused process-wide by `ResolveProductionMetalBackend()`. Likewise,
[`DeviceScratchArena`](../src/gpu/scratch.cpp) already reuses an allocation when
its retained capacity is sufficient. The current lifetime of the prepared
workflow prevents those evaluator arenas and most host-vector capacities from
surviving an independent encode.

### Recommended first slice: lazy materialization

Introduce an internal execution/materialization plan after geometry validation
and per-attempt backend selection. It should distinguish at least:

- CPU pipeline;
- Metal exact coefficients;
- Metal fully resident or throughput;
- maximum-error control;
- final Butteraugli diagnostics; and
- public diagnostic APIs that promise complete output planes.

Use that plan to allocate only required output storage. For an ordinary
fully-resident encode, `PipelineStorage` should retain the final frame, score
history, and control result, but not allocate diagnostic quant fields, a block
distance map, or reconstructed RGB. `PreparedQuantizationPipeline` should make
its final-output planes lazy as well. Some initial fields may remain necessary
for control or reconfiguration; their removal must be driven by actual caller
use rather than by the final materialization flags alone.

Avoid the complete `opsin -> coding_opsin` copy by letting the internal prepared
pipeline borrow the workflow-owned immutable Opsin view for the synchronous
encode lifetime. A more complete version can remove `padded_linear`: transform
the real source extent into a padded Opsin destination, then extend the
already-transformed right and bottom edges. Because the transform is pointwise,
that ordering should preserve padded values while eliminating one owned
three-plane float image.

Backend selection is per attempt and automatic target-size searches can vary
the attempted Butteraugli target, so lazy storage must still support a later CPU
or Metal choice. The goal is not to assume that one backend applies to every
attempt; it is to materialize each attempt's actual requirements on demand.

### Recommended second slice: persistent workspace leases

After dead allocations are removed, preserve useful capacity across independent
images with an internal `VarDctEncoderWorkspace` or `MetalEncodingWorkspace`.
Do not cache an entire `PreparedWorkflow` unchanged.

The C API documents that `gjxl_encode` may run concurrently on one context.
Consequently, one context-wide mutable workspace would either race or serialize
all calls. A safe design is a leaseable pool:

- acquire and return a workspace under a short lock;
- keep all encode and GPU work outside that lock;
- give each batch worker one natural workspace lease;
- key or reconfigure workspaces by backend and required capacity;
- discard poisoned state after a device, upload, or completion error; and
- enforce a high-water memory policy so a single large image is not retained
  forever by every worker.

Prepared AQ currently uses backend, view identity, and options to decide whether
an evaluation is compatible. Reusing the same host allocation for a different
image can preserve its pointer and geometry while changing its contents. A
persistent workspace therefore requires an explicit image generation/reset
operation. Pointer identity alone must not allow a new image to reuse the old
device pixels or Butteraugli reference.

### Expected result and gates

The whole 4K preparation phase is about 61.5 ms, so that is the absolute ceiling,
not an achievable saving. The allocator profile makes a 15-30 ms first-pass
engineering target plausible across priorities 2 and 3, but only a measured A/B
can turn that into a claim.

Required coverage includes automatic CPU fallback, forced CPU and Metal,
target-size repeated attempts, maximum-error, final-score diagnostics, concurrent
C API calls, batch workers, allocation failures, and device-error invalidation.

## Priority 3: validate and move image pixels once

### Current repeated passes

[`EdgeExtend()`](../src/codestream/workflow.cpp) checks every input sample for
finiteness while copying it, and performs `min()` coordinate clamping inside the
entire padded destination loop.

[`LinearRgbToOpsin()`](../src/codec/color_transform.cpp) then:

1. allocates a full three-plane temporary result;
2. scans each input row for finite samples;
3. performs the Opsin transform;
4. scans all transformed samples for finiteness; and
5. copies the complete temporary result into the destination.

Quantization-matrix statistics scan Opsin again. Metal evaluator preparation
also validates its original image before upload. These checks are individually
defensible at public API boundaries but redundant inside one already-validated
synchronous workflow.

### Implementation

Add an internal checked/trusted path rather than weakening the standalone public
color-transform contract:

- validate source values during their first copy or transform;
- write transformed values directly into workflow-owned Opsin storage;
- combine output-finite detection with the transform loop or its vector
  reduction;
- bulk-copy the real part of each source row and fill only the padded right
  edge, then duplicate only the final row into bottom padding; and
- carry an internal finite-validation precondition into later consumers that
  would otherwise rescan the same storage.

Finite input alone does not prove finite output. The public linear-float API can
receive very large finite values that overflow during matrix arithmetic, so the
output check must be fused, not silently deleted. The public
`LinearRgbToOpsin()` function should also retain failure atomicity: a failed
standalone call must not leave a partially transformed destination.

The color transform currently creates fresh `std::thread` workers for each
parallel invocation. That overhead is lower priority. Earlier shared-executor
experiments were not unconditionally faster, so full-image allocations and
passes should be removed before scheduling is redesigned.

### Expected result and gates

The current cross-profile attributes about 10.2 ms per 4K encode to GJXL's
color transform versus about 2.2 ms to libjxl's corresponding operation. A
5-8 ms opportunity is plausible, with additional allocator benefit overlapping
priority 2.

Tests must cover NaN, infinity, huge finite values, strided inputs, overlapping
views where allowed, unchanged public output on failure, scalar/NEON parity, and
complete codestream hashes. Any numerical approximation change also requires
decoded Butteraugli comparison; direct-write and pass-fusion changes should aim
for exact bytes.

## Priority 4: align and accelerate chromaticity statistics

### The heuristic is grounded

GJXL's quantization-matrix scale statistics are not an ungrounded local search.
Pinned libjxl's
[`PixelStatsForChromacityAdjustment`](../third_party/libjxl/lib/jxl/enc_frame.cc)
computes the same maximum X edge, B-minus-Y edge, and exposed-blue statistic and
uses the same decision thresholds.

The behavioral difference is when and how the pass is run. Libjxl skips the
pixel scan at tiers faster than effort 7 and returns before it in maximum-error
mode. GJXL currently computes the statistics during `PrepareWorkflow()` before
`SelectQuantizationMatrixScales()` returns default scales for maximum-error.

### Implementation

Match libjxl's control policy:

- efforts 1-6: use the distance/default scale path without pixel statistics;
- efforts 7-10: retain pixel statistics; and
- maximum-error: skip the pass entirely.

For the retained effort-7 and effort-9 path:

- hoist current and previous row pointers outside the pixel loop;
- remove per-sample finite checks only when priority 3 supplies a trusted
  validated Opsin view;
- vectorize horizontal and vertical X differences, B-minus-Y differences, and
  exposed-blue products; and
- if necessary, use independent row stripes that each include one prior row and
  produce three local maxima for deterministic final reduction.

Do not initially fuse this pass into Opsin conversion. Vertical-neighbor state
and parallel row scheduling make fusion harder to qualify, while a standalone
SIMD maximum reduction is comparatively contained.

### Expected result and gates

The current sampled attribution is about 16.8 ms for GJXL versus 5.9 ms for
libjxl at 4K, suggesting roughly 8-11 ms of effort-7 optimization headroom. The
full pass should disappear at low efforts and in maximum-error mode.

The thresholds `0.015`, `0.022`, `0.026`, `0.28`, `0.33`, `0.38`, and `0.13`
directly affect the frame header. Add scalar-oracle and `nextafter` fixtures
around each boundary. Vectorization must preserve per-pixel operation ordering
or explicitly qualify any codestream changes caused by contraction or rounding.

## Priority 5: shorten the Metal-to-serializer handoff

### Current boundary

Encoding-only fully-resident execution already omits reconstructed-RGB,
block-distance, and final-quant-field materialization. The remaining final frame
readback in
[`MetalPreparedAqEvaluation`](../src/gpu/metal/metal_aq_evaluation.cpp) contains
primarily:

- strategy-batch-major quantized AC coefficients;
- quantized DC coefficients;
- raw quantization; and
- small color-correlation maps.

Metal buffers use `MTL::ResourceStorageModeShared`, and
[`CopyDeviceToHost()`](../src/gpu/metal/metal_backend.cpp) is a checked
`memcpy` from the mapped buffer. On Apple Silicon this is not a PCIe transfer;
the cost is memory bandwidth, cache coherence, and the subsequent duplicate
host representation.

[`AssembleVarDctEncoderFrame()`](../src/codec/vardct_frame.cpp) then allocates
and zeros group-major AC storage, copies raw quantization and DC, reconstructs
floating DC, scatters each transform's coefficients into its AC group, and calls
the deep `VarDctEncoderFrame::valid()` scan. The evaluator separately scans the
readback for poison values before assembly.

The layout conversion is structural: Metal's final coefficient buffer is
organized by strategy batch, channel, and anchor, while the serializer consumes
fixed-capacity AC-group-major channel storage.

### Low-risk implementation

1. Read raw quantization with one contiguous copy; its current device row stride
   equals its width.
2. Give internal frame assembly writable final spans so raw quantization and
   quantized DC can land directly in owned frame storage rather than temporary
   vectors.
3. Fold poison detection into the coefficient repack loop instead of scanning
   every coefficient immediately before reading it again.
4. Separate construction invariants from the public deep `valid()` audit. A
   successfully checked constructor should not rescan all DC values, active
   coefficients, and zero padding before committing its own result.
5. Reuse final-frame vector capacity across attempts where the ownership model
   permits it.

These changes preserve the existing GPU output layout and serializer input
contract, so they should precede more architectural work.

### Medium-risk implementation

Add a final Metal packing kernel that converts strategy-batch-major output into
the AC-group-major representation expected by the serializer. The kernel can
also initialize required padding deterministically. Compare its dispatch,
completion, and mapped-memory costs against the current approximately 12.9 ms
host assembly phase; a faster kernel in isolation is insufficient if it adds a
new synchronization boundary.

### Architectural implementation

Let the internal serializer consume a read-only frame view over the completed
shared Metal buffer. This can eliminate both the explicit readback `memcpy` and
an owned coefficient copy, but it requires:

- retaining the evaluator and buffer lease through serialization;
- a backend-independent read-only frame/view contract;
- explicit Metal completion and coherence rules;
- safe error and cancellation lifetimes;
- alignment and padding guarantees; and
- concurrency and memory-pressure qualification.

This is a zero-copy coefficient handoff, not GPU entropy coding. rANS remains on
the CPU.

### Expected result and gates

The approximately 10.2 ms readback and 12.9 ms assembly measurements form a
23.1 ms gross boundary. Only part is removable without changing the layout.
Low-risk direct copies and scan fusion should be measured before assigning a
numerical saving; GPU packing or a mapped frame view is justified only if the
post-priority-1 profile still exposes this boundary.

Validation must pin group/channel coefficient order, active counts, zero
padding, DC reconstruction, raw quantization, CfL maps, deterministic hashes,
pinned-`djxl` acceptance, evaluator lifetime on failure, and concurrent
contexts.

## Priority 6: separate ordinary tokenization from maximum compression

### Current residual tournament architecture

Balanced and high-density encoding now select one block-context map, one
coefficient-order representation, and one entropy family. Their AC path still
uses the reusable intermediate representation built for maximum compression:

1. [`BuildSimpleAcGroupTokenTemplateForEncoder()`](../src/codestream/ac_group.cpp)
   validates and collects anchors, allocates three nonzero maps, appends each
   value, and appends one four-byte descriptor per token.
2. `MaterializeSimpleAcGroupContextsForEncoder()` builds block-context values
   and walks every descriptor again to emit a separate `uint16_t` context
   array.
3. Entropy construction walks the resulting split value/context streams again
   to build fixed-HybridUint histograms.
4. Final emission traverses the selected token streams, as it necessarily must.

This split representation was a major improvement for maximum compression:
many candidate context maps can reuse coefficient values and traversal order.
For a one-representation encode, however, the descriptor allocation and second
context pass are unnecessary.

Pinned libjxl's
[`TokenizeCoefficients()`](../third_party/libjxl/lib/jxl/enc_entropy_coder.cc)
computes the final block context and immediately appends the `(context, value)`
token in one coefficient traversal. Although the function is dispatched through
Highway, its central advantage here is not an extraordinary vectorized
coefficient loop. It performs less generalized work and uses per-group scratch.

### Ordinary one-pass token path

Add a dedicated internal ordinary encoder path that receives the already chosen
coefficient orders and block-context map and, for each AC group:

- traverses anchors and coefficients once;
- writes split `values` and final `contexts` arrays directly;
- reuses one nonzero-map and anchor scratch object per participating worker;
- precomputes natural orders once per encode rather than lazily in every group;
  and
- optionally produces cache-local balanced histogram statistics while token
  values are hot.

Keep the current template/context-materialization path for maximum compression,
where its reuse is justified. The two paths should share the actual coefficient
context formulas and validation helpers so behavior does not drift.

### Effort-aware coefficient-order work

Pinned libjxl's
[`ComputeCoeffOrder()`](../third_party/libjxl/lib/jxl/enc_coeff_order.cc)
deterministically samples approximately half of eligible blocks when deriving a
custom coefficient order at the effort-7-like Squirrel tier if only DCT8 is in
use. GJXL currently scans every coefficient at every ordinary effort.

Resolve coefficient-order effort explicitly rather than inferring it solely
from entropy behavior:

- effort 7: use the same deterministic DCT8-only half-sampling policy;
- effort 8: retain full order statistics, matching libjxl's slower tier;
- efforts 9-10/high density: retain full statistics; and
- maximum compression: retain full statistics.

This is deliberately byte-changing. The objective is effort behavior and
speed/density alignment, not byte identity with GJXL's previous effort-7
codestream. Reproducing libjxl's PRNG and traversal policy does not imply that
the two encoders will select identical orders when their quantized frames or
strategies differ.

Also remove redundant trusted-frame validation inside serializer helpers. The
top-level encoder already validates the complete frame; internal calls for DC
tokenization, block-context calculation, and coefficient orders should use
validated variants rather than repeatedly invoking a deep frame audit.

### Balanced histogram construction

The balanced direct-ANS partition uses one fixed HybridUint configuration. Its
current histogram builder re-encodes every ordered token to accumulate symbol
counts and extra-bit totals. The ordinary tokenization task can accumulate
those exact integer statistics concurrently with final-context construction.

Do not scatter every token directly into a process-wide table of approximately
1 KiB histograms. A previous experiment removed the nominal histogram pass but
increased context-materialization work by an order of magnitude because writes
jumped across thousands of cache lines. The viable design is:

1. accumulate group-local populations only for contexts touched by that group;
2. retain compact touched-context and sparse-symbol bounds;
3. reduce groups in deterministic group order into the global context
   histograms; and
4. pass those populations into `PrepareDirectAnsPartition()` so it skips its
   full token scan.

For high-density effort 9, retain the legitimate search across multiple
HybridUint configurations and precise histogram forms. The one-pass token
representation still applies, but an effort-7 fixed-configuration population
shortcut must not silently remove effort-9 search work.

### Expected result and gates

The raw 4K phase labels report 21.9 ms for GJXL and 2.2 ms for libjxl, but the
boundary mismatch assigns coefficient-order work only to GJXL. After correcting
that mismatch, the earlier profile estimated approximately 8.7 ms of 4K delay
from GJXL's coefficient-order/token-preparation behavior. The separate
entropy-construction gap was approximately 8.9 ms. Those are the intended
targets.

Final rANS/model emission should not be redesigned as part of this work. At 4K
GJXL spends 8.9 ms there versus libjxl's 21.3 ms. rANS is serial within a stream,
and moving it to Metal would add dispatch and synchronization to a phase that is
already ahead.

The one-pass representation and prepared-population changes should preserve
exact ordinary codestream bytes. Effort-7 coefficient-order sampling should be
qualified through encoded size, decoded pixels, Butteraugli, and deterministic
repetition rather than old-byte equality. Effort 9 and maximum compression must
retain their current search intensity and pinned hashes unless explicitly
changed by a separate proposal.

## Recommended implementation sequence

The priority numbers describe causal importance, but the safest implementation
order separates low-risk work elimination from lifetime and layout redesign:

1. Add allocation, zeroed-byte, copied-byte, validation-pass, and frame-handoff
   counters where the current wall stages do not distinguish them.
2. Implement priority 2's lazy materialization and duplicate-Opsin removal.
3. Implement priority 3's internal direct-write, single-validation color path.
4. Implement priority 4's effort/mode gate and vectorized retained statistics.
5. Incorporate the independently owned effort-7 zero-update change and capture
   a fresh matched profile.
6. Implement priority 5's low-risk direct frame copies and scan fusion.
7. Implement priority 6's one-pass ordinary tokenization and effort-aware
   coefficient-order policy.
8. Add cache-local balanced histogram population reduction.
9. Re-profile before considering persistent workspace pooling, a Metal
   group-packing kernel, or a mapped zero-copy frame view.

This sequence prevents two common forms of wasted optimization: building a
persistent pool around allocations that should not exist, and redesigning the
GPU coefficient layout before the ordinary serializer's required representation
has stabilized.

## Qualification contract

Every retained optimization should use a fresh Release build, alternating
independent processes, warmups, multiple samples, and representative 1080p, 4K,
and higher-resolution inputs. Report complete-encode wall time separately from
stage wall time, aggregate worker time, and GPU timestamps.

The functional matrix must cover:

- efforts 7, 8, 9, and 10;
- explicit maximum compression;
- forced CPU, forced Metal, and automatic fallback;
- fully-resident, exact-coefficients, and supported throughput paths;
- single-target, target-size repeated attempts, and maximum-error control;
- final Butteraugli diagnostics on and off;
- explicit CPU thread budgets and automatic scheduling;
- batch encoding and concurrent calls on one C API context; and
- injected allocation, upload, completion, readback, and numeric failures.

For byte-preserving changes, retain selected block-map/order metadata, entropy
modes, cluster counts, model/token bit accounting, complete SHA-256 hashes,
pinned-`djxl` acceptance, and decoded-pixel comparison. For intentionally
byte-changing effort alignment, retain deterministic output and compare encoded
size and decoded perceptual quality against both the parent GJXL build and the
matched libjxl effort.

## Explicit non-goals

- Do not restore the maximum-compression representation tournament at an
  ordinary effort.
- Do not call aggregate worker time wall-clock delay.
- Do not infer that a Metal kernel is slow from the complete pipeline without a
  matched operation boundary.
- Do not cache one mutable prepared workflow in `GJXLContext` and serialize
  concurrent calls.
- Do not remove public finite-input or failure-atomicity guarantees to speed up
  a trusted internal caller.
- Do not replace effort-9 HybridUint and precise-histogram search with the
  effort-7 fixed-population shortcut.
- Do not prioritize GPU entropy coding while CPU final emission is already
  faster than libjxl and repeated host passes remain.
