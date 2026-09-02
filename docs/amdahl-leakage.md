# Amdahl leakage after entropy-behavior alignment

- Status: active implementation roadmap
- Original profile revision: `af3a9e6`
- Current roadmap baseline: `c81ff95`
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

The roadmap priorities are:

| Priority | Area | Current signal or status | Next implementation target |
| ---: | --- | ---: | --- |
| 2 | Preparation and storage | Direct padded-Opsin slice: padded-4K input preparation 75.09 -> 54.42 ms and peak RSS 1.161 -> 1.043 GiB | Re-profile before considering persistent workspace leases |
| 3 | Color conversion, validation, and copies | Direct workflow transform plus identity-bound finite-input provenance implemented; redundant scans cost 4.61 ms at 1080p and 18.49 ms at 4K | Qualified; proceed to priority 4 |
| 4 | Quantization-matrix chromaticity statistics | Gate plus trusted NEON pass implemented; retained effort-7 preparation improved by 5.39 ms at 1080p and 25.49 ms at 4K | Qualified; incorporate effort-7 update policy and re-profile |
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

### Landed first slice: two output contracts

Commit `3d1b93b` and the first storage slice recorded in `40b6a11` made fully
resident Metal the default for qualified automatic single-target encodes and
removed unused host storage from that path. The implementation does not carry a
six-way execution plan. It uses two concrete output contracts, with the existing
materialization flags handling only optional results inside those contracts:

| Caller or mode | Output contract | Host behavior |
| --- | --- | --- |
| Metal exact coefficients | Lean encoding | Materializes preprocessed Opsin lazily, then returns the frame, score history, and control result |
| Metal fully resident or throughput | Lean encoding | Returns the frame, score history, and control result without a host preprocessed-Opsin image |
| CPU codestream adapter | Complete compatibility | Allocates `PipelineStorage` only if CPU is actually selected |
| Metal maximum-throughput adapter | Complete compatibility | Allocates `PipelineStorage` only when this explicit frame-only mode is selected |
| Public AQ and quantization-pipeline APIs | Complete diagnostics | Preserves the promised quant fields, masks, block-distance map, reconstructed RGB, frame, and scores |
| Maximum-error control or final-score collection | Modifier on either contract | Adds only the required control result or terminal perceptual evaluation; it is not another storage plan |

`PreparedWorkflow` now keeps a small `EncodingArtifacts` result for the normal
encoding path. Its complete `PipelineStorage` is a lazy compatibility allocation
used only by the CPU and maximum-throughput adapters. The default resident path
therefore does not construct host diagnostic quant fields, block distances, or
reconstructed RGB.

`PreparedQuantizationPipeline` no longer owns a second set of final quantization,
block-distance, reconstructed-RGB, frame, or score-history results. Adaptive-
quantization providers already promise atomic output, so they commit directly
to the selected caller-owned destination. The host preprocessed-Opsin image is
also allocated only when the exact or CPU implementation needs it. CPU
Butteraugli reference preparation is eager only for an explicitly forced CPU
encode and remains available lazily for automatic fallback.

Some initial quantization, strategy, and pixel-mask fields remain because the
resident frontend currently reads them back and uses them for AC search and AQ
reconfiguration. Their retention follows actual use rather than the final
diagnostic materialization flags.

Automatic backend policy is narrower than the original plan assumed. Qualified
automatic single Butteraugli-target encoding may select resident Metal and fall
back to CPU if Metal is unavailable. Automatic target-byte, target-BPP, and
maximum-error searches stay entirely on CPU so one search never mixes CPU and
resident rate curves. Forced Metal continues to support resident maximum-error
control. Lazy compatibility storage is consequently needed for genuine fallback,
not for arbitrary backend changes between target-size attempts.

### First-slice measurements

The retained A/B used Release builds and five alternating independent process
pairs per workload, with two warmups and five measured samples per process. The
table reports the median of process medians. Measurements were captured on
patch-equivalent commits `535acfb` and `7732b9f`; their changes are recorded in
the `3d1b93b` and `40b6a11` branch history named here.

| Workload | Stage | Parent | First storage slice | Change |
| --- | --- | ---: | ---: | ---: |
| padded 1080p | Input preparation | 23.359 ms | 18.953 ms | -4.406 ms (-18.9%) |
| padded 1080p | Complete encode | 174.466 ms | 171.169 ms | -3.297 ms (-1.9%) |
| padded 4K | Input preparation | 97.576 ms | 82.201 ms | -15.375 ms (-15.8%) |
| padded 4K | Complete encode | 699.764 ms | 654.247 ms | -45.517 ms (-6.5%) |

Complete-encode timing was noisier than the preparation-stage result, so its
improvement is directional rather than a precise throughput promise. Every
sample selected Metal and retained its encoded size: 410,072 bytes at 1080p and
1,606,911 bytes at 4K. The resident encoding test also compares the lean and
complete-output frame and codestream byte-for-byte.

Three alternating padded-4K process pairs, each with one warmup and one measured
sample under `/usr/bin/time -l`, reduced median maximum resident set size from
1.56 GiB to 1.22 GiB: approximately 342 MiB, or 21%. The pinned-libjxl Release
matrix passed 66 of 67 tests; the only failure was the pre-existing CPU
quantization golden mismatch, reproduced unchanged on the parent.

### Landed second slice: borrow the coding Opsin image

Commit `55048f2` makes `PreparedQuantizationPipeline::coding_opsin` borrow the
workflow-owned immutable Opsin view for the synchronous prepared lifetime.
Exact and CPU paths
still own a separate preprocessed image because Gaborish inversion changes their
search-domain pixels; resident paths no longer create a duplicate of the
unfiltered coding image.

Each successful pipeline preparation receives a monotonic source generation.
`PreparedAdaptiveQuantization` records that generation and discards its
source-dependent evaluator and resident views when a new preparation is bound,
even if the caller reused identical host addresses. AC-search scratch is retained
because it caches capacity rather than source pixels. A focused regression test
rewrites one image in place, re-prepares through the same views, and verifies the
reused path against a fresh resident frame and codestream.

Five alternating Release process pairs measured the following preparation-stage
change relative to `40b6a11`, with two warmups and five samples per process:

| Workload | Parent | Borrowed Opsin | Change |
| --- | ---: | ---: | ---: |
| padded 1080p | 19.031 ms | 17.196 ms | -1.835 ms (-9.6%) |
| padded 4K | 89.961 ms | 76.447 ms | -13.514 ms (-15.0%) |

All five paired 4K preparation results favored the borrowed view. Complete
encode timing remained GPU-noise-limited: the median paired difference was
-6.6 ms at 1080p and -1.6 ms at 4K, with wide ranges, so this slice makes no
standalone end-to-end speed claim. Every sample selected Metal and retained the
same encoded size as its parent cohort.

Three alternating padded-4K RSS pairs reduced median maximum resident set size
from 1.24 GiB to 1.17 GiB, approximately 70 MiB or 5.5%. The pinned-libjxl
Release matrix again passed 66 of 67 tests; the sole failure was the unchanged,
pre-existing CPU quantization golden mismatch.

### Implemented third slice: direct padded Opsin preparation

`PreparedWorkflow` now borrows the caller's immutable linear-RGB view for the
synchronous encode and all target-size attempts. It no longer owns
`padded_linear`. The internal `LinearRgbToPaddedOpsin()` preparation primitive
transforms source rows directly into the workflow-owned padded Opsin image,
fills each transformed right edge, and duplicates only the final transformed
row into bottom padding.

Input and transformed-output finite checks are fused into the scalar or NEON
conversion loop. The internal destination stays private and is discarded after
failure, so it does not require another full-image staging buffer. The public
`LinearRgbToOpsin()` entry point still converts through private scratch and
commits only after success, preserving its in-place support and failure
atomicity.

Five alternating independent Release process pairs compared this change with
the `55048f2` baseline. Each process used two warmups and five measured samples;
the table reports the median of process medians:

| Workload | Stage | `55048f2` | Direct padded Opsin | Change |
| --- | --- | ---: | ---: | ---: |
| padded 1080p | Input preparation | 17.240 ms | 9.375 ms | -7.865 ms (-45.6%) |
| padded 1080p | Complete encode | 175.513 ms | 157.849 ms | -17.664 ms (-10.1%) |
| padded 4K | Input preparation | 75.088 ms | 54.422 ms | -20.666 ms (-27.5%) |
| padded 4K | Complete encode | 655.957 ms | 615.225 ms | -40.732 ms (-6.2%) |

All five paired preparation and complete-encode results favored the candidate
at both sizes. Every sample selected Metal and preserved encoded size: 410,072
bytes at 1080p and 1,606,911 bytes at 4K. A separate forced exact-coefficient
Metal smoke retained its 434,392-byte output; its one-sample timing is not a
performance claim. A cross-binary CLI smoke over the repository fixture was
byte-identical before and after the slice for CPU, fully resident Metal, and
exact-coefficient Metal.

Three alternating padded-4K RSS pairs reduced median maximum resident set size
from 1.161 GiB to 1.043 GiB, approximately 121.5 MiB or 10.2%. A targeted test
compares strided direct output bit-for-bit with the former pad-then-transform
result, checks row-stride padding and non-finite rejection, and verifies public
atomicity when finite input overflows during conversion. It passes in both the
native NEON build and a forced-scalar build. The Release matrices
passed 61 of 62 tests without libjxl and 66 of 67 with pinned libjxl; in each,
the sole failure was the unchanged pre-existing CPU quantization golden.

### Deferred option: persistent workspace leases

Consider preserving useful capacity across independent images with an internal
`VarDctEncoderWorkspace` or `MetalEncodingWorkspace` only after duplicate image
ownership and direct-write passes have landed and a fresh profile still
justifies it. Do not cache an entire `PreparedWorkflow` unchanged.

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

### Remaining gates

The direct-transform change must preserve borrowed-view lifetime, failure
atomicity, and prepared-GPU invalidation. Required coverage includes automatic
CPU fallback, forced CPU and Metal, target-size repeated attempts, maximum-error,
final-score diagnostics, concurrent C API calls, batch workers, allocation
failures, and device-error invalidation.

## Priority 3: validate and move image pixels once

### State after the direct-write and provenance slices

The workflow previously ran [`EdgeExtend()`](../src/codestream/workflow.cpp)
over the complete padded destination, then called the atomic public
[`LinearRgbToOpsin()`](../src/codec/color_transform.cpp), which allocated and
copied another full padded result. Both workflow-only passes and the owned
padded-linear image are now gone.

The retained internal path:

- validates source values while transforming them;
- writes transformed values directly into workflow-owned Opsin storage;
- combines output-finite detection with the transform loop or its vector
  reduction; and
- fills only the transformed right edge and duplicates the final transformed
  row into bottom padding.

Quantization-matrix statistics still scan Opsin. The ordinary fully resident
workflow no longer repeats finite-value validation inside Metal evaluator
preparation. `PreparedQuantizationPipeline` records the exact immutable RGB and
Opsin views validated by the direct transform, and the backend-neutral resident
frontend checks both identities before using a private validated-preparation
capability. A different source view falls back to the public validating entry
point. Geometry, layout, strategy, EPF, option, and resident-device-view checks
remain unconditional.

The trust contract is not present on `AqEvaluationPreparation` or the public
`PrepareAqEvaluation()` API. Direct public GPU calls, exact-coefficient paths,
and other callers without matching preparation provenance retain their finite
scans. This keeps an optimization established by one synchronous workflow from
becoming a caller-controlled validation bypass.

Finite input alone does not prove finite output. The public linear-float API can
receive very large finite values that overflow during matrix arithmetic, so the
output check must be fused, not silently deleted. The public
`LinearRgbToOpsin()` function should also retain failure atomicity: a failed
standalone call must not leave a partially transformed destination.

The color transform currently creates fresh `std::thread` workers for each
parallel invocation. That overhead is lower priority. Earlier shared-executor
experiments were not unconditionally faster, so full-image allocations and
passes should be removed before scheduling is redesigned.

### Result and validation

The direct-write slice exceeded the original 5-8 ms estimate because it removed
both the persistent padded-linear image and the public transform's padded
scratch/copy from the workflow. At padded 4K, input preparation improved by
20.7 ms in the retained cohort and complete encode by 40.7 ms, with a 121.5 MiB
peak-RSS reduction. The larger end-to-end result includes allocator and memory-
traffic effects outside the removed loop and must not be interpreted as color-
transform CPU time alone.

Temporary timers around the production `ValidateFiniteImage()` calls measured
21 preparations at each resolution across seven independent processes. The
medians were:

| Finite validation | Padded 1080p | Padded 4K |
| --- | ---: | ---: |
| Original linear RGB | 2.29 ms | 9.38 ms |
| Coding Opsin | 2.33 ms | 9.16 ms |
| Combined critical-path scan | 4.61 ms | 18.49 ms |

Seven alternating stage-profile pairs compared the retained implementation
with `3313bbe`. Median `frontend.prepare_evaluator` wall time improved from
29.89 to 26.39 ms at padded 1080p (11.7%) and from 129.94 to 108.88 ms at
padded 4K (16.2%). All seven 1080p pairs and six of seven 4K pairs favored the
provenance path. Complete-encode timing remained noisier than this bounded
stage, so the scan and evaluator numbers are the causal qualification for the
slice rather than an inflated end-to-end claim.

Public-preparation tests inject NaN into original RGB and infinity into coding
Opsin and require rejection before any GPU allocation or submission. A prepared
pipeline test also substitutes a different non-finite RGB view and verifies
that provenance does not follow it. Effort-7-like, high-density, and
maximum-error CLI smoke encodes remained byte-for-byte identical to `3313bbe`
and were accepted by pinned `djxl` 0.12.0.

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

The behavioral difference was when and how the pass ran. Libjxl skips the pixel
scan at tiers faster than effort 7 and returns before it in maximum-error mode.
GJXL previously computed the statistics during `PrepareWorkflow()` even when
`SelectQuantizationMatrixScales()` would ignore them for maximum-error.

### Implementation

The first slice now matches libjxl's control policy:

- efforts 1-6: use the distance/default scale path without pixel statistics;
- efforts 7-10 and the explicit effort-9-like high-density override: retain
  pixel statistics;
- maximum-error: skip the pass entirely.

Maximum compression continues to control only entropy/codestream search, so it
does not independently enable the pixel pass at low effort. The gate is an
explicit internal policy queried by `PrepareWorkflow()`; the standalone
statistics helper retains its complete validation contract.

The second slice optimizes the retained effort-7 and effort-9-like path:

- current and previous row pointers are hoisted outside the pixel loop;
- per-sample finite checks are removed only for the exact Opsin view validated
  by the synchronous RGB-to-Opsin transform;
- horizontal and vertical X differences, B-minus-Y differences, and
  exposed-blue products use four-wide NEON reductions; and
- scalar handling remains for vector tails and non-NEON targets.

The standalone internal helper retains its checked scalar behavior. A separate
finite-Opsin entry point makes the trust boundary explicit and leaves results
unchanged on failure. Independent row stripes were not added: the vectorized
pass brought preparation close enough to the no-statistics effort-6 floor that
thread-launch and reduction complexity are not justified by the retained
profile.

Do not initially fuse this pass into Opsin conversion. Vertical-neighbor state
and parallel row scheduling make fusion harder to qualify, while a standalone
SIMD maximum reduction is comparatively contained.

### Effort/mode-gate result

Seven alternating independent-process pairs used a Release build, two warmups,
and five samples per process with fully resident Metal at effort 6. The table
reports the median of each seven-process cohort:

| Input | Baseline preparation | Gated preparation | Removed wall time |
| --- | ---: | ---: | ---: |
| Padded 1080p | 9.28 ms | 3.25 ms | 6.03 ms (64.9%) |
| Padded 4K | 49.60 ms | 15.29 ms | 34.32 ms (69.2%) |

Every paired preparation comparison favored the gate. Complete-encode timing
also improved in every pair, but those results combine the eliminated pass with
the intentional low-effort matrix-scale and codestream change, so they are not
used as the causal timing claim. On the synthetic benchmark, effort-6 output
became 10.7% smaller at both resolutions; final Butteraugli changed from 1.597
to 1.589 at 1080p and from 1.647 to 1.699 at 4K. These are policy-alignment
observations, not a matched-quality density claim.

Effort 7, explicit high density, and maximum-error smoke encodes remained
byte-for-byte identical to `1e8bb0e`. The maximum-error result demonstrates
that the formerly unused scan was removed without changing its fixed 2/2
matrix scales. Candidate codestreams, including the deliberately changed
effort-6 output, were accepted by `djxl` 0.12.0.

### Retained-pass result and gates

Seven alternating independent-process pairs compared the optimized path with
`c81ff95`, again using Release builds, two warmups, and five samples per process.
Both sides used fully resident Metal at effort 7, so the matrix statistics were
retained and codestream behavior was identical:

| Input | Scalar preparation | Trusted NEON preparation | Removed wall time |
| --- | ---: | ---: | ---: |
| Padded 1080p | 9.96 ms | 4.57 ms | 5.39 ms (54.1%) |
| Padded 4K | 41.41 ms | 15.92 ms | 25.49 ms (61.6%) |

Every preparation pair favored the optimized path. Complete-encode cohort
medians were noisy at 1080p (159.23 versus 164.09 ms) and improved at 4K
(590.24 versus 561.01 ms); five of seven 1080p pairs and six of seven 4K pairs
favored the optimized build. The bounded preparation stage is therefore the
causal result, while a fresh matched end-to-end profile should wait for the
independently owned effort-7 update-policy change.

The checked scalar and trusted vector paths produce exactly equal statistics on
degenerate, strided, vector-width, and scalar-tail fixtures. Efforts 6, 7, 9,
and 10, explicit high density, maximum compression, and maximum-error smoke
encodes were byte-for-byte identical to `c81ff95`; every candidate was accepted
by `djxl` 0.12.0. The previously sampled 16.8 ms statistic attribution used a
different workload and sampled-CPU boundary, so it should not be compared
directly with the 25.49 ms 4K preparation-stage reduction.

The thresholds `0.015`, `0.022`, `0.026`, `0.28`, `0.33`, `0.38`, and `0.13`
directly affect the frame header. Existing `nextafter` selection fixtures cover
every boundary, and the retained SIMD path is checked against the scalar oracle.
Any future change to operation contraction or rounding requires renewed
codestream qualification.

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

The priority numbers describe causal importance, but the implementation order
separates low-risk work elimination from lifetime and layout redesign:

1. **Completed in `3d1b93b`:** make fully resident Metal the ordinary default
   while retaining exact coefficients as an explicit compatibility path.
2. **Completed in `40b6a11`:** give encoding and public diagnostics two concrete
   output contracts; remove dead resident host materialization and defer CPU-only
   preparation until CPU is selected.
3. **Completed in `55048f2`:** remove duplicate workflow-to-prepared Opsin
   ownership, with explicit borrowed-view lifetime and prepared-state
   invalidation.
4. **Completed in `3313bbe`:** remove padded linear RGB and the
   workflow's atomic color-transform scratch/copy through a checked direct-write
   path; preserve the public transform contract.
5. **Completed in `1e8bb0e`:** bind finite-input provenance to the
   exact workflow RGB/Opsin views and skip the measured duplicate scans only in
   private fully resident preparation; retain public GPU validation.
6. **Completed in `c81ff95`:** apply priority 4's effort/mode gate;
   preserve high-density behavior and keep maximum compression orthogonal.
7. **Completed in the current slice:** vectorize priority 4's retained
   effort-7/high-density statistics under explicit finite-Opsin provenance.
8. Incorporate the independently owned effort-7 zero-update change and capture
   a fresh matched profile.
9. Implement priority 5's low-risk direct frame copies and scan fusion.
10. Implement priority 6's one-pass ordinary tokenization and effort-aware
   coefficient-order policy.
11. Add cache-local balanced histogram population reduction.
12. Re-profile before adding more counters or considering persistent workspace
    pooling, a Metal group-packing kernel, or a mapped zero-copy frame view.

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
