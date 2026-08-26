# Metal adaptive-quantization implementation roadmap

This document owns the plan for accelerating gjxl's iterative adaptive-
quantization (AQ) evaluation loop with Metal. The CPU algorithm and supported
codec subset remain documented in [`quantization.md`](quantization.md). The
native perceptual metric, its intermediate oracles, and the standalone Metal
Butteraugli operation remain documented in
[`butteraugli.md`](butteraugli.md).

The objective is an end-to-end reduction in the time spent by
`FindBestQuantization`, not isolated kernel speedups. The default AQ policy
performs two quant-field updates and therefore three complete
encode/reconstruct/measure evaluations. A useful Metal path must keep the
large images and intermediate planes resident across those evaluations and
limit synchronization to the data required by the CPU update policy.

The current public GPU abstraction supports buffer allocation, host transfers,
batched transforms, and explicit synchronization. It does not yet model
strided device images, reusable scratch, multi-kernel codec operations, or
reductions. Existing Metal DCT kernels and their independent CPU oracles are
the transform foundation; they are not by themselves an AQ implementation.

## Goals

- Accelerate the complete AQ encode/reconstruct/measure evaluation on Metal.
- Retain the CPU implementation as the readable executable specification and
  fallback path.
- Keep quant-field convergence policy and iteration order on the CPU initially.
- Keep original, coding, reconstruction, filtering, and perceptual images
  device-resident when doing so removes avoidable transfers.
- Reuse device-image, submission, scratch, and reduction infrastructure across
  reconstruction, filtering, Butteraugli, and later codec operations.
- Preserve accepted encoder decisions: exact raw quant values and documented
  tolerances for floating-point fields, reconstructed images, and scores.
- Establish resident and full end-to-end performance evidence at realistic
  image sizes before enabling automatic backend selection.

## Non-goals

- Moving initial quant-field generation or AC-strategy search into this effort.
  They precede iterative AQ and have separate dataflow and benchmarks.
- Moving quant-field update policy, convergence decisions, or iteration order
  to the GPU before profiling demonstrates a need.
- Retuning Butteraugli, quantization matrices, loop filters, or encoder policy.
- Requiring bit-identical CPU and GPU floating-point intermediates.
- Claiming complete JPEG XL encoder or bitstream parity for features outside
  the supported CPU reference subset.
- Designing entropy or bitstream coding as part of the first AQ acceleration
  path.

## Ownership boundaries

### CPU quantization reference

[`quantization.md`](quantization.md) owns the algorithmic order, supported AC
strategies, fixed coefficient-coding subset, known encoder deviations, and CPU
integration tests. CPU results are the primary oracle for GPU reconstruction
and AQ decisions.

### Standalone Butteraugli implementation

[`butteraugli.md`](butteraugli.md) owns native CPU Butteraugli, pinned libjxl
differential tests, intermediate-stage fixtures, the explicit device
Butteraugli operation, prepared-reference behavior, and Butteraugli-specific
numerical tolerances.

### Shared GPU infrastructure

This roadmap owns the requirements and integration of device image views,
multi-kernel submission, reusable scratch, and generic reductions because the
AQ path is their first cross-operation consumer. The types must remain generic
and live below codec-operation APIs; Butteraugli- or reconstruction-specific
fields must not leak into the shared contracts.

### AQ orchestration

The Metal AQ operation belongs under `src/gpu/ops/` and must depend on the
backend abstraction rather than Metal C++ types. Backend-specific pipeline
creation, command encoding, resource validation, and synchronization remain in
`src/gpu/metal/`.

## Evaluation boundary

The first Metal implementation keeps field construction and policy on the CPU
and moves the throughput-heavy image path to the GPU:

```text
CPU, once per frame
  adjusted initial quant field
  selected AC strategies
  original linear RGB + padded coding opsin
               |
               v
prepare device evaluation state
  upload static images, strategies, tables, and options
  prepare original Butteraugli reference
  allocate reusable scratch at maximum required capacity
               |
               v
CPU, once per evaluation
  quant field -> quantizer + raw quant field
  final CfL map + EPF inverse sigma
               |
               v
one Metal submission
  forward transforms + coefficient coding
  dequantization + inverse transforms + inverse CfL
  loop filters
  opsin-to-linear RGB
  Butteraugli distance map + scalar score
  strategy-aware 16-norm block reduction
               |
               v
one bounded readback
  block-distance map + scalar score
               |
               v
CPU quant-field update policy
```

Keeping quantizer construction, final CfL, and EPF field generation on the CPU
is an initial boundary, not a permanent performance claim. Each may move only
after the complete resident path is measured and a separate GPU port can
preserve its decisions. The first implementation must not obscure their upload
cost in end-to-end measurements.

The final evaluation may additionally read back the reconstructed image and
other caller-visible output required by the existing CPU API. A future encoder
handoff may retain final coefficients on the device, but that representation is
outside the first AQ milestone and must not be invented implicitly by scratch
buffers.

## Residency and transfer contract

### Prepared state

The prepared state is bound to one backend device, source geometry, strategy
grid, Butteraugli option set, and coefficient-coding configuration. It owns or
retains device-resident copies of:

- Original unpadded linear RGB.
- Padded coding opsin.
- Complete AC-strategy metadata.
- Quantization tables and transform constants required by the selected
  strategies.
- Prepared Butteraugli reference data.
- Reusable coefficient, image, filter, perceptual, and reduction scratch.

Preparation must finish validation and allocation before any evaluation is
submitted. A failed preparation leaves no partially usable state.

### Per-evaluation inputs

The initial implementation uploads only iteration-dependent data:

- Raw quant field and quantizer parameters.
- Final color-correlation map.
- EPF inverse-sigma field and any enabled-filter controls.

The float quant field remains a CPU policy input unless profiling justifies a
device conversion. Every transfer must be reported in the full-E2E benchmark.

### Per-evaluation outputs

Intermediate AQ evaluations read back only:

- One block-resolution distance per base block.
- The aggregate Butteraugli score.

The pixel-resolution distance map must remain device-resident. Reading it back
to perform the 16-norm reduction on the CPU is not an acceptable steady-state
path. The final evaluation may additionally commit the reconstructed linear RGB
image and existing CPU-visible state.

## Shared GPU contracts

The shared substrate must provide the following without exposing Metal types in
public codec headers.

### Device planes and images

A device plane view needs an owning buffer, byte offset, element type, extent,
and row stride. A three-plane image view additionally needs an explicit planar
layout or three plane views. Validation must reject:

- Empty or overflowing extents and strides.
- Buffer ranges that exceed the allocation.
- Buffers owned by another backend or device.
- Unsupported aliasing between inputs, outputs, and scratch.
- Layouts not supported by a selected kernel.

Contiguous images remain the preferred internal representation, but tests must
exercise valid strided layouts so kernels do not acquire undocumented packing
assumptions.

### Submission and synchronization

An AQ evaluation should encode its dependent kernels into one backend
submission. It may use more than one command encoder where a blit or resource
transition requires it, but must not commit and synchronize after each leaf
operation. Submission failure and command-buffer failure must be distinguishable
from invalid caller input.

The shader build must compile and link multiple controlled in-tree `.metal`
sources into the existing gjxl metallib. Backend creation binds exact exported
function names and validates required pipeline state up front; runtime
reflection or prefix-based discovery is not part of the contract. Coherent
codec operations belong above the backend and must not expand `GpuBackend` into
one virtual method per private leaf kernel.

Caller-visible output is committed only after validation and successful GPU
completion. Invalid requests must submit no work. Tests may read intermediate
device planes explicitly; production APIs must not hide diagnostic readbacks.

### Scratch planning

Scratch capacity is derived from checked geometry and the enabled pipeline
before the first evaluation. The steady-state iteration path performs no device
allocations. The planner must document which temporary planes alias in time and
which must remain live across kernels. Benchmarks report both total persistent
bytes and peak scratch bytes.

### Reductions

The substrate must support maximum reduction for Butteraugli scores and the
strategy-aware AQ block reduction. Reduction order may differ from the CPU, but
the result must satisfy a fixed tolerance chosen before performance tuning.
Partial threadgroups, odd dimensions, and padded image edges require direct
oracle coverage.

## Milestones

All milestones below are pending. Milestone 0 is the implementation entry
point; later milestones must not be treated as complete until their stated
exit criteria pass.

### 0. Refresh the AQ baseline and freeze the evaluation contract

- Add timing boundaries inside one CPU AQ evaluation for field construction,
  coefficient coding, reconstruction, loop filters, color conversion,
  Butteraugli, and block reduction.
- Refresh the complete pipeline baseline after native CPU Butteraugli became
  the production facade.
- Measure zero-update and default two-update AQ separately.
- Record native CPU allocation or peak-resident evidence in addition to the
  existing libjxl-managed-byte benchmark.
- Write the concrete prepared-state, per-evaluation input, output, and failure
  contracts before adding GPU kernels.

Exit criterion: Release measurements identify the cost of every evaluation
stage and the documentation fixes the first CPU/GPU boundary without relying on
an assumed bottleneck.

### 1. Add reusable device-image and submission infrastructure

- Define backend-neutral device plane and three-plane image views.
- Add checked conversion from view geometry to buffer byte ranges.
- Extend the build to compile and link multiple in-tree Metal source files into
  one metallib with exact, creation-time pipeline bindings.
- Add a backend operation mechanism that can encode several dependent kernels
  in one submission.
- Add reusable scratch allocation with explicit lifetime and capacity rules.
- Add test utilities for guarded uploads, poisoned outputs, and intentional
  intermediate readback.
- Implement and directly test a pointwise operation, separable convolution, and
  maximum reduction over odd and strided extents.

Exit criterion: shared infrastructure passes CPU-oracle tests on a real Metal
device and a complete multi-kernel test incurs one submission and no
steady-state allocation.

### 2. Define and validate the prepared AQ evaluation operation

- Add a GPU operation that separates one-time preparation from repeated
  evaluations.
- Upload static images, strategies, and tables during preparation.
- Allocate buffers for the largest supported evaluation before the first
  submission.
- Validate backend/device ownership, geometry, offsets, strides, aliasing,
  options, strategy support, and all size arithmetic.
- Add disabled/unavailable-backend behavior without adding a Metal dependency to
  the CPU codec library.
- Provide a test-only no-op or staged path that verifies transfer and lifetime
  contracts before reconstruction kernels are connected.

Exit criterion: malformed requests fail before submission, valid prepared state
can execute repeated evaluations without allocation, and destruction safely
waits for outstanding work.

### 3. Port coefficient coding and reconstruction

- Gather source pixels for every selected transform and channel.
- Reuse the registered Metal forward and inverse transforms for all seven
  supported strategies.
- Implement DC/LLF separation, AC quantization/dequantization, coefficient
  layout, and inverse color correlation required by the CPU reference subset.
- Scatter reconstructed transforms into the padded opsin image without gaps,
  overlap, or stale pixels.
- Initially read back reconstructed opsin for direct comparison before attaching
  downstream filters.

Exit criterion: forward coefficients, quantized coefficients, reconstructed
opsin, and multi-transform output match independent CPU oracles for every
strategy, including asymmetric impulses, deterministic random blocks,
quantization boundaries, partial groups, and NaN-poisoned multi-block output.

### 4. Port loop filters and color conversion

- Port the enabled Gaborish and EPF paths used by the AQ reference subset.
- Preserve disabled-filter behavior without unnecessary copies or dispatches.
- Port cropped opsin-to-linear RGB conversion with the existing intensity
  target contract.
- Keep reconstructed opsin, filter scratch, and reconstructed linear RGB
  resident in the prepared evaluation state.
- Compare each stage independently before chaining it to reconstruction.

Exit criterion: reconstructed linear RGB meets its fixed CPU-oracle tolerance
for odd source extents, padded edges, every supported filter configuration, and
the varied quantization corpus.

### 5. Integrate the device Butteraugli operation

- Implement the device operation and intermediate parity milestones owned by
  [`butteraugli.md`](butteraugli.md).
- Prepare the original image once and reuse its opsin and frequency data across
  AQ evaluations.
- Consume reconstructed linear RGB without host readback.
- Produce a device-resident distance map and aggregate score.
- Preserve explicit one-shot and prepared-comparison benchmark paths.

Exit criterion: the complete Metal distance map and score pass the Butteraugli
corpus, and repeated prepared comparisons require no reference recomputation,
device allocation, or pixel-image transfer.

### 6. Add device AQ block reduction and CPU policy integration

- Reduce the pixel distance map to the transform-aware 16-norm block map on the
  GPU.
- Cover multiblock strategies, partial source edges, and every supported
  strategy footprint.
- Read back only block distances and the score for intermediate evaluations.
- Run the unchanged CPU clamp, power, rounding-progress, and bounds policy.
- Execute the default two-update loop using one GPU submission and one bounded
  readback per evaluation.
- Commit final outputs only after the last evaluation succeeds.

Exit criterion: score history and floating-point AQ fields remain within their
fixed tolerances, every raw quant value matches exactly, and no intermediate
pixel-resolution plane crosses the CPU/GPU boundary.

### 7. Harden residency, failure behavior, and final output

- Reuse prepared state across repeated calls with compatible geometry and
  options, or reject incompatible reuse explicitly.
- Verify concurrent prepared states do not share mutable scratch.
- Exercise device loss, command-buffer errors where injectable, allocation
  failure, malformed descriptors, and destruction with outstanding work.
- Define final reconstructed-image readback and any later encoder handoff
  separately from scratch layout.
- Report persistent and peak scratch memory by image size.

Exit criterion: failures never expose partially committed CPU output, resource
lifetimes are explicit, and the steady-state path has no per-evaluation device
allocation.

### 8. Establish the end-to-end performance and rollout gate

- Benchmark CPU, Metal-resident, and full-E2E evaluation paths with balanced
  order and repeated independent Release processes.
- Measure one evaluation, default iterative AQ, and the complete quantization
  pipeline separately.
- Include preparation, uploads, synchronization, readback, and final-output
  download in clearly labeled full-E2E results.
- Determine a measured CPU/Metal crossover and retain CPU fallback below it.
- Verify automatic backend selection never changes accepted AQ decisions.
- Document unsupported hardware and backend-unavailable behavior.

Exit criterion: Metal provides a stable full-E2E speedup for the target image
sizes, all correctness gates pass, and the documented fallback policy handles
small images or unavailable Metal devices without changing the CPU contract.

## Validation matrix

GPU validation is layered so a final round trip cannot hide a wrong
intermediate:

1. Leaf kernels compare every output element with an independent CPU oracle.
2. Stage tests compare coefficient coding, reconstruction, filtering, color
   conversion, Butteraugli, and block reduction independently.
3. Evaluation tests compare reconstructed images, distance maps, block maps,
   scores, and failure behavior.
4. Iterative tests compare the entire score history, final float quant field,
   exact raw quant field, and final reconstruction.
5. Pipeline tests compare the CPU and Metal paths through the supported
   integration boundary.

The corpus must include:

- Identical images, spatial and coefficient impulses, and deterministic random
  input.
- Flat, gradient, textured, high-contrast, and chromatic images.
- Dimensions below one block, exactly block-aligned, odd, narrowly shaped, and
  padded at the right and bottom edges.
- Encoding-tile, color-correlation-tile, and AC-group boundaries represented by
  the supported CPU subset.
- All seven supported AC strategies, including transpose pairs and mixed grids.
- Default and representative non-default Butteraugli and loop-filter options.
- Strided host inputs, device offsets, poisoned padding, invalid descriptors,
  and overflow attempts.

Metal tolerances must be established from differential evidence before tuning.
Stable fixtures should retain tight per-stage limits; broad sweeps may use a
separately documented boundary tolerance where float transform or quantization
discontinuities require it. A failing decision-level test must not be converted
to diagnostics or fixed by silently widening a tolerance.

## Performance methodology

Performance claims require rebuilt Release binaries and at least three
independent benchmark processes. Each process must warm every path, balance or
rotate CPU/resident/E2E order, and collect enough samples to report a median and
observed range. Sequential one-shot comparisons are diagnostic only.

Required workloads are:

- Synthetic 128x96 for continuity with the existing CPU baseline.
- An odd, heavily padded integration fixture.
- The 510x532 Flower crop used by Butteraugli validation.
- Padded 480p, 720p, and 1080p workloads.

Report at least:

- One-time preparation time.
- Resident GPU time for one evaluation.
- Full-E2E time for one evaluation.
- Default two-update iterative AQ time.
- Complete quantization-pipeline time with AC search reported separately.
- Upload, synchronization/readback, and final-output-download contributions.
- Persistent device bytes and peak scratch bytes.
- CPU/Metal crossover and run-to-run variation.

An isolated kernel speedup does not satisfy a milestone whose data still
crosses the host boundary between adjacent stages. Likewise, a resident GPU
speedup does not justify rollout if full-E2E AQ is slower or numerically changes
encoder decisions.

## Implementation order

Milestone 0 is the first implementation task. Milestone 1 then establishes the
shared substrate required by both reconstruction and Butteraugli. After
Milestone 2 fixes residency and lifetime contracts, reconstruction/filter work
(Milestones 3 and 4) and the standalone Butteraugli operation (Milestone 5) may
proceed independently. Milestone 6 joins them into the iterative loop;
Milestones 7 and 8 harden and qualify the result for rollout.

Removing libjxl from the production dependency graph is tracked independently
in [`butteraugli.md`](butteraugli.md). It may proceed in parallel and does not
block starting Metal infrastructure or AQ profiling.
