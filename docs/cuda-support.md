# CUDA backend support analysis

- Status: implementation in progress on `feat/cuda`
- Date: 2026-09-03
- Initial target: a forced CUDA backend, followed by production qualification

Implementation progress as of this revision:

- portable CPU-only, independently selectable Metal, and independently
  selectable CUDA builds are in place;
- CUDA owns a non-blocking stream, device-scoped RAII allocations, synchronous
  checked transfers, event-backed submissions, deterministic failure
  injection, and backend/device ownership validation;
- all nine VarDCT transform shapes and the shared affine, convolution,
  symmetric-convolution, and maximum-reduction primitives pass real-device
  conformance on compute capability 8.6;
- forced `maximum-throughput` encoding now keeps initial quantization,
  inverse Gaborish, initial CfL, DCT8 coefficient decisions, and quantized
  frame state on CUDA. Its odd-size/padded conformance fixture matches CPU
  initial-field tolerances and produces byte-identical codestreams;
- the prepared CUDA maximum-throughput operation is deterministic across
  reuse, performs no steady-state device allocations, and preserves
  caller-visible outputs across injected submission failure;
- public C++, C, Rust, CLI, package-export, and diagnostic vocabulary now
  includes CUDA without changing existing C enum values; and
- automatic selection deliberately remains Metal-only until CUDA passes the
  full workflow and qualification gates described below.

## Executive finding

A CUDA backend is technically feasible, and the existing GPU architecture was
clearly designed with a second backend in mind. `BackendKind` already contains
`kCuda`, while buffers, typed device images, submissions, transforms, image
primitives, AC-strategy evaluation, prepared Butteraugli, and prepared adaptive
quantization all have backend-neutral contracts.

The work is nevertheless substantially larger than translating the Metal DCT
kernels. The fully resident Metal path now implements almost the complete
image-analysis half of the encoder: initial quantization, chroma-from-luma,
AC-strategy candidate evaluation, quantizer construction, coefficient coding,
reconstruction, loop filtering, opsin conversion, Butteraugli, metric
reduction, and the dependent AQ update policy. The Metal host, header, and
shader implementation is roughly 20,000 lines and exposes about one hundred
distinct functional kernel paths once alternative DCT implementations are
excluded.

The expected difficulty therefore depends on the requested endpoint:

| Endpoint | Difficulty | Cumulative one-engineer estimate |
| --- | --- | ---: |
| CUDA buffers, submissions, copies, and smoke tests | Moderate | 3-5 weeks |
| DCT and reusable primitive conformance | Moderate | 5-8 weeks |
| Forced `maximum-throughput` CUDA encoding | Moderately hard | 8-12 weeks |
| Exact-coefficient CUDA workflow | Hard | 14-20 weeks |
| Fully resident, production-qualified parity | Very hard | 26-40 weeks |

These are engineering-week estimates for an experienced CUDA/C++ engineer
with access to codec expertise. Full parity is likely a six-to-nine-month
single-engineer project, or approximately three to five calendar months for
two strong engineers where the work can be parallelized. Correctness,
architecture, and qualification contain enough serial dependencies that team
size will not divide calendar time linearly.

## Existing foundation

The shared GPU substrate is a strong starting point:

- [`GpuBackend`](../src/gpu/backend.h) owns backend identity, allocation and
  submission accounting, synchronous host transfers, and packed transform
  operations.
- [`DeviceBuffer`](../src/gpu/buffer.h) records both backend kind and backend
  instance identity. It already reserves `BackendKind::kCuda`.
- [`DevicePlaneView`](../src/gpu/image.h) and `DeviceImage3` provide typed,
  strided, non-owning device views with checked byte-range and overlap logic.
- [`GpuSubmission`](../src/gpu/submission.h) has explicit completion semantics.
  Repeated and concurrent waits must return one cached status, and destruction
  does not implicitly wait.
- [`DeviceScratchArena`](../src/gpu/scratch.h) provides checked aligned
  suballocation from one reusable device allocation.
- [`GpuImagePrimitives`](../src/gpu/ops/primitives.h) defines a small optional
  capability for affine operations, separable convolution, the codec's
  symmetric 5x5 convolution, and maximum reduction.
- [`GpuAcStrategyEvaluation`](../src/gpu/ops/ac_strategy.h) leaves candidate
  traversal, non-overlap merging, and deterministic tie-breaking on the CPU
  while evaluating expensive same-strategy batches on the device.
- [`PreparedDeviceButteraugli`](../src/gpu/ops/butteraugli.h) fixes reference
  preparation, comparison, readback, lifetime, and failure behavior without
  exposing Metal types.
- [`PreparedAqEvaluation`](../src/gpu/ops/aq_evaluation.h) defines resident
  inputs, optional materialization, reconfiguration, memory accounting, and
  failure-atomic output behavior.

This separation means a CUDA backend does not require redesigning the encoder
or public GPU operation contracts from scratch. The native CPU implementation
also remains a readable executable specification and supplies numerical and
decision-level oracles for CUDA tests.

## Functional scope of a complete backend

The current Metal implementation contains the following broad kernel families:

| Family | Current Metal scope | CUDA requirement |
| --- | --- | --- |
| DCT | Nine shapes, forward and inverse, three implementations | One correct implementation per required shape and direction initially |
| AC strategy | Gather, fused forward transforms, residual/inverse, and cost | Seven production strategy paths plus CPU-compatible cost output |
| Initial AQ | Gradient, erosion, modulation, sorting, quantizer selection | Required for resident and maximum-throughput modes |
| Reconstruction | Quant selection, coefficient coding, inverse transforms, scatter | Required for exact and resident perceptual modes |
| Postprocessing | Gaborish, EPF, and opsin-to-linear conversion | Required for the perceptual tail |
| Butteraugli | Prepared reference, psychoacoustic stages, Malta, masks, final map, reduction | Required for exact and resident Butteraugli control |
| AQ policy | Block reduction, maximum-error reduction, dependent field updates | Required for complete resident rate control |

There are four intentionally distinct behavior tracks, documented in
[`metal-encoding-performance.md`](metal-encoding-performance.md):

- `exact-coefficients` preserves CPU raw quantization, encoder-frame decisions,
  and codestream bytes. The GPU begins at the reconstruction/perceptual tail.
- `fully-resident` is the production Metal default. It is deterministic for a
  fixed backend but is not required to be byte-identical to CPU.
- `throughput` shares the resident implementation but may use a reduced
  diagnostic or iteration policy in non-encoding APIs.
- `maximum-throughput` uses DCT8 only, constructs a frame directly from the
  adjusted initial field, and omits reconstruction and perceptual scoring.

CUDA should preserve these explicit contracts rather than silently changing
quality or compatibility behavior under one generic GPU label.

## Metal design to retain

### Prepared, resident operations

The most important Metal design choice is to prepare immutable frame state once
and keep large images and intermediates resident. Static images, strategies,
transform metadata, quantization tables, Butteraugli reference data, and
scratch are allocated or uploaded during preparation. Repeated evaluation then
performs no device allocation.

This maps well to CUDA device memory, stream-ordered execution, and eventually
CUDA Graphs. The no-steady-state-allocation property should be a CUDA acceptance
criterion, not merely an optimization goal.

### Explicit ownership and lifetime rules

Buffers are tied to one backend instance, device views are non-owning, and
prepared operations retain any outstanding submission before reusing or
destroying scratch. These rules prevent a large class of cross-device,
use-after-free, and asynchronous lifetime failures.

The CUDA implementation should preserve the same ownership checks even though
raw CUDA pointers would otherwise make it easy to bypass them.

### Failure-atomic output

Metal validates complete descriptors before submission. Invalid requests
submit no work. Upload, submission, execution, numeric, or readback failures do
not partially commit caller-visible output. Operational failures invalidate the
prepared state so it cannot silently reuse corrupted storage.

The CUDA implementation should distinguish:

- argument or compatibility rejection before launch;
- immediate launch/submission failure;
- asynchronous execution failure discovered at event synchronization;
- device-side numeric error flags; and
- host staging or readback failure.

All paths must retain the existing output-atomicity contract.

### Exact and resident acceptance tracks

The exact-coefficient path is a useful compatibility boundary. It keeps
threshold-sensitive coefficient decisions on the CPU and uses the GPU for the
more numerically tolerant inverse-transform and perceptual tail. The resident
path is allowed to make backend-specific floating-point decisions but is held
to determinism, decodability, size, and decoded-quality gates.

CUDA should adopt the same separation. Requiring CUDA and Metal resident paths
to emit identical bytes would constrain both backends to their least natural
arithmetic and execution order.

### CPU-owned deterministic search policy

AC-strategy traversal, non-overlap merging, and tie-breaking remain on the CPU.
The device receives large candidate batches and returns scalar costs. This
keeps policy readable and makes backend comparison straightforward while still
moving the expensive work.

Moving the merge or traversal to CUDA should require profiling evidence and a
new deterministic contract; it should not be part of the initial port.

### Up-front pipeline validation

Metal loads the embedded library, creates required pipeline states, and checks
launch limits when the backend is constructed. A forced backend therefore
fails early rather than halfway through an encode.

CUDA should similarly choose a device, validate its required features and
memory limits, establish kernel variants, and construct any pools or reusable
events before reporting the backend as available.

### Differential tests and performance discipline

The Metal work has strong stage-level CPU differential tests, poisoned-output
checks, allocation and submission counters, concurrency tests, failure
injection, GPU timestamps, and recorded end-to-end performance gates. It also
measures the complete public encoder rather than treating a fast leaf kernel as
an encoder speedup.

Those practices should become shared GPU conformance and qualification suites
that both backends run.

## Metal design not to copy blindly

### Platform and target coupling

The root [`CMakeLists.txt`](../CMakeLists.txt) currently rejects every non-Apple
platform, unconditionally enables Objective-C++, and publicly links
`gjxl_codestream` to `gjxl_metal`. This prevents even the CPU targets from
serving as a portable foundation.

The build should instead provide independently controlled targets:

- `GJXL_ENABLE_METAL`, available only on Apple platforms;
- `GJXL_ENABLE_CUDA`, available when a supported CUDA toolchain is selected;
- a portable CPU-only build with neither GPU backend; and
- an internal backend resolver that depends only on the enabled concrete
  factories.

Metal frameworks, Objective-C++, `metal-cpp`, shader compilation, and metallib
embedding must remain inside the conditional Metal branch. CUDA compilation,
the CUDA runtime, architecture selection, and generated device objects must
remain inside a separate `gjxl_cuda` target.

### Original Metal-specific public vocabulary

At the time of the initial analysis, [`VarDctEncodingOptions`](../src/codestream/workflow.h)
exposed `kMetal`, a Metal-only execution summary, and a Metal-named AQ field.
The C API and Rust wrapper likewise exposed only automatic, CPU, and Metal
backend variants.

The first implementation checkpoints completed the required vocabulary work:

- `VarDctBackendPreference::kCuda`;
- `VarDctExecutionBackend::kCuda`;
- `GJXL_BACKEND_CUDA`, appended without renumbering existing C values;
- a Rust `Backend::Cuda` variant;
- `--backend cuda` in command-line tools; and
- backend-neutral `gpu_aq_mode` terminology.

Because the project is version `0.0.1`, the C++ field was renamed directly
rather than retaining two independently writable aggregate members. The CLI
accepts legacy `--metal-aq` as an alias for the canonical `--gpu-aq` spelling.

### Brittle device qualification and process-global caching

Automatic Metal selection currently recognizes one exact device-name string
and stores one system-default backend behind a process-global `std::once_flag`.
This is too narrow for NVIDIA's range of devices and prevents recovery from a
transient initialization failure.

The internal backend descriptor should report at least:

- backend kind and stable device identity;
- device ordinal;
- available memory and relevant execution limits;
- supported operation capabilities and AQ modes;
- supported profiling capabilities; and
- qualification state for automatic selection.

Forced selection should accept an explicit CUDA device ordinal. Automatic
selection should initially remain conservative and be enabled per measured
device class, geometry range, and AQ mode. A failed one-time initialization
must not permanently poison every later encode unless the device itself is in
an unrecoverable state.

### Unified-memory assumptions

[`MetalBackend`](../src/gpu/metal/metal_backend.cpp) allocates every buffer with
`MTL::ResourceStorageModeShared`. Host copies are direct `memcpy` operations
over `MTL::Buffer::contents()`, and final frame assembly borrows completed
buffer ranges synchronously.

That is appropriate for Apple Silicon but should not be reproduced with CUDA
managed memory. On a discrete NVIDIA GPU it risks unpredictable page migration
and disguises expensive transfers.

CUDA should use:

- explicit device allocations;
- pinned host upload and readback staging;
- asynchronous copies associated with the operation stream;
- reusable allocation pools after correctness is stable; and
- explicit transfer-byte and transfer-time accounting.

The synchronous `GpuBackend` copy methods are sufficient for the first correct
implementation. Prepared CUDA operations may then manage pinned staging and
asynchronous copies internally without prematurely broadening the public base
interface.

### Final coefficient handoff

Metal can synchronously consume completed shared storage while converting its
strategy-batch-major coefficients into the CPU serializer's group-major
layout. CUDA must transfer those coefficients across the device boundary.

The preferred eventual design is:

1. retain strategy-batch-major storage for CUDA coefficient kernels;
2. pack the final result into the serializer's group-major representation on
   the GPU;
3. copy one contiguous result into pinned, candidate-owned host storage; and
4. commit the completed `VarDctEncoderFrame` only after successful event
   completion, readback, and validation.

Writing directly to mapped host memory is unlikely to be a good default for
the coefficient kernels. Explicit device packing followed by a large
contiguous asynchronous copy should be the baseline to measure.

### Monolithic prepared AQ state

`MetalPreparedAqEvaluation` has accumulated geometry planning, validation,
resource layout, host staging, Metal dispatch encoding, profiling, exact
handoffs, several resident modes, frame assembly, diagnostics, failure
injection, and a mutable operation state machine. Its many `frame_only_*`,
`resident_*`, and exact-mode booleans encode combinations that are increasingly
difficult to reason about.

The CUDA implementation should not duplicate this class wholesale. First
extract backend-independent immutable planning:

- validated source, coding, block, tile, and filter geometry;
- canonical strategy anchors and per-strategy batches;
- coefficient offsets and final transform layouts;
- quantization-table and constant preparation;
- persistent, staging, and scratch capacity requirements;
- selected execution features and valid feature combinations; and
- requested output/readback materialization.

A shared `PreparedAqPlan` can be consumed by separate Metal and CUDA executors.
Native resource ownership, launch encoding, profiling, and synchronization
should remain backend-specific.

The preparation flags should also be grouped into a structured feature or mode
description rather than extended with more interacting booleans.

### Manually duplicated host/device ABI

AQ and Butteraugli parameter structures are declared independently in C++ and
Metal source. Host-side size assertions catch some mistakes, but they cannot
prove that the shader declaration retained the same field offsets and
semantics. Quantization formulas and constants also have separate CPU and Metal
implementations.

Adding a third handwritten CUDA copy would increase drift risk. Prefer one of:

- generating host, MSL, and CUDA POD declarations from a small schema;
- a carefully restricted shared header where language compatibility permits;
  or
- generated offset/size manifests plus exhaustive ABI tests.

Decision-sensitive formulas should use shared generated test vectors even if
the source syntax cannot be shared directly.

### Apple-specific kernel structure

The production Metal configuration selects SIMD-group matrix DCTs for the
seven AQ strategies. CUDA warp geometry, register pressure, shared-memory bank
behavior, and preferred block sizes differ. The CUDA port should preserve the
math and coefficient layout, not Metal launch geometry or SIMD-group
implementation details.

The recommended DCT progression is:

1. simple scalar or matrix FP32 kernels as the correctness oracle;
2. separable shared-memory transforms;
3. factored radix-2 variants where they win end to end; and
4. warp-specialized variants selected by profiling.

Tensor-core arithmetic should not be used in decision-sensitive paths until
its output and quality behavior have an explicit acceptance contract.

### Submission and transfer model

The common submission abstraction maps naturally to a CUDA event, but the base
backend has no explicit stream, event-dependency, or asynchronous-copy API. A
single CUDA stream is sufficient for the first correct implementation and
preserves ordering, but it may serialize otherwise independent prepared
objects.

The initial CUDA backend should own a default stream and let prepared AQ or
Butteraugli objects own or lease private streams. Independent prepared objects
must be thread-safe even before true overlap is optimized. A bounded stream
pool can be introduced after concurrency and memory-pressure profiling.

The resident policy contains many launches with stable allocations and mostly
stable geometry. Once the ordinary stream implementation is correct, it is a
strong candidate for CUDA Graph capture. Graphs should optimize an established
execution plan rather than become the initial correctness mechanism.

### Metal-only tests and absent native CI

Several important tests instantiate Metal directly even though most of their
contracts are backend-neutral. They should be parameterized through a small
backend test factory. Metal-specific pipeline selection, profiling, and shader
capture tests should remain concrete.

The repository currently has Rust workflows but no comprehensive native C++ or
GPU CI. CUDA support should not be considered production-ready without
automated CPU-only builds and real-NVIDIA functional testing.

## Recommended CUDA architecture

### Backend object

`CudaBackend` should implement the shared `GpuBackend` base plus only the
optional capabilities that are complete and qualified at each milestone. It
should own:

- CUDA device identity and context policy;
- one default stream and reusable event resources;
- allocation and optional memory-pool state;
- selected DCT/kernel variants;
- profiling capability information; and
- deterministic failure-injection state for tests.

`CudaBuffer` should be an RAII `DeviceBuffer` containing an explicit device
pointer and allocation metadata. It must validate both backend instance and
device ownership before every operation.

`CudaSubmission` should record completion with a CUDA event and cache one
translated status with `std::call_once`, preserving the concurrent `Wait()`
contract. Immediate launch errors should be reported before a successful
submission is returned; asynchronous failures belong to the submission's
completion status.

### Prepared operations

Prepared CUDA AQ and Butteraugli operations should consume shared immutable
plans but own their native streams, events, device allocations, pinned staging,
and graph instances. Stable pointers should be preferred so graph capture and
memory reuse remain possible later.

The operation should expose the same ready, busy, and invalid state semantics
as Metal. Destruction must wait for outstanding work before releasing device or
host staging memory.

### Math and kernel strategy

CUDA kernels should initially use ordinary FP32 arithmetic and explicit
decision-sensitive compile settings. Global fast-math should remain disabled.
Maximum reduction may use a standard CUDA reduction implementation because
finite maximum is order-independent; sum-, norm-, and threshold-sensitive
operations require fixed tolerances and CPU differential tests.

Metal source is a valuable description of fusion and dataflow, but the CPU
implementation remains the semantic oracle. Where Metal and CPU differ, the
documented exact/resident contract determines which result CUDA must follow.

## Implementation sequence

### Phase 0: portable build and backend-neutral workflow

1. Remove the unconditional non-Apple CMake failure.
2. Make Objective-C++ and Metal dependencies conditional.
3. Add a CPU-only Windows and Linux configuration.
4. Add optional `gjxl_cuda` build plumbing without functional kernels.
5. Generalize backend enums, AQ naming, workflow selection, summaries, CLI,
   C API, and Rust API.
6. Introduce an internal backend descriptor and resolver.
7. Preserve all existing Metal behavior and numerical output.

Exit criterion: CPU builds and tests run without Metal; macOS Metal behavior is
unchanged; an unavailable forced CUDA request returns the correct status.

### Phase 1: shared conformance suite and CUDA substrate

1. Parameterize buffer, view, ownership, copy, submission, concurrency,
   primitive, and transform tests.
2. Implement CUDA status translation, device selection, buffers, streams,
   events, and submissions.
3. Implement synchronous base transfers and pinned staging helpers.
4. Port affine, convolution, and maximum-reduction primitives.
5. Implement simple forward and inverse DCTs.
6. Add failure injection and allocation/submission accounting.

Exit criterion: CUDA passes the generic substrate and transform tests, invalid
descriptors submit no work, and concurrent waits return identical status.

### Phase 2: maximum-throughput vertical slice

The existing maximum-throughput mode is the smallest clean end-to-end CUDA
milestone. It requires prepared AQ but intentionally omits AC search,
reconstruction, Butteraugli, and dependent perceptual updates.

Port:

- inverse Gaborish preprocessing where enabled;
- initial CfL;
- initial quant gradient, erosion, modulation, and selection;
- strategy-aware field adjustment;
- resident quantizer and raw-quant construction;
- DCT8 coefficient coding; and
- final frame readback and assembly.

Exit criterion: forced CUDA produces deterministic, independently decodable
codestreams for odd, padded, small, 1080p, and 4K inputs; no score history is
reported; repeated warm execution performs no device allocation.

Current progress: the vertical slice is implemented and exposed through an
explicitly forced CUDA workflow. Real-device tests cover odd padded geometry,
CPU initial-field tolerances, byte-identical CPU frame serialization,
deterministic prepared-operation reuse, zero steady-state device allocations,
and failure-atomic direct and public calls. Small/1080p/4K coverage and an
independent decoder gate remain before this phase's qualification exit
criterion is complete.

This is a vertical architecture and transfer proof, not qualification of the
default quality path.

### Phase 3: exact-coefficient workflow

Port and validate:

- all production transform shapes;
- AC-strategy candidate evaluation;
- inverse coefficient reconstruction and pixel scatter;
- Gaborish and EPF postprocessing;
- opsin-to-linear conversion;
- prepared Butteraugli and score reduction; and
- maximum-error reduction where required.

Exit criterion: the exact track preserves CPU raw quantization, encoder frame,
codestream bytes, control outcome, and existing numerical tolerances.

### Phase 4: fully resident AQ

Port:

- resident initial quantization and AC-search handoff;
- final resident CfL;
- strategy-aware quant adjustment and quantizer selection;
- forward coefficient caching;
- resident coefficient decisions;
- dependent Butteraugli policy updates;
- final-frame-only materialization;
- maximum-error resident control; and
- optional diagnostic materialization.

Only after ordinary stream execution is correct should kernel fusion, CUDA
Graphs, allocation pools, stream pooling, or architecture-specific variants be
considered.

Exit criterion: all four GPU modes satisfy their distinct contracts; resident
results are deterministic for a fixed CUDA backend, independently decodable,
finite after decoding, and within established size and perceptual-quality
gates.

### Phase 5: production qualification

Add:

- real-GPU CI on at least two NVIDIA architecture classes;
- CPU-only Windows, Linux, and macOS builds;
- Compute Sanitizer coverage in a scheduled job;
- toolkit and host-compiler compatibility matrices;
- device-loss, out-of-memory, launch, completion, numeric, and readback tests;
- concurrent prepared operations and concurrent public contexts;
- warm and cold 1080p/4K benchmarks;
- H2D/D2H byte and timing accounting;
- peak VRAM and host-pinned-memory accounting;
- exact-mode hashes and resident-mode determinism checks;
- pinned `djxl` acceptance and decoded-pixel checks; and
- named-corpus size and Butteraugli comparisons.

Automatic CUDA selection should remain disabled until this phase produces a
documented device and workload qualification range.

## Major risks

### Numerical decision drift

CUDA contraction, transcendentals, and reduction order can move values across
quantization or search thresholds. The exact path limits this risk; resident
CUDA requires its own deterministic quality gates. Global fast-math is the
largest avoidable early risk.

### Transfer-bound performance

Metal's shared-memory handoff does not predict discrete-GPU performance. A CUDA
kernel can be faster while the public encoder is slower because input upload,
final coefficient readback, CPU repacking, or synchronization dominates.
Every performance claim must use the complete in-memory encode boundary.

### Memory capacity

Prepared AQ owns large persistent and staging arenas, and earlier Metal
measurements reached hundreds of megabytes at 1080p. CUDA must measure peak
device memory separately from host memory, reject impossible preparations
atomically, and account for concurrent encodes. Pooling should follow, not
precede, a verified resource plan.

### Third implementation drift

CPU and Metal already duplicate some numerical logic and parameter layouts.
CUDA increases the maintenance burden unless shared planning, generated
constants, ABI checks, and differential fixtures are established early.

### Overfitting one GPU

The current Metal automatic path is deliberately qualified for one named Apple
GPU. CUDA spans a much wider device and memory range. Kernel selection and
automatic enablement must be based on explicit properties and measured device
classes, not one development machine's name or timings.

### Premature optimization

Tensor cores, managed memory, graph capture, multi-stream scheduling, GPU
entropy coding, and a device-native serializer are all plausible future work.
None should block the first correct backend. The existing Metal roadmap's
evidence-driven, complete-workflow measurement discipline should determine
which of them is justified.

## Acceptance contract

A production CUDA backend should meet all of the following:

- CPU-only builds remain independent of CUDA and Metal.
- Forced CUDA fails early and clearly when unavailable or unsupported.
- Automatic selection falls back only before pipeline execution; runtime CUDA
  failures are not silently retried on CPU.
- Exact mode preserves CPU decision and byte-level contracts.
- Resident modes are deterministic per backend and meet explicit size and
  decoded-quality gates.
- Invalid inputs allocate and submit no work.
- Operational failure commits no caller-visible partial output.
- Repeated prepared evaluation performs zero steady-state device allocations.
- Independent prepared objects are thread-safe and may progress without
  sharing mutable scratch.
- Final codestreams are accepted by the pinned independent decoder.
- Performance is reported at the complete public encode boundary, alongside
  transfer, synchronization, memory, and output-quality evidence.

## Recommendation

The first implementation effort should combine Phase 0 and Phase 1: make the
build and workflow genuinely backend-neutral, factor the conformance tests, and
land the CUDA runtime substrate with simple correct transforms. That work is
valuable even if later performance results change the scope.

The forced maximum-throughput path should then serve as the first go/no-go
milestone. It proves end-to-end CUDA ownership, preprocessing, coefficient
generation, transfer, frame assembly, and serialization without first
committing to the much larger Butteraugli and resident reconstruction port.

If that slice shows a credible complete-workflow result, proceed through the
exact-coefficient path before fully resident AQ. This orders the work from the
strongest correctness oracle to the most backend-specific performance path and
avoids cloning the current Metal monolith before the common architecture has
been improved.
