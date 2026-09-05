# CUDA backend support analysis

- Status: functionally complete for explicit CUDA selection; production
  qualification remains in progress
- Date: 2026-09-03
- Supported target: explicit CUDA encoding on non-macOS hosts with a supported
  NVIDIA device and CUDA toolkit

Implementation progress as of this revision:

- portable CPU-only, independently selectable Metal, and independently
  selectable CUDA builds are in place;
- each CUDA execution lane owns a non-blocking stream, device-scoped RAII
  allocations, stream-ordered checked transfers, event-backed submissions,
  deterministic failure injection, and backend/device ownership validation;
- all nine VarDCT transform shapes and the shared affine, convolution,
  symmetric-convolution, and maximum-reduction primitives pass real-device
  conformance on compute capability 8.6;
- all seven production AC-strategy candidate paths run on CUDA, support both
  packed and checked resident inputs, retain CPU-owned traversal and merging,
  and match CPU cost estimates within the existing Metal test tolerance;
- forced `maximum-throughput` encoding now keeps initial quantization,
  inverse Gaborish, initial CfL, DCT8 coefficient decisions, and quantized
  frame state on CUDA. Its odd-size/padded conformance fixture matches CPU
  initial-field tolerances and produces byte-identical codestreams;
- the prepared CUDA maximum-throughput operation is deterministic across
  reuse, performs no steady-state device allocations, and preserves
  caller-visible outputs across injected submission failure;
- prepared Butteraugli now runs the complete reference cache, psychoacoustic
  decomposition, Malta/L2 difference, masking, multiscale composition, and
  NaN-aware maximum reduction on CUDA. It uses one prepared allocation and
  one submission per comparison, is deterministic across reuse, and matches
  the CPU oracle on expanded, single-scale, multiscale, strided, identity,
  and non-default-option fixtures (worst observed absolute error
  `2.06e-5` on compute capability 8.6);
- exact-coefficient adaptive quantization now keeps CPU coefficient decisions
  authoritative and hands grouped, dequantized coefficients to CUDA at the
  inverse-transform boundary. CUDA owns mixed-strategy reconstruction, pixel
  scatter, Gaborish, all three EPF pass variants, opsin-to-linear conversion,
  Butteraugli block reduction, and normalized maximum-error reduction;
- the exact evaluator uses separate persistent and staging arenas, performs no
  steady-state device allocations, preserves final CPU frame/codestream bytes,
  supports odd padded source geometry and strategy reconfiguration, and
  invalidates atomically after submission, completion, numeric, or readback
  failure;
- fully resident AQ supports all seven production strategies, strategy-aware
  field adjustment, resident quantizer selection, cached forward transforms,
  final CfL, adjusted coefficient decisions, DC and AC coding,
  reconstruction, loop filtering, Butteraugli feedback, and resident
  maximum-error control. Its dependent Butteraugli policy is fused into one
  stream submission, so the two-evaluation field update stays on the device;
- the public workflow supplies CUDA with resident source/preprocessing,
  initial-quantization, initial-CfL, and AC-strategy-search results. It can
  materialize only the final frame and optional diagnostics rather than
  reading back and re-uploading the quant field between evaluations;
- exact-coefficient, fully-resident, throughput, and maximum-throughput CUDA
  modes work through C++, C, Rust, and the CLI. Distance, maximum-error, and
  target-size rate control are covered at the public-workflow boundary;
- forced CUDA was exercised on odd padded 1919x1079 input in all four modes and
  on odd padded 3839x2159 input in fully-resident mode. Every output was
  accepted by the pinned independent libjxl decoder and contained only finite
  decoded RGB samples. At 1919x1079, exact CUDA and CPU emitted the same
  codestream SHA-256, while requesting the optional final resident score did
  not change resident codestream bytes;
- native Rust builds now support macOS, Linux, and Windows, with an explicit
  `cuda` feature. The Windows CUDA test creates a real CUDA context and encodes
  through the public C API;
- the pinned conformance decoder is portable to Windows, including executable
  suffixes, a `clang-cl` workaround for the pinned revision, and local staging
  of its runtime DLLs. All 22 codestream fixtures and all 52 CUDA-enabled CTest
  tests pass in the measured Windows configuration; and
- automatic selection deliberately remains Metal-only until CUDA passes the
  cross-device qualification gates described below.

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

Before the CUDA work, the root [`CMakeLists.txt`](../CMakeLists.txt) rejected
every non-Apple platform, unconditionally enabled Objective-C++, and publicly
linked `gjxl_codestream` to `gjxl_metal`. This prevented even the CPU targets
from serving as a portable foundation.

The implemented build now provides independently controlled targets:

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

The concrete `CreateCudaBackend` factory accepts an explicit CUDA device
ordinal and validates it before constructing the backend; the higher-level
production resolver currently selects ordinal zero. Automatic selection should
remain conservative and be enabled per measured device class, geometry range,
and AQ mode. A failed one-time initialization must not permanently poison every
later encode unless the device itself is in an unrecoverable state.

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

Each implemented CUDA backend owns one non-blocking stream and serializes
stream submission and host transfers with a mutex. Production worker threads
are assigned to one of two persistent backend lanes, while explicitly created
backends retain one private lane. Prepared AQ and Butteraugli objects own
independent arenas, so work on different production lanes can overlap without
sharing per-image device buffers. Two lanes bound simultaneous GPU execution
without creating one stream per context. The cap follows local single-stream
profiling in which maximum-throughput encoding stopped improving between two
and four in-flight 1080p requests.

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

## CUDA architecture

### Backend object

`CudaBackend` implements the shared `GpuBackend` base and exposes the image
primitive, AC-strategy, prepared Butteraugli, and prepared AQ capabilities. It
owns:

- explicit CUDA device identity and runtime-device restoration;
- one private non-blocking stream and event-backed submissions;
- validated device allocations and launch limits;
- the correct FP32 DCT and functional-kernel dispatch paths; and
- deterministic failure-injection state for tests.

The functional backend has since gained shared private memory pools,
two production execution lanes, and profiled kernel specializations; see the
[optimization study](cuda-optimization-s1.md). Graph capture remains an
unimplemented optimization, not a correctness requirement.

`CudaBuffer` is an RAII `DeviceBuffer` containing an explicit device pointer
and allocation metadata. Operations validate both backend instance and device
ownership before using it.

`CudaSubmission` records completion with a CUDA event and caches one
translated status with `std::call_once`, preserving the concurrent `Wait()`
contract. Immediate launch errors are reported before a successful
submission is returned; asynchronous failures belong to the submission's
completion status.

### Device allocation policy

On devices with stream-ordered allocation support, CUDA backends use
`cudaMallocFromPoolAsync` and `cudaFreeAsync` on their existing streams.
Backends with the same device ordinal and retention policy share one
gjxl-private pool, including the two persistent production lanes. This does
not share per-image buffers or weaken backend-ownership validation, and
does not modify the application's default/current CUDA memory pool.

The default release threshold is `min(totalGlobalMem / 2, 4 GiB)`. It retains
unused memory for repeated encoding and is not a cap on live allocations.
Separate custom thresholds create separate pools; their retention can add
up. Zero requests release of unused storage at synchronization points.
Callers creating explicit C++ backends can choose a policy:

```cpp
gjxl::CudaBackendOptions options;
options.memory_pool_release_threshold_bytes = uint64_t{1} << 30;  // 1 GiB
// Alternatively, options.use_stream_ordered_allocation = false selects
// the legacy cudaMalloc/cudaFree policy instead of any memory pool.
std::unique_ptr<gjxl::GpuBackend> backend;
gjxl::Status status = gjxl::CreateCudaBackend(options, &backend);
if (!status.ok()) return status;
```

The public `gpu/cuda/cuda_backend.h` declares these options and a cache
release function. After quiescing encodes and releasing unneeded prepared
objects/buffers, a C++ caller can release unused memory while retaining
the backend lanes:

```cpp
gjxl::Status status = gjxl::TrimCudaDeviceMemory(0);
if (!status.ok()) return status;
```

Trimming synchronizes CUDA work on the selected device in the current
context, so it may wait for other users of that context. It leaves live
allocations valid and trims only gjxl pools. Concurrent encoding can
immediately grow the cache again. Pool ownership ends after the final
backend/buffer/submission state releases it; the registry holds only weak
references. Applications sharing a memory-constrained GPU can lower the
threshold, explicitly trim, or select the legacy allocator. There is no
automatic cross-pool out-of-memory recovery.

The feature is guarded for CUDA 11.2+, with a runtime capability check;
unsupported devices and older-toolkit builds use the legacy allocator.
Other pool setup errors are returned to the caller. Qualification currently
covers CUDA 11.8 on the RTX 3060 Laptop GPU, including forced legacy behavior,
not an older toolkit or an unsupported physical device. Retained cache use
and cold/warm timing are reported separately in the
[S32 study](cuda-optimization-s1.md#stream-ordered-allocation-follow-up-s32).
Release behavior follows NVIDIA's
[stream-ordered allocator documentation](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator).

### Prepared operations

Prepared CUDA AQ and Butteraugli operations consume the shared descriptors and
own persistent and staging device arenas. They submit through the backend
stream and keep stable device pointers across evaluations; that preserves the
option of later graph capture or pooling without making either part of the
correctness mechanism.

The operations expose the same ready, busy, and invalid state semantics as
Metal. Destruction waits for outstanding work before releasing device or host
staging memory.

Fully-resident AQ allocates host RGB reconstruction staging only when a
caller requests that diagnostic image. Encoding-only requests leave these
three host planes unallocated, saving `3 * source_width * source_height *
sizeof(float)` bytes of host storage (99.46 MB at 3839x2159). The GPU still
reconstructs RGB for perceptual evaluation; device arenas and required frame
readbacks are unchanged. The first diagnostic request allocates all three
staging planes before submitting work, and later requests reuse them.
Readback validation still precedes publication to the caller's image, so
failure does not expose a partial reconstruction.

Required fully-resident AC-coefficient host staging is allocated without
initial value initialization. The synchronous readback overwrites its full
extent before validation or frame assembly can access it. This avoids an
extra full-image host clear, but does not reduce its capacity or readback
volume. Final frame coefficients retain the same group/channel order and
zero-filled unused edge tails.

The fully-resident mixed-strategy forward DCT reads coding-image rectangles
directly, and its inverse DCT writes reconstruction-image rectangles directly.
Both use the existing factorized FP32 arithmetic and validated channel-major
anchor batches. The separate gathered-pixel and inverse-pixel arrays are no
longer allocated: this removes `2 * 3 * padded_width * padded_height *
sizeof(float)` bytes of device staging (199.07 MB at padded 4K), plus the
associated gather/scatter launches and device-memory round trips. Cached
forward coefficients remain available for repeated AQ evaluations. Exact
coefficient mode and the DCT8-only maximum-throughput path are unchanged.

The resident coefficient kernel computes its small forward/inverse DC bases
once per anchor block and reuses them across channels and samples. It retains
the original FP32 formulas, constants, and accumulation order. Direct access
to scale, bias, threshold, and sharpness values avoids dynamically indexed
thread-local array copies; CUDA 11.8 reports a 32-byte rather than 112-byte
stack frame, with 256 bytes of shared basis storage per block. The remaining
stack belongs to the cosine large-argument path, not a zero-stack claim.
This changes neither launch counts nor requested device allocation sizes or
host/device transfers. A guarded original-kernel oracle covers all seven
shapes, both quantization modes, changed-input reuse, and error paths.

Resident AC quantization, X/B prediction, and color restoration now share
one per-coefficient pass. Each thread keeps its reconstructed Y value and
completes X/B restoration before storing those channels. An explicit final
FMA preserves the unfused kernel's rounding boundary. The shared-basis,
DC-extraction, and pre-LLF barriers remain; three other block barriers and
intermediate reconstruction accesses are removed. A separate bounded entry
keeps four 256-thread blocks feasible on the qualified SM86 device without
changing the unbounded arithmetic/performance reference kernels.

### Math and kernel strategy

CUDA kernels use ordinary FP32 arithmetic and explicit decision-sensitive
compile settings. Global fast-math remains disabled.
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
and failure-atomic direct and public calls. Pinned libjxl accepts the small and
odd padded 1919x1079 maximum-throughput outputs, whose decoded samples are all
finite. Fully-resident 3839x2159 coverage additionally demonstrates that the
shared allocation and geometry path scales to odd padded 4K on the measured
6 GB device. Repeating the 4K maximum-throughput case and the full matrix on
other device classes remains qualification work rather than a functional gap.

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

Current progress: this phase is implemented for the forced exact track. The
CUDA evaluator validates and groups all seven production strategies, stages
the CPU frame's quantized AC and DC/LLF decisions into dequantized transform
batches, applies final CfL before upload, and starts device work at inverse
DCT. Reconstruction, scatter, Gaborish, EPF passes 0/1/2, opsin-to-linear,
prepared Butteraugli, 16-norm block feedback, and normalized maximum-error
feedback then remain on CUDA. An optional exact-linear handoff can skip the
reconstruction tail for Butteraugli-only callers.

Real-device differential coverage uses mixed transforms, an odd 257x17 source
padded to 264x24, non-default Gaborish/EPF/Butteraugli parameters, strided and
poisoned host outputs, both adaptive-quantization control modes, and injected
completion failure. The exact Butteraugli workflow stays within the existing
`2e-3` numerical contract (observed errors were below `3e-5` in block feedback
and below `4e-6` in reconstructed RGB on compute capability 8.6). The
maximum-error track stays within `2e-4`, preserves the CPU policy outcome, and
emits byte-identical final codestreams. Compute Sanitizer reports zero memory
errors. The odd padded 1919x1079 public workflow emits the same SHA-256
codestream as CPU and passes pinned-libjxl decode with finite output. Broader
corpus and cross-architecture gates remain part of production qualification
rather than functional exact-mode implementation.

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

Current progress: the direct resident evaluator is implemented for all seven
production transform strategies. It keeps the adjusted field, selected
quantizer and raw-quant grid, cached forward coefficients, final CfL, adjusted
coefficient decisions, inverse reconstruction, loop filters, color conversion,
and metric inputs in CUDA memory. Both Butteraugli and maximum-error control
produce valid frames and codestreams; iteration-zero mixed-strategy output is
byte-identical to the CPU oracle, while later iterations are tested against the
resident determinism contract because fixed final CfL intentionally differs
from ordinary CPU evaluation. Bounded and full materialization agree exactly,
caller output remains unchanged after injected completion failure, all 52
CUDA-enabled tests pass, and Compute Sanitizer reports zero memory errors on
compute capability 8.6.

The fused end state is now implemented. CUDA accepts the integrated pipeline's
resident source, inverse-Gaborish selection, initial quantization, initial CfL,
and AC-strategy-search handoff. `EvaluateResidentButteraugliPolicy` initializes,
evaluates, and updates the dependent quant field on the operation stream; the
ordinary two-evaluation public path no longer reads back and re-uploads the
field. Optional final scoring adds a diagnostic evaluation without changing
the frame or codestream.

All four modes pass the public-workflow contract on odd padded 1919x1079 input,
including independent decode and finite-sample checks. Fully-resident CUDA also
passes the same gates at odd padded 3839x2159 on a 6 GB compute-capability 8.6
device. This completes the functional Phase 4 scope. It does not by itself
qualify automatic selection across the NVIDIA product range.

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

Local qualification in this revision covers a Windows 11 host, CUDA 11.8,
MSVC 19.37, and an RTX 3060 Laptop GPU (compute capability 8.6, 6 GB). It
includes the complete CTest suite, CPU-only and CUDA builds, the public Rust
wrapper in both modes, Compute Sanitizer, deterministic and failure-injection
tests, all four public CUDA modes at odd padded 1080p, fully-resident CUDA at
odd padded 4K, pinned-decoder acceptance, finite decoded samples, an exact-mode
CPU/CUDA hash match, and scored/unscored resident hash stability.

The remaining qualification work is deliberately external to that single-host
evidence: automated real-GPU CI on at least one second architecture class,
Linux/toolkit-version coverage, concurrent public-context stress, measured
peak VRAM and pinned-host memory, explicit transfer accounting, repeatable cold
and warm performance baselines, and named photographic-corpus quality/size
comparisons.

Automatic CUDA selection should remain disabled until this phase produces a
documented device and workload qualification range.

## Build and use

CUDA is independent of Metal and is opt-in at configuration time:

```sh
cmake -S . -B build-cuda -G Ninja \
  -DGJXL_ENABLE_CUDA=ON \
  -DGJXL_ENABLE_METAL=OFF
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```

CMake uses its ordinary CUDA compiler and architecture discovery. Toolchain
files and CI should set `CMAKE_CUDA_ARCHITECTURES` explicitly when producing
portable artifacts instead of relying on the development machine's native
architecture.

The CLI requires explicit CUDA selection while qualification is conservative:

```sh
build-cuda/gjxl_encode input.pfm output.jxl \
  --backend cuda \
  --gpu-aq fully-resident \
  --distance 1.0
```

`--gpu-aq` also accepts `exact-coefficients`, `throughput`, and
`maximum-throughput`. The rate-control options are the same as for CPU and
Metal. The C API selects `GJXL_BACKEND_CUDA`, and Rust consumers enable the
safe crate's `cuda` feature:

```sh
cargo test --manifest-path rust/Cargo.toml --workspace --features cuda
```

On Windows, setting `CUDA_PATH` is the most direct way for the Rust build
script to find `cudart`; it also recognizes `CUDACXX` and
`CMAKE_CUDA_COMPILER`. Linux additionally falls back to
`/usr/local/cuda/lib64`.

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

The functional implementation has now completed Phases 0 through 4 in the
order proposed by the initial analysis: portable substrate, reusable
primitives, maximum-throughput vertical slice, exact coefficients, and finally
the fused resident pipeline. The next highest-value work is qualification, not
additional kernel surface.

Keep CUDA explicitly selected while collecting evidence on a second NVIDIA
architecture, Linux and newer toolkit combinations, a named photographic
corpus, concurrent contexts, transfer volume, memory pressure, and complete
workflow performance. Use exact mode as the byte-level regression oracle and
resident mode as the deterministic quality/performance track. Enable automatic
selection only for device, geometry, and mode ranges supported by recorded
data.

The architectural critique remains relevant after functional completion. In
particular, a shared immutable `PreparedAqPlan`, generated host/device ABI
checks, and backend-parameterized conformance tests would reduce long-term
Metal/CUDA drift. Those should be incremental refactors with unchanged output
contracts, not prerequisites for using the forced CUDA backend.
