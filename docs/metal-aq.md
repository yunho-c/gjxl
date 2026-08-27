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

The public GPU substrate now supports buffer allocation, host transfers,
strided device images, reusable scratch, per-command submission handles,
batched transforms, a fixed optional image-primitive capability, and an
optional prepared AQ operation. The prepared Metal path now completes
coefficient coding, reconstruction, loop filtering, and cropped opsin-to-linear
conversion while keeping both reconstructed images resident. Butteraugli, AQ
block reduction, and production policy integration remain unavailable.

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

Butteraugli Milestone 7 now fixes the backend-neutral preparation, synchronous
comparison, device-output, readback, validation, and lifetime contracts. Its
test-only staged path exercises this substrate but contains no Butteraugli
kernels. This does not complete the Metal AQ Milestone 5 integration gate.

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
submitted. The prepared object is non-copyable, is bound to the backend that
created it, and exposes no partially initialized state. Preparation failure
returns no object.

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

### Synchronization, concurrency, and failure

The initial operation is host-synchronous at the CPU policy boundary. One call
encodes one Metal submission, waits for its bounded block-map and score
readback, and returns only after those values are available. The implementation
must not hide additional submissions or full-resolution diagnostic readbacks.

A prepared object is not concurrently reentrant because its evaluations reuse
mutable scratch. Independent prepared objects may execute concurrently and
must not share mutable storage. Geometry, strategy, option, or backend mismatch
is rejected before submission and leaves the prepared object usable.

Invalid input submits no work. Allocation or validation failure during
preparation returns no object. Submission, command-buffer, synchronization, or
readback failure commits no caller-visible CPU output and invalidates the
prepared object; the caller must prepare a new object before retrying. Backend
unavailability remains an explicit status handled by CPU fallback above the
operation rather than by silently switching implementations inside it.

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

Every successful non-empty transform or image-primitive submission returns an
owning `GpuSubmission`. `Wait()` is thread-safe, idempotent, and returns its
cached completion status. The handle retains the native objects required to
wait and may outlive the backend; destroying the handle does not implicitly
wait. Failed validation or command-buffer creation resets the output handle and
submits no work. Empty image-primitive sequences are invalid. Zero-transform
batches retain their established no-op contract and return success with a null
handle.

Prepared operations must retain all outstanding submissions and wait before
destroying, resizing, or reusing referenced scratch. Backend allocation and
submission counters are safe to read while independent submissions are being
created concurrently.

The shader build must compile and link multiple controlled in-tree `.metal`
sources into the existing gjxl metallib. Backend creation binds exact exported
function names and validates required pipeline state up front; runtime
reflection or prefix-based discovery is not part of the contract. Coherent
codec operations belong above the backend and must not expand `GpuBackend` into
one virtual method per private leaf kernel. The fixed affine,
separable-convolution, and maximum-reduction command set is exposed through the
optional `GpuImagePrimitives` capability instead of the core backend interface.
Prepared Butteraugli and AQ operations will own their private kernel sequences
rather than extending that command variant.

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

Milestones 0 through 4 are complete. Milestones 5 through 8 remain pending and
must not be treated as complete until their stated exit criteria pass.

### 0. Refresh the AQ baseline and freeze the evaluation contract — complete (2026-08-26)

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

The implementation adds a diagnostic-only internal profiling entry point that
runs the production `FindBestQuantization` implementation. It atomically
reports loop setup, every evaluation, quant-field updates, and output commit.
Each evaluation records field construction, coefficient coding,
reconstruction, loop filters, color conversion, Butteraugli, and block
reduction. The ordinary API remains unchanged and does not read the clock.

The benchmark now rotates seven phases and supports the required padded
workloads. Measurements below used an Apple M4 Pro with 14 CPU cores and 48 GB
RAM, macOS 15.6, AppleClang 17, and a Release build of this Milestone 0 tree
based on `bff6146`. Each of three independent processes performed three warmup
rotations and five measured rotations. Reported medians are the median of the
three run medians; parenthesized ranges span all 15 measured samples.

| Workload (source -> coding) | One evaluation (ms) | Two-update AQ (ms) | Complete pipeline (ms) |
| --- | ---: | ---: | ---: |
| Synthetic 128x96 -> 128x96 | 13.826 (13.347–14.260) | 40.882 (40.057–42.504) | 53.195 (51.746–54.278) |
| Odd 121x89 -> 128x96 | 12.289 (11.965–12.627) | 36.525 (35.877–37.218) | 49.332 (48.855–51.093) |
| Flower 510x532 -> 512x536 | 298.712 (293.901–309.105) | 898.888 (886.225–917.357) | 1177.842 (1163.011–1235.693) |
| Padded 480p 854x479 -> 856x480 | 454.288 (448.320–462.507) | 1366.242 (1343.159–1404.186) | 1734.224 (1711.211–1782.266) |
| Padded 720p 1279x719 -> 1280x720 | 1011.521 (992.109–1026.342) | 3031.718 (2996.328–3073.570) | 3844.369 (3800.606–3899.127) |
| Padded 1080p 1919x1079 -> 1920x1080 | 2285.346 (2255.584–2315.131) | 6863.434 (6773.744–6885.900) | 8713.679 (8577.378–8746.698) |

The complete-pipeline totals include the following independently measured CPU
preprocessing and search medians:

| Workload | Initial quant | Gaborish | Initial CfL | AC search |
| --- | ---: | ---: | ---: | ---: |
| Synthetic 128x96 | 0.386 | 0.647 | 1.237 | 9.377 |
| Odd 121x89 | 0.377 | 0.660 | 1.517 | 10.231 |
| Flower 510x532 | 8.102 | 14.030 | 29.192 | 222.521 |
| Padded 480p | 12.279 | 22.012 | 39.640 | 290.632 |
| Padded 720p | 26.991 | 47.929 | 91.918 | 646.087 |
| Padded 1080p | 60.883 | 107.115 | 203.984 | 1455.114 |

Internal one-evaluation medians in milliseconds identify the throughput-heavy
boundary directly:

| Workload | Field construction | Coefficient coding | Reconstruction | Loop filters | Color conversion | Butteraugli | Block reduction | Evaluation total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic 128x96 | 0.685 | 0.752 | 0.844 | 2.747 | 0.109 | 8.629 | 0.023 | 13.812 |
| Odd 121x89 | 0.582 | 0.645 | 0.740 | 2.723 | 0.094 | 7.446 | 0.021 | 12.272 |
| Flower 510x532 | 14.989 | 16.292 | 19.555 | 58.700 | 2.227 | 185.830 | 0.478 | 297.453 |
| Padded 480p | 23.965 | 25.547 | 30.634 | 89.698 | 3.974 | 282.372 | 0.717 | 452.004 |
| Padded 720p | 53.657 | 57.290 | 69.257 | 190.811 | 7.960 | 626.878 | 1.741 | 1004.808 |
| Padded 1080p | 122.608 | 130.287 | 153.798 | 425.433 | 17.539 | 1413.317 | 4.124 | 2279.490 |

Butteraugli is approximately 62% of evaluation time throughout the measured
range, with loop filtering the next largest stage. Field construction remains
on the CPU initially because it is a smaller stage and produces only
block/tile-resolution iteration inputs; its transfer cost remains part of the
future full-E2E gate.

Peak memory was measured by running one workload per process under
`/usr/bin/time -l`. These are process-wide maximum resident-set values, not
allocator-tracked native scratch bytes, and include benchmark inputs, outputs,
libraries, and allocator retention.

| Workload | Median peak RSS | Three-process range |
| --- | ---: | ---: |
| Synthetic 128x96 | 27.5 MiB | 22.0–43.1 MiB |
| Odd 121x89 | 27.2 MiB | 23.4–31.0 MiB |
| Flower 510x532 | 468.5 MiB | 423.7–478.4 MiB |
| Padded 480p | 654.9 MiB | 652.9–696.3 MiB |
| Padded 720p | 973.6 MiB | 971.1–1046.7 MiB |
| Padded 1080p | 1992.6 MiB | 1952.5–2074.8 MiB |

Reproduce one process with `just quantization-benchmark`, or select one
workload explicitly, for example:

```sh
just quantization-benchmark padded_1080p 5 3
/usr/bin/time -l build/release/gjxl_quantization_benchmark \
  --workload padded_1080p --samples 5 --warmups 3
```

The measured stage split and memory scaling validate the documented boundary:
the GPU effort starts with shared residency/submission/scratch infrastructure,
then connects reconstruction, filtering, and prepared Butteraugli without
full-resolution host round trips. These Milestone 0 measurements include no
Milestone 1 GPU work.

### 1. Add reusable device-image and submission infrastructure — complete (2026-08-26)

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

The backend-neutral substrate now provides typed mutable and const device
planes, explicit three-plane images, checked byte-range conversion, backend
instance ownership, overlap checks, and a reusable aligned scratch arena.
Buffers from a different backend instance are rejected even when both backends
select the same physical device. Scratch growth is preparation-only; planned
slices remain non-owning and require the arena to outlive submitted work.

`GpuBackend` now contains only the shared buffer, copy, and transform surface.
Backends opt into the fixed `ImagePrimitiveCommand` set through
`GpuImagePrimitives`; a lightweight range-test backend therefore has no image
primitive dependency or stub method. `SubmitImagePrimitiveSequence` validates
a complete non-empty span before creating a command buffer, then encodes it in
order through one compute encoder and one commit.

Transforms and primitive sequences both return per-command `GpuSubmission`
handles, replacing backend-wide latest-command synchronization. Repeated and
concurrent `Wait()` calls return one cached status, and a handle retains the
native command buffer, queue, and device so that it may outlive its backend.
Invalid descriptors and injected command creation failures clear the output
handle and commit nothing; completion failures remain attached to the exact
submitted command. Successful allocation and submission counters are owned by
`GpuBackend`, use atomic storage, and make concurrent and steady-state
contracts directly testable. Command creation and completion failures have
distinct `SubmissionFailed` and `DeviceError` statuses.

The Metal host implementation is split by responsibility:
`metal_backend.cpp` owns factory, buffer, copy, transform, and capability setup;
`metal_submission.cpp` owns command-buffer lifecycle and completion errors; and
`metal_primitives.cpp` owns primitive pipelines, validation, and encoding. One
narrow internal header shares the concrete backend state and buffer resolution
surface.

The shader build compiles `dct.metal` and `primitives.metal` separately and
links both into `gjxl.metallib`. Backend creation binds the exact affine,
horizontal-convolution, vertical-convolution, and maximum-reduction entry
points. The initial convolution primitive supports odd 1–33 tap float kernels,
truncated and renormalized edges, strided planes, and exact in-place
input/output through distinct scratch. Maximum reduction is multi-pass and
returns the exact maximum of finite float inputs.

Release validation ran on the M4 Pro described above. The guarded `17x11`
affine -> five-tap convolution -> maximum chain used nonzero offsets and
different row strides. After one warmup, each of three repeated executions
incremented the committed-submission counter by exactly one and the allocation
counter by zero. Its 4096-byte arena used 1284 bytes at peak. Affine matched its
CPU oracle exactly; the chained convolution's maximum absolute error was
`1.19209e-7`, below the fixed `2e-5 + 2e-5 * abs(expected)` gate. Direct one-
and 33-tap cases, an in-place constant case, partial threadgroups, all-negative
reduction input, and a tail maximum also passed; reduction results were
bit-exact with the maximum of the downloaded logical plane.

The complete reference-enabled suite passed 32/32 tests and the
reference-disabled suite passed 26/26. Both include real Metal DCT and primitive
execution. Guarded prefixes, suffixes, and row padding remained poisoned, and
foreign ownership, overflow, misalignment, partial overlap, insufficient
scratch, empty sequences, injected submission failure, and multiple independent
injected completion failures were covered directly. Submission handles were
also exercised after backend destruction, through repeated and concurrent
waits, and from concurrent submission threads.

### 2. Define and validate the prepared AQ evaluation operation — complete (2026-08-26)

Milestone 2 establishes the operation and lifetime boundary without claiming a
working GPU AQ result. Production evaluation remains unavailable until the
reconstruction, filtering, Butteraugli, and block-reduction stages are
connected in later milestones.

#### Public operation contract

- Add `src/gpu/ops/aq_evaluation.h` with a non-copyable
  `PreparedAqEvaluation` and an optional `GpuAqEvaluation` factory capability.
  Querying a backend without that capability returns `Unavailable`; do not add
  an AQ method to `GpuBackend` or an AQ command to `ImagePrimitiveCommand`.
- Preparation accepts the original unpadded linear RGB image, padded coding
  opsin image, complete strategy grid, coefficient-coding options, loop-filter
  options, Butteraugli options, and opsin intensity target. The prepared object
  copies these host inputs and does not retain their views.
- One evaluation accepts only the CPU-produced raw quant field,
  `QuantizerParams`, the two final color-correlation planes, and the EPF
  inverse-sigma plane. Its eventual host-visible result is one block-resolution
  distance map and one score. Final reconstructed-image readback remains a
  later explicit extension rather than implicit scratch output.
- The production `Evaluate` entry point is host-synchronous and atomically
  commits its bounded result only after upload, one GPU submission, successful
  `Wait()`, and readback. During Milestone 2 it returns `Unavailable`; only an
  internal test probe may exercise the staged submission path.

#### Prepared-state and failure contract

- Validate source, padded, block, and color-tile geometry; complete coverage by
  the supported seven DCT strategies; finite supported options; every checked
  byte calculation; and all output pointers before allocating or uploading.
- Canonicalize each strategy cell as an explicit strategy-and-anchor record for
  device upload. Do not expose or copy `AcStrategyGrid`'s private byte encoding.
- Upload the two source images, canonical strategy grid, and quantization tables
  during preparation. Allocate all persistent, per-evaluation staging,
  readback, and contract-probe storage before returning the prepared object.
- Record persistent bytes, staging bytes, and peak planned scratch separately.
  Milestone 2 sizes only the staged path; each later kernel milestone must
  extend the checked plan before enabling that stage. Do not present the probe's
  capacity as the final AQ scratch requirement.
- The backend must outlive its prepared objects. A prepared object owns all
  device buffers and any outstanding `GpuSubmission`, is not concurrently
  reentrant, and waits before destroying or reusing referenced storage.
  Independent prepared objects may use one backend concurrently.
- Descriptor or compatibility rejection submits nothing and leaves the object
  ready. Upload, submission, completion, or readback failure leaves caller
  output unchanged and permanently invalidates the object. Preparation failure
  clears the output object and leaks no allocation.

#### Metal staging and build boundary

- Keep the backend-neutral descriptors under `src/gpu/ops/` and the concrete
  prepared state, exact pipeline binding, validation, encoding, and resource
  layout under `src/gpu/metal/`. The CPU codec target must not link Metal; later
  policy integration queries the optional capability and owns CPU fallback.
- Add one private contract-probe pipeline that derives guarded output from every
  class of static and per-evaluation upload. Compare it with an independent CPU
  oracle, encode the probe and its reduction in one submission, and never route
  its synthetic output through the production `Evaluate` entry point.
- Bind probe functions by exact name at backend creation. The probe is
  transitional test infrastructure and must be removed when the first complete
  production evaluation replaces it.

#### Required validation

- Cover a padded odd source, mixed supported strategies, non-packed host views,
  color-tile edges, poisoned output, and maximum checked dimensions that can be
  represented without allocating a large test image.
- Reject mismatched and overflowing geometry, incomplete or unsupported
  strategies, invalid quantizer and option values, missing capability, null
  outputs, and reuse after an operational failure without committing work.
- After one warmup, run at least three probes with distinct per-evaluation data;
  each must add exactly one submission and zero allocations while matching its
  CPU oracle. Run two independently prepared objects concurrently.
- Inject submission and completion failure, destroy a prepared object with an
  outstanding probe, and verify idempotent waiting, unchanged caller output,
  invalidation, and resource release.
- Run the full reference-enabled and reference-disabled Release matrices,
  including real Metal DCT and image-primitive tests.

Exit criterion: the public contract and unavailable path are stable; malformed
requests fail before submission; repeated real-Metal contract probes use one
submission and no steady-state allocation; independent prepared objects do not
share mutable state; and destruction safely waits for outstanding work. No
probe result is exposed as a completed AQ evaluation.

The implementation adds the backend-neutral `GpuAqEvaluation` capability and
non-copyable `PreparedAqEvaluation` state without changing `GpuBackend` or the
fixed image-primitive command set. The Metal backend binds the exact private
`gjxl_aq_contract_probe` function when the backend is created. Production
`Evaluate` remains explicitly `Unavailable`, submits no work, and leaves output
unchanged until the real pipeline is connected.

Preparation validates checked host layouts, padded geometry, supported options,
finite static samples, and complete coverage by the seven supported strategy
types before allocating. It packs the original and coding images, explicit
strategy-and-anchor records, and all five quantization-table families into one
persistent arena. A separate staging arena owns quant, EPF, color-correlation,
probe-output, reduction, and scalar-result storage. At the Milestone 2
boundary, the guarded 89x57 to 96x64 validation case used 183552 persistent
bytes, 2564 staging bytes, and 8 peak scratch bytes. The current Milestone 3
layout extends the same checked preparation plan and reports 257792 persistent
bytes and 311552 staging/peak-scratch bytes for that case. The resident
reconstructed image is included only in persistent storage; coefficient, DC,
diagnostic-probe, and existing contract-probe storage are included in the
staging upper bound.

The internal probe samples every class of prepared and per-evaluation upload,
then performs maximum reduction in the same compute encoder and command-buffer
commit. Its independent CPU oracle obtains quantization matrices through
`GetDefaultQuantizationMatrix` rather than sharing the device table-offset
logic. The odd padded, strided, mixed-strategy test observed a maximum absolute
map error of 9.53674e-7. After warmup, three distinct evaluations each added
one submission and zero allocations. Two separately prepared objects also ran
concurrently through one backend without sharing mutable scratch.

Validation covers missing capability, maximum representable and overflowing
geometry, incomplete and unsupported strategies, invalid options, static
non-finite samples, invalid quant and EPF fields, invalid quantizer parameters,
null output, poisoned row padding, and same-object reentry. Injected submission,
completion, and post-completion readback failures preserve poisoned caller
output and permanently invalidate the prepared object. Destruction with an
outstanding probe observes a completed `Wait()` before releasing its arenas.

Both Release matrices pass on the real Metal device: 34/34 tests with the
pinned libjxl reference enabled and 28/28 with it disabled. These matrices also
retain the DCT and fixed image-primitive GPU coverage. Reproduce them with:

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON
cmake --build build/release -j
ctest --test-dir build/release --output-on-failure

cmake -S . -B build/no-reference -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON \
  -DGJXL_ENABLE_LIBJXL_REFERENCE=OFF
cmake --build build/no-reference -j
ctest --test-dir build/no-reference --output-on-failure
```

### 3. Port coefficient coding and reconstruction — complete (2026-08-26)

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

The concrete prepared implementation is declared in the private
`metal_aq_evaluation_internal.h` boundary. Preparation and shared state handling
remain in `metal_aq_evaluation.cpp`; coefficient orchestration, readback, and
diagnostics live in `metal_aq_reconstruction.cpp`. No AQ method was added to
`GpuBackend`, and the fixed `ImagePrimitiveCommand` set remains unchanged. A
private offset-aware transform encoder lets one prepared AQ submission reuse
whichever scalar or SIMD pipelines the backend selected for all seven supported
strategies.

Preparation enumerates transform anchors in row-major order, groups them by
strategy for dispatch, and uploads explicit anchor coordinates. It allocates
packed channel-major gathered pixels, preserved forward coefficients, exact
`int32_t` quantized coefficients, reconstruction coefficients, three block-DC
planes, a device error flag, and three persistent reconstructed-opsin planes.
The diagnostic snapshot restores row-major transform order only after the
entire submission, device-error check, and readback succeed.

One compute encoder resets every destination to a sentinel, gathers transform
pixels, invokes each selected forward DCT batch, extracts DC/LLF, quantizes Y,
applies forward CfL to X/B, quantizes X/B, dequantizes and restores inverse
CfL/LLF, invokes each inverse DCT batch, and scatters the complete padded opsin
image. Non-finite scaled coefficients and `int32_t` overflow set a device error
that permanently invalidates the prepared object. Production `Evaluate`
remains `Unavailable`; full reconstructed-image readback is an internal
Milestone 3 diagnostic only, while the device image remains resident for
Milestone 4.

Quantization table-family offsets and coefficient helpers are shared by the
Milestone 2 contract probe and the reconstruction shader. Only the new
coefficient/reconstruction Metal translation unit is compiled with safe math
and floating-point contraction disabled. Existing DCT shader compilation and
numerics are unchanged.

Real-Metal validation runs both single-strategy grids for all seven strategies
and a mixed grid containing all seven, with structured random, asymmetric
impulse, chromatic, and flat-DC samples under both all-scalar and all-SIMD DCT
selections. Forward coefficients are compared
with the independent double-precision DCT oracle at `1e-5 + 5e-5 * abs(x)` for
DCT8 and `2e-5 + 5e-5 * abs(x)` otherwise. Quantized coefficients match the CPU
frame exactly; DC uses `2e-4 + 5e-5 * abs(x)` and reconstructed opsin uses
`7.5e-4 + 1e-4 * abs(x)`.

The direct safe-math probe covers every strategy, both rectangular transpose
pairs, every XYB channel, raw quant 1 and 256, non-default matrix multipliers,
threshold and ties-to-even inputs, LLF zeroing, non-finite input, and finite
overflow. Its quantized output is exact and dequantization passes
`2e-6 + 2e-6 * abs(x)`. After preparation, three reconstruction repeats each
add exactly one submission and zero device allocations. Invalid descriptors
submit nothing; injected submission, completion, post-completion readback, and
device numeric failures leave diagnostic output unchanged and invalidate the
object. The shared Milestone 2 state tests continue to cover same-object
reentry, independent prepared-state concurrency, and destruction with
outstanding work.

### 4. Port loop filters and color conversion — complete (2026-08-26)

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

The Metal backend now binds three exact AQ postprocess entry points for
normalized decoder Gaborish, EPF, and opsin-to-linear RGB. Gaborish preserves
the CPU mirror boundary and channel-specific normalization. One EPF pipeline
implements passes 0, 1, and 2 with the same patch and pixel SAD neighborhoods,
block-border scaling, inverse-sigma lookup, and pass sequences `{1}`, `{1,2}`,
and `{0,1,2}`. Color conversion dispatches only the unpadded source extent, so
the right and bottom crop never requires a separate copy. The new shader unit
uses safe, precise FP32 math with floating-point contraction disabled.

Preparation allocates three persistent source-sized reconstructed-linear
planes. Its option-derived filter plan allocates zero scratch images when all
filters are disabled, one for a single enabled stage, and two for every longer
Gaborish/EPF chain. Encoding alternates those images without copying. With both
filters disabled, color conversion reads reconstructed opsin directly and the
plan reports zero filter and copy dispatches. The default `89x57 -> 96x64`
prepared case now reports 319232 persistent bytes and 459008 staging/peak-
scratch bytes, including two padded filter images.

The direct Metal diagnostic covers source/coding extents `5x3 -> 8x8`, aligned
`16x8`, narrow `3x17 -> 8x24`, odd `17x9 -> 24x16`, and heavily padded
`121x89 -> 128x96`. It crosses Gaborish enabled/disabled with EPF iteration
counts 0 through 3, then repeats all eight configurations with non-default
weights, channel scales, pass scales, border scaling, and intensity target.
Every filtered pixel passes `2e-5 + 2e-5 * abs(x)`, isolated color conversion
passes `1e-4 + 5e-5 * abs(x)`, and the combined stage passes
`2e-4 + 1e-4 * abs(x)`. Observed maxima on the M4 Pro were `1.19209e-7`,
`2.38419e-7`, and `8.34465e-7`, respectively.

A second diagnostic encodes Milestone 3 reconstruction and all Milestone 4
work into one submission with no intermediate synchronization or readback. Its
odd padded mixed grid contains all seven supported strategies and exercises
three quant-field/global-scale variants. Reconstructed opsin retains its
Milestone 3 tolerance; filtering and color conversion use that exact Metal
reconstruction as their independent CPU-oracle input. Three repeats add three
submissions and zero allocations. Validation also covers malformed and
non-finite input, invalid inverse sigma, finite overflow, atomic diagnostic
commit, independent-state concurrency, and injected submission, completion,
and post-completion readback failure.

Fresh AppleClang 17 Release matrices pass all 35 reference-enabled and all 29
reference-disabled tests, including real Metal execution for the new stage and
all prior AQ, DCT, primitive, and Butteraugli coverage. Production `Evaluate`
remains explicitly unavailable until the resident Butteraugli and block-map
stages are connected. This milestone makes no performance or crossover claim.

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

Milestones 0 through 4 are complete. The validated prepared-operation contract
now carries resident reconstructed opsin through loop filtering into cropped
linear RGB. The standalone Butteraugli operation in Milestone 5 is the next
full-resolution stage; Milestone 6 joins it to device block reduction and the
iterative CPU policy. Milestones 7 and 8 harden and qualify the result for
rollout.

Removing libjxl from the production dependency graph is tracked independently
in [`butteraugli.md`](butteraugli.md). It may proceed in parallel and does not
block starting Metal infrastructure or AQ profiling.
