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
The cross-stage plan for reaching the complete encoder's `50x` performance
target is maintained in
[`metal-encoding-performance.md`](metal-encoding-performance.md).

The public GPU substrate now supports buffer allocation, host transfers,
strided device images, reusable scratch, per-command submission handles,
batched transforms, a fixed optional image-primitive capability, and an
optional prepared AQ operation. The prepared Metal path now completes
coefficient coding, reconstruction, loop filtering, cropped opsin-to-linear
conversion, prepared Butteraugli comparison, and strategy-aware block reduction
or, for maximum-error control, compares coding and filtered reconstructed opsin
and reduces per-transform maxima in one submission. Reconstructed images and
pixel-resolution metric inputs remain
resident, while the bounded GPU policy returns only the final quant field,
block map, and score history. Reconstructed-image and encoder-frame
materialization, complete-pipeline switching, and a decision-preserving
automatic rollout are now available. The qualified rollout keeps exact CPU
coefficient coding, dequantization, inverse CfL, and DC-to-low-frequency
conversion, then uses Metal inverse transforms, source-domain filtering, color
conversion, Butteraugli, and block reduction. The fully resident float path
remains available through the prepared operation but is not selected
automatically.

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

Butteraugli Milestones 7 through 9 fix the backend-neutral operation contract,
complete Metal map/score implementation, prepared-reference cache, numerical
gates, and standalone qualification. Metal AQ Milestone 5 consumes that
operation without extending the shared backend contract.

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

Milestone 6 established the bounded policy result: final quant field,
block-distance map, score history, and exact raw-quant decisions. Milestone 7
adds reconstructed-image and `VarDctEncoderFrame` materialization from the last
resident evaluation and uses that path in the complete GPU quantization
pipeline. A future encoder handoff may retain final coefficients on the device,
but that representation is not invented implicitly by scratch buffers.

## Residency and transfer contract

### Prepared state

The prepared state is bound to one backend device, source geometry, strategy
grid, active metric and its options, and `SimpleVarDctCodestreamProfile`. The profile
is the authoritative source for quantization-matrix scales, loop-filter
settings, and the opsin intensity target. The prepared state owns or retains
device-resident copies of:

- Original unpadded linear RGB.
- Padded coding opsin.
- Complete AC-strategy metadata.
- Quantization tables and transform constants required by the selected
  strategies.
- Prepared Butteraugli reference data when Butteraugli is active.
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
- The aggregate metric score.
- For maximum-error control, three actual channel maxima per transform.

The pixel-resolution distance map must remain device-resident. Reading it back
to perform the 16-norm reduction on the CPU is not an acceptable steady-state
path. Bounded policy calls, including their last evaluation, return only this
result. Full calls request reconstructed linear RGB, quantized AC, and
quantized DC from the last evaluation after the same wait, then atomically
commit the field, block map, score history, image, and assembled encoder frame.
Exact raw quant values remain a decision-level acceptance oracle, not an
additional public output.

### Synchronization, concurrency, and failure

The operation is host-synchronous at the CPU policy boundary. One evaluation
encodes one Metal submission and waits once. Intermediate calls read back the
bounded block map and score; an explicitly requested final call additionally
downloads its final image/frame payload without another submission.

A prepared object is not concurrently reentrant because its evaluations reuse
mutable scratch. Independent prepared objects may execute concurrently and
must not share mutable storage. Geometry, strategy, option, or backend mismatch
is rejected before submission and leaves the prepared object usable.

Invalid input submits no work. Allocation or validation failure during
preparation returns no object. Upload, submission, command-buffer,
synchronization, device-numeric, or readback failure commits no caller-visible
CPU output and invalidates the prepared object; the caller must prepare a new
object before retrying. Backend unavailability remains an explicit status
handled by the caller rather than by silently switching implementations inside
the bounded GPU policy.

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

Every successful non-empty transform, image-primitive, or AC-candidate
submission returns an owning `GpuSubmission`. `Wait()` is thread-safe,
idempotent, and returns its cached completion status. The handle retains the
native objects required to wait and may outlive the backend; destroying the
handle does not implicitly wait. Failed validation or command-buffer creation
resets the output handle and submits no work. Empty image-primitive sequences
are invalid. Zero-transform and zero-candidate batches retain their established
no-op contracts and return success with a null handle.

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
Staged AC candidate evaluation follows the same rule through the optional
`GpuAcStrategyEvaluation` capability and returns its own submission handle.

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

Milestones 0 through 9 are complete.

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
  opsin image, complete strategy grid, `SimpleVarDctCodestreamProfile`, and
  Butteraugli options. The prepared object copies these host inputs and does
  not retain their views.
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
  -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON \
  -DGJXL_ENABLE_LIBJXL_REFERENCE=ON
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
pixels, invokes each selected forward DCT batch, extracts DC/LLF, applies the
simple codestream's decoder-equivalent DC quantization and default DC CfL,
quantizes Y AC, applies forward AC CfL to X/B, quantizes X/B AC, dequantizes and
restores inverse CfL/LLF, invokes each inverse DCT batch, and scatters the
complete padded opsin image. Non-finite scaled coefficients and `int32_t`
overflow set a device error that permanently invalidates the prepared object.
Production `Evaluate` remains `Unavailable`; full reconstructed-image readback
is an internal Milestone 3 diagnostic only, while the device image remains
resident for Milestone 4.

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
pairs, every XYB channel, raw quant 1 and 256, non-default matrix scales,
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
plan reports zero filter and copy dispatches. At the Milestone 4 boundary, the
default `89x57 -> 96x64` prepared case reported 319232 persistent bytes and
459008 staging/peak-scratch bytes, including two padded filter images.

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

### 5. Integrate the device Butteraugli operation — complete (2026-08-27)

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

The AQ and Butteraugli branches are now reconciled in one backend. Metal binds
both pipeline registries and exposes both optional coherent operations; the
combined metallib retains each shader unit's established arithmetic flags.
Fresh combined Release baselines passed 38/38 reference-enabled and 32/32
reference-disabled tests before AQ integration.

AQ preparation uploads the original linear image, prepares and retains one
`PreparedDeviceButteraugli`, and caches its main and optional half-resolution
psycho-image decomposition in one submission. The AQ state now owns a
source-sized float32 distance map. Each diagnostic evaluation submits the
existing reconstruction/filter/color chain, checks its scalar device error,
then passes `reconstructed_linear_` directly to the prepared Butteraugli
comparison. The linear image, map, and score never pass through a host image
between stages. Butteraugli owns one following synchronous submission; merging
the two command buffers is deliberately deferred to Milestone 6's one-
submission evaluation gate.

The mixed `91x57 -> 96x64` integration case contains all seven supported AC
strategies, Gaborish plus all three EPF passes, non-default filter and intensity
options, and three raw-quant/global-scale variants. An isolated oracle feeds
the exact Metal linear reconstruction into native CPU Butteraugli and retains
the standalone `1.5e-3` map/score limit. The complete CPU reconstruction-
through-Butteraugli comparison uses a fixed `2e-3` accumulated limit. Observed
maximum errors on the M4 Pro were `7.24792e-5` and `8.7738e-5`, respectively.
Three repeated comparisons add six evaluation submissions and zero allocations
after preparation; the companion Milestone 4 diagnostic accounts for the
other three submissions in the combined test.

Preparation adds exactly three prepared-state allocations (persistent,
staging, and the Butteraugli arena) and one reference-cache submission.
Backend-neutral Butteraugli memory accounting is included in AQ staging and
peak-scratch reports. The default `89x57 -> 96x64` case now reports 319232
persistent bytes, 1306036 staging bytes, and 1047824 peak-scratch bytes.
Validation preserves atomic diagnostic output for Butteraugli submission,
completion, and post-comparison readback failures, rejects a null snapshot
without submitting, invalidates the AQ state after operational failure, and
runs independent prepared states concurrently. The standalone Butteraugli
corpus continues to cover all leaf stages, small/odd/strided geometry,
reference-cache reuse, and the full numerical boundary.

At the Milestone 5 boundary, production `Evaluate` remained unavailable until
Milestone 6 added the strategy-aware block reduction, bounded readback, and CPU
quant-field policy integration. Milestone 5 inherited the standalone
Butteraugli performance evidence but made no complete-AQ speedup claim. Its
fresh final Release matrices passed all 38 reference-enabled and all 32
reference-disabled tests.

### 6. Add device AQ block reduction and CPU policy integration — complete (2026-08-27)

- Reduce the pixel distance map to the transform-aware 16-norm block map on the
  GPU.
- Cover multiblock strategies, partial source edges, and every supported
  strategy footprint.
- Make production `PreparedAqEvaluation::Evaluate` return only block distances
  and the score for every policy evaluation, including the last one.
- Factor the deterministic CPU policy driver so the CPU reference evaluator and
  prepared GPU evaluator share the same clamp, power, rounding-progress, bounds,
  and iteration logic.
- Run the unchanged CPU clamp, power, rounding-progress, and bounds policy.
- Execute the default two-update loop using one GPU submission and one bounded
  readback per evaluation.
- Atomically commit the bounded policy outputs only after the last evaluation
  succeeds.
- Keep the public complete quantization pipeline on CPU AQ until Milestone 7;
  do not run an extra CPU evaluation solely to materialize reconstructed RGB or
  a `VarDctEncoderFrame`.

Exit criterion: score history and floating-point AQ fields remain within their
fixed tolerances, every raw quant decision matches exactly, each evaluation uses
one submission and one bounded readback, no pixel-resolution plane crosses the
CPU/GPU boundary, and the CPU reference path remains bit-for-bit unchanged.

The implementation factors one internal evaluator interface and policy driver
from `FindBestQuantization`. The CPU evaluator still owns reconstructed linear
RGB, `VarDctEncoderFrame`, and profiling, while
`RunGpuAdaptiveQuantizationPolicy` explicitly requires the optional prepared-AQ
capability and never falls back. The shared driver owns initial strategy
adjustment, bounds, second-update clamp, power update, raw-rounding progress,
and iteration order. Existing CPU score, field, raw-quant, and profiled-path
pins remain unchanged.

Metal now appends its already validated prepared Butteraugli comparison to the
AQ command encoder and dispatches one 256-thread reduction group per transform
anchor for each of the seven strategy batches. The reduction squares four
times, averages only source pixels inside right and bottom edges, takes the
sixteenth root, multiplies by `1.2`, and fills every covered base block. The
transitional contract-probe shader, pipeline, and storage were removed; its
split submit/finish seam now exercises the production evaluation.

The isolated reduction corpus covers all seven strategies, both rectangular
transpose pairs, aligned and mixed grids, and right/bottom partial edges. Its
maximum absolute error was `2.38419e-7` against the fixed `2e-6` oracle gate.
The existing `91x57 -> 96x64` chained CPU oracle observed `3.05176e-5` maximum
production block-map/score error against the fixed `2e-3` accumulated gate.
Each production evaluation added exactly one submission and zero device
allocations after preparation, preserved poisoned host padding, and read back
only the block map and scalar score.

Bounded policy comparisons cover zero, one, and two updates with default and
representative non-default filter and Butteraugli options. Observed maxima were
`7.62939e-5` for score history, `4.65393e-4` for block distance, and
`1.02818e-5` for the final float field, all below the fixed `2e-3` gate; every
final raw-quant value and quantizer parameter matched the corresponding CPU
frame exactly. Missing capability, invalid and overflowing strided descriptors,
atomic output, upload/submission/completion/numeric/readback failure,
post-failure rejection, reuse after descriptor rejection, non-reentrancy,
outstanding-work destruction, and independent concurrent states are covered.

For the documented `89x57 -> 96x64` prepared case, the completed layout uses
`319232` persistent bytes, `1305524` staging bytes, and `1047312` peak-scratch
bytes. Preparation still adds three device allocations and one reference-cache
submission. This milestone makes no performance, crossover, automatic
selection, or complete-pipeline claim.

Fresh AppleClang 17 Release matrices pass `55/55` tests with the pinned
Butteraugli reference enabled and `49/49` with it disabled. Both include the
installed static consumer and all CPU/Metal AQ targets. The separately built
pinned libjxl decoder also accepts all 21 codestream-conformance fixtures and
the checked workflow sample.

### 7. Materialize final output and connect the complete pipeline — complete (2026-08-27)

- Define and atomically materialize final reconstructed linear RGB and the
  existing `VarDctEncoderFrame` separately from scratch layout.
- Connect Metal AQ to the public complete quantization pipeline without a
  redundant CPU final evaluation; keep any later device-resident encoder
  handoff as a separate extension.
- Extend the completed bounded-path lifetime and failure guarantees to final
  image/frame materialization and complete-pipeline output.
- Report persistent and peak scratch memory by image size.

Exit criterion: failures never expose partially committed CPU output, resource
lifetimes are explicit, the steady-state path has no per-evaluation device
allocation, and the complete GPU pipeline produces the existing reconstructed
image and encoder-frame outputs atomically.

`PreparedAqEvaluation::Evaluate` now accepts an optional final-output
descriptor. The shared policy marks its last evaluation explicitly, so bounded
calls retain the Milestone 6 readback while full calls use that same submission
and wait to download reconstructed linear RGB, quantized AC, and authoritative
quantized DC. Invalid final descriptors submit nothing; failures after an
accepted submission leave bounded and final output untouched and invalidate
the prepared state.

The reconstruction kernel retains quantized DC before replacing its working DC
planes with decoder-equivalent values. An internal codec assembler combines
those integers with the final host-owned raw field, quantizer, CfL, EPF
sharpness, strategy grid, and batch-major quantized AC views. It packs the
existing row-major fixed-capacity AC groups, reconstructs the DC cache from the
authoritative integers, zeros unused group tails, and validates the candidate
`VarDctEncoderFrame`; no source transform or reconstruction is repeated on the
CPU.

`RunGpuAdaptiveQuantization` atomically returns the existing full AQ output,
while `RunGpuAdaptiveQuantizationPolicy` remains the bounded interface.
Quantization orchestration now injects both AC-search and AQ providers:
`RunCpuQuantizationPipeline` is unchanged, and `RunGpuQuantizationPipeline`
preflights prepared-AQ support before using GPU search and GPU AQ without a CPU
fallback. The public codestream workflow remains CPU-selected pending the
Milestone 8 rollout decision.

Full-AQ comparisons cover zero, one, and two updates with default and
non-default profiles. The existing bounded maxima remain `7.62939e-5` for
score, `4.65393e-4` for block distance, and `1.02818e-5` for the float field;
the maximum final reconstructed-RGB error is `5.36442e-6`. Raw quant, quantizer,
CfL, EPF sharpness, quantized and decoder-equivalent DC, grouped AC storage,
complete frame state, and supported codestream bytes are exact.

The default two-update complete pipeline observes maxima of `1.14441e-5` for
the quant field, `7.39694e-5` for block distance, `1.03533e-4` for score, and
`6.3777e-6` for reconstructed RGB, with exact frame and codestream output. The
separate 257x17 AC-group-edge corpus keeps its deliberately float-sensitive
block/RGB results under a fixed `2.5e-2` boundary (`2.21866e-2` and
`2.01165e-2` observed) while retaining exact frame and codestream decisions.

Final materialization adds no submission and no device allocation: preparation
still adds three allocations and one reference-cache submission, and each
policy evaluation adds exactly one submission. The non-default prepared memory
figures are:

| Source -> coding geometry | Persistent | Staging | Peak scratch |
| --- | ---: | ---: | ---: |
| 89x57 -> 96x64 | 319232 | 1306804 | 1048592 |
| 128x96 -> 128x96 | 640512 | 2945588 | 2330752 |
| 510x532 -> 512x536 | 13215232 | 64976692 | 51408544 |
| 854x480 -> 856x480 | 19849728 | 97888436 | 77391888 |
| 1280x720 | 44514816 | 219901748 | 173821312 |
| 1920x1080 | 100099072 | 494763060 | 391082528 |

Missing capability, invalid/strided final output, atomic descriptor rejection,
upload/submission/completion/numeric/readback failure, reuse and invalidation,
outstanding destruction, poisoned padding, independent prepared states, and
complete-pipeline atomic statistics are covered. This milestone makes no
performance, crossover, automatic-selection, or rollout claim.

Fresh AppleClang 17 Release matrices pass `55/55` tests with the pinned
Butteraugli reference enabled and `49/49` with it disabled. Both include the
installed static consumer, and the separately built pinned libjxl decoder
accepts all 21 codestream-conformance fixtures and the checked workflow sample.

### 8. Establish the end-to-end performance and rollout gate — complete (2026-08-27)

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

The expanded Flower and resolution sweep exposed decision discontinuities in
the fully resident float reconstruction path: a first Flower diagnostic had
40 different raw-quant values, a `0.114605` float-field delta, and a
`0.048073` score delta. The fixed `2e-3` decision gate was not relaxed. The
Milestone 8's rollout evaluator instead computes the authoritative coefficient
frame,
decoder reconstruction, loop filters, and linear RGB with the unchanged CPU
reference, uploads that exact comparison image, and executes prepared
Butteraugli plus strategy-aware block reduction in one Metal submission. GPU
AC search and the CPU AQ update policy remain unchanged. The direct prepared
operation still supports the complete resident reconstruction chain for tests
and future precision work. At that milestone, automatic and forced workflow
selection used the decision-preserving exact-linear composite; Milestone 9
below supersedes that handoff.

`VarDctEncodingOptions` now selects `kAutomatic`, `kCpu`, or `kMetal`, and the
summary reports the backend that actually ran. The CLI exposes the same policy
as `--backend auto|cpu|metal`. The production metallib is embedded as a
`679800`-byte library image, so installed consumers need no shader path or
resource bundle. Backend construction is lazy and cached once per process.
Automatic selection is qualified only for the exact backend name
`Metal: Apple M4 Pro` and padded coding geometries with at least `128x96`
pixels, area at least `12288`, and minimum dimension at least `96`. Other
devices, smaller images, backend unavailability, or missing capabilities use
CPU before pipeline execution. Forced Metal ignores the device qualification
and size cutoff but requires both capabilities. Once GPU pipeline execution
begins, every error is returned atomically; it never triggers a CPU retry.

Three independent AppleClang 17 Release processes used one warmup rotation and
three measured rotations of all 16 phases. Each workload first enforced exact
frame and codestream output and the unchanged `2e-3` numeric gate. Cells below
are ranges of process medians in milliseconds; `Metal one` and `Metal AQ2`
include the exact CPU prefix, preparation, upload, synchronization, bounded
readback, and final download as applicable.

| Workload | CPU one | Metal one | CPU AQ2 | Metal AQ2 | CPU pipeline | Metal pipeline | CPU public | Metal cold public |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic 128x96 | 14.043–14.835 | 9.069–9.427 | 42.195–43.174 | 24.139–24.683 | 53.873–55.026 | 27.435–28.584 | 57.189–58.302 | 32.159–34.226 |
| Odd 121x89 -> 128x96 | 12.443–13.013 | 8.317–9.850 | 37.025–38.152 | 21.592–21.865 | 49.945–51.200 | 23.780–26.245 | 53.957–56.640 | 31.686–32.576 |
| Flower 510x532 | 303.998–307.460 | 136.965–148.400 | 908.629–962.251 | 393.205–404.173 | 1187.000–1216.841 | 458.216–565.296 | 1203.493–1566.048 | 484.971–496.848 |
| Padded 480p | 461.052–517.509 | 201.176–348.622 | 1376.035–1821.417 | 597.875–786.046 | 1754.996–1961.197 | 695.578–901.443 | 1785.945–2458.735 | 717.864–890.746 |
| Padded 720p | 1035.719–1225.345 | 455.652–546.217 | 3109.737–3665.874 | 1322.697–1691.286 | 3947.403–4130.377 | 1517.808–1806.671 | 3992.456–4295.592 | 1573.067–1744.441 |
| Padded 1080p | 2341.748–2362.551 | 1005.584–1033.969 | 7020.374–7100.934 | 2912.645–2916.732 | 8873.342–9012.107 | 3322.967–3397.221 | 8953.741–9086.560 | 3411.637–3475.737 |

Cold public-workflow speedups were `1.70–1.78x` at 128x96,
`1.66–1.78x` on the odd fixture, `2.48–3.15x` on Flower,
`2.44–2.76x` at 480p, `2.46–2.54x` at 720p, and `2.61–2.65x` at
1080p. The crossover sweep was:

| Coding geometry | CPU public median range | Metal cold median range | Result |
| --- | ---: | ---: | --- |
| 32x24 | 4.646–4.856 | 5.945–9.906 | CPU won every process |
| 64x48 | 14.894–15.902 | 13.477–15.362 | Metal medians won, sample ranges overlapped |
| 96x64 | 28.997–30.720 | 20.068–21.164 | Metal ranges did not overlap CPU |
| 128x96 | 57.189–58.302 | 32.159–34.226 | Metal won with rollout headroom |

The measured stable crossover bracket is therefore `(64x48, 96x64]`; the
automatic floor deliberately advances one full measured step to 128x96.

The profiled prepared perceptual tail separates upload, submission, wait, GPU
timestamps, and readbacks. Process-median ranges were:

| Workload | Input upload | Submit | Completion wait | GPU time | Bounded readback | Final readback | Commit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flower | 0.272–3.404 | 0.063–0.100 | 4.834–29.746 | 3.625–9.578 | 0.006–0.012 | 1.205–1.974 | 0.186–0.943 |
| 1080p | 1.932–10.990 | 0.099–0.108 | 23.497–77.058 | 17.635–18.334 | 0.025–0.064 | 8.873–12.791 | 1.620–6.696 |

The complete corpus observed maximum errors of `1.838893e-3` for the final
float quant field, `3.623962e-5` for the block map, and `1.144409e-5` for the
score history. Reconstructed RGB, raw quant, the complete encoder frame, and
serialized codestream bytes were exact. Preparation and the perceptual tail
retain the Milestone 7 allocation contract: one reference-cache submission at
preparation, exactly one submission per evaluation, and zero steady-state
device allocations. Current prepared memory is:

| Source -> coding geometry | Persistent | Staging | Peak scratch |
| --- | ---: | ---: | ---: |
| 128x96 | 639488 | 2945588 | 2330752 |
| 121x89 -> 128x96 | 604416 | 2698292 | 2154696 |
| 510x532 -> 512x536 | 13195264 | 64976692 | 51408544 |
| 854x479 -> 856x480 | 19789824 | 97756084 | 77292880 |
| 1279x719 -> 1280x720 | 44376576 | 219592244 | 173589432 |
| 1919x1079 -> 1920x1080 | 99822080 | 494296372 | 390734776 |

Fresh AppleClang 17 Release matrices pass `55/55` tests with the pinned
Butteraugli reference enabled and `49/49` with it disabled. Both matrices
include the installed static-library consumer, which creates the backend from
the embedded metallib. The separately built pinned libjxl decoder accepts all
21 codestream-conformance fixtures and the checked workflow sample. Reproduce
one balanced process with `just quantization-benchmark all simd 3 1`; the
ranges above combine three independent invocations.

### 9. Explore and qualify the reconstruction handoff — complete (2026-08-27)

- Localize the decision error in the fully resident evaluator rather than
  widening the fixed `2e-3` gate.
- Compare exact-linear, exact-reconstructed-opsin, exact-coefficient, and fully
  resident CPU/Metal handoffs under one benchmark and oracle harness.
- Exercise default and non-default quality targets, filtering options, smooth
  and textured synthetic images, odd padding, natural images, and 720p/1080p
  inputs.
- Select the deepest boundary that preserves the authoritative encoder frame,
  integrate it into bounded AQ and the complete workflow, and remove the
  temporary policy-level selector used for the comparison.

The investigation first found a correctness bug independent of transform
precision: the resident path filtered the padded coding extent, whereas the CPU
reference crops reconstructed opsin to the source extent before Gaborish and
EPF. Metal filter parameters and dispatches now use the source width and height
while retaining the padded device stride. An isolated postprocess oracle covers
the corrected right and bottom boundary behavior.

Four handoffs were then compared:

| Boundary | CPU work per evaluation | Metal work | Decision result |
| --- | --- | --- | --- |
| Exact linear | coefficient coding through color conversion | Butteraugli and block reduction | Stable, but leaves most AQ work on CPU |
| Exact reconstructed opsin | coefficient coding and inverse reconstruction | filters through block reduction | Stable at the default target after the crop fix |
| Exact coefficients | coefficient coding, dequantization, inverse CfL, and DC/LLF conversion | inverse transforms through block reduction | Selected production boundary |
| Fully resident | policy-side fields only | forward transforms through block reduction | Rejected; float coefficient ties compound across policy updates |

The selected evaluator uploads packed float reconstruction coefficients after
the CPU reference has made the quantized-AC and quantized-DC decisions, applied
dequantization and inverse CfL, and replaced transform low frequencies from the
authoritative frame DC. Metal begins with the inverse transforms and retains
reconstruction, source-domain filtering, opsin-to-linear conversion, prepared
Butteraugli comparison, and strategy-aware reduction in the same submission.
The final `VarDctEncoderFrame` is assembled from the exact uploaded coefficient
decisions, so there is no redundant CPU reconstruction or final perceptual
evaluation. Preparation storage is reused: each evaluation still adds exactly
one submission and zero device allocations. Device memory is unchanged from
Milestone 8; the new host preparation workspace is one packed float coefficient
array.

| Source -> coding geometry | Device persistent | Device staging | Device peak scratch | Packed host coefficients |
| --- | ---: | ---: | ---: | ---: |
| 128x96 | 639488 | 2945588 | 2330752 | 147456 |
| 510x532 -> 512x536 | 13195264 | 64976692 | 51408544 | 3293184 |
| 1279x719 -> 1280x720 | 44376576 | 219592244 | 173589432 | 11059200 |
| 1919x1079 -> 1920x1080 | 99822080 | 494296372 | 390734776 | 24883200 |

The fully resident variant is faster, but it is not a safe encoder boundary.
Sparse one-unit quantized-DC and quantized-AC differences can cross policy
rounding thresholds and amplify on the next update. A natural 1080p crop
reached `0.123` final-field and `0.524` block-map error, and a smooth-gradient
case reached `0.112` score-history error. CPU transforms accumulate in double
precision while Metal transforms use float arithmetic; duplicating the CPU
dequantization, CfL, and DC/LLF preparation proved sufficient to retain safe
float inverse transforms without pretending those coefficient ties are
interchangeable.

After Milestone 9, the rejected boundary was promoted from a private diagnostic
to an explicit public experimental mode so its errors and candidate fixes can
be measured without test-only shims. `GpuAdaptiveQuantizationMode` selects
`kExactCoefficients`, `kFullyResident`, or `kThroughput` in bounded AQ, full AQ,
and the complete iterative GPU quantization pipeline. The separate
`kMaximumThroughput` value selects only the public frame-only workflow.
`VarDctEncodingOptions::metal_aq_mode` and the CLI's `--metal-aq` option carry
the same choice through codestream generation. Both resident modes require
forced Metal, are reported in the workflow summary, and never participate in
automatic selection. Their output is atomic and structurally valid but is
intentionally not covered by the CPU decision-parity promise. Both also apply
inverse Gaborish through one three-channel Metal primitive submission and use
the deterministic tilewise pixel-domain initial-CfL seed. Exact-coefficient
mode retains the
original CPU Gaborish and DCT-domain iterative initial-CfL paths. After strategy
selection, each resident mode computes one fast strategy-aware final-CfL map
from the adjusted initial quant field and reuses it across perceptual
evaluations; the exact path retains evaluation-local quant-dependent maps.
The separate throughput mode retains the resident coefficient path while
limiting the complete quantization pipeline to one AQ update. Direct fully
resident AQ APIs still honor their requested iteration count; the bounded
policy is therefore explicit rather than a hidden reinterpretation of
`kFullyResident`.

Resident coefficient coding now composes the pinned `AdjustQuantBlockAC`
heuristic into the existing per-transform Metal kernel. It selects one shared
raw quant from Y, X, and B, retains the adjusted Y dead-zone thresholds, and
recomputes EPF inverse sigma from the adjusted anchor on device. The final
frame reads back the block-grid raw-quant field, but the decision adds no
allocation, command submission, coefficient readback, or pixel-sized host
boundary. Direct fixtures cover all seven strategies, and a mixed-strategy
integration check verifies the batched decision and EPF field against CPU
oracles. This does not make resident output CPU-bit-exact: its adjustment acts
on Metal FP32 forward coefficients rather than the CPU double-precision
coefficients.

Maximum-throughput mode bypasses AC search with a complete DCT8 grid, adjusts
the initial quant field once, and invokes `PreparedAqEvaluation::EncodeFrame`.
The Metal implementation runs reset, inverse Gaborish, gather,
forward-transform, and coefficient-quantization commands in one submission,
then reads back only device error state, quantized DC, quantized AC, and the
block-grid adjusted raw-quant field required by frame serialization.
`AqEvaluationPreparation::frame_only` omits the original-image upload and
prepared Butteraugli reference. No inverse transform reconstruction, decoder
loop-filter evaluation, linear-RGB reconstruction, Butteraugli comparison,
block reduction, or score readback runs, so the workflow reports an empty
score history.

The production corpus covers 13 built-in workloads at Butteraugli targets
`1.0` and `1.2`, plus four independent 1919x1079 natural, HDR-like, and
high-contrast images at `1.2`. The selected path observed maxima of
`1.838893e-3` for the final float field, `2.745390e-4` for the block map,
`2.551079e-5` for score history, and `5.960464e-6` for reconstructed RGB. Raw
quantization, complete frame state, and codestream bytes were exact. Direct
exact-coefficient evaluation remained below the same `2e-3` gate, preserved
poisoned host padding, and retained the one-submission/zero-allocation
contract.

Targets `0.5` and `2.0` were also explored. Even an exact-linear handoff can
cross a policy discontinuity at `0.5`, while moving the inverse reconstruction
to Metal introduced additional threshold cases at both extremes. Automatic
workflow selection is therefore qualified only for finite Butteraugli targets
in `[1.0, 1.2]`, in addition to the existing device and geometry gates. Outside
that window automatic mode selects CPU before pipeline execution. Forced Metal
remains explicit for diagnostics and callers that accept the unqualified
quality range, returns any failure atomically, and never retries on CPU.

Three independent Apple M4 Pro Release processes each used one warmup and
three measured rotations of all 22 exploratory phases. The cells are ranges
of process medians. The committed benchmark removes the temporary policy
handoff matrix and retains 18 durable production and prepared-operation
phases. `--gpu-aq fully-resident` now selects the resident policy, complete
pipeline, and public-workflow phases while retaining the CPU baselines and
printing their numerical and codestream deltas.

| Workload | CPU AQ2 | Metal AQ2 | CPU complete pipeline | Metal complete pipeline | CPU public | Metal cold public |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128x96 | 27.943–29.646 ms | 5.551–5.764 ms | 35.850–37.042 ms | 8.223–8.274 ms | 38.075–40.797 ms | 15.726–16.716 ms |
| Flower 510x532 | 640.883–660.979 ms | 100.495–102.700 ms | 824.642–845.000 ms | 133.000–143.928 ms | 848.807–855.194 ms | 148.533–152.821 ms |
| 1279x719 -> 1280x720 | 2193.056–2226.364 ms | 324.751–335.515 ms | 2779.794–2805.685 ms | 463.664–480.656 ms | 2813.722–2828.943 ms | 502.435–516.657 ms |
| 1919x1079 -> 1920x1080 | 4974.302–4990.734 ms | 706.189–726.516 ms | 6247.826–6291.163 ms | 1023.175–1030.042 ms | 6345.863–6374.548 ms | 1084.067–1117.815 ms |

Paired cold-public speedups were `2.29–2.59x`, `5.60–5.74x`, `5.47–5.63x`,
and `5.70–5.87x`, respectively. Complete-pipeline speedup was `5.80–6.00x` at
720p and `6.08–6.11x` at 1080p; selected-boundary AQ itself reached
`6.86–7.07x` at 1080p. The non-production fully resident AQ mode reached
`12.2–12.8x` there, demonstrating that 10x compute throughput is possible but
not decision-safe. A production-default 10x result would require an exact or
decision-equivalent GPU coefficient coder; the tested float implementation
does not meet that prerequisite, so the mode remains explicitly opt-in.

A selective CPU-repair handoff was also considered. At the default target the
larger fixtures showed only one to six one-unit quantized-DC mismatches and no
AC mismatch; the 1080p `0.5` experiment also exposed one AC mismatch. Sparsity
alone is not a correctness contract: the float device result cannot prove on
which side of the CPU double-accumulation rounding boundary the reference lies.
Conservatively finding every repair candidate still needs a validated
transform-specific error interval, and recomputing all candidates with the CPU
transform returns the measured coefficient-coding cost to the critical path.
No false-negative repair bound was established across all seven strategies, so
the repair prototype is retained as future precision research rather than a
production shortcut.

Fresh AppleClang 17 Release matrices pass `55/55` tests with the pinned
Butteraugli reference enabled and `49/49` with it disabled. Both include the
installed static-library consumer. Targeted tests observed `2.38419e-7`
maximum block-reduction error, `1.19209e-7` filter error, `2.38419e-7` color
error, `4.57764e-5` chained production-evaluation error, `3.58582e-4`
policy block-map error, `1.06812e-4` policy score error, and `5.30481e-6`
policy reconstructed-RGB error. The separately built pinned libjxl decoder
accepts all 21 codestream-conformance fixtures and the checked workflow sample.
`git diff --check` is clean.

## Validation matrix

GPU validation is layered so a final round trip cannot hide a wrong
intermediate:

1. Leaf kernels compare every output element with an independent CPU oracle.
2. Stage tests compare coefficient coding, reconstruction, filtering, color
   conversion, Butteraugli, and block reduction independently.
3. Evaluation tests compare distance maps, block maps, scores, submission and
   readback counts, and failure behavior.
4. Milestone 6 iterative tests compare the entire score history, final float
   quant field, and exact raw-quant decisions without requiring final image or
   frame materialization.
5. Milestone 7 final-output tests compare reconstructed images, encoder-frame
   state, and deterministic codestream bytes.
6. Pipeline tests compare the CPU and Metal paths through the supported
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
- Padded 1440p and 4K workloads for the
  `gpu_iterative_aq_two_updates_e2e` phase only. These high-resolution cases do
  not execute separate CPU AQ baselines, exploratory prepared subphases, the
  complete quantization pipeline, or public codestream workflow phases. The
  production exact-coefficient Metal mode retains its documented CPU decision
  boundary inside the GPU AQ operation.

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

Milestones 0 through 9 are complete. The prepared operation can carry resident
reconstructed opsin through filtering, cropped linear RGB, prepared
Butteraugli comparison, and strategy-aware block reduction in one submission;
the shared CPU policy requests final RGB and frame materialization only from
its last evaluation. Milestone 9 supersedes Milestone 8's conservative
exact-linear rollout with the qualified exact-coefficient boundary: CPU owns
the coefficient decisions, dequantization, inverse CfL, and DC/LLF conversion,
while Metal owns inverse transforms and the complete image/perceptual tail.
The fully resident path is a first-class experimental opt-in because it does
not satisfy the unchanged decision gate. Exact coefficients remain the default
and the only automatic mode; automatic selection additionally requires a
Butteraugli target in `[1.0, 1.2]`.

The rate-control RC2 extension also adds the alternate resident maximum-error
tail. Its strategy-aware kernel ignores padded pixels and returns only the
block map and per-transform channel maxima required by the unchanged CPU
policy. Exact-coefficient Metal preserves CPU frame and codestream decisions;
fully resident maximum-error remains an explicit experimental mode. Automatic
maximum-error requests continue to use CPU.

The pre-Milestone-6 2026-08-27 codestream integration made the encoder frame
profile the single CPU/GPU option contract and added modular DC quantization
parity to the resident Metal reconstruction. Its AppleClang 17 Release matrices
passed 54/54 tests with the pinned reference enabled and 48/48 with it disabled,
including codestream conformance, scalar/dispatched Butteraugli differentials,
Metal DCT and AC-strategy search, and the full staged Metal AQ diagnostics.

The integration checkpoint also moves staged AC candidate evaluation out of
`GpuBackend` into `GpuAcStrategyEvaluation`. Metal returns one caller-owned
submission for the complete multi-strategy batch; search, tests, and benchmarks
wait that handle directly. The core backend consequently retains only buffer,
copy, and transform operations, with no codec-specific AC methods or
backend-wide synchronization state. Factored-radix DCT selection and the shared
device-basis fast path remain unchanged.

Removing libjxl from the production dependency graph is tracked independently
in [`butteraugli.md`](butteraugli.md). It may proceed in parallel and does not
block starting Metal infrastructure or AQ profiling.
