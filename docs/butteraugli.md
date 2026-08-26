# Native Butteraugli implementation roadmap

This document tracks replacement of the production libjxl Butteraugli
dependency with gjxl-owned CPU and Metal backends. The pinned libjxl revision
`e8ff09762481785938d8e4e01333ed3917571161` remains the correctness oracle
until both native implementations satisfy the validation gates below.

The work is intentionally parity-first. A plausible perceptual metric is not
enough: small differences in blur boundaries, nonlinear transforms, masking,
or reductions can change adaptive-quantization decisions.

## Current state

`ComputeButteraugliDistance` exposes a backend-neutral, view-based CPU API. Its
current implementation runs the gjxl-owned native scalar backend with
call-local scratch, producing a pixel-resolution distance map and its maximum
score. gjxl also owns the transform-aware 16-norm reduction of that map and the
iterative quant-field update.

`GJXL_ENABLE_LIBJXL_REFERENCE=OFF` builds the facade and complete iterative
quantization pipeline without libjxl or Highway. Reference-enabled builds keep
the pinned libjxl translation units and Highway for differential tests,
golden regeneration, and comparative benchmarks. Removing the now-unused
reference linkage from the production library target remains Milestone 6.

The current Metal abstraction supports allocation, transfers, synchronization,
and batched transforms. It does not yet model image operations, multi-pass
compute graphs, or reusable scratch storage. Those shared contracts and the
complete resident AQ integration are owned by
[`metal-aq.md`](metal-aq.md); this document owns only the standalone device
Butteraugli operation and its perceptual validation.

## Goals

- Provide a readable scalar CPU implementation owned by gjxl.
- Match the pinned libjxl distance map, score, and downstream AQ decisions
  within documented numerical tolerances.
- Keep libjxl available as an optional differential-test oracle rather than a
  production requirement.
- Add a Metal implementation without hiding host/device transfers behind the
  synchronous CPU-view API.
- Reuse GPU primitives and scratch-management infrastructure across later
  codec operations.
- Preserve atomic output behavior on every failing CPU path.

## Non-goals

- Retuning Butteraugli or designing a different perceptual metric.
- Supporting libjxl features outside the parameters used by the current gjxl
  contract.
- Moving iterative AQ policy or convergence decisions to the GPU before
  profiling justifies it.
- Requiring bit-identical CPU and GPU floating-point evaluation. Numerical and
  encoder-decision parity are the acceptance criteria.

## Backend boundary

Keep `ComputeButteraugliDistance` as the stable host-facing CPU facade. The
native implementation is the production path; test-only internal adapters let
the same fixtures evaluate native CPU, pinned scalar libjxl, and dispatched
libjxl in one reference-enabled build.

Metal should receive an explicit device-buffer operation, for example a
`ButteraugliBatch` under `src/gpu/ops/`, rather than silently uploading and
downloading through the CPU facade. The operation should accept device-resident
reference and distorted images, write a device-resident distance map, and
produce a score through an explicit reduction/readback contract.

Repeated AQ evaluations compare several reconstructions with one original
image. Once baseline parity is established, a prepared-reference object may
cache the original image's opsin and frequency decomposition. This is an
optimization milestone, not part of the initial correctness contract.

## Validation strategy

The reference-enabled configuration must remain available while native CPU and
Metal parity are maintained. Tests should compare complete distance maps, not
only the aggregate score or selected samples. Test-only fixtures generated from
the pinned source should also cover intermediate stages without exposing those
stages as production APIs.

The differential corpus must cover:

- Identical images and isolated RGB impulses.
- Flat fields, gradients, texture, sharp contrast, and chromatic errors.
- Deterministic random images at low, normal, and out-of-range finite values.
- Dimensions below 8, exactly 8, odd dimensions, narrow images, and realistic
  image sizes.
- Default and representative non-default values for `hf_asymmetry`,
  `x_multiplier`, and `intensity_target`.
- Strided inputs and outputs, poisoned padding, invalid arguments, allocation
  failures where injectable, and atomic failure behavior.

Before implementation begins, record pinned libjxl results on the supported
CPU architectures and choose separate stage, final-map, and score tolerances
from observed floating-point variance. CPU and Metal may use different
tolerances. In addition to numerical error, end-to-end tests must compare the
AQ score history, final quant field, and raw quant field. Any decision
difference requires an explicit test and rationale.

Benchmarks must rebuild the tested targets, alternate implementations over the
same images, report variance, and distinguish one-time reference preparation
from per-comparison work.

## Milestones

### 1. Expand the libjxl oracle harness — complete (2026-08-26)

- Factor fixture generation so the same inputs can drive two backends.
- Compare every distance-map pixel rather than four selected samples.
- Add a test-only generation path for pinned blur, opsin, frequency, Malta,
  masking, and final-composition fixtures.
- Add small, odd, strided, impulse, deterministic-random, and real-image
  cases.
- Add option coverage and end-to-end iterative-AQ parity checks.
- Record initial correctness tolerances and libjxl timing/memory baselines.

Exit criterion: the harness detects deliberate perturbations to blur borders,
opsin constants, masking, Malta stencils, and score reduction.

The infrastructure work added exact scalar-generated full maps and 27
intermediate planes, an exact regeneration check, live full-map differential
coverage, injected allocation failures, a real-image crop, a two-update
iterative-AQ pin, and a dedicated interleaved benchmark. The test comparison
limits are
`1e-5 + 5e-6 * abs(expected)` per map pixel, `1e-5` for aggregate scores,
`1e-7` for identity maps, and `2e-5` for pinned iterative-AQ floating outputs.
Raw quant values and poisoned padding remain exact checks.

The scalar header is a reproducible source fixture, while facade parity is
checked against the live dynamically dispatched oracle. On an M4 Pro with
AppleClang 17, pinned libjxl's own scalar and dispatched Highway targets differ
by more than the map/score comparison limits. Those differences are recorded
instead of hiding them by widening the limits:

| Fixture group | Maximum absolute error | Maximum relative error | Maximum score error | Maximum limit ratio |
|---|---:|---:|---:|---:|
| Four 32x24 full maps | 0.000135899 | 0.000137113 | 0.0000467300 | 6.10x |
| 16x12 intermediate planes | 0.000701904 | 0.0252102 | n/a | 4.71x |

The relative stage maximum occurs near zero. Exact scalar regeneration and the
fixed facade-vs-live limits are separate gates. Scalar-vs-dispatched
compatibility has its own assertion-based absolute limits: `3e-4` per full-map
pixel, `1.5e-3` per intermediate-stage value, and `1e-4` for aggregate scores.
These rounded limits are slightly more than twice the observed M4 Pro maxima;
they do not weaken native facade-vs-live comparisons. Other CPU architectures
use the same policy and must provide new evidence before changing it.

Release timings below combine three independent runs. Each run used three
warmup rotations followed by 15 samples, rotating the order of one-shot,
reference-preparation, and prepared-comparison phases. The median column is the
range of the three run medians; the observed column spans all 45 samples.
These are baselines, not improvement claims.

| Workload | Phase | Run-median range (ms) | All observed samples (ms) |
|---|---|---:|---:|
| Synthetic 128x96 | One shot | 3.215–3.232 | 2.999–3.443 |
| Synthetic 128x96 | Reference preparation | 0.846–0.849 | 0.844–0.897 |
| Synthetic 128x96 | Prepared comparison | 2.073–2.079 | 2.049–2.530 |
| Flower 510x532 | One shot | 69.266–70.225 | 68.099–71.791 |
| Flower 510x532 | Reference preparation | 19.761–19.898 | 18.701–20.684 |
| Flower 510x532 | Prepared comparison | 50.350–50.438 | 48.970–55.051 |

Memory numbers are peak live bytes observed through the supplied
`JxlMemoryManager`. They intentionally exclude comparator objects,
standard-library containers, allocator metadata, and other process memory.
The prepared-comparison total includes retained reference state; the
incremental column subtracts that retained baseline.

| Workload | One-shot peak | Preparation peak | Prepared retained | Comparison total peak | Comparison incremental peak |
|---|---:|---:|---:|---:|---:|
| Synthetic 128x96 | 3,507,968 B (3.345 MiB) | 1,932,288 B (1.843 MiB) | 1,349,376 B (1.287 MiB) | 3,317,504 B (3.164 MiB) | 1,968,128 B (1.877 MiB) |
| Flower 510x532 | 61,789,440 B (58.927 MiB) | 31,976,064 B (30.495 MiB) | 22,360,320 B (21.324 MiB) | 58,310,400 B (55.609 MiB) | 35,950,080 B (34.285 MiB) |

### 2. Establish native CPU image and blur primitives — complete (2026-08-26)

- Introduce internal owned-image and reusable scratch types with checked extent
  arithmetic.
- Implement normalized separable Gaussian convolution with libjxl-compatible
  boundary behavior.
- Cover all kernel radii used by opsin dynamics, frequency separation, and
  masking.
- Keep the first implementation scalar and deterministic.

Exit criterion: every blur output matches libjxl over the differential corpus,
including borders and images smaller than a kernel.

gjxl now owns contiguous single-plane and planar three-channel float storage,
checked atomic resizing, reusable transposed/kernel scratch, and a scalar blur
for all five pinned sigmas. The non-aliasing 5-tap path mirrors coordinates
outside the image, repeating each edge pixel once. The 7-, 13-, 15-, and
33-tap paths instead clip support at each edge, renormalize each coordinate,
and apply two transposed scalar passes. Both modes preserve strided input and
output padding; validation and scratch allocation finish before output writes.

The five 16x12 scalar blur goldens pass the fixed
`1e-5 + 5e-6 * abs(expected)` limit. On an M4 Pro with AppleClang 17, the
maximum native-versus-scalar absolute error was `0`; the maximum
native-versus-dispatched absolute error over every reference and distorted
plane, all channels, all sigmas, and the complete differential corpus was
`1.78813934e-07`, below the fixed `1.5e-3` stage cap. Direct-oracle allocation
failure injection covers every managed allocation without leaks or partial
output. This milestone adds primitives only; the public facade remains on the
pinned libjxl path until native pipeline integration in later milestones.

### 3. Implement native opsin and frequency decomposition — complete (2026-08-26)

- Port absorbance mixing, gamma response, and linear RGB to XYB conversion.
- Implement LF/MF, MF/HF, and HF/UHF separation.
- Implement range removal, amplification, clamping, and X-by-Y suppression.
- Expose internal stage hooks to differential tests without making them public
  API.

Exit criterion: all intermediate planes match the pinned oracle within the
stage tolerances for every corpus category.

The native module now converts linear RGB to opsin-dynamics XYB and separates
it into three LF planes, three MF planes, two HF planes, and two UHF planes.
The implementation preserves the pinned absorbance coefficients, gamma
approximation, intensity scaling, low-frequency conversion, range transforms,
maximum clamps, X-by-Y suppression, and scalar evaluation order. In
particular, scalar multiply-adds explicitly preserve Highway scalar's
intermediate rounding instead of depending on the compiler's contraction
mode.

Opsin conversion stages the complete result before committing it, supports
exact input/output aliasing, and preserves strided output padding on failure.
Frequency decomposition builds a complete candidate image and move-commits it
only after all stages are finite. Reusable opsin and frequency scratch storage
owns the temporary blurred images and Gaussian state. The public Butteraugli
facade remains on libjxl.

All 13 native opsin/frequency planes match the existing 16x12 scalar goldens
exactly. On an M4 Pro with AppleClang 17, the maximum native-versus-dispatched
absolute error over both images in every differential-corpus fixture was
`0.000915527344`, below the fixed `1.5e-3` stage cap. The direct single-image
oracle covers every managed-allocation failure without leaks or partial output,
and the native tests cover invalid/non-finite inputs, exact in-place opsin
conversion, constant fields, threshold behavior, small images, strides,
padding, and scratch reuse.

### 4. Implement native difference and masking stages — complete (2026-08-26)

- Implement symmetric and asymmetric L2 terms.
- Implement LF and full Malta neighborhood responses with exact edge rules.
- Implement masking precomputation, fuzzy erosion, and channel combination.
- Implement the final distance-map composition and maximum-score reduction.

Exit criterion: the complete native CPU map and score pass full-map
differential tests, including small-image expansion and cropping behavior.

The native implementation now contains symmetric and asymmetric L2 terms,
the 16-direction LF and full Malta responses, masking precomputation and
step-three fuzzy erosion, AC/DC channel composition, and maximum-score
reduction. Malta reads zero outside its 9x9 support at image borders. The
complete-map path separately edge-replicates dimensions smaller than eight,
crops the expanded result, and adds exactly one half-resolution scale for
images at least 15x15. All results are staged before output writes, reusable
scratch owns the intermediate images, and the public facade remains on
libjxl pending Milestone 5.

The strict 16x12 scalar stage goldens and four 32x24 scalar map/score goldens
pass. A dedicated pinned-scalar executable also covers every eligible stage,
complete map, and score over the full corpus. On an M4 Pro Release build, the
maximum complete-corpus scalar errors were `3.05175781e-05` for a stage and
`4.76837158e-07` for both the map and score; every value passes the strict
scalar map formula and `1e-5` score limit.

Pinned libjxl's dispatched path differs from its scalar path most visibly on
expanded 1x1 impulses. Native-versus-dispatched comparison therefore remains
a separate cross-target gate using the existing architecture-independent
`0.0015` stage cap for stages, maps, and scores, without changing the original
facade/golden limits. The measured maxima were `0.000679016113` for stages and
`0.00114440918` for both maps and scores. No architecture-specific tolerance
branches are used.

### 5. Integrate and harden the native CPU backend — complete (2026-08-26)

- Make the native implementation available through
  `ComputeButteraugliDistance` while retaining a test-only way to select the
  libjxl oracle.
- Preserve input validation, finite-value checks, strided views, and atomic
  output commits.
- Run iterative AQ and complete quantization-pipeline parity tests.
- Benchmark scalar native CPU against libjxl/Highway before choosing
  optimization work.
- Preserve upstream license notices for algorithm code adapted from libjxl.

Exit criterion: native CPU can run the complete pipeline with
`GJXL_ENABLE_LIBJXL_REFERENCE=OFF`, and the enabled build passes differential
and end-to-end parity tests.

The public facade now maps `ButteraugliOptions` directly to the native scalar
implementation and creates scratch storage per call. This keeps concurrent
calls independent while preserving strided views, exact output aliasing,
poisoned padding, and atomic map/score commits. Invalid caller inputs are
reported as `InvalidArgument`, allocation failures as `OutOfMemory`, and
non-finite computed results as internal failures. The facade, iterative AQ,
and full CPU quantization-pipeline tests are built in both reference modes;
the unavailable-backend target has been removed.

The four strict scalar full-map goldens produced a maximum native-facade map
error of `4.76837158e-07`, a maximum score error of `4.76837158e-07`, and a
maximum strict-limit ratio of `0.0193271134`. Across the complete scalar
corpus, the maximum native difference-stage error was `3.05175781e-05`; map
and score errors remained `4.76837158e-07`. The dispatched libjxl comparison
remains a separate architecture-independent gate: observed stage error was
`0.000679016113`, and map and score errors were `0.00114440918`, all below the
fixed `0.0015` cap. Scalar goldens retain the
`1e-5 + 5e-6 * abs(expected)` map limit and `1e-5` score limit.

The existing two-update AQ pin passes without revision. Maximum observed
errors were `5.7220459e-06` for score history, `9.83476639e-06` for the final
quant field, and `1.52587891e-05` for block distances; raw quant values matched
exactly. All remain under the fixed `2e-5` pin limit. Five complete pipeline
fixtures now pin all 15 score-history values, 72 final quant-field values, and
72 raw quant decisions from the pre-integration pinned scalar libjxl path. The
native run's maximum errors were `1.1920929e-07` for score history and
`5.96046448e-08` for final quant fields, while every raw quant decision matched
exactly. Comparisons use the same `2e-5` floating-point limit and exact raw
quant equality.

Release timings below combine three independent benchmark processes. Each
process used three warmup rotations followed by 15 samples of four
equal-fixture phases. The median column is the range of the three process
medians; the observed column spans all 45 samples. The scalar implementation
is currently slower than dispatched libjxl on both fixtures, so optimization
work should follow this end-to-end evidence rather than infer a speedup from
the integration alone.

| Workload | Phase | Run-median range (ms) | All observed samples (ms) |
|---|---|---:|---:|
| Synthetic 128x96 | Native one shot | 8.237–8.929 | 8.032–15.158 |
| Synthetic 128x96 | Dispatched libjxl one shot | 3.300–3.619 | 3.027–7.193 |
| Synthetic 128x96 | Libjxl reference preparation | 0.881–0.940 | 0.859–2.420 |
| Synthetic 128x96 | Prepared libjxl comparison | 2.265–2.688 | 2.125–5.027 |
| Flower 510x532 | Native one shot | 184.862–188.086 | 183.099–190.338 |
| Flower 510x532 | Dispatched libjxl one shot | 72.001–73.212 | 70.868–83.998 |
| Flower 510x532 | Libjxl reference preparation | 20.283–20.757 | 19.413–21.430 |
| Flower 510x532 | Prepared libjxl comparison | 52.096–52.490 | 50.162–68.145 |

The benchmark continues to report only bytes observed through libjxl's
supplied `JxlMemoryManager`, explicitly labeled as libjxl-managed memory. Those
numbers do not measure native allocations, standard-library allocations,
allocator metadata, or total process memory.

### 6. Remove libjxl from the production dependency graph

- Remove dormant libjxl and Highway linkage from `gjxl_codec`.
- Restrict libjxl includes, sources, and Highway linkage to reference tests and
  reference benchmarks.
- Keep an explicit build option for maintainers who need live differential
  validation.
- Verify installation and downstream static linking without the submodule.

Exit criterion: the default library and its consumers build and run without
initializing libjxl, while a reference-enabled CI configuration continues to
guard parity.

This build/dependency cleanup may proceed in parallel with Metal work. It is not
a prerequisite for profiling AQ, adding shared GPU infrastructure, or beginning
the device Butteraugli operation. Before removing the production reference
linkage, document whether the slower native scalar CPU path is an accepted
fallback or requires a separate CPU optimization effort.

### 7. Define the standalone device Butteraugli operation

- Depend on the backend-neutral device-image, submission, scratch, and
  reduction contracts established by Milestone 1 of
  [`metal-aq.md`](metal-aq.md).
- Add an explicit operation under `src/gpu/ops/`; do not route device buffers
  through the synchronous CPU-view facade.
- Accept device-resident reference and distorted linear RGB images and write a
  device-resident pixel distance map plus an aggregate score.
- Specify options, extent, stride, offset, buffer-capacity, device-ownership,
  and aliasing requirements.
- Separate one-time preparation, repeated comparison, optional diagnostic
  readback, and score readback in the operation contract.
- Validate all host-visible descriptors before submitting GPU work.

Exit criterion: operation and prepared-state contracts are documented and
covered by validation/lifetime tests before Butteraugli kernels replace any
test stub or staged path.

### 8. Port Butteraugli leaf stages to Metal

- Port opsin conversion and pointwise nonlinear operations.
- Port separable blurs and frequency decomposition.
- Port symmetric/asymmetric difference, Malta neighborhoods, masking, fuzzy
  erosion, final composition, and maximum-score reduction.
- Reuse scratch planes according to a documented lifetime schedule with no
  steady-state allocation.
- Compare every eligible intermediate output with the native scalar CPU oracle
  and pinned libjxl fixtures.
- Cover partial threadgroups, images smaller than a kernel, odd and strided
  dimensions, poisoned output, and unsupported aliasing.

Exit criterion: the Metal distance map and score meet fixed numerical
tolerances on the complete corpus without uninitialized, stale, or
out-of-bounds output.

### 9. Add prepared-reference reuse and qualify standalone performance

- Cache the original image's opsin and frequency decomposition for repeated AQ
  comparisons.
- Prove one-shot and prepared paths are numerically equivalent within the same
  decision-level contract.
- Reuse device buffers and scratch across comparisons without retaining mutable
  state between independent prepared objects.
- Benchmark preparation, resident comparison, and full-E2E comparison
  separately with balanced execution order.
- Report upload/readback costs, persistent bytes, peak scratch bytes, and the
  CPU/Metal crossover on the workload set defined in
  [`metal-aq.md`](metal-aq.md).

Exit criterion: repeated prepared comparisons require no reference
recomputation or steady-state allocation and provide a stable standalone
full-E2E speedup at the target image sizes.

The strategy-aware 16-norm block reduction, CPU quant-field update, resident
reconstruction chain, and complete AQ speedup gate are not Butteraugli
milestones. They are tracked in [`metal-aq.md`](metal-aq.md), which consumes the
device-resident distance map produced here.

### 10. Consolidate and maintain Butteraugli

- Remove Butteraugli-specific transitional selectors and duplicate scratch
  paths after CPU and Metal contracts stabilize.
- Document numerical tolerances, known CPU/Metal deviations, standalone memory
  use, and prepared-reference behavior.
- Keep pinned full-map and intermediate fixtures so ordinary native tests do
  not require libjxl.
- Retain periodic reference-enabled CI or an explicit compatibility job for
  intentional libjxl revision updates.

Exit criterion: native CPU is the standalone correctness baseline, the device
operation is an optional accelerated implementation, and libjxl is required
only for deliberate compatibility validation.

## Recommended implementation order

Milestones 1 through 5 establish the correctness baseline and are complete.
Milestone 6 is independent build cleanup and may proceed in parallel. Shared
device-image and submission infrastructure is now validated by Milestone 1 of
[`metal-aq.md`](metal-aq.md); complete the operation contract in Milestone 7
before porting leaf kernels in Milestone 8. Prepared-reference optimization and
standalone qualification follow in Milestone 9. Full iterative-AQ integration
proceeds only through the gates in `metal-aq.md`, while the native CPU map
remains the primary readable oracle.
