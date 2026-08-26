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
current implementation adapts gjxl images to the pinned libjxl implementation,
which produces a pixel-resolution distance map and its maximum score. gjxl
already owns the transform-aware 16-norm reduction of that map and the
iterative quant-field update.

The build compiles only the libjxl translation units required by Butteraugli,
plus Highway. `GJXL_ENABLE_LIBJXL_REFERENCE=OFF` removes that dependency, but
the distance computation and complete iterative quantization pipeline then
return `Unavailable`.

The current Metal abstraction supports allocation, transfers, synchronization,
and batched transforms. It does not yet model image operations, multi-pass
compute graphs, or reusable scratch storage.

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

Keep `ComputeButteraugliDistance` as the stable host-facing facade. During the
transition, split its implementations behind internal entry points so native
CPU and libjxl can be evaluated on the same inputs in one test process.

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

The libjxl reference must remain enabled while native parity is established.
Tests should compare complete distance maps, not only the aggregate score or
selected samples. Test-only fixtures generated from the pinned source should
also cover intermediate stages without exposing those stages as production
APIs.

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

### 1. Expand the libjxl oracle harness — infrastructure complete; acceptance pending

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
fixed facade-vs-live limits are separate gates; the scalar-vs-dispatched values
above are a diagnostic baseline, not an assertion-based acceptance gate.
Milestone 1 remains pending until a cross-target tolerance policy is adopted.

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

### 2. Establish native CPU image and blur primitives

- Introduce internal owned-image and reusable scratch types with checked extent
  arithmetic.
- Implement normalized separable Gaussian convolution with libjxl-compatible
  boundary behavior.
- Cover all kernel radii used by opsin dynamics, frequency separation, and
  masking.
- Keep the first implementation scalar and deterministic.

Exit criterion: every blur output matches libjxl over the differential corpus,
including borders and images smaller than a kernel.

### 3. Implement native opsin and frequency decomposition

- Port absorbance mixing, gamma response, and linear RGB to XYB conversion.
- Implement LF/MF, MF/HF, and HF/UHF separation.
- Implement range removal, amplification, clamping, and X-by-Y suppression.
- Expose internal stage hooks to differential tests without making them public
  API.

Exit criterion: all intermediate planes match the pinned oracle within the
stage tolerances for every corpus category.

### 4. Implement native difference and masking stages

- Implement symmetric and asymmetric L2 terms.
- Implement LF and full Malta neighborhood responses with exact edge rules.
- Implement masking precomputation, fuzzy erosion, and channel combination.
- Implement the final distance-map composition and maximum-score reduction.

Exit criterion: the complete native CPU map and score pass full-map
differential tests, including small-image expansion and cropping behavior.

### 5. Integrate and harden the native CPU backend

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

### 6. Remove libjxl from the production dependency graph

- Make native CPU the default implementation.
- Restrict libjxl includes, sources, and Highway linkage to reference tests and
  reference benchmarks.
- Keep an explicit build option for maintainers who need live differential
  validation.
- Verify installation and downstream static linking without the submodule.

Exit criterion: the default library and its consumers build and run without
initializing libjxl, while a reference-enabled CI configuration continues to
guard parity.

### 7. Add reusable Metal image-operation infrastructure

- Define device image views with extent, channel layout, and stride contracts.
- Add image-operation dispatch without expanding transform-specific types.
- Add reusable scratch allocation and explicit lifetime rules.
- Support sequencing multiple kernels in one command buffer and reducing a
  plane to a scalar maximum.
- Add test utilities for intermediate device-plane readback and comparison.

Exit criterion: pointwise operations, separable blur, and maximum reduction
have direct CPU-oracle tests over odd and strided dimensions.

### 8. Port Butteraugli leaf stages to Metal

- Port opsin conversion and pointwise nonlinear operations.
- Port separable blurs and frequency decomposition.
- Port Malta neighborhood kernels, masking, erosion, and final composition.
- Reuse scratch planes according to a documented lifetime schedule.
- Compare each intermediate output with both native CPU and libjxl.

Exit criterion: the Metal distance map and score meet their numerical
tolerances on the complete corpus without uninitialized or out-of-bounds
output.

### 9. Integrate Metal with iterative AQ

- Add explicit backend selection to the internal quantization orchestration.
- Keep reconstructed images and perceptual intermediates device-resident where
  that reduces transfers without coupling policy to Metal.
- Evaluate prepared-reference caching across repeated AQ comparisons.
- Measure complete encode/reconstruct/measure iterations, not isolated kernels
  alone.

Exit criterion: Metal preserves accepted AQ decisions and provides a stable,
repeatable end-to-end speedup over native CPU for representative image sizes.

### 10. Consolidate and maintain

- Remove transitional implementation selectors and duplicate scratch paths.
- Document numerical tolerances, known CPU/Metal deviations, memory use, and
  backend-selection behavior.
- Keep pinned full-map fixtures so ordinary tests do not require libjxl.
- Retain periodic reference-enabled CI or an explicit compatibility job for
  intentional libjxl revision updates.

Exit criterion: native CPU is the standalone correctness baseline, Metal is an
optional accelerated backend, and libjxl is required only for deliberate
compatibility validation.

## Recommended implementation order

Complete milestones 1 through 6 before treating the Metal backend as a product
path. Metal infrastructure may begin after the CPU blur contract is stable, but
the native CPU map should remain the primary readable oracle. Within the Metal
work, prioritize reusable convolution and scratch-management primitives before
Butteraugli-specific fusion or tuning.
