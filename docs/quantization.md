# Quantization and adaptive quantization roadmap

This document tracks the CPU quantization and adaptive-quantization reference
pipeline. The implementations provide executable specifications, pinned
libjxl parity data, and independent correctness oracles for later GPU kernels.
The cross-operation Metal implementation plan, residency contract, validation
matrix, and rollout gate live in [`metal-aq.md`](metal-aq.md).

The current reference revision is libjxl
`e8ff09762481785938d8e4e01333ed3917571161`.

## Pipeline order

Initial quantization, AC-strategy search, and iterative adaptive quantization
are distinct stages. The intended dependency order is:

```text
opsin image ────────────────> initial quant field and masking maps
     │
     └─> Gaborish preprocessing ─> first-pass CfL
                                      │
initial quant field + masks + CfL ────┴─> AC-strategy search
                                               │
initial quant field + selected strategies ─────┴─> adjusted quant field
                                                        │
adjusted field ─> raw quant field ─> second-pass CfL ───┤
                                                        │
                        encode/reconstruct/measure <────┘
                                      │
                                      └─> iterative AQ
```

AC-strategy search is a sibling and consumer of the initial quantization
heuristic. Iterative AQ runs after strategies and the initial raw quant field
have been selected.

## Milestones

### 1. DCT foundation — complete

- Maintain an independent double-precision CPU transform oracle.
- Support forward and inverse Metal transforms for DCT8, DCT16x16,
  DCT32x32, DCT16x8, DCT8x16, DCT32x16, and DCT16x32.
- Validate coefficient layout, forward output, inverse output, and round trips.
- Retain scalar and SIMD-group benchmark coverage.

### 2. Quantization foundation — complete

- Represent quantizer state and block-resolution raw quant fields.
- Provide pinned default quantization matrices for the supported strategies.
- Quantize and dequantize AC coefficients.
- Convert between multiblock low-frequency coefficients and the DC grid.
- Keep shared geometry and fast-math contracts header-only.

Relevant implementations:

- [`quantization.cpp`](../src/codec/quantization.cpp)
- [`dc_conversion.cpp`](../src/codec/dc_conversion.cpp)
- [`fast_math.h`](../src/util/fast_math.h)

### 3. Initial adaptive-quantization estimate — complete

- Compute `InitialQuantDC` from the Butteraugli target.
- Compute the initial DCT8 quant field from a padded opsin image.
- Produce the block-resolution strategy mask.
- Produce the blurred pixel-resolution masking field.
- Match pinned libjxl outputs, including samples across encoding-tile edges.

Relevant implementation:

- [`adaptive_quantization.cpp`](../src/codec/adaptive_quantization.cpp)

### 4. AC-strategy data model — complete

- Add an `AcStrategyGrid` contract to
  [`ac_strategy.h`](../src/core/ac_strategy.h).
- Encode the selected strategy for every 8x8 base block.
- Distinguish a multiblock transform's top-left anchor from its covered cells.
- Reject out-of-bounds placements and overlapping transforms.
- Support iteration over transform anchors without visiting covered cells as
  independent transforms.
- Add a production CPU transform and candidate-cost oracle for search.

This milestone is shared infrastructure for both AC search and quant-field
adjustment; it should not contain search policy.

Relevant implementations:

- [`ac_strategy.h`](../src/core/ac_strategy.h)
- [`dct.cpp`](../src/codec/dct.cpp)
- [`ac_strategy.cpp`](../src/codec/ac_strategy.cpp)

### 5. AC-search preprocessing — complete

- Implement the CPU Gaborish preprocessing used before strategy selection.
- Implement the first-pass chroma-from-luma map consumed by AC cost estimates.
- Validate intermediate images and CfL factors against pinned libjxl.

The initial quant field intentionally runs on the pre-Gaborish opsin image.
Strategy selection consumes the preprocessed image and first-pass CfL result.

Relevant implementations:

- [`gaborish.cpp`](../src/codec/gaborish.cpp)
- [`chroma_from_luma.cpp`](../src/codec/chroma_from_luma.cpp)

### 6. CPU AC-strategy search — complete

- Implement libjxl-compatible entropy and information-loss estimates.
- Evaluate transform candidates and choose non-overlapping placements.
- Initially limit the candidate set to strategies supported end-to-end:
  DCT8, DCT16x8, DCT8x16, DCT16x16, DCT32x16, DCT16x32, and DCT32x32.
- Compare individual candidate costs and complete strategy maps with pinned
  libjxl fixtures.
- Add adversarial tests for image edges, partial candidate regions, ties, and
  multiblock overlap.

Additional strategies must not enter the search until their transforms,
coefficient layouts, quantization matrices, and reconstruction paths are all
supported.

Relevant implementation:

- [`ac_strategy.cpp`](../src/codec/ac_strategy.cpp)

### 7. Post-search quantization finalization — complete

- Implement `AdjustQuantField` for the selected multiblock strategies.
- Convert the adjusted float field to the raw quant field.
- Compute the second-pass CfL map from strategies and raw quant values.
- Generate the EPF control and sharpness fields required by reconstruction.
- Validate adjusted fields and raw quant values independently.

Relevant implementations:

- [`adaptive_quantization.cpp`](../src/codec/adaptive_quantization.cpp)
- [`chroma_from_luma.cpp`](../src/codec/chroma_from_luma.cpp)
- [`epf.cpp`](../src/codec/epf.cpp)

### 8. CPU reconstruction path — complete

- Produce transformed coefficients for the selected strategy grid.
- Apply DC/LLF conversion and AC quantization/dequantization.
- Reconstruct pixels with the matching inverse transforms.
- Apply inverse CfL, Gaborish, and EPF behavior needed by the encoder's
  perceptual round trip.
- Compare intermediate coefficients and reconstructed images with libjxl.

The CPU reference stores modular-stream quantized DC separately from AC and
retains its decoder-equivalent floating-point reconstruction for the AQ round
trip. Production coefficient coding applies the pinned `AdjustQuantBlockAC`
cross-channel shared-quant decision and adjusted Y dead zones. The fixed-raw-
quant mode remains an explicit diagnostic oracle. Experimental resident Metal
coefficient coding applies the same heuristic to its device FP32 forward
coefficients; those inputs are not claimed to be identical to the CPU's
double-precision forward transform.

Relevant implementations:

- [`vardct_frame.h`](../src/codec/vardct_frame.h)
- [`reconstruction.cpp`](../src/codec/reconstruction.cpp)
- [`loop_filter.cpp`](../src/codec/loop_filter.cpp)
- [`gaborish.cpp`](../src/codec/gaborish.cpp)
- [`epf.cpp`](../src/codec/epf.cpp)

### 9. Perceptual scoring — complete

- Implement the CPU Butteraugli-compatible distance map required by AQ.
- Compute the aggregate perceptual score used for convergence.
- Validate both the full distance map and scalar score against pinned fixtures.
- Include flat, textured, high-contrast, and chromatic test images.

The CPU metric uses gjxl's native scalar Butteraugli implementation through the
backend-neutral, view-based `ComputeButteraugliDistance` facade. The pinned
libjxl implementation remains available only as a differential oracle when
`GJXL_ENABLE_LIBJXL_REFERENCE=ON`; the option defaults to `OFF`. The production
codec target has no libjxl or Highway dependency, and the native facade,
iterative AQ, and complete quantization pipeline build and run without either
reference dependency.

The native implementation, its intermediate-stage oracles, and the standalone
device Butteraugli operation are tracked in [`butteraugli.md`](butteraugli.md).
The complete resident reconstruction and AQ integration path is tracked in [`metal-aq.md`](metal-aq.md).

Relevant implementation:

- [`butteraugli.cpp`](../src/codec/butteraugli.cpp)

### 10. Iterative adaptive quantization — complete

- Implement the `FindBestQuantizer`-style encode/reconstruct/measure loop.
- Update the quant field from the perceptual distance map.
- Match libjxl convergence, clamping, and stopping behavior.
- Keep iteration order deterministic.
- Keep Butteraugli and maximum-error policies independently testable.

`FindBestQuantization` performs the default two field updates and three
measurements used by libjxl. It derives the same asymmetric bounds from the
adjusted initial field, applies the `0.2` under-target power, advances a field
value by one quantizer scale when rounding would otherwise stall, and applies
the second-update clamp toward the initial field. A caller may request zero to
four Butteraugli updates.

Maximum-error mode uses the alternate pinned transform-local update rule over
normalized XYB reconstruction error. It performs five updates and a fixed final
verification, ignores padded source pixels, and reports whether the selected
field met the limit, exhausted the iteration budget, or reached the
representable quantization bound. See [`rate-control.md`](rate-control.md) for
the public contract and Metal completion boundary.

Each measurement recomputes raw quantization, final CfL, and EPF sigma before
coefficient coding, reconstruction, loop filtering, XYB-to-linear conversion,
Butteraugli comparison, and transform-aware 16-norm block reduction. The API
accepts an unpadded linear reference and a padded XYB coding image, and commits
all outputs only if every evaluation succeeds.

Relevant implementations:

- [`adaptive_quantization.cpp`](../src/codec/adaptive_quantization.cpp)
- [`maximum_error.cpp`](../src/codec/maximum_error.cpp)
- [`color_transform.cpp`](../src/codec/color_transform.cpp)
- [`adaptive_quantization_loop_test.cpp`](../tests/adaptive_quantization_loop_test.cpp)

### 11. CPU integration gate — complete for the supported reference subset

GPU porting begins only after the CPU pipeline satisfies all of the following:

- End-to-end fixtures pass through initial quantization, strategy selection,
  quant-field adjustment, quantization, reconstruction, and iterative AQ.
- Intermediate fields and supported fixed-path outputs have pinned libjxl
  parity coverage.
- Independent oracles cover layout-sensitive transforms and coefficient paths.
- A varied image corpus covers odd dimensions, padding, tile boundaries,
  smooth gradients, texture, edges, and saturated colors.
- Stage-level CPU timings establish performance baselines.
- Accuracy tolerances and known deviations are documented explicitly.

`RunQuantizationPipeline` is the backend-neutral integration boundary. It runs
initial AQ on the pre-Gaborish XYB image, applies inverse Gaborish when enabled,
computes first-pass CfL, delegates strategy selection to an injected provider,
initializes EPF sharpness, and invokes the iterative AQ loop. The CPU wrapper
uses `FindAcStrategyGrid`; the GPU wrapper uses `FindAcStrategyGridGpu`. Their
outputs expose the initial maps, a completed
`VarDctEncoderFrame`, the final float quant field, score history, block distance
map, and reconstructed linear RGB image.

The integration corpus covers odd dimensions and edge padding, gradients,
texture, hard edges, saturated primaries, and a 64-pixel CfL tile boundary.
Earlier pinned fixtures cover initial-AQ encoding-tile boundaries, individual
AC candidate costs and maps, coefficient layout/reconstruction, CfL, EPF, and
Butteraugli map values.

Relevant implementations:

- [`quantization_pipeline.cpp`](../src/codec/quantization_pipeline.cpp)
- [`quantization_pipeline_test.cpp`](../tests/quantization_pipeline_test.cpp)
- [`encoding_benchmark.cpp`](../benchmarks/encoding_benchmark.cpp)

### 12. Encoder-facing VarDCT frame — complete

`VarDctEncoderFrame` is the owned handoff between encoder analysis and future
entropy/bitstream coding. A successful final AQ evaluation commits one frame
containing source and padded geometry, the selected strategy grid, raw quant
field, quantizer, final CfL map, EPF sharpness, the retained codestream profile
and its three-bit X/B matrix scales, three modular-stream `int32_t` DC planes,
decoder-equivalent reconstructed DC, and grouped quantized AC coefficients.
Callers may release or reuse all borrowed inputs after frame construction.

AC storage follows the JPEG XL 256x256-pixel group grid. Each group owns three
fixed 65,536-element `int32_t` channel rows. Groups are indexed in row-major
order; within a group, complete transforms are appended in row-major anchor
order, and each transform contributes its native coefficient layout
contiguously. Edge groups expose their used coefficient count and guarantee a
zero-filled unused tail. A transform that would cross a group boundary is
rejected atomically.

DC uses one sample per 8x8 base block and channel. Quantization follows
libjxl's default 4:4:4 path with `extra_dc_precision = 0`: Y is quantized first,
X has no DC CfL prediction, and B is predicted from decoder-reconstructed Y
with a factor of one. The quantized `int32_t` planes are authoritative; the
frame also caches their decoder-equivalent floating-point reconstruction so AQ
measures actual DC loss. Modular gradient prediction and entropy tokenization
remain a later bitstream milestone.

Relevant implementations:

- [`vardct_frame.h`](../src/codec/vardct_frame.h)
- [`vardct_frame.cpp`](../src/codec/vardct_frame.cpp)
- [`dc_quantization.cpp`](../src/codec/dc_quantization.cpp)
- [`vardct_frame_test.cpp`](../tests/vardct_frame_test.cpp)
- [`dc_quantization_test.cpp`](../tests/dc_quantization_test.cpp)

## Historical CPU performance baseline

Release build on an Apple M4 Pro with 48 GB RAM, measured on 2026-08-25. The
synthetic workload is 128x96 linear RGB, includes a 64-pixel CfL boundary, and
uses two AQ updates. Values are the median of the three run medians; ranges
span all 15 samples from three consecutive invocations. This historical
baseline predates the fixed-row frame storage in milestone 12 and must be
refreshed before using it to judge that representation's cost.

This measurement predates the native Butteraugli facade and the later shared
image/scratch refactors. It remains a historical comparison point, not the
current Metal-AQ baseline. The refreshed full-pipeline, per-evaluation, and
peak-RSS results are recorded under completed Milestone 0 of
[`metal-aq.md`](metal-aq.md).

| Stage | Median | Observed range |
| --- | ---: | ---: |
| Initial quant field | 0.343 ms | 0.324–0.370 ms |
| Inverse Gaborish | 0.586 ms | 0.557–0.621 ms |
| Initial CfL | 1.117 ms | 1.078–1.185 ms |
| AC-strategy search | 9.494 ms | 9.357–9.763 ms |
| Iterative AQ | 25.914 ms | 24.931–26.678 ms |
| Complete pipeline | 37.778 ms | 36.583–38.890 ms |

Reproduce the baseline with:

```sh
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_BUILD_TESTS=ON \
  -DGJXL_BUILD_BENCHMARKS=ON
cmake --build build-release -j --target gjxl_encoding_benchmark
./build-release/gjxl_encoding_benchmark
```

## Batched Metal AC candidate evaluation

The Metal backend can evaluate one or more same-strategy candidate batches in
one command buffer and one compute encoder. It gathers candidate image regions,
runs the selected forward DCT,
computes quantization residuals and rate terms, runs the inverse DCT, and
reduces the masked information loss to one cost per candidate. Full planar
opsin, mask, and quantization-matrix buffers remain resident; only 24-byte
candidate descriptors and scalar costs need to cross the CPU/GPU boundary.
Search traversal and selection remain on the CPU.

Release measurements on an Apple M4 Pro on 2026-08-25 used simdgroup DCTs and
three independent invocations of 12 samples. Each invocation balances all six
CPU/resident/E2E measurement orders, immediately warms each GPU path before a
timed sample, and times at least 16 GPU submissions per sample. `E2E` includes
candidate upload, command submission and synchronization, and cost download;
image, mask, matrix residency, candidate construction, and quant-norm
aggregation are outside timing. Maximum batch sizes hold coefficient-pixel
work constant at 4096 DCT8-equivalent candidates.

| Strategy | Candidates | CPU median range | Metal E2E median range | E2E speedup range |
| --- | ---: | ---: | ---: | ---: |
| DCT8 | 4096 | 16.758–16.855 ms | 0.430–0.512 ms | 32.9–39.0x |
| DCT16x8 | 2048 | 18.970–19.554 ms | 0.438–0.501 ms | 38.7–44.7x |
| DCT8x16 | 2048 | 17.471–18.313 ms | 0.388–0.435 ms | 42.1–47.2x |
| DCT16x16 | 1024 | 22.588–22.757 ms | 0.402–0.428 ms | 53.1–56.6x |
| DCT32x16 | 512 | 31.185–31.480 ms | 0.393–0.591 ms | 53.3–79.3x |
| DCT16x32 | 512 | 29.299–29.544 ms | 0.462–0.578 ms | 51.0–63.4x |
| DCT32x32 | 256 | 38.236–38.270 ms | 0.470–0.613 ms | 62.5–81.4x |

Batching is necessary to amortize roughly 0.12 ms of command-buffer latency.
The smallest tested E2E batch that was faster in all three invocations was 32
candidates for DCT16x8 and DCT8x16, and 8 candidates for DCT16x16, DCT32x16,
DCT16x32, and DCT32x32. DCT8 at 32 candidates remained around break-even, so
its crossover lies above that tested point. A single large transform is also
around break-even; this is a throughput optimization, not a latency win.

Each invocation validates 10,783 CPU/GPU costs before timing. Two candidates
landed on a quantization discontinuity where float Metal DCT output rounded a
coefficient differently from the double-precision CPU DCT. The worst cost
difference was 0.566%; all other candidates met the tight unit-test tolerance.
The benchmark retains a 1% hard gate, while dedicated non-boundary fixtures
retain the tighter parity check for both scalar and simdgroup implementations.

Reproduce the benchmark with:

```sh
just ac-strategy-benchmark
```

The same benchmark accepts `device-quant` as its third recipe argument. That
mode binds checked resident opsin, mask, and quant-field views and poisons the
descriptor quant norms to `1.0`, so validation proves that the device field is
authoritative while timing includes its shader-side strategy aggregation:

```sh
just ac-strategy-benchmark 4096 12 device-quant
```

This mode was added while evaluating whether the residual kernel should share
one quant-norm calculation across a whole threadgroup. Both a threadgroup
barrier and a barrier-free SIMD-group broadcast were tested against the direct
per-thread expression on the Apple M4 Pro. Max-batch medians varied by strategy
and the proposed broadcasts did not provide a stable aggregate win, so the
production shader retains the direct expression. The benchmark coverage is
retained to keep future resident-field changes measurable instead of assuming
that source-level duplicate work survives Metal compilation or dominates the
kernel.

### Complete staged search

`FindAcStrategyGridGpu` precomputes every candidate anchor that the existing
search can request, grouped by strategy. It submits all seven groups through a
single command buffer and encoder, sharing scratch storage between groups, then
downloads the scalar costs. The original CPU traversal consumes the resulting
table without changing merge order, priorities, boundary checks, or tie policy.
For a complete 8x8-block color tile this stages 258 candidates; enumerating
every geometrically valid 32-point anchor would stage 320.

Candidate evaluation is exposed through the optional
`GpuAcStrategyEvaluation` capability rather than codec-specific virtual methods
on `GpuBackend`. A successful non-empty batch returns its own caller-owned
`GpuSubmission`; the search waits that exact submission before cost readback.
This keeps validation and completion failures attached to the command that
produced the costs and permits independent candidate submissions without a
backend-wide “latest command” synchronization slot.

Release measurements on an Apple M4 Pro on 2026-08-26 used three independent
invocations of 12 samples with alternating CPU/GPU order. GPU E2E includes
strided host packing, candidate and quant-norm construction, all buffer
allocations and uploads, one command submission and synchronization, cost
readback, and CPU decision traversal. Values below are ranges across the three
invocation medians.

| Image | Candidates | CPU median range | GPU E2E median range | Speedup range |
| --- | ---: | ---: | ---: | ---: |
| 64x64 | 258 | 2.784–2.813 ms | 0.630–1.132 ms | 2.48–4.63x |
| 128x96 | 752 | 8.328–8.417 ms | 0.603–1.631 ms | 5.11–13.90x |
| 256x192 | 3,096 | 33.311–33.821 ms | 1.984–3.505 ms | 9.52–17.38x |
| 512x384 | 12,384 | 132.686–134.425 ms | 6.310–10.166 ms | 13.11–22.15x |
| 480p (856x480 padded) | 25,594 | 273.142–276.185 ms | 9.771–14.291 ms | 19.51–28.49x |
| 720p (1280x720) | 57,720 | 619.433–625.679 ms | 20.474–22.030 ms | 29.14–31.02x |
| 1080p (1920x1080) | 130,380 | 1,391.920–1,417.010 ms | 42.388–45.615 ms | 30.58–33.10x |

The 480p case models a 16:9 854x480 source padded by two columns to complete
the final 8x8 block. GPU samples remain variable, especially for the smaller
allocations, so the table supports a substantial end-to-end throughput win but
not a claim that resource-management variance has been eliminated. At 1080p,
however, all three GPU run medians remained within 3.3 ms. Full-grid parity
tests cover full and partial color tiles,
three Butteraugli targets, strided inputs, invalid-input atomicity, and several
deterministic source phases. Quantization-boundary sensitivity remains governed
by the leaf-cost accuracy contract above.

Reproduce this benchmark with:

```sh
just ac-strategy-search-benchmark
```

### Full pipeline integration

`RunGpuQuantizationPipeline` injects GPU AC candidate evaluation and prepared
GPU adaptive quantization into the common codec orchestration without adding a
GPU dependency to `gjxl_codec`. In exact-coefficient mode, initial
quantization, Gaborish preprocessing, first-pass CfL, search decisions, and the
deterministic AQ update policy remain on the CPU. Fully resident mode instead
uses the backend image primitives for inverse Gaborish and selects the existing
deterministic tilewise pixel-domain initial-CfL seed; initial quantization and
policy decisions remain on the CPU. It prepares one strategy-aware final-CfL
map from the adjusted initial field and reuses that map for every throughput
evaluation; exact mode continues to recompute quant-dependent final CfL. The
direct prepared operation can perform coefficient coding, reconstruction,
filters, color conversion, and either Butteraugli or transform-local
maximum-error block reduction in one Metal submission. The qualified workflow
uses a
decision-preserving coefficient boundary: CPU coefficient coding,
dequantization, inverse CfL, and DC/LLF conversion prepare exact packed
reconstruction coefficients, then one prepared Metal submission performs
inverse transforms and source-domain filters. Butteraugli mode then performs
color conversion, comparison, and block reduction; maximum-error mode compares
resident coding and filtered reconstructed opsin directly. Both retain exact
coefficient decisions and avoid a redundant CPU reconstruction or metric pass.

The optional prepared-AQ capability is preflighted before pipeline work. A
backend without it returns `Unavailable` without changing output or search
statistics, and there is no silent CPU fallback. The bounded
`RunGpuAdaptiveQuantizationPolicy` API remains available when callers need only
the final field, block map, and score history; `RunGpuAdaptiveQuantization`
materializes the existing full adaptive-quantization output atomically. Their
source-compatible default overloads use the exact-coefficient composite
evaluator. Explicit overloads accept `GpuAdaptiveQuantizationMode`: the
production `kExactCoefficients` mode preserves CPU coefficient decisions,
while experimental `kFullyResident` runs forward transforms and coefficient
coding on the GPU, uses GPU Gaborish preprocessing and the faster initial-CfL
seed, and may change the quant field, frame, and codestream. The complete GPU
pipeline exposes the same explicit mode for both Butteraugli and maximum-error
control. Automatic maximum-error requests remain on CPU; forced Metal requests
may select either mode. This two-value public
contract replaces the temporary generic handoff selector without exposing the
discarded exact-linear or exact-opsin policy prototypes.

The benchmark accepts `--gpu-aq exact-coefficients|fully-resident`. Exact mode
retains the fixed rollout gate; fully resident mode completes normally and
reports CPU deltas for quant field, block map, score history, reconstructed
RGB, frame coefficients, and codestream bytes so numerical experiments do not
need a private API.

The Metal backend uses the same transform-dispatch helper for public DCT
batches and candidate evaluation. Scalar, SIMD-group, and factored radix-2
implementations therefore share packing, partial-threadgroup guards, and buffer
bindings. Candidate and full-search tests cover all three implementations.

The default two-update complete-pipeline fixture keeps quant field, block map,
score history, and reconstructed RGB below the fixed `2e-3` accumulated gate;
its observed maxima are `1.14441e-5`, `7.39694e-5`, `1.03533e-4`, and
`6.3777e-6`, respectively. Raw quantization, all frame state, and serialized
codestream bytes are exact. A separate 257x17 source crosses the 256-pixel
AC-group boundary and verifies the 1x3-block edge group with zero updates. Its
float-DCT-sensitive maximum block/RGB deviations are `2.21866e-2` and
`2.01165e-2`, below a separately fixed `2.5e-2` narrow-geometry boundary; its
frame and codestream remain exact.

The Milestone 9 corpus keeps the unchanged `2e-3` gate and observes maxima of
`1.838893e-3` for the float field, `2.745390e-4` for block distance,
`2.551079e-5` for score, and `5.960464e-6` for reconstructed RGB at validated
targets `1.0` and `1.2`. Raw quant, complete frame state, and codestream bytes
are exact across the built-in resolution sweep and four additional 1080p
images. Automatic workflow selection is limited to that target interval and is
documented in [`metal-aq.md`](metal-aq.md).

Relevant implementations:

- [`quantization_pipeline.h`](../src/codec/quantization_pipeline.h)
- [`quantization_pipeline.cpp`](../src/gpu/ops/quantization_pipeline.cpp)
- [`quantization_gpu_pipeline_test.cpp`](../tests/quantization_gpu_pipeline_test.cpp)

## Accuracy scope and known deviations

The CPU pipeline is an executable reference for the currently supported seven
DCT strategies, including the adjusted shared-quant coefficient path. It is
not yet a complete libjxl encoder replacement:

- Default 4:4:4 DC quantization and DC chroma-from-luma are modeled. Modular DC
  entropy tokenization and adaptive DC smoothing are not modeled yet.
- Resampling-specific AQ bypasses, HDR transfer functions, and non-default
  opsin matrices are outside the current contract.
- Exact pinned parity applies to the individual numerical stages and complete
  CPU coefficient decisions. The broader score trajectory is tested for
  deterministic composition and the libjxl update rule, not claimed to be
  bit-identical to every full libjxl encoder configuration.

These deviations do not block GPU ports of the established leaf operations.
They do block claiming complete encoder or bitstream parity.

## GPU-porting boundary

The actionable GPU roadmap is [`metal-aq.md`](metal-aq.md). This document
remains authoritative for CPU algorithm order, supported strategies, known
codec deviations, and reference outputs.

Initial quantization and AC-strategy search are siblings of iterative AQ and
remain separate GPU operations. Within iterative AQ, quant-field convergence,
clamping, and rounding-progress decisions stay on the CPU. Intermediate Metal
evaluations read back only the block-distance map and score required by that
unchanged policy. Direct resident calls may also read back reconstructed RGB,
quantized AC, and quantized DC. The qualified rollout retains the exact CPU
coefficient frame and uploads its dequantized, inverse-CfL, DC/LLF-replaced
coefficient stream. Metal performs inverse transforms and the image/perceptual
tail, and the last evaluation downloads the reconstructed source image while
returning an exact authoritative frame.

Every GPU stage must be compared with the CPU reference and, where applicable,
the pinned libjxl oracle. Direct intermediate comparisons, final AQ decisions,
resident timings, and full end-to-end timings are all required; a GPU round
trip or isolated kernel speedup is not sufficient evidence of parity or useful
acceleration.
