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

The CPU reference preserves floating-point DC separately, matching VarDCT's
AC/DC split. It intentionally does not yet run libjxl's optional
`AdjustQuantBlockAC` encoder heuristic; fixed-raw-quant coefficient coding and
decoder reconstruction are parity-tested independently of that heuristic.

Relevant implementations:

- [`coeff_store.h`](../src/core/coeff_store.h)
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
- Add maximum-error behavior only when the primary perceptual loop is stable.

`FindBestQuantization` performs the default two field updates and three
measurements used by libjxl. It derives the same asymmetric bounds from the
adjusted initial field, applies the `0.2` under-target power, advances a field
value by one quantizer scale when rounding would otherwise stall, and applies
the second-update clamp toward the initial field. A caller may request zero to
four updates; maximum-error mode remains deliberately outside this milestone.

Each measurement recomputes raw quantization, final CfL, and EPF sigma before
coefficient coding, reconstruction, loop filtering, XYB-to-linear conversion,
Butteraugli comparison, and transform-aware 16-norm block reduction. The API
accepts an unpadded linear reference and a padded XYB coding image, and commits
all outputs only if every evaluation succeeds.

Relevant implementations:

- [`adaptive_quantization.cpp`](../src/codec/adaptive_quantization.cpp)
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

`RunCpuQuantizationPipeline` is the integration boundary. It runs initial AQ
on the pre-Gaborish XYB image, applies inverse Gaborish when enabled, computes
first-pass CfL, selects strategies, initializes EPF sharpness, and invokes the
iterative AQ loop. Its outputs expose the initial maps, selected strategy grid,
final float and raw quant fields, final CfL, score history, block distance map,
and reconstructed linear RGB image.

The integration corpus covers odd dimensions and edge padding, gradients,
texture, hard edges, saturated primaries, and a 64-pixel CfL tile boundary.
Earlier pinned fixtures cover initial-AQ encoding-tile boundaries, individual
AC candidate costs and maps, coefficient layout/reconstruction, CfL, EPF, and
Butteraugli map values.

Relevant implementations:

- [`quantization_pipeline.cpp`](../src/codec/quantization_pipeline.cpp)
- [`quantization_pipeline_test.cpp`](../tests/quantization_pipeline_test.cpp)
- [`quantization_benchmark.cpp`](../benchmarks/quantization_benchmark.cpp)

## Historical CPU performance baseline

Release build on an Apple M4 Pro with 48 GB RAM, measured on 2026-08-25. The
synthetic workload is 128x96 linear RGB, includes a 64-pixel CfL boundary, and
uses two AQ updates. Values are the median of the three run medians; ranges
span all 15 samples from three consecutive invocations.

This measurement predates the native Butteraugli facade and the later shared
image/scratch refactors. It remains a historical comparison point, not the
current Metal-AQ baseline. Milestone 0 of
[`metal-aq.md`](metal-aq.md) requires a refreshed full-pipeline and per-evaluation
breakdown before GPU implementation choices are treated as performance claims.

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
cmake --build build-release -j --target gjxl_quantization_benchmark
./build-release/gjxl_quantization_benchmark
```

## Accuracy scope and known deviations

The CPU pipeline is an executable reference for the currently supported seven
DCT strategies and fixed-raw-quant coefficient path. It is not yet a complete
libjxl encoder replacement:

- DC is preserved as floating-point LLF data. Modular DC quantization,
  DC chroma-from-luma, and adaptive DC smoothing are not modeled yet.
- The optional encoder-side `AdjustQuantBlockAC` heuristic is not applied.
- Maximum-error AQ, resampling-specific AQ bypasses, HDR transfer functions,
  and non-default opsin matrices are outside the current contract.
- Exact pinned parity applies to the individual numerical stages and fixed
  coefficient fixtures. Because of the DC and coefficient-heuristic omissions,
  the complete score trajectory is tested for deterministic composition and
  the libjxl update rule, not claimed to be bit-identical to a full libjxl
  encode round trip.

These deviations do not block GPU ports of the established leaf operations.
They do block claiming complete encoder or bitstream parity.

## GPU-porting boundary

The actionable GPU roadmap is [`metal-aq.md`](metal-aq.md). This document
remains authoritative for CPU algorithm order, supported strategies, known
codec deviations, and reference outputs.

Initial quantization and AC-strategy search are siblings of iterative AQ and
remain separate GPU efforts. Within iterative AQ, quant-field convergence,
clamping, and rounding-progress decisions initially stay on the CPU. The Metal
path accelerates the complete encode/reconstruct/measure evaluation and reads
back only the block-distance map and score required by that unchanged policy.

Every GPU stage must be compared with the CPU reference and, where applicable,
the pinned libjxl oracle. Direct intermediate comparisons, final AQ decisions,
resident timings, and full end-to-end timings are all required; a GPU round
trip or isolated kernel speedup is not sufficient evidence of parity or useful
acceleration.
