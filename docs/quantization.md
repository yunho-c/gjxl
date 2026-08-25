# Quantization and adaptive quantization roadmap

This document tracks the CPU reference pipeline that must be established before
moving additional quantization and adaptive-quantization work to Metal. The CPU
implementations provide executable specifications, pinned libjxl parity data,
and independent correctness oracles for later GPU kernels.

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

### 4. AC-strategy data model — next

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

### 5. AC-search preprocessing

- Implement the CPU Gaborish preprocessing used before strategy selection.
- Implement the first-pass chroma-from-luma map consumed by AC cost estimates.
- Validate intermediate images and CfL factors against pinned libjxl.

The initial quant field intentionally runs on the pre-Gaborish opsin image.
Strategy selection consumes the preprocessed image and first-pass CfL result.

### 6. CPU AC-strategy search

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

### 7. Post-search quantization finalization

- Implement `AdjustQuantField` for the selected multiblock strategies.
- Convert the adjusted float field to the raw quant field.
- Compute the second-pass CfL map from strategies and raw quant values.
- Generate the EPF control and sharpness fields required by reconstruction.
- Validate adjusted fields and raw quant values independently.

### 8. CPU reconstruction path

- Produce transformed coefficients for the selected strategy grid.
- Apply DC/LLF conversion and AC quantization/dequantization.
- Reconstruct pixels with the matching inverse transforms.
- Apply inverse CfL, Gaborish, and EPF behavior needed by the encoder's
  perceptual round trip.
- Compare intermediate coefficients and reconstructed images with libjxl.

### 9. Perceptual scoring

- Implement the CPU Butteraugli-compatible distance map required by AQ.
- Compute the aggregate perceptual score used for convergence.
- Validate both the full distance map and scalar score against pinned fixtures.
- Include flat, textured, high-contrast, and chromatic test images.

### 10. Iterative adaptive quantization

- Implement the `FindBestQuantizer`-style encode/reconstruct/measure loop.
- Update the quant field from the perceptual distance map.
- Match libjxl convergence, clamping, and stopping behavior.
- Keep iteration order deterministic.
- Add maximum-error behavior only when the primary perceptual loop is stable.

### 11. CPU integration gate

GPU porting begins only after the CPU pipeline satisfies all of the following:

- End-to-end fixtures pass through initial quantization, strategy selection,
  quant-field adjustment, quantization, reconstruction, and iterative AQ.
- Intermediate fields and final outputs have pinned libjxl parity coverage.
- Independent oracles cover layout-sensitive transforms and coefficient paths.
- A varied image corpus covers odd dimensions, padding, tile boundaries,
  smooth gradients, texture, edges, and saturated colors.
- Stage-level CPU timings establish performance baselines.
- Accuracy tolerances and known deviations are documented explicitly.

## GPU-porting boundary

After the CPU integration gate, port compute-heavy leaf operations while
retaining the CPU pipeline as the reference implementation. A likely order is:

1. Gaborish and masking convolutions.
2. CfL statistics and map generation.
3. AC candidate transform and cost evaluation.
4. Quantization, dequantization, and reconstruction.
5. EPF and Butteraugli distance-map computation.

Search traversal, candidate selection, convergence decisions, and AQ
orchestration should initially remain on the CPU. They should move to the GPU
only when profiling demonstrates that doing so is worthwhile and the CPU/GPU
parity tests can preserve deterministic behavior.

Each GPU milestone must be checked against both the CPU reference and pinned
libjxl outputs; a GPU round trip alone is not sufficient evidence of parity.
