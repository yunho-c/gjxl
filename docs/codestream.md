# VarDCT codestream roadmap

This document tracks the work required to serialize a completed
`VarDctEncoderFrame` as a valid JPEG XL codestream. The first milestone targets
a deliberately small, deterministic profile derived from `libjxl-tiny`; later
work may generalize the profile without reopening the frontend representation.

The numerical reference revision is libjxl
`e8ff09762481785938d8e4e01333ed3917571161`. The serializer design is based on
the local `/Users/yunhocho/GitHub/libjxl-tiny` encoder, with its bitstream code
adapted to consume GJXL's completed frame instead of recomputing transforms and
quantization from pixels.

## Current foundation

The frontend handoff is complete for the initial profile. A valid
`VarDctEncoderFrame` owns:

- the exact serialization-critical source, quantization, DC, coefficient-order,
  loop-filter, and modular profile used by the frontend;
- original and padded frame geometry;
- the complete AC-strategy grid;
- the raw quant field and quantizer parameters;
- the final per-tile AC chroma-from-luma maps;
- per-block EPF sharpness;
- authoritative three-plane `int32_t` quantized DC;
- decoder-equivalent reconstructed DC used by AQ and reconstruction; and
- quantized AC in fixed-capacity 256x256-pixel groups.

The current seven-strategy set is DCT8, DCT16x16, DCT32x32, DCT16x8,
DCT8x16, DCT32x16, and DCT16x32. Complete transforms are appended to each AC
group in row-major anchor order, and unused edge-group tails are zero.

Relevant implementations:

- [`codestream.h`](../src/codec/codestream.h)
- [`vardct_frame.h`](../src/codec/vardct_frame.h)
- [`vardct_frame.cpp`](../src/codec/vardct_frame.cpp)
- [`dc_quantization.cpp`](../src/codec/dc_quantization.cpp)
- [`reconstruction.cpp`](../src/codec/reconstruction.cpp)

## Dependency order

The intended implementation sequence is:

```text
simple profile contract
        │
        v
bit writer + entropy primitives
        │
        ├───────────────┐
        v               v
DC + metadata tokens   AC tokens
        │               │
        └───────┬───────┘
                v
       headers + section assembly
                │
                v
       pinned-decoder conformance
                │
                v
          public API + CLI
```

Token generation should remain separable from entropy encoding. Pure token
streams are easier to compare against independent fixtures and allow entropy
model changes without changing coefficient traversal.

## Initial codestream profile

The first writer emits a raw JPEG XL codestream beginning with the `0xFF 0x0A`
marker. It does not emit an ISO BMFF container or metadata boxes.

| Property | Initial value |
| --- | --- |
| Frame | One regular, final VarDCT frame |
| Color transform | XYB |
| Chroma sampling | 4:4:4 |
| Source metadata | Linear sRGB, floating-point samples |
| Intensity target | 255 nits |
| Extra channels | None |
| Passes | One |
| Upsampling | None |
| Quantization matrices | JPEG XL defaults |
| X/B matrix scales | `x_qm_scale = 2`, `b_qm_scale = 2` |
| DC precision | `extra_dc_precision = 0` |
| DC CfL | Default X=0 and B=1 factors |
| Adaptive DC smoothing | Skipped |
| Coefficient orders | Default natural orders |
| Gaborish | Enabled with default weights |
| EPF | Two iterations with default parameters |
| Modular transforms | None |
| LZ77 | Disabled |

The writer must reject unsupported frame state before emitting bytes. It must
not silently round arbitrary coefficient-matrix multipliers to a representable
three-bit scale or signal loop-filter settings different from those used by
AQ.

## Milestones

### 0. Serialization profile contract

Close the remaining gap between a numerically complete frame and a
serialization-safe frame.

Deliverables:

- Define a `SimpleVarDctCodestreamProfile` or equivalent exact-bit contract.
- Thread serialization-critical settings through the pipeline and retain them
  with the completed frame.
- Represent X and B matrix multipliers using the serialized three-bit scales,
  deriving numerical multipliers from `pow(1.25, scale - 2)`.
- Retain or validate the loop-filter mode used by AQ.
- Add `ValidateSimpleCodestreamFrame` without weakening the general
  `VarDctEncoderFrame::valid()` invariant.
- Reject unsupported color metadata, extra channels, passes, custom matrices,
  custom coefficient orders, loop filters, and DC modes.
- Preserve atomic output behavior on every validation failure.

Initial-profile acceptance criteria:

- Default pipeline output passes validation.
- Gaborish-disabled and non-default EPF configurations are rejected.
- Non-representable matrix multipliers cannot reach the writer.
- Quantizer parameters and all dimensions fit their JPEG XL encodings.
- Validation tests cover each rejected profile dimension independently.

### 1. Bit writer and entropy primitives

Add a backend-independent `gjxl_codestream` library. Port only the small
bitstream-writing portion of `libjxl-tiny`; do not import its image,
transform, AQ, or quantization frontend.

Deliverables:

- Byte storage and an unaligned little-endian bit writer.
- Bounded allotments and byte alignment.
- Signed-to-unsigned packing.
- A typed token containing a context and full `uint32_t` value.
- JPEG XL HybridUint tokenization.
- Huffman construction and prefix-code serialization.
- Context-map serialization.
- Entropy-code optimization across section token streams.
- TOC size coding and byte-aligned section concatenation.
- Preserved BSD attribution for adapted `libjxl-tiny` sources.

GJXL must not retain `libjxl-tiny`'s 16-bit temporary token staging. DC and AC
are stored as `int32_t`, so token collection must preserve full packed values.
The preferred implementation uses a HybridUint configuration capable of
representing the full `uint32_t` token range and signals that configuration in
the entropy code. A deliberately narrower implementation must instead reject
out-of-range tokens explicitly; truncation and debug-only assertions are not
acceptable.

Acceptance criteria:

- Byte-exact tests cover cross-byte writes, alignment, append, and final size.
- HybridUint boundary vectors cover zero, sign packing, every exponent change,
  `UINT16_MAX`, and `UINT32_MAX`.
- Huffman and context-map fixtures match the independent tiny/reference
  writer for the same histograms.
- TOC fixtures cover each size-selector transition.
- Oversized allocations and section sizes return a status rather than relying
  on assertions.

Reference sources:

- `/Users/yunhocho/GitHub/libjxl-tiny/encoder/enc_bit_writer.*`
- `/Users/yunhocho/GitHub/libjxl-tiny/encoder/enc_entropy_code.*`
- `/Users/yunhocho/GitHub/libjxl-tiny/encoder/enc_cluster.*`
- `/Users/yunhocho/GitHub/libjxl-tiny/encoder/enc_huffman_tree.*`

### 2. DC-group and AC-metadata tokenization

Generate the modular tokens stored in each 2048x2048-pixel DC group. Frame
planes use X/Y/B order; the modular stream is emitted in Y/X/B order.

Deliverables:

- Derive the DC-group grid from frame geometry.
- Slice `quantized_dc()` into group-local block rectangles.
- Apply the JPEG XL clamped-gradient predictor independently within each DC
  group.
- Emit packed residuals with the 45 DC contexts.
- Emit the DC-group modular header with `extra_dc_precision = 0`, the global
  tree, default weighted predictor settings, and no transforms.
- Slice and emit the two color-correlation maps.
- Emit raw AC-strategy codes at transform anchors.
- Emit `raw_quant - 1` at anchors using a type that preserves raw quant 256.
- Emit the frame's actual EPF sharpness for every base block.
- Emit the transform-anchor count and AC-metadata modular header.
- Return token vectors separately from entropy-coded bytes.

Acceptance criteria:

- Token tuple fixtures pin `(context, value)` independently of Huffman coding.
- Dimensions 1, 8, 9, 2048, and 2049 exercise padding and DC-group edges.
- Raw quant 1 and 256 round-trip through metadata tokens.
- Nonconstant EPF and CfL fixtures prove that values are not hardcoded.
- Large positive and negative DC predictor residuals are preserved exactly.
- Multiple DC groups reset their predictor state correctly.

Reference implementation:

- `/Users/yunhocho/GitHub/libjxl-tiny/encoder/enc_frame.cc`, especially
  `WriteDCTokens`, `WriteACMetadataTokens`, and `WriteDCGroup`.

### 3. Seven-strategy AC-group tokenization

Consume the grouped AC rows already stored in `VarDctEncoderFrame`; do not
re-run the DCT, CfL, quantization, or DC extraction.

Deliverables:

- Traverse every group in row-major order.
- Traverse transform anchors in the same row-major order used while building
  each fixed coefficient row.
- Generate the default natural coefficient order for all seven strategies.
- Skip the strategy's complete low-low-frequency region.
- Count nonzero AC coefficients using an `int32_t` count.
- Replicate the scaled nonzero count across every covered base block.
- Maintain the top/left nonzero map used for context prediction.
- Emit the nonzero-count token and coefficient tokens in Y/X/B order.
- Preserve full signed coefficient values during packing.
- Require the final source offset to equal the group's
  `used_coefficient_count`.

The tiny implementation only supports DCT8, DCT16x8, and DCT8x16 and uses
scratch assumptions sized for those transforms. GJXL should generalize the
scalar tokenization path directly instead of restricting the established
frontend strategy set. The default natural-order generator in pinned libjxl is
small and should be ported with direct parity tests.

Acceptance criteria:

- Every supported strategy has an asymmetric pinned natural-order fixture.
- Every supported strategy has a pinned token-stream fixture.
- Fixtures cover zero AC, dense AC, last-scan-position nonzero, and signed
  extremes.
- Multiblock fixtures verify LLF skipping and nonzero-map replication.
- 256- and 257-pixel cases verify full and edge AC groups.
- Multiple transforms in one group prove coefficient-offset accounting.
- Corrupted or inconsistent group consumption fails atomically.

Reference implementations:

- `/Users/yunhocho/GitHub/libjxl-tiny/encoder/enc_group.cc`
- `third_party/libjxl/lib/jxl/ac_strategy.cc`

### 4. Headers, global sections, and assembly

Assemble the tokenized data into the raw codestream.

Deliverables:

- Codestream marker and size header.
- Initial-profile image metadata.
- Regular VarDCT frame header.
- Serialized `global_scale` and `quant_dc` from the frame's quantizer.
- DC global section with default dequantization and DC correlation.
- AC global section with default matrices and coefficient orders.
- One entropy model shared across DC-group and AC-metadata streams.
- One entropy model shared across AC-group streams.
- Section layout in JPEG XL order:
  - DC global;
  - all DC groups;
  - AC global; and
  - all AC groups.
- TOC emission and byte-aligned section concatenation.
- A filesystem-independent API:

```cpp
[[nodiscard]] Status EncodeVarDctCodestream(
  const VarDctEncoderFrame& frame,
  std::vector<uint8_t>* output);
```

The function validates the complete frame and profile before committing
output. Failure leaves the caller's byte vector unchanged.

Acceptance criteria:

- Empty or invalid frames are rejected.
- A valid one-block frame produces a deterministic nonempty codestream with
  the `0xFF 0x0A` marker.
- Quantizer selector boundaries have byte-level fixtures.
- Single- and multiple-section TOCs have byte-level fixtures.
- Repeated encoding produces byte-identical output.
- The initial implementation contains no file I/O and no dependency on Metal.

Reference implementation:

- `/Users/yunhocho/GitHub/libjxl-tiny/encoder/enc_file.cc`
- `/Users/yunhocho/GitHub/libjxl-tiny/encoder/enc_frame.cc`

### 5. Decoder conformance gate

Compilation, token fixtures, and internal reconstruction do not prove that a
codestream is valid. Completion requires decoding with an independent JPEG XL
decoder.

Deliverables:

- A reproducible conformance target using `djxl` from the pinned libjxl
  revision.
- A convenient local smoke path using an installed `djxl`, when available.
- Decoding to a floating-point output format suitable for numerical
  comparison.
- Comparison against the pipeline's reconstructed linear RGB after the same
  crop, loop filter, color transform, and intensity target.
- Persisted failing codestreams and decoded artifacts for diagnosis.

Required corpus:

- 1x1 and single-block images;
- odd dimensions and block padding;
- 256/257-pixel AC-group boundaries;
- 2048/2049-pixel DC-group boundaries;
- one forced fixture for each supported strategy;
- flat, asymmetric impulse, deterministic random, gradient, texture, hard
  edge, and saturated-primary inputs;
- raw quant endpoints and large signed DC/AC token values; and
- multiple AC and DC groups.

Acceptance criteria:

- Every produced codestream is accepted by pinned `djxl`.
- Decoded dimensions and color metadata match the source contract.
- Decoded pixels meet an assertion-based tolerance against GJXL's
  decoder-equivalent reconstruction.
- Corruption tests fail cleanly and never masquerade as successful encodes.
- At least one small deterministic fixture has a pinned codestream hash.

### 6. Public encoding workflow — complete (2026-08-27)

Provide an end-to-end path only after the frame serializer passes decoder
conformance.

Deliverables:

- Compose color conversion, quantization/AQ, frame construction, and
  `EncodeVarDctCodestream` without duplicating either frontend or writer state.
- Add a small command-line encoder, initially accepting the repository's
  chosen floating-point interchange format.
- Add a `just encode` workflow with input, output, and perceptual-distance
  arguments.
- Report dimensions, encoded byte count, strategy counts, and actionable
  errors.
- Ensure partial output files are never left behind after failure.

Acceptance criteria:

- A checked-in sample can be encoded, decoded with pinned `djxl`, and compared
  automatically.
- The CLI output is deterministic for a fixed build and input.
- The core codestream library remains independent of command-line parsing and
  filesystem policy.

`EncodeLinearRgbVarDctCodestream` is the backend-neutral, in-memory public
workflow. It edge-extends strided linear-sRGB input to the codec block extent,
converts it to XYB, runs the native CPU quantization and default two-update AQ
pipeline, passes the resulting owned `VarDctEncoderFrame` directly to
`EncodeVarDctCodestream`, and atomically commits the byte vector and optional
analysis summary. The summary reports dimensions, encoded bytes, score history,
and transform-anchor counts without exposing pipeline scratch storage.

The `gjxl_encode` frontend accepts three-channel linear-RGB PFM input and a
Butteraugli target. It writes through a same-directory temporary file,
synchronizes it, and renames it over the destination only after the complete
codestream succeeds. Invalid options, malformed or non-finite PFM input,
encoding failure, and output failure cannot expose a partial destination. The
codec and codestream libraries contain no command-line parsing or filesystem
policy.

The checked `17x13` sample encodes to 291 bytes at target `1.0`; its codestream
SHA-256 is
`48abd331b4b4e37f0b158af86ef7c766c72ed760a51ce6903a415bf2544031c7`.
Pinned `djxl` decodes it as linear sRGB with native Butteraugli distance
`0.999045551` from the input. The workflow also has an independent in-memory
FNV-1a codestream pin, deterministic repeated-encode coverage, strided input,
strategy reporting, invalid-input atomicity, installed-consumer coverage, and
a generated-sample freshness check.

Encode a PFM with:

```sh
just encode testdata/codestream_sample.pfm output.jxl 1.0
```

Relevant implementations:

- [`workflow.h`](../src/codestream/workflow.h)
- [`gjxl_encode.cpp`](../tools/gjxl_encode.cpp)
- [`pfm.cpp`](../src/io/pfm.cpp)
- [`codestream_workflow_test.cpp`](../tests/codestream_workflow_test.cpp)
- [`RunCodestreamCliTest.cmake`](../cmake/RunCodestreamCliTest.cmake)

## Completion definition

The initial codestream milestone is complete when all of the following are
true:

- the frame/profile validator prevents decoder-header mismatches;
- quantized DC, AC metadata, and all seven AC strategies are serialized from
  the completed frame without recomputation;
- token and bitstream paths preserve their declared integer range;
- output construction is atomic and deterministic;
- raw `.jxl` output decodes with pinned libjxl across the required corpus; and
- decoded pixels match the frontend's decoder-equivalent reconstruction under
  assertion-based tolerances.

The initial profile and public CPU workflow now satisfy this definition. The
pinned conformance target retains the direct reconstruction comparison for all
21 serializer fixtures and additionally exercises the checked public-workflow
sample through the installed-style CLI and independent decoder. The completed
Release validation passed all 39 reference-disabled tests, all 46
reference-enabled tests, and the full 21-fixture pinned-decoder corpus plus the
workflow sample.

## Deferred work

The following are explicitly outside the first codestream milestone:

- ISO BMFF containers, Exif, XMP, previews, and thumbnails;
- ICC profiles and non-linear or non-sRGB source metadata;
- alpha or other extra channels;
- chroma subsampling;
- custom quantization matrices;
- non-default X/B quantization-matrix scales;
- custom Gaborish or EPF parameters;
- non-default DC correlation and extra DC precision;
- adaptive DC smoothing;
- custom coefficient orders;
- progressive or multi-pass coding;
- LZ77;
- entropy-model or size optimization beyond a correct deterministic model;
- lossless and modular-only image coding; and
- GPU tokenization or entropy coding.

These features may extend the frame/profile contract later. They must not be
silently inferred or partially signaled by the initial writer.
