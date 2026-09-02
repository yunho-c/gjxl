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
| X/B matrix scales | libjxl target/pixel heuristic; `2/2` for maximum error |
| DC precision | `extra_dc_precision = 0` |
| DC CfL | Default X=0 and B=1 factors |
| Adaptive DC smoothing | Skipped |
| Coefficient orders | Natural or serializer-selected custom orders |
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
  externally supplied custom coefficient orders, loop filters, and DC modes.
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
- AC global section with default matrices and selected coefficient orders.
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
converts it to XYB, runs the selected quantization and default two-update AQ
pipeline, passes the resulting owned `VarDctEncoderFrame` directly to
`EncodeVarDctCodestream`, and atomically commits the byte vector and optional
analysis summary. The summary reports dimensions, encoded bytes, score history,
transform-anchor counts, the selected CPU or Metal backend, and the selected
Metal AQ mode without exposing pipeline scratch storage.
The complete public-boundary profiling contract and `50x` optimization plan are
maintained in
[`metal-encoding-performance.md`](metal-encoding-performance.md).

The default `kAutomatic` preference uses the embedded, process-cached Metal
backend only on the qualified Apple M4 Pro geometry range and Butteraugli
target interval `[1.0, 1.2]` established in [`metal-aq.md`](metal-aq.md);
small images, targets outside that interval, unqualified devices, unavailable
Metal, or missing capabilities use CPU before pipeline execution. Explicit
`kCpu` and `kMetal` overrides are available. Forced Metal bypasses the
automatic size, target, and device gates but never falls back; the broader
quality range is an explicit unqualified override. Operational errors after
GPU work starts are returned atomically instead of retrying on CPU. The
default density policy performs two AQ updates; the explicit high-density
policy performs four on CPU, exact-coefficient Metal, or fully-resident Metal.
Entropy search is resolved independently: efforts 1-8 use the balanced
single-representation serializer, efforts 9-10 and `kHighDensity` use the
effort-9-like high-density serializer, and
`VarDctCompressionMode::kMaximumCompression` explicitly restores the former
exhaustive map/order/coder tournament. Direct serializer calls default to
balanced behavior. See
[`entropy-behavior-alignment.md`](entropy-behavior-alignment.md) for the source
comparison, behavior matrix, and qualification.

Forced Metal additionally accepts `GpuAdaptiveQuantizationMode::kFullyResident`,
`kThroughput`, and `kMaximumThroughput` as experimental first-class options.
The first two keep forward
transforms and coefficient coding on Metal, apply inverse Gaborish through
Metal, and use a deterministic tilewise pixel-domain initial-CfL seed. They do
not promise the CPU reference's quant field, frame, or codestream bytes. Their
final-CfL map is strategy-aware but fixed from the adjusted initial field
across AQ evaluations. Automatic and CPU preferences reject those modes rather
than silently selecting a different implementation; exact coefficients remain
the default and the only automatically selected Metal AQ mode.
Fully resident and throughput encoding apply the configured updates, then
quantize the resulting field into the final frame without reconstructing and
scoring that field one more time. Their reported score history therefore
covers the evaluated update fields, while
`final_butteraugli_score_evaluated` is false. Setting
`collect_final_butteraugli_score` performs the terminal evaluation without
changing the configured update count or encoded frame. Complete throughput
diagnostic API calls retain their separate explicit one-update policy.
Maximum-throughput mode instead fixes every transform to DCT8, applies the
resident `AdjustQuantBlockAC` shared-quant decision to the adjusted initial
field, and stops before inverse reconstruction or perceptual scoring. Its
summary score history is empty; the policy must be evaluated with an
independent decoder and quality metric.

The `gjxl_encode` frontend accepts three-channel linear-RGB PFM input and one
of `--distance`, `--maximum-error`, `--target-bytes`, or `--target-bpp`, plus
`--effort 1..10`, `--maximum-compression`, `--backend auto|cpu|metal`, and
`--metal-aq exact-coefficients|fully-resident|throughput|maximum-throughput`.
`--collect-final-score` opts resident Metal encoding into the terminal
diagnostic evaluation; the default report says that the final score was not
evaluated.
`--high-density` selects four AQ updates for Butteraugli-target and target-size
control and high-density entropy. Efforts 9-10 select the same entropy behavior
without otherwise becoming aliases for the four-update compatibility option.
`--maximum-compression` changes only entropy/codestream search; it does not
silently change AQ, backend, rate control, or thread limits. High density is
rejected with maximum-error, throughput, and maximum-throughput policies rather
than being ignored or silently overridden.
Size searches accept
`--size-tolerance`, `--max-attempts`, and
`--size-selection under-budget|closest`. All three experimental modes require
`--backend metal`; automatic maximum-error control remains CPU-only, and
maximum-throughput mode does not support maximum-error control. The
CLI also reports prepared-source, selected-attempt, aggregate size-search, and
end-to-end timing from the profiled workflow API. The
frontend writes through a
same-directory temporary file,
synchronizes it, and renames it over the destination only after the complete
codestream succeeds. Invalid options, malformed or non-finite PFM input,
encoding failure, and output failure cannot expose a partial destination. The
codec and codestream libraries contain no command-line parsing or filesystem
policy.

The `just encode` workflow also accepts normal single-frame image files when
the optional ImageMagick 7 `magick` executable is available. The repository's
`tools/encode_image.py` wrapper applies stored orientation, converts the image
to linear RGB, composites transparency over a white linear-RGB background,
forces three color channels, and writes a temporary PFM before invoking the
unchanged `gjxl_encode` binary. PFM input bypasses ImageMagick entirely.
Animated and multi-page inputs are rejected rather than selecting a frame
silently. Set `GJXL_MAGICK` to an alternate `magick` executable or
`GJXL_ALPHA_BACKGROUND=black` to use a black alpha-compositing background.
Input conversion is outside the encoder workflow timing reported by the CLI.

DC- and AC-group section writers run independently on up to eight host workers
after the shared entropy codes are finalized. Each worker owns one `BitWriter`;
the TOC and final assembly retain canonical section order, so parallelism does
not change codestream bytes or failure atomicity.

The checked `17x13` sample encodes to 263 bytes at target `1.0`; its balanced
codestream SHA-256 is
`e4566239f5e15dd67a4716d26da662728c88ffcae19bdf93ff28c2b8df6c8504`.
Maximum compression emits the former default 255-byte codestream with SHA-256
`e5577ebf76a37bf56a93db61b2ccf1fc959292a3d13d6489baf2e7f5b6105558`.
Pinned `djxl` decodes it as linear sRGB with native Butteraugli distance
`1.09415638` from the input. The workflow also has an independent in-memory
FNV-1a codestream pin, deterministic repeated-encode coverage, strided input,
strategy reporting, invalid-input atomicity, installed-consumer coverage, and
a generated-sample freshness check.

Encode a PFM with:

```sh
just encode testdata/codestream_sample.pfm output.jxl 1.0
# PNG, JPEG, and other formats supported by the installed ImageMagick build:
just encode input.png output.jxl 1.0
# Or call gjxl_encode directly with --backend cpu|metal.
# Experimental resident path:
build/release/gjxl_encode --distance 1.0 --backend metal \
  --metal-aq fully-resident testdata/codestream_sample.pfm output.jxl
# Add --collect-final-score when the terminal encoded-field diagnostic is
# required; it does not change the fully resident codestream.
# Four-update density search on the exact-coefficient path:
build/release/gjxl_encode --distance 1.0 --high-density \
  testdata/codestream_sample.pfm output.jxl
# Preserve the former exhaustive serializer policy without changing AQ:
build/release/gjxl_encode --distance 1.0 --maximum-compression \
  testdata/codestream_sample.pfm output.jxl
# Or the explicitly bounded one-update policy:
build/release/gjxl_encode --distance 1.0 --backend metal \
  --metal-aq throughput testdata/codestream_sample.pfm output.jxl
# Or the speed-first frame-only policy:
build/release/gjxl_encode --distance 1.0 --backend metal \
  --metal-aq maximum-throughput testdata/codestream_sample.pfm output.jxl
# Closest absolute serialized size within the bounded attempt budget:
build/release/gjxl_encode --target-bytes 4096 --size-selection closest \
  testdata/codestream_sample.pfm output.jxl
```

Relevant implementations:

- [`workflow.h`](../src/codestream/workflow.h)
- [`gjxl_encode.cpp`](../tools/gjxl_encode.cpp)
- [`encode_image.py`](../tools/encode_image.py)
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
22 serializer fixtures and additionally exercises the checked public-workflow
sample through distance, maximum-error, and target-size encodes using the
installed-style CLI and independent decoder. The initial milestone's completed
Release validation passed all 39 reference-disabled tests and all 46
reference-enabled tests; the corpus has since expanded from 21 to 22 serializer
fixtures and three workflow modes.

## Deferred work

The following are explicitly outside the first codestream milestone:

- ISO BMFF containers, Exif, XMP, previews, and thumbnails;
- ICC profiles and non-linear or non-sRGB source metadata;
- alpha or other extra channels;
- chroma subsampling;
- custom quantization matrices;
- custom Gaborish or EPF parameters;
- non-default DC correlation and extra DC precision;
- adaptive DC smoothing;
- progressive or multi-pass coding;
- LZ77;
- broader entropy clustering and size optimization beyond the bounded
  prefix/ANS optimizer;
- lossless and modular-only image coding; and
- GPU tokenization or entropy coding.

These features may extend the frame/profile contract later. They must not be
silently inferred or partially signaled by the initial writer.

## Compression-density roadmap

The current writer prioritizes a correct, deterministic subset over exhaustive
compression optimization. Its bounded prefix optimizer searches up to 32
clusters and selects HybridUint configurations per cluster. Adaptive
block-context selection is available for sufficiently large images, and the
writer can serialize either prefix or ANS entropy models. ANS retains the
prefix-selected context partition, then independently searches a bounded set of
HybridUint configurations, histogram representations, and alphabet widths.
Ordinary Butteraugli encoding uses libjxl's target- and pixel-dependent X/B
quantization-matrix scales; maximum-error encoding retains `2/2`.

A directional local comparison used the Release encoder, four natural images
around 500 pixels in size, identical linear-PFM inputs, raw `.jxl` output, and
libjxl tools at `05be1775629f7fdc01beb70c118043b4b0c69d2a`.
External Butteraugli scores were measured at 255 nits. The aggregate encoded
sizes were:

| Target distance | Initial | After #1 | After #2 | After #3 | After #4 | After #5 | After #7 | #7 vs #5 | `cjxl -e 7` | #7 gap | `cjxl -e 9` | #7 gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 153,464 B | 153,278 B | 155,391 B | 152,722 B | 152,111 B | 148,137 B | 147,238 B | -0.61% | 151,714 B | -3.0% | 147,386 B | -0.1% |
| 2 | 94,430 B | 94,327 B | 95,072 B | 92,746 B | 92,065 B | 87,714 B | 87,004 B | -0.81% | 93,414 B | -6.9% | 88,578 B | -1.8% |

These are same-target rather than matched-quality results. The external scores
were close but not identical, so the table indicates the scale of the current
gap; it is not a BD-rate result or a claim of equal visual quality. Before the
matrix-scale heuristic, the flower input at target distance 1 produced 45,963
bytes at external distance 1.42955. With the heuristic it produced 46,454
bytes at 1.43541. `cjxl -e 7` produced 45,014 bytes at 1.42125, and
`cjxl -e 9` produced 43,496 bytes at 1.39970. As another directional signal,
`cjxl -e 1` and `-e 2` produced 47,368-byte files at identical external
distance 1.58041, while `-e 3` produced 45,363 bytes at the same distance. The
4.2% reduction suggests useful entropy-modeling headroom, but the effort tier
changes more than one variable and does not isolate ANS from clustering or
other encoder decisions.

The recommended implementation order is below. Effort and size reductions are
rough hypotheses for general natural images, are not additive, and should be
replaced by corpus measurements as each step lands.

1. **Complete (2026-08-28): improve the existing prefix-code optimizer before
   adding ANS.** The always-on bounded search considers cluster caps 1, 2, 4,
   8, 16, and 32 using exact serialized model-plus-token cost. It reuses a
   deterministic farthest-first seed order, refines the best candidate and its
   next larger cap with prefix-code costs, and selects among four HybridUint
   configurations per final cluster. The former eight-cluster, fixed-config
   result remains a fallback and is replaced only when the complete serialized
   cost is strictly smaller.

   The four-image natural corpus improved by 0.12% at distance 1 and 0.11% at
   distance 2, with no individual regression. The flower result improved from
   46,082 to 45,963 bytes (0.26%). Across the 22 pinned serializer fixtures,
   aggregate size improved from 29,466 to 28,580 bytes (3.01%), again with no
   regression. Since entropy coding does not alter decoded coefficients, the
   independent reconstruction and quality results remain unchanged.

   On the Apple M4 Pro flower benchmark with two warmups and five samples, CPU
   public-workflow median time changed from 910.437 ms to 936.360 ms (+2.85%),
   while its entropy-optimization component changed from 6.381 ms to 27.889 ms.
   Metal public-workflow median changed from 109.255 ms to 131.322 ms (+20.2%)
   because the same host entropy work is a larger fraction of that path. The
   raw-sample schema now reports model and token bits separately plus selected
   DC/AC cluster counts; the flower stream used 6,456 model bits, 359,939 token
   bits, 8 DC clusters, and 16 AC clusters. The natural-image gain is therefore
   below the original 1-4% hypothesis, and the remaining gap is unlikely to be
   closed by prefix clustering alone.

2. **Implemented (2026-08-28), with density acceptance still open: enable the
   libjxl X/B quantization-matrix-scale heuristic.** The workflow computes the
   pinned libjxl X-edge, B-Y edge, and exposed-blue statistics once over the
   unpadded opsin image. Every Butteraugli attempt, including target-size
   searches and all CPU/Metal AQ modes, combines those statistics with the
   attempt's target to select the serialized three-bit scales. AQ,
   reconstruction, coefficient coding, and the frame header all consume that
   same profile. Maximum-error mode remains exactly `2/2` and its checked
   codestream hash did not change.

   This policy did not produce the hypothesized same-target reduction on the
   four-image corpus: relative to #1, aggregate size increased 1.38% at target
   1 and 0.79% at target 2. External quality moved in both directions. A
   bounded per-image target search against the pre-change external
   Butteraugli scores found a 152,209-byte aggregate at the target-1 quality
   points, 0.70% below the 153,278-byte baseline. At target 2 it found 95,831
   bytes, 1.59% above the 94,327-byte baseline. The closest available scores
   differed by 0.0001 to 0.0129 at target 1 and 0.0056 to 0.0657 at target 2,
   so these are directional matched-quality results rather than a
   rate-distortion curve.

   A balanced five-round Apple M4 Pro comparison alternated the pre-change and
   heuristic binaries, with one warmup and one measured public-workflow sample
   per round. CPU median changed from 914.790 to 925.462 ms (+1.17%); Metal
   median changed from 122.987 to 132.191 ms (+7.48%). The CPU result satisfies
   the planned 5% guardrail, while Metal does not. The implementation is
   therefore complete and independently decodable, but the current
   two-iteration AQ policy does not yet justify calling this a general
   compression-density win.

3. **Complete (2026-08-28): add serializer-local custom coefficient orders.**
   The always-on candidate counts all coefficient zeros by channel and the five
   on-wire families used by the seven supported strategies. It retains natural
   orders when both block dimensions are below five, otherwise applies the
   pinned-libjxl quantized zero-count stable sort after the exact LLF prefix.
   Nondefault orders are converted to natural-rank Lehmer permutations, encoded
   with eight contexts, and serialized once in AC global. AC tokens and order
   signaling use the same validated orders.

   Natural and custom AC candidates receive independent entropy models. Their
   exact complete-codestream byte sizes include headers, TOC selectors, section
   padding, order signaling, and token payloads; custom wins only when strictly
   smaller, so ties retain natural order. Common DC sections are written once,
   independent entropy searches run concurrently, and only the selected full
   codestream is assembled. This policy is internal to the serializer and adds
   no public encoding option.

   On the four-image PFM corpus, custom order itself was selected for three of
   four images at each target. Exact candidate sizes changed from 153,245 to
   152,722 bytes at distance 1 (-0.34%) and from 92,972 to 92,746 bytes at
   distance 2 (-0.24%). The complete #3 change versus #2 is larger: -1.72% and
   -2.45%, respectively. Most of that additional gain comes from a required
   prefix correction exposed by the order payload: JPEG XL's one-symbol
   Huffman alphabet consumes zero prefix bits. Cost modeling, token writing,
   and validation now agree on that rule, and the natural zero-order mask also
   uses its canonical two-bit selector. Pinned libjxl decoded the before/after
   corpus to byte-identical PFM output.

   The 22 pinned serializer fixtures improved in aggregate from 28,580 to
   25,679 bytes (-10.15%), predominantly because small synthetic histograms
   exercise the corrected one-symbol prefix case heavily. Every fixture and
   the distance, maximum-error, and target-size public workflows decode under
   pinned libjxl. Unit coverage pins the Lehmer stream, the small-image gate,
   all five order families and seven strategies, invalid-order atomicity,
   deterministic derivation, and both outcomes of exact byte selection.

   A five-round alternating Apple M4 Pro comparison used one warmup and one
   measured flower encode per process. CPU serializer median changed from
   25.765 ms (25.576-26.607) to 26.568 ms (26.388-27.194), +3.12%; complete CPU
   workflow median changed from 935.565 ms (900.927-948.646) to 915.694 ms
   (891.932-935.512), -2.12% within the broader pipeline variation. Metal
   serializer median changed from 25.816 ms (25.519-34.190) to 27.734 ms
   (26.544-34.410), +7.43%; complete Metal workflow median changed from
   119.011 ms (115.152-127.786) to 127.471 ms (119.268-134.978), +7.11%.
   Concurrent candidate optimization kept the entropy phase essentially flat;
   the remaining host cost is mostly the second AC traversal and model setup.
   Raw benchmark schema version 3 records natural/custom candidate sizes and
   the selected family mask alongside entropy bits, clusters, and phase timing.

4. **Complete (2026-08-28): make block-context maps adaptive for sufficiently
   large images.** Images below 1,024 base blocks retain the exact compact-map
   path and checked codestream bytes. Larger images consider the compact map,
   the JPEG XL default map, one- and two-context channel maps, and a
   deterministic occurrence-clustered strategy map. Images with at least 8,192
   base blocks additionally consider a median raw-quant split. Every map is
   crossed with the eligible natural/custom coefficient order, receives an
   independent prefix model, and is selected by exact complete-codestream bytes
   including map signaling, entropy models, TOC selectors, and padding. The
   compact candidate is always present, and exact ties prefer natural order and
   then the earlier map.

   On the established four-image PFM corpus, all eight target/image cases
   selected an alternate map. Aggregate size changed from 152,722 to 152,111
   bytes at distance 1 (-0.40%) and from 92,746 to 92,065 bytes at distance 2
   (-0.73%), with no individual regression. One-, two-, and three-context maps
   all won at least one case. Pinned `djxl` accepted every output, and each
   decoded PFM was byte-identical to its pre-change decode. A separate
   1024x768 natural-image check selected the ten-context, one-quant-threshold
   candidate; its benchmark codestream was 181,288 bytes versus 181,517 bytes
   for the compact candidate, and pinned `djxl` accepted the result. This is a
   real gain, but below the original 1-2% hypothesis.

   A five-round alternating Apple M4 Pro comparison used one warmup and one
   measured flower encode per round. CPU serializer median changed from 29.396
   ms (28.134-29.844) to 54.739 ms (53.767-56.527), while complete CPU workflow
   median changed from 919.861 to 941.536 ms (+2.36%). Metal serializer median
   changed from 28.631 ms (27.395-29.687) to 55.177 ms (53.702-65.450), while
   complete Metal workflow median changed from 129.585 to 160.602 ms (+23.9%).
   The size search is host-side and intentionally bounded away from thumbnails,
   but its cost is material on the faster Metal frontend. Raw benchmark schema
   version 4 records the candidate count, compact-candidate bytes, selected map
   index/context count, and raw-quant-threshold count.

5. **Complete (2026-08-29): add a native ANS entropy path with an exact prefix
   fallback.** The writer now supports JPEG XL's 12-bit ANS tables with
   deterministic count normalization, alias-based encoder lookup tables,
   general and one-/two-symbol histogram serialization, reverse-order state
   updates, and full-width HybridUint extra bits. Production code has no
   libjxl dependency; the adapted model and state logic retains upstream BSD
   attribution.

   Each DC, AC, and eligible coefficient-order model first runs the established
   prefix optimizer. ANS reuses that optimized context partition and per-cluster
   HybridUint configuration, then competes on its exact serialized model and
   payload bits. Strict local ties retain prefix coding. Because section padding,
   TOC selectors, and interactions between independently selected models can
   erase a local bit win, every block-map/order candidate additionally measures
   the complete all-prefix codestream. The all-prefix form wins complete-byte
   ties. This bounded comparison deliberately does not enumerate every mixed
   prefix/ANS combination, but it guarantees that enabling ANS cannot make a
   selected complete candidate larger than its former prefix form.

   On the established four-image PFM corpus, all eight target/image cases used
   ANS for AC, while DC and coefficient-order models selected prefix or ANS
   independently. Aggregate size changed from 152,111 to 148,137 bytes at
   distance 1 (-2.61%) and from 92,065 to 87,714 bytes at distance 2 (-4.73%).
   The combined reduction was 8,325 bytes (-3.41%), with every individual case
   improving. Pinned `djxl` accepted all eight outputs, and every decoded PFM
   was byte-identical to its #4 counterpart. The checked 17x13 sample retained
   its 259-byte codestream and SHA-256 because the exact all-prefix tie fallback
   preserved its previous bytes.

   A five-round alternating Apple M4 Pro comparison used one warmup and one
   measured flower encode per round. CPU serializer median changed from 53.146
   ms (52.022-56.373) to 62.002 ms (59.801-78.186), +16.7%; complete CPU
   workflow median changed from 954.003 to 965.412 ms, +1.20%. Metal serializer
   median changed from 54.851 ms (52.903-59.325) to 60.152 ms
   (59.195-65.062), +9.66%; complete Metal workflow median changed from 149.283
   to 152.101 ms, +1.89%. The additional complete-prefix measurement roughly
   doubled section-writing time, but remained a small fraction of either full
   workflow. Raw benchmark schema version 5 reports the selected DC, AC, and
   coefficient-order entropy modes.

   Unit coverage pins deterministic ANS model/payload construction, exact cost
   attribution, malformed reverse-map atomicity, and an encoder fixture that
   selects ANS. The focused CLI, generated-sample, encoder, entropy, raw-schema,
   and installed-decoder smoke gates all pass.

6. **Complete (2026-08-29): offer four adaptive-quantization updates as an
   optional high-density mode.** `VarDctDensityMode::kHighDensity` and the CLI's
   `--high-density` opt into four updates; the default remains two and retains
   its checked codestream bytes. The policy is supported by CPU,
   exact-coefficient Metal, fully-resident Metal, and target-size searches.
   Maximum-error has its own fixed hard-bound search, while throughput and
   maximum-throughput intentionally perform less work, so those combinations
   are rejected atomically.

   Extra updates improve quality at the same requested target rather than
   directly minimizing bytes. On the four-image PFM corpus, same-target size
   changed from 148,137 to 148,489 bytes at distance 1 (+0.24%) and from 87,714
   to 87,991 bytes at distance 2 (+0.32%); every changed case had a lower
   external Butteraugli score. A bounded six-step per-image target search then
   matched the #5 external scores within 0.000954 at distance 1 and 0.000631 at
   distance 2. At those directional matched-quality points, aggregate size was
   147,688 bytes (-0.30%) and 87,148 bytes (-0.65%), respectively. Two of the
   eight cases were already unchanged after four updates. Pinned `djxl`
   accepted every same-target output, and the conformance target now includes
   the checked high-density public workflow.

   A five-round alternating Apple M4 Pro comparison used one warmup and one
   measured flower encode per mode and round. CPU quantization-pipeline median
   changed from 760.846 to 1,144.865 ms (+50.5%), and complete CPU workflow
   median changed from 960.907 to 1,345.529 ms (+40.0%). Metal quantization
   changed from 63.971 to 87.426 ms (+36.7%), and complete Metal workflow from
   157.274 to 179.700 ms (+14.3%). The sub-1% directional density gain therefore
   comes with a deliberately substantial runtime cost, which is why the mode is
   explicit. Raw benchmark schema version 6 records `density=default|high`.

7. **Complete (2026-08-29): make ANS model selection native to ANS.** ANS still
   reuses the prefix optimizer's bounded context partition, but no longer reuses
   its prefix-cost HybridUint choice. Each cluster screens eight configurations:
   the former four plus `{3,1,0}`, `{3,2,0}`, `{4,1,0}`, and `{5,2,0}`. The
   selected population competes across the flat form and all 12 allowed
   precision shifts. General histogram headers use the format's repeated-count
   coding, and alphabet widths 5 through 8 compete on exact serialized
   model-plus-token bits. The existing exact prefix fallback remains unchanged,
   so a larger ANS result cannot displace prefix coding.

   On the established four-image PFM corpus, aggregate size changed from
   148,137 to 147,238 bytes at distance 1 (-0.61%) and from 87,714 to 87,004
   bytes at distance 2 (-0.81%). Every individual case improved. Histogram
   precision selection supplied 756 and 584 bytes of those reductions;
   ANS-specific HybridUint selection supplied the remaining 143 and 126 bytes.
   Header RLE alone saved only 31 and 12 bytes, while the flat form accounted
   for 6 and 7 aggregate bytes. A 28-configuration experiment matched the
   eight-choice result at distance 1 and improved distance 2 by only four more
   bytes, so the exhaustive search was not retained. Pinned `djxl` accepted all
   eight production outputs, and each decoded PFM was byte-identical to its #5
   baseline.

   A seven-round alternating Apple M4 Pro comparison used one warmup and one
   measured flower encode per binary and round. Complete CPU workflow medians
   were 1,385.965 ms before and 1,383.156 ms after (-0.20%, within run-to-run
   noise). Complete forced-Metal workflow medians were 290.555 and 306.334 ms
   (+5.43%). The bounded eight-choice search therefore avoids the roughly 6%
   complete-CPU overhead observed for the rejected 28-choice experiment, but
   its host-side work remains more visible on the faster Metal frontend.

   Unit coverage pins flat and non-flat precision selection, the two dominant
   added HybridUint configurations, sparse-histogram RLE, exact model and
   payload accounting, and malformed-representation atomicity. The full Release
   suite passes apart from the inherited pinned quantization-score mismatch.

LZ77, adaptive DC smoothing, additional transform decisions, patches/dots/
splines, chroma subsampling, and GPU entropy coding are not first-line size
work. They target narrower content, change quality semantics, or require more
implementation scope for less likely general-image benefit. Each accepted
optimization should be evaluated on a broader corpus with decoded-output
validation and matched-quality size comparisons, not only the same nominal
distance.

### CPU thread-budget requalification

The Release `gjxl_encoding_benchmark` accepts `--cpu-threads auto|N` for its
public-workflow scopes. Raw-sample schema 10 records the requested value as
`cpu_threads` (`0` means automatic) and each encode's observed
`peak_cpu_participants`. The peak is diagnostic-only internal profiling data;
normal library encodes do not enable its atomic participant tracker.

A post-prefix-statistics run on Apple M4 Pro compared automatic, 1, 2, 4, and
8 CPU threads at distance 1.2, effort 7, and default density. Each budget had
two independent benchmark processes in forward then reverse order. Every
process performed one warmup and three measured encodes, giving six CPU
samples per cell. The benchmark's exact-coefficient CPU/Metal validation also
confirmed identical codestreams within every process.

| Budget | Observed peak | Flower 510x532 total | Flower codestream | Padded 720p total | Padded 720p codestream |
| --- | ---: | ---: | ---: | ---: | ---: |
| automatic | 12 | 949.207 ms | 32.367 ms | 3227.548 ms | 154.625 ms |
| 1 | 1 | 1053.443 ms | 156.183 ms | 3739.456 ms | 749.551 ms |
| 2 | 2 | 994.152 ms | 85.649 ms | 3336.950 ms | 377.007 ms |
| 4 | 4 | 949.914 ms | 51.061 ms | 3137.765 ms | 208.820 ms |
| 8 | 8 | 917.185 ms | 31.177 ms | 3049.494 ms | 148.568 ms |

The explicit cap held exactly in all 48 explicit-budget CPU samples, and each
workload produced one encoded size across all budgets. Eight threads had the
lowest total median in this limited M4 Pro study, 3.4% below automatic on
Flower and 5.5% below automatic on padded 720p. This is not enough evidence to
replace the automatic default: it establishes that the post-optimization
scheduler still honors its cap and that an eight-thread consumer preference is
competitive on this machine. Broader real-image and hardware coverage should
precede a default-policy change.

## Codestream performance profile and optimization priorities

A symbolized Samply capture on Apple M4 Pro isolated the host codestream tail
after the prefix-density and custom-coefficient-order changes. The capture used
the Release `gjxl_encoding_benchmark` binary at
`56a0790d3549bbd2bebe80522808ff47f7891ef2`, Kodak image 01 as a 768x512 PPM,
the Metal public workflow in `maximum-throughput` mode at distance 1.2, five
warmups, 400 measured encodes, and 1 kHz all-thread sampling. It contained
37,656 positive-CPU samples, 27,886.277 ms of sampled CPU delta, and 99.78%
weighted leaf-symbol resolution.

Sampling raised the run's reported latency, so these percentages are hotspot
attribution rather than benchmark speedups. The capture also predates adaptive
block-context candidates, ANS selection, and high-density AQ. Those features
make a new current-head capture necessary before claiming current percentages;
source inspection nevertheless confirms that the profiled prefix-clustering
and Huffman paths remain in the current writer.

| Stack or mutually exclusive entropy leaf group | Share of all sampled CPU | Share within prefix optimization |
| --- | ---: | ---: |
| `OptimizeEntropyCode` inclusive | 77.90% | 100.00% |
| `OptimizeAcCandidate` inclusive | 76.43% | 98.10% |
| Huffman-tree construction | 32.17% | 41.30% |
| Histogram clustering and distance | 24.18% | 31.03% |
| Allocation and memory traffic | 9.34% | 11.99% |
| Entropy model and configuration construction | 7.28% | 9.34% |
| Synchronization and waits | 0.01% | 0.01% |

The hottest flat leaves were `CreateHuffmanTree` at 17.83% of all sampled CPU,
`ClusterHistogramsFromSeeds` at 12.15%, Huffman `SetDepth` at 9.77%,
`HistogramDistance` at 7.59%, and `BuildEntropyCodeForPartition` at 6.51%.
Inclusive rows overlap and must not be added. The leaf-group rows partition only
samples whose stacks include `OptimizeEntropyCode`.

The call stacks and dependency structure suggest the following implementation
order.

1. **Complete (2026-08-29): remove unused exact costs from shape-only
   screening.**
   `FastClusterHistograms` computes `HistogramBitCost` for every prepared
   histogram even when `fill_to_limit` selects only `HistogramShapeDistance`
   and the seed-index path returns before exact assignment. Likewise,
   `AssignHistogramsToSeeds` computes and updates exact bit costs when
   `use_shape_distance` is true, although those comparisons use only counts and
   totals. The capture attributed 8.21% of all sampled CPU to Huffman work
   directly beneath `ClusterHistogramsFromSeeds` and another 1.89% beneath
   `FastClusterHistograms`. Not all of that 10.10% is removable, but this is the
   strongest exact-codestream-preserving first experiment.

   The retained implementation skips prepared costs only when shape-based seed
   discovery returns before exact assignment. Shape-based seed assignment also
   omits its unused input and accumulating-cluster costs. Exact-distance seed
   discovery and assignment retain their former cost updates, while downstream
   compaction or refinement reconstructs final codes from unchanged counts.

   Five alternating Apple M4 Pro process pairs compared the parent and optimized
   Release binaries on Kodak image 01, using the Metal public workflow at
   distance 1.2 and `maximum-throughput`, with three warmups and 20 measured
   samples per process. Per-pair entropy-optimization medians improved by
   7.55-8.60%, codestream encoding by 4.30-7.18%, and complete workflow time by
   5.66-6.73%. Across the pooled 100 samples per binary, entropy changed from
   58.834 to 54.009 ms (-8.20%), codestream encoding from 77.264 to 72.975 ms
   (-5.55%), and complete workflow time from 97.826 to 91.583 ms (-6.38%). All
   200 samples retained identical encoded size, entropy bits and cluster counts,
   entropy modes, coefficient-order choice, and block-context choice.

   Matched 1 kHz Samply captures used five warmups and 400 encodes per binary.
   Sampled CPU delta fell 8.51% overall, 10.96% inside `OptimizeEntropyCode`, and
   20.10% under `CreateHuffmanTree`. Direct Huffman work beneath seeded
   clustering fell 70.43%, from 7.02% to 2.27% of sampled CPU; the corresponding
   direct fast-clustering work fell 33.12%, from 1.41% to 1.03%. Exact
   `HistogramDistance` work changed by only -1.06% and synchronization/waits by
   -1.38%, supporting the intended work-elimination mechanism. These sampled
   percentages remain attribution rather than timing claims.

   The focused entropy, encoder, single-image workflow, and batch-workflow tests
   pass, as do all 22 pinned conformance fixtures and four public workflows under
   pinned `djxl`. The complete Release suite remains 58/59: the sole
   `quantization_pipeline` score mismatch exactly reproduces the inherited
   parent failure and is unrelated to codestream entropy.

2. **Complete (2026-08-29): reuse exact histogram state and Huffman scratch.**
   Histograms now carry explicit bit-cost validity. Successful count mutations
   invalidate the cache, while the source histograms compute their exact costs
   once before clustering so every candidate copy can reuse them. Independently
   mutable cluster histograms continue to recompute after each merge.

   `CreateHuffmanTree` now uses one uninitialized fixed-capacity array for the
   128-symbol, 257-node maximum and reuses it across depth-limit retries. An
   in-place total-order sort uses count ascending and symbol descending, exactly
   matching the old stable sort's descending-symbol insertion order. Focused
   tests pin the equal-count depths and canonical bits and exercise a forced
   depth-limit retry. Eagerly value-initializing the complete scratch array was
   rejected after it made entropy optimization about 9-11% slower; only nodes
   written by the current attempt are initialized. A fixed-buffer merge-sort
   experiment was also slower and was discarded.

   Five alternating Apple M4 Pro process pairs compared the parent `a251caa`
   binary and the retained Release binary on Kodak image 01 under the same Metal
   public-workflow, distance-1.2, maximum-throughput setup, with three warmups
   and 20 measured samples per process. Per-pair entropy medians improved by
   0.51-2.94%. Across the pooled 100 samples per binary, entropy optimization
   changed from 52.580 to 51.811 ms (-1.46%), codestream encoding from 70.647 to
   69.812 ms (-1.18%), and complete workflow time from 89.883 to 88.360 ms
   (-1.69%). All 200 samples retained identical encoded size, entropy bits and
   clusters, entropy modes, coefficient-order choice, and block-context choice.

   Matched 1 kHz Samply captures used five warmups and 400 encodes per binary.
   Sampled CPU delta fell 5.67% overall, 4.76% inside `OptimizeEntropyCode`, and
   6.39% under `CreateHuffmanTree`; exact `HistogramDistance` work changed by
   -1.74%. Allocator leaves beneath `CreateHuffmanTree` fell from 586.668 ms to
   zero, directly confirming the fixed-scratch mechanism. The replacement sort
   itself was 31.41% hotter than the former stable sort. These sampled CPU deltas
   are attribution evidence rather than timing claims.

   A later retained-head investigation measured 493,756 Huffman-tree calls in
   one Kodak-01 encode. The mean populated alphabet was 19.32 leaves, with p50
   21, p95 26, p99 27, and a maximum of 51; 99.82% had at most 27 leaves. Only
   6,984 calls needed depth-limit retries. This strongly favored small-array
   candidates over another full-width merge sort.

   Exact-order prototypes covered node insertion thresholds 8/16/24/32,
   packed count-and-symbol keys with `std::sort`, insertion, and Shell sort,
   indirect one-byte symbol sorting, and allocating stable sort as a reference.
   A same-process benchmark over 4,096 evenly sampled real histograms used 15
   balanced rounds and 200 repetitions per round. Both 32-leaf node insertion
   and packed keys with an eight-leaf insertion cutoff reduced complete
   tree-construction CPU by about 7.6% relative to the retained node
   `std::sort`; indirect sorting lost once tree materialization was included.

   That isolated result did not survive the full optimizer. In reversed-order
   200-encode Samply captures, inclusive `CreateHuffmanTree` CPU was effectively
   unchanged at 16,986.377 ms for insertion versus 16,980.602 ms for the parent
   (+0.03%). Five alternating Release process pairs used three warmups and 20
   samples each. Their median paired changes were a 3.72% entropy-optimization
   regression, a 0.68% codestream-encoding regression, and a noisy 0.50%
   complete-workflow improvement; entropy improved clearly in only one pair.
   Every run reported the same 110,996-byte output size. Substantial unrelated
   host load increased variance, but neither the profiles nor the timing pairs
   met the retention gate. All sort prototypes were removed, and the existing
   exact-order `std::sort` remains. Further exact-candidate reuse is now a
   stronger target than another local Huffman-sort substitution.

   The optimized and parent sample codestreams are byte-identical with SHA-256
   `4f3013a085debbb78d93043d67bffa0587cd155e62d5e84f54cadf2dbf5f0d1d`.
   The focused entropy, encoder, single-image workflow, and batch-workflow tests
   pass, as do all 22 pinned conformance fixtures and four public workflows. The
   complete Release suite remains 58/59 with the exact inherited
   `quantization_pipeline` score mismatch.

3. **Investigated and rejected (2026-08-29): globally bounded task
   parallelism.** The experiment replaced the six fresh-thread section waves
   with one lazy process-wide executor capped at eight workers. Indexed status
   slots retained deterministic error order, and executor workers cooperatively
   drained the shared queue while waiting on nested work so full-pool nesting
   could not deadlock. Concurrent callers shared the same ceiling. The six
   unique normalized cluster-cap evaluations then wrote independent indexed
   results in parallel, followed by the unchanged ascending-cap, strict-`<`
   serial reduction.

   Focused coverage exercised zero/one-task fallback, exactly-once execution,
   deterministic failures, allocation and unexpected-exception mapping,
   concurrent callers, the active-worker bound, and full-occupancy nesting.
   One hundred repetitions of that executor test passed. Repeated entropy
   optimization produced identical models and costs. The candidate matched the
   parent byte-for-byte on the checked small fixture and a 768x512 Kodak-derived
   encode; all 22 pinned decoder fixtures passed. The complete Release suite was
   59/60, with only the exact inherited `quantization_pipeline` mismatch of
   `4.4524669647216797e-05`.

   Five Latin-square process rounds compared the parent, executor-only, and
   executor-plus-cap binaries on Kodak image 01. Each process used three warmups
   and 20 measured Metal public-workflow samples at distance 1.2 and
   `maximum-throughput`. Pooled medians were:

   | Phase | Parent | Executor only | Executor + caps |
   | --- | ---: | ---: | ---: |
   | Entropy optimization | 52.140 ms | 52.409 ms (+0.52%) | 51.103 ms (-1.99%) |
   | Codestream encoding | 70.253 ms | 71.488 ms (+1.76%) | 71.297 ms (+1.49%) |
   | Complete workflow | 90.313 ms | 91.913 ms (+1.77%) | 91.170 ms (+0.95%) |

   Cap parallelism improved entropy time in every paired round by 1.29-3.60%
   relative to the parent, but complete workflow time improved in only one of
   five rounds; its paired range was -0.34% to +2.24%. Executor-only workflow
   time regressed in all five rounds by 0.91-3.66%. All 300 samples retained the
   same encoded bytes, entropy bits and clusters, entropy modes, coefficient
   order, and block-context choice.

   Three independently launched full batch matrices used one warmup, three
   samples, batch sizes 1/2/4/8, and all five benchmark workloads. The table
   below reports the median of each process's batch-time median for batch eight:

   | Workload | Parent | Executor only | Executor + caps |
   | --- | ---: | ---: | ---: |
   | 64x64 | 20.533 ms | 26.542 ms (+29.27%) | 24.652 ms (+20.06%) |
   | 512x384 | 190.722 ms | 242.920 ms (+27.37%) | 290.300 ms (+52.21%) |
   | 1080p | 1,021.094 ms | 1,363.740 ms (+33.56%) | 1,435.783 ms (+40.61%) |
   | 4K | 3,960.405 ms | 5,508.151 ms (+39.08%) | 5,282.166 ms (+33.37%) |

   For batch sizes 2-8 across 512x384, 1080p, and 4K, executor-only time
   regressed by 20.15-39.08% and executor-plus-cap time by 26.67-52.21%.
   `VarDctBatchEncoder` outer workers block while inner work is restricted to
   the global pool, so the eight-worker ceiling underuses this 14-core M4 Pro.
   The previous per-image section waves allow materially more host concurrency;
   their thread-creation cost is not the limiting factor for this workload.

   Both code changes were therefore removed. A new Samply capture was not run
   after the retention gates failed because it would profile a non-shipping
   variant. A future revisit should unify outer and inner scheduling with a
   hardware-sized or measured adaptive budget, rather than applying the former
   single-image eight-worker limit process-wide. Caller participation and task
   granularity should be measured independently before adding cap tasks again.

4. **Complete (2026-08-29): count exact ANS candidate payload sizes without
   materializing them.** `MeasureAnsCode` previously sent every section through
   the complete production token writer, including model revalidation, reverse
   chunk allocation, temporary bit-writer growth, and a second pass that copied
   the chunks into their forward serialized order. An ANS section's exact size
   is its 32-bit final state, all HybridUint extra bits, and one 16-bit chunk for
   each renormalization event. The renormalization count still depends on the
   evolving reverse ANS state, so the retained implementation shares that exact
   state traversal between the production writer and a count-only sink. The
   counter discards chunk values and performs no payload allocation or writes.
   Entropy-model serialization remains unchanged and validates the completed
   candidate once before its sections are counted.

   Focused coverage compares the measured cost with real serialized payloads
   across multiple independently reset sections, including an empty section,
   multiple contexts and clusters, HybridUint extra bits, and repeated
   renormalization. One- and two-symbol histograms retain the same equality
   check. Existing deterministic model/payload, malformed reverse-map atomicity,
   pinned codestream-hash, decoder-conformance, and public-workflow checks also
   pass.

   Five alternating parent-`863b826`/optimized Release process pairs used Kodak
   image 01, the Metal public workflow at distance 1.2 and
   `maximum-throughput`, two warmups, and nine measured samples per process. The
   host had substantial unrelated editor load, including one visibly
   contaminated parent process, so medians of the paired improvements are
   directional rather than final clean qualification: entropy optimization
   improved 5.16%, codestream encoding 4.08%, and complete workflow time 3.89%.
   The production section-writing phase was effectively flat at a 0.12% median
   paired regression. Every run reported the same 110,996-byte output size.

   Matched 1 kHz all-thread Samply captures used five warmups and 400 encodes.
   CPU delta inside `OptimizeAnsEntropyCode` fell from 27,463.855 to 19,562.470
   ms (-28.77%). The former measurement-only `WriteAnsTokenStream` subtree,
   which accounted for 19,946.150 ms, 10.18% of all sampled CPU, and 72.63% of
   the ANS phase, disappeared. ANS validation beneath the phase fell 76.12%,
   from 1,230.339 to 293.821 ms. Production ANS writing remained about 1.2% of
   total sampled CPU. Surrounding prefix work was hotter in the second capture,
   consistent with the active-host timing noise, so the profile supports the
   work-elimination mechanism rather than a whole-workflow speedup claim.

   The complete Release suite remains 58/59. Its only failure is the exact
   inherited `quantization_pipeline` score mismatch, reproduced by the parent
   build; all codestream, conformance, encoder, entropy, single-image workflow,
   and batch-workflow tests pass. The focused entropy test also passes 100
   consecutive repetitions.

5. **Complete (2026-08-29): bound histogram work and reuse immutable source
   state.** A retained-head diagnostic encode covered all 64 prefix optimizers
   and recorded 441,372 exact histogram-distance calls plus 1,031,678 shape
   comparisons. Their combined symbol spans averaged 21.49 and 20.22 entries,
   respectively, out of the fixed 128-entry alphabet; neither exceeded 40.
   Exact invalid-cost requests were 20.65% duplicates, while 40.34% of 8,572
   final prefix-code requests repeated a prior prefix histogram. The span result
   offered cheap unconditional work elimination; the duplicate result required
   a separate cache experiment because exact collision checks and stored counts
   are not free.

   Histograms now retain their highest potentially populated symbol. Count
   merges, shape distance, Huffman cost, refinement scoring, final prefix-code
   construction, and token-bit accumulation stop at that bound. Exact distance
   constructs only the bounded merged counts in uninitialized stack scratch
   instead of copying a complete histogram and mutating it through two
   full-alphabet passes. Source histogram costs remain prepared once by
   `OptimizeEntropyCode`; seed discovery and assignment now read those immutable
   sources directly rather than copying the complete source vector for every
   cap. Exact paths reject an unexpectedly unprepared source cost.

   A 1,024-entry per-optimizer direct-mapped cost cache was also tested. Hash
   matches performed full span, total-count, and count-array equality checks, so
   collisions could not change output. Five alternating process pairs showed
   entropy regressions of 3.12-5.53% and complete-workflow regressions of
   0.46-9.41%. Hashing, comparisons, count copies, and cache footprint cost more
   than the avoided trees, so the cache was removed. Storing prefix depths in
   every histogram was not pursued: after the retained changes, all Huffman work
   below `BuildPrefixCode` is only 0.36% of sampled CPU, below the cost/risk of
   enlarging every mutable histogram.

   Five final alternating Release process pairs compared parent `9c75afd` with
   the retained implementation on Kodak image 01 at distance 1.2 using the
   Metal public workflow and `maximum-throughput`. Each process used three
   warmups and 20 measured samples. Per-pair entropy-optimization improvements
   were 35.43-36.85%, codestream-encoding improvements were 26.84-29.78%, and
   complete-workflow improvements were 19.81-24.98%. Across the pooled 100
   samples per binary, medians changed from 49.680 to 31.761 ms (-36.07%) for
   entropy, 67.677 to 48.608 ms (-28.18%) for codestream encoding, and 86.770
   to 66.492 ms (-23.37%) for the complete workflow. All 200 samples had one
   identical output-size, entropy-model, coefficient-order, and block-context
   tuple.

   Three smaller alternating checks on padded 1080p improved entropy by
   19.47-21.01% and complete workflow time by 8.76-11.98%. Three sparse padded
   4K checks improved entropy by 4.85-8.06%, while complete workflow time ranged
   from a 4.52% improvement to a 1.24% regression; those three-sample 4K results
   are directional rather than a stable end-to-end claim. Every checked 1080p
   and 4K sample retained one identical output-decision tuple.

   Matched 1 kHz, all-thread Samply captures used five warmups and 400 encodes.
   Sampled CPU delta fell 54.58% overall and 61.88% inside
   `OptimizeEntropyCode`. Exact `HistogramDistance` fell 58.54%, the union of
   prefix clustering fell 70.48%, clustering work excluding Huffman fell
   88.50%, and Huffman construction fell 47.09%. `memmove` directly below
   `ClusterHistogramsFromSeeds` fell from 8,482.198 to 17.279 ms, confirming the
   source-copy mechanism. Four full Kodak PFM encodes are byte-identical to the
   parent; Kodak 01 remains 110,996 bytes with SHA-256
   `d6cfa967fa95ddd4848206eea642976c7bd1f4e7bd0d0937a110bfd99842895c`.
   The complete Release suite remains 58/59 with only the exact inherited
   `quantization_pipeline` score mismatch.

6. **Complete (2026-08-29): reduce repeated ANS candidate screening.**
   The first retained step aggregates each cluster's token values into a
   deterministic value-sorted `(value, count)` sequence. Each of the eight
   HybridUint configurations now encodes every distinct value once and weights
   its symbol population and extra-bit contribution by the checked count,
   instead of re-encoding every token eight times. The ordered section tokens
   remain unchanged for exact ANS state traversal, so model selection,
   tie-breaking, and serialized output are unaffected.

   Five alternating Release process pairs compared parent `0e0428e` with the
   aggregated implementation on `flower_510x532`, using the Metal public
   workflow at distance 1.2 and `maximum-throughput`. Each process used two
   warmups and nine measured samples. Per-pair entropy-optimization improvement
   was 7.13-8.27%, codestream encoding improved 3.65-6.81%, and complete
   workflow time improved 2.59-6.79%. The median of the five process medians
   changed from 40.958 to 37.754 ms for entropy optimization (-7.82%), 54.694
   to 51.502 ms for codestream encoding (-5.84%), and 73.864 to 70.470 ms for
   the complete workflow (-4.59%). Every sample retained the same 43,427-byte
   Metal codestream. A separate one-sample semantic sweep retained identical
   CPU and Metal codestream sizes for all 15 built-in workloads through padded
   4K, including the same aggregate byte sink. The focused entropy test passes
   100 consecutive runs, and the complete Release suite remains 58/59 with only
   the exact inherited `quantization_pipeline` score mismatch. Width-dependent
   statistic caching and safe exact-candidate pruning remain separate follow-up
   steps so their effects can be measured independently.

   The second step precomputes width validity, exact HybridUint header bits, and
   the combined screening estimate for every `(cluster, configuration, width)`
   tuple. It constructs each valid width once and caches its validated exact
   model size plus a rigorous payload lower bound consisting of the 32-bit
   final state for every section and the selected configurations' exact
   extra-bit totals. Before token traversal, a width is discarded only when its
   complete token coder is identical to another width with an equal-or-smaller
   model, or when its exact model-plus-payload lower bound cannot beat an
   already measured candidate. Equality preserves the original smaller-width
   tie preference. The floating entropy estimate ranks configurations but is
   never used as a pruning proof.

   Temporary diagnostic counters over one `flower_510x532` workflow encode
   observed 36 ANS optimizers and 144 valid width candidates. This workload had
   no complete token-coder equivalences and its payload bounds were too loose
   to reject a width before traversal. A five-pair, 20-sample alternating check
   accordingly showed no standalone speed win: four uncontaminated pairs put
   entropy optimization between a 0.15% and 1.30% regression, while one process
   was visibly contaminated across unrelated stages. This step is retained as
   correctness-preserving candidate infrastructure, not promoted as a speedup;
   the next step must share exact count-only work across surviving widths and
   pass its own retention gate.

   The final step groups surviving widths only when their context map, selected
   HybridUint configurations, and normalized symbol frequencies are identical.
   It converts each ordered token to a HybridUint symbol once for the group,
   then advances a separate exact rANS state through each width's own reverse
   map. The state transition is shared with the production writer. Thus every
   survivor retains its exact section-reset, extra-bit, and renormalization
   count without allocating or materializing candidate payloads, while the
   common token/configuration work is no longer repeated up to four times.
   Cached validated model sizes are combined with those exact token counts;
   final selection explicitly preserves the original smaller-width tie order.

   Five alternating Release process pairs compared parent `97cf864` with the
   shared traversal on `flower_510x532` under the same Metal public workflow,
   distance 1.2, and `maximum-throughput` boundary. Each process used three
   warmups and 30 measured samples. The median paired improvement was 6.47% for
   entropy optimization, 5.41% for codestream encoding, and 3.92% for the
   complete workflow. Entropy improved in all five pairs (0.09-11.35%);
   codestream and complete time each had one effectively flat/noisy pair
   (-0.45% and -0.14%, respectively). All samples retained the same
   43,427-byte output. A separate Metal-only semantic sweep matched the parent
   codestream sizes and aggregate byte sink across all 15 built-in workloads
   through padded 4K. Focused entropy, encoder, conformance, single-image, and
   batch-workflow coverage passes, including 100 consecutive entropy runs.

   A final five-pair comparison measured the complete three-step series against
   original parent `0e0428e`, again with three warmups and 30 samples per
   process. Median paired improvement was 13.66% for entropy optimization,
   10.92% for codestream encoding, and 8.72% for the complete Metal workflow.
   All five pairs improved all three boundaries. One broadly slower parent
   process raised the maxima to 20.48%, 16.00%, and 17.17%; across the other
   four pairs, the respective ranges were 12.93-13.77%, 10.39-11.41%, and
   7.80-8.88%. Section writing was not changed by this series.

   Future speed comparisons must not use the 510x532 flower alone. The retained
   regression set is `flower_510x532`, `padded_1080p`, `padded_1440p`, and
   `padded_4k`, run with `just encoding-regression-benchmark`. Real
   high-resolution images are added with `just encoding-image-benchmark IMAGE`;
   normal image conversion happens once before warmups and is excluded from the
   reported encoder stages.

   A post-integration high-resolution audit compared parent `0e0428e` with
   current head `4b6b27e` under Metal fully-resident AQ at distance 1.0. Three
   alternating process pairs used two warmups and five samples for the built-in
   workloads. Median paired improvement was 14.99% for the complete padded
   1080p workflow and 5.22% for padded 4K; entropy optimization improved 19.70%
   and 11.27%, respectively. A separate 4672x5584 photographic input used one
   warmup and three samples per process. The source JPEG SHA-256 was
   `980b46d44ffdc73e2f1ddfe1ec1e4b208cc8eb647cf3871ba7386b3cb4f83654`.
   It improved the complete workflow in
   every pair by 10.23-14.89% and entropy optimization by 24.75-29.30%, with
   byte-identical direct-CLI output. The unchanged quantization phase remained
   noisy, and the direct CLI reached roughly 4-5 GB maximum resident memory, so
   isolated wall-clock observations on this 26-megapixel case can vary despite
   the retained codestream speedup.

   A high-resolution follow-up replaces the full occurrence sort inside
   `AggregateValues` for inputs of at least 4,096 values. Common raw values
   below 65,536 are counted in a bounded dense table; only distinct larger
   values enter a sparse map and sorted suffix. Smaller inputs retain the
   original sort. Temporary diagnostics on padded 4K at distance 1.2 showed
   that the consequential arrays contained roughly 100 thousand to 9.5 million
   values but generally only 5-200 distinct values, and all values took the
   dense path.
   The added entropy fixture crosses the counting threshold and includes
   `UINT32_MAX`, so both dense and sparse behavior are exercised.

   Three rotated Release rounds compared `951df02`, current `4b6b27e`, and the
   counted implementation. Each independent process used the Metal public
   workflow, fully-resident SIMD AQ, two warmups, and five measured samples.
   Percentages below are medians of three paired process-median changes; a
   negative value means that counting is faster.

   | 3839x2159 workload | Total vs current | Entropy vs current | Total vs `951df02` | Entropy vs `951df02` |
   | --- | ---: | ---: | ---: | ---: |
   | Padded, distance 1.2 | -35.67% | -52.45% | +7.44% | +22.83% |
   | Padded, distance 1.0 | -22.32% | -30.36% | +12.65% | +24.27% |
   | Flower | +5.69% | -4.91% | +8.76% | +2.36% |
   | Keong Macan | -5.70% | -15.26% | -1.66% | -5.38% |
   | Riaphotographs | -0.61% | -8.70% | +3.52% | -2.24% |
   | Bliznaca | -9.53% | -3.54% | -2.78% | +3.21% |

   All 90 current/counted benchmark sample decision records are identical,
   including the 1,638,673-byte padded distance-1.2 codestream. A separate
   seeded 3839x2159 PFM encode is byte-identical between those builds and
   decodes successfully with `djxl`. `951df02` predates native ANS selection
   and produces 1,640,942 bytes for padded 4K at distance 1.2, so its speed
   advantage also carries a 2,269-byte (0.138%) size cost in this case. The
   complete rebuilt Release suite remains 58/59 with only the exact inherited
   `quantization_pipeline` score mismatch reproduced by the unmodified current
   build.

7. **Complete (2026-09-01): reuse AC token values and traversal order across
   block-context candidates.** The former candidate loop tokenized every AC
   coefficient stream independently for each block-context map and again for
   natural versus custom coefficient order. Those passes repeated coefficient
   loads, zero-density state, signed packing, value storage, and allocation even
   though only the final context labels depended on the block map.

   The retained implementation builds one map-independent template per
   coefficient-order choice. Each group stores its values in one contiguous
   `uint32_t` array and a compact four-byte descriptor containing the local
   context and block-context inputs. Candidate preparation resolves only a
   two-byte context array. Prefix optimization, ANS optimization, exact
   count-only measurement, and final token writing accept a read-only split
   value/context view, so candidate streams share the original values without
   rebuilding interleaved eight-byte tokens. Descriptor and block-key storage
   is released after all contexts have been resolved and before entropy search.

   Raw workflow schema 8 separates aggregate template-construction and context-
   materialization work and records their pass/token counts. On the 4672x5584
   `doughnuts` PFM at distance 1.0, the selected search built two templates
   containing 22,115,824 tokens total, then materialized 12 context variants
   covering 132,694,944 token positions. The exact parent performed 12 full
   tokenization passes. In the quiet final parent run, aggregate full-
   tokenization work was 1,696.864 ms; the two candidate runs used
   147.310-188.621 ms for template construction and 246.579-261.710 ms for
   context materialization. These are summed worker times, not latency.

   Four final independent Release processes ran parent/candidate/candidate/
   parent with one warmup and three samples each under the fully-resident Metal
   workflow. One parent process was broadly contaminated. Against the quiet
   parent median, the two candidate medians reduced AC-tokenization wall time
   from 375.754 ms to 229.661-233.748 ms (-38.88% to -37.79%) and codestream
   time from 2,181.610 ms to 1,816.156-1,946.515 ms (-16.75% to -10.78%).
   Entropy optimization ranged from an 8.67% improvement to a 0.13% regression,
   consistent with variable memory pressure. Every sample remained 2,690,877
   bytes. The nearest quiet distance-1.2 pair improved codestream time 8.22%
   and AC tokenization 38.21% at an unchanged 2,348,582 bytes. The closest
   padded-4K comparisons improved codestream time 10.66-11.91% at an unchanged
   2,132,228 bytes; the second parent there was also contaminated.

   Alternating one-sample memory measurements on the high-resolution input
   reduced maximum RSS from 5,126,012,928 to 4,935,778,304 bytes (-3.71%) and
   peak footprint from 14,138,386,224 to 13,478,008,256 bytes (-4.67%). A
   direct CLI comparison produced byte-identical 2,690,877-byte codestreams
   with SHA-256
   `74dab8b328b12d3a485006b5197059df5d580e1b2aa80ee35a2dfdbc34f2d249`;
   `djxl` 0.12.0 decoded the candidate to PFM. Split/interleaved parity tests
   cover prefix and ANS model selection, costs, exact counting, serialized
   payloads, and malformed-view atomicity. The complete Release suite is 61/62
   with only the exact inherited `quantization_pipeline` score mismatch.

   A second retained step reuses exact value statistics between the prefix and
   ANS searches. Prefix HybridUint configuration search already collects every
   selected cluster's values and reduces them to deterministic, value-sorted
   `(value,count)` populations. ANS formerly traversed all ordered tokens again,
   copied their raw values into new cluster vectors, and repeated the same
   aggregation before trying its configurations. The prefix optimizer now
   returns those populations with their context partition, and ANS consumes
   them directly. It still traverses the original ordered streams for exact
   rANS state/renormalization costs, so symbol order and every coding decision
   remain unchanged. Before reuse, the internal path checks context-map
   identity, cluster count, strictly increasing values, nonzero/overflow-safe
   weights, stream validity, and the exact total token count. Malformed input
   leaves model and cost outputs unchanged.

   Raw workflow schema 9 retains ANS value-collection and aggregation phases as
   zero-valued work-elimination sentinels and adds prepared-value validation.
   On padded 4K at distance 1.2, two stable alternating Release pairs used one
   warmup and three samples per process. Entropy optimization improved
   10.38-10.68%, codestream encoding improved 9.03-9.34%, and the complete
   fully-resident Metal workflow improved 7.54-7.66%; every sample remained
   1,638,673 bytes. The removed collection and aggregation accounted for
   1,163-1,174 ms of aggregate parent worker time. A final schema-9 rerun had
   one broadly contaminated parent process; against the quiet parent, its two
   candidates still improved entropy by 13.67-21.61% and codestream time by
   11.46-16.25%. Prepared-value validation itself took only 0.129-0.147 ms of
   aggregate worker time.

   Two high-resolution distance-1.0 pairs on the 4672x5584 PFM improved entropy
   optimization by 6.60% and 13.10%, codestream encoding by 5.78% and 14.23%,
   and complete workflow time by 6.70% and 6.07%, respectively, at an unchanged
   2,690,877 bytes. A three-pair 510x532 flower check was mixed at this shorter
   boundary: entropy changed by -7.08%, -20.27%, and +11.87%, while the pooled
   process-sample median improved 3.37%. Direct parent/candidate CLI output on
   the high-resolution input remained byte-identical at SHA-256
   `74dab8b328b12d3a485006b5197059df5d580e1b2aa80ee35a2dfdbc34f2d249`,
   and `djxl` 0.12.0 decoded the candidate to PFM. Focused parity coverage
   compares prepared and legacy prefix/ANS models, exact costs, and malformed
   prepared-input atomicity; it passes 100 consecutive runs. The rebuilt
   Release suite remains 61/62 with only the inherited
   `quantization_pipeline` score mismatch.

   A third experiment attempted to fuse the remaining fixed-HybridUint prefix
   histogram pass into block-context materialization. It accumulated one exact
   128-bin population per resolved AC context while each token was already in
   hand, then moved those populations directly into prefix clustering. On
   padded 4K at distance 1.2 this reduced aggregate prefix-histogram work from
   518.505 ms to 9.578-9.799 ms, but it scattered updates across thousands of
   roughly 1 KiB histograms. Aggregate context-materialization work rose from
   135.535 ms to 2,046.490-2,252.485 ms, and AC-tokenization wall time rose from
   90.519 ms to 393.242-401.531 ms. Against the quiet parent, codestream time
   regressed 22.56-28.91% despite identical 1,638,673-byte output. The entire
   code experiment was removed. A future revisit would need cache-local group
   histograms plus a reduction, not direct random writes to the global context
   table.

   A fourth retained step removes growth from the selected prefix partition's
   raw-value collection. The fixed-HybridUint source histograms already contain
   each initial population, so the selected context partition now reduces those
   totals into exact cluster offsets, allocates one contiguous `uint32_t`
   buffer, and fills each cluster through a bounds-checked cursor. Mutable-span
   aggregation sorts or counts each cluster directly in its flat segment. The
   owning aggregation overload remains available to the legacy ANS path.

   Three extended padded-4K distance-1.2 pairs alternated independent parent
   and candidate Release processes with one warmup and five samples each. The
   aggregate prefix-value-collection counter improved in every pair by 3.38%,
   6.55%, and 5.93%; the median paired changes were 5.93% for value collection,
   3.88% for entropy optimization, 2.75% for codestream encoding, and 4.63% for
   the complete workflow. The pooled 15-sample medians improved 4.59%, 5.70%,
   6.23%, and 6.69%, respectively. Unrelated AC-tokenization and GPU timing
   remained noisy, including one pair whose complete workflow regressed 4.02%,
   so the targeted counter is the firmer attribution. All 30 samples retained
   identical entropy decisions and the same 1,638,673-byte output.

   On the 4672x5584 photographic input, the nearest quiet pair reduced value
   collection from 1,105.616 to 919.501 ms (-16.83%); subsequent parent runs
   were allocator-contaminated while candidate collection remained in a much
   narrower 883.874-992.742 ms range. A direct distance-1.0 CLI comparison was
   byte-identical at 2,690,877 bytes and SHA-256
   `74dab8b328b12d3a485006b5197059df5d580e1b2aa80ee35a2dfdbc34f2d249`,
   and `djxl` 0.12.0 decoded the candidate to PFM. Alternating padded-4K memory
   checks reduced maximum RSS by 8.79-9.38%, while reported peak footprint rose
   0.93-1.00%; high-resolution memory results were mixed. Focused entropy,
   encoder, workflow, and raw-profile tests pass, including mutable-span small
   and counted aggregation coverage.

8. **Retain the implemented serializer-effort tradeoff.** Balanced and
   high-density modes now commit one derived map/order representation, choose
   Prefix or ANS before model construction, and use direct ANS policies modeled
   after pinned libjxl efforts 7 and 9. The former exhaustive map/order/coder
   tournament is available only through maximum compression. The twelve-photo
   qualification measured a 2.15x median balanced codestream speedup and a
   1.16% median byte reduction relative to the former path; one high-density
   input regressed 2.55% and remains a named stress case. Preserve the
   three-policy conformance and measurement gates in
   [`entropy-behavior-alignment.md`](entropy-behavior-alignment.md) rather than
   restoring the outer tournament to an ordinary effort tier.

9. **Defer GPU entropy coding until a residual profile justifies it.** The
   dominant exact operation builds many small, branch-heavy, depth-limited
   128-symbol Huffman trees with deterministic tie behavior. Seed assignment
   also mutates cluster state between decisions, and the CPU serializer needs
   the result immediately. Those properties are a poor match for low-latency
   Metal dispatch even on unified memory. A future multi-image implementation
   could batch histogram reductions, shape-distance matrices, or many
   independent tree builds, but it would require enough cross-image work to
   amortize dispatch and synchronization. CPU work elimination, scratch reuse,
   and bounded task scheduling should be measured first.

For exact-output changes, retain the current deterministic serializer fixtures,
complete-codestream hashes, pinned-`djxl` acceptance, and decoded-pixel checks.
Performance qualification should add call counts and time for shape screening,
exact refinement, each cap, each block-map/order candidate, Huffman retries,
and executor queue/run time. Use warmups plus repeated alternating builds for
latency, and independent repeated batch processes for throughput. Re-profile
the winning implementation so a shifted hotspot, rather than the original
sample percentages, determines the next optimization.
