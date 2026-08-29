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
Throughput mode additionally performs one AQ update instead of the default two;
fully resident mode continues to honor the requested iteration count.
Maximum-throughput mode instead fixes every transform to DCT8, applies the
resident `AdjustQuantBlockAC` shared-quant decision to the adjusted initial
field, and stops before inverse reconstruction or perceptual scoring. Its
summary score history is empty; the policy must be evaluated with an
independent decoder and quality metric.

The `gjxl_encode` frontend accepts three-channel linear-RGB PFM input and one
of `--distance`, `--maximum-error`, `--target-bytes`, or `--target-bpp`, plus
`--backend auto|cpu|metal` and
`--metal-aq exact-coefficients|fully-resident|throughput|maximum-throughput`.
`--high-density` selects four AQ updates for Butteraugli-target and target-size
control. It is rejected with maximum-error, throughput, and maximum-throughput
policies rather than being ignored or silently overridden.
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

The checked `17x13` sample encodes to 259 bytes at target `1.0`; its codestream
SHA-256 is
`82f7936f5fc932dd0b484705e9f01d1e18e3e11aa8a7545b8cc082acf136af17`.
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
# Four-update density search on the exact-coefficient path:
build/release/gjxl_encode --distance 1.0 --high-density \
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
writer can serialize either prefix or ANS entropy models.
Ordinary Butteraugli encoding uses libjxl's target- and pixel-dependent X/B
quantization-matrix scales; maximum-error encoding retains `2/2`.

A directional local comparison used the Release encoder, four natural images
around 500 pixels in size, identical linear-PFM inputs, raw `.jxl` output, and
libjxl tools at `05be1775629f7fdc01beb70c118043b4b0c69d2a`.
External Butteraugli scores were measured at 255 nits. The aggregate encoded
sizes were:

| Target distance | Initial | After #1 | After #2 | After #3 | After #4 | After #5 | #5 vs #4 | `cjxl -e 7` | #5 gap | `cjxl -e 9` | #5 gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 153,464 B | 153,278 B | 155,391 B | 152,722 B | 152,111 B | 148,137 B | -2.61% | 151,714 B | -2.4% | 147,386 B | +0.5% |
| 2 | 94,430 B | 94,327 B | 95,072 B | 92,746 B | 92,065 B | 87,714 B | -4.73% | 93,414 B | -6.1% | 88,578 B | -1.0% |

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

LZ77, adaptive DC smoothing, additional transform decisions, patches/dots/
splines, chroma subsampling, and GPU entropy coding are not first-line size
work. They target narrower content, change quality semantics, or require more
implementation scope for less likely general-image benefit. Each accepted
optimization should be evaluated on a broader corpus with decoded-output
validation and matched-quality size comparisons, not only the same nominal
distance.

## Codestream performance profile and optimization priorities

A symbolized Samply capture on Apple M4 Pro isolated the host codestream tail
after the prefix-density and custom-coefficient-order changes. The capture used
the Release `gjxl_quantization_benchmark` binary at
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
   itself was 31.41% hotter than the former stable sort, so a faster
   allocation-free total-order sort remains a possible follow-up. These sampled
   CPU deltas are attribution evidence rather than benchmark timing claims.

   The optimized and parent sample codestreams are byte-identical with SHA-256
   `4f3013a085debbb78d93043d67bffa0587cd155e62d5e84f54cadf2dbf5f0d1d`.
   The focused entropy, encoder, single-image workflow, and batch-workflow tests
   pass, as do all 22 pinned conformance fixtures and four public workflows. The
   complete Release suite remains 58/59 with the exact inherited
   `quantization_pipeline` score mismatch.

3. **Use bounded task parallelism after reducing work.** The seed order is a
   serial prerequisite, but the six cap evaluations are independent once it is
   available. They can produce indexed results in parallel followed by the
   existing deterministic, strict-cost reduction. Per-cluster prefix-code
   construction is also independent, although its tasks may be too small unless
   chunked. The current `RunParallelSections` creates and joins fresh
   `std::thread`s for each call; the profile recorded 5,027 thread records for
   400 encodes. A shared persistent executor with a concurrency budget is a
   better foundation than nested worker creation.

   Concurrency policy must distinguish latency from throughput. A single image
   can use idle cores for cap-level work. `VarDctBatchEncoder` already uses
   persistent outer workers for independent images, so a saturated batch should
   normally spend its budget across images rather than multiply inner workers
   and oversubscribe the host. Measure single-image wall latency and batch
   images/second separately; CPU-time reduction cannot be inferred from a
   wall-latency improvement.

4. **Offer an explicit serializer-effort tradeoff if rate changes are allowed.**
   `maximum-throughput` reduces AQ work but still invokes the full prefix search
   for each eligible entropy candidate. A speed-oriented serializer policy
   could cheaply screen coefficient-order and block-context candidates, search
   fewer cluster caps, or retain the legacy partition when predicted savings are
   small. These entropy choices do not alter coefficients, decoded pixels, or
   Butteraugli distortion, but they may increase file size. Candidate winner
   rates, per-candidate time, bytes saved, and complete-codestream size must
   define this policy; one losing custom-order example is not sufficient.

5. **Defer GPU entropy coding until a residual profile justifies it.** The
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
