# Frozen Modular CPU oracle, version 1

`capture.cpp` generates deterministic stage fixtures and measures real workflow
calls. `gjxl_modular_capture` links only native libraries. The separate
`gjxl_modular_reference_capture` target additionally checks the exact fixtures
against the clean pinned libjxl build. Neither target is installed or exported.

Run the native gate without a reference build:

```text
cmake --build BUILD --target gjxl_modular_capture
python tools/modular_oracle/verify.py BUILD/gjxl_modular_capture testdata/modular OUTPUT
```

Use `.exe` on Windows. CTest registers `modular_frozen_oracle`; configuring
`GJXL_MODULAR_ORACLE_BUILD` also registers `modular_frozen_reference` and enables
the independently linked capture executable. The latter compares transformed
planes, shapes/shifts, tree tokens, all 14 predictions and 16 properties, leaf
residual/context tokens, and every decoded sample. Palette ordering uses an
independent ordered dictionary and pinned shape application. Broader geometry,
transform-prefix and arithmetic-extrema checks remain in `modular_policy` and
`modular_transform`; this compact corpus complements them.

The normal Native Libraries CI matrix builds the pinned reference with
`gjxl_pinned_decoder`, then reconfigures with `GJXL_MODULAR_ORACLE_BUILD` set to
that build. All 11 `modular_*` tests run on macOS, Linux and Windows under both
C++20 and C++23, before the broad build/test steps. Explicit qualification targets
make missing reference configuration a build failure; Python is required when
the oracle is enabled so CLI and frozen-reference checks cannot be omitted.
CI retains the CMake caches, Modular JUnit report, frozen artifacts and test logs.
The remaining suite excludes `modular_*` to avoid running the qualification twice.

An installed native package can run the same gate using the matching private
source headers, without exporting those headers or linking libjxl:

```text
cmake -S tools/modular_oracle -B CAPTURE -DCMAKE_PREFIX_PATH=INSTALL -DGJXL_SOURCE_DIR=SOURCE
cmake --build CAPTURE
python tools/modular_oracle/verify.py CAPTURE/gjxl_modular_capture testdata/modular OUTPUT
```

## Corpus and provenance

The generator in `Corpus()` is part of the versioned fixture specification.
All six formats occur: Gray8, Gray16, RGB8, RGBA8, RGB16, RGBA16. Inputs are packed
little-endian integers with sRGB semantics and unassociated alpha. Transparent
pixels retain nonzero color. Seven categories cover tiny images, 16-bit ramps,
screen-like blocks, palettes, deterministic xorshift noise, alpha, and photography.
Freeze dimensions are 1x1, 257x9 and 125x125. Benchmark dimensions are 1x1,
513x257 and 625x125. The photographic benchmark tiles five copies horizontally.

`testdata/modular/photo.json` records the pinned source and SHA-256. The CC0
photograph is subsampled at coordinates `(4*x, 4*y)` without interpolation or
transfer conversion. Its license is retained beside the fixture. To regenerate:

```text
python tools/modular_oracle/prepare_photo.py third_party/libjxl/testdata/external/wesaturate/500px/tmshre_riaphotographs_srgb8.png OUTPUT.ppm
```

`testdata/modular/oracle-v1.json` freezes 286 artifacts across 14 prescribed
fixture/coder combinations. The manifest records the native base revision,
pinned reference revision, capture source hash and input hash. Qualification
records retain the complete source hashes for the working-tree additions.
`verify.py` never writes expected hashes. A mismatch retains actual files for
inspection. Compare integer records and independently check the changed stage
before proposing a separately reviewed oracle version; never regenerate the
manifest as part of a test run to make a failure disappear.

## Record schema

All `.bin` integers are exactly eight bytes, little endian. Signed values use
two's-complement modulo 2^64. There are no native structs, pointers, padding,
locale-dependent numbers or wall-clock values. Enum values come from the
versioned native declarations. Record lengths are inferred from counts below
and the file length; rows follow canonical stream/channel/raster order.

| Artifact | Records |
| --- | --- |
| `input.raw` | Packed little-endian source bytes |
| `input.bin` | width, height, channels, bits, packed-format enum, entropy enum, CPU participants (1) |
| `policy.bin` | RCT, transform count, node count; seven WP coefficients; four weights; each transform: kind/begin/count/colors/horizontal/in-place; each node: property/split/left/right/predictor/offset/multiplier |
| `source.bin`, `transformed-N.bin` | channel count, metadata count; per channel: width/height/hshift/vshift/role then signed raster samples. N=0 is after RCT; N>0 includes the first N transforms |
| `streams.bin` | per stream: role/id/section/slice count/token count, then each slice: channel/x/y/width/height; includes empty streams |
| `tree-tokens.bin`, `tokens.bin` | context/value pairs, respectively tree serialization and packed signed residuals |
| `decisions.bin` | per sample: stream index/channel/x/y/leaf index/prediction/residual; 16 properties; predictions 0..13. WP value/property is zero when the prescribed tree does not use WP |
| `model.bin` | mode/context count/cluster count/ANS alphabet log; context map; each HybridUint config: split/msb/lsb; prefix mode: each cluster's degenerate symbol, 128 depths, 128 codewords; ANS mode: each cluster's frequency count/method/omit position then normalized frequencies |
| `populations.bin` | sorted cluster/symbol/count triples, from the resolved HybridUint configuration, before ANS normalization |
| `model.raw` | serialized global entropy model, including no-LZ77 marker; last byte zero padded |
| `stream-bits.bin` | model bit count, then stream-index/payload-bit-count pairs; empty payloads have zero bits; stream headers/tree/TOC are excluded |
| `stream-N.raw` | actual entropy payload, zero padded final byte; stream headers excluded |
| `final.jxl` | complete native file, checked against profile-enabled and four-participant calls and independently decoded in the reference target |
| `learned-single.bin`, `learned-split.bin` | separately frozen learner decisions over this transformed image; same record schema as policy; these decisions are policy baselines, not arithmetic truth or a complete transform program |

The exact original transform settings live in `policy.bin`. Search is disabled
for arithmetic fixtures. Timing and heuristic improvements may be reviewed
independently of these prescribed decisions.

## Measurement protocol

```text
BUILD/gjxl_modular_reference_capture benchmark testdata/modular/photo.ppm OUTPUT 3
python tools/modular_oracle/summarize.py OUTPUT/timings.csv OUTPUT/summary.json
```

Run on an otherwise idle machine. Each configuration has one first call (run 0)
and three subsequent calls; first call does not imply a cold process or cache.
Each call creates a fresh domain and fresh output before timing. Times include
admission and publication, exclude fixture generation, reference decoding,
recording and domain construction. All native results are deterministic across
runs. Serial/four-participant runs, prefix/ANS, and prescribed/search policies
are separate configurations. Selected policies and final files are retained per
configuration; replaying the selected policy must produce identical bytes.
The CSV also records how many complete candidates search encoded.

Private opt-in scopes measure exclusive caller wall time: input unpacking and
allocation; RCT/palette/squeeze; learner and palette probe; tokenization including
joined workers; global data-model optimization; tree/model/header/payload
emission including joined workers; final headers/TOC/assembly/publication copy.
Nested input/transform scopes do not double count. Search accumulates work for
all candidates, including losers. Preparation planning, admission, layout,
some cleanup and result publication remain in complete-call time but outside
named scopes. Their sum must not exceed complete-call latency. Hooks use no
clock or allocation when disabled. They do add a thread-local null check.

`managed_peak` is observed managed backing capacity, not RSS. `planned_capacity`
is the conservative admission bound, not actual allocation. Caller input/output,
fixture/reference/trace storage, stacks, and control objects are excluded.
Reference memory columns are zero meaning **unmeasured**, never zero allocation.
Reference effort is fixed at 1, lossless and keep-invisible, using the same image
and one calling thread without a parallel runner. It repeats once for each
native entropy mode; the summary retains both sample sets. Native search and
reference effort are unrelated policies. This is context, not an effort-equivalence
or speed/size-parity claim. Throughput is source megapixels per complete-call
second. The corpus establishes a baseline; it is not a broad compression study.
