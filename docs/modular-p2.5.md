# P2.5: CPU scheduling and public lossless integration

P2.5 exposes the bounded lossless Modular profile through C++, C, Rust and the
native CLI. Automatic Modular execution uses CPU. Explicit Metal/CUDA requests
are unsupported by the Modular encoder; context creation can report unavailable
first when the requested backend is not built. VarDCT quality, dispatch and batch
semantics remain independent.

## Public interfaces

The installed `gjxl/modular.hpp` header provides `ModularImageView`,
`ModularEncodingOptions`, `ModularEncodingSummary` and `EncodeModularImage`.
It depends only on common geometry, status and execution-domain types. Transform,
predictor, tree, stream and storage-planning types remain private. Link
`gjxl::codestream`.

```cpp
#include <gjxl/modular.hpp>
std::vector<uint8_t> encoded;
gjxl::ModularImageView image{rgba_bytes, {width, height}, stride_bytes,
                            gjxl::ModularPixelFormat::kRgba16,
                            gjxl::ModularByteOrder::kBigEndian};
gjxl::ModularEncodingOptions options;
options.cpu_thread_count = 4;
options.search = true;
gjxl::ModularEncodingSummary summary;
auto status = gjxl::EncodeModularImage(image, options, &encoded, &summary);
```

All six gray/RGB/RGBA 8/16-bit layouts are supported. Byte order is explicit for
16-bit input; unaligned buffers, padded byte strides and an unpadded final row
are allowed. RGB samples use sRGB and gray samples use its transfer function;
alpha is unassociated at the color depth. Invisible RGB and all alpha samples
are preserved. No float conversion occurs. Defaults select prefix coding and
identity/gradient policy. `search=true` enables the bounded P2.3/P2.4 selector;
`ModularEntropy::kAns` selects ANS independently. There is no distance or quality
parameter. Output and summary replacement are atomic. The summary reports source
extent, encoded bytes, resolved CPU backend and whole-call wall seconds,
including admission waits.

The C API adds `GJXLModularOptions`, `gjxl_modular_options_init` and
`gjxl_encode_modular`. It reuses contexts, domain handles, `GJXLImageView`,
`GJXLBuffer`, thread-local errors and `gjxl_buffer_free`. Null Modular options
select defaults. Four-byte options prefixes are valid; absent search/entropy
fields default to zero, and larger future tails are ignored on input. Existing
pixel enum values 1/2 retain RGB8/RGBA8; values 3..9 add gray8 and explicitly
little/big-endian gray16/RGB16/RGBA16. Existing VarDCT `gjxl_encode` still rejects
unsupported formats and nonopaque alpha. Both routes now share the C publication
helper and its existing allocation/ownership contract.

Rust bindgen automatically exposes the new C declarations. The safe wrapper adds
`ModularOptions`, `ModularEntropy`, all integer `PixelFormat` variants and
`Context::encode_modular`. `ImageView::new` checks the corresponding byte layout.
The existing `encode` method retains its VarDCT behavior.

## Scheduling and resources

The private workflow defaults to one thread and remains callable as the serial
CPU oracle. Public CPU limits accept zero for automatic or 1..256 as an upper
bound; this implementation uses at most four participants, further limited by
nonempty streams and the shared domain. One participant selects the serial path.

Tokenization first computes canonical stream offsets into one preallocated token
array. Workers write disjoint spans and stream-view entries; weighted state is
local to each active stream. Tree learning and entropy-model construction remain
serial. Global headers, global samples and tree/model serialization are serial;
independent remaining streams emit to distinct section writers using a fixed,
read-only entropy model. TOC and file assembly retain canonical order. Worker
completion order therefore does not affect bytes.

`RunModularStreams` uses the existing CPU worker-group permits, inherited managed
resource context, worker launch and join mechanism. Tokenization and emission have
separate fault-injection launch sites. Partial launch failure joins every started
worker before returning, and publication remains atomic. Small/global-only images
avoid unnecessary worker launches.

Shape-based storage planning adds canonical-offset storage, status/thread records,
concurrent weighted states and concurrent entropy-emission scratch. Search bounds
include the same parallel candidate envelopes and retained-winner overlap. The C
adapter admits additional publication-copy overlap before acquiring CPU capacity.
Concurrent Modular and VarDCT calls share the same domain's memory and CPU caps;
no mixed-mode batch API is introduced.

## Exact integer CLI

```sh
gjxl_encode --modular --input-color-space srgb --search --threads 4 input.pam output.jxl
gjxl_encode --modular --input-color-space srgb --entropy ans input.ppm output.jxl
```

The private PNM reader supports binary P5 (gray), P6 (RGB), and P7 PAM with
`GRAYSCALE`, `RGB` or `RGB_ALPHA` tuple types. `MAXVAL` must be 255 or 65535;
16-bit file samples are big endian. It rejects truncated/trailing raster data,
unsupported tuple types, dimensions/ranges, unknown PAM metadata and malformed
headers. The CLI requires `--input-color-space srgb` as an explicit declaration:
PNM does not reliably identify the intended transfer function, and samples must
not silently be relabeled. Other declarations are rejected. File loading happens
before encoding admission; this frontend storage is not part of the managed
encoder envelope.

The Modular parser rejects VarDCT-only controls such as `--distance`. It dispatches
before the existing PFM/linear-float path and uses the same atomic output-file
writer. The existing normal-image/float wrappers are not lossless Modular input
routes. Use integer PGM/PPM/PAM with the explicit source declaration.

## Qualification and next step

The [machine-readable record](modular-p2.5-results.json) records final commands,
source and artifact hashes, platform constraints and baseline failures. Tests
cover independently decoded public C/C++ output, serial/parallel policy and
transform parity, sized C prefixes, publication allocation failures, partial
worker-launch failure and recovery, mixed-mode domain caps, CLI malformed-input
atomicity, native Rust wrappers, and installed C11/C++20/C++23 consumers and
relocation. P2.6 remains the CPU-oracle freeze and GPU-boundary milestone.

| Check | Result |
| --- | --- |
| Public C/C++ format, byte-order, coder and search combinations | 72 exact serial/parallel cases |
| C publication allocation failures | 312 positions, atomic output |
| Worker launch faults | 8 first/partial-launch cases across both stages |
| Mixed Modular/VarDCT domain | Four concurrent calls; memory/CPU caps and draining pass |
| CLI files / invalid requests | 20 exact independent decodes / 12 atomic rejections |
| Native Rust tests / independent file decodes | 12 pass / 18 exact |
| Private workflow allocation failures | 51,134 positions across 44 cases |
| Frozen default / transformed / policy Modular files | 2,802 / 626 / 1,646 unchanged |
| Frozen VarDCT artifacts / pinned decoder fixtures | 208 unchanged / 23 pass |
| Installed C11 and C++20/C++23 consumers, header manifest and relocation | Pass |
| Final qualification across broad and focused runs | 123/132 pass; same nine baseline failures |

The initial 132-test broad run omitted the MSVC environment initialization, so
the toolchain probe could not find standard headers and the installed-consumer
configure could not find `kernel32.lib`. Both passed after restoring that
environment in an 11-test focused run. That rerun also qualified the final shared
C publication helper and headers; all 11 tests passed. The private workflow had
passed the broad run and separately passed against the reference-disabled installed
libraries with matching private source headers.

The nine remaining baseline failures are the two Windows contract-test build
failures, five Butteraugli reference/differential targets, missing generated
Butteraugli goldens, and the known 255x256 DC-processing oracle mismatch. Eight
corresponding broad-build failures remain. No P2.4 passing test was lost.

Native Modular's include closure has 65 files and no VarDCT-policy, GPU or libjxl
headers. The 90-file installed package adds only `gjxl/modular.hpp` to the header
surface; private Modular headers and reference exports remain excluded. Raw
qualification evidence is under `build/modular-p2.5/`. Rust artifacts are retained
there in `rust-artifacts/`; the candidate build retains the CLI and oracle files.

This qualification uses the audited Windows x64 MSVC toolchain under ARM64
emulation and pinned libjxl `e8ff09762481785938d8e4e01333ed3917571161`.
Rust uses the x64 MSVC host toolchain under the same emulation. No GPU, native
ARM64, other-platform, sanitizer or corpus performance qualification is claimed.
