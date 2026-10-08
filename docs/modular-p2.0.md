# P2.0: Modular foundations and independent reference infrastructure

This milestone implements the foundations in
[the architecture plan](modular-architecture.md#p20-shared-metadata-channels-grouping-and-oracle-infrastructure).
It does not enable whole-image Modular encoding. The first native RGB8 workflow,
including complete admission and output publication, remains P2.1.

## Implemented contracts

| Component | Responsibility |
| --- | --- |
| `codec/frame_metadata.*` | Private image/sample/color/alpha values and an explicit frame-mode discriminator, without writer dependencies |
| `codec/modular/image.*` | Managed integer planes, channel descriptors, bounded borrowed views, and image storage planning |
| `codec/modular/geometry.*` | Source-pixel geometry with 256-pixel groups and 2048-pixel DC groups; no DCT padding |
| `codestream/modular/stream_plan.*` | Canonical stream IDs, section assignments, channel rectangles, borrowed slices, and plan storage bounds |
| `codestream/headers_internal.h`, `headers.cpp` | Metadata-driven common headers and their storage bound |
| `codestream/vardct/headers.cpp` | Existing public VarDCT compatibility adapters and VarDCT-only section syntax |
| `tests/modular_reference.*`, `modular_oracle_test.cpp` | Independent pinned decoding, header/TOC parsing, and prescribed-stage reference checks |

All new native headers remain private. Existing public signatures, installed
header lists, package components, VarDCT validation, and workflow dispatch remain
unchanged. Reference targets are optional and never installed or linked by
production targets.

### Channels, arithmetic and ownership

`ModularImage` owns one managed record array and one contiguous `int32_t` plane
per channel. Allocation ownership is inherited from the caller's resource scope;
P2.1 preparation should use its preparation scope. Views carry a backing span,
element stride, extent, shifts, and role. A nonzero input offset is represented
by a subspan. Validation checks the last addressed element, overflow and pointer
range before any slicing. A borrowed view must not outlive its backing owner.

Identity preparation can represent every unsigned 8/16-bit sample exactly:
`0 <= sample <= 65535 < INT32_MAX`. Arithmetic intermediates use `int64_t`.
For a later identity/gradient tokenizer, `N + W - NW` lies in
`[-65535, 131070]`; clamping to neighboring samples gives `[0, 65535]`, so its
residual lies in `[-65535, 65535]` and its signed-packed value fits in 17 bits.
No floating-point normalization is required. This is not a range proof for
future transforms or configurable predictors; each needs its own proof and
independent checks before being enabled.

Metadata channels form an explicit leading prefix. Other channels can have
different extents and shifts. Empty streams are represented, but empty channel
descriptors are rejected. No transforms or packed-input conversion are performed
by this milestone. The existing extracted weighted predictor is unchanged.

Image and stream-plan bounds are allocation-free on success and describe fresh
output backing. Atomic replacement retains the previous output while building
the new one: callers add the old backing to their admission plan. Failed
construction leaves the previous value intact and releases partial allocations.
Managed allocation errors retain the original resource-plan-exceeded status.
The common header plan includes growing destination history and one temporary
writer; these coexist during append. These component bounds do not establish
complete-call admission for a future encoder.

### Streams and sections

The global stream contains metadata and the leading nonmetadata channels whose
width and height are both at most 256. Scanning stops at the first larger
channel; a later small channel is still grouped. Channels outside that prefix
use DC groups when `min(hshift, vshift) >= 3`, otherwise ordinary groups for the
single pass. Group rectangles are shifted before clipping to each channel;
clipping the source rectangle first would lose odd shifted edge samples.

With `D` DC groups, canonical IDs are global `0`, Modular DC `1 + D + group`,
and Modular group `1 + 3*D + 17 + group`. The intervening IDs reserve the
VarDCT DC, AC-metadata, and quantization-table roles required by the format.
IDs are independent of section positions. Stream and channel IDs are checked
against the signed 32-bit static-property range before plan allocation.

For multiple groups, section order is DC global, DC groups, AC global, then
ordinary groups. AC global has a section even though it has no Modular stream.
For one group and one pass, every contribution maps to section zero. Empty
group streams retain their identities and have no channel slices; they must
not cause duplicate global-channel tokenization in P2.1.

Descriptors admit shifts 0 through 30. A grouped channel's shifts must fit its
group dimension (at most 8 for ordinary groups, 11 for DC groups); otherwise the
planner reports unsupported instead of silently producing empty rectangles.
Nonmetadata extents must fit the source extent after ceiling division by their
shifts. Future transforms may extend these capabilities with their own tests.

### Header capabilities

The common writer supports the existing float32 linear-sRGB/XYB VarDCT metadata
and integer gray/RGB/unassociated-RGBA at 8 or 16 bits. Integer metadata uses
source-space sRGB or its grayscale equivalent; alpha has the same bit depth,
no dimension shift, and no premultiplication. Modular frame headers signal one
regular final frame, one pass, 256-pixel groups, no upsampling, no XYB, and no
Gaborish or EPF. Other metadata combinations are explicitly rejected.

The writer permits 32-bit Modular buffers at either source depth. This conservative
declaration corresponds to level 10 in the pinned encoder's level checks; no
level-5 compatibility is claimed. A narrower declaration requires a range proof
for every enabled transform and stream, independently of the native plane type.
The writer retains
the precise legacy VarDCT field ordering through compatibility adapters. Image
headers are padded relative to their own beginning before append; frame headers
remain bit-aligned for subsequent TOC writing. Tests cover all eight destination
bit offsets, invalid metadata, allocation failures and enclosing allotments.

## Independent checks

The reference is clean libjxl
`e8ff09762481785938d8e4e01333ed3917571161`. The test-only adapter:

- Parses native image/frame headers and synthetic TOCs with actual pinned
  readers, checking color, depth, alpha, frame mode, filters, groups and sections.
- Compares canonical IDs and shifted/clipped rectangles with pinned utilities.
- Generates lossless reference fixtures, extracts raw codestreams when the
  reference encoder returns containers, and decodes with the public decoder API
  into `uint8_t` or `uint16_t` output. It requests source-space output and keeps
  unassociated alpha and invisible colors. Unsupported metadata and truncated
  input are rejected. This fixture extraction is not production container support.
- Exercises actual weighted prediction/state updates, one-leaf and split
  prescribed-tree tokenization, native tree emission parsed by the reference,
  and forced RCT forward/inverse arithmetic. RCT remains reference-only.

The 24 integer round trips cover gray/RGB/RGBA, both depths, and extents 1x1,
17x19, 257x3, and 2049x1. Samples include high-byte differences, extrema and
opaque/partial/zero alpha with nonzero RGB. Full native Modular files will use
the same decoder helper in P2.1. Reference-generated round trips do not claim
that a native whole-image encoder already exists.

## Reproduction and qualification

Use an audited compiler from [the storage toolchain contract](storage-toolchain.md).
The local Windows environment is the same as Phase 1: MSVC 19.37.32826/STL
202305, Release x64/C++20, Windows SDK 10.0.26100.0, on a Windows ARM64 host
running the x64 binaries under emulation. The pinned library uses x64 clang-cl.
Keep the storage compatibility guard enabled.

Build the pinned reference using the existing
[Phase 1 reference instructions](../tools/modular_extraction/README.md#independent-tests),
then configure a native build with the new optional path:

```powershell
cmake -S . -B build/modular-p2.0/check -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 -DGJXL_ENABLE_METAL=OFF -DGJXL_ENABLE_CUDA=OFF "-DGJXL_MODULAR_ORACLE_BUILD=$pwd/build/modular-phase1/pinned-libjxl-x64"
cmake --build build/modular-p2.0/check --target gjxl_modular_foundation_test gjxl_modular_oracle_test --parallel 6
ctest --test-dir build/modular-p2.0/check -R '^modular_(foundation|oracle)$' --output-on-failure
```

Leave `GJXL_MODULAR_ORACLE_BUILD` empty to omit the reference targets. The native
foundation test remains available without libjxl. The oracle configuration checks
that the source associated with the supplied build is clean and at the pinned
revision, following the existing direct-DC-oracle convention.

### Recorded results

The changes are based on `6bdfbb0`. The machine-readable
[qualification record](modular-p2.0-results.json) records source and binary
identities, commands, test outcomes and evidence hashes. Raw local logs and
captures are under `build/modular-p2.0/`; the broad suite reused the audited
`build/modular-phase1/candidate` build without replacing Phase 1's retained logs
or committed evidence.

| Check | Result |
| --- | --- |
| Native foundation and independent Modular oracle tests | Both pass |
| Integer reference-generated round trips | 24/24 exact, including transparent RGB and 16-bit samples |
| Native headers parsed independently | 240 metadata/frame cases across formats, extents and bit offsets |
| VarDCT artifact parity | 204 original/stage artifacts plus four photographic artifacts match the frozen baseline exactly |
| Pinned VarDCT decoder conformance | 23/23 fixtures pass |
| Complete native build with tests/reference/Metal/CUDA disabled | Pass |
| Installed C11 and C++20/C++23 consumers, header compilation and relocation | Pass |
| Reference pin rejection and installed dependency audit | Pass; no reference libraries or new private headers in the installed interface |
| Effective CTest result after the isolated retry | 117/126 pass; every Phase 1 candidate pass is retained |

The first broad run passed 116/126. `cpu_workflow_legacy_dc_tree` exceeded its
180-second timeout; an isolated retry with a 600-second limit passed in 270.50
seconds. Both records are retained. Test duration is not a performance gate or a
speedup claim.

The remaining nine failures/not-run tests are unchanged from Phase 1: the two
Windows contract-test build failures, five Butteraugli reference/differential
targets, generated Butteraugli goldens, and the direct DC-processing oracle.
The latter still reports the same baseline-reproduced mismatch at 255x256,
channel 2, x=8, y=2, pattern 3: reference -5850, native -5852. See the
[Phase 1 failure accounting](modular-phase1/README.md#tests-and-retained-failures).
These failures are not omitted from the totals or treated as successful checks.

GPU execution, native ARM64, other platforms and sanitizers require separate
qualification. No compression or speedup claim is made by this milestone.

To repeat the complete regression check, build all native tests with the same
reference configuration, retain any known build failures, then run CTest with a
600-second timeout on this host. Compare artifacts using the existing
`tools/modular_extraction` capture/evidence tools against the retained
`baseline-stages` (204 artifacts) and photographic baseline (four artifacts).
Use separate capture directories; never update the expected Phase 1 bytes.
