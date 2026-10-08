# P2.2: integer formats and alpha preservation

This milestone extends the private scalar Modular workflow from P2.1 to grayscale,
RGB and unassociated RGBA, each at 8 or 16 unsigned bits. Public C++/C/Rust APIs
and CLI integration remain P2.5. The architecture and later milestones are in
[the Modular plan](modular-architecture.md#p22-preserve-more-source-formats-and-extra-channel-semantics).

## Input contract

`PackedModularImageView` carries a bounded byte span, extent, row stride in bytes,
`PackedModularFormat`, and `SampleByteOrder`. The six formats are `kGray8`,
`kRgb8`, `kRgba8`, `kGray16`, `kRgb16`, and `kRgba16`. Grayscale uses gray sRGB;
RGB/RGBA use sRGB sample space. No linearization or normalization occurs.

Byte order is explicitly little-endian or big-endian, defaulting to little-endian.
Both orders are accepted for 8-bit input and have no effect on samples. Invalid
format/order values are rejected. Sixteen-bit loads combine bytes directly:
unaligned offsets and odd byte strides are valid, independently of host byte order.
There is no native-endian mode.

Offsets use a subspan. Stride must cover the complete packed row; backing need
only reach the final sample, without final-row padding. Validation checks geometry,
row arithmetic, final addressed length and pointer-difference limits before
admission or allocation. Existing dimension, pixel-count, stream-ID, TOC and
managed storage limits continue to apply. Errors leave published output unchanged.

RGBA always means unassociated alpha at the color depth, with zero dimension shift.
Alpha is the fourth plane and has `ChannelRole::kAlpha`; the first three planes
are color. RGB values at zero alpha remain intact. There is no premultiplication,
unpremultiplication or invisible-pixel optimization. Associated alpha, mixed alpha
depths, gray-plus-alpha, shifted channels and arbitrary extra-channel types are
not input formats in this milestone.

## Architecture and numeric bounds

`ResolveModularInput` resolves metadata, channel descriptors and sample width
without allocating. Source validation, frame preparation and workflow storage
planning share that description. `ModularEncoderFrame::Prepare` copies source
integers into immutable completed identity frames with one, three or four signed
32-bit planes. `TokenizeIdentity` retains P2.1's canonical stream traversal and
channel-local gradient predictor resets, with one context and one global tree leaf.

`EncodeModularImageOwned` publishes managed result backing; `EncodeModularImage`
publishes an ordinary vector. Both use the existing encoding options, admission,
execution-domain and atomic-publication contracts. RGB8 entry points remain thin
forwarding adapters. Frame serialization, prefix/ANS machinery, section layout,
common metadata writers and VarDCT's opaque-alpha validation are unchanged.
No private interface is installed, and production code does not link libjxl.

For 16-bit input, samples and clamped predictions are in `[0,65535]`.
`left + top - top_left` is in `[-65535,131070]`; prediction and residual arithmetic
use `int64_t`. Residuals are in `[-65535,65535]`, and signed-packed values are at
most 131070. Plane storage and token conversion therefore preserve the full range.
The signed-16-buffer-sufficient declaration is false for 16-bit input. The existing
8-bit identity/gradient proof applies to grayscale and alpha as well as RGB, so
all 8-bit formats set that declaration.

Storage planning now scales plane records, samples, total tokens and maximum
stream payloads by the resolved channel count. The shared full-uint32 token
emission bound already covers 17-bit packed residuals. Byte order changes neither
planning nor encoded output. Preparation, modeling, emission, assembly and
publication retain the original phase lifetimes; existing owned output remains
charged separately during replacement.

## Qualification

The environment remains Release x64 under emulation on Windows ARM64, using the
audited MSVC 19.37.32826/STL 202305 toolchain and Windows SDK 10.0.26100.0.
The independent libjxl oracle is pinned to
`e8ff09762481785938d8e4e01333ed3917571161`, built with x64 clang-cl.
Exact commands, hashes and test outcomes are in
[the machine-readable record](modular-p2.2-results.json).

The expanded native conformance harness compares independently generated reference
tokens and decodes complete native files back into source-space integers. Expected
channel counts and depths are specified independently of native format resolution.
It checks color/alpha metadata, dimension shift, buffer declarations, TOC placement,
exact reconstruction and rejection of truncated files. Equivalent LE/BE inputs
must emit identical bytes. Each format retains P2.1's eight patterns and 19 small,
narrow, odd and boundary extents, plus a random 2049x2049 image. Additional fixtures
exercise numeric transitions, a complete 16-bit value sweep, high-byte-only changes,
and transparent colored pixels with mixed alpha. Inputs have nonzero offsets,
nontrivial strides and no final-row padding; 16-bit rows are deliberately unaligned.

| Check | Result |
| --- | --- |
| Native complete-file integer comparisons | 2,802 exact; all truncated copies rejected |
| Independent prescribed-gradient token cases | 1,401 exact |
| Frozen P2.1 RGB8 codestreams | 306 unchanged |
| Managed allocation failure injection | 11,292 positions across 26 cases |
| Frozen VarDCT artifacts | 208 unchanged |
| Pinned VarDCT decoder conformance | 23/23 pass |
| Reference-disabled native build and installed workflow consumer | Pass |
| Installed C11 and C++20/C++23 consumers, headers and relocation | Pass |
| Complete CTest run | 119/128 pass; all P2.1 passes retained |
| Native Modular include closure | 47 files; no VarDCT-policy, GPU or libjxl headers |

The workflow tests additionally check invalid enums, dimensions, short buffers,
insufficient/overflowing strides, allocation-free planning, channel-dependent token
counts, failure atomicity, retained-output replacement, cross-thread destruction,
recovery, source immutability and execution-domain behavior. Format-specific fault
sweeps cover both global and multi-group streams with both entropy coders.

The complete CTest run used four parallel tests and a 600-second per-test timeout,
finishing in 299.02 seconds. The nine retained failures/not-run tests match P2.1:
two Windows contract-test build failures, five Butteraugli reference/differential
targets, missing generated Butteraugli goldens, and the direct DC-processing oracle.
The latter still reports the previously reproduced 255x256 mismatch at channel 2,
x=8, y=2, pattern 3: reference -5850 versus native -5852. The broad build retains
the same eight failed targets. These are unresolved baseline failures, not passes;
see [the earlier accounting](modular-phase1/README.md#tests-and-retained-failures).

Raw evidence is under `build/modular-p2.2/`. Conformance artifacts are grouped by
format and byte order beneath the candidate build's `modular-conformance` directory;
frozen RGB8 hashes are compared without updating their expected values. The
reference-disabled workflow consumer links installed static libraries using the
matching private source headers; its dependencies are Windows/MSVC runtimes only.

To reproduce the focused qualification with the existing pinned reference build:

```powershell
. ./build/modular-phase1/msvc-env.ps1
cmake --build build/modular-phase1/candidate --target gjxl_modular_workflow_test gjxl_modular_conformance_test gjxl_modular_foundation_test gjxl_modular_oracle_test --parallel 4
ctest --test-dir build/modular-phase1/candidate -R '^modular_(workflow|conformance|foundation|oracle)$' --output-on-failure --timeout 600
```

The machine-readable record includes the complete reference-disabled build/install,
installed-consumer, artifact-capture, and full regression commands. Keep the pinned
expected hashes unchanged when repeating the comparisons.

GPU execution, native ARM64, other platforms and sanitizers remain unqualified.
This milestone makes no compression-ratio, speed or unrestricted level-5 claim.
