# P2.1: native RGB8 Modular CPU encoding

This milestone implements the first complete native workflow from
[the architecture plan](modular-architecture.md#p21-the-smallest-independently-decodable-cpu-encoder),
based on `952ab2f`. The encoder is private and callable by native tests;
public C++/C/Rust and CLI integration remains P2.5.

## Qualified encoding contract

Input is a bounded span of packed RGB8 bytes in sRGB sample space, with explicit
source extent and byte row stride. Offsets use a subspan; trailing row padding
is allowed, and only the last addressed pixel must be present. Validation checks
nonempty dimensions, stride, backing length, pointer range and arithmetic before
managed allocation. The original bytes remain unchanged on success and failure.

The workflow emits one regular final Modular frame in a raw codestream, with
256-pixel groups, one pass, identity transforms and a single global gradient leaf
(offset zero, multiplier one, context zero). There is no XYB conversion,
linearization, quantization, resampling, Gaborish, EPF or LZ77. Gray, RGBA and
16-bit input remain P2.2. No new interface is installed.

`EncodeRgb8ModularOwned` retains managed result backing; `EncodeRgb8Modular`
publishes an ordinary byte vector after all fallible work succeeds. Private
`ModularEncodingOptions` accepts an execution domain and prefix or ANS entropy
coding. Prefix is the default. ANS builds from the prefix partition using the
existing shared optimizer; there is no complete-file coder competition or effort
search. Null domains select the shared default domain. Both routes admit one
CPU participant and launch no workers.

Input dimensions retain the common header/geometry limits, the format limit of
2^40 pixels, and signed 32-bit stream IDs. Checked storage and TOC bounds may
reject requests before allocation. The workflow is not an image streaming API;
it prepares full integer planes and retains all tokens for shared model creation.

### Numeric and buffer declaration proof

Prepared samples are in `[0,255]`. Stream-local first pixels predict zero, the
first row uses its left neighbor, and the first column uses its top neighbor.
Elsewhere `left + top - top_left` lies in `[-255,510]` and is clamped between
left and top. Predictions therefore remain in `[0,255]`, residuals are in
`[-255,255]`, and signed-packed residuals are at most 510. Arithmetic uses
`int64_t`; owned planes remain `int32_t`.

The private image metadata now carries an explicit buffer-sufficiency declaration.
This proven identity/gradient RGB8 path signals that signed 16-bit Modular sample
buffers suffice. Existing metadata producers default to the earlier conservative
32-bit permission. The field does not select native plane storage. Additional
transforms/predictors require their own proof before reusing the declaration.
The writer rejects the currently unqualified narrower declaration for 16-bit and
floating-point sources. No unrestricted level-5 compatibility claim is made:
large image dimensions/pixel counts have separate level restrictions.

## Layering and data flow

- `codec/modular/frame.*` owns bounded source validation and preparation into an
  immutable completed identity frame. `gradient.h` contains predictor arithmetic
  with no entropy or writer dependencies.
- `codestream/modular/tokenization.*` scans P2.0 channel slices into one exact
  managed token array and borrowing stream views. Coordinates and prediction
  reset per channel and stream. The canonical planner supplies channel ordering
  and IDs; callers must not fabricate or edit the plan passed to the tokenizer.
- `codestream/modular/frame_encoder.*` builds one shared model, emits streams,
  assembles sections and copies the final writer into managed publication backing.
  The internal frame handoff requires canonical plan/token provenance from the
  workflow; it is not a validator for externally supplied frames or trees.
- `codestream/modular/storage_plan.*` composes checked allocation-free bounds;
  `workflow.*` performs admission, lifetime control and atomic publication.

The common `WorkflowAdmission` class is separated into a private header without
VarDCT planning or GPU header dependencies. Existing VarDCT callers retain their
compatibility header, planner, domain checks and cache-trimming behavior. The
Modular implementation does not import VarDCT policy or production libjxl code.

### Frame syntax and the single-context entropy correction

DC-global starts with the default DC-matrix bit, followed by the shared tree,
shared sample model, and the global Modular stream header. Images no larger than
one group place their samples globally. Larger images still emit the global
header, but no global sample payload or ANS state. Empty group streams emit no
header or payload. Empty DC-group and AC-global sections remain present in the
TOC. Single-group frames combine contributions in section zero without inserting
padding between them; byte padding belongs only to section boundaries.

Independent complete-file decoding exposed a preexisting shared entropy writer
bug: prefix and ANS models with exactly one context emitted an explicit three-bit
`{0}` context map. Pinned `DecodeHistograms` skips that field for one context.
Both model writers now omit it. Standalone context-map writing and the internal
validation used to serialize arrays of prefix codes retain their existing meaning.
The sparse ANS unit fixture consequently changes from 59 to 56 model bits;
its histogram and token expectations remain unchanged. Frozen VarDCT artifacts
are compared separately, without updating their expected bytes.

## Resource lifetime contract

Planning depends only on geometry and the chosen entropy mode. Bounds describe
fresh owners, including their allocation/growth history and replacement peaks;
they do not estimate RSS or compression ratio. The complete reservation is the
maximum of these explicitly composed phases:

| Phase | Coexisting managed backing |
| --- | --- |
| Preparation | Three integer planes and descriptors, stream plan, token array and view table |
| Model preparation | Stream plan, tokens/views, prefix optimization; for ANS, retained prefix partition plus ANS optimization |
| Section emission | Stream plan, tokens/views, shared model, section records/writers, tree/model/token writer scratch |
| Assembly | Stream plan, retained sections, header temporary, TOC size array and final writer |
| Publication | Stream plan, final writer and fresh exact result backing |

Planes are released after tokenization. Tokens and models are released before
assembly; sections are destroyed before publication backing is allocated. Bounds
include the maximum bit-writer allotments, not only actual emitted bytes. Edge
groups use conservative full-group payload bounds. Existing owned results stay
charged separately during replacement and are never counted as fresh allocations
inside the new reservation. Caller-owned input and previously published vectors
remain outside the managed-memory boundary.

Memory admission precedes preparation and CPU participation. The CPU scope lasts
through publication. Managed failures preserve their original status, including
the typed resource-plan-exceeded reason. Partial work is destroyed before return;
previous output is preserved. Owned results retain charges beyond the producing
reservation and may be destroyed on another thread. Nested calls reuse the
outer reservation/participant only under the existing domain identity rules.

## Qualification and reproduction

The environment is the audited P2.0 Windows configuration: MSVC 19.37.32826,
STL 202305, Release x64/C++20, Windows SDK 10.0.26100.0, running under x64
emulation on an ARM64 host. Pinned libjxl is
`e8ff09762481785938d8e4e01333ed3917571161`, built separately with x64 clang-cl.
The storage-toolchain guard remains enabled.

New native workflow tests run without libjxl. Optional `modular_conformance`
uses the existing pinned reference target and retains generated `.jxl` files in
the test build's `modular-conformance` directory. Each image gets an independent
prescribed-tree token comparison and two complete-file checks, one per coder.
Reference tiles are constructed from source pixels without native rectangle or
view helpers. Pinned decoding requests source-space integer output.

Fixtures include zero, maximum, a nonzero constant, ramps, impulses, checkerboards,
deterministic random values and independent color channels; 1x1, rows/columns,
odd sizes, 255/256/257 and 2047/2048/2049 boundaries on both axes, and a random
2049x2049 image. Every native file checks header fields, section placement, exact
samples and truncation rejection. Existing P2.0 tests continue to exercise wider
metadata and independent tree/predictor contracts.

The workflow tests cover allocation-free planning, invalid input, overflow,
unsupported mode, too-small admission, terminal forced underplanning, every
managed backing failure position on representative single/multi-group inputs,
owned/public equality, replacement, cross-thread destruction, deterministic
recovery and unchanged source bytes. Shared-domain tests cover simultaneous calls,
nested reservation/CPU reuse and rejection of a different explicit nested domain.

Results and exact source/evidence hashes are recorded in
[the machine-readable qualification record](modular-p2.1-results.json).

| Check | Result |
| --- | --- |
| Native prescribed-tree token comparisons | 153/153 exact |
| Native RGB8 files decoded with pinned libjxl | 306/306 exact across prefix and ANS; all truncated copies rejected |
| Independent metadata/header checks | 248 cases, including both buffer declarations |
| Existing reference-generated integer round trips | 24/24 exact |
| Managed allocation failure injection | 2,537 positions across six single/multi-group and entropy cases |
| Frozen VarDCT artifacts | 204 primary/stage plus four photographic artifacts unchanged |
| Existing pinned VarDCT decoder conformance | 23/23 fixtures pass |
| Complete reference-disabled native build | Pass |
| Workflow test rebuilt against reference-disabled installed libraries | Pass; Windows/MSVC runtime dependencies only |
| Installed C11 and C++20/C++23 consumers, headers and relocation | Pass |
| Complete CTest run | 119/128 pass; all 117 P2.0 passes retained |
| Native Modular include closure | 47 files; no VarDCT-policy, GPU or libjxl headers |

The complete run used a 600-second per-test timeout and needed no isolated retry.
The nine remaining failures/not-run tests are the same as P2.0: two Windows
contract-test build failures, five Butteraugli reference/differential targets,
generated Butteraugli goldens, and the direct DC-processing oracle. The broad
build retains eight known failed targets. The DC oracle still reports the
baseline-reproduced 255x256 mismatch at channel 2, x=8, y=2, pattern 3:
reference -5850 versus native -5852. These are retained failures, not passing
checks. See [the earlier failure accounting](modular-phase1/README.md#tests-and-retained-failures).

Raw evidence lives under `build/modular-p2.1/`. The reference-disabled workflow
consumer uses matching private source headers to exercise the installed static
libraries; it does not install or introduce a public Modular entry point. Its
small CMake harness is preserved in the qualification record. The native fixture
hashes and the committed Phase 1 expected hashes are recorded individually.

Build the optional pinned targets with the existing Phase 1 reference build:

```powershell
. ./build/modular-phase1/msvc-env.ps1
cmake -S . -B build/modular-phase1/candidate "-DGJXL_MODULAR_ORACLE_BUILD=$pwd/build/modular-phase1/pinned-libjxl-x64"
cmake --build build/modular-phase1/candidate --target gjxl_modular_workflow_test gjxl_modular_conformance_test gjxl_modular_foundation_test gjxl_modular_oracle_test --parallel 6
ctest --test-dir build/modular-phase1/candidate -R '^modular_(workflow|conformance|foundation|oracle)$' --output-on-failure --timeout 600
```

For the complete regression run, retain known build failures with Ninja `-k 0`,
then run the full CTest suite with `--parallel 4 --timeout 600`. Use separate P2.1
capture/evidence directories for the frozen VarDCT comparisons. The manifest lists
the reference-disabled build/install and capture commands too.

GPU execution, native ARM64, other platforms and sanitizers remain unqualified.
This milestone makes no compression-ratio or performance claim.
