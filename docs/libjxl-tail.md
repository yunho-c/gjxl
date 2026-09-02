# Libjxl VarDCT tail experiment plan

This document plans an opt-in encoder experiment that keeps GJXL's completed
VarDCT frontend decisions but replaces GJXL's native entropy and codestream
tail with the corresponding pinned libjxl implementation. The purpose is to
measure how much end-to-end time remains when GJXL's CPU or Metal frontend is
paired with a mature alternative serializer.

The experiment is not a claim that libjxl is necessarily faster. It is a
controlled way to answer that question on the same completed frame. The first
implementation is benchmark-only and is disabled by default; promotion to a
supported encoder option is a later decision based on correctness, performance,
maintenance cost, and output-size results.

[`codestream.md`](codestream.md) remains authoritative for GJXL's supported
bitstream profile. [`metal-encoding-performance.md`](metal-encoding-performance.md)
remains authoritative for the public-workflow timing boundary and benchmark
protocol. This document owns the hybrid-backend experiment and the additional
validation required to make its result interpretable.

## Implementation status

As of 2026-09-01, Milestones 1 and 2 are implemented through the controlled
bridge seam:

- the opt-in build verifies the pinned base
  `e8ff09762481785938d8e4e01333ed3917571161` and prints the applied libjxl
  patch revision;
- libjxl patch `30aefc190f790282327999633d43ae3e63aab5e6` adds the private
  `EncodePrecomputedVarDctFrame` declaration and skeletal implementation;
- default builds select an unavailable adapter stub and do not configure or
  link the full libjxl encoder;
- enabled builds configure static `jxl-internal` and `jxl_threads`, reach the
  libjxl bridge from a completed GJXL frame, and receive the expected
  controlled `Unsupported` result; and
- focused tests verify default-build unavailability, enabled-build bridge
  reachability, option validation, and caller-output atomicity;
- libjxl patch `61f8f8d1751397765547668574e71afea37913e2` constructs ordinary
  `CodecMetadata`, `FrameHeader`, `PassesEncoderState`, quantizer, strategy,
  control-field, qDC, DC, and AC coefficient state from the GJXL handoff;
- the qDC path copies authoritative X/Y/B integers into libjxl's Y/X/B
  modular channel order and uses libjxl's ordinary `DequantDC` path without a
  float requantization; and
- field-specific pre/post-copy digests match for both a small fixture and a
  mixed-strategy 2057x257 fixture spanning partial AC groups and two DC groups.

The Milestone 0 results captured so far are provisional. They cover a small
fixture, pinned-decoder conformance, and initial padded-1080p profiles, but not
yet the complete independent-process, 4K, real-photograph, target-1.0/1.2
matrix required by the milestone. Milestone 3 codestream production is not yet
claimed: the enabled bridge currently validates and copies the completed frame,
then returns the controlled `Unsupported` result before entropy coding.

## Questions the experiment must answer

The work has three separate outputs. They must not be collapsed into one
speedup number.

1. **Perfect-tail Amdahl bound:** how fast would the current GJXL workflow be
   if its entire measured codestream phase took zero time?
2. **Same-frame tail comparison:** how do the GJXL and libjxl tails compare
   when both serialize the exact same `VarDctEncoderFrame`?
3. **Hybrid public workflow:** how fast is the complete GJXL frontend plus the
   libjxl tail, including state conversion and every per-frame allocation or
   copy required by the bridge?

For each timed sample, define:

```text
T_frontend       = T_gjxl_total - T_gjxl_tail
T_hybrid_tail    = T_adapter + T_libjxl_tail

perfect-tail bound = T_gjxl_total / T_frontend
measured hybrid    = T_gjxl_total / (T_frontend + T_hybrid_tail)
```

The libjxl result is an attainable counterfactual, not the mathematical
maximum. The zero-time tail is the Amdahl bound. Both values belong in the
final report.

## Scope

The primary experiment replaces the complete serialization tail after a
validated `VarDctEncoderFrame` exists:

- VarDCT DC and AC-metadata modular streams;
- libjxl block-context model selection;
- coefficient-order selection;
- AC coefficient tokenization;
- histogram construction, clustering, and entropy coding;
- DC, global AC, DC-group, and AC-group section writing;
- frame header, table of contents, and final raw codestream assembly.

The experiment retains GJXL's:

- input preparation and linear-RGB-to-XYB conversion;
- AC-strategy search;
- adaptive quantization and Butteraugli policy;
- DCT, quantization, coefficient adjustment, CfL, and EPF decisions;
- authoritative quantized DC and AC coefficients; and
- selected CPU or Metal frontend mode.

An entropy-primitives-only adapter may be useful as a later diagnostic, but it
is not the primary deliverable. Replacing only model construction or rANS while
retaining GJXL tokenization, candidate search, headers, and section assembly
does not answer the complete-tail question.

### Initial profile

The first bridge supports only the profile already accepted by
`ValidateSimpleCodestreamFrame`:

- one regular final VarDCT frame;
- three XYB color channels, 4:4:4, and no extra channels;
- one pass and no upsampling;
- default quantization matrices with the frame's encoded X/B scales;
- `extra_dc_precision = 0` and no adaptive DC smoothing;
- default loop-filter signaling;
- no modular transforms or LZ77; and
- GJXL's current seven AC strategies.

Unsupported state must return an error before output bytes are committed. The
bridge must not silently change the profile to one libjxl happens to accept.

### Non-goals for the first experiment

- No stable public C or C++ API.
- No runtime dependency on a system libjxl.
- No fallback from a requested libjxl tail to the GJXL tail.
- No JPEG recompression, lossless Modular mode, progressive passes, extra
  channels, animation, containers, or streaming encoding.
- No claim that codestreams from the two tails should be byte-identical.
- No target-byte or target-BPP measurements until the fixed-target experiment
  is complete.
- No entropy or codestream optimization performed as part of the bridge work.

## Current handoff and intended architecture

`VarDctEncoderFrame` is already the correct semantic handoff. It owns the
geometry, strategies, raw quantization, quantizer, CfL, EPF, quantized DC,
decoder-equivalent DC, profile, and fixed-capacity AC-group rows required by a
serializer. GJXL's workflow currently calls `EncodeVarDctCodestream` only after
the CPU or Metal quantization pipeline has completed.

The implementation should preserve that boundary:

```text
linear RGB
    |
    v
GJXL CPU or Metal frontend
    |
    v
authoritative VarDctEncoderFrame
    |
    +--------------------------+
    |                          |
    v                          v
GJXL native tail       GJXL-to-libjxl state adapter
                               |
                               v
                       libjxl prepared-frame tail
                               |
    +--------------------------+
    |
    v
raw JPEG XL codestream bytes
```

The workflow switch belongs at the current serializer call in
[`workflow.cpp`](../src/codestream/workflow.cpp), not inside the quantization
pipeline. Both tail implementations consume a `const VarDctEncoderFrame&` and
commit output only after successful completion.

## Libjxl integration strategy

Libjxl's public `EncodeFrame` entry point starts from pixels or an
`ImageBundle`; using it would repeat color conversion, heuristics, AQ, DCT, and
quantization. The required post-coefficient functions in `enc_frame.cc` are in
an anonymous namespace and are not linkable from a normal GJXL translation
unit. A narrow pinned-libjxl internal bridge is therefore required.

For the experiment, carry one reviewable libjxl patch commit based on the
documented upstream pin. The submodule patch should:

- add an internal, non-installed header such as
  `lib/jxl/enc_precomputed_vardct.h`;
- define a GJXL-independent view of precomputed VarDCT state;
- add `EncodePrecomputedVarDCTFrame` in `enc_frame.cc`, where it can reuse the
  existing private helper sequence;
- add a direct quantized-DC modular-stream entry point instead of converting
  qDC to float and rounding it back to integers; and
- expose optional phase timings and state digests only through the internal
  bridge contract.

The bridge must not include GJXL headers or know about GJXL types. The root
repository owns the adapter from `VarDctEncoderFrame` to the libjxl view. The
libjxl patch remains small enough to rebase or discard independently.

If retaining an upstream-exact submodule gitlink becomes a hard requirement,
the same patch may instead be applied to a private copy in the build tree.
That packaging alternative must not change the bridge semantics, and the
source revision plus patch hash must be recorded in benchmark artifacts. Do
not include `enc_frame.cc` directly from a GJXL translation unit.

### Build configuration

Add an option named `GJXL_ENABLE_LIBJXL_TAIL_EXPERIMENT`, defaulting to `OFF`.
When disabled:

- installed libraries and headers are unchanged;
- GJXL does not link the full libjxl encoder; and
- explicitly requesting the libjxl tail returns `Status::Unavailable`.

When enabled:

- configure the pinned libjxl tree as static and `EXCLUDE_FROM_ALL`;
- disable libjxl tests, tools, examples, JNI, docs, JPEG transcoding, viewers,
  TCMalloc, and unrelated optional codecs;
- link a private `gjxl_libjxl_tail` adapter target to `jxl-internal` and the
  libjxl thread runner;
- do not install libjxl targets or expose their include directories through a
  public GJXL target; and
- arrange the existing Butteraugli-reference option to reuse the already
  configured Highway/libjxl targets instead of adding Highway twice.

The build must verify and print the libjxl base revision and patch revision.
The benchmark JSON must record both.

## Internal configuration contract

Do not add the experimental selector to the public `VarDctEncodingOptions` in
the first patch. Add an internal contract along these lines:

```cpp
enum class VarDctCodestreamBackend {
  kGjxl,
  kLibjxl,
};

struct VarDctCodestreamBackendOptions {
  VarDctCodestreamBackend backend = VarDctCodestreamBackend::kGjxl;
  int libjxl_effort = 7;
};
```

The libjxl effort is explicit and belongs to the tail, not the GJXL frontend.
Map efforts `1...10` exactly as libjxl does, to
`SpeedTier(10 - effort)`. Do not silently derive libjxl effort from GJXL's
frontend effort. Record both values.

Add one internal serializer dispatch that accepts a completed frame, backend
options, an optional reusable libjxl context, output bytes, and a backend-aware
profile. Existing production entry points pass the default GJXL backend. A new
testing/benchmark entry point may select libjxl when the build option is on.

The serializer must have these behavioral guarantees for both backends:

- validate before emitting caller-visible bytes;
- leave output and profile unchanged on failure;
- never fall back between backends;
- report the backend actually used; and
- preserve deterministic output for fixed input, options, thread count, and
  pinned revisions.

## State mapping

The adapter should copy or construct only serialization-critical state. It
must not invoke libjxl frontend heuristics that can replace GJXL decisions.

| GJXL source | Libjxl destination | Required handling |
| --- | --- | --- |
| `frame.geometry()` | `FrameHeader` and `FrameDimensions` | Preserve original and padded dimensions and initial-profile group sizing. |
| `frame.profile()` | `CodecMetadata` and `FrameHeader` | Signal linear sRGB source metadata, XYB, one pass, 4:4:4, X/B matrix scales, and identical loop filters. |
| `frame.strategies()` | `PassesSharedState::ac_strategy` | Copy every raw strategy and anchor/covered-block relationship; reject unsupported values. |
| `frame.quantizer()` | `DequantMatrices` and libjxl `Quantizer` | Construct from the exact serialized `global_scale` and `quant_dc`; compute the default matrices before use. |
| `frame.raw_quant_field()` | `PassesSharedState::raw_quant_field` | Copy the complete padded block grid exactly. |
| `frame.epf_sharpness()` | `PassesSharedState::epf_sharpness` | Copy the block grid exactly. |
| `frame.color_correlation()` | `PassesSharedState::cmap` | Preserve base factors plus signed X/B tile maps. |
| `frame.quantized_dc()` | qDC state and VarDCT DC modular streams | Consume integer qDC directly. Do not round-trip through float DC. |
| `frame.dc()` | `dc_storage` when required by libjxl helpers | Copy decoder-equivalent DC only; it must not become the source of qDC. |
| `frame.GetAcGroup()` | one `ACImageT<int32_t>` row per group | Copy all three 65,536-element rows or the used prefix plus a verified zero tail. |
| GJXL frontend distance | libjxl tail `CompressParams` | Use only where libjxl context heuristics require it; it must not trigger AQ or coefficient recomputation. |

After the copy, libjxl should own its block-context choice, coefficient orders,
tokenization, histogram construction, entropy representation, and section
layout. Those decisions are part of the tail being measured.

### Quantized DC is a hard correctness boundary

Libjxl's existing `ModularFrameEncoder::AddVarDCTDC` accepts floating-point DC
and computes qDC using `round`. Although GJXL retains decoder-equivalent DC and
uses matching quantizer equations, an unnecessary float round trip creates a
tie and platform-sensitivity risk.

Add an internal libjxl method that accepts the authoritative three-plane qDC
integers, remaps GJXL's X/Y/B planes to libjxl's VarDCT modular Y/X/B channel
order, populates `shared.quant_dc`, and derives `dc_storage` using libjxl's
ordinary dequantization path. The method must share predictor/tree selection
with `AddVarDCTDC` so the only change is the source representation.

The initial implementation does not pass its correctness gate if it relies on
re-quantizing `frame.dc()`.

## Profiling contract

GJXL's current `VarDctCodestreamProfile` fields describe the native tail and
must not be repurposed with approximate libjxl meanings. Add:

- a `codestream_backend` discriminator;
- a `LibjxlTailProfile` substructure; and
- backend-independent `total_nanoseconds` and output-size reporting.

The libjxl subprofile should separate at least:

- adapter validation and state copy;
- libjxl state/header initialization;
- qDC and AC-metadata modular preparation;
- block-context and coefficient-order computation;
- AC tokenization;
- histogram/model construction;
- group/section writing; and
- header, TOC, and output assembly.

These are elapsed phase times unless explicitly named as aggregate worker
time. Do not add overlapping phases to compute total latency. The complete
hybrid tail timer includes adapter work and every per-frame libjxl allocation.

Thread-runner creation and other reusable context initialization are reported
as `context_setup_nanoseconds`. Report two explicit boundaries:

- **complete-call:** includes context setup when the caller supplies no warm
  context; and
- **warm-context:** reuses an already-created thread runner and allocator
  context but still includes every frame-dependent copy and allocation.

The public-workflow result must state which boundary it uses. The primary
maximum-throughput experiment uses a warm context, matching the existing warm
Metal-backend convention, while also reporting the complete-call result.

Increment the encoding benchmark's raw-sample schema when these fields are
added. Update its CLI regression test and the performance documentation in the
same change.

## Threading contract

Libjxl must receive an explicit thread count for retained comparisons. Do not
compare GJXL automatic scheduling with an unspecified libjxl default.

Use these primary settings:

- one active CPU participant, with libjxl running on the calling thread; and
- eight active CPU workers for the parallel-tail throughput comparison.

When the requested count is greater than one, create the libjxl runner once
outside the warm timed region. Record its worker count and whether the calling
thread participates or waits. Extend GJXL's participant instrumentation or add
a separate libjxl active-worker field before making equal-core claims; the
current thread-local GJXL tracker cannot observe libjxl-owned worker threads.

Do not allow a libjxl tail invoked inside an already-active GJXL parallel scope
to create another unbounded pool. The first bridge may reject that condition;
the supported solution is a shared runner or a bounded adapter, not silent
oversubscription.

## Implementation milestones

Estimates assume one engineer familiar with GJXL and the pinned libjxl source.
They are planning ranges, not commitments. The trustworthy experimental path
is expected to take roughly 7-12 focused days; public productization is
separate.

### 0. Freeze the measurement and revision contract

**Estimate:** 0.5-1 day

Deliverables:

- Record the GJXL revision, libjxl base revision, compiler, Release flags,
  machine, OS, thread counts, frontend mode, effort values, and corpus.
- Capture current native-tail codestream hashes, decoded outputs, byte sizes,
  and raw profiles for the selected corpus.
- Compute the perfect-tail Amdahl bound per raw sample rather than from ratios
  of unrelated medians.
- Choose the initial primary matrix: Butteraugli targets `1.0` and `1.2`, CPU
  and exact-coefficient Metal frontends, one and eight CPU participants, and
  padded 1080p/4K plus real photographic inputs.

Exit criteria:

- The baseline is reproducible from a saved command and contains raw samples,
  not only a summary table.
- Every retained native codestream passes the pinned decoder and decoded-pixel
  checks.

### 1. Build and call the pinned internal bridge

**Estimate:** 1-2 days

Root changes:

- Add `GJXL_ENABLE_LIBJXL_TAIL_EXPERIMENT` and private libjxl target setup to
  `CMakeLists.txt`.
- Add a GJXL-owned adapter target and an unavailable stub selected by the
  option.
- Keep normal builds and installed exports free of the full libjxl dependency.

Pinned-libjxl changes:

- Add the internal precomputed-frame view and bridge declaration.
- Add a skeletal implementation in `enc_frame.cc` that validates metadata and
  returns a minimal controlled error before doing state conversion.
- Make the bridge callable from a focused GJXL test without exporting it from
  libjxl's public API.

Exit criteria:

- Default GJXL configuration and focused tests remain unchanged.
- The enabled configuration builds `jxl-internal`, the thread runner, and the
  bridge from a clean tree.
- Disabled builds return `Status::Unavailable` for an explicit libjxl request.
- Enabled builds reach the bridge and preserve output atomicity on its expected
  controlled failure.

### 2. Implement and audit state conversion

**Estimate:** 2-3 days

Deliverables:

- Construct the exact `CodecMetadata`, `FrameHeader`, frame dimensions, one-pass
  splitter, default dequant matrices, and quantizer.
- Allocate and populate strategies, raw quantization, EPF, CfL, qDC/DC, and
  `ACImageT<int32_t>` rows.
- Implement the direct quantized-DC modular path.
- Run libjxl's AC-metadata construction on the copied state.
- Add a test-only digest covering dimensions, strategies, quantization fields,
  EPF, CfL, qDC, used AC counts, and all copied coefficient bytes.

Fixtures must cover:

- each of GJXL's seven AC strategies;
- negative, zero, and positive DC/AC coefficients;
- nontrivial CfL and EPF maps;
- non-default representable X/B matrix scales;
- partial edge groups and zero tails;
- multiple AC groups and multiple DC groups; and
- invalid or unsupported frame state.

Exit criteria:

- The post-copy libjxl digest matches the GJXL source digest field by field.
- qDC is integer-identical before modular prediction.
- No pixel-domain input, DCT, quantization, AQ, or AC-strategy heuristic is
  called by the bridge.
- Sanitizer builds report no out-of-bounds reads of edge-group coefficient
  tails.

### 3. Complete libjxl's post-coefficient tail

**Estimate:** 2-3 days

After state conversion, run the pinned libjxl sequence for:

1. block-context model selection for the requested libjxl effort;
2. coefficient-order computation;
3. coefficient tokenization;
4. DC/metadata modular tree and token preparation where required;
5. global AC histogram and entropy-model encoding;
6. DC, DC-group, and AC-group serialization;
7. frame-header and TOC writing; and
8. byte-aligned final assembly.

Refactor a private helper from libjxl's existing one-shot path if that avoids
duplicating the tail sequence. Both the normal libjxl encoder and the
precomputed bridge should call the same helper. Do not copy a second version of
the order/token/group logic into GJXL.

Exit criteria:

- Small, one-group, multi-group, and partial-edge fixtures produce valid raw
  JPEG XL codestreams.
- Repeated encodes have identical hashes for fixed options and thread count.
- One-thread and eight-worker encodes decode to the same pixels and, unless
  libjxl itself proves otherwise, produce identical codestream bytes.
- GJXL-tail and libjxl-tail codestreams from the same frame decode to bitwise
  identical pixel arrays using the same pinned decoder.
- Different entropy coding, context maps, orders, section sizes, and final raw
  bytes are accepted and reported rather than treated as a failure.

If decoded pixels differ, stop the performance work. First compare frame
headers, qDC, AC metadata, coefficient rows, and loop-filter signaling. Do not
introduce a numerical tolerance until the semantic mismatch is understood.

### 4. Add internal backend selection and profiles

**Estimate:** 1-2 days

Deliverables:

- Add the internal backend enum, options, and serializer dispatch.
- Select the dispatch at the existing post-pipeline call in `workflow.cpp`.
- Preserve the current public entry points by passing `kGjxl` internally.
- Add a benchmark/testing workflow entry point that accepts tail options and a
  reusable libjxl context.
- Add the backend-aware profile and raw-sample fields.
- Report selected backend and libjxl effort in text and JSON output.

Initially accept the libjxl tail only for Butteraugli-target and maximum-error
single attempts. Reject target-byte and target-BPP requests with a clear error;
those modes serialize multiple candidates and allow backend size to change the
selected frontend target.

Exit criteria:

- GJXL remains the default in every production entry point and CLI invocation.
- Explicit libjxl selection never silently falls back.
- Native GJXL outputs and profiles remain byte-for-byte/schema-compatible when
  the experimental option is off.
- Failure leaves codestream, summary, and profile outputs unchanged.

### 5. Add the same-frame benchmark

**Estimate:** 1-2 days

Add a `codestream-tail` scope to `gjxl_encoding_benchmark`. This scope must run
the selected frontend once, retain the resulting `VarDctEncoderFrame`, and
then serialize that object repeatedly through both tails.

The benchmark must:

- perform untimed correctness encodes first;
- warm both serializers and the libjxl context;
- alternate which tail runs first;
- emit every raw sample rather than only medians;
- separate adapter copy time from libjxl-internal tail time;
- report output bytes and SHA-256 for every backend;
- record libjxl effort and both thread policies;
- optionally retain codestreams and decoded outputs for audit; and
- reject a comparison if the completed frame changes between backends.

Add a deterministic frame fingerprint over all serialization-critical state.
Write that fingerprint into every same-frame result record. Equality of input
filenames or frontend options is not sufficient evidence that the tails saw
the same frame.

Exit criteria:

- CLI tests cover valid/invalid backend names, unavailable builds, effort
  bounds, raw JSON, and artifact paths.
- A saved same-frame run contains matching frame fingerprints, validated
  decoded pixels, size deltas, and per-tail timings.
- Tail-only timing excludes frontend work but includes all frame-dependent
  adapter work.

### 6. Run the end-to-end Amdahl experiment

**Estimate:** 1-2 days after correctness gates pass

Use fresh Release builds. For each retained configuration:

1. start at least three independent processes;
2. warm the selected frontend and both tails;
3. alternate GJXL/libjxl order for at least five paired samples per process;
4. keep CPU participant counts explicit;
5. validate one output from each process with pinned `djxl` and decoded-pixel
   comparison; and
6. retain raw JSON, build manifests, hashes, sizes, and commands.

Primary workloads:

- representative small and one-group fixtures, to expose fixed overhead;
- Flower or an equivalent medium photographic input;
- padded 1080p and padded 4K, for existing performance continuity; and
- at least two native high-resolution photographs, to avoid drawing a result
  only from synthetic or duplicate-heavy padded data.

Primary tail settings:

- libjxl effort 7 as the standard reference;
- a sweep of efforts 3, 5, 7, and 9 to show the speed/size envelope; and
- one and eight active CPU participants.

Report per workload and frontend mode:

| Result | Required statistic |
| --- | --- |
| GJXL frontend | median and raw wall time |
| GJXL native tail | median and raw wall time |
| libjxl state adapter | median and raw wall time |
| libjxl internal tail | median and raw wall time |
| Complete GJXL workflow | median and raw wall time |
| Complete hybrid workflow | median and raw wall time |
| Perfect-tail Amdahl bound | computed per sample, then summarized |
| Measured hybrid speedup | paired per sample/process |
| Encoded size delta | bytes and percentage |
| Decoded equivalence | explicit pass/fail |

Do not compare these numbers directly with an unrelated `cjxl` profile. A
whole libjxl encode includes a different frontend, while this experiment feeds
libjxl GJXL's completed coefficient state.

Exit criteria:

- The result distinguishes same-frame tail speedup, complete hybrid speedup,
  and the zero-tail bound.
- No conclusion relies on aggregate worker time as wall-clock latency.
- Speed, output size, thread count, effort, quality/decoded identity, and
  revision are visible together.
- Conclusions are limited to the measured corpus and hardware.

### 7. Decide whether to promote the backend

**Estimate:** decision gate; 1-3 additional weeks if promoted

The experiment may remain an internal research tool. Promote it to a supported
option only if all of these are true:

- the hybrid produces a material repeatable end-to-end gain;
- size regressions are acceptable or controllable with a documented effort;
- decoded equivalence is exact for the supported profile;
- the full libjxl build/link cost is acceptable to downstream users;
- the pinned bridge patch has a maintainable ownership and update strategy;
- cross-platform CI can build and exercise it; and
- target-size feedback behavior has been separately characterized.

If promoted:

- add a public codestream-backend enum with `kGjxl` as the stable default;
- add actual-backend reporting to `VarDctEncodingSummary`;
- define behavior for builds without libjxl support;
- test C API and package/export behavior;
- support target-byte/BPP modes as complete end-to-end policies, not as the
  same-frame experiment; and
- document the dependency, license, binary-size, and determinism implications.

If the libjxl tail is not faster, retain the result. It still establishes that
GJXL's native tail is not the limiting difference under the tested conditions,
or identifies which libjxl subphase and data distribution defeated the
expected gain.

## Test matrix

### Focused unit and integration tests

- Backend-option validation and unavailable-build behavior.
- Frame-profile rejection before bridge entry.
- Field-by-field state digest parity.
- Direct qDC modular construction and dequantized-DC parity.
- Every supported strategy and edge-group layout.
- Output atomicity for invalid input and injected allocation failure where
  practical.
- Determinism across repeated runs and supported thread counts.
- Pinned decoder acceptance and `jxlinfo` structural checks.
- Exact decoded-pixel equality between tails for the same frame.

Suggested root tests:

- extend `codestream_encoder_test` for backend dispatch and atomicity;
- add `libjxl_tail_test`, built only with the experiment option;
- extend `codestream_workflow_test` for single-attempt integration; and
- extend `encoding_benchmark_cli_test.py` for the new scope and JSON schema.

### Existing regression gates

At minimum, run:

```sh
cmake -S . -B build/libjxl-tail -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_BUILD_TESTS=ON \
  -DGJXL_BUILD_BENCHMARKS=ON \
  -DGJXL_ENABLE_LIBJXL_TAIL_EXPERIMENT=ON

cmake --build build/libjxl-tail -j

ctest --test-dir build/libjxl-tail \
  -R '^(codestream_encoder|codestream_workflow|libjxl_tail|encoding_benchmark_cli)$' \
  --output-on-failure

cmake --build build/libjxl-tail --target codestream-conformance -j
```

Also configure and run focused codestream tests once with the experimental
option off. That build is the compatibility gate for ordinary users.

## Expected file ownership

The exact split may change during implementation, but the intended ownership
is:

| Area | Expected files |
| --- | --- |
| Build option and private libjxl targets | `CMakeLists.txt`, possibly one helper under `cmake/` |
| Backend-neutral dispatch | `src/codestream/encoder_backend_internal.h/.cpp` |
| GJXL-to-libjxl adapter and warm context | `src/codestream/libjxl_tail_internal.h/.cpp` plus an unavailable stub |
| Workflow selection | `src/codestream/workflow.cpp`, `workflow_internal.h` |
| Profiling | `encoder_internal.h`, `workflow_internal.h`, `benchmarks/encoding_benchmark.cpp` |
| Pinned private bridge | `third_party/libjxl/lib/jxl/enc_precomputed_vardct.h`, `enc_frame.cc`, `enc_modular.*` |
| Correctness tests | `tests/libjxl_tail_test.cpp`, existing codestream/workflow tests |
| Benchmark CLI tests | `tests/encoding_benchmark_cli_test.py` |
| Result documentation | this file and, after measurement, `metal-encoding-performance.md` |

Keep the libjxl-specific types behind the private adapter so the native GJXL
tail, installed headers, and downstream consumers do not acquire libjxl
implementation details.

## Main risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Libjxl accidentally reruns frontend work | Invalid comparison and misleading timing | Audit the bridge call graph; prohibit pixel input and frontend heuristic entry points. |
| Float DC re-quantization changes qDC | Different decoded image | Add a direct integer-qDC modular path and exact qDC tests. |
| Header or matrix-scale mismatch | Same coefficients decode differently | Compare all serialization-critical state and require exact decoded pixels. |
| Different thread counts or nested pools | False speedup or regression | Use explicit participant counts, record runner behavior, and prevent nested unbounded pools. |
| Adapter copy dominates | Hybrid misses the expected bound | Measure the copy separately but include it in complete hybrid time; optimize ownership only after the first result. |
| Libjxl effort changes size substantially | Speed result is not an equivalent tradeoff | Report a speed/size effort sweep and keep decoded reconstruction fixed. |
| Target-size search selects a different frontend frame | No longer a same-frame comparison | Defer target-size modes and label their later result as end-to-end policy behavior. |
| Private libjxl patch drifts | Maintenance burden | Keep one narrow patch commit, pinned base revision, focused bridge tests, and no public libjxl ABI dependency. |
| Full libjxl linkage bloats normal builds | Downstream cost despite experimental status | Default the option off and keep libjxl targets private/non-installed. |
| Historical profiles are treated as current | Incorrect Amdahl bound | Capture fresh same-revision baselines before measuring the bridge. |

## Completion definition

The experiment is complete when:

- one completed GJXL frame can be serialized by both tails without rerunning
  frontend work;
- both outputs decode to exactly the same pixels through the pinned decoder;
- all serialization-critical input state is fingerprinted and identical;
- adapter, libjxl-tail, native-tail, and complete workflow times are separately
  observable;
- output sizes, efforts, threads, revisions, and timing boundaries accompany
  every result;
- multi-process paired benchmarks quantify the measured hybrid speedup and the
  perfect-tail Amdahl bound; and
- the result supports an explicit decision to retain the bridge as a research
  tool, promote it, or remove it.
