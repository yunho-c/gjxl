# Prediction-aware DC quantization and adaptive DC smoothing

The public encoder automatically enables prediction-aware DC quantization and
adaptive DC smoothing at efforts 4–10. Efforts 1–3 use ordinary rounding and
no smoothing. Both controls remain independently overridable. Weighted residual
coding and size-adaptive predefined DC trees remain the lossless defaults.

The [e4 qualification](dc-e4-qualification/REPORT.md) supports this as an
anti-banding policy, with additional encode time and scene-dependent rate/texture
tradeoffs. The earlier [rate study](dc-processing-qualification/REPORT.md) used
a gradient/round/no-smoothing baseline named `default`; its measurements and
original decision are historical, not a description of today's automatic policy.

## Controls and reconstruction contract

```sh
build/release/gjxl_encode --backend cpu --effort 4 --distance 1 \
  --dc-prediction weighted --dc-quantization prediction-aware \
  --adaptive-dc-smoothing input.pfm output.jxl
```

Omitting a control follows effort for that control. Use
`--dc-quantization round --no-adaptive-dc-smoothing` for the former lossy defaults,
`--dc-quantization prediction-aware --no-adaptive-dc-smoothing` for quantize alone,
and `--dc-quantization round --adaptive-dc-smoothing` for smoothing alone.
`--dc-quantization auto` explicitly restores automatic quantization. Both CLI
switches for smoothing reject duplicates or conflicts. The quality benchmark
accepts the same controls and reports their resolved values using resident Metal.
`gjxl_dc_processing_benchmark` additionally supports independently calibrated
distances in one shared-backend process; see `tools/dc_coding/README.md`.

C++ `VarDctEncodingOptions::dc_quantization` defaults to `kAutomatic`.
`adaptive_dc_smoothing` is `std::optional<bool>`: `nullopt` follows effort,
while explicit `false` or `true` overrides it. Result summaries always contain
resolved quantization and a plain smoothing boolean. CPU, resident Metal,
compatibility, target-size retry, and batch admission use the same resolution;
low-level DC primitives continue to require explicit modes. Prediction-aware mode uses the selected
`dc_prediction`, one extra precision bit, a 0.62 residual deadzone, and even
integer residual quantization beyond magnitude two. Predictor state resets
per 256x256 DC group and channel; Y is coded before chroma. Lossless residual
coding is still separately selectable.

The C struct remains 28 bytes with unchanged field offsets. Quantization values
0/1 retain their round/prediction-aware meanings; `GJXL_DC_QUANTIZATION_AUTOMATIC`
is 2. The smoothing field accepts `GJXL_DC_SMOOTHING_DISABLED` (0),
`GJXL_DC_SMOOTHING_ENABLED` (1), and `GJXL_DC_SMOOTHING_AUTOMATIC` (2).
The initializer selects automatic for both. Missing fields in the 12-, 16-,
20-, and 24-byte layouts also follow effort; fields present with explicit 0/1
retain those choices. No caller memory beyond `struct_size` is read or written.

Rust adds `DcQuantization::Automatic` and changes `adaptive_dc_smoothing` from
`bool` to `Option<bool>`. The default is `None`; existing explicit boolean callers
should use `Some(false)` or `Some(true)`. This distinguishes an override from an
effort-dependent default.

The completed frame stores authoritative DC integers and an **unsmoothed**
dequantized cache. Precision scales are reflected in validation, assembly,
and codestream headers. CPU reconstruction filters a temporary DC image before
conversion to transform low frequencies, so AQ evaluates the signaled decoder
behavior. Smoothing uses the maximum normalized gap across all three channels,
the base DC steps, explicit fused operations, and unchanged borders.

The new precision path rounds the dequantized blue product before adding
reconstructed Y, matching the pinned decoder. The precision-zero rounding path keeps
its legacy arithmetic to avoid changing established outputs as a side effect.
The coefficient-coding API requires nonzero extra precision for prediction-aware
quantization; the lower-level primitive can independently test grids 0..3.

CPU resource plans include the candidate integer image, optional weighted
predictor scratch, and both DC smoothing images. Invalid inputs and managed
allocation failures leave public outputs unchanged. GPU options retain the
requested mode/predictor for cache identity. The device path extracts all DC
before group quantization, optionally smooths into a separate plane, and embeds
low frequencies before inverse transforms. Exact-coefficient inputs are never
requantized. Final-only output omits smoothing work because it has no
reconstruction consumer. Device storage, workflow admission, and profiling
include all added buffers and dispatches.

Prediction-aware chroma subtraction uses explicit FMA on CPU and Metal. A
higher-precision boundary test exposed differing implicit contraction; the
explicit operation matches the pinned native toolchain and removes the backend
difference. Ordinary rounding retains its previous arithmetic.

## Automatic-policy validation

A fresh Release build in `build/dc-policy-e4-20260914` passes 21 selected native
tests, including CPU/Metal defaults and independent overrides, mixed-effort
batches, all three workflow storage plans, admission, C ABI layouts and legacy
sizes, CLI protocol, and the relocated installed consumer. The new policy test
compares automatic and explicit memory plans at all ten efforts; full encoding
checks cover efforts 3, 4, and 7 on CPU and Metal. The original workflow hash is
still pinned using explicit gradient/round/no-smoothing settings.

All 21 retained photographic cases reproduce the qualified bytes: e3 uses the
previous default, while e4/e7 use the previously measured `both` outputs at
12/24/48MP. Ten main-CLI checks reproduce corrected gradient streams for all
four explicit combinations and automatic selection at e3/e4; six duplicate or
conflicting smoothing requests reject without writing output. These are output
identity checks, not new timing measurements. All 11 Rust tests pass with a
fresh Cargo target/native build. The earlier qualification artifacts are
unchanged. See [the validation record](dc-policy-update-validation.json).

## Exact-coefficient merge regression

The broader merge review exposed an existing interaction in the exact-coefficient
Metal path when both DC controls were enabled. That path prepared low frequencies
on the CPU, then overwrote them with Metal's FP32 DC-to-LLF conversion. Its small
rounding differences could change later AQ decisions. On the existing 128x96
e7 fixture, CPU produced 537 bytes and exact-coefficient Metal produced 778;
the maximum score-history difference was 0.136601 against a 0.002 tolerance.
The same failure reproduced with explicit `both` on the pre-policy main revision.

Exact input now keeps smoothing and DC-to-LLF conversion on the CPU before the
inverse-transform handoff. The fixture produces matching 537-byte streams, with
a maximum score-history difference of 0.0000107884. A causal control kept CPU
smoothing but restored Metal LLF conversion and reproduced the original failure.
Stored DC integers and the unsmoothed frame cache retain their meanings; the
fully resident path continues to perform its DC processing on Metal.

The host admission plan includes the smoothing destination and atomic candidate
alongside group offsets. Allocation-failure and reuse tests exercise this scratch.
The workflow regression covers all four DC combinations with both predictors at
e4/e7 without relaxing the existing score tolerance. CLI checks pin the new
ordinary, maximum-compression, and maximum-error outputs, and separately retain
the old hashes with explicit round/no-smoothing controls. Follow-up results are
recorded in [the merge validation record](dc-policy-merge-validation.json).

A fresh Release export passes 125/126 native checks (the 11 worker-launch
fault-injection tests were excluded). The only failure is the unchanged CPU
`quantization_pipeline` golden mismatch, independently reproduced on main.
All 11 Rust tests, formatting, and Clippy pass with a fresh native build.
All 21 photographic output identities at 12/24/48MP and ten gradient/CLI
identities match the retained qualification; six invalid CLI cases also pass.
These are correctness checks, not new performance measurements.

## Historical implementation validation

- Full Release build succeeds; the final suite passes 139/140 tests. The only failure is the
  previously reproduced baseline `quantization_pipeline` golden mismatch
  documented in `dc-prediction.md`. This includes the relocated installed
  consumer and all default Metal regression tests. Logs:
  `build/release/dc-wavefront-full-build-v1.log` and
  `build/release/dc-final-full-tests-v1.log`.
- All 108 full-qualification default-output checks are byte-identical to
  retained pre-change gradient artifacts. The full run also preserves all
  five outputs from 36 overlapping pilot cases. These regression checks are
  separate from the accepted complete-call timing. Identities and counts are
  retained in `docs/dc-processing-qualification/audit.json` and the raw study.
- `tests/dc_processing_oracle_test.cpp` calls the actual pinned libjxl
  `QuantizeWP`, `QuantizeGradient`, `DequantDC`, and `AdaptiveDCSmoothing`.
  The expanded test passes 864 quantization cases and 96 smoothing cases
  exactly, with three quantizers, eight extents, six signal patterns, all
  supported nonzero precisions, and group boundaries. See
  `build/release/dc-metal-final-tests-v1.log` and the final full suite.
- `tests/dc_processing_test.cpp` covers aliases, row padding, concurrent
  invocations, residual boundaries, coupled-channel smoothing, tiny images,
  invalid values/modes, and managed allocation-failure atomicity.
- `gjxl_codestream_conformance_test --dc-processing --keep-artifacts` passes
  50 cases using the freshly built pinned `djxl` and `jxlinfo`. This checks
  separate/combined modes, both predictors, mixed transforms, horizontal and
  vertical group boundaries, and precision 2/3 controls. Full RGB reconstruction
  uses the existing conformance tolerance for CPU/SIMD transform differences;
  it is not an exact decoded-pixel claim for the lossy modes. Artifacts are in
  `build/dc-cpu-conformance-post-fma-v1`, with the log in
  `build/release/dc-cpu-conformance-post-fma-v1.log`.
- Expanded isolated AQ and complete CPU workflow tests pass with the planned
  memory allowances and unchanged outputs relative to unconstrained execution.
  See `build/release/dc-cpu-storage-tests-v2.log`.
- Focused ASan/UBSan tests for weighted prediction, both DC primitives, the
  direct oracle, and coefficient reconstruction pass. See
  `build/sanitize/dc-processing-final-tests-v1.log`. Only the external oracle adapter
  omits UBSan vptr checking because pinned libjxl is built without RTTI;
  native library sanitizer settings are unchanged. Managed backing ownership
  is checked separately; macOS LeakSanitizer is unavailable.
- All ten Rust tests pass in a fresh target directory/native build, including
  all separate/combined DC choices and repeated output equality
  (`build/dc-final-rust-tests-v1.log`). The fresh build avoids the existing
  `gjxl-sys/build.rs` limitation that does not track every native source edit.
  C API tests
  cover defaults, legacy struct sizes, explicit controls, determinism, and
  invalid enum/flag rejection.

The optional direct oracle uses `-DGJXL_DC_ORACLE_BUILD=...` and requires a
clean pinned source checkout. Its external libraries enter only the test
executable. The retained reference build is `build/dc-libjxl-reference`, sourced
from this worktree's clean `third_party/libjxl` at
`e8ff09762481785938d8e4e01333ed3917571161`.

## Resident validation

- Mixed-transform reconstruction, repeated reconfiguration, transitions between
  resident and exact-coefficient input, and separate frame-only encoding pass
  CPU comparisons (`dc-metal-integration-tests-v1.log`).
- The parallel device traversal passes 1,410 exact comparisons of CPU integers,
  raw dequantization, and smoothing, including five signal patterns, three
  precisions, multiple groups, and resident quantizers. Twenty invalid / overflow
  cases reject without publishing output. See
  `dc-wavefront-integration-test-output-v1.log` for captured test output.
- Metal uses diagonals within each group, with five full error planes. This
  replaces the serial GPU prototype after its first pilot exposed substantial
  overhead. The exact dependency derivation and memory tradeoff are recorded
  in `dc-reconstruction-design.md`. CPU retains its compact two-row predictor.
- Whole resident and compatibility workflows pass their planned memory budgets
  and repeated/cache execution checks. The expanded resident matrix covers
  efforts 3/4/7, both predictors, each lossy option, and final evaluation on/off.
  Profiled and ordinary outputs match. This also passes after the diagonal
  traversal / scratch change (`dc-wavefront-integration-tests-v1.log`).
- All 50 final Metal-generated codestream cases pass the independent decoder
  and existing float-RGB tolerance (`dc-wavefront-conformance-v1.log`), with
  artifacts under `build/dc-wavefront-conformance-v1`.
- The final full Release run passes 139/140. The newly exposed precision
  mismatch is fixed; the known baseline golden mismatch remains. Logs are
  under `build/release/`.

## Qualification and default policy

`build/dc-processing-qualification-v1` retains all 270 curve cases, 107 completed
calibration probes, and ten accepted timing cases with 1,000 complete encode and
1,600 complete decode samples. Edge effort 3 (combined) and effort 7
(quantization) exhausted the declared calibration budgets and remain excluded
from matched-quality timing. Their complete rate/quality curves remain visible.

The [portable report and CSVs](dc-processing-qualification/REPORT.md) compare
each control separately and together, including negative results and a
piecewise-linear interpolation sensitivity check. That original rate-only decision retained ordinary rounding and disabled
smoothing. The later photographic and gradient [e4 qualification](dc-e4-qualification/REPORT.md)
adds visual anti-banding evidence and complete-encode costs. The approved current
policy enables both at e4 and above while preserving explicit overrides; it does
not claim universal matched-quality rate savings.
