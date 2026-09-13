# Prediction-aware DC quantization and adaptive DC smoothing

These are separate, opt-in lossy experiments on `feat/dc-coding`. CPU and
resident Metal integration and bounded matched-quality qualification are
complete. The [qualification report](dc-processing-qualification/REPORT.md)
records mixed rate benefits, encoding costs, and two unresolved compact
calibration targets; the evidence does not support enabling either by default.
Weighted residual coding and size-adaptive predefined DC trees are now the
encoding defaults. Ordinary rounding and disabled smoothing remain unchanged;
see [the combined-default decision](dc-small-trees/README.md). The measurements
below retain their original gradient/round/no-smoothing control, named `default`
in the frozen study artifacts.

## Controls and reconstruction contract

```sh
build/release/gjxl_encode --backend cpu --effort 4 --distance 1 \
  --dc-prediction weighted --dc-quantization prediction-aware \
  --adaptive-dc-smoothing input.pfm output.jxl
```

Omit either of the final two switches to test the other independently.
`--dc-quantization round` explicitly selects the default. The quality benchmark
accepts the same switches and records them using fully resident Metal.
`gjxl_dc_processing_benchmark` additionally supports independently calibrated
distances in one shared-backend process; see `tools/dc_coding/README.md`.

C++ `VarDctEncodingOptions` and its result summary append `dc_quantization`
and `adaptive_dc_smoothing`. Prediction-aware mode uses the selected
`dc_prediction`, one extra precision bit, a 0.62 residual deadzone, and even
integer residual quantization beyond magnitude two. Predictor state resets
per 256x256 DC group and channel; Y is coded before chroma. Lossless residual
coding is still separately selectable.

The C options append `GJXLDcQuantization dc_quantization` and a uint32 smoothing
flag (only 0/1 accepted). `GJXLEncoderOptions` is now 28 bytes; its 12-, 16-,
20-, and 24-byte predecessors use `struct_size` to select missing-field defaults:
weighted prediction, ordinary rounding, and disabled smoothing. Rust exposes `DcQuantization::{Round, PredictionAware}` and
`adaptive_dc_smoothing: bool`.

The completed frame stores authoritative DC integers and an **unsmoothed**
dequantized cache. Precision scales are reflected in validation, assembly,
and codestream headers. CPU reconstruction filters a temporary DC image before
conversion to transform low frequencies, so AQ evaluates the signaled decoder
behavior. Smoothing uses the maximum normalized gap across all three channels,
the base DC steps, explicit fused operations, and unchanged borders.

The new precision path rounds the dequantized blue product before adding
reconstructed Y, matching the pinned decoder. The precision-zero default keeps
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
difference. Ordinary default rounding retains its previous arithmetic.

## Validation retained locally

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
piecewise-linear interpolation sensitivity check. The measured incremental
lossy gains do not justify changing ordinary DC rounding or disabled smoothing.
The later promotion of lossless weighted prediction and adaptive predefined
trees is independent of this decision; both lossy controls remain opt-in.
