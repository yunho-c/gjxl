# DC residual prediction

The native encoder supports two lossless representations of quantized DC:
`gradient` (the existing default) and `weighted` (the default JPEG XL weighted
predictor with its matching error contexts and fixed tree). Selecting weighted
prediction preserves reconstructed pixels. It does not enable prediction-aware
DC quantization or adaptive DC smoothing. Those separately controlled lossy
experiments are described in [dc-processing.md](dc-processing.md).

The choice is independent of effort and compression mode. Weighted prediction
can reduce photographic file sizes, but it can increase small/synthetic files
and adds encoder and decoder work. No automatic selection policy is enabled.

```sh
gjxl_encode --distance 1 --effort 4 --dc-prediction weighted input.pfm output.jxl
```

For C++, set `VarDctEncodingOptions::dc_prediction` to
`VarDctDcPrediction::kWeighted`; direct serialization accepts the same field in
`VarDctCodestreamOptions`. The workflow summary records the selection. For C,
initialize `GJXLEncoderOptions` and set `dc_prediction` to
`GJXL_DC_PREDICTION_WEIGHTED`. Older C option sizes select gradient, including the
original 12-byte and subsequent 16-byte layouts. Rust exposes
`EncoderOptions::dc_prediction` with `DcPrediction::Gradient` and `Weighted`.

The predictor, context property, and tree signaling are changed together. AC
metadata residuals retain their existing representation; DC and metadata share
entropy models, so their joint model costs may change. Both entropy selection
and final emission use the selected DC tree. Managed scratch is accounted by
the serializer and complete-workflow storage plans; the five error arrays are
reused between channels and add at most 10,320 bytes of element storage for a
DC group. Groups are processed sequentially.

The implementation derives from the restricted prototype at `f4094bb` on
`diagnostics/writer-rate-attribution`. Its historical effort-7 report is retained
in that worktree under `docs/native-weighted-dc/`. Those frozen-frame results
are separate from current encoder qualification and from matched-quality
comparisons against stock libjxl.

## Validation status

Implementation and qualification are isolated on `feat/dc-coding`, based on
`2c936fa9`. The default policy is unchanged.

- The 45-case pinned weighted-predictor oracle passes in Release and with
  ASan/UBSan. It covers strided and boundary geometries, five signal families,
  deterministic/concurrent calls, and atomic invalid-input rejection.
- Serializer storage planning and allocation-failure sweeps cover both
  predictors, entropy behaviors, and profiled/unprofiled paths. Focused Release
  and ASan/UBSan checks pass.
- Workflow, C/C++ API, relocated installed consumer, and explicit decoder CLI
  checks pass. The weighted-only checkpoint passed 9/9 Rust native tests;
  the final CPU/Metal quantization/smoothing integration passes 10/10 in a
  fresh Rust/native build, including the separate/combined mode checks.
- The first full Release run passed 135/137. The new missing installed header
  was corrected and the relocated C++20/C++23 consumer now passes. The remaining
  quantization golden mismatch reproduces exactly in a freshly built unchanged
  `2c936fa9` baseline (`0.24919039011001587` versus `0.24914586544036865`).
  The final expanded Release suite passes 139/140, with that sole failure.
- C API ASan/UBSan reaches an existing null `metal-cpp` shared-pointer member call
  during Metal backend creation. A fresh baseline sanitizer build reproduces
  the same stack at `metal_backend.cpp:565`. It is recorded separately from
  native predictor/serializer sanitizer coverage; it is not suppressed.

Logs live under `build/release/dc-*.log`, `build/sanitize/dc-*.log`,
`build/baseline/dc-check-test.log`, `build/dc-baseline-sanitize-test.log`, and
`build/dc-rust-tests.log`.

Two 12-case pilots passed exact base-gradient output and decoded-pixel checks.
The first exposed noisy cross-process encode timing; the second uses alternating
calls in one process. Its completed-case resume audit also passes. A standalone
decoder helper was cross-checked against the pinned `djxl` linear-float output,
and rejects deliberately unequal decoded pixels. The 612-case current-effort
study and all 18 timing cases (20 pairs each) are complete. All base-gradient
bytes and decoded-pixel checks pass. Gradient remains the default because of
compact regressions and the complete-call cost at low efforts. See the
[qualification report](dc-prediction-qualification/REPORT.md) and
[`tools/dc_coding/README.md`](../tools/dc_coding/README.md).
