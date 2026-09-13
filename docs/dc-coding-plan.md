# DC coding implementation and qualification

Historical plan and progress log. The initial gradient-default decisions below
were superseded by the approved [weighted/adaptive default](dc-small-trees/README.md).
Prediction-aware quantization and adaptive DC smoothing remain opt-in.

Base: `2c936fa96a334d7f67e04abf85fb38ccfcdf73f5`. Work is isolated on
`feat/dc-coding`; existing worktrees and historical evidence are preserved.

This records the full implementation and bounded qualification sequence.
Completing prediction support alone does not complete the project.

1. Integrate native weighted DC prediction as an explicit option, retaining the
   gradient default and fallback. Port the predictor, context property and tree
   signaling together. Include C/C++ options, CLI and benchmark provenance.
2. Use a constant-time context lookup and reuse predictor scratch. Account all
   owned storage in the existing resource budget. Qualify correctness against
   pinned libjxl, decoded-pixel equality, determinism, invalid-input atomicity,
   allocator failures, and sanitizers. Preserve default gradient bytes.
3. Qualify size, complete encode and decode cost on current e3/e4 and higher
   efforts, with photographic and compact controls. Use fresh Release builds,
   explicit manifests, resumable ledgers, alternating timing and quiet-machine
   checks. Decide automatic selection from independent qualification; do not
   silently impose an unqualified image-size or effort threshold.
4. Implement prediction-aware DC quantization and adaptive DC smoothing as
   separate opt-in experiments. Maintain decoder-equivalent DC reconstruction
   in CPU and resident Metal/AQ paths and validate against pinned libjxl.
   Evaluate each separately and together at measured matched quality, including
   timing. Retain only evidence-supported default policies, and document
   negative results or tradeoffs as well as improvements.

## Status

- Isolated worktree and baseline were checked before starting measurements;
  existing worktrees and unrelated changes remain preserved.
- Weighted prediction is integrated through serialization, resource planning,
  public C/C++/Rust options, and CLIs. Gradient remains the default.
- Direct pinned oracle, serializer allocation-failure/sanitizer coverage,
  workflow/API, and relocated installed consumers pass. Baseline builds
  reproduce the pre-existing quantization golden mismatch and Metal
  initialization UBSan failure; see `docs/dc-prediction.md` for boundaries.
- Two 12-case pilots pass unchanged gradient bytes and identical weighted
  decoded pixels. The first pilot exposed cross-process timing noise; the
  second uses alternating complete calls in one warm process. Saved records
  pass a resume audit and the decoder pixel hash matches independent `djxl`.
- The 612-case weighted qualification and all 18 timing cases (20 pairs each)
  are complete under `build/dc-weighted-qualification-v1`. All baseline-gradient
  output and weighted decoded-pixel checks pass, including the completed-case
  resume audit. Portable evidence is in `docs/dc-prediction-qualification/`.
- Policy decision: retain gradient by default; expose weighted explicitly.
  Photographic savings are useful, but compact regressions and low-effort
  complete-call overhead do not support an unconditional automatic switch.
- Prediction-aware quantization and adaptive smoothing are implemented on CPU,
  including frame reconstruction, header signaling, AQ, resource bounds, and
  C/C++/Rust/CLI controls. The direct pinned oracle passes 864 quantization and
  96 smoothing cases. Fifty independent-decoder CPU cases cover separate/combined
  modes, mixed transforms, group boundaries, and extra precision 2/3 controls.
- Resident Metal integration now covers reconstruction, exact coefficients,
  frame-only/final-only output, completed frames, cache reconfiguration,
  profiling, and resource bounds. The diagonal CPU/device primitive comparison
  passes 1,410 exact cases and 20 invalid-input checks; 50 Metal-generated
  codestream cases pass the independent decoder. A higher-precision fused
  arithmetic mismatch was fixed explicitly on both backends.
- Final validation passes 139/140 Release tests, four focused ASan/UBSan
  targets, and all ten Rust tests in a fresh native build. Only the established
  baseline `quantization_pipeline` golden mismatch remains in Release.
  See `docs/dc-processing.md` for retained test boundaries and logs.
- Lossy qualification has all 270 curve cases (1,350 JXL outputs), all 108
  default regression checks, and 180 pilot-output comparisons complete.
  Ten of twelve preselected cases have measured matched-quality encode/decode
  timing; two edge controls remain explicitly unresolved after their declared
  calibration budgets. All completed data and exclusions are audited.
  Portable results, interpolation sensitivity, and the policy decision are in
  `docs/dc-processing-qualification/REPORT.md`.
- The full implementation and bounded qualification sequence is complete.
  Keep ordinary rounding and smoothing disabled by default; expose the lossy
  controls explicitly. Incremental rate benefits are inconsistent and do not
  justify prediction-aware quantization's measured encoding overhead. The
  qualification checkpoint was saved before staging or committing.

## Evidence to retain

Source diff/revision and shader hashes; compiler/build settings; input and
decoder/scorer identities; exact-output and oracle checks; test and sanitizer
logs; complete-call and decode timing samples; rate/quality ledgers; a report
that distinguishes historical fixed-frame evidence from current results.

The final source, tests, documentation, native benchmarks, and selected
validation logs are retained in `build/dc-coding-final-checkpoint-v1` with a
hash manifest. Raw measurement artifacts remain in their separate frozen study
directories; the final checkpoint does not replace those identities.
