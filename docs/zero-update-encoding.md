# Evaluation-free zero-update encoding

Resident Metal encoding now skips perceptual evaluation when the resolved AQ
policy has zero updates and `collect_final_butteraugli_score` is false. This
applies to default-density efforts 1–3, including forced resident target-size
attempts. The score history is empty and
`final_butteraugli_score_evaluated` is false.

Previously the workflow forced a terminal score for zero-update policies.
That prepared the Butteraugli reference, reconstructed and filtered the encoded
image, and evaluated its perceptual error without using that error to change
any encoding decision.

## Execution and storage

The first final-frame pass now resets device error state, constructs the
resident quantizer, computes forward coefficients and final CfL, and writes
the final adjusted integer coefficients. Repeated use preserves the existing
forward-coefficient/CfL reuse rules. Completed-frame order-population counting
and serialization still run.

An `evaluation_free` preparation option is part of the evaluator cache identity.
Switching between scored and unscored policies rebuilds the evaluator as
needed. The option omits Butteraugli reference preparation and storage,
reconstructed linear RGB, reconstruction coefficients, and loop-filter scratch.
The search-domain image used by inverse Gaborish for AC selection remains when
needed. Shared small quantizer/control/diagnostic buffers remain allocated.

The final coefficient specialization performs its existing numeric validation
but does not write reconstructed coefficients. Its unused reconstruction
argument binds valid scratch, avoiding a full coefficient-sized allocation.
The frame reset kernel preserves injected numeric-error handling. No inverse
transform, reconstructed-image filtering, color conversion, or metric dispatch
runs in the final pass. Decoder loop-filter settings are unchanged.

Explicit final-score requests, complete diagnostic APIs, CPU encoding, exact
coefficient mode, maximum-error control, and policies with AQ updates retain
their evaluation behavior. High-density mode still resolves to four updates.

## Qualification

Regression coverage includes:

- Byte equality between scored and evaluation-free public encoding at efforts
  1–3, with effort 4/7/10 and CPU controls.
- First-use and repeated ordinary/profiled resident coefficient equality,
  empty scores/readbacks, no reference submission, and smaller device storage.
- Scored/unscored cache transitions in both directions on one prepared pipeline.
- Rejection of score requests on an evaluation-free preparation and atomic
  output behavior under upload, submission, completion, numeric, and readback
  failures.
- Cold/warm bounded workflow execution, profiling bounds, and target-size retries.

Release qualification on 2026-09-10 used Apple Clang, libc++ and C++20:

- The full suite passed 124/126 tests. `quantization_pipeline` reproduced the
  inherited CPU golden mismatch (actual `0.24919039011001587`, expected
  `0.24914586544036865`); the older review build reproduced the same failure.
  `metal_aq_evaluation` failed once at the existing assertion “Resident final
  CfL dispatch was not profiled exactly once.” Its cause is unconfirmed: ten
  subsequent candidate runs and ten baseline runs all passed. This is not a
  clean full-suite result.
- All six affected test executables passed on the final build. The public
  workflow, Metal evaluator and Metal quantization pipeline also passed with
  `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`.
- Three profiled effort-1 encodes per benchmark image reported zero scores,
  no reference/Butteraugli stages and no final reconstruction/filter stages.
  Each profiled result matched its unprofiled warmup bytes.
- Representative outputs decoded successfully with pinned `djxl` revision
  `e8ff0976`. Before/after codestream SHA-256 values matched in every benchmark
  process, including the effort-4 controls.

## Complete-call timing

Baseline: the study's Release build at `a7b5f57`. Candidate: this worktree based
on `1b108ce`, whose encoder sources match that study revision before this patch.
Both used the same benchmark source, forced fully resident Metal, default
density, distance 1.9 and eight participating CPU threads. Each independent
process performed one validation encode, two warmups and three timed public
encode calls. Input loading and file writes were outside the timed boundary.

Seven before/after process pairs alternated AB/BA order for effort 1. The
latencies below are medians of process medians; reductions are medians of
within-pair ratios, so the displayed columns need not divide exactly.

| Image | Dimensions | Before (ms) | After (ms) | Paired latency reduction |
| --- | --- | ---: | ---: | ---: |
| Kodak 10 | 512 × 768 | 21.15 | 18.13 | 15.8% |
| CLIC `0c49a5cce349020bbba2f97ae41e90ba` | 2048 × 1358 | 64.30 | 49.68 | 23.0% |
| Unsplash campus interior | 4249 × 2824 | 280.32 | 155.78 | 43.4% |

Three effort-4 control pairs per image changed by −0.7%, −2.5%, and +0.6%
latency, respectively. Another GJXL `rate_probe` workload was active throughout
sampling and is recorded in the artifact directory. These are local paired
observations under background load, not isolated performance qualification or
new corpus-wide Pareto points. Efforts 2–3 share the zero-update path and have
byte-parity regression coverage; their latency was not independently timed.

The device inventory calculated by `ComputeResidentWorkflowStoragePlan` fell
by approximately 47% for these images. At 12 MP it fell from 3,886,643,771 to
2,048,892,615 bytes. This is a preflight device-storage bound, not measured RSS
or the complete host/device peak. Shared frontend storage remains.

Evidence is in `/tmp/gjxl-zero-update-20260910/`: `benchmark.py`, input cases,
binary/source manifests and patch, all raw timing reports and JXL outputs,
`benchmark-summary.json`, `background-processes.log`, storage calculations,
profile and pinned-decoder logs, and test logs. The benchmark summary retains
all paired ratios and codestream hashes. The original comparison plot has not
been regenerated.

To reproduce qualification from this worktree:

```sh
cmake -S . -B build/zero-update -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON \
  -DGJXL_ENABLE_LIBJXL_REFERENCE=OFF
cmake --build build/zero-update -j 8
SDKROOT=$(xcrun --show-sdk-path) ctest --test-dir build/zero-update --output-on-failure
```
