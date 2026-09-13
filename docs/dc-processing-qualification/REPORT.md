# DC quantization and smoothing: qualification and policy

Keep ordinary DC rounding and disabled adaptive DC smoothing as the defaults.
Both experiments are implemented for CPU and resident Metal/AQ, separately and
together, and remain explicitly selectable. Prediction-aware quantization has
some rate benefit, but its incremental encoding cost and inconsistent gains do
not support enabling it automatically. Smoothing is approximately rate-neutral
on these photographic medians; combining the two often costs rate.

Weighted residual prediction remains a separate explicit option. Its stronger
evidence is in the [lossless-prediction report](../dc-prediction-qualification/REPORT.md).
No automatic effort, image-size, or content threshold was fitted or enabled.

## Controls and measured scope

The candidate is the native implementation on `feat/dc-coding`, based on
`2c936fa96a334d7f67e04abf85fb38ccfcdf73f5`. The accepted run is
`build/dc-processing-qualification-v1`; it freezes sources, build commands,
binaries, decoder libraries, input hashes, scorer identity, and machine state.
Measurements use fully resident Metal, eight CPU threads, no stage profiling,
and no final diagnostic score. The machine is `Mac16,7`, macOS 15.6 / 24G84,
on AC power.

| Control | Residual coding | DC quantization | Adaptive smoothing |
| --- | --- | --- | --- |
| `default` | Gradient | Ordinary rounding, precision 0 | Off |
| `weighted` | Weighted | Ordinary rounding, precision 0 | Off |
| `quantize` | Weighted | Prediction-aware, precision 1 | Off |
| `smooth` | Weighted | Ordinary rounding, precision 0 | On |
| `both` | Weighted | Prediction-aware, precision 1 | On |

The quantization experiment includes the extra precision bit, prediction-residual
deadzone, and even-integer residual rule as one policy. It does not isolate
precision alone. Gradient prediction-aware quantization is covered by correctness
tests, but the full rate/time matrix here uses the weighted quantizer.

The preselected curve cohort has six CLIC photographs, six Kodak photographs,
two 3840x2160 photographs (planter and bedroom noise), and four compact synthetic
controls (edge, gradient, saturated, texture). Images were selected from the
retained input manifest before the full run: evenly spread sorted CLIC/Kodak
entries and the first/last sorted 4K entries. Efforts 3/4/7 and distances
0.5/0.7/1/1.4/2 give **270 complete cases and 1,350 retained JXL outputs**.

Quality is measured on independently decoded finite linear-sRGB float pixels
using the pinned fast-ssim2 implementation at
`c3867954c7bec8a761951df9256354b305fd0cff`. Its adapter uses full or 256-row strip
mode automatically according to geometry; the 4K inputs use strip mode.
The pinned libjxl decoder is built from
`e8ff09762481785938d8e4e01333ed3917571161`. Decoder identity is checked against
the helper, and every gradient/weighted pair has exactly equal decoded pixels.

## Rate at measured decoded quality

Each image/effort uses the same quality interval for all five controls. The
reported BD-rate integrates PCHIP interpolation of log(file bytes) against
measured SSIMULACRA2. Strictly dominated samples are removed from the frontier;
all observations remain in `curves.csv`. There is no extrapolation. All 54
image/effort groups have sufficient common support under the declared protocol.

The table gives the median image BD-rate change **relative to weighted coding**,
so it isolates the incremental lossy policy. Negative values mean fewer bytes
at matched quality. These are per-corpus medians, not a pooled corpus score.

| Corpus | Images | Effort | Quantization | Smoothing | Both |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIC | 6 | 3 | -0.495% | +0.083% | -0.025% |
| CLIC | 6 | 4 | +0.004% | +0.075% | +0.885% |
| CLIC | 6 | 7 | -0.586% | +0.053% | +0.192% |
| Kodak | 6 | 3 | +0.105% | +0.103% | +0.540% |
| Kodak | 6 | 4 | +0.054% | +0.112% | +0.739% |
| Kodak | 6 | 7 | -0.184% | +0.033% | +0.118% |
| 4K photographs | 2 | 3 | -0.645% | +0.280% | +0.272% |
| 4K photographs | 2 | 4 | -0.711% | -0.041% | +0.108% |
| 4K photographs | 2 | 7 | -1.264% | +0.325% | +0.096% |
| Compact controls | 4 | 3 | +1.283% | -0.015% | +2.129% |
| Compact controls | 4 | 4 | +2.024% | -0.059% | +2.216% |
| Compact controls | 4 | 7 | +1.490% | +0.032% | +2.325% |

The gains vary substantially by image. CLIC quantization ranges from -6.188%
to +1.021% at effort 7. The combined policy's compact effort-7 range reaches
+31.263%. Individual compact curves need particular care: the edge control's
common quality spans are only 0.189, 0.906, and 0.643 points at efforts 3/4/7.

A saved-data sensitivity check replaces PCHIP with piecewise-linear log-rate
interpolation over the identical frontiers and intervals. CLIC quantization
medians become -0.545%, -0.072%, and -0.707%; Kodak becomes +0.090%, +0.117%,
and -0.152%. CLIC effort-4 smoothing changes sign (-0.011% instead of +0.075%).
The largest photographic per-image difference between these methods for the
three lossy controls is 0.242 percentage points. Compact differences reach
3.556 points. Tiny signed changes therefore should not be interpreted as robust
wins. The broader conclusion of mixed gains and no automatic default remains.

## Complete encode and decode cost

Timing uses one preselected CLIC image, Kodak 01, the 4K planter, and the edge
control at each effort. Each control's distance is calibrated to the default's
distance-1 measured quality within **0.05 SSIMULACRA2**. Calibration retains all
107 completed probes and allows at most ten new probes per lossy control.

**Ten of twelve cases calibrated and completed timing.** The nine photographic
cases and edge effort 4 are included. Edge effort 3's combined policy exhausted
the probe budget with a best error of +0.086853 points; edge effort 7's quantizer
exhausted it at +0.117632. Both cases remain explicitly unresolved and have no
accepted matched-quality timing. A normal resume cannot reset their budgets.

Accepted timing contains 1,000 complete encode samples and 1,600 complete decode
samples. Encoding uses a shared warmed process/backend, three warmups per
control, and 20 rotated/reversed rounds. Every control occupies each position
four times. The timer surrounds `EncodeLinearRgbVarDctCodestream`; input file
loading, output file writes, and validation are outside it. Decoding uses
fresh single-thread decoder creation, output allocation, decode, and decoder
destruction, with three warmups and 20 alternating pairs per comparison.
The returned decoded pixel buffer is freed after the timer.

Every timed codestream matches its calibrated artifact. Decoder output is
finite and repeat-deterministic and matches the independently scored pixel hash.
The largest accepted quality error is 0.049729 points. Lossy controls are not
required to have equal decoded pixels to the default.

The following ranges span the **nine photographic timing cases**, relative to
the gradient default at the calibrated target. File-size changes are point
comparisons, distinct from the curve BD-rate above. Negative time changes mean
faster complete calls.

| Control | File-size change | Paired encode-time change | Paired decode-time change |
| --- | ---: | ---: | ---: |
| Weighted | -3.139% to +0.397% | -2.157% to +3.490% | +1.331% to +3.911% |
| Quantization | -7.981% to +1.101% | +7.463% to +26.188% | +1.282% to +14.526% |
| Smoothing | -6.257% to +0.430% | +0.531% to +4.869% | +0.301% to +15.866% |
| Both | -6.070% to +1.288% | +9.344% to +28.102% | +1.741% to +14.482% |

Relative to the **weighted control within the same encode rounds**, quantization
adds 6.979% to 27.759% median time and is slower in 172/180 photographic rounds.
The combined policy adds 7.126% to 27.353% and is slower in 175/180 rounds.
Smoothing alone ranges from -1.284% to +0.958%, with 89/180 slower rounds;
these small differences do not establish a consistent incremental encoding cost.
The implementation can omit the filter when final-only output has no
reconstruction consumer, while still signaling smoothing to the decoder.

The edge effort-4 matched point illustrates the compact risk: file size grows
23.692% for quantization, 9.231% for smoothing, and 21.077% for both against the
gradient default (weighted alone grows 9.385%). The narrow edge curves and
discrete changes explain why its point comparison and integrated BD-rate should
be reported separately.

Paired medians are medians of within-round or within-pair ratios, not ratios of
standalone medians. `timing-paired.csv` also retains interquartile ranges and
slower-round counts. The Mac remained interactive; guards stopped/resumed the
run around background CPU spikes and checked before/after accepted timing
cases. Ordinary scheduling and power variability remain. Small deltas are not
universal latency guarantees, and the two unresolved targets are not hidden.

## Correctness and engineering checks

- All 108 overlapping default encodes are byte-identical to retained baseline
  outputs. All 36 overlapping pilot cases preserve all five JXL outputs
  (180 byte comparisons). All 1,350 full-run curve artifacts pass the final audit.
- The final Release suite passes **139/140**. Only the quantization golden
  mismatch reproduced on the unchanged baseline remains. The relocated installed
  consumer and expanded resource-budget/API/Metal tests pass. The final log is
  `build/release/dc-final-full-tests-v1.log`.
- The pinned direct CPU oracle passes 864 quantization and 96 smoothing cases.
  The parallel Metal primitive passes 1,410 exact CPU/device comparisons and
  20 invalid/overflow rejection cases. Independent-decoder conformance covers
  50 CPU and 50 final Metal cases, with the established RGB transform tolerance.
- Focused ASan/UBSan tests pass for weighted prediction, DC primitives, the
  pinned oracle, and reconstruction. The known baseline Metal-initialization
  UBSan issue is not claimed fixed. See [the implementation record](../dc-processing.md)
  for the exact logs, floating-point contract, and sanitizer boundaries.
- All ten Rust tests pass in a fresh native build, including separate/combined
  controls and repeated output equality (`build/dc-final-rust-tests-v1.log`).

The initial serial GPU quantizer was superseded after a partial pilot exposed
large overhead. Its 26 completed cases remain in `build/dc-processing-pilot-v1`.
The diagonal implementation preserves all 130 overlapping serial-pilot outputs
in the subsequent pilot. Serial pilot timing is diagnostic only and is not used
for the qualified rate/time tables above.

## Adoption decision and evidence

Retain weighted DC prediction as an explicit rate-versus-time choice, and retain
the quantization/smoothing implementations as opt-in experiments. Keep gradient,
ordinary rounding, and smoothing disabled by default. Correct decoder-equivalent
CPU/Metal integration is useful infrastructure, but correctness alone does not
establish a beneficial default policy. Additional content-specific selection or
quantization-kernel optimization would require a separately declared validation
set and new timing evidence.

This is an incremental GJXL comparison. Libjxl supplies the oracle and decoder;
the experiment does **not** directly measure how much of the overall matched-quality
GJXL-versus-libjxl gap has closed. Historical frozen-frame writer attribution and
this measured-quality frontend experiment answer different questions.

Portable data: [curves](curves.csv), [BD-rate](bd-rate.csv),
[rate summaries](rate-summary.csv), [timing](timing.csv),
[paired timing](timing-paired.csv), [interpolation sensitivity](sensitivity.csv),
[sensitivity summaries](sensitivity-summary.csv), [completion summary](summary.json),
and [audit/provenance](audit.json). Raw inputs, codestreams, probes, commands,
source snapshot, failed attempts, and sample ledgers remain under the accepted
build directory. The final source/test/documentation snapshot, native benchmark
binaries, and validation logs are retained separately in
`build/dc-coding-final-checkpoint-v1`. The [collector README](../../tools/dc_coding/README.md) describes
reproduction. The portable tables and sensitivity audit can be regenerated
without running a codec or metric:

```sh
python3 tools/dc_coding/processing_study.py report --output build/dc-processing-qualification-v1
python3 docs/dc-processing-qualification/analyze.py build/dc-processing-qualification-v1
```
