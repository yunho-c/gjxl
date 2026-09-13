# Native weighted DC prediction: current-effort qualification

This is the historical predictor-only qualification. Its initial decision
retained gradient by default: weighted reduced photographic rates at identical
decoded pixels, but compact regressions and complete-call cost counted against
an unconditional switch, particularly at efforts 3 and 4. No predictor image-size
or effort threshold was fitted or enabled.

The subsequent [combined-default decision](../dc-small-trees/README.md) accepts
those tradeoffs and enables weighted prediction with size-adaptive predefined
trees. The measurements and frozen artifacts below are unchanged.

## Size and correctness

The candidate is the native integration on `feat/dc-coding`, based on
`2c936fa96a334d7f67e04abf85fb38ccfcdf73f5`, including constant-time context lookup
and reused, budget-accounted predictor scratch. The unchanged baseline was
built freshly from that revision. All 612 cases preserve baseline gradient
codestream bytes and decode weighted output to exactly the same finite linear
RGB float pixels. All 1,836 retained outputs passed a completed-case resume audit.

Positive savings mean smaller complete files. Three distances (0.6, 1, 2) are
repeated observations of each image; counts below are image-distance points.

| Corpus | Effort | Points | Smaller / larger | Median saving | Saving range |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIC, 32 photographs | 3 | 96 | 95 / 1 | 1.172% | −0.044% to 8.931% |
| CLIC | 4 | 96 | 95 / 1 | 1.208% | −0.050% to 9.008% |
| CLIC | 7 | 96 | 95 / 1 | 1.324% | −0.046% to 10.583% |
| Kodak, 24 photographs | 3 | 72 | 66 / 6 | 0.382% | −0.575% to 2.830% |
| Kodak | 4 | 72 | 66 / 6 | 0.408% | −0.606% to 2.973% |
| Kodak | 7 | 72 | 66 / 6 | 0.402% | −0.611% to 3.565% |
| Four 4K photographs | 3 | 12 | 12 / 0 | 0.892% | 0.538% to 2.218% |
| Four 4K photographs | 4 | 12 | 12 / 0 | 0.908% | 0.562% to 2.369% |
| Four 4K photographs | 7 | 12 | 12 / 0 | 1.076% | 0.562% to 2.553% |
| Eight compact controls | 3 | 24 | 7 / 17 | −0.072% | −6.840% to 0.836% |
| Compact | 4 | 24 | 10 / 14 | −0.060% | −9.385% to 0.441% |
| Compact | 7 | 24 | 8 / 16 | −0.207% | −7.429% to 0.546% |

These are current GJXL gradient-versus-weighted comparisons. They do not measure
the overall BD-rate gap against stock libjxl. Weighted prediction changes the
lossless coding of already quantized coefficients, allowing exact pixel
equality to establish equal quality at each point.

## Complete encode and decode cost

All 18 preselected timing cases completed 20 alternating AB/BA pairs. The six
inputs comprise one CLIC photograph, one Kodak photograph, two 4K photographs,
and two compact controls, each at efforts 3/4/7 and distance 1. Encode samples
share one warmed process/backend and surround the entire uninstrumented public
call. Decoder samples include fresh single-thread decoder creation, allocation,
decode, and destruction. Every timed output matches its retained reference.

The 4K complete-encode results show the effort dependence most clearly:

| Input | Effort | Gradient median | Weighted median | Paired delta | Paired change | Slower pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Planter | 3 | 63.715 ms | 67.526 ms | +4.475 ms | +7.157% | 19/20 |
| Planter | 4 | 107.007 ms | 110.815 ms | +4.170 ms | +3.894% | 18/20 |
| Planter | 7 | 184.240 ms | 187.467 ms | +3.289 ms | +1.786% | 18/20 |
| Sun forest | 3 | 55.903 ms | 58.836 ms | +3.678 ms | +6.630% | 15/20 |
| Sun forest | 4 | 99.460 ms | 102.706 ms | +3.322 ms | +3.367% | 20/20 |
| Sun forest | 7 | 171.858 ms | 175.672 ms | +4.113 ms | +2.381% | 18/20 |

The selected CLIC photograph's paired encode changes are +4.161%, +3.109%, and
+2.493% for efforts 3/4/7, with 12/20, 17/20, and 20/20 slower pairs respectively.
Across the timed CLIC/4K cases, paired decode changes range from +1.689% to
+3.714%. Kodak and compact encode deltas are often small or mixed in sign;
their full samples and sign counts are retained in `timing.csv` and the raw run.

Paired medians are medians of within-pair differences/ratios, and need not equal
the difference/ratio of the standalone medians. The machine remained an
interactive Mac on AC power. Collection stopped and resumed around CPU spikes,
including VS Code renderer activity; ordinary background variability remains.
These local results support the rate/time tradeoff, not precise universal
latency guarantees. The earlier pilot's cross-process encoder timings are
excluded from these results.

## Engineering gates and evidence

The pinned 45-case weighted predictor oracle, native serializer storage and
allocation-failure checks, focused ASan/UBSan tests, workflow/API/CLI tests,
relocated C++20/C++23 installed consumers, and nine Rust native tests pass.
Baseline builds reproduce the pre-existing quantization golden mismatch and
the Metal initialization UBSan issue; neither was hidden or retuned for this
change. See [`../dc-prediction.md`](../dc-prediction.md) for exact boundaries.

Accepted raw evidence is `build/dc-weighted-qualification-v1`. It retains source
and build identity, input hashes, decoder libraries, all codestreams, command
logs, failed/paused attempts, machine snapshots, and timing samples. Portable
results are [`summary.json`](summary.json), [`size.csv`](size.csv),
[`size-summary.csv`](size-summary.csv), and [`timing.csv`](timing.csv).
The [collector README](../../tools/dc_coding/README.md) describes reproduction.

Prediction-aware DC quantization and adaptive DC smoothing have their own
[implementation and measured-quality qualification](../dc-processing-qualification/REPORT.md).
They change reconstruction; this lossless-prediction result does not qualify them.
