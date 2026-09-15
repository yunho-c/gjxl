# Resident quantizer dispatch qualification

For fields of at most 8,192 entries, the resident quantizer computes exact
median/MAD selection and quantizer finalization in one 256-thread group.
Eight private histograms and an integer SIMD prefix scan preserve the existing
floating-point arithmetic. Raw-quant generation remains a separate dispatch,
reducing the quantizer stage from 20 dispatches to two. Larger fields, SIMD widths
other than 32, and pipelines that cannot launch 256 threads retain the parallel
radix path. Profiling storage continues to reserve the parallel-path bound.

The cutoff keeps headroom below the measured crossover: all five synthetic
field patterns improved at 8,192 entries, while constant fields were nearly flat
at 16,384 and larger repetitive fields regressed.

Qualification on Apple M4 Pro compared the implementation with `d49d95d` at
freshly verified fast-SSIM2 80 +/- 0.1, using eight CPU participants. The balanced
run comprised 520 blocks and 1,560 timed complete encode calls, with three
warmups per block. Inputs were preloaded native linear-float pixels; file I/O
and metric computation were outside the timed call. All 130 matched outputs
were byte-identical between the two implementations.

| Corpus/path | Images | Geometric mean encode-time change | Faster images |
|---|---:|---:|---:|
| Kodak / new path | 24 | -1.85% | 21/24 |
| CLIC + Unsplash / fallback | 41 | +0.13% | 20/41 |
| Full corpus | 65 | -0.60% | 41/65 |

The preceding pilot measured a 1.91% Kodak reduction. Same-binary controls
ranged from -0.57% to +0.64% across four images. These results support a modest
warm-encode corpus benefit; they do not establish cold-start performance,
other-device performance, or current libjxl effort-6/7 speed parity.

All 390 rate-replay codestreams were byte-identical across 65 images and six
distances (0.55, 1.0, 1.9, 2.8, 4.6, 6.4), preserving rate and decoded quality.
The permanent oracle passes 672 guarded cases both normally and with Metal API
and shader validation. It compares CPU median/MAD, both GPU paths, quantizer
words, raw quant, error bits, scratch state, padding, and guards. Reconstruction,
quantization-pipeline, and completed-frame tests also pass with both validation
layers. Profiling tests cover launch shape and fields at and above the cutoff.

The final full suite passed 138/139 tests. The inherited `quantization_pipeline`
pinned-score failure remains: actual `0.24919039011001587`, expected
`0.24914586544036865`. An earlier AC-strategy failure did not recur in five
unchanged-binary reruns per arm, 50 instrumented runs per arm, or the final full
suite; its cause remains unexplained. The tested snapshots included a pre-existing
Python CLI fixture change, which is excluded from this commit.

The measured source snapshot and detailed ledgers are retained locally under
`build/e8-speed-quantizer-group-d49d95d-20260915/`: `group-v4`, `control-warm`,
`crossover`, `full65-v4`, `rate-v4`, `verification-v4`, and `integration`.
The retained detailed reports are `docs/e8-speed/quantizer-dispatch.{md,json}`.
