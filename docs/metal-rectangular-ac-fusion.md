# Small rectangular Metal AC candidate fusion

The tuned 16x8 and 8x16 AC candidate paths retain X/Y/B coefficients in
threadgroup memory through residual, inverse DCT and loss production. This
removes one dispatch and the intervening device coefficient traffic per batch.
The scalar cost finalizer, arithmetic order, reduction trees and CPU merge/tie
decisions are unchanged. Each optional pipeline is selected independently on
Apple9-family devices after checking SIMD width, thread count and memory limits;
the existing split implementation remains the fallback.

The 16x8 kernel uses 192 threads and the 8x16 kernel uses 96. Each reports
9,984 bytes of static threadgroup memory on the measured device. Larger AC
families still require coefficient scratch, so this does not reduce the overall
scratch allocation. Two additional pipelines also add setup and library cost.

## Isolated fusion screen

Baseline: `473ad7a482305d091bad8e01420876139e4c9237`. The candidate contains
rectangular fusion without the separate fine Butteraugli profiling changes.
Release builds, Apple M4 Pro (20 GPU cores, 48 GiB), macOS 15.6, Xcode 26.3;
fully-resident Metal, distance 1.2, effort 7.

| Boundary and workload | Baseline ms | Fusion ms | Paired change |
| --- | ---: | ---: | ---: |
| 16x8 AC stage, padded 4K | 5.404 | 4.341 | -19.67% |
| 8x16 AC stage, padded 4K | 5.491 | 5.046 | -8.29% |
| All AC stages, padded 4K | 37.632 | 36.066 | -4.25% |
| Whole call, padded 4K | 219.602 | 217.713 | -1.38% |
| Whole call, padded 1080p | 66.826 | 66.717 | -0.23% |
| Whole call, synthetic 128x96 | 9.146 | 9.174 | +0.65% |

Stage measurements use three alternating process pairs, two warmups and three
samples per process. Separate whole-call measurements use five alternating
pairs, three warmups and seven samples. Times are medians of process medians;
changes are medians of paired ratios. Negative changes mean less time. The
whole-call boundary includes CPU codestream encoding and excludes backend setup
and input loading. Tiny-image measurements are noisy; no tiny-image win is claimed.

## Correctness and reproduction

The isolated Release build passed `metal_ac_strategy`,
`metal_ac_strategy_search` and `metal_aq_evaluation`. The guarded AC probe passed
320 cases both normally and under Metal API/shader validation. It compares
quantization norms, channel rates, loss sums and final costs bitwise against the
split path, checks guards and verifies that fused coefficient scratch is
untouched. Cases cover both shapes, five batch sizes, eight input patterns, two
quantization sources and two stride paddings. Three canonical image encodes
(planter 4K, Kodak 17 and padded stress 4K) matched the baseline byte for byte.
These are focused checks, not a claim that the complete Release suite is green.

Build `gjxl_metal_ac_candidate_probe` with `GJXL_BUILD_BENCHMARKS=ON`, then run:

```sh
CANDIDATE_BUILD/gjxl_metal_ac_candidate_probe BASELINE.metallib CANDIDATE.metallib
```

`tools/metal_dataflow/qualify.py --ac-candidate` integrates the normal and
validated probe runs with the existing qualification workflow. The local raw
records remain in `build/kernel-tuning/evidence/rect-stage`, `rect-wall`,
`rect-parity-screen`, `rect-tests.log`, `rect-ac-probe.log` and
`rect-ac-probe-validation.log`. The stage/wall identity files record executable
and metallib hashes; `rect-commands.json` records commands and environments.
Frozen source and build exports remain in `build/kernel-tuning/rect-src` and
`build/kernel-tuning/rect`.
