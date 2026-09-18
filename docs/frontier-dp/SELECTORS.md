# Selector policy and Metal comparison

This is an experimental checkpoint on `experiment/gpu-frontier-dp`, based on
`bb7b714e858ffe7668a84c8588745a99b7015f39`, Apple M4 Pro, macOS 15.6.
It extends the original frontier experiment in REPORT.md. Production defaults
are unchanged. The remaining resident integration is tracked in IMPLEMENTATION.md.

## Complete-encode rate-quality pilot

Retained run: `build/frontier/results/rate-pilot-20260917/`.
The manifest freezes the diagnostic binary, source copies, input PFM hashes,
decoder and metric tools. All 180 codestreams, decoded-image hashes, raw records,
commands, logs and metric observations are retained. Decoded PFMs are temporary.

Six images were selected before observing results: Kodak 01, 12, 24 and first,
middle, last lexically sorted CLIC 2024 test PFMs (32 available). Each was encoded
at efforts 5 and 8, distances 0.7, 1.2, 2, 4, 7, with CPU greedy, full frontier,
and rectangle DP. The two new policies consume the same frozen FP32 costs,
including the existing DCT8 discount. They compare exact integer objective sums
and preserve the original greedy map whenever it ties or beats their proposal.

These CPU diagnostic overrides establish policy effects; their encoder times do
not measure the prospective GPU implementation. All outputs decoded and scored.
SSIMULACRA2 uses the existing fast-ssim2 scorer; Butteraugli uses explicit linear
sRGB and intensity target 80. PCHIP interpolates log(bytes); Akima is a sensitivity
check. No extrapolation or nonmonotonic curve is accepted.

| Metric interval | Full frontier PCHIP / Akima | Rectangle + greedy PCHIP / Akima | Covered image-effort curves |
|---|---:|---:|---:|
| SSIMULACRA2 75–85 | −0.111% / −0.107% | −0.170% / −0.153% | 12 / 12 |
| SSIMULACRA2 60–80 | −0.304% / −0.278% | −0.231% / −0.216% | 10 / 12 |
| Butteraugli 1.5–4, logarithmic quality axis | −0.185% / −0.172% | −0.120% / −0.173% | 12 / 12 |

Each aggregate weights image-effort curves equally. Lower BD-rate is better.
The CLIC image `60430844748882ff66b9a5598ef06dd6` does not bracket score 60 in
either effort, and is excluded from the 60–80 row. These are exploratory
five-point curves on six images, not a corpus qualification or calibrated
point comparison. Full per-curve outcomes are in `analysis.json`.

The gains are small and not universal. At SSIMULACRA2 75–85, frontier wins 7 of
12 curves and rectangle wins 9 of 12; worst PCHIP regressions are +0.573% and
+0.856%. Butteraugli worst regressions are +2.785% and +5.058%. The proxy optimum
therefore does not warrant changing the normal selector policy by itself.

## Captured-cost GPU comparison

Retained run: `build/frontier/results/selector-comparison-20260917-v2/`.
This run reuses and hashes the original candidate captures, freezes the new
probe/shaders/sources, and runs three separate processes per image with three
warmups and seven samples per process. Image order rotates between processes;
threadgroup order rotates within processes. Figures below are medians of process
medians at a fixed thread count per policy: frontier 512, rectangle 256,
greedy 64, and rectangle + greedy 256. Preparation and required fallback
dispatches are included. Scoring, input upload, graph construction, AQ metadata
and the rest of the encoder are excluded.

The combined rectangle policy includes the cost and work of GPU greedy selection
and an exact objective comparison, retaining greedy on ties. Pure rectangle
timing alone does not represent the policy used by the rate-quality pilot.

| Input | GPU frontier | GPU rectangle | GPU greedy | GPU rectangle + greedy | Earlier CPU greedy control |
|---|---:|---:|---:|---:|---:|
| alpine_24mp | 9.606 ms | 1.843 ms | 0.865 ms | 2.690 ms | 9.105 ms |
| alpine_3mp | 1.245 ms | 0.337 ms | 0.423 ms | 0.776 ms | 1.214 ms |
| campus_24mp | 9.603 ms | 1.848 ms | 0.559 ms | 2.397 ms | 10.139 ms |
| forest_24mp | 9.814 ms | 2.788 ms | 0.734 ms | 3.654 ms | 11.359 ms |
| synthetic_4k | 3.007 ms | 0.571 ms | 0.497 ms | 1.070 ms | 3.001 ms |

All 20,606 captured tiles matched the production CPU greedy map and the exact
CPU frontier/rectangle oracles, including combined-policy decisions. None needed
the wide fallback. Rectangle + greedy loses some frontier proxy improvement on
only 78 tiles: 23 alpine 24MP, 4 alpine 3MP, 46 campus, 0 forest, 5 synthetic 4K.
It recovers over 99.4% of the full frontier improvement in every captured case.

Self-tests now cover 2,816 tiles across all 64 partial-tile dimensions, including
1,131 wide cases and 192 invalid-input cases, near ties, area ties, signed zero,
subnormals, extreme finite values, and deterministic random costs. Greedy is
checked against the actual production CPU selector, not another transcription.
Both ordinary optimized and Metal API/shader-validation runs pass at all four
thread counts. Guard regions, policy bits, normalization, costs and maps are
checked outside timing.

The full synthetic suite also runs with a nontrivial DCT8 discount, checking
correctly rounded binary32 multiplication at the normal/subnormal boundary.
The policy preparation uses integer multiplication and rounding in that small
range to preserve the CPU bits despite the GPU's subnormal behavior.
Portable summaries, provenance hashes and validation results are retained in
[evidence/](evidence/); complete logs and codestreams remain in the run directories.

Two implementation hazards were found. Metal FP32 arithmetic flushes subnormal
operands on this device, so the policy-preserving greedy sum uses integer
round-to-nearest/ties-to-even addition when a nonzero operand is subnormal.
Ordinary values use hardware FP32 addition in CPU order. Also, inlining the
mutating greedy helpers produced incorrect maps in ordinary kernels while
shader validation masked the failure. Keeping these helpers out of line passes
both builds. This is an observed compiler-sensitive workaround; the underlying
compiler defect has not been isolated. Future toolchains must retain both checks.

## Implementation decision

Start resident integration with the existing greedy policy on GPU. It offers
substantial selection-time savings while preserving the status-quo map on the
tested inputs. Keep rectangle + greedy as an explicit experimental alternative;
it is much cheaper than full frontier search and retains nearly all captured
proxy benefit. Keep full frontier as the correctness reference. No end-to-end
speedup, production numerical guarantee, or default rate improvement is claimed
at this checkpoint. Effort 10's dense bank remains outside the prototype.

Reproduce after configuring `GJXL_BUILD_FRONTIER_EXPERIMENT=ON`:

```sh
cmake --build build/frontier --target gjxl_quality_benchmark gjxl_frontier_probe
python3 tools/frontier_dp/rate_experiment.py --output <new-rate-directory>
python3 tools/frontier_dp/analyze_rate.py <rate-directory>
python3 tools/frontier_dp/compare_selectors.py \
  --captures build/frontier/results/kernel-probe-20260917 \
  --output <new-kernel-directory>
```

Rate collection is resumable with verified frozen inputs. Kernel comparison
requires a fresh directory. The diagnostic environment variable is
`GJXL_AC_SEARCH_EXPERIMENT=greedy|frontier|rectangle`; it exists only in the opt-in
experimental build. `GJXL_FRONTIER_CAPTURE_DIR` separately enables candidate
capture and intentionally adds CPU-control repetitions, so it must be unset for
ordinary timing or rate runs.
