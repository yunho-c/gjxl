# Shuffle tails for the four remaining Metal AC strategies

Both magnitude/count and loss shuffle tails are selected for DCT8, DCT32,
32×16 and 16×32. The four affected GPU stages improve
11.39% combined on padded 4K and
11.38% on Planter 4K. Whole-encode paired changes are
-0.75% and
-1.57%, respectively.
The Planter-4K whole-call estimate is noisy and is not a reliable improvement
claim. All 56 corpus/policy comparisons remain byte-identical.

Baseline: `fcb1c644075c4ef777852d542f839c796ac2a74c`, including shared DCT bases
and [shuffle tails in the three complete fused candidates](metal-ac-shuffle-tail.md).
These measurements are incremental to that baseline. They do not include the
previous three-shape improvement or any new channel grouping or basis sharing.

Measured September 8, 2026 on Apple M4 Pro / Mac16,7, 20 GPU cores, 48 GiB,
macOS 15.6, Apple Clang 17, Release C++20/libc++.

## Scope and arithmetic

The four remaining families already have specialized forward and inverse DCTs.
They retain three dispatches: forward DCT, residual/inverse/weighted loss,
then the cross-channel cost finalizer. Channels remain in separate threadgroups,
with device coefficients between forward and residual/inverse. The experiment
changes only the reductions within their existing `fused-tuned` loss kernels.

The shared reduction helpers are the same ones qualified for the three complete
fused candidates. New compile-time macro arguments independently select the
magnitude/count and loss tails for each staged kernel. The four selected entry
points are `gjxl_ac_strategy_dct8_residual_inverse_compact_loss`,
`gjxl_ac_strategy_dct32_residual_inverse_tuned_loss`,
`gjxl_ac_strategy_dct32x16_residual_inverse_compact_loss` and
`gjxl_ac_strategy_dct16x32_residual_inverse_tuned_loss`.

The DCT16, 16×8 and 8×16 staged fallback kernels keep both flags false. Their
complete candidate kernels retain the previous optimization. Other inverse
modes retain their current code. No new pipeline entry points, host scheduling
changes, coefficient layouts, basis sharing or final-cost changes are added.

The upper shared-memory tree runs through stride 32 and its barrier. The first
32 lanes then use shuffle-down offsets 16, 8, 4, 2, 1, preserving the original
tree that feeds lane zero. Every shuffle source lane remains active. There is
no generic SIMD sum or reassociation of the per-coefficient arithmetic.
The tail writes the completed channel rate or loss directly to its output.

Ten reduction-loop barriers are removed per channel threadgroup, five from
each tree. Since these paths have three channel threadgroups per candidate,
that is 30 removed barrier executions across the candidate's groups. DCT8's
two reduction loops drop from 12 barriers to 2, DCT32's from 20 to 10, and
each rectangle's from 18 to 8. The surrounding publication and scratch-reuse
barriers remain, including the inverse helper's unconditional entry barrier.

## Resources and launches

The guarded probe queries static memory; dynamic storage is the three existing
`setThreadgroupMemoryLength` allocations. Both libraries report identical
storage. The recorded stage dispatches confirm the unchanged launch sizes.

| Strategy | Forward threads | Residual/inverse/loss threads | Static bytes | Dynamic bytes | Total bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| DCT8 | 32 | 32 | 256 | 768 | 1,024 |
| DCT32 | 128 | 512 | 4,096 | 12,288 | 16,384 |
| 32×16 | 128 | 128 | 5,120 | 6,144 | 11,264 |
| 16×32 | 64 | 256 | 5,120 | 6,144 | 11,264 |

The wider tuned launches remain intact. Full scratch arrays still serve the
upper reduction tree and inverse/loss stages, so this is not an allocation or
RSS reduction. No new occupancy or hardware-limiter capture was taken.

## Separate-tail screen

Magnitude/count alone, loss alone and both each passed 1,200 guarded cases
normally and another 1,200 under Metal API/shader validation before timing.
The screen uses three alternating process pairs at padded 1080p and padded 4K;
each process has three warmups and seven measured calls. Padded-4K paired changes:

| Variant | DCT8 | DCT32 | 32×16 | 16×32 | Four combined | All AC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Magnitude/count | -10.23% | -6.03% | -8.58% | -7.90% | -7.83% | -5.00% |
| Loss | -5.11% | -4.53% | -3.02% | -4.20% | -4.14% | -2.56% |
| Both | -12.00% | -9.95% | -11.88% | -12.38% | -11.35% | -7.25% |

Both tails give the best result for every shape at both resolutions. The
combined variant was selected for fresh Release qualification and a separate
five-pair stage confirmation.

## GPU stage confirmation

Five alternating process pairs per input, three warmups and seven measured
calls per process. Supported GPU stage timestamps cover all three dispatches
for each family, including its unchanged forward DCT and cost finalizer.
Individual dispatch timestamps are unavailable on this device.

| Padded-4K stage | Baseline ms | Shuffle ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| DCT8 | 2.805 | 2.470 | -11.87% | 5/5 |
| DCT32 | 6.378 | 5.740 | -10.04% | 5/5 |
| 32×16 | 5.169 | 4.560 | -11.88% | 5/5 |
| 16×32 | 5.113 | 4.480 | -12.36% | 5/5 |
| Four affected stages combined | 19.476 | 17.258 | -11.39% | 5/5 |
| All AC stages | 30.590 | 28.352 | -7.34% | 5/5 |

Planter 4K's affected total changes from 19.459 to
17.249 ms (-11.38% paired).
Padded 1080p and synthetic 128×96 are also included in the raw confirmation.
Dispatch inventories, entry points and geometry match across all variants.

## Whole workflow

Eleven alternating process pairs per input, three warmups and seven measured
calls per process. The larger pair count was chosen before this campaign,
because the preceding three-shape experiment showed substantial whole-call
variability. Every pair is retained; stage and whole-call campaigns are separate.

The two sides use the identical frozen benchmark executable, differing only
in the explicit `--metallib` path. The shader library matches the fresh Release
build used for qualification. GPU profiling and Metal API/shader validation
are disabled during whole-workflow timing.

The boundary is `metal-public-workflow`, fully resident AQ, SIMD, fused-tuned
AC, distance 1.2, effort 7, final score disabled. It includes input preparation,
backend selection, quantization, codestream encoding and summary assembly.
These are warm workflow measurements; file I/O and cold startup are outside
the measured boundary.

| Input | Baseline ms | Shuffle ms | Paired change | Faster pairs | Pair range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic 128×96 | 9.979 | 10.049 | -2.89% | 7/11 | -9.43% to +11.94% |
| Padded 1919×1079 | 62.133 | 61.869 | -1.13% | 8/11 | -4.79% to +2.27% |
| Padded 3839×2159 | 209.665 | 207.312 | -0.75% | 9/11 | -4.17% to +0.95% |
| Kodak 17 | 20.180 | 20.309 | -0.13% | 7/11 | -4.15% to +2.19% |
| Planter 1080p | 69.839 | 69.312 | -1.52% | 9/11 | -5.51% to +0.74% |
| Planter 4K | 246.206 | 243.768 | -1.57% | 6/11 | -13.10% to +7.47% |

The strongest whole-workflow observations are padded 4K and the two 1080p
inputs. Padded 4K and Planter 1080p improve in 9/11 pairs; padded 1080p improves
in 8/11. Kodak 17 is effectively flat. Planter 4K improves in only 6/11 pairs,
with a -13.10% to +7.47% pair range, so its negative median does not establish
a repeatable whole-encode benefit. The tiny input is also noisy; no reliable
tiny-image speedup is claimed. Every original pair is retained without trimming
or a further timing rerun to seek a more favorable result.

The combined tails are retained for the consistent larger-image GPU-stage
reduction, exact output parity and modest whole-encode gains on the padded
and 1080p workloads. They add no static/dynamic threadgroup storage or dispatches.

Times are medians of process medians. Changes are medians of paired relative
changes, so they need not equal the ratio of displayed times. Grouped GPU times
are summed within each sample before taking medians. Pair ranges show observed
spread, not confidence intervals. Stage gains must not be presented as equal
whole-encode gains. Results are specific to this device, compiler and workload.

## Correctness and retained evidence

- Three variants × 1,200 normal and 1,200 API/shader-validation cases: 7,200
  guarded cases. Comparisons include coefficients, quant norms, channel rates,
  loss and final costs, guards and untouched buffer padding.
- The existing default split-oracle probe, fused-baseline probe and validated
  fused-baseline probe each pass 900 cases on the three earlier shapes: 2,700
  compatibility cases. Their invocation modes and defaults are preserved.
- The expanded probe built by the fresh Release build passes another 1,200
  staged cases. Overall: 11,100 guarded cases across the qualification runs.
- New mode: `gjxl_metal_ac_candidate_probe BASELINE.metallib CANDIDATE.metallib
  --staged-reductions`. It covers four shapes, five batch sizes, ten patterns,
  three quant-norm sources and two row paddings. It explicitly preserves the
  tuned 512-thread DCT32 and 256-thread 16×32 inverse launches.
- Fresh Release: 123/124, with only the inherited `quantization_pipeline`
  mismatch at index 1, actual `0.24919039011001587` versus expected
  `0.24914586544036865`. A fresh baseline run reproduces the identical values.
- The three focused Metal suites pass under API/shader validation.
- All 56 corpus/policy encodes match byte for byte, covering 38 canonical
  images, efforts 1–10, high density, maximum compression, final score,
  alternate AQ, rate control and CPU maximum-error fallback. Three decoded
  pairs match using pinned `djxl` revision `e8ff0976`.
- Five timing campaigns: 208 processes and 1,456 measured calls. Runtime and
  raw-output hashes, nonoverlapping intervals and unchanged dispatch geometry
  are audited. Qualification runs do not overlap timing.

Frozen sources, libraries, patches, probes and raw evidence remain under
`build/ac-shuffle-remaining/`. `prepare.py`, `screen.py`, `qualify.py`,
`confirm.py`, `analyze.py` and `verify.py` record the construction and checks.
`analysis.json` contains all paired results; `verification.json` records the
final source/report qualification. Preparation requires the guarded baseline
and refuses to overwrite existing variant directories.

Selected metallib SHA-256:
`c7957b7fcb49e6700685dc8a5611dd7fe735d648a6973c6913bda34e2c38730d`.
After production promotion and formatting cleanup, the compiled library is
byte-identical to this qualified and timed artifact. The diagnostic probe
source matches the one used in the fresh Release build.

To repeat timing with the frozen artifacts, use a new output directory:

```sh
python3 tools/metal_dataflow/compare.py \
  --baseline-build build/ac-shuffle-remaining/baseline \
  --candidate-build build/ac-shuffle-remaining/both \
  --workload padded_4k --pairs 11 --warmups 3 --samples 7 \
  --output build/ac-shuffle-remaining-repeat/wall
```

Add `--profile` with a separate output directory for GPU stage measurements.
