# Shared DCT basis in fused Metal AC candidates

Shared basis staging on the measured M4 Pro reduces the three affected 4K AC
stages by about 19% combined and improves the complete encode by 1.24% on
Planter 4K and 1.37% on padded 4K. Both whole-call
comparisons improve in six of seven independent pairs. Outputs remain exact.

This investigation is based on `264460c128cd087800fc0dea3657d3c8ed7ba5ba`,
measured September 8, 2026 on Apple M4 Pro / Mac16,7, 48 GiB, macOS 15.6,
Apple Clang 17, Release C++20/libc++. The qualified shader change is implemented
in the production fused kernels. The original experiment and complete timing
evidence remain frozen under `build/shared-dct-basis/`.

The lead came from the earlier
[performance-limiter analysis](../logs/metal-profile/20260908T191457Z-padded_4k-fully-resident-6e39e381cbd6/limiter-analysis/README.md).
That capture predates the current CPU serializer and coefficient-population
improvements. All timings below use the current qualified runtime, including
the population kernel. The old occupancy/limiter counters motivated the
experiment; they are not new measurements of the optimized shader.

## Implementation and synchronization

Only `src/gpu/metal/kernels/ac_strategy.metal` changes at runtime. The affected
entry points
are `gjxl_ac_strategy_dct16_candidate_loss_parallel`,
`gjxl_ac_strategy_dct16x8_candidate_loss_parallel`, and
`gjxl_ac_strategy_dct8x16_candidate_loss_parallel`. Shape names here follow the
shader's rows × columns convention.

The original kernels allocate a basis for each of X/Y/B and stage it in both
the forward and inverse helpers. The new kernels allocate one square basis, or
one vertical/horizontal pair, for the entire candidate threadgroup:

1. The wrapper assigns basis element `i` to global thread `tid`, advancing by
   `3 * Workers`. Each element has exactly one writer across the three channels.
2. Forward helpers retain their existing pixel gathering. Their existing
   unconditional threadgroup barrier publishes both pixels and the shared basis.
3. The basis stays read-only through all three channels' forward transforms,
   residual calculation and inverse transforms. No inverse restaging occurs.
4. The inverse helper's existing barrier remains: it also publishes residual
   coefficients and completes rate/magnitude use before magnitude storage is
   reused for inverse pixels. Removing basis stores does not remove that
   synchronization obligation.

A compile-time `StageBasis` parameter defaults to true for existing split
helpers. Only the three fused wrappers select false after staging their shared
bases. Matrix loads, multiply/accumulate order, scaling, coefficient layout,
cross-channel Y publication, magnitude/loss trees and final cost calculation
retain their original arithmetic and barriers. All recorded dispatch shapes
and inventories match the baseline. This experiment does **not** include the
suggested shuffle reduction or color-correlation changes.

This is coordinated staging, not simultaneous writes through aliased channel
pointers. At source level, six logical basis copies (three channels, twice)
become one. Compiler elimination and physical cache traffic were not measured,
so that is not a claim of sixfold measured basis bandwidth reduction.

## Queried threadgroup storage

These are actual `staticThreadgroupMemoryLength()` results from the compiled
pipelines, matching the source-level prediction:

| Fused family | Threads/group | Baseline bytes | Shared-basis bytes | Reduction |
| --- | ---: | ---: | ---: | ---: |
| DCT16 | 192 | 15,360 | 13,312 | 13.33% |
| 16×8 | 192 | 9,984 | 7,424 | 25.64% |
| 8×16 | 96 | 9,984 | 7,424 | 25.64% |

The per-channel coefficient, residual, magnitude and nonzero arrays stay
separate. No device-buffer allocation, resource plan, submission count or
handoff changes. These are shader threadgroup-capacity savings, not process
memory or RSS measurements.

## GPU stage measurements

The first screen used three alternating process pairs on each of two 4K inputs.
A separate confirmation used five pairs each for padded 4K, Planter 4K,
padded 1080p and synthetic 128×96. Every process used three warmups and seven
measured calls. Timings use supported GPU stage timestamps; dispatch-boundary
timestamps are unavailable on this device. Each reported AC stage includes
the existing candidate kernel and cost finalizer.

The five-pair padded-4K confirmation:

| GPU stage | Baseline ms | Shared-basis ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| DCT16 candidates | 7.191 | 6.086 | −15.27% | 5/5 |
| 16×8 candidates | 4.362 | 3.806 | −12.75% | 5/5 |
| 8×16 candidates | 5.088 | 3.508 | −31.06% | 5/5 |
| Three affected stages combined | 16.846 | 13.615 | −19.16% | 5/5 |
| All AC stages | 36.687 | 33.402 | −8.95% | 5/5 |

Planter 4K gives −15.22%, −12.79% and −31.44% for the same three families.
Its combined affected stages fall from 16.742 to 13.623 ms (−18.71% paired),
and all AC stages improve 8.99%. Every affected stage improves in all five
pairs. Padded 1080p shows the same direction: −15.25%, −12.48%, −31.11%.

Grouped durations are summed **within each sample**, then reduced to process
medians. They need not equal sums of the displayed individual-stage medians.
The independent first screen showed similar 4K results: roughly −15%, −13%,
and −31%, respectively.

The actual storage reduction and repeated speedup support the original lead.
They do not identify how much comes from occupancy, cache behavior or removed
staging work. No new occupancy or thread-launch-limiter capture was collected,
and this experiment does not isolate shared capacity from inverse basis reuse.

## Complete workflow

Seven alternating independent-process pairs per input, three warmups and seven
measured calls per process. The **benchmark executable is byte-identical on
both sides**; `--metallib` selects the baseline or candidate library. This holds
host code and executable layout constant. The complete candidate build embeds
the exact same candidate metallib and is independently used for correctness.

Policy is fully-resident Metal, SIMD DCT, fused-tuned AC, distance 1.2, effort 7,
without a requested final score. The boundary is the benchmark's public encoding
workflow, excluding file I/O and process/backend startup. GPU stage profiling,
Metal validation and sanitizers are disabled for this campaign.

| Input | Baseline ms | Shared-basis ms | Paired change | Faster pairs | Pair range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic 128×96 | 10.782 | 10.545 | −1.91% | 4/7 | −12.15% to +19.38% |
| Padded 1919×1079 | 64.744 | 63.663 | −1.68% | 6/7 | −2.47% to +1.43% |
| Padded 3839×2159 | 214.632 | 211.272 | −1.37% | 6/7 | −2.81% to +0.23% |
| Kodak 17, 512×768 | 20.646 | 20.258 | −1.23% | 7/7 | −5.01% to −0.97% |
| Planter 1920×1080 | 69.824 | 68.991 | −1.08% | 7/7 | −2.37% to −0.39% |
| Planter 3840×2160 | 217.706 | 214.418 | −1.24% | 6/7 | −2.57% to +0.45% |

Milliseconds are medians of process medians. Percentages are medians of the
per-pair ratios; they need not equal the ratio of separately displayed times.
Negative means faster. Ranges are observed variation, not confidence intervals.
The tiny synthetic case is too noisy to establish a whole-call benefit. The
other five inputs show consistent modest improvements, with especially strong
and repeatable evidence at the affected GPU-stage boundary. These are warm
M4 Pro results, not cold-start or multi-device qualification.

## Correctness

- An expanded guarded probe compares the current fused kernels directly with
  the shared-basis versions: **900/900 cases**, repeated **900/900 under Metal
  API and shader validation**. It covers all three shapes, candidate counts
  1/2/7/33/257, contiguous and padded strides, candidate/precomputed/forward
  quant norms, random/constant/impulse/checkerboard/large/nonfinite inputs,
  invalid masks, invalid CfL, out-of-range anchors and invalid quantization.
  Quant norms, channel rates, losses and final costs match bit-for-bit. Input
  and output guards remain intact; coefficient scratch stays poisoned.
- The complete Release build passes **123/124 tests**. The sole failure is the
  inherited `quantization_pipeline` fixture: index 1, actual
  `0.24919039011001587`, expected `0.24914586544036865`.
- `metal_ac_strategy`, `metal_ac_strategy_search` and `metal_aq_evaluation`
  pass with Metal API and shader validation enabled. Existing AQ coverage
  exercises retained results, reuse, trimming, errors and concurrent callers.
- All **56 corpus/policy pairs** are byte-identical. Coverage includes the
  38 canonical inputs, efforts 1–10, maximum compression, high density,
  final-score requests, alternate Metal AQ policies, rate control and a CPU
  fallback case. Three decoded pairs match using pinned `djxl`
  `e8ff09762481785938d8e4e01333ed3917571161`.

The final audit checks the frozen runtime, unchanged original sources and
preexisting dirty files, actual dispatch geometry, raw samples, output hashes,
and non-overlapping timing/qualification intervals. No build, test, validation
or other benchmark ran alongside timed processes. The campaigns contain
136 timing processes and 952 measured calls: 364 instrumented stage calls and
588 ordinary whole-workflow calls.

## Artifacts and reproduction

Local evidence is under `build/shared-dct-basis/`:

- `prepare.py`, `kernels/`, `candidate.patch`, `commands.json`, `identity.json`:
  the exact one-file shader change, guarded probe generation, compiler/linker
  commands and frozen identities. The patch passes `git apply --check` at the
  pinned revision. Existing barriers remain in the same order.
- `candidate/`: copied baseline benchmark plus the changed metallib, used for
  timing. Its library SHA-256 is
  `e67910c3bedc3866d61bc23daa4ea6a6daecfb5b2976d7b9d4659569293fff9c`.
- `src/`, `release/`, `full-build-commands.json`, build/configure logs: isolated
  complete build from the pinned Git source with the shader patch and the same
  Metal C++ submodule. Its metallib is byte-identical to `candidate/`.
- `probe.cpp`, `probe.log`, `probe-validation.log`: guarded comparisons and
  queried pipeline storage. The probe extends the existing repository probe;
  it also sets the square fixture's block count to four.
- `qualify.py`, `qualification-commands.json`, `ctest.xml`,
  `metal-validation.xml`, their logs, `parity/`,
  `qualified-build-identity.json`: qualification commands and results.
- `stage-screen/`, `stage-confirm/`, `wall/`, their logs, `timing.py`,
  `timing-commands.json`: raw samples, exact commands, input/runtime hashes,
  timestamps, process medians and paired results.
- `analyze.py`, `analysis.json`, `evidence-hashes.json`: reproducible grouped
  summaries and evidence audit. The three changed stages are summed per sample.
- `commit/`: promotion checks, the original investigation report and preserved
  file hashes. Comments and macro formatting were cleaned up during promotion;
  the rebuilt metallib remains byte-identical to the measured candidate.

The expanded `benchmarks/metal_ac_candidate_probe.cpp` is retained in the
repository. It accepts baseline and candidate metallib paths and runs 900 cases
against the split implementation by default. Add `--fused-baseline` to compare
two fused implementations directly. Both modes pass after promotion, and the
fused comparison also passes with Metal API/shader validation enabled. These
checks use the promoted probe and the exact measured shader library, so the
previous full-suite and timing evidence still applies without another timing
campaign.

Use a fresh artifact directory for a rerun and preserve the original frozen
builds/results. `prepare.py` builds the shader-only candidate and probe against
`build/coefficient-population/final`. The complete source snapshot comes from
`git archive 264460c`, the pinned Metal C++ submodule, and `candidate.patch`;
replay `full-build-commands.json` with fresh source/build paths. Direct compiler
commands use `SDKROOT` from `xcrun --show-sdk-path`.

Run `qualify.py`, then `timing.py` sequentially on a quiet machine. The separate
first screen uses `tools/metal_dataflow/compare.py` with the same builds,
`--workload padded_4k`, the canonical Planter 4K input,
`--pairs 3 --warmups 3 --samples 7 --profile`; each record retains its exact
command. Finally run `analyze.py` to verify identities and recompute the report
data. Scripts using existing timing directories resume frozen records; use new
directories to collect new samples. The frozen scripts' source/revision guards
target the original `264460c` checkout; use a matching isolated checkout to
reproduce that investigation. `commit/verify.py` records the promoted source's
correspondence to its qualified and measured shader artifact.
