# Metal AC scratch sizing and complete candidate fusion

On the measured Apple M4 Pro, effort-7 fully resident 4K encoding improves by
**21.0–22.7 ms (10.2–10.8%)** against the original allocation/kernel combination.
Both 4K inputs improve in all six paired whole-call comparisons. Correcting
allocation sizes without changing kernels accounts for **13.5–13.7 ms** of that
improvement. These are paired statistics, not differences between unrelated
historical captures.

Base revision: `4db0bf7342e26fcf09f6cddaba30de2a0f4ca0bc`. Worktree:
`/Users/yunhocho/GitHub/gjxl-metal-ac-scratch`, branch `perf/metal-ac-scratch`.
Measured September 11, 2026. Raw evidence and reproducible scripts are retained
under `build/ac-scratch/` in this worktree.

## Implementation

`GpuAcStrategyEvaluation::GetAcStrategyScratchRequirements` reports the scratch
needed by the selected backend pipelines. Its default preserves the original
full coefficient-sized ranges. Metal reports scalar channel losses for A when
its selected forward and inverse/loss kernels support them, and zero B when the
complete candidate kernel is available. Validation uses the same query and
ignores an unused B pointer. Search planning takes independent maxima for A/B;
zero-sized scratch is released instead of allocated.

The backend-independent storage plan remains a conservative admission bound.
Passing a backend produces the actual fresh allocation requests. This change
does not make the earlier backend-independent admission reservation tighter.
Scalar, factored, split, compact, and missing-optional-pipeline paths retain the
storage their actual kernels need. Prepared buffers still reuse sufficient
capacity; the ordinary public workflow retains its existing release behavior.

Four additional optional Apple-family-9 pipelines complete candidate fusion for
8×8, 32×16, 16×32, and 32×32. The previous 16×16, 16×8, and 8×16 fused kernels
remain. Selection checks SIMD width, thread count, and static threadgroup memory.

The new kernels preserve Metal's SIMD matrix transforms, quantization arithmetic,
loss reduction tree, and CPU placement/tie rules. Two three-channel arrays have
successive roles: forward coefficients become residual coefficients, while
forward pixels become magnitude-reduction storage and then inverse pixels.
Residuals stay in per-lane registers until every channel finishes reading Y.
Integer warp counts use cells in the upper magnitude half only after that half
is dead. Unconditional barriers protect Y reads, reduction storage reuse, and
inverse output publication. No separate Y copy or DCT factorization change is
needed.

Queried static storage is 1,792 bytes / 96 threads for 8×8; 17,408 bytes /
384 threads for 32×16; 17,408 bytes / 192 threads for 16×32; and **28,672 bytes /
384 threads for 32×32**, below this device's 32,768-byte limit. Shape names follow
the shader's rows-by-columns convention.

An initial variant reduced magnitudes entirely in registers. The guarded probe
found a one-ULP rate mismatch for a large-magnitude 8×8 case. That variant was
rejected; the retained implementation preserves the shared-memory magnitude
reduction while reusing its backing. The failed shader and diagnostic log remain
in `fusion-register-rates-v1.metal` and `evidence/fusion-probe-detail.log`.

## Storage at 3840×2160

These are requested device-buffer sizes, not measurements of process RSS.

| Storage | Original | Sizing only | Sizing plus fusion |
| --- | ---: | ---: | ---: |
| Scratch A | 290.391 MiB | 1.483 MiB | 1.483 MiB |
| Scratch B | 290.391 MiB | 213.047 MiB | 0 |
| A + B | 580.781 MiB | 214.530 MiB | 1.483 MiB |
| Channel rates | 2.966 MiB | 2.966 MiB | 2.966 MiB |

Final resident AC device requests including descriptors, matrices, costs and
rate scratch total **19,347,936 bytes (18.452 MiB)**. Other encoding stages are
outside this number. The metallib grows from 2,311,148 to 2,398,716 bytes
(+87,568 bytes); four additional pipelines add preparation work. Backend setup
is outside the timed encoding boundary below.

## Measurement design

Release, fully resident Metal, SIMD implementation, fused-tuned AC, effort 7,
distance 1.2, automatic CPU thread budget, no requested final score. The machine
was on AC power and reported no thermal/performance warning. The normal desktop
remained running; process snapshots are retained with timing records.

The primary comparison uses one **byte-identical executable** for all three
arms. Original and sizing-only arms use the same frozen original metallib; the
fusion arm uses the new metallib. A diagnostic-only scratch override recreates
the original allocation contract. That override and the AC timestamp logging
are absent from the retained production source. Their exact experimental patch,
binaries, library hashes, commands, and logs are retained.

Six independent process rounds cover all six arm orders. Workload/mode order is
randomized with a saved seed. Each process has three warmups and seven measured
encodes, following the benchmark's validation encode. The public-call boundary
includes CPU codestream work and excludes input loading, filesystem output,
process startup and backend construction. All samples and pairs are retained.
No build, test, or other study job overlaps a timing process.

Whole-call timing has logging and GPU profiling disabled. Separate ordinary
submissions with timestamp logging measure the driver interval
`kernelEndTime - kernelStartTime` and command-buffer GPU execution. These are
separate campaigns; their medians are not additive. The earlier one-round
`sizing-screen` was noisier and is not used for the retained speedup claim.

| Whole-call workload | Original median ms | Sizing median ms | Fusion median ms | Sizing paired delta | Fusion paired delta | Fusion faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Padded 4K | 210.625 | 196.353 | 187.666 | −13.723 ms / −6.61% | −22.688 ms / −10.79% | 6/6 |
| Planter 4K | 212.672 | 198.486 | 188.567 | −13.483 ms / −6.33% | −21.008 ms / −10.19% | 6/6 |
| Padded 1080p | 63.680 | 59.834 | 58.116 | −3.791 ms / −5.94% | −5.669 ms / −8.90% | 6/6 |
| Kodak 17 | 20.181 | 19.932 | 19.306 | −0.167 ms / −0.83% | −0.832 ms / −4.12% | 6/6 |

Medians are medians of process medians; deltas and percentages are medians of
within-round paired differences/ratios, so the columns need not subtract.
Fusion also beats sizing alone in all six pairs for each of these four inputs.
The 128×96 case is noisy: fusion versus sizing has a +0.007 ms paired median,
three faster pairs of six, and a −1.75 to +2.21 ms paired range. No tiny-image
speedup is claimed.

| 4K diagnostic boundary | Original | Sizing only | Fusion |
| --- | ---: | ---: | ---: |
| Padded driver interval | 17.268 ms | 6.967 ms | 0.652 ms |
| Planter driver interval | 17.923 ms | 7.203 ms | 0.652 ms |
| Padded AC GPU execution | 27.176 ms | 27.126 ms | 24.715 ms |
| Planter AC GPU execution | 27.191 ms | 27.196 ms | 24.941 ms |

Allocation sizing leaves GPU work essentially unchanged. Fusion removes almost
all of the driver preparation interval and also improves total AC GPU execution.
The driver baseline here is lower than the earlier approximately 25 ms capture;
only contemporaneous paired values support the savings above.

A separate three-round, two-warmup/three-sample stage capture shows why individual
kernel timings should not decide the allocation-aware combination. Against
sizing-only, the paired 4K stage changes are −0.777 ms for 8×8, −1.386 ms for
32×16, −0.451 ms for 16×32, and **+0.331 ms for 32×32**. The complete combination
wins despite the small 32×32 GPU-stage regression and eliminates B entirely.
Unchanged families' small variations are not credited as improvements.

## Qualification and artifacts

- Sizing-only Release suite: 126/127 pass. The sole CPU `quantization_pipeline`
  failure is reproduced on the frozen baseline with the identical score and
  difference (`0.24919039011001587` versus `0.24914586544036865`).
- Minimum-sized scratch executes across scalar, SIMD tuned/compact/wide/split,
  and factored modes. One-byte-short A/B fails before submission; empty/overflow
  query contracts and exact tuned-versus-wide costs are checked.
- The four-family fusion probe passes **1,200/1,200 bitwise guarded cases**,
  repeated under Metal API and shader validation. It checks quant norms, channel
  rates, losses, final costs, buffer guards and untouched coefficient scratch;
  coverage includes batch tails, padded strides, three norm sources, invalid
  geometry/masks/CfL, nonfinite data and large inputs.
- Metal AC, search, and AQ tests pass with both validation layers enabled.
- All **38 corpus images** produce byte-identical codestreams for original versus
  sizing-only and original versus fusion. These are exact-byte comparisons;
  no new independent decoder run is claimed.
- The final production-source suite initially passes 125/127. Updating the CLI
  test's expected dispatch names/counts for the four new fused kernels makes
  that test pass on rerun, leaving only the identical inherited CPU fixture
  failure. See `evidence/final-tests.log`, `evidence/final-cli-retest.log`, and
  `evidence/baseline-inherited-failure.log`.
- The final production metallib is byte-identical to the measured fusion
  artifact. Three additional public-CLI smoke comparisons against baseline,
  including 4K, pass after removing diagnostic switches from production source.
  `evidence/final-audit.json` records source/artifact hashes and collection checks.

Timing inputs, commands, raw samples and process snapshots are in `comparison/`
and `comparison-small/`; `paired-analysis.json` contains all process medians,
paired deltas and ranges. `stage-comparison/` retains stage samples and dispatch
inventories. `artifacts/` contains frozen experimental executables and metallibs.
`evidence/` contains source patches, build/test logs, environment and hashes.
`{sized,fusion}-parity/` contains exact outputs and comparison manifests.

Reanalyzing results launches no encodes:

```sh
python3 build/ac-scratch/analyze_comparison.py build/ac-scratch/comparison
python3 build/ac-scratch/analyze_comparison.py build/ac-scratch/comparison-small
```

`measure.py`, `profile_stages.py` and `qualify_outputs.py` are explicit collection
scripts. Each requires a fresh output directory. The original arm in the timing
controller requires the frozen diagnostic executable; production intentionally
has no full-scratch override.
