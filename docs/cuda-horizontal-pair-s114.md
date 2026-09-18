# Paired horizontal filtering and Malta tile screening (S114)

Date: 2026-09-08. Starting revision: `cf0ac56`, branch `feat/cuda`.
Windows 11, RTX 3060 Laptop (sm86), CUDA 11.8, MSVC 14.37, Release.

## Retained change

The 7-, 13-, and 15-tap horizontal Butteraugli convolutions now evaluate two
adjacent outputs per thread. The production kernels match the measured
prototype instruction-for-instruction. This is retained as a GPU-stage
optimization: the instrumented horizontal-filter intervals save 3.01–3.51 ms
per odd-4K encode. **Uninstrumented whole-encode results remain mixed; this
is not a claim of a general throughput improvement.**

Malta scheduling, vertical filters, both 33-tap paths, serial CPU tile merging,
S111 composition/reduction fusion and opt-in compact coefficient storage are
unchanged. There is no runtime selector, compatibility adapter, new allocation,
readback, host synchronization or launch. The existing 256-thread, 256x4
horizontal tile and its shared-memory capacity are retained.

For two outputs, K+1 shared input values feed 2K products. Each output retains
its ascending tap/FMA order, ascending normalization sum and rounded division.
The interior normalization is still computed once per block. At edges, excluded
taps are skipped, including their weights; multiplying a zero halo by a NaN
weight would not be equivalent. Odd-width pairs store only valid outputs.
Every thread finishes the cooperative load and barrier before partial tiles
skip outputs. This is logical reuse, not a measured DRAM-traffic reduction.

Registers for 7/13/15 taps are 31/38/39, versus 31/39/39 previously. Shared
storage remains 4,224/4,344/4,384 bytes. The prototype compilation reports zero
stack frames or spills. The clean production object and dense/compact release
and scoped-ASAN executables preserve the other 73 retained kernels exactly.
Two new reference kernels preserve the previous 7/15-tap horizontal bodies
for frequency-split differential testing; those references do not route
through the new paired implementation. Public/internal declarations are
unchanged.

## Screening and rejected alternatives

The screen compared two and four adjacent horizontal outputs, and four Malta
tile alternatives, all with 256 threads. Copied horizontal controls and the
copied 32x64 full/LF Malta controls, including flat-grid variants, were native
instruction-identical to the retained kernels.

| Malta output tile | Shared bytes | Nominal scaled values / output |
|---|---:|---:|
| 32x64, retained | 11,520 | 1.40625 |
| 64x32 | 11,520 | 1.40625 |
| 64x48 | 16,128 | 1.31250 |
| 64x64 | 20,736 | 1.265625 |
| 32x96 | 16,640 | 1.354167 |

These ratios describe full interior tiles, including the four-pixel halo.
They are not measured memory transactions or occupancy.

All twelve paired 4K Malta stage/replicate medians favor 64x32 by 2.33–4.26%.
However, its zero-response HD stages regress by up to 3.80%, while larger
tiles generally fare worse. Merely reducing repeated halo arithmetic did not
improve the measured result. The 64x32 geometry has the same tile area and
nominal halo ratio as the baseline, so its 4K gain cannot be attributed to
that ratio. No Malta policy change or new size threshold is retained.

For the retained horizontal tap counts, synthetic odd-4K pair medians improve
by 3.65–6.66%, and HD medians by 2.27–8.44%, across the two pair labels and two
replicates. Four-output groups are less effective and regress for seven taps.
The generic 33-tap pair has a large 4K microbenchmark gain, but it is not the
already-specialized joint 33-tap production path; neither 33-tap route changes.
500x333 and 125x83 cases were also screened. Their short intervals and larger
duplicate-control variation do not support precise small-image speed claims.

There are 56 screen jobs: 32 horizontal and 24 Malta. Each uses six untimed
qualification bursts, six warmup Williams-square rows, and twelve measured
rows, with six labels per row. Opposite-order replicates are retained.
Horizontal bursts contain eight launches; Malta bursts contain four.
Uploads, output checks and scratch checks are outside event intervals, and
every burst is checked bit-for-bit. The horizontal inputs/weights are synthetic.
Malta uses twelve frozen S65 captures: six first-full-scale stages each from
odd HD and 4K, not fresh current-encoder captures or later AQ iterations.

## In-encode attribution and whole-call limits

A separate diagnostic binary changes only the three target horizontal filters.
It links the S111 serial-CPU libraries, not the rejected S112 CPU scheduler.
Every odd-4K encode reaches 37 target launches. Labels 0/2 are native-identical
baselines and labels 1/3 are identical pairs. Four-label Williams squares
shuffle labels and rows, with four warmup rows and twelve measured rows per
cohort. Each candidate must match frozen bytes and the baseline summary.

The event-instrumented binary brackets each target launch on its existing
stream. Event creation and elapsed-time readback are outside the encode timer.
The sum of these intervals includes any enclosed launch gaps; it is not a
profiler kernel-duration sum. Instrumented whole-call timings are kept separate
from the uninstrumented experiment.

Each cell below reports replicate 0 / replicate 1. Differences are medians of
per-round paired differences between the two candidate-label mean and the two
baseline-label mean, not differences between independent medians.

| Workload | Baseline filter intervals, ms | Pair interval difference, ms | Pair interval difference |
|---|---:|---:|---:|
| flower 500 | 0.6917 / 0.7160 | -0.0778 / -0.0489 | -10.46% / -6.65% |
| odd HD | 2.2968 / 2.3332 | -0.2127 / -0.2563 | -9.48% / -11.12% |
| odd 4K | 15.8595 / 15.9450 | -3.5146 / -3.0121 | -22.81% / -18.96% |

The HD and 4K interval gains exceed their duplicate-label interval contrasts.
Small-image duplicate variation is more material. No clock-based normalization
is applied, and these measurements do not establish why the in-encode gain is
larger than the isolated synthetic gain.

The corresponding **uninstrumented complete-call** paired differences are:

| Workload | Difference, ms | Difference |
|---|---:|---:|
| flower 500 | +0.071 / +0.040 | +0.307% / +0.152% |
| odd HD | +0.227 / +0.367 | +0.319% / +0.465% |
| odd 4K | -8.789 / +2.055 | -2.907% / +0.683% |

Whole-call duplicate contrasts reach several milliseconds. The instrumented
4K whole-call contrast also changes sign between replicates. Retaining the
local device-work improvement does not erase these slower observations or
establish a batch-throughput gain. All 68 timing jobs, including both encoder
experiments, have no overlap with another recorded agent job. No power, clock,
priority, affinity, firewall or driver settings were changed.

## Correctness, failures and reproducibility

The explicit encode campaigns pass 2,070 checks: 840 diagnostic encodes, 960
production dense/compact checks, 222 scoped-ASAN checks, and 48 integrated
memcheck checks. Fourteen previously rejected inputs retain their expected
errors. Production coverage reuses S108 quality, low-distance, target-size,
maximum-error, high-amplitude and batch cases, including final scoring,
independent backends and the shared batch driver. Bytes and complete summary
serialization match the frozen oracles. This reuses prior decoded-quality
evidence; it is not a new independent-decoder campaign. The ASAN builds
instrument the harness and resident host implementation, not every linked
library.

The clean production build passes 78/78 CTest tests. The committed frequency
test expands from 320 to 880 fixtures, each with three-stage reuse, including
pair/tile boundaries, padding, zero/negative/subnormal/exceptional weights and
exceptional inputs. Its previous horizontal bodies remain independent
references. The standalone scalar-GPU horizontal oracle checks 612 fixtures
across 7/13/15/33 taps and all six diagnostic labels, with full release and
host-ASAN repetitions. Malta passes 10,752 fixtures in both release and host
ASAN. All eight prototype CUDA sanitizer jobs and four production-linked
frequency sanitizer jobs are clean. The latter use 32 scoped fixtures; this
is not sanitizer coverage of all 880 production fixtures. Integrated memcheck
also reports no device leaks. Forty retained runtime artifacts remain unchanged.

Failed setup attempts are preserved rather than counted as passes: missing
`<string>` in the initial horizontal host harness, an exception thrown through
an `extern "C"` diagnostic callback under `/EHsc`, and a scoped-test build
placing archives before `/link` under `/TP`. Two premature native inspections
targeted missing event executables; a preflight request stopped before launch
for the same reason. These were harness/build failures, not GPU correctness
failures. The initial ASAN build completed the Malta executables before failing
on the horizontal source; its successful Malta products are explicitly tested.
The aggregate validator was also adjusted to this CTest version's actual
success-summary format while independently requiring 78 passed test rows.

Before any replay timing, inspection found that the default-stream pageable
uploads needed an explicit completion boundary before work on a non-blocking
stream. Both replay harnesses were rebuilt with that synchronization outside
timing, rechecked at 4K, and native-audited. No timing from the earlier replay
binaries is used. No admin/firewall blocker was observed.

Evidence is frozen under `U:/gjxl-cuda-diagnostics/s114`; the final clean build
is its `build-cuda` directory. The archive includes sources, manifests,
executable/native hashes, failed logs, per-window measurements, production
checks and the parent Butteraugli source. Historical prototype builds must
use that parent source (or starting revision `cf0ac56`), not the newly paired
production source included by the same diagnostic translation unit. The
validator verifies the parent content against its recorded original hash,
including the original Windows newline representation.

Relevant scripts in `build-cuda-ninja/profiles` are `s114_campaign.py`,
`s114_analyze.py`, `s114_within_campaign.py`, `s114_within_analyze.py`,
`s114_production_campaign.py`, `s114_validate.py` and `s114_freeze.py`.
Campaign writers use exclusive output names and are not in-place rerun tools.
After freezing, verification is:

```text
python build-cuda-ninja/profiles/s114_validate.py --frozen
```

The backend is not exhausted. Malta's size-dependent tradeoff needs more
mechanism evidence before changing its policy; whole-encode performance and
qualification beyond this Windows/sm86 machine remain open.
