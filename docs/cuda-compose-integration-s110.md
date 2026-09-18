# Integrated composition / maximum reduction (S110)

Date: 2026-09-08. Starting revision: `173f860`, branch `feat/cuda`.
Windows 11, RTX 3060 Laptop (sm86), CUDA 11.8, MSVC 14.37, Release.

## Scope and mechanism

This study takes S106's standalone composition/maximum prototype through the
actual prepared Butteraugli and fully-resident encoder lifecycles. Compact
coefficient storage remains OFF in both encoder timing paths, isolating this
change from S107–S109. CPU tile scheduling is unchanged.

The new internal `LaunchCudaButteraugliCompose` writes the multiscale distance
map and produces the first 256-wide maximum reduction in one kernel. AQ still
needs the materialized map: its store and later per-transform reads remain.
The first partials occupy existing reduction scratch A; subsequent passes
alternate B/A, with the last pass writing the score directly. At <=256 pixels
the fused pass writes the score without touching either scratch pointer.

The composed float is stored before invalid-value sanitization. Nonfinite or
negative logical values make the score NaN, without changing the map. All
lanes, including tail lanes, participate in the same shared-memory reduction
tree. The fused and subsequent reduction kernels share that tree in source.
In-place composition is safe because every output thread reads its own main
pixel; subscale input and reduction storage are disjoint.

The tested integration replaces only the multiscale composition and first
maximum pass. Expanded/cropped and single-scale comparisons retain their
existing maximum reduction. It adds no allocations, readbacks, host/device
synchronization, compatibility adapter, or runtime option. The map and score
are still ready before the stream's dependent AQ kernels.

At 3839x2159 this removes one logical 33,153,604-byte map read and one launch
per comparison. The ordinary d1.2/e7 timing cases execute two such comparisons
per encode (verified by diagnostic branch counters). These are logical
traffic counts, not new DRAM-counter measurements. S106 measured about 29%
lower isolated composition-plus-maximum time at this geometry; that earlier
microbenchmark is not an integrated encoder speedup claim.

The diagnostic sm86 compilation uses 18 registers and 1024 shared bytes for
the fused kernel, versus 16 registers for separate composition and 10
registers / 1024 shared bytes for maximum reduction. All three report zero
stack and spills. No fast-math or device/power/security setting was changed.

## Correctness and lifecycle qualification

The permanent `cuda_compose_maximum` test compares against independent copies
of the prior separate GPU kernels, not the production reduction helper:

- 480 cases: 15 geometries, two layouts, separate/in-place output, eight value
  patterns. Sizes include 1x1, 225/256/289 pixels, 255/256/257 pixels,
  65,535/65,536/65,537 pixels, 500x500, odd HD and odd 4K.
- Distinct padded main/sub/output strides, unaligned pointer prefixes, row and
  allocation guards, immutable inputs, reusable scratch, and null scratch
  pointers for a one-block reduction.
- Bitwise complete maps and finite scores, signed zeros, NaN in main/subscale
  inputs, infinity, a negative final pixel, and finite-input overflow. Invalid
  scores must be NaN; the map is not sanitized into a different result.

The integrated encoder repeats S108's frozen byte and complete-summary
oracles: 480 checked encodes in 40 successful jobs, covering photographic
content, effort/distance/final-score settings, size/error searches, successful
high-range inputs, independent backends and public-driver batches. Seven
previously rejected extreme inputs fail with the same errors. Those seven
are not successful qualification cases.

Both timing campaigns also validate every result. Together with the initial
qualification they contain 2,942 exact encode checks; the production campaign
has 864 measured encodes and the within-process campaign has 960. This is
deterministic parity against the existing CUDA oracle, not a new independent
decoder or cross-device quality campaign.

## Within-process timing

One diagnostic executable contains both routes. Labels 0/2 select separate
passes; labels 1/3 select fusion. The selector and branch counters exist only
in a mechanically generated diagnostic source copy, never in tracked runtime
code. Each encode asserts that exactly the expected two branches executed,
then checks frozen codestream bytes and a fresh complete reference summary.

Five inputs run twice in reverse case order with different seeds. Each job
uses one reference, four warm Williams-design rounds, and 24 measured rounds
of four shuffled labels: 113 checks per job, 1,130 total. The outer boundary
includes a complete in-memory encode on a persistent backend, but excludes
input I/O, backend creation, result comparisons and returned-result destruction.
Distance is 1.2, effort 7, CPU threads automatic, final score OFF.

Ranges below are the eight descriptive paired medians per input (four
fused/separate combinations across two repetitions). They are correlated,
not eight independent experiments. Negative means faster.

| Input | Fused/separate paired medians | Duplicate-label controls |
|---|---:|---:|
| Flower 500 | -1.95% to +1.37% | -0.66% to -0.15% |
| Keong 500 | -1.88% to +0.02% | -2.70% to +1.78% |
| Keong 2000 | -1.84% to +3.51% | -0.31% to +1.31% |
| Odd HD | -1.62% to +0.10% | -2.46% to +1.51% |
| Odd 4K | -1.75% to +2.40% | +0.22% to +1.23% |

HD generally leans faster, but every input has mixed-sign candidate medians.
The second Keong 2000 and 4K repetitions have all-positive candidate paired
medians; those observations are retained, not discarded. Quantization-stage
deltas also change sign. The whole-call data does not establish a repeatable
general encoder improvement from this small stage elimination.

## Production-linked timing

Two separately linked executables use the same frozen S108 encode harness:
the S108 dense baseline and the integrated S110 candidate. Six repetitions
alternate mode order and reverse case order. Five single-image cohorts use
two warm and eight measured loops. Two-image HD/4K cohorts use two warm and
four measured loops, with both independent backends and the public batch
driver; batch requests use two CPU threads each. All measurements retain the
complete returned result until after timing and compare bytes/full summaries.

There are 108 jobs, 1,332 checked encodes and 864 measured encodes. Timing is
serial across jobs; only the intended concurrent requests within a batch
overlap. No build, sanitizer, CTest or telemetry poll overlaps either timing
campaign. Light editing and ordinary machine-state drift remain limitations.

The table reports medians of each process's measured-loop median. The paired
percentage is separately computed as the median of six paired process
percentages; it is not the ratio of the two displayed aggregate medians.

| Input / mode | Separate ms | Fused ms | Paired median change | Faster pairs / 6 |
|---|---:|---:|---:|---:|
| Flower 500 | 21.118 | 21.039 | -1.20% | 4 |
| Keong 500 | 22.717 | 22.018 | -1.63% | 3 |
| Keong 2000 | 154.125 | 154.467 | -0.45% | 4 |
| Odd HD | 79.827 | 78.602 | -2.07% | 4 |
| Odd 4K | 334.849 | 327.188 | -0.57% | 3 |
| HD x2, independent | 137.640 | 136.602 | -2.07% | 4 |
| HD x2, public driver | 136.474 | 135.945 | -3.32% | 4 |
| 4K x2, independent | 769.933 | 751.622 | -2.29% | 5 |
| 4K x2, public driver | 772.585 | 769.812 | -0.64% | 4 |

All aggregate paired medians are favorable, but every cohort has mixed-sign
repetitions. Single-image 4K pairs range from -5.81% to +8.78%; Flower ranges
from -16.67% to +18.99%. The result does not resolve a sub-millisecond stage
saving reliably at the complete-call boundary. In particular, the displayed
multi-millisecond aggregate differences must not be attributed wholly to
composition fusion or added to S107/S109's observations.

Within-process timing ran 04:46:08.619–04:48:31.993 UTC; production-linked
timing ran 04:49:34.590–04:54:36.687 UTC on September 8. The entire timing set,
including unfavorable repetitions and duplicate controls, is preserved.

## Safety and final source

All four Compute Sanitizer tools pass 384 focused kernel cases each: the
first 12 geometries through 65,537 pixels, retaining all layouts, aliases and
value patterns. Memcheck reports zero leaked bytes; racecheck reports zero
errors, warnings or hazards; initcheck and synccheck report zero errors.

Four integrated memchecks pass: seven prepared-reference cases, independent
mixed-quality batches, public-driver mixed-quality batches and the four-item
sample rate-search fixture. The last three contribute 36 additional exact
encode checks. All four report zero errors and leaks. A separate release run
passes all 31 prepared Butteraugli cases, including cached-reference reuse,
identity restoration, guard/allocation invariants and failure invalidation.
These prepared-operation tests also retain the existing CPU tolerance oracle.

The tested integration is preserved as `s110/integrated_kernels.cu`, with its
original isolated `build-cuda` directory and statically linked probes. Final
source restores the separate composition route and retains the new fusion
primitive plus permanent differential test. The shared reduction helper is
still used by both entry points; the encoder's launch sequence is unchanged.
There is no new runtime selector or compatibility layer.

The restored source is built independently in `s110/build-final` so the
candidate binaries are not overwritten. Its clean CTest run passes **77/77**
tests in **168.26 seconds**, including the new 480-case kernel test, existing
prepared-operation tests, and the install-consumer check.

## Disposition and next step

Do not change default encoder routing on these measurements. S110 closes the
prepared-lifecycle, bitwise-output, concurrent-context and integrated-memory
questions left open by S106, but does not demonstrate a repeatable general
complete-encode speedup. Retain the qualified primitive as a building block
for a larger composition/AQ-reduction fusion; do not add an opt-in switch for
this small experiment.

The larger opportunity is to remove another consumer's map traffic, not just
this first maximum pass. A composition plus strategy-aware AQ reduction must
preserve the transform score definition, optional final-score/map consumers,
scratch lifetimes and exact downstream field decisions. It remains an
unimplemented hypothesis here. The separate S106 CPU scheduling candidate
still needs concurrency/budget qualification before any routing change.
Prior S93 erosion-tiling and streaming-proxy evidence also argues against
repeating arithmetic-only work at that already bandwidth-heavy boundary.
None of these results establishes that the encoder is globally maxed out.

## Evidence and reproduction

Evidence root: `U:/gjxl-cuda-diagnostics/s110`. Scripts and diagnostic sources:
`build-cuda-ninja/profiles/s110_*`. The manifest includes both build outputs,
linked probes, integrated source snapshot, oracle/spec inputs, reports,
analysis and relevant source hashes. Existing 40-file retained runtime
hashes are checked without rebuilding those binaries. User scratch files are
not read, edited or staged. No firewall/admin prompt or permission blocker
was observed; no firewall, driver, clock, power or priority setting changed.

The runner requires exit zero and an explicitly flushed completion marker;
sanitizer summaries alone are insufficient. The validator recomputes analysis,
checks exact-summary oracles and artifact hashes, and verifies all 176 task
GPU jobs are serial except the intentional requests inside batch jobs.

```powershell
python build-cuda-ninja/profiles/s110_validate.py --frozen
```

The campaign/build scripts use exclusive named output files. Do not rerun a
named job into the frozen evidence root; use a new study directory. Build
scripts configure the current working source, so reproducing the integrated
candidate requires the archived integrated source, not the final restored
encoder route. The diagnostic selector is not a production feature.
