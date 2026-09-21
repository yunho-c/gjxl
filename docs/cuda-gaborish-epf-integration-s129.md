# Resident Gaborish/first-EPF integration (S129)

Starting revision: `b8a0173`, branch `feat/cuda`; Windows, MSVC 14.37,
CUDA 11.8, RTX 3060 Laptop / sm86, 2026-09-08.

Outcome: **do not promote this candidate yet**. It is bit-exact on actual
resident inputs, but the large synthetic interval win does not survive
reliably in the encoder. The two 4K stage changes are +0.36% and −1.44%,
with larger duplicate-control variation. Follow-up counters retain roughly
48% less DRAM traffic but expose the same occupancy cost and changing SM
frequencies. Production remains S127; reducing the fused tile's resource
cost and explaining its workload-context sensitivity are next.

## Hypothesis and isolated integration

[S128](cuda-gaborish-epf-fusion-s128.md) found a promising raw-shared
Gaborish/first-EPF candidate for pass 1, but not a general pass-0 win. Its
synthetic benchmarks used resident-style pitches with deliberately unaligned
guarded pointers. S129 tests the same compiled candidate inside the fully
resident encoder, using actual reconstruction and inverse-sigma data.
Production source is not changed during this qualification experiment.

The diagnostic resident owner links S127 production libraries and the exact
S128 GPU object. It changes only `EncodePostprocess`, plus diagnostic hooks.
Gaborish followed by EPF pass 1 becomes one raw-shared kernel for one- or
two-iteration filtering. Three-iteration filtering keeps the separate
Gaborish/pass-0 path; disabled Gaborish and zero EPF keep their existing
paths. S127's final-EPF/color fusion remains in place.

| Gaborish-enabled profile | First boundary | Later work / live result |
| --- | --- | --- |
| One EPF, perceptual | Gaborish + pass 1 → RGB | Return RGB directly |
| One EPF, maximum error | Gaborish + pass 1 → scratch 1 XYB | Ordinary color conversion; scratch 1 remains live for maximum error |
| Two EPF, perceptual | Gaborish + pass 1 → scratch 1 XYB | S127 fused pass 2 → RGB |
| Two EPF, maximum error | Gaborish + pass 1 → scratch 1 XYB | Pass 2 → scratch 0 XYB, then color conversion |

Both control and candidate follow this owner structure. The control dispatch
uses production Gaborish and production pass 1; the candidate substitutes
S128 raw-shared mode 3. Original logical stage numbering is retained, so
`FinalFilteredImage()` still selects the correct maximum-error consumer data.
The candidate deliberately leaves scratch 0 untouched at the fused boundary.
Preparation, allocation counts/sizes, coefficient ownership and subsequent
consumers do not change. This isolates the fusion from any future scratch
lifetime optimization; no memory-capacity saving is claimed.

The full diagnostic executables contain 222 GPU bodies: all 213 S127 bodies
unchanged, plus S128's nine controls/candidates. The two replay executables
contain only the 21-body S128 object. Extracted ELF module hashes verify
fifteen executables: thirteen retain nine unchanged modules and replace only
the audited exact-kernel module; both replays use that same module. No CUDA
code is recompiled or silently retuned in S129. Diagnostic selectors and
events are not proposed public APIs or compatibility layers.
Two later standalone counter executables are checked separately against the
same device-module hash, bringing the final executable inventory to seventeen.

## Actual inputs, pitches and alignment

Three diagnostic encodes capture both first-filter boundaries for Flower
500, HD and odd 4K. The capture synchronizes the producer stream and copies
only logical image/sigma rows, retaining original pitches, parameter float
bits, route and input-pointer alignment in a versioned header. It does not
read uninitialized row padding. Each following candidate encode matches the
frozen codestream and reference summary: six checked encodes in total.
Capture synchronization is absent from the timing executables.

All six captures are Gaborish plus pass 1 → XYB in the default two-EPF
profile. At 3839x2159, reconstruction/Gaborish/XYB pitches are 3840, packed
RGB pitch is 3839 and sigma pitch is 480. Input and sigma pointers are all
256-byte aligned in every capture. This closes S128's actual-input and
allocation-alignment gap for the default route.

Replay restores the captured parameters and row pitches, poisons padding,
and exercises both XYB and RGB output routes with zero- and nineteen-float
allocation prefixes. Original, native-identical clone, direct and raw-shared
outputs match bitwise through three reuses. Inputs, sigma, row/allocation
guards and expected scratch behavior remain intact. Release, host ASAN and
GPU memcheck each pass 288 guarded executions, for 864 total; memcheck
reports zero errors and leaks. RGB replay uses the captured two-EPF input
data with the color epilogue; these are not mislabeled as captures of a
one-EPF encode. Actual one-EPF profile behavior is checked separately below.

## Profile and public-workflow checks

Release and host-ASAN overlays pass the existing CUDA AQ test, including its
CPU comparisons, exact/resident maximum-error behavior, prepared reuse,
deferred metadata and failure/concurrency contracts. This is a relinked test,
not a new full CTest campaign. ASAN instruments the resident owner and test
driver, not every production library or GPU instructions.

The profile matrix compares original/candidate AQ results over two contents,
zero through three EPF iterations, Gaborish on/off, ordinary/custom filter and
intensity parameters, perceptual/maximum-error control and zero/two AQ
updates. Every case checks whether fusion or the expected non-fused path
actually ran. Quant fields, block maps, score histories, maximum-error
results, complete RGB planes and resident frame population are checked;
serialization results must agree, including unchanged empty output for
unsupported writer profiles.

Release, host ASAN and memcheck each pass all 128 pairs: 384 bitwise AQ
comparisons. Eight pairs per run serialize identically; the remaining 120
match the existing unsupported-profile writer rejections. Thus only 24
compared pairs produce successful codestreams, not all 384. Matrix memcheck
also reports zero errors and leaks. Profiles with one EPF and maximum-error
scoring exercise the preserved final XYB scratch ownership directly.

Fixed-candidate dense and compact overlays also pass the S108 frozen-oracle
corpus: 960 ordinary encodes, 222 under host ASAN and 48 under encoder
memcheck, plus fourteen matching expected rejection cases. These cover
photographs, padded HD/4K, quality/final-score settings, byte and maximum-error
search, extreme finite ranges, compact width/overflow decisions, serial
reuse, independent contexts and the public batch driver. All four encoder
memcheck jobs report zero errors and leaks. Compact ownership stays opt-in.
The fixed selector is never changed in these drivers; explicit per-call
route assertions are supplied by the matrix and timing drivers, not an
inferred destructor footer. Batch correctness is not a batch speedup claim.

## Timing protocol

Separate within-process wall-time and CUDA-event executables use duplicate
baseline labels 0/2 and duplicate raw-shared labels 1/3. Every encode must
match the frozen codestream, reference summary, AC coefficient width and
owner size, and execute exactly two first-filter boundaries. The event build
also asserts two intervals. Each interval spans the complete Gaborish/first-
EPF boundary, not just EPF alone or the later S127 EPF/color boundary.

Flower 500, HD and odd 4K each run twice in reversed case order. Each job
uses four warm and twelve measured rounds of four balanced/shuffled labels.
Distance is 1.2, effort 7, CPU thread count automatic, pool allocation and
final scoring off. Backend construction, image I/O, output comparisons and
returned-result destruction are outside the complete-call wall timer.
Release/host-ASAN preflights check both timing drivers before measurement.

All GPU correctness work finishes before measured timing begins. The job
validator checks every measured interval against all recorded jobs. Read-
only NVML enforced-power-limit endpoints are captured outside the encode
wall timer. Clock, power, priority, affinity and security settings are not
changed, and no artificial idle period is inserted. Equal limit readings
are not evidence of equal operating clocks within an encode. Instrumented
whole-call results remain a separate population from uninstrumented timing.

## Measured results

Matched-round medians below are milliseconds; negative deltas mean faster.
Each round averages baseline labels 0/2 and candidate labels 1/3 before
subtraction. Percentages are formed within rounds as well. Separate medians
are not additive or confidence intervals. Duplicate columns compare labels
that execute identical code.

Uninstrumented complete encodes:

| Input / window | Baseline encode | Candidate delta | Paired change | Baseline duplicate | Candidate duplicate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flower 500 / 0 | 19.729 | +0.073 | +0.38% | −0.054 | +0.271 |
| Flower 500 / 1 | 19.959 | −0.239 | −1.19% | +0.589 | −0.247 |
| HD / 0 | 76.821 | −1.138 | −1.53% | −1.279 | −1.305 |
| HD / 1 | 78.025 | −3.033 | −3.85% | +1.598 | −0.325 |
| 4K / 0 | 314.943 | −1.445 | −0.46% | −3.684 | −19.263 |
| 4K / 1 | 322.878 | −13.088 | −4.02% | −11.265 | +3.220 |

Separately instrumented jobs, summing both first-filter boundaries per encode:

| Input / window | Baseline stage | Stage delta | Stage change | Stage duplicates (baseline, candidate) | Instrumented encode delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flower 500 / 0 | 0.157440 | −0.012032 | −7.67% | 0.000000, −0.000512 | −0.261 |
| Flower 500 / 1 | 0.171008 | −0.026880 | −15.58% | +0.000512, −0.002048 | −0.022 |
| HD / 0 | 0.988160 | −0.034048 | −3.38% | −0.026112, 0.000000 | −1.202 |
| HD / 1 | 1.023232 | −0.013568 | −1.29% | −0.030208, +0.008704 | +2.642 |
| 4K / 0 | 12.453376 | +0.008448 | +0.36% | +0.077824, +0.652288 | +4.194 |
| 4K / 1 | 11.946752 | −0.178176 | −1.44% | +0.262656, +0.446464 | +4.581 |

Five of six primary stage medians and 22/24 individual candidate/control
stage medians favor fusion. However, one HD and one 4K individual comparison
regress, and the large-image stage deltas are small against duplicate-label
variation. Both 4K instrumented whole-call medians regress. The uninstrumented
4K window with a 13 ms improvement has duplicate variation reaching 19 ms;
it is not evidence that this boundary saves 13 ms. S128's 14.5–15.3% short-
burst or 34–37% sustained XYB savings must not be substituted for these
actual in-encoder results.

All 1,680 power-limit endpoints in the preflight/measured timing processes
are 40 W, with zero endpoint changes. Uninstrumented measurement runs from
16:31:20 to 16:32:19 UTC, followed by event measurement through 16:33:19 UTC.
The RDP-associated limit transition seen in S128 does not recur in these
endpoint records. Stable limits do not remove the observed timing variance
or establish stable operating clocks.

## Investigating the synthetic/encoder gap

The inverse-sigma captures rule out widespread EPF bypass as an explanation.
Neither 4K capture bypasses any pixels; both Flower captures also have none.
HD bypasses only 0.038% and 0.065% of logical pixels in its two captures.
All captured sigma values are finite. The captured default color scale is
1.0 rather than S128's fixed 0.255; this epilogue is not part of the default
first-boundary XYB route.

A follow-up factor experiment uses the first 4K capture, then substitutes
S128's synthetic ramp and sigma pattern while retaining the captured filter
parameters. Both data populations run with zero- and nineteen-float prefixes,
production/direct/raw-shared modes, and two reversed-order repetitions:
24 Nsight Compute captures. Each process executes one reference, eight warm
chains and one profiled chain, then checks output bits, flags and input/scratch
integrity. Twelve additional host-ASAN preflights cover every data/prefix/mode
combination. Kernel replay is used with clock/cache control disabled. This
is an isolated replay, not counters captured inside a running encoder.

| Data / prefix floats | Baseline DRAM MB | Raw-shared DRAM MB | DRAM change | Raw-shared global-load Msectors | Raw-shared SM MHz |
| --- | ---: | ---: | ---: | ---: | ---: |
| Captured / 0 | 396.78–396.81 | 207.18–207.27 | about −47.78% | 6.503–6.505 | 1277.6–1278.1 |
| Captured / 19 | 397.04–397.05 | 206.69–206.98 | about −47.91% | 5.516–5.518 | 1500.8–1501.9 |
| Synthetic / 0 | 396.83–396.84 | 207.19–207.22 | about −47.79% | 6.501–6.502 | 1449.1–1468.6 |
| Synthetic / 19 | 397.11–397.20 | 207.02–207.06 | about −47.87% | 5.517 | 1465.5–1518.3 |

Executed work is identical across the two data populations and alignments:
216,729,529 baseline warp instructions versus 185,139,360 raw-shared
(−14.58%). Predicated thread FFMA counts are 406,131,649 versus 419,853,403
(+3.38%). The direct candidate executes 224,015,208 warp instructions
(+3.36%), despite reducing DRAM traffic to about 199 MB. Raw-shared achieved
occupancy stays about 49.4%, versus baseline EPF's 98.6–98.7%; changing the
data does not remove the shared-storage cost.

Alignment does change memory transaction demand: aligned raw-shared loads
about 18% more global sectors than the nineteen-float-prefix version, while
aligned direct fusion loads about 55.0 million sectors versus 46.2 million.
The deliberately unaligned fixture was therefore not a performance-neutral
layout substitution. Nevertheless, the roughly 48% DRAM reduction survives
both factors, and same-layout captured/synthetic instruction and traffic
counts are effectively unchanged.

Profiler-observed SM clocks differ materially between these short captures,
while DRAM clocks are around 5495 MHz. For example, captured/raw-shared runs
take about 1.828 ms aligned at 1278 MHz and 1.577 ms unaligned at 1501 MHz.
These durations are not a clock-matched alignment comparison or an estimate
of the in-encoder saving. No clock normalization is used. The exact cause of
the in-encoder loss remains unresolved: the evidence confirms traffic/work
savings and an occupancy penalty, not a single proven performance limiter.

Two short Nsight Systems traces then test whether launch gaps inflate the
encoder interval. Each runs the existing event executable through one
reference and four label encodes at 4K, preserving frozen output checks.
The analysis matches the two first-filter boundaries per encode to actual
kernel records on their stream. For the four non-reference baseline
encodes, combined Gaborish-to-EPF gaps are only 0.006688–0.016224 ms, versus
8.965–11.263 ms executing the kernels. Event time exceeds kernel time by
0.019–0.037 ms for these controls and 0.013–0.164 ms for the fused encodes.
These observed intervals are predominantly GPU execution, not multi-
millisecond gaps between the two launches.

The traces add ten exact encodes and twenty 40 W limit endpoints. CPU
sampling/context-switch tracing is disabled; CUDA/NVTX tracing and SQLite
export complete without an elevation or firewall blocker. The short traced
population has no twelve-round timing qualification and is not substituted
for the main measurements. Its purpose is to rule out a large event-gap
accounting explanation in the observed windows, not prove a new speedup.

## Decision and next experiment

Keep this raw-shared implementation diagnostic. Do not introduce default
fusion dispatch, a geometry threshold or a public compatibility switch from
these results. The candidate is correct, but a dependable large-image stage
improvement has not been demonstrated in the resident workload. Production
remains S127; allocation policy and compact defaults stay unchanged.

The next concrete tile experiment is to reuse shared storage: compute the
filtered halo into per-thread temporaries, finish all raw-window readers,
then reuse the raw shared allocation for the EPF tile. Pass 1 currently
keeps 17,328 raw bytes and 15,552 filtered bytes simultaneously. Reuse could
remove that simultaneous second footprint, but adds a barrier and may raise
register pressure or spill. It must be compiled/audited and tested, not
assumed faster. Start with the aligned captured inputs and qualify any gain
inside the encoder before repeating broad promotion checks. Keep pass 0's
earlier regression and maximum-error XYB ownership requirements explicit.

Separately, the resident filter scratch lifetimes remain an opportunity for
lower memory capacity without depending on this unqualified fusion. No
scratch allocation is removed here. Neither this candidate nor the backend
has been established to be maxed out.

## Reproduction and audit

Evidence is under `U:/gjxl-cuda-diagnostics/s129`, with diagnostic source and
generators under `build-cuda-ninja/profiles/s129_*`. The frozen S128 GPU object
is linked directly against S127 libraries; no historical binary or oracle is
regenerated. Core entry points are `s129_prepare.py`, `s129_build.ps1`,
`s129_check_resume.py`, `s129_measure.py` and their analysis scripts.
`s129_capture_stats.py`, `s129_hot_counters.py` and `s129_trace.py` preserve
the follow-up checks. `summary.json` covers the initial 2,076 exact encodes;
`final_summary.json` and the final validator add the ten timeline encodes
and supplementary counter evidence. Reproduce with frozen sources in a new
root: output creation is exclusive, not an overwrite of earlier evidence.

The first native-audit checker had a stray quote in an integer index. Python
stopped before any extraction; the failed script/log are retained. The v2
checker fixes only that syntax error, as recorded in
`checker_correction.json`, and the qualification resumes at the audit.
No C++ or GPU code changes were needed. The original failed checker is
archived but deliberately not parsed as a valid Python dependency graph.

The final validator checks 219 recorded jobs: 204 accepted, fourteen matching
expected rejections and the one corrected checker failure. It verifies
generation/input/log/executable hashes, exact resident-overlay scope, native
module identities, profile-matrix coverage, captured data/parameters, guarded
replays, raw counter exports, timeline kernel/gap accounting, balanced timing
labels and paired medians, and forty retained runtime hashes. All twelve
measured timing jobs overlap no other recorded job.

Totals are 2,086 frozen-oracle encodes (252 under host ASAN), 384 compared AQ
profile pairs including 24 serialized pairs, 864 guarded hot replay
executions and six clean CUDA memcheck jobs. The factor study adds 240 logical
counter chains plus 120 host-ASAN preflight chains; each ten-chain process
checks its final output, not every intermediate repeated output. Timing
records contain 1,680 power-limit endpoints and traces add twenty. The AQ
tests are additional coverage, not folded into an invented encode count.

Production source/tests/build files remain unchanged; there is no fresh full
CTest run or new default runtime in this study. Final source snapshots and
the SHA-256 artifact manifest preserve all findings independently of future
live-source edits.
