# Gaborish/first-EPF tile fusion (S128)

Starting revision: `d4d1a99`, branch `feat/cuda`; Windows, MSVC 14.37,
CUDA 11.8, RTX 3060 Laptop / sm86, 2026-09-08.

Outcome: raw-shared Gaborish/first-EPF fusion is a promising **pass-1-only
candidate**, not a production change. Clean 4K short bursts save about
14.5–15.3% for XYB and 17.9–18.7% for RGB, including user-requested repeats.
Pass 0 regresses in short HD/4K bursts. All differential and sanitizer checks
pass, but real resident inputs, alignment and whole-encode qualification
remain necessary. Runtime stays at S127.

## Hypothesis and scope

[S127](cuda-epf-color-production-s127.md) removes the final EPF/color
intermediate. At the other end of the filter sequence, Gaborish still writes
three full XYB planes that the first EPF pass immediately reads. S128 tests
computing those Gaborish values inside the EPF tile instead. This is a new
producer/consumer boundary, not a rerun of the mask-fusion candidate rejected
for default promotion in [S104](cuda-candidate-qualification.md).

The experiment leaves production unchanged. It tests three routes:

| Route | Boundary | Resident use |
| --- | --- | --- |
| 0 | Gaborish + EPF pass 0 → XYB | Three EPF iterations |
| 1 | Gaborish + EPF pass 1 → XYB | Two EPF iterations, or one with maximum-error scoring |
| 2 | Gaborish + EPF pass 1 → linear RGB | One EPF iteration with perceptual scoring |

Route 2 includes S127's color epilogue, so it compares against Gaborish followed
by the already-fused EPF/color kernel, not an obsolete three-kernel baseline.
Every baseline boundary uses two launches; each new fused boundary uses one.
Earlier/later filters, zero-EPF behavior, coefficient ownership, allocation
policy and encoder scheduling are outside this isolated experiment.

## Two tile-loading alternatives

Both candidates retain the production 32x32 output tile, 256 threads and EPF
arithmetic. The **direct** candidate computes each Gaborish-filtered halo cell
from global input and stores it in the ordinary shared EPF tile. It avoids the
global intermediate but duplicates Gaborish work in overlapping halos.

The **raw-shared** candidate first loads an expanded three-channel raw image
window into shared memory, then computes the filtered EPF tile from it. It
adds a barrier and shared storage but avoids repeated global neighbor loads.
The raw window is 40x40 for pass 0 and 38x38 for pass 1; the filtered windows
are 38x38 and 36x36, respectively.

Boundary arithmetic is deliberately not reassociated. EPF reflects the
filtered-image coordinate first; Gaborish reflects its immediate neighbors
around that real center. Simply reflecting a combined virtual neighborhood
can reverse operand order at an edge, changing floating-point results.
The raw-shared candidate selects a bounded real-coordinate window, shifting
it toward the image interior at partial right/bottom tiles and replicating
unused padding for small images. Canonical Gaborish sample order is retained.
An independent one-dimensional mapping check covers 374,514 axis-neighbor
relations over widths 1–257, larger boundary sizes and sampled large-image
tiles. The GPU fixtures also check both axes together.

All logical Gaborish centers remain covered, including when EPF bypasses a
block. Non-finite Gaborish values are sanitized before EPF and OR `1u` into
the shared error word. EPF and optional color sanitation retain their existing
behavior. Repeated halo evaluation may repeat an atomic OR, but must not
invent or lose an error bit.

## Native code and resource cost

The diagnostic GPU object contains 21 bodies: S127's twelve original bodies,
three EPF control clones and six candidate specializations. Every original
body is instruction-identical to the production object; all three control
clones match their corresponding production EPF bodies after symbol
normalization. No other GPU optimization is hidden in the comparison.

| Route | Control EPF registers | Direct registers | Raw-shared registers | Direct shared bytes | Raw-shared bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 56 | 56 | 56 | 17,328 | 36,528 |
| 1 | 40 | 43 | 40 | 15,552 | 32,880 |
| 2 | 55 | 55 | 55 | 15,552 | 32,880 |

All nine new bodies have zero stack frame and spill loads/stores. Gaborish
itself is unchanged in the controls. The higher shared-memory footprint is
a real occupancy tradeoff, not free storage. Executables link the same GPU
object; extracted ELF module hashes verify device-code identity across hosts.

## Correctness and benchmark protocol

The guarded fixture reuses S126's allocation/row guards and sixteen
input/sigma patterns: signed zero, random/ramp/checkerboard data, large finite
values, subnormals, NaNs, infinities, threshold-adjacent bypass values, mixed
bypass rows, and positive/NaN sigma. Twenty-four geometries include tiny and
single-row/column images, tile boundaries, odd sizes and a 4096x3 strip.
Two padding layouts, three routes, ordinary/custom Gaborish weights and
three changed-input reuses compare production, a native-identical control,
direct fusion and raw-shared fusion. Full output bits and error flags match;
input/sigma and guards remain intact; fused paths leave the unused global
Gaborish scratch unchanged. Release and host ASAN each pass 55,296 pipeline
executions. This is a differential test of actual rebuilt candidate code,
not a claim that synthetic fixtures replace eventual encoder qualification.
Memcheck, racecheck, initcheck and synccheck each pass another 6,912 executions
with zero errors/hazards; memcheck reports zero leaked allocations. Including
the ordinary 6,912-execution preflight, the total is 145,152 guarded pipeline
executions. Host ASAN instruments the fixture/driver, not device instructions.

The initial short pilot uses the deliberately distinct pitches of the guarded
fixture: coding-width input, intermediate width +3, and source-width output.
It is retained separately. A second host driver changes only construction
of the fixture to resident-style strides: input and Gaborish intermediate use
coding width; XYB output uses coding width and RGB uses source width. An
additional seven words of padding exercise another alignment. This correction
changes no GPU code and is documented rather than pooling the two layouts.
Both drivers retain the fixture's 19-float allocation prefix, so active
pointers are deliberately unaligned by 76 bytes. "Resident-style" describes
the row pitches, not a reproduction of resident allocation alignment. Color
scale is fixed at 0.255 in this fixture. Real captures and the production
parameter matrix remain future integration gates.

The main resident-layout campaign uses 500x500, 1919x1079 and 3839x2159,
all three routes and both padding choices, with two repetitions in reversed
case order: 36 jobs. Inputs remain synthetic ramps, not captured reconstructions.
Each job has six warm and twelve measured rounds for each of 4- and 64-chain
bursts. Six balanced labels comprise production, a native-identical control,
two direct labels and two raw-shared labels. Every label occupies each
position twice in the measured rounds.

CUDA events enclose launches on one nonblocking stream. Upload completion
precedes that stream's work; output checks and NVML power-limit endpoints are
outside the event interval. Every burst's final output and accumulated error
flags are checked; the report counts checked bursts separately from the
number of logical chains executed inside them. Candidate/control differences
are paired within rounds before taking medians, with individual comparisons
and duplicate-label differences retained. Power-limit readings are not
per-kernel clock measurements.

## Initial timings and the interrupted power regime

The 36-job campaign ran from 15:40:27 to 15:57:11 UTC. Near its end, the
read-only enforced-power-limit readings changed from 40,000 mW to values
between 71,536 and 74,181 mW. Seven jobs contain this changed regime: the
second HD/pad-0/route-0 case and the final six 500x500 cases. There are seven
within-burst endpoint changes. The user subsequently reported a brief RDP
visit and requested repeats. This is a temporal association, not proof that
RDP caused the transition; no clock, power, RDP or security settings were
changed by the experiment.

All original results are retained. A separate, explicitly post-hoc eligibility
report excludes those seven **whole jobs** from qualification against the
initial 40 W regime. It does not trim selected rounds or normalize timings.
The remaining 29 jobs include all twelve 4K jobs. Their 4K ranges below span
two repetitions and both padding layouts; negative percentages mean faster.

| Route | Burst chains | Baseline ms/chain | Direct change | Raw-shared change | Raw-shared delta ms/chain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0: pass 0 → XYB | 4 | 2.249–2.264 | +3.20–+3.60% | +2.86–+5.05% | +0.065–+0.114 |
| 1: pass 1 → XYB | 4 | 1.703–1.710 | −9.33–−8.78% | −15.28–−14.84% | −0.260–−0.254 |
| 2: pass 1 → RGB | 4 | 1.834–1.841 | −6.38–−5.45% | −18.66–−18.09% | −0.343–−0.333 |
| 0: pass 0 → XYB | 64 | 8.138–8.552 | −15.44–−13.90% | −38.59–−35.67% | −3.182–−3.036 |
| 1: pass 1 → XYB | 64 | 6.242–6.582 | −21.68–−19.39% | −37.00–−33.98% | −2.412–−2.126 |
| 2: pass 1 → RGB | 64 | 6.703–7.133 | −23.35–−18.65% | −39.48–−36.81% | −2.824–−2.485 |

All sixteen individual raw-shared-versus-control medians per route/burst
agree with the corresponding primary sign at 4K. The pass-0 short-burst
regression is real evidence against indiscriminate fusion. At HD, short
bursts similarly favor raw-shared pass 1: 14.52–16.98% for XYB and
17.94–18.99% for RGB. Sustained HD results have substantial duplicate-label
scatter: only 10/16 and 13/16 individual comparisons favor those routes,
respectively. The original raw, unfiltered 36-job report has 61/72 favorable
raw-shared primary medians, but that pooled count is not a qualification gate.

Long 4K bursts are much slower per chain than short bursts even when all
enforced-limit endpoints read 40 W. Equal limit readings do not establish
equal operating clocks or otherwise stationary GPU state. The 34–39%
long-burst pass-1 savings must not be transferred to a real encoder workload
or described as a general throughput improvement.

## Requested repeats

After the sanitizer/profiler sequence completed, the seven affected cases
were repeated with a new balanced schedule seed, followed by two pad-0 4K
pass-1 cross-checks. They ran from 16:06:55 to 16:09:34 UTC. Every output check
passes; **all 3,888 before/after limit readings are 40,000 mW**, with zero
endpoint changes. These are nine additional jobs, not overwritten originals.
The table shows raw-shared paired median changes; ranges for 500x500 cover
both padding layouts.

| Geometry / route | Four-chain change | 64-chain change |
| --- | ---: | ---: |
| 500x500 / pass 0 → XYB | −2.89–−0.57% | +5.85–+10.33% |
| 500x500 / pass 1 → XYB | −16.16–−13.08% | −2.31–−1.68% |
| 500x500 / pass 1 → RGB | −17.97–−15.55% | −8.14–−5.92% |
| 1919x1079 / pass 0 → XYB | +3.53% | −24.02% |
| 3839x2159 / pass 1 → XYB | −14.55% (−0.248 ms) | −36.28% (−2.413 ms) |
| 3839x2159 / pass 1 → RGB | −17.94% (−0.330 ms) | −36.93% (−2.580 ms) |

All sixteen individual raw-shared/control comparisons in the two 4K repeat
jobs favor fusion. Small-image pass-0 short bursts remain marginal/mixed
against duplicate controls, while all eight long-burst comparisons regress.
Small-image pass-1 XYB long bursts have only 6/8 favorable individual
comparisons despite favorable primary medians; RGB has 8/8. Direct fusion
regresses in every 500x500 long-burst primary median. The repeat therefore
supports the large pass-1 result without turning it into a geometry- and
load-independent win. The sustained-state caveat above still applies even
after the limit readings have settled.

## Counter evidence

Eighteen Nsight Compute captures cover three routes, production/direct/raw-
shared modes and two reversed-order repetitions at 3839x2159, resident
strides. Each executable runs one reference, eight warm chains and one
profiled chain; only the last boundary is inside the profiler region. Output
bits and error flags are checked afterward. Kernel replay is used with clock
and cache control disabled. These captures follow the power transition, are
not concurrent with event timing, and are not presumed to share its state.

All baseline boundaries move about 397 MB of measured DRAM traffic. Direct
fusion reduces that to about 199 MB but increases global-load sectors by
19–37% and executed warp instructions by 3.36–7.10%, because halo work still
loads neighboring samples repeatedly. Raw-shared performs more of this work
locally:

| Raw-shared route | DRAM MB | DRAM change | L2 traffic change | Global-load sectors change | Executed warp instructions change | Achieved occupancy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 211.36–211.66 | −46.79–−46.70% | about −55.05% | about −82.63% | −8.29% | 33.02% |
| 1 | 206.57–207.13 | −47.98–−47.84% | about −60.13% | about −85.76% | −14.58% | 49.42% |
| 2 | 206.48–206.69 | −48.00–−47.95% | about −60.21% | about −85.75% | −14.14% | 49.42–49.43% |

Shared-load wavefronts rise by about 91%, 154% and 156%, respectively.
Predicated thread FFMA counts rise by 2.41%, 3.38% and 2.85% for both fused
alternatives because Gaborish is recomputed in overlapping halos. Less
traffic does not mean less of every kind of work.

Shared-memory limits allow only two resident raw-shared blocks per SM for
pass 0 and three for pass 1. Theoretical occupancy falls from the baseline
EPF's 66.67%, 100% and 66.67% to 33.33%, 50% and 50%. This supplies a concrete
resource cost consistent with the pass-0 short-burst regression, not proof
that occupancy alone causes it. Profiled SM frequencies vary between runs;
profiler durations are not substituted for the paired event measurements.

## Decision and next qualification

Retain both candidates as isolated diagnostic code and advance raw-shared
pass 1 to real fully resident integration testing. Preserve separate
Gaborish/pass-0 for three-iteration filtering and existing zero-EPF behavior.
No compatibility layer, public option or production dispatch change is added
by S128. The next experiment should first hold allocation policy and all
other kernels fixed, verify actual reconstruction/sigma captures and output
contracts, then measure both the in-encoder boundary and uninstrumented
complete encodes. Maximum-error scoring must still receive filtered XYB.

The resident owner currently counts logical filter stages when selecting
scratch images. Fusing its first two stages may allow a later, separately
qualified allocation-lifetime improvement; one 3840x2160 three-channel float
image is 99,532,800 bytes. That is an opportunity, **not memory saved here**.
Pass-0's shared-storage/occupancy cost also motivates a distinct future tile
experiment, not another unchanged rerun. Neither production throughput nor
maxed-out status is established by this isolated campaign.

## Reproduction and audit

Artifacts are under `U:/gjxl-cuda-diagnostics/s128`; drivers and generators
are under `build-cuda-ninja/profiles/s128_*`. `s128_prepare.py` extracts the
S127 EPF and Gaborish bodies, `s128_build.ps1` compiles CUDA 11.8/sm86 without
fast math and links Release or host-ASAN fixtures, and `s128_native.py`
checks the native bodies. `s128_campaign.py` owns the initial guarded-layout
preflights/pilots and CUDA sanitizers. `s128_resident_campaign.py` owns the
resident-pitch preflights, pilots and main campaign; `s128_repeat.py` owns
the requested repeats. Analyses, the separate post-hoc eligibility report,
counter capture/export/analysis and compact result summaries are retained
alongside the raw logs. Output creation is exclusive; reproduce in a fresh
diagnostic root using the frozen sources, not over the original artifacts.

The final validator accepts all 109 recorded jobs and checks log/executable
hashes, generation/input hashes, all twelve unchanged original GPU bodies,
three native-identical controls, one common device module across seven
executables, timing label balance and paired medians, the 29/7 initial-regime
partition, repeat coverage, sanitizer footers, raw counter exports, and forty
retained runtime hashes. The 49 timing jobs overlap no other recorded job.
Across pilots, preflights, main measurements and repeats there are 10,944
checked bursts, 342,720 logical benchmark chains, 58 single-chain references
and 21,888 power-limit endpoints; 432 bursts use host ASAN. The eighteen
counter captures add 180 logical chains. Checks do not imply that every
repeated chain has a separately downloaded output.

Production source/tests/build files remain unchanged from S127. There is no
fresh full-encoder build, CTest campaign or production speed claim in S128.
The final source snapshot and SHA-256 artifact manifest preserve the evidence
for later audit. Racecheck took about eight minutes with continuing child
process heartbeats and successful progress; it was not blocked by a firewall
or elevation prompt. No such blocker appeared during this experiment.
