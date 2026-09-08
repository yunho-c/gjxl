# Gaborish / first-EPF shared-storage reuse (S130)

All-channel reuse is correct and improves the measured resident filter
stage by 10.6-18.7%. Select it for production qualification; reject the
channel-wise schedule. Whole-encode changes remain inconclusive, and this
study does not change the S127 production default.

## Question and scope

[S129](cuda-gaborish-epf-integration-s129.md) did not establish a dependable
resident-encoder gain from S128's raw-shared Gaborish / first-EPF fusion.
Although the kernel reduced work and DRAM traffic, it kept both the raw and
filtered three-channel tiles in shared memory. Its captured pass-1 occupancy
was about 49%, versus about 99% for the unfused EPF kernel. This study tests
whether reusing the shared allocation improves that tradeoff.

The experiment remains an isolated CUDA object and resident-owner overlay.
It does not change allocation policy, compact coefficient defaults, public
APIs, production source, tests or CMake. The original S127 runtime and frozen
oracles are reused. There is no compatibility layer or new public switch.

## Two reuse schedules

Both designs retain 32-by-32 output tiles, 256 threads, the existing Gaborish
arithmetic, canonical boundary mirroring, and the EPF / optional color
arithmetic. A flat `__shared__ float storage[3 * kRawSize]` first holds the
raw input window. The filtered halo is calculated into thread-local values;
after all relevant readers finish, it is stored into a compact prefix of
the same allocation for EPF consumption. This uses neither a union nor
overlapping object lifetimes.

All-channel reuse holds the three channels' filtered values in registers
before one read-completion barrier and the compacted stores. Channel-wise
reuse computes and stores one channel at a time, with a read-completion
barrier per channel. Its compacted channel `c` ends no later than raw
channel `c + 1` begins because `kTileSize <= kRawSize`. Thus those stores
cannot overwrite the next channel before it is read. The final barrier
completes all filtered stores before EPF starts. No thread exits before
these collective barriers.

The raw-shared design executes two block barriers, all-channel reuse three,
and channel-wise reuse five. Fully unrolled local slots keep the temporary
arrays out of local memory in the compiled variants. These are resource and
synchronization changes, not evidence of a speedup by themselves.

| Route | Old raw-shared registers / shared bytes | All-channel reuse | Channel-wise reuse |
| --- | ---: | ---: | ---: |
| Pass 0 to XYB | 56 / 36,528 | 58 / 19,200 | 59 / 19,200 |
| Pass 1 to XYB | 40 / 32,880 | 46 / 17,328 | 40 / 17,328 |
| Pass 1 to linear RGB | 55 / 32,880 | 58 / 17,328 | 56 / 17,328 |

All six new variants have zero compiler-reported stack, spill stores and
spill loads. These shared-byte counts exclude the driver's per-block
reservation; profiler occupancy must account for that reservation too.

The isolated launcher has five modes: production, native-identical EPF
clone, old raw-shared fusion, all-channel reuse and channel-wise reuse.
The real encoder compares four variants, omitting the clone. The encoder
overlay fuses only Gaborish with first pass 1 when the profile requests one
or two EPF iterations. Maximum-error evaluation retains XYB ownership;
one-EPF perceptual evaluation may finish in linear RGB. No-Gaborish,
zero-EPF and three-EPF profiles retain their existing dispatch. The prior
pass-0 fusion regression is not silently enabled.

## Correctness and provenance

The full guarded fixture runs 24 geometries, two unequal-pitch layouts,
three routes, sixteen input patterns, two Gaborish parameter sets, three
changed-input reuses and five modes. Release and host ASAN each execute
69,120 comparisons with 19-float prefixes and 37-float suffix guards.
The patterns include nonfinite and extreme values, subnormals, exact sigma
bypass boundaries and adjacent representable values. Output bits, all
guards, immutable input/sigma data, error flags and scratch contents are
checked; fused modes must leave global Gaborish scratch unchanged.

The smaller aligned sanitizer population covers 1-by-1, 33-by-33 and
65-by-17 images with both layouts and all remaining factors: 8,640
executions per run. Ordinary preflight and CUDA memcheck, racecheck,
initcheck and synccheck bring the guarded total to 181,440 executions.
Racecheck completed with zero hazards, and the other tools reported zero
errors. Memcheck also reported zero leaked bytes.

The six S129 captures are reused with their original data, filter parameter
bits and row pitches. Each replay compares both XYB and RGB routes,
aligned and 19-float-offset layouts, three buffer reuses and all five
modes. The RGB replay uses the captured two-EPF boundary's inputs; it is
not presented as a capture from a one-EPF profile.

SASS comparison verifies that the 27-kernel diagnostic module contains all
21 S128 bodies unchanged plus six new reuse bodies. Full encoder binaries
retain the other nine S127 GPU modules byte-for-byte, for 228 linked kernel
bodies. Standalone and full-encoder Release / ASAN binaries use the same
new GPU module. No fast-math flags are added.

## Measurement design

The captured-data counter probe uses the aligned 3839-by-2159 S129 call-0
input and actual filter parameters. Each process runs a baseline reference,
eight warm chains and one profiled chain. Its final output, input guards,
error flags and fused scratch are checked. Four variants are captured in
forward and reverse order, using kernel replay, clock control `none` and
cache control `none`. These short profiler durations are not substituted
for resident-encoder measurements.

The encoder comparison uses Flower 500, padded HD (1919 by 1079) and
padded 4K (3839 by 2159), with the same frozen codestream oracles as S129.
Each process includes two labels per variant. An eight-label Williams
schedule balances every label across positions and every distinct ordered
within-round predecessor pair over eight rounds. Label and row orders are
seeded and shuffled per eight-round block. There are eight warm rounds,
then sixteen measured rounds, and two independently ordered process
replicates per image. A reference plus 24 eight-label rounds yields 193
exact encodes per measured process.

Paired differences use the average of the two labels for each variant
within a round, then report the median across measured rounds. The two
labels also provide an unchanged duplicate-control difference for each
variant. Four individual label-pair comparisons are retained alongside
each averaged comparison. The event-instrumented and uninstrumented
populations are separate: the former sums the two Gaborish / first-EPF
boundaries per encode; the latter measures the complete public encode.
Every encode checks frozen bytes, summary and coefficient storage width /
size, and verifies that only the selected filter branch ran.

NVML enforced-power-limit samples bracket every encode outside the timed
boundary. They detect endpoint limit changes, not constant GPU clocks or
absence of intermediate changes. No power, clock, firewall or privilege
settings are changed. All timed processes must overlap no other recorded
job. CPU-only build work is allowed alongside earlier correctness tests,
not alongside measurements.

## Captured-data counters

Both reuse schedules raise theoretical occupancy from three blocks / 50%
to five blocks / 83.33%. Achieved occupancy is about 82.4% for all-channel
and 82.6% for channel-wise reuse, versus 49.4% for old raw-shared fusion.
Thus the intended occupancy improvement is measured, not merely predicted
from the source allocation size.

| Variant | Executed warp instructions | DRAM MB, two captures | Profiled ms, two captures |
| --- | ---: | ---: | ---: |
| Baseline Gaborish + EPF | 216,729,529 | 396.84 / 396.85 | 1.880 / 1.882 |
| Old raw-shared fusion | 185,139,360 | 207.23 / 207.31 | 1.682 / 1.650 |
| All-channel reuse | 186,696,024 | 203.48 / 203.14 | 1.401 / 1.571 |
| Channel-wise reuse | 316,900,464 | 205.65 / 205.75 | 2.742 / 2.742 |

Instruction counts repeat exactly. All-channel reuse adds only about 0.84%
warp instructions to old fusion, retaining about a 13.9% reduction versus
the two-kernel baseline. Channel-wise reuse adds about 71.2% to old fusion
despite the same improved occupancy. Predicated-on FFMA thread instruction
counts are identical across all three fused variants (419,853,403), versus
406,131,649 for the baseline. The channel-wise increase is therefore not
additional FFMA work. Its repeated channel-local coordinate / loading
schedule and synchronization do not provide an efficient implementation of
the shared-capacity saving.

The all-channel main XYB body has 2,744 static SASS instruction slots and
three `BAR.SYNC` instructions; channel-wise has 5,296 and five, versus
1,152 and two for old raw-shared fusion. These counts include 14, 14 and
nine NOP slots respectively. No instruction-cache counters were collected,
so body growth is not asserted to be a measured instruction-cache stall
cause. Likewise, higher occupancy alone does not establish the cause or
magnitude of a timing improvement.

Clock differences still matter in these short captures. All-channel SM
clocks are about 1446 and 1277 MHz, old fusion about 1411 and 1443 MHz,
and channel-wise about 1277 and 1276 MHz. These are not clock-matched
comparisons, and no clock normalization is applied. Counter timings favor
all-channel reuse and reject the channel-wise schedule, but the complete
resident encoder remains the qualification boundary.

## In-encoder stage results

All-channel reuse improves the summed two-boundary stage in all six
measured image / replicate combinations. Negative differences favor reuse.
The baseline and reuse values are paired within each eight-label round;
medians of different paired comparisons are not algebraically additive.

| Image / replicate | Baseline ms | All-channel change ms | Change % | Duplicate baseline / reuse differences ms |
| --- | ---: | ---: | ---: | ---: |
| Flower / 0 | 0.161248 | -0.026112 | -16.52 | +0.002560 / -0.001024 |
| Flower / 1 | 0.169984 | -0.032512 | -18.65 | +0.004608 / +0.003072 |
| HD / 0 | 1.094912 | -0.190720 | -17.65 | -0.035328 / +0.028160 |
| HD / 1 | 1.195008 | -0.196352 | -16.89 | +0.009216 / -0.000512 |
| 4K / 0 | 12.705536 | -1.809152 | -13.92 | -0.057344 / -0.695808 |
| 4K / 1 | 11.609600 | -1.200896 | -10.57 | +0.339968 / +0.594944 |

All 24 individual all-channel-versus-baseline label-pair median differences
are also negative. At 4K they range from -1.883 to -1.266 ms in replicate 0
and -1.784 to -0.922 ms in replicate 1. All-channel reuse improves directly
over old raw-shared fusion in all six stage comparisons too: about 6-8%
on Flower, 12-14% on HD and 9-12% on 4K.

Old raw-shared fusion's 4K stage regresses by 5.12% and 4.54% against the
baseline in this population. Channel-wise reuse regresses the stage in
every case: about 32-39% on Flower, 45-49% on HD and 50-54% on 4K. The
shared-memory capacity reduction alone is therefore not an adequate
selection criterion.

## Complete-encode results and decision

The separate uninstrumented population does not show a repeatable total
encode improvement. Every image's all-channel change reverses sign between
process replicates, with substantial unchanged-control differences.

| Image / replicate | Baseline ms | All-channel change ms | Change % | Duplicate baseline / reuse differences ms |
| --- | ---: | ---: | ---: | ---: |
| Flower / 0 | 20.088175 | +0.470475 | +2.35 | +0.292900 / -0.240100 |
| Flower / 1 | 20.995125 | -0.392125 | -1.88 | +0.577600 / -0.316600 |
| HD / 0 | 84.639250 | -2.521050 | -3.11 | -0.501100 / -0.187350 |
| HD / 1 | 83.219350 | +1.046850 | +1.27 | -0.343950 / -2.436400 |
| 4K / 0 | 345.925375 | -1.389075 | -0.32 | +0.408150 / -15.897550 |
| 4K / 1 | 358.064900 | +4.806675 | +1.39 | +17.942800 / -2.473900 |

The instrumented population's whole-encode all-channel changes at 4K are
also mixed (-10.280 and +7.254 ms). The stage improvement must not be
reported as an equivalent whole-encode percentage or used to select one
favorable total-time replicate. This study establishes a useful local
optimization, not a dependable complete-workflow speedup.

All 4,848 timing / preflight enforced-power-limit endpoints are 40 W, with
zero before/after endpoint changes. No firewall or elevation blocker was
encountered. Fixed endpoint limits did not eliminate total-time variation;
the study does not identify a single cause of that variation.

Advance all-channel pass-1-to-XYB reuse to a clean production integration
and its own baseline/candidate qualification. It fixes the occupancy
penalty without the channel-wise schedule's large added instruction work,
and its local gain now survives actual encoder inputs and paired repeats.
Do not promote old raw-shared fusion, channel-wise reuse, pass-0 fusion or
an unmeasured geometry threshold. Pass-0 and pass-1-to-RGB variants are
correctness-tested here but do not receive a new performance qualification
from this default two-EPF-to-XYB timing population.

Keep scratch allocation changes separate from this kernel selection.
Production integration must preserve maximum-error XYB ownership, bypass
profiles, numerical flags and frozen outputs, and must audit the actual
linked runtime rather than assuming the diagnostic object will compile
identically after refactoring. The backend has not been established to be
maxed out.

## Reproduction and final audit

Artifacts are under `U:/gjxl-cuda-diagnostics/s130`; diagnostic sources and
generators are under `build-cuda-ninja/profiles/s130_*`. The core entry
points are `s130_prepare.py`, `s130_core.py`, `s130_encoder_prepare.py`,
`s130_hot_prepare.py`, `s130_matrix_prepare.py` and `s130_after_core.py`.
The latter begins only after the core checks and native audits have
completed, then serializes the encoder matrix, preflights, counters and
both measurement populations. Outputs are created exclusively. Reproduce
from the frozen source snapshots in a new root, not by overwriting this
evidence or rebuilding its historical baselines.

The final validator checks all 72 recorded jobs, all accepted, and verifies
that the twelve timed jobs overlap no other recorded job. It checks source,
input, oracle, executable and log hashes; native module/body identities;
static instructions and resource records; guarded populations; matrix
coverage; raw counter exports; timing schedule balance, branch counts,
paired statistics and power endpoints; and forty retained runtime hashes.

Coverage totals are 181,440 guarded kernel executions, 1,080 guarded
captured-input replay executions, 2,424 frozen-oracle encodes (54 under
host ASAN), and 768 compared AQ profile pairs. Those pairs include 48
successfully serialized pairs and 720 matching writer-rejection pairs;
rejected profiles are not counted as successful encodes. Six CUDA sanitizer
jobs are clean: four on the guarded kernels plus replay and profile-matrix
memcheck. Counter processes add 80 logical chains and host-ASAN counter
preflights add forty; each ten-chain process checks its final output, not
every intermediate warm output.

The first unrecorded final-validator invocation found a checker-only JSON
comparison mismatch: integer power-limit histogram keys become strings
when serialized. Normalizing the recomputed result through the same JSON
representation fixes the comparison. The original checker and exact
single-line correction are preserved; no experimental source, binary,
counter or measurement was changed or rerun for this correction. All 72
recorded experiment jobs had already passed.

`final_summary.json`, source snapshots and the SHA-256 artifact manifest
preserve the final decision and evidence. Production source/tests/build
files remain unchanged, and there is no fresh full CTest run or new default
runtime in this isolated study.
