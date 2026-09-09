# CUDA low/medium parameter-carried weights (S147)

## Scope and mechanism

S146 established that moving coefficients into retired ring rows reduced
the S144 rolling convolution's executed work, with repeatable sustained
stage gains on the two largest inputs. It still executed more instructions
than baseline and increased shared-load traffic. S147 tests passing the
33 weights by value as kernel parameters. Starting commit: `79b7e18`.
Production source remains S143 during this diagnostic study.

Each diagnostic device case owns a 132-byte host weight payload copied from
the same source as its device weight allocation. New launch wrappers take
this payload explicitly; there is no global lookup, fixed Gaussian table,
device-to-host fetch, or compatibility fallback. The unchanged horizontal
convolution and reference paths still read the original device allocation.

The two new vertical kernels retain the 64-row, three-plane shared ring and
24-output-row chunks. Tap FMAs read immutable parameter values directly.
Thread 0 performs the original ordered normalization on the device, once
per CTA. Each chunk publishes that scalar in one unused ring slot through
the existing barrier; the existing read-completion barrier protects it
from the next chunk. No reciprocal, host sum, tap reordering, or new barrier
is introduced. Integer liveness simulation checks all six chunks across
48- and 96-output-row tiles, including wraparound.

## Native code and qualification

The GPU object contains 85 bodies: all 83 prior bodies are instruction-exact,
plus the two new kernels. Ten release/host-ASAN executables (full guards,
scoped guards, replay, counters, and ownership tests) contain identical
multisets of three GPU modules. The two auxiliary modules remain unchanged.

| Vertical body | Registers | Shared bytes | Static instructions | Static shared loads | Local/stack/spills |
| --- | ---: | ---: | ---: | ---: | --- |
| Retained baseline 48 | 54 | 30,856 | 1,736 | 219 | 0 |
| S146 retired weights 48/96 | 55 | 24,576 | 2,080 | 267 | 0 |
| S147 parameter weights 48/96 | 46 | 24,576 | 1,984 | 211 | 0 |

Both new bodies have 651 static FFMAs, 594 with direct constant-parameter
operands, and no explicit LDC or local-array load/store instructions. Their
constant-0 footprint is 604 bytes versus S146's 476 bytes. Static counts are
not executed-work or throughput measurements.
The driver confirms four 256-thread blocks (32 warps) per SM for both new
kernels, versus three blocks (24 warps) for the retained baseline.

The fixture adaptation preserves both permanent oracles and every existing
assertion; it only adds per-device-case weight ownership and changes the
candidate dispatch. Release and host-ASAN each pass 2,760 fixtures across
six labels, with three reuse stages and bitwise checks of sixteen arrays,
guards, and padding. Both pass the flattened-grid tall-only cases.

Separate ownership tests capture vertical graphs for two independent cases
and streams, poison caller-owned and temporary host weight copies, let the
temporary copies expire, and launch both graphs repeatedly. A second
generation changes device and host weights, including negative taps, then
rebuilds and rechecks the graphs. Release and host-ASAN each pass 72 checked
graph executions across both tile variants and three input-pattern pairings.
This is diagnostic evidence for launch/graph value ownership, not an
integrated production-plan API qualification.

Both the scoped fixture suite (180 fixtures) and the ownership suite pass
memcheck, initcheck, synccheck, and racecheck. Neither memcheck reports leaks,
and both race checks report zero hazards. Main race checking takes 592.426
seconds with per-label progress; ownership race checking takes 45.546
seconds. No privilege/firewall blocker is observed.

## Measurement protocol

The ordinary matrix reuses the same 24 hash-checked current-path payloads
captured in S145. All 24 are preflighted in release and host-ASAN at 128
horizontal-plus-vertical pairs per graph. Twelve selected captures receive
timing at both 4 and 128 pairs, with two reverse-order process repetitions.
Six duplicate labels (baseline 0/3, new 48 1/4, new 96 2/5) use a randomized
Williams design with six warmup and twelve measured rounds. Sixteen-array
validation occurs after every graph, outside event timing. These are stage
measurements, not whole-encoder throughput.

The counter comparison uses baseline, S146 48/96, and S147 48/96 on current
full 4K. Each process warms its selected vertical variant ten times, then
profiles one explicitly delimited direct launch. Both variant orders are
kept. Clock and cache controls are disabled. Counter runs diagnose executed
work and resources, not ordinary throughput.

## Ordinary results

All 48 capture preflights and 48 timing processes pass. The 5,184 event
windows include 3,456 measured windows and 228,096 measured horizontal-plus-
vertical pairs. All 11,520 ordinary preflight/timing power-limit endpoints
report 40 W. The replay harness also passes memcheck and initcheck, with
zero leaks. No observations are dropped.

Primary comparisons average the two family-label medians; all four cross-
label candidate/baseline pairs are retained. Negative means faster.

| Burst | Parameter family | Favorable primary | Favorable cross-label | Median delta across captures/repeats |
| --- | --- | ---: | ---: | ---: |
| 4 pairs | 48 rows | 20/24 | 79/96 | -0.95% |
| 4 pairs | 96 rows | 14/24 | 56/96 | -0.65% |
| 128 pairs | 48 rows | 16/24 | 62/96 | -2.70% |
| 128 pairs | 96 rows | 12/24 | 55/96 | -0.20% |

Full-resolution results, preserving both repetitions:

| Input | Burst | Baseline ms/pair | Parameter 48 delta | Parameter 96 delta |
| --- | ---: | ---: | ---: | ---: |
| 3839x2159 | 4 | 2.8861 / 2.8740 | -3.76% / -3.07% | -8.02% / -6.02% |
| 3839x2159 | 128 | 9.1364 / 9.1532 | -1.32% / -1.92% | -4.74% / -4.89% |
| 1919x1079 | 4 | 0.7082 / 0.7057 | -6.17% / -5.98% | -4.88% / -4.43% |
| 1919x1079 | 128 | 1.7066 / 1.8147 | -4.54% / -5.10% | -3.50% / -4.38% |
| 2000x2000 flower | 4 | 1.3907 / 1.3788 | -8.27% / -7.71% | -5.35% / -5.11% |
| 2000x2000 flower | 128 | 3.9972 / 4.0584 | -5.16% / -3.73% | -3.65% / -3.92% |

Every cross-label comparison favors both parameter variants for these three
full-resolution cases, at both burst lengths and repetitions. Packed-output
half 4K also favors both variants across all cross-label pairs: sustained
48 improves -9.95%/-9.12%, and 96 -10.02%/-3.63%. Padded-output half 4K
improves -4.74%/-4.43% for 48, with all cross-label pairs favorable; 96
improves -3.26%/-4.40%, with 3/4 then 4/4 favorable pairs.

Smaller inputs do not support a blanket switch. The 96-row tile regresses
about 22-29% on the 250x250 flower halves, and around 2-3% on short half-HD
and full 500x500. Sustained 48 also regresses on several small cases and
both packed-output 1000x1000 flower repetitions. The matching padded-output
1000x1000 primary gain reverses on repeat. All cases remain in `report.json`.

Duplicate variation remains material: the largest absolute duplicate delta
is -0.103796 ms (-5.56%) for the second sustained-HD baseline. The largest
relative delta is +12.73% (+0.037688 ms) for the first sustained packed-output
half-HD baseline. S146/S147 ordinary results are separate campaigns, not a
single interleaved retired-weight/parameter-weight throughput comparison.
Stage gains do not establish whole-encoder speedups or eliminate the prior
operating-state caveats.

## Executed-work evidence

Ten release/host-ASAN counter preflights, separate memcheck/initcheck, and
ten profiler runs pass with exact outputs. Each profile contains the one
requested direct kernel inside the explicitly delimited region. Both orders
agree on executed instruction totals.

| Vertical body | Executed warp instructions vs baseline | Global-load sectors vs baseline | Shared-load wavefronts vs baseline |
| --- | ---: | ---: | ---: |
| S146 retired 48 | +9.76% | +0.01% | +8.29 to +8.35% |
| S146 retired 96 | +5.53% | -13.07 to -13.05% | +7.84 to +7.90% |
| S147 parameter 48 | +8.13% | -2.08% | -6.84 to -6.62% |
| S147 parameter 96 | +4.22% | -14.16 to -14.14% | -6.95 to -6.79% |

Compared with the corresponding S146 ring, parameter 48 executes 1.49% fewer
warp instructions and parameter 96 1.24% fewer. Shared-load wavefronts fall
13.81-13.97% and 13.57-13.76%, respectively. This is a smaller dynamic
instruction reduction than the static count change, but removes S146's
additional shared-load traffic. The mechanism is supported by both native
constant operands and matched executed-work counters.

DRAM reads remain essentially unchanged relative to S146. Parameter 96
preserves the halo advantage: 12.37-12.46% fewer DRAM read bytes and
7.18-7.25% fewer L2 bytes than baseline. Parameter 48's DRAM reads remain
within 0.1% of baseline. Predicated-on thread FFMAs are identical to S146
and 0.023% above baseline due to retained partial-x lanes. Shared-load bank
conflict counts are lower in both orders, but are not isolated as the sole
performance cause. Observed active warps are 22.46/22.55 for baseline,
30.36/30.42 for parameter 48, and 31.15/31.15 for parameter 96.

## Decision and next work

Parameter-carried weights are a qualified prototype improvement: they
reduce register demand, executed instructions, and shared traffic relative
to retired-row weights, with repeatable sustained stage gains on larger
inputs. No runtime policy is retained in S147. Small-image regressions rule
out blanket replacement, and an integrated production ownership change and
whole-encoder performance qualification are still required.

The next step is an integrated geometry-policy experiment, retaining the
original small-plane body and testing parameter 48/96 on larger planes.
Current `CudaPreparedDeviceButteraugli::PrepareStorage` already creates the
33 CPU weights before uploading them. Retaining that payload in the prepared
plan and passing it through `LaunchPsycho` can supply kernel parameters
without device readback or a global cache. The low/medium plan and tests
should express ownership explicitly; no compatibility layer is needed.
Host argument-marshalling cost, current reference/distorted paths, wide and
compact coefficient modes, complete codestream identity, reuse/failure
invariants, and integrated short/sustained throughput must be qualified.

Evidence root: `build-cuda-ninja/profiles/s147-artifacts/`; source and
drivers: `build-cuda-ninja/profiles/s147_*` and `*_s147.py`. All 156 recorded
jobs are terminal and accepted, including twelve CUDA sanitizer jobs.
The 135 GPU jobs are non-overlapping; independent streams are exercised
within the dedicated ownership jobs. Ten linked executables share the same
three GPU modules. The forty historical retained runtime files keep their
hashes. `verify_s147.py` checks the fixture adaptation, source, inputs,
native bodies/modules, jobs, timing balance, counters, and frozen identities;
`freeze_s147.py` archives the evidence and source dependencies.

The counter driver's v2 only gives its independent module extraction a
different output directory from the expanded native audit; prior extraction
files and the original driver remain untouched. No clocks, power limits,
thermal policy, firewall, affinity, or priority settings changed. No
privilege/firewall blocker occurred. Protected scratch files remain
untouched. The encoder is not established to be maxed out.
