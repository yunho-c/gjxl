# CUDA rolling-row sustained-work qualification (S145)

## Scope

S144's rolling low/medium convolution improved isolated large-image short
bursts, but increased executed instructions and regressed smaller images.
S145 tests the actual horizontal-plus-vertical pair with 4 or 128 consecutive
pairs per graph, using fresh current fully resident inputs. It does not
change production source or select a production tile policy.

S144 qualified candidates and narrowed the next question to sustained work
and operating state. The starting commit is `236c0f9`; S143 remains the
retained runtime implementation.

## Current-path capture and binary fidelity

A diagnostic-only host hook in the normal Butteraugli launch wrapper captures
every low/medium call in one encode. Its 221 native kernel bodies exactly
match S143. Four capture executables (wide/compact, release/host ASAN) contain
identical multisets of ten GPU modules; nine modules match S143 byte-for-byte,
and the remaining module differs in symbol/source identity, not instructions.
The two replay executables link the frozen S144 prototype object and contain
the same three GPU modules as the previously qualified S144 probes.

Eight release capture encodes cover wide/compact storage for flower 500x500,
flower 2000x2000, padded HD, and padded 4K. Two host-ASAN capture encodes cover
the smaller flower and 4K in wide storage. Every encode matches its frozen
codestream oracle and verifies the requested coefficient storage width.
Each produces six captures, for 60 files total. Capture synchronization and
readback are deliberately intrusive; capture runtimes are not performance
evidence.

The files reduce to 24 unique payloads: corresponding wide, compact, and
ASAN captures are byte-identical. All 24 are replay-preflighted. The twelve
fresh HD/4K payloads are not byte-identical to same-geometry historical S73
captures. S144/S145 timings must not be interpreted as same-input paired
measurements or as isolating the cause of the payload differences.

## Preregistered ordinary protocol

The three families are retained baseline, rolling 48, and rolling 96.
The unsuccessful warp-weight-only family is excluded. Labels 0/3, 1/4, and
2/5 are dispatch-identical duplicates. Every process has six warmup and
twelve measured rounds in a randomized six-label Williams design, with
balanced positions and ordered predecessor pairs in each six-round block.

The 48 release/ASAN preflight processes cover every unique payload with
128-pair graphs. The 48 timing processes cover twelve selected captures,
both burst lengths, and two opposite-order process repetitions. For each
image case, selected call indices are 0 (reference main), 1 (reference half),
and 3 (distorted half with padded output). Other captured payloads receive
preflight coverage but not ordinary timing coverage in this matrix.

Every graph is followed by exact checks of all sixteen arrays, including
input, output, unused storage, and guards. These checks/readbacks are outside
CUDA-event timing and create host gaps between bursts. Both burst lengths
measure the same complete low/medium horizontal-plus-vertical pair, not the
entire encoder. Enforced-power-limit queries are read-only endpoints and do
not measure active-kernel clocks.

The new replay harness passes memcheck and initcheck; memcheck reports no
leaks. All 48 sustained preflights pass. Kernel correctness, synchronization,
and race evidence from S144 applies through the exact linked-module match;
S145 does not introduce a new convolution body.

## Cycle-rate diagnostic

A separate diagnostic executable adds the hash-checked S119 aligned probe
module, leaving all three replay modules unchanged. It samples twenty
microseconds before and after pairs 0, burst/2, and burst-1. Each probe reads
the GPU global timer, local cycle counter, and SM identifier in the same
ordered sequence, with aligned timer endpoints and checked device guards.

Here labels 0/1/2 have no probes and labels 3/4/5 are their probe-observed
counterparts; these are **not** duplicate-control labels. Observer effects
must be reported separately from ordinary performance. Four instrumented
processes cover full and half 4K with both burst lengths, after the ordinary
matrix finishes. Samples describe adjacent probe kernels, not instructions
inside convolution, and are not independently measured physical SM clocks.

## Ordinary results

All 48 timing processes complete with exact outputs. They contain 5,184
event windows, including 3,456 measured windows and 228,096 measured
horizontal-plus-vertical pairs. All 11,520 preflight/timing power-limit
endpoints report 40 W. Primary comparisons average the two family-label
medians; cross-label comparisons retain all four baseline/candidate pairs.
Negative deltas mean faster. No observations are discarded.

| Burst | Family | Favorable primary comparisons | Favorable cross-label pairs | Median delta across captures/repeats |
| --- | --- | ---: | ---: | ---: |
| 4 pairs | Rolling 48 | 9/24 | 40/96 | +0.53% |
| 4 pairs | Rolling 96 | 10/24 | 41/96 | +0.20% |
| 128 pairs | Rolling 48 | 5/24 | 20/96 | +2.67% |
| 128 pairs | Rolling 96 | 8/24 | 30/96 | +4.55% |

Representative full-resolution comparisons, preserving both process repeats:

| Input | Burst | Baseline ms/pair | Rolling 48 delta | Rolling 96 delta |
| --- | ---: | ---: | ---: | ---: |
| 3839x2159 | 4 | 2.8732 / 2.8713 | -2.15% / +1.49% | -5.25% / -3.29% |
| 3839x2159 | 128 | 9.0963 / 9.1428 | +3.62% / +3.67% | +0.25% / +0.36% |
| 1919x1079 | 4 | 0.7126 / 0.7106 | -4.94% / -5.58% | -3.56% / -3.56% |
| 1919x1079 | 128 | 1.7083 / 1.7005 | +1.23% / +2.33% | -0.17% / +1.75% |
| 2000x2000 flower | 4 | 1.3531 / 1.4339 | -4.38% / -6.40% | -4.49% / -4.00% |
| 2000x2000 flower | 128 | 3.7927 / 3.9155 | +1.84% / +0.77% | -0.46% / -0.39% |

The packed-output half-4K case retains a small rolling-96 sustained gain
(-1.83%/-1.56%, all eight cross-label pairs favorable), but this does not
justify a general geometry rule. The similarly sized full-HD case does not
reproduce it, and most smaller inputs regress. All twelve selected captures,
both output-stride layouts, and both repetitions are retained in `report.json`.

Duplicate controls are not uniformly tight. The largest absolute duplicate
delta is -0.125952 ms for rolling 48 in the second short full-4K process.
The largest relative duplicate delta is +15.83% for rolling 48 in the first
sustained 1000x1000 packed-output flower process. The apparent -10.37%
rolling-48 gain on sustained 500x500 flower reverses to +3.80% on repeat;
it is not a retained speedup.

## Decision and next hypothesis

The cycle diagnostic completes eight release/ASAN preflights, four timed
processes, and separate memcheck/initcheck runs. Its four timed processes
record 1,368 valid same-SM samples (864 measured, 432 warmup, 72 graph-qualification samples),
with no guard changes or migrated samples. All 912 timed-process power-limit
endpoints report 40 W. Baseline medians from the sustained observations are:

| Input | Pair in burst | Before / after cycles per global-timer ns | Interval between adjacent probes (ms) |
| --- | ---: | ---: | ---: |
| Full 4K | 0 | 1.0938 / 1.0929 | 3.0075 |
| Full 4K | 64 | 0.2257 / 0.2267 | 9.6282 |
| Full 4K | 127 | 0.2110 / 0.2110 | 10.2948 |
| Half 4K | 0 | 1.2824 / 1.2814 | 0.7076 |
| Half 4K | 64 | 0.2647 / 0.2599 | 2.1601 |
| Half 4K | 127 | 0.2560 / 0.2560 | 2.2333 |

The rolling variants also end near 0.210-0.211 cycles/ns for full 4K and
0.258-0.259 for half 4K. These observations establish a substantial change
in local cycle/timer ratio during the instrumented sustained bursts. They
support an operating-state explanation for the time variation, but do not
isolate the physical cause or prove that adjacent-probe rates equal the
convolution's effective rate throughout its execution. A constant reported
power limit is not a constant-frequency guarantee.

Probe-on/off median differences are material: +1.36% to +8.02% in the short
processes, -3.01% to +3.96% in the sustained processes. These include both
instrumentation and possible operating-state differences; they are not
ordinary speedup estimates or a causal measurement of probe overhead alone.
The full sample records, per-family rates, interval gaps, and observed/
unobserved timings are preserved in `clock_analysis.json`.

No new runtime policy is retained. The short-burst large-image improvement
does not survive this sustained full-pair qualification reliably. The
instruction/resource tradeoff identified in S144 remains the relevant target.
An untested next design is to store shared weights and normalization in the
eight ring rows outside the currently live 56-row window, moving their
logical location each chunk. Per-thread retained coefficient values could
repopulate those retired slots without reloading/summing weights per warp.
This would target S144's extra instruction/load work while retaining the
24 KiB shared-memory footprint. Ring-slot liveness, barriers, register use,
native instructions, exactness, and sustained performance all require testing;
the idea itself is not an improvement claim.

Evidence root: `build-cuda-ninja/profiles/s145-artifacts/`; drivers and generated
sources: `build-cuda-ninja/profiles/s145_*`, with late reporting/verification
helpers named `*_s145.py`. Historical artifacts and the user's unrelated
scratch files remain untouched. No clocks, power limits, thermal policy,
firewall, affinity, or priority settings were changed. No privilege/firewall
blocker occurred. All 134 recorded jobs are terminal and accepted. The forty
retained historical runtime files keep their recorded hashes. `verify_s145.py`
rederives the numerical reports and checks the source, input, linked-module,
job-log, sample, and artifact identities; `freeze_s145.py` archives the evidence
and its source dependencies. The encoder is not established to be maxed out.
