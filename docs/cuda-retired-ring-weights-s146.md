# CUDA low/medium weights in retired ring rows (S146)

## Scope and mechanism

S145 did not retain the S144 rolling-row convolution: short-burst gains did
not reliably survive sustained horizontal-plus-vertical work. S146 tests
whether removing its warp-distributed weight overhead improves that tradeoff.
Starting commit: `83ffb99`. Production remains S143 pending qualification.

The diagnostic kernels retain the three 64-by-32 shared float planes and
24-output-row chunks. Each chunk consumes 56 rows, leaving eight unused.
Thirty-three weights and the ordered normalization occupy 34 consecutive
float slots in two of those unused rows of the first plane. Their starting
rows for the four possible chunks are 56, 16, 40, and 0. They do not overlap
the current loaders or any live pixel. An integer simulation checks every
read and write across both 48- and 96-output-row tiles, including wraparound.

Threads 0 through 32 each retain one weight; thread 0 computes the original
ordered normalization once per CTA. Each chunk republishes them at its new
location. The existing loader-publication barrier also publishes the weights;
the existing read-completion barrier protects retired rows from overwrite.
No barrier or shared allocation is added. The nine original ordered FMA
chains, divisions, output reconstruction, and signed input addressing remain.
Partial-x lanes continue through the cooperative barriers.

## Native code and correctness

The GPU object contains 83 kernel bodies: all 81 S144 bodies match their
previous instructions exactly, plus the two new kernels. The six release/
host-ASAN guard, scoped-guard, and replay executables contain identical
multisets of three GPU modules, with the two auxiliary modules unchanged.

| Vertical body | Registers | Shared bytes | Static instructions | Local/stack/spills |
| --- | ---: | ---: | ---: | --- |
| Retained 48 | 54 | 30,856 | 1,736 | 0 |
| S144 ring 48/96 | 42 | 24,576 | 3,680 | 0 |
| S146 retired weights 48/96 | 55 | 24,576 | 2,080 | 0 |

The static reduction is not an executed-instruction or throughput claim.
S146 eliminates all shuffle instructions in these two bodies; both retain
the same 651 static FFMAs as the baseline, versus 1,035 in the S144 bodies.
Compiler-generated control-flow alternatives contribute to static counts.
The driver confirms four 256-thread blocks (32 warps) per SM for both new
kernels, versus three blocks (24 warps) for baseline; the extra registers
do not reduce the rolling design's occupancy capacity on this device.

Release and host-ASAN each pass 2,760 fixtures: six labels times 460 fixtures.
Every fixture checks three buffer-reuse stages against both established
reference paths, bitwise across all sixteen arrays and guard/padding storage.
Coverage includes signed zero, extreme magnitudes, NaN/infinity, alternate
weights, tiny/odd dimensions, and packed/padded layouts. Both also pass the
flattened-grid tall-only fixtures. Memcheck, initcheck, synccheck, and
racecheck each pass 180 scoped fixtures covering 1x97, 33x97, and 65x193,
both stride layouts, five patterns, and six labels. Racecheck reports zero
hazards; memcheck reports zero leaks. Its 602-second runtime was active
instrumentation, with per-label progress, not an observed privilege blocker.

## Performance protocol

The ordinary experiment uses the same 24 immutable current-path capture
payloads as S145. All are preflighted in release and host-ASAN with 128-pair
graphs. Twelve selected captures cover main reference, half reference, and
half distorted/padded output from each of the four image cases. Each is
timed at 4 and 128 consecutive horizontal-plus-vertical pairs, with two
opposite-order process repetitions. These are stage timings, not full-encoder
throughput measurements.

Labels 0/3, 1/4, and 2/5 are dispatch-identical baseline, new 48, and new 96
duplicates. Every process uses a randomized six-label Williams design with
six warmup and twelve measured rounds, balanced positions and predecessor
pairs. Every graph receives sixteen-array bitwise validation outside event
timing. Read-only enforced-power-limit endpoints accompany each graph.

The separate counter harness compares baseline, old 48/96, and new 48/96 on
the current full-4K payload. Each variant receives ten direct vertical
warmups followed by one explicitly delimited direct target launch. Profiler
collection begins only at that region; both variant orders are preserved.
Clock control and cache control are disabled. Counter runs provide structural
evidence, not ordinary throughput estimates.

## Ordinary results

All 48 release/host-ASAN capture preflights and all 48 timing processes pass.
There are 5,184 event windows, including 3,456 measured windows and 228,096
measured horizontal-plus-vertical pairs. All 11,520 ordinary preflight/timing
power-limit endpoints report 40 W. The replay harness also passes memcheck
and initcheck, with zero leaks. No observations are discarded.

Primary comparisons average the two family-label medians; cross-label
comparisons preserve all four candidate/baseline pairs. Negative is faster.

| Burst | New family | Favorable primary | Favorable cross-label | Median delta across captures/repeats |
| --- | --- | ---: | ---: | ---: |
| 4 pairs | 48 rows | 18/24 | 70/96 | -1.13% |
| 4 pairs | 96 rows | 14/24 | 56/96 | -0.97% |
| 128 pairs | 48 rows | 17/24 | 63/96 | -2.82% |
| 128 pairs | 96 rows | 15/24 | 60/96 | -1.16% |

Full-resolution results, preserving both process repetitions:

| Input | Burst | Baseline ms/pair | New 48 delta | New 96 delta |
| --- | ---: | ---: | ---: | ---: |
| 3839x2159 | 4 | 2.8844 / 3.0083 | -4.88% / -6.62% | -6.54% / -8.75% |
| 3839x2159 | 128 | 9.1499 / 9.1475 | -1.17% / -1.39% | -2.83% / -1.94% |
| 1919x1079 | 4 | 0.7137 / 0.7065 | -6.07% / -5.63% | -5.72% / -4.74% |
| 1919x1079 | 128 | 1.7381 / 1.7161 | -4.40% / +0.99% | -1.94% / -1.92% |
| 2000x2000 flower | 4 | 1.3592 / 1.3952 | -5.22% / -7.34% | -5.04% / -5.33% |
| 2000x2000 flower | 128 | 4.0476 / 4.0845 | -3.23% / -3.07% | -4.20% / -3.05% |

All cross-label comparisons favor both new variants for full 4K and full
2000x2000, at both burst lengths and repetitions. This is stronger evidence
than the old rings' S145 sustained results, although S145 and S146 are separate
campaigns, not a single interleaved old/new throughput comparison.

HD and half-resolution cases remain mixed. Sustained HD's 48-row primary
gain reverses on repeat, with only 2/4 then 1/4 favorable cross-label pairs;
96 rows has 2/4 then 3/4 despite favorable primary averages. Packed-output
half 4K also reverses for 48 rows (-6.71% to +1.18%). Its 96-row primary gains
shrink from -3.92% to -0.46%. Padded-output half 4K favors 48 rows by -2.41%/
-2.65%, but its first repetition has only 3/4 favorable cross-label pairs.

Tiny flower halves consistently regress: new 96 is about 24-31% slower,
and new 48 about 2-4% slower. Full 500x500 also regresses in short bursts.
Duplicate variation is material elsewhere: the largest absolute duplicate
delta is -0.206688 ms (-11.22%) for the first sustained-HD baseline; the
largest relative delta is -16.38% (-0.034268 ms) for the first sustained
500x500 baseline. The -17.65% new-48 primary result on one half-HD sustained
repeat must not be generalized. Every case and duplicate remains in
`analysis.json` and `report.json`.

## Executed-work evidence

The counter harness passes ten release/host-ASAN preflights and separate
memcheck/initcheck runs. Its two executables contain the same three GPU
modules as the six guard/replay executables. All ten profiler runs complete
with exact outputs and the requested single target inside the explicit
profiler region. Both orders agree on executed instruction totals.

| Vertical body | Executed warp instructions vs baseline | Global-load sectors vs baseline | Shared-load wavefronts vs baseline |
| --- | ---: | ---: | ---: |
| S144 ring 48 | +16.37% | +14.46 to +14.47% | -6.83 to -6.76% |
| S144 ring 96 | +10.14% | -5.67 to -5.66% | -7.24 to -7.20% |
| S146 new 48 | +9.76% | +0.07 to +0.08% | +8.61 to +8.71% |
| S146 new 96 | +5.53% | -13.07 to -13.04% | +7.97 to +8.04% |

Relative to the corresponding old ring, new 48 executes 5.68% fewer warp
instructions and new 96 4.18% fewer. Global-load sectors fall 12.57% and
7.81-7.85%, respectively, while shared-load wavefronts rise about 16-17%.
Thus shared coefficient loads replace warp-distribution work; the much
larger static-code reduction is not the dynamic reduction. Both still
execute more instructions than the retained baseline.

New 96 retains the old ring's halo-traffic advantage: DRAM reads are
12.32-12.36% below baseline and L2 bytes 7.08-7.11% below. New 48's DRAM/L2
traffic is essentially unchanged. Predicated-on thread FFMAs are identical
between old/new rings and 0.023% above baseline, consistent with retaining
partial-x lanes. Shared bank conflicts do not consistently improve; all
raw counts are retained rather than attributing the gain to fewer conflicts.
Observed active warps are 22.38 for baseline, 30.46-30.49 for new 48, and
31.13-31.15 for new 96. The occupancy-capacity gain is realized here.

## Decision and next hypothesis

Retired-row coefficient storage works and improves the rolling design's
executed-work tradeoff. Full 4K and full 2000x2000 now show repeatable
sustained stage gains, making large-plane use a candidate for further
qualification. No production policy is retained in S146: blanket replacement
is contradicted by small-image regressions, HD controls remain noisy, and
there is no integrated encoder-throughput qualification yet.

The remaining extra instructions and shared coefficient traffic suggest a
bounded follow-up: pass the 33 weights by value as kernel parameters, testing
whether constant-parameter operands remove coefficient publication/loads
while retaining the 24 KiB ring. The existing ordered device normalization
can remain, avoiding a reciprocal or host/device arithmetic change. This
would require a separately qualified ownership/API change for production;
it must not silently hardcode Gaussian weights or introduce compatibility
fallbacks. It is an untested hypothesis, not a claimed improvement.

Evidence root: `build-cuda-ninja/profiles/s146-artifacts/`; drivers and
generated sources: `build-cuda-ninja/profiles/s146_*` and `*_s146.py`.
All 152 recorded jobs are terminal and accepted; the 129 GPU jobs are
verified non-overlapping. Eight CUDA sanitizer jobs pass. The forty retained
historical runtime files keep their recorded hashes. `verify_s146.py`
rederives the timing and counter reports and checks source, input, native
body/module, job-log, and artifact identities. `freeze_s146.py` archives the
evidence and source dependencies. No clocks, power limits, thermal policy,
firewall, affinity, or priority settings changed. No privilege/firewall
blocker occurred, and protected scratch files remain untouched. The encoder
is not established to be maxed out.
