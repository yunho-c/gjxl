# CUDA low/medium rolling-row experiment (S144)

## Scope and hypothesis

This is a diagnostic experiment against S143 (`25a599f`), not an integrated
runtime change. S143's fresh production-label 4K traces attribute 10.261 / 9.211
ms to the low/medium **vertical** convolution (wide / compact snapshots). The
S143 report originally called this horizontal; S144 corrects that label without
altering its frozen evidence. Those snapshots are not a controlled comparison
of coefficient layouts.

The retained vertical kernel computes three adjacent output rows per warp,
sharing each loaded sample across up to three output rows while preserving
the ordered 33-tap sums and divisions. Its 32-column by 48-output-row tile
stages three 80-row input planes. S62 already explored simpler changes to
tile height, output rows per thread, and thread count; S73/S92 explored
channel phasing. S144 instead retains all three channels together and rolls
a shared row window within each block.

Four diagnostic families have two dispatch-identical labels each:

| Family | Labels | Output rows/block | Registers/thread | Shared bytes | API maximum blocks/SM |
| --- | --- | ---: | ---: | ---: | ---: |
| Retained baseline | 0 / 4 | 48 | 54 | 30,856 | 3 |
| Warp-distributed weights | 1 / 5 | 48 | 39 | 30,720 | 3 |
| Rolling rows | 2 / 6 | 48 | 42 | 24,576 | 4 |
| Rolling rows | 3 / 7 | 96 | 42 | 24,576 | 4 |

All use 256 threads and have zero stack, local-memory allocation, or native
local loads/stores. These are capacity limits on the tested SM86 device, not
measurements of achieved occupancy. Its 102,400-byte shared-memory capacity
fits four 24,576-byte blocks plus the per-block reserved allocation.

Each warp holds the 33 weights across its lanes and broadcasts each tap using
a full-mask shuffle. Lane zero sums normalization in the original order.
Invalid columns remain active for shuffles and barriers; only stores are
column-guarded. This removes the 136-byte shared weight/normalization storage
that would otherwise prevent the fourth rolling block from fitting.

The rolling variants use three 64-row shared rings and compute 24 output rows
per chunk. The first chunk loads 56 rows; each subsequent chunk loads 24 new
rows and retains the 32-row overlap. Publication and overwrite barriers are
unconditional across the block. No reciprocal approximation, reordered sum,
channel phasing, address-math optimization, or runtime compatibility layer is
part of this experiment.

## Native and correctness evidence

The prototype GPU object contains 81 bodies: all 78 original Butteraugli
bodies are instruction-identical to S143, plus three experimental bodies.
The release guard probe, host-ASAN guard probe, scoped sanitizer probe, and
both replay builds contain identical multisets of three linked GPU modules.
They use the hash-checked frozen S143 CUDA library for the AQ helper modules.

Static native body size increases: 1,736 instructions for baseline, 3,288
for warp-distributed weights, and 3,680 for either rolling variant. These
counts include different control-flow paths; they are not executed work
counts. Lower register allocation alone is not evidence of faster execution.

Release and host ASAN each passed 3,680 guarded fixtures, with three reuse
stages and two independent reference paths. Scoped checks cover 1x97, 33x97,
and 65x193 dimensions, packed/padded storage, five input patterns, and all
eight labels. Separate tall tests exercise flattened-grid limits.
Memcheck, initcheck, synccheck, and racecheck all passed the scoped fixture
suite; memcheck reported zero leaks, and racecheck zero errors or warnings.
Race checking completed in 783 seconds without a privilege/firewall blocker.

## Measurement protocol

The input and executable hashes and test order were pinned in `protocol.json`
before timing. Eighteen unchanged historical S73 captures cover flower, HD,
and 4K inputs with full/half-resolution and packed/padded output layouts.
They are historical stage inputs, not newly captured S144 encoder executions.

The preregistered preflight has 72 processes: 18 captures, release/ASAN, and
vertical-only/full horizontal-plus-vertical graph boundaries. The initial
timing screen has 36 processes: two opposite-order repetitions of all 18
captures, vertical-only, four launches per graph, eight warmup rounds and
16 measured rounds. Each process uses an eight-label randomized Williams
design with balanced positions and ordered predecessor pairs. All 16 device
arrays are checked exactly after every graph, outside CUDA-event timing.
The host validation/readback gaps make this a short-burst stage screen.

Read-only enforced-power-limit endpoints surround every checked graph.
They do not measure active-kernel clocks or prove a constant operating state.
Duplicate labels expose within-process variation. Any promising result needs
full-pair, sustained-burst, and current fully resident evidence before retention.

## Screening results

All 72 replay preflights and 36 timing processes passed. There are 6,912
timed windows (4,608 measured, 2,304 warmup), representing 18,432 measured
vertical launches. All 15,552 preflight/screen power-limit endpoints report
40 W. The primary comparison averages the two per-label medians for each
family; cross-label comparisons retain all four candidate/baseline pairs.
Negative deltas below mean faster; no observation is removed.

| Captured geometry group (six comparisons each) | Rolling 48 delta | Rolling 96 delta | Favorable primary comparisons, either rolling family |
| --- | ---: | ---: | ---: |
| Full 4K, 3839x2159 | -5.67% to -3.48% | -7.99% to -6.35% | 6/6 |
| Half 4K, 1920x1080, multiple output strides | -8.22% to +74.97% | -5.53% to +32.60% | 5/6 |
| Full HD, 1919x1079 | -7.66% to -7.20% | -5.55% to -4.90% | 6/6 |
| Half HD, 960x540 | +2.48% to +4.69% | +6.20% to +7.89% | 0/6 |
| Full flower, 510x532 | +7.78% to +8.44% | +3.50% to +3.99% | 0/6 |
| Half flower, 255x266 | +10.99% to +15.13% | +70.05% to +79.17% | 0/6 |

Each rolling family is favorable in 17/36 primary and 70/144 cross-label
comparisons. Each wins all 24 cross-label comparisons for full 4K and all 24
for full HD. Warp-distributed weights without the ring is unfavorable in all
36 primary comparisons (median +5.09%; only 3/144 cross-label pairs favorable).
This is not an across-the-board improvement or a justified production policy.

The first packed-output half-4K process (`short_r0_01`) is conspicuously
bimodal: identical baseline windows switch between roughly 0.51 and 3.61 ms
per launch. Its duplicate-label medians differ by -3.089536 ms. The same
capture's reverse-order repeat has a 0.498752 ms primary baseline and rolling
48/96 deltas of -7.38%/-4.82%. Another half-4K capture's baseline changes from
3.628352 to 0.510208 ms between repetitions even though its duplicate labels
are close within the first process. Balanced ordering and duplicate controls
therefore do not establish a stationary operating state. The unfavorable
bimodal comparisons remain in the totals above.

## Counter evidence

Eight Nsight Compute profiles use the same first full-resolution 4K capture,
four kernel families in forward/reverse order, with clock and cache controls
disabled. The baseline is its first matching direct launch; experimental
kernels are their first matching graph nodes. They are structural-work
observations, not an execution-context-matched replacement for ordinary timing.

| Family | Executed warp instructions versus baseline | Global-load sectors versus baseline | Observed active warps/SM, two profiles |
| --- | ---: | ---: | ---: |
| Baseline | reference | reference | 22.49 / 22.57 |
| Warp-distributed weights | +6.47% | +14.39% / +14.40% | 22.56 / 22.63 |
| Rolling 48 | +16.37% | +14.38% / +14.38% | 30.51 / 30.53 |
| Rolling 96 | +10.14% | -5.71% / -5.74% | 31.18 / 31.15 |

The dynamic increase is much smaller than the static body-size increase.
All three prototypes execute about 0.023% more predicated-on thread FFMAs;
they keep invalid final-column lanes active for shuffles. Shared-load
wavefronts decrease about 6.9-7.5%, but shared bank-conflict counts are not
consistently improved. Rolling 48 does not materially reduce DRAM reads.
Rolling 96 reduces DRAM reads by 12.24%/12.25% and L2 traffic by 7.06%/7.09%,
while DRAM writes stay within about 0.13% of baseline. Full raw counters and
both repetitions, including increased load-sector and instruction counts,
are preserved.

The observed occupancy increase supports the ring's resource hypothesis;
the 96-row ring additionally amortizes its halo over more output rows. This
does not isolate occupancy as the sole cause of the timing gains: the designs
also change loads, synchronization, instruction count, and scheduling. The
warp-weight-only control demonstrates that lower registers/shared allocation
without an extra resident block is insufficient here.

## Timing follow-up and decision

Two Nsight Systems traces complete successfully, each with 809 kernels,
200 graph launches, 384 event records, and exact output checks. All 800
additional trace power-limit endpoints report 40 W. The half-resolution trace
reproduces baseline kernel-interval averages around 0.495-0.515 ms and
3.615-3.623 ms per launch. The full-resolution trace also contains roughly
2 ms and 15 ms baseline kernel-interval averages. The largest event duration
minus summed kernel duration is only 0.045691/0.046756 ms per launch in the
half/full traces, including warmups. Inter-node gaps are negligible in these
recorded timelines. Thus the reproduced multi-millisecond swings are inside
reported kernel intervals, not explained solely by gaps around graph launches.

This does not identify the physical cause: CUPTI kernel start/end intervals
are not active-SM cycle measurements and can include operating-state or
preemption effects. Nor does profiling establish the exact cause of the
earlier unprofiled observations. One read-only, unsynchronized `nvidia-smi`
observation during `short_r0_02` reports 1,282 MHz, 66 C, P3, and 25.09 W;
it was taken after the bimodal `short_r0_01` process and is not a clock
measurement for its slow kernels. The telemetry limitations are preserved.

Keep S143 as the retained implementation. S144 identifies promising large-
geometry candidates, not a production policy: short bursts and historical
stage captures do not qualify sustained full-pair or fully resident encoding.
Next test 48/96-row selection with sustained horizontal-plus-vertical work,
fresh current resident captures, and a geometry policy that retains baseline
for small inputs. Check GPU-active operating state before interpreting the
bimodal half-resolution runs. Do not repeat the already unsuccessful
warp-weight-only variant or choose tile height from shared-memory capacity
alone. No S144 production CUDA/test change or compatibility layer is retained.

Evidence root: `build-cuda-ninja/profiles/s144-artifacts/`. Experiment drivers
and generated diagnostic sources are `build-cuda-ninja/profiles/s144_*`.
The initial link attempt omitted required AQ helpers; its failed log and map
are preserved. The versioned recovery links the frozen qualified S143 library.
No historical artifact, power/clock policy, firewall setting, affinity, or
process priority was changed.

The numerical reports, job logs, native/module audits, counter exports,
timeline databases, and source snapshots are verified by `verify_s144.py`
and archived by `freeze_s144.py`. Failed initial link artifacts remain intact.
The forty retained historical runtime files retain their recorded hashes.
