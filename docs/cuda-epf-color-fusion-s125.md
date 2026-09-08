# Final EPF and color conversion fusion (S125)

Starting revision: `adfd38b`, branch `feat/cuda`. Windows, CUDA 11.8,
MSVC 14.37, Release, RTX 3060 Laptop / sm86; 2026-09-08.

**Outcome: qualified isolated candidate, not yet integrated.** All 48
case/burst-length matched medians favor fusion. At 4K, pass 2 is about 49%
faster in four-chain bursts and 37–40% faster in 128-chain bursts. Independent
counter captures show roughly half the boundary's DRAM traffic with unchanged
FFMA work. Production remains S124; real resident layouts, inputs and whole
encodes are the next qualification gate, not an established speedup here.

## Question and scope

The fully resident reconstruction path runs its final EPF pass into three
filtered XYB planes and immediately reads them in Opsin-to-linear conversion.
S105's earlier counters measured substantial DRAM utilization in both pass 2
and color conversion. This motivates removing their intermediate write/read,
not another arithmetic-only EPF change. The earlier compact ownership,
composition/AQ fusion and CPU tile scheduling studies retain their respective
dispositions; this is a different producer/consumer boundary.

The diagnostic fuses color conversion into the existing 32x32 EPF tile with
256 threads. It tests final pass 1 (one EPF iteration) and final pass 2 (two
or three iterations). Pass 0 is never the final pass under the existing
iteration policy. Zero-iteration reconstruction still needs its ordinary
color conversion and is outside this experiment.

The fused path writes RGB directly and leaves all three intermediate planes
untouched. It removes one launch and six logical float transfers per pixel
from this boundary: three intermediate stores and three subsequent loads.
At 3839x2159 that is 198,921,624 logical bytes per chain, before cache effects.
This is a source-level traffic count, not a new DRAM-counter measurement.
The benchmark still allocates intermediate storage for its control. No
allocation or resident-memory-capacity reduction is claimed.

The resident owner uses `FinalFilteredImage()` for maximum-error evaluation.
That consumer still requires filtered XYB. A future integration must preserve
it, while the perceptual path can potentially avoid the final XYB materialization.
The source-sized RGB outputs and coding-stride EPF inputs also need their
actual layout contracts respected. No production source is changed here.

## Arithmetic and native-code controls

The prototype mechanically copies the current EPF body and color arithmetic.
Mirrored halo coordinates, all-lane loading/barrier placement, candidate order,
FMA usage, divisions, EPF nonfinite sanitization and error-bit ORs remain in
the same order. A bypass pixel is converted directly without EPF sanitization,
matching the original separate kernels; bypass continues to later per-thread
rows rather than terminating the thread.

All ten original bodies in the included production translation unit match
their S124 production-linked bodies exactly. Both copied separate-pass
controls are native-identical to their corresponding EPF kernels. The
diagnostic object and release/host-ASAN correctness and timing executables
all contain the same fourteen bodies. No other GPU dependency is linked into
these standalone probes.

| Final pass | Separate EPF registers | Fused registers | Shared bytes, either route | Stack / spills |
| --- | ---: | ---: | ---: | ---: |
| 1 | 40 | 55 | 15,552 | 0 / 0 |
| 2 | 38 | 40 | 13,872 | 0 / 0 |

The register increase, particularly for pass 1, is a real cost to measure,
not evidence of an automatically faster implementation. The separate color
kernel uses 20 registers and no shared memory.

## Qualification

Release and scoped host-ASAN each execute 27,648 guarded pipelines over
24 geometries, two padding choices, both final passes, sixteen input/sigma
patterns, two color scales and three changed-input reuse stages. For each
configuration/stage, the original separate pipeline supplies a reference;
the native-identical copied control and fused candidate must match every
output bit and the complete error word, including a preexisting error bit.
All input/sigma contents and allocation/row guards are checked. The control's
intermediate planes match the original; the fused route must leave theirs
entirely poisoned and untouched.

The patterns include ordinary random/ramp/checker values, signed zeros,
subnormals, extreme finite exponents, NaNs, infinities, bypassed nonfinite
pixels, mixed bypass rows, values on both sides of the exact bypass threshold,
and unusual sigma values. Geometries include single pixels/rows/columns,
31/32/33 boundaries, odd tiles and 4096x3. Host ASAN instruments the standalone
harness, not CUDA device code or the driver.

Memcheck, racecheck, initcheck and synccheck each execute another 3,456
guarded pipelines over 1x1, 33x33 and 65x17 with both paddings and the same
patterns/passes/scales/reuse. All report zero errors or hazards; memcheck
also reports zero leaks. In total these fixture jobs execute **69,120
guarded pipelines**, including **46,080 bitwise control/candidate comparisons**
against separate-pipeline references. This is not a new CPU differential
oracle or complete-encoder qualification.

Two initial harness runs fail with inconsistent error flags. The harness
used pageable default-stream H2D copies immediately before work on a
nonblocking stream, without a completion dependency. Adding default-stream
synchronization after setup/reset resolves the failures without changing
the GPU object. The original and diagnostic failure logs, binaries and
pre-correction fixture snapshots are retained. The first sanitizer launcher
also used a nonexistent executable path; no child process started. Its empty
log and correction record are retained, and the corrected path completes
all four tools. Neither incident was an admin/firewall/permission block.

## Timing protocol

Each process uses one fixture with a deterministic ramp-like pattern, not a
captured encoder EPF input. Input/sigma values remain unchanged within the
process. Each burst repeats the same idempotent EPF/color chain four or 128
times and checks its final RGB result against a separate-pipeline reference.
Every fused burst also verifies untouched intermediate storage. The reported
per-chain time is the complete CUDA-event burst interval divided by its
chain count; individual inner results are not separately read back.

Labels 0/1 run original/native-identical separate pipelines. Labels 2/3 run
the same fused pipeline. Four warm rounds and eight measured rounds use
shuffled cyclic orders with every label occupying each position equally.
Three shapes (500x500, 1919x1079 and 3839x2159), two input padding choices and
both passes run twice, with the second case order reversed: 24 measured
processes. Eight additional release/host-ASAN preflight processes precede
the timing population.

The two layouts use input stride `width + pad`, intermediate stride
`width + pad + 3`, RGB stride `width + pad + 7`, and sigma stride
`ceil(width/8) + pad`, for pad 0 or 7. Thus even the packed-input case has
padded outputs. These are useful layout probes but are not the resident
encoder's coding-stride input and packed-RGB combination.

Setup uploads, scratch/output poisoning, result readback, bitwise checks and
read-only NVML enforced-limit queries lie outside each event interval. These
operations occur between bursts: this is sustained work *within a burst*,
not uninterrupted whole-process EPF saturation. Each burst records a power
limit before and after; equal endpoints do not prove stable clocks inside.
CUDA-event intervals can include host submission gaps. No clock, power,
priority, affinity or security settings are changed, and no artificial sleep
is inserted. Timed jobs do not overlap other recorded jobs, builds, sanitizer
runs or heavy artifact hashing. Light editing and ordinary shared-machine
activity remain measurement limitations.

## Timing results

The following ranges span the four matched medians for each shape/pass/burst
length (two input paddings, two process orders). They are not confidence
intervals. Negative deltas mean faster. Each round first averages its two
separate labels and its two fused labels, then forms the difference/ratio;
the reported median is over eight measured rounds. Percentages are not ratios
of separately aggregated time medians.

| Shape | Final pass | Chains/burst | Per-chain delta (ms) | Paired percentage change |
| --- | ---: | ---: | ---: | ---: |
| 500x500 | 1 | 4 | -0.0280 to -0.0219 | -40.6% to -31.5% |
| 500x500 | 1 | 128 | -0.0205 to -0.0156 | -23.3% to -15.6% |
| 500x500 | 2 | 4 | -0.0263 to -0.0253 | -42.9% to -39.6% |
| 500x500 | 2 | 128 | -0.0270 to -0.0252 | -38.3% to -32.1% |
| HD | 1 | 4 | -0.1755 to -0.1718 | -40.9% to -39.9% |
| HD | 1 | 128 | -0.4829 to -0.1905 | -41.8% to -18.6% |
| HD | 2 | 4 | -0.2040 to -0.2021 | -49.0% to -48.6% |
| HD | 2 | 128 | -0.4515 to -0.3359 | -51.6% to -43.4% |
| 4K | 1 | 4 | -0.6803 to -0.6672 | -40.8% to -40.1% |
| 4K | 1 | 128 | -1.7448 to -1.6085 | -33.1% to -32.1% |
| 4K | 2 | 4 | -0.8000 to -0.7976 | -49.5% to -49.4% |
| 4K | 2 | 128 | -1.7654 to -1.6585 | -39.9% to -37.4% |

All 48 aggregated matched medians are favorable, not necessarily every
individual round. The 4K separate pass-2 chain medians rise from 1.614–1.616 ms
in short bursts to 4.378–4.508 ms in long bursts; fusion does not make the
operating-state effect disappear. Long-burst duplicate differences are also
substantial: the largest absolute per-chain duplicate median is about
0.281 ms at 4K and 0.277 ms at HD. These observations remain in the result,
alongside the consistent candidate advantage.

Across measured and preflight jobs, 2,560 bursts execute 153,600 logical
chains, plus 32 process references. Every burst's terminal output is checked;
this is not 153,600 independent readback checks. Host-ASAN preflights cover
128 of those bursts. All 5,120 unprofiled power-limit endpoints report 40 W,
with no endpoint change. No per-kernel clock correction is inferred from that.

## Memory and instruction counters

Eight separate Nsight Compute 2025.2 captures cover both passes and separate/
fused routes in opposite process orders. They use the same synthetic 4K
packed-input/padded-output fixture and native GPU object as the timing probe.
A reference and eight warm chains precede the profiler range, which contains
one logical chain: two kernels for the separate route, one for fused.
Final RGB, error flags, inputs and unused fused intermediates are checked.
Kernel replay adds profiler executions beyond those logical harness launches.

Cache control and clock control are both `none`; profiling starts only at the
explicit range. These are instrumented workload observations, not another
unprofiled throughput cohort or a claim of identical cache/clock exposure.
The table sums additive counters across the two baseline kernels and compares
that boundary with the single fused kernel. MB denotes decimal megabytes.

| Pass | Separate DRAM read + write (MB) | Fused DRAM read + write (MB) | Separate L2 traffic (MB) | Fused L2 traffic (MB) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 397.321–397.470 | 199.355–199.383 | 496.375–497.305 | 277.962–278.146 |
| 2 | 397.244–397.274 | 199.322–199.396 | 480.365–480.817 | 263.728–263.782 |

Both repetitions measure approximately 49.8% less DRAM traffic. Executed
warp instructions are exactly 139,847,930 -> 121,147,455 for pass 1 and
101,958,651 -> 82,868,022 for pass 2, respectively about 13.4% and 18.7% less.
All eight captures execute 430,996,852 predicated-on thread FFMA instructions.
The fused design removes materialization/address work without reducing those
floating-point operations. Shared-wavefront and global-sector counters are
retained in the raw report; no reduction of EPF neighborhood math is claimed.

## Disposition, remaining gates and evidence

Proceed to an isolated resident-encoder integration. The measured traffic
reduction and consistent local timing improvement justify that next step;
they do not qualify an unconditional production change. Required checks are
actual coding-stride inputs/packed RGB, captured hot EPF inputs, zero/one/two/
three EPF iterations, Gaborish combinations, custom color scales and sigma
parameters, maximum-error XYB consumption, changed-input/prepared reuse,
failure behavior, and complete single/batch encode byte/summary equality.
Whole-encode timing must retain duplicate controls, sustained-load exposure
and adverse observations. No compatibility layer or speculative runtime
policy is added in this study. The backend is not established to be maxed out.

Evidence is in `U:/gjxl-cuda-diagnostics/s125/`; sources and scripts are
`build-cuda-ninja/profiles/s125_*`. Key records are `generation.json`,
`qualification_inputs.json`, `bench_inputs.json`, `native.json`,
`analysis.json`, `counter_sources.json`, `counter_inputs.json` and
`counter_analysis.json`, plus all job logs/records, native dumps and profiler
reports. The measured campaign runs from 14:14:15 to 14:20:09 UTC; counter
captures run separately from 14:21:19 to 14:21:38 UTC.

The initial timing parser incorrectly expected nine fields in an eight-field
row and stopped before emitting an analysis. Its pinned source remains;
`s125_analyze_v2.py` corrects only that field count. The validator independently
recomputes rows, position balance, paired statistics and counter totals from
raw logs, so a success marker alone is not the acceptance criterion.

`s125_validate.py` checks all 70 terminal job records (68 successful and two
rejected pre-fix harness runs), the separately recorded no-process launch
failure, input/executable/log hashes, native identities, guard/sanitizer
counts, timing isolation and the unchanged 40 retained runtime files. S124's
current production source snapshots also remain unchanged. This study does
not claim a new CTest run or complete-encode campaign. `s125_freeze.py`
validates and archives sources with transitive Python dependencies before
hashing the evidence; `s125_validate.py --frozen` verifies it afterward.
