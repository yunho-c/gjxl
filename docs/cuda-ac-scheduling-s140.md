# CUDA AC candidate packing and channel ordering (S140)

Disposition: both schedules are unpromoted. Each is unfavorable in all four
4K whole-encode comparisons. Production remains the
[S138](cuda-fused-ac-integration-s138.md) evaluator; the unpromoted
[S139](cuda-fused-final-cost-s139.md) final-cost fusion is not enabled.

## Isolated hypotheses

Two diagnostic schedules retain S138's arithmetic, per-transform lane count,
shared coefficient/residual tiles, immutable Y copy, quant-norm preparation,
final-cost kernel and global scratch layout:

- Channel-major ordering retains 192 threads and the same candidates per block,
  but assigns channel first, then candidate, to transform groups. Smaller-shape
  warps can read the same channel's matrix entries and take uniform
  channel-dependent branches. Matrix transaction/cache benefits are hypotheses,
  not measured bandwidth claims.
- Half-block packing retains candidate-major channel ordering but uses 96
  threads, half as many complete candidates per block and half the shared
  storage. More blocks are needed for the same candidates. This may improve
  resource-limited concurrency, or lose throughput to extra blocks and changed
  register allocation.

The generated kernel changes only template scheduling parameters, the mapping
from thread group to candidate/channel, and the shared-Y index. Output indices
remain global candidate/channel indices. Complete candidates stay together;
inactive tails hit every barrier without descriptor loads. The original
residual and loss helpers are reused. No arithmetic approximation or fast-math
relaxation is introduced. Only ignored diagnostic sources contain selectors;
the normal source and allocation contracts are unchanged.

## Native resources and modeled occupancy

Starting revision `317e038`, branch `feat/cuda`, September 8, 2026. Windows,
RTX 3060 Laptop (`sm_86`), CUDA 11.8, MSVC 14.37, Release. Fresh diagnostic GPU
objects and callers link against frozen S138 libraries. Scoped host ASAN uses
fresh clang-cl callers plus hash-checked S138 validator/search, pipeline and
resident objects whose sources have not changed.

There are 235 kernel bodies: 220 S138 bodies are instruction/control-identical,
one source-unchanged quant-norm body matches the historical S134 variant, and
fourteen new scheduled bodies have zero stack/local storage and no static
local-memory loads/stores. Twelve statically linked GPU executables, including
trace and ASAN callers, have identical ten-module sets. Eight modules match the
frozen S138 production encoder.

The resource query loads that exact extracted GPU module through the CUDA
driver, checks function attributes and calls its occupancy API at each actual
block size, with zero dynamic shared memory and no cache/attribute changes.
It launches no test kernels. The device reports 1,536 maximum resident threads,
102,400 shared bytes, 65,536 registers and sixteen blocks per SM. The following
values are default-driver resource ceilings, not measured achieved occupancy.

| Shape | Registers: baseline / channel-major / half-block | Modeled warps per SM: baseline / channel-major / half-block |
| --- | ---: | ---: |
| 8×8 | 40 / 40 / 40 | 48 / 48 / 48 |
| 16×8 | 39 / 40 / 40 | 48 / 48 / 48 |
| 8×16 | 56 / 55 / 60 | 36 / 36 / 30 |
| 16×16 | 48 / 48 / 50 | 30 / 30 / 30 |
| 32×16 | 56 / 64 / 64 | 30 / 30 / 30 |
| 16×32 | 88 / 92 / 92 | 18 / 18 / 18 |
| 32×32 | 78 / 78 / 78 | 12 / 12 / 15 |

Thus halving the block is not a general occupancy improvement. It improves
the queried 32×32 ceiling, lowers 8×16, and leaves the others unchanged.
Channel ordering changes no queried occupancy ceiling. Register counts alone
do not establish the reason for any complete-encode timing change.

A separate read-only driver query reports 1,024 reserved shared bytes per
block. The baseline 32×32 kernel has 33,536 static shared bytes, so three blocks
would require `3 * (33,536 + 1,024) = 103,680` bytes, exceeding the device's
102,400-byte SM capacity. This explains why summing only visible static shared
storage incorrectly suggests that three blocks fit. Half-block packing uses
16,768 static bytes; five blocks fit while six require 106,752 bytes. The
reservation-aware bounds agree with the occupancy API's two versus five
blocks, or twelve versus fifteen warps. This is a capacity explanation, not
proof of achieved occupancy or end-to-end performance causality.

## Correctness and benchmark design

The focused fixture covers fifteen geometries, including thin images, partial
tiles and axes through 257 blocks. Both host/signed device CfL and
descriptor/device-derived quant norms give 288 cases and 59,752 descriptors
per execution. It obtains baseline final costs, channel rates and losses from
the normal S138 batch, resets a guarded strided-input arena, and requires three
evaluations of each schedule to match all output bits exactly while preserving
every byte outside the three live outputs. Forward scratch remains zero.

The focused fixture passes Release, scoped ASAN and all four CUDA sanitizers.
Memory/init/synchronization checking reports zero errors, race checking reports
zero hazards, and memcheck reports zero leaked bytes. The exhaustive extension
covers every block width/height pair from 1 through 19, in all four CfL/norm
combinations: 8,804 cases and 539,236 descriptors per execution in Release and
ASAN. Together those grids execute 19,336 baseline batches and 58,008 batches
per candidate schedule. Repeated evaluations are not independent images.
The broad AC contract test passes all three paths in Release and ASAN, with
329 submissions per family per execution, including invalid input/range cases.

Complete-encode inputs match S133/S138: Flower 500, padded HD from 1919×1079,
padded 4K from 3839×2159, and Flower 2000 (fourfold nearest-neighbor replication,
not a native 2000-square photo). Distance is 1.2, effort 7, fully resident, with
wide and opt-in compact coefficients; the default coefficient width is unchanged.

Six labels duplicate all three families: baseline 0/3, channel-major 1/4 and
half-block 2/5. Six-round Williams blocks balance positions and ordered
predecessors. Each timed process has one reference encode, six warm rounds and
eighteen measured rounds. Two process passes reverse case/width order. Every
encode checks frozen bytes, summary, coefficient width/storage and seven
expected AC-batch selections. Power endpoints are recorded for every encode.
The preregistered analysis reports all observations and duplicate controls,
and separately identifies processes with identical power endpoints; varying
power is not relabeled as controlled latency evidence.

This experiment is not a fresh normal CMake build or CTest run. The four
complete-encode inputs at one quality/effort setting do not replace broad
quality/search/batch promotion qualification. No new Metal, Linux, independent
decoder or second-GPU qualification is claimed.

## Complete-encode results

Sixteen timed processes contain 1,728 measured encodes, plus reference and
warmup calls. No other recorded build or diagnostic job overlaps a timed
process. Preflights/timing check 2,432 frozen encodes; the traces add 56, for
2,488 total, including 56 scoped host-ASAN encodes. Every check passes.
All 4,976 power endpoints, including traces, report 40,000 mW. Constant limits
do not prove constant clocks, cache/placement or absent external activity.
No power, clock, priority, affinity or firewall policy was changed, and no
elevation/firewall blocker was encountered.

The tables give median within-round mean candidate-label time minus mean
baseline-label time. Negative is faster. They are not differences of separate
medians; quantization is nested inside whole-call time, so savings are not
additive.

### Channel-major ordering

| Case / width | Whole-call delta r0 / r1 (ms) | Quantization delta r0 / r1 (ms) |
| --- | ---: | ---: |
| Flower 500 / wide | −0.310 / −0.108 | +0.000 / −0.098 |
| Flower 500 / compact | −0.468 / +0.164 | −0.052 / +0.146 |
| HD / wide | +1.780 / −1.862 | +1.096 / −0.426 |
| HD / compact | −1.029 / +0.211 | −0.130 / −0.410 |
| 4K / wide | +4.231 / +7.333 | +2.901 / +3.060 |
| 4K / compact | +7.370 / +1.017 | +3.277 / +6.898 |
| Flower 2000 / wide | +0.215 / +2.582 | −0.712 / +2.286 |
| Flower 2000 / compact | −0.428 / −0.018 | +2.253 / +0.401 |

### 96-thread candidate packing

| Case / width | Whole-call delta r0 / r1 (ms) | Quantization delta r0 / r1 (ms) |
| --- | ---: | ---: |
| Flower 500 / wide | −0.361 / +0.348 | −0.127 / +0.075 |
| Flower 500 / compact | −0.250 / −0.048 | −0.073 / +0.154 |
| HD / wide | +0.942 / +0.824 | +0.358 / −1.051 |
| HD / compact | −2.589 / −1.317 | −1.079 / −1.488 |
| 4K / wide | +8.927 / +1.843 | +4.946 / +5.858 |
| 4K / compact | +16.974 / +7.382 | +6.982 / +3.711 |
| Flower 2000 / wide | +1.852 / −2.144 | +0.917 / +1.789 |
| Flower 2000 / compact | +1.651 / −1.422 | +0.562 / +2.386 |

Channel-major ordering is favorable in 7/16 whole-call primary comparisons
and 22/64 cross-label pairs; quantization is 6/16 and 22/64. Half-block packing
is 7/16 and 27/64 for whole calls, and 5/16 and 32/64 for quantization. Both
are unfavorable in all four 4K primary results for both whole calls and
quantization. Duplicate controls remain substantial: equivalent half-block
labels differ by +25.889 ms in the second wide-4K process. These results do not
justify a precise universal regression estimate or a post-hoc small-image
selector, but they do not justify replacing S138 either.

## Traces and unresolved local/whole-call difference

Eight complete-process Nsight Systems captures contain 48 labeled windows and
eight reference encodes. All CUDA calls succeed. Launch counts, all unrelated
kernel-name/count multisets, and memcpy counts/payload bytes are unchanged.
Each path still launches seven quant-norm, seven evaluator and seven final-cost
kernels. Trace launch dimensions verify 192-thread unchanged grids for
channel-major ordering and 96-thread near-doubled grids for half-block packing,
including odd-tail rounding. Both schedules retain zero global forward scratch.

The following summed evaluator durations average the two labels per family.
They are short instrumented captures, not substitutes for balanced whole-call
timings.

| Case / width | Baseline / channel-major / half-block AC time (ms) |
| --- | ---: |
| HD / wide | 3.330 / 3.447 / 3.235 |
| HD / compact | 3.543 / 3.555 / 3.347 |
| 4K / wide | 20.029 / 16.731 / 17.078 |
| 4K / compact | 19.906 / 23.104 / 17.107 |
| Flower 2000 / wide | 6.784 / 6.248 / 6.050 |
| Flower 2000 / compact | 6.672 / 6.724 / 6.422 |
| Flower 500 / wide | 0.485 / 0.484 / 0.455 |
| Flower 500 / compact | 0.484 / 0.485 / 0.453 |

Half-block packing is locally favorable in all eight captures; channel-major
ordering is favorable in three. That does not overturn the unfavorable
complete-encode results. Unchanged GPU work also varies: in wide 4K, half-block
AC time falls by about 2.95 ms while other kernel time rises by about 7.41 ms.
In compact 4K, channel-major AC time rises by about 3.20 ms while other kernel
time falls by about 26.48 ms. The captured whole-call means themselves differ
from the larger timed campaign. The record does not establish a full causal
explanation for this variation, and the local gains are not relabeled as
encoder speedups.

## Next action and evidence

Keep S138's candidate-major 192-thread schedule. The shared-memory capacity
bound suggests a more direct experiment: remove the duplicate immutable Y
coefficient copy while retaining complete candidates and the current block
size. X/B residual quantization could finish reading Y before Y's tile is
overwritten, or all residual stores could be delayed behind a read-completion
barrier. Those approaches trade synchronization and register liveness against
shared storage; neither is implemented or qualified here. Test exact bits,
barriers, native resources, and complete-encode performance independently.
The encoder is not established to be maxed out.

All sixty recorded jobs are terminal and accepted. Four CUDA sanitizer jobs
qualify. Forty retained runtime files and all reused frozen S138 inputs keep
their hashes. No previous artifacts were removed or overwritten.

Evidence root: `build-cuda-ninja/profiles/s140-artifacts/`; drivers and generated
prototypes are `build-cuda-ninja/profiles/s140_*`. Key records include
`before.json`, `inputs.json`, `trace_inputs.json`, `native_scan.json`,
`linked.json`, `occupancy_analysis.json`, `reserved_shared.json`,
`within_analysis.json`, `trace_analysis.json`, `trace_parts.json`,
`timing_summary.json`, job logs, `final_summary.json`,
`source_snapshot_index.json` and `artifact_hashes.json`.
