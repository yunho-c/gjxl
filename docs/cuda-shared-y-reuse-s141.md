# CUDA shared-Y coefficient reuse (S141)

Disposition: neither all-shape variant is promoted. Each is unfavorable in
three of four 4K whole-call primary comparisons. Normal production remains
[S138](cuda-fused-ac-integration-s138.md). The two variants test the shared-memory
capacity bound established by [S140](cuda-ac-scheduling-s140.md); neither changes
candidate packing, launch count, global scratch, quant-norm preparation or
final-cost composition.

## Isolated changes

Both variants retain candidate-major 192-thread blocks and complete three-channel
candidates. They remove the separate immutable shared Y coefficient copy, read
the original padded Y channel tile instead, and add one block barrier:

- Phased variant: X/B perform residual quantization first; all threads reach a
  barrier before Y replaces its own coefficients. The original residual helper's
  warp reductions stay within whole 8/16/32-thread channel groups. No conditional
  phase calls a helper with an internal block barrier.
- Deferred-store variant: all channels compute residuals into compile-time-indexed
  per-lane arrays, finish their rate reductions, then reach an unconditional
  read-completion barrier before storing residuals into the channel tiles. The
  existing caller barrier still protects the subsequent inverse transform.

Both preserve the forward and inverse pass order, coefficient normalization,
CfL arithmetic (including Y's zero factor), rounding and FP32 halving trees.
Padded natural-coordinate Y addressing replaces the packed-copy addressing.
Invalid candidates retain their existing NaN behavior; inactive complete-candidate
tails avoid descriptor loads but reach all barriers. This is shared coefficient
consumption inside AC search, not a new global compact-coefficient format.

All changes are generated ignored diagnostic sources. No production or test
source, allocation contract, default coefficient width or compatibility layer
is changed. The normal S138 evaluator is the baseline inside the same executable.

## Build and native resources

Starting revision `479c154`, branch `feat/cuda`, September 8, 2026 local time
(September 9 UTC). Windows, RTX 3060 Laptop `sm_86`, CUDA 11.8, MSVC 14.37 Release.
Fresh GPU objects and callers link against frozen S138 libraries. Scoped host
ASAN uses fresh clang-cl callers and source-identical, hash-checked frozen S138
host objects; CUDA sanitizer checks cover device execution separately.

The diagnostic GPU build emits 21 unused-variable warnings: seven each for
`residual_values`, `v` and `u` in discarded compile-time branches. They are
retained in the build log, not hidden or described as a warning-free build.
Other three build jobs have no warnings/errors. Every new kernel has zero
stack/local storage and no static local-memory loads/stores despite the
deferred arrays. Across 235 bodies, 220 S138 bodies remain instruction/control-
identical; one source-unchanged quant-norm body matches the historical S134
variant; fourteen bodies are new. Twelve statically linked GPU executables
have identical ten-module sets, eight modules matching S138 production.

Driver queries load the exact extracted GPU module and use actual block sizes,
zero dynamic shared memory and unchanged default function/cache attributes.
They launch no test kernels. These are resource ceilings, not achieved occupancy.

| Shape | Static shared bytes: baseline / either variant | Registers: baseline / phased / deferred | Modeled warps per SM: baseline / phased / deferred |
| --- | ---: | ---: | ---: |
| 8×8 | 8,960 / 6,912 | 40 / 40 / 44 | 48 / 48 / 36 |
| 16×8 | 8,576 / 6,528 | 39 / 39 / 46 | 48 / 48 / 36 |
| 8×16 | 8,960 / 6,912 | 56 / 54 / 62 | 36 / 36 / 30 |
| 16×16 | 17,152 / 13,056 | 48 / 52 / 52 | 30 / 36 / 36 |
| 32×16 | 16,768 / 12,672 | 56 / 71 / 78 | 30 / 24 / 24 |
| 16×32 | 17,152 / 13,056 | 88 / 94 / 104 | 18 / 18 / 12 |
| 32×32 | 33,536 / 25,344 | 78 / 79 / 76 | 12 / 18 / 18 |

The device again reports 102,400 shared bytes per SM and 1,024 reserved shared
bytes per block. For 32×32, three baseline blocks require 103,680 bytes and do
not fit; three candidate blocks need only 79,104 bytes, while four would need
105,472. The occupancy API confirms two versus three blocks, or twelve versus
eighteen warps. The 16×16 ceiling also improves. Other shapes show the register
tradeoff: fewer shared bytes do not guarantee more resident work or lower time.

## Correctness and timing design

The guarded focused fixture covers fifteen block geometries, thin images and
partial tiles through 257 blocks, host/signed-device CfL and descriptor/device
quant norms. Each execution covers 288 cases and 59,752 descriptors, obtaining
normal S138 costs/rates/losses and requiring three arena-reset evaluations of
each candidate to match all output bits while preserving every other byte.
Forward scratch remains zero. The exhaustive extension covers every block
width/height pair from 1 through 19 in all four CfL/norm modes: 8,804 cases and
539,236 descriptors per execution. The broad contract fixture checks all three
paths, including invalid inputs/ranges, with 329 submissions per family.

The focused fixture passes Release, scoped ASAN and all four CUDA sanitizers;
the exhaustive and contract fixtures pass Release and ASAN. Together the six
focused and two exhaustive runs execute 19,336 baseline batches and 58,008
batches per candidate. Repeated evaluations are not independent images.
Memory/init/synchronization checking reports zero errors; race checking reports
zero hazards; memcheck reports zero leaked bytes. Race checking completed on
the original live process after roughly 220 seconds; it was not restarted.

Complete-encode inputs are the frozen S133/S138 Flower 500, padded HD from
1919×1079, padded 4K from 3839×2159 and Flower 2000 (fourfold nearest-neighbor
replication, not a native large photograph). Distance 1.2, effort 7, fully
resident, wide and opt-in compact. Six labels duplicate each family: baseline
0/3, phased 1/4, deferred 2/5. Six-round Williams blocks balance positions and
ordered predecessors. Each timed process uses one reference, six warm rounds
and eighteen measured rounds; the second pass reverses image/width order.
Every encode checks frozen bytes, summary, coefficient width/storage and all
seven expected AC-batch selections. Every encode has two power-limit endpoints.

This is not a fresh normal CMake/CTest campaign, broad quality/effort promotion
qualification, independent decoder test, or new Metal/Linux/second-GPU evidence.
No power, clock, priority, affinity or firewall policy is changed.

## Results

Sixteen timed processes contain 1,728 measured encodes, with reference and
warmup calls in addition. Preflights/timing verify 2,432 frozen encodes and the
traces verify 56 more: 2,488 total, including 56 scoped host-ASAN encodes. All
checks pass. No other recorded build or diagnostic job overlaps a timed process.

All 4,976 power-limit endpoints are retained: 4,976 at 40,000 mW.
16/16 timed processes have identical endpoints throughout. Constant enforced
limits do not establish constant clocks, power draw, cache/placement or absence
of external activity. No elevation/firewall blocker was encountered.

The primary statistic is the median within-round mean candidate-label time
minus mean baseline-label time, not the difference of separate medians.
Negative is faster. Quantization is nested within the whole call, so these
deltas are not additive.

### Phased X/B then Y

| Case / width | Whole-call delta r0 / r1 (ms) | Quantization delta r0 / r1 (ms) |
| --- | ---: | ---: |
| Flower 500 / wide | +0.035 / +0.326 | +0.197 / +0.128 |
| Flower 500 / compact | +0.117 / +0.326 | +0.155 / +0.232 |
| HD / wide | +2.884 / +0.080 | +0.899 / +0.814 |
| HD / compact | +0.800 / −2.032 | −0.057 / +0.320 |
| 4K / wide | +6.720 / +16.770 | +11.439 / +7.640 |
| 4K / compact | −0.719 / +5.795 | +3.034 / +5.569 |
| Flower 2000 / wide | +1.570 / −4.262 | +2.073 / −2.716 |
| Flower 2000 / compact | +3.659 / −0.088 | +3.178 / +1.850 |

Favorable primary comparisons: 4/16 whole-call and
2/16 quantization; cross-label pairs: 24/64 and
7/64 respectively. At 4K, 1/4 whole-call primary results are favorable.

### Deferred residual stores

| Case / width | Whole-call delta r0 / r1 (ms) | Quantization delta r0 / r1 (ms) |
| --- | ---: | ---: |
| Flower 500 / wide | +0.023 / +0.214 | +0.109 / +0.111 |
| Flower 500 / compact | −0.351 / −0.074 | −0.113 / +0.024 |
| HD / wide | +1.864 / −3.812 | +0.178 / −0.217 |
| HD / compact | +0.446 / −2.748 | −0.397 / −0.493 |
| 4K / wide | +8.358 / +10.263 | +6.235 / +6.441 |
| 4K / compact | −3.159 / +3.498 | −1.762 / +3.989 |
| Flower 2000 / wide | −1.977 / −9.162 | +1.068 / −5.144 |
| Flower 2000 / compact | +1.753 / +0.036 | +0.311 / −3.491 |

Favorable primary comparisons: 7/16 whole-call and
7/16 quantization; cross-label pairs: 39/64 and
38/64 respectively. At 4K, 1/4 whole-call primary results are favorable.

The largest absolute whole-call duplicate-label control is −16.488 ms
(4K / wide, r1, labels 4 minus 1). Equivalent-label
variation must not be attributed to the algorithm or omitted from the result.

### Trace results

Eight complete-process Nsight Systems CUDA/NVTX captures contain 48 labeled
windows plus eight reference encodes. All CUDA calls succeed. Each variant
retains seven quant-norm, seven evaluator and seven final-cost launches;
192-thread blocks, grids, unrelated kernel-name/count multisets and memcpy
counts/payloads are unchanged. These short instrumented captures are not
substitutes for the balanced whole-call campaign.

| Case / width | Baseline / phased / deferred summed evaluator time (ms) |
| --- | ---: |
| HD / wide | 3.546 / 4.414 / 3.564 |
| HD / compact | 3.547 / 4.415 / 3.554 |
| 4K / wide | 21.599 / 22.505 / 20.217 |
| 4K / compact | 21.558 / 18.791 / 22.404 |
| Flower 2000 / wide | 6.418 / 8.214 / 5.925 |
| Flower 2000 / compact | 7.150 / 8.648 / 6.987 |
| Flower 500 / wide | 0.485 / 0.600 / 0.474 |
| Flower 500 / compact | 0.485 / 0.600 / 0.475 |

Summed evaluator time favors phased reuse in 1/8 captures and deferred
stores in 5/8. The capacity improvement is not itself a latency result.

The 4K trace decomposition below separates the targeted 32×32 evaluator from
other shapes and unrelated GPU work. Values average the two labels per family;
summed GPU durations are not wall time and do not include host gaps.

| Width / family | 32×32 evaluator (ms) | All evaluators (ms) | Other GPU kernels (ms) |
| --- | ---: | ---: | ---: |
| wide / baseline | 3.734 | 21.599 | 127.425 |
| wide / phased | 3.846 | 22.505 | 130.417 |
| wide / deferred | 2.946 | 20.217 | 132.784 |
| compact / baseline | 3.637 | 21.558 | 132.876 |
| compact / phased | 3.513 | 18.791 | 131.935 |
| compact / deferred | 3.128 | 22.404 | 142.606 |

## Evidence and disposition

The `native_scan` and `linked` analysis children wrote complete reports, but
their wrappers then failed because job metadata used the same report names.
Both wrappers exited 1. Original logs/reports remain untouched; no successful
wrapper metadata is inferred. A separately recorded read-only audit verifies
the reports and module hashes. `wrapper_issues.json` records both failures.
They are logging failures, not qualified successful wrapper jobs or evidence
of a CUDA kernel failure. No artifact is rerun or overwritten to conceal them.

Keep S138's evaluator. The all-shape phased variant has no compelling timing
case. Deferred stores are more promising locally: both 16×16 and 32×32 are
faster in all eight short captures, matching the two improved resource
ceilings. At 4K, the deferred 32×32 kernel falls from 3.734 to 2.946 ms (wide)
and from 3.637 to 3.128 ms (compact). But the all-shape variant also lowers
resource ceilings elsewhere, and unchanged GPU work varies substantially.
These observations do not prove a causal explanation for whole-call regressions
or justify promoting a selector based on this same data.

Next independently test deferred reuse only for 32×32, then for 32×32 plus
16×16, retaining the normal kernels for other shapes. Add repeated per-stage
GPU-event timing and read-only clock/throttle telemetry alongside an
uninstrumented whole-call campaign. That should test whether the local gains
persist while distinguishing them from encoder-wide variation; the telemetry
must not be confused with power/clock control. Neither selective variant nor
the new measurement harness is implemented or qualified here. The encoder is
not established to be maxed out.

All 59 recorded jobs are terminal and accepted; the two additional wrapper
metadata failures are preserved separately. All four CUDA sanitizer jobs
qualify. Forty retained runtime files and reused frozen inputs retain their
hashes. No previous artifacts were removed or overwritten.

Evidence root: `build-cuda-ninja/profiles/s141-artifacts/`; scripts/prototypes:
`build-cuda-ninja/profiles/s141_*`. Key records are `before.json`, `inputs.json`,
`trace_inputs.json`, `native_scan.json`, `linked.json`, `occupancy_analysis.json`,
`reserved_shared.json`, `wrapper_issues.json`, `recovered_audits.json`,
`within_analysis.json`, `trace_analysis.json`, `trace_parts.json`,
`timing_summary.json`, logs, `final_summary.json`, `source_snapshot_index.json`
and `artifact_hashes.json`.
