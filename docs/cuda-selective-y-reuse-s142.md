# CUDA selective shared-Y reuse and stage telemetry (S142)

Disposition: selective prototypes qualified for an integration experiment,
not promoted by this turn. Production remains
[S138](cuda-fused-ac-integration-s138.md). This independently tests the selective
hypotheses suggested by [S141](cuda-shared-y-reuse-s141.md), not a retrospective
selector promoted from S141's short traces.

## Variants and unchanged computation

All paths retain candidate-major 192-thread blocks. Six labels duplicate three
families: normal S138 evaluation (0/3), deferred residual stores for 32×32 only
(1/4), and deferred stores for 32×32 plus 16×16 (2/5). Every other shape keeps the
normal evaluator. Quant-norm preparation, final-cost composition, launch grids,
global scratch and the zero-forward-scratch contract remain unchanged.

The GPU object is reused byte-for-byte from frozen S141. Its exact-output,
barrier and resource qualification is not inferred from a reimplementation.
The new diagnostic batch dispatcher calls its deferred-store instantiations
only for the selected shapes. No production source, test source, coefficient
format, default width, arithmetic approximation or compatibility layer changes.

S141's resource queries apply to those same evaluator bodies: 32×32 static
shared storage is 25,344 instead of 33,536 bytes, with a modeled resource ceiling
of eighteen instead of twelve warps per SM. For 16×16, 13,056 instead of 17,152
shared bytes gives thirty-six instead of thirty warps. These are resource
ceilings, not achieved occupancy or throughput measurements. The experiment
avoids the other shapes whose register tradeoffs were unfavorable in S141.

Starting revision `f08e1d3`, branch `feat/cuda`, September 8, 2026 local time
(September 9 UTC). Windows, RTX 3060 Laptop `sm_86`, CUDA 11.8, MSVC 14.37 Release.
Fresh batch objects and callers use frozen S138 libraries and the hash-checked
S141 GPU object. Scoped host ASAN uses fresh clang-cl callers plus the same
source-identical frozen S138 host objects as the preceding experiments.

## Separate event and telemetry caller

The ordinary caller has no new GPU-event records or clock probes. It retains
the existing enforced-power-limit endpoints. A separate instrumented batch
and caller records a CUDA event immediately before and after each of the seven
evaluator launches. Fourteen events are allocated once and reused. After the
encode returns and endpoint telemetry is read, every ending event query must
succeed; this checks readiness at query time, not the exact return instant.
The probe adds no event/stream/device synchronization to the timed interval. It checks stage
count, shape, selector, event-pair stream and lifecycle, then obtains elapsed
times outside the measured encode call. Samples are buffered until process end.

These are GPU-stream stage intervals: they can include host-enqueue gaps, not
just kernel execution. They are not substituted for uninstrumented whole-call
latency. Instrumentation changes the workload between encodes, so absolute
times across the two caller families are not treated as paired measurements.

Read-only NVML queries before/after instrumented encodes record SM and memory
clocks, power draw, temperature, performance state and throttle-reason masks.
Every field retains its return status and every endpoint its QPC bounds.
Unsupported fields remain explicit rather than being silently replaced with
zero or interpolated. Endpoint telemetry is not continuous clock monitoring
and does not prove a state held throughout a kernel. No power, clock, thermal,
priority, affinity or firewall policy is changed.

The first seven-encode probe passes all frozen checks, 49 event intervals and
fourteen telemetry endpoints. All six telemetry APIs return success. Those
endpoints report SM clock 1,282 MHz, memory clock 5,500 MHz, GPU temperature
59°C, P3 and throttle mask 36 (`0x24`). The pinned CUDA 11.8 NVML header maps
that mask to software power-cap and software thermal slowdown. These are
driver-reported active flags, not an established physical explanation for
whole-call variance. No elevated privilege was required.

## Native audit and its failed initial assumption

The ordinary executable has all 235 S141 instruction/control bodies unchanged.
The reused deferred evaluators have no stack/local storage or static local-memory
loads/stores. Three fresh build jobs pass without warnings/errors; the reused
S141 object's original diagnostic warnings remain in its frozen evidence.

The first module audit failed an overly strong all-executables-identical
assumption. Its failed job metadata, log and extracted modules are retained.
A second read-only audit reuses those extracts rather than overwriting them:
twelve ordinary and six instrumented executables each have internally identical
ten-module sets, with nine modules common between the two groups.

In the differing two-kernel module, final-cost code is identical. The ordinary
quant-norm body matches S141's historical variant; the instrumented body matches
retained S138 exactly. Both have 35 registers, zero shared/stack/local storage
and unchanged source. Each is common to all three families within its executable.
This compiler variation is explicitly retained; the two build families are not
claimed to be binary-identical or directly paired whole-call measurements.

## Qualification and benchmark design

The guarded focused fixture covers fifteen block geometries through 257-block
thin axes, partial tiles, host/signed device CfL and descriptor/device quant
norms: 288 cases and 59,752 descriptors per execution. Three reset-arena
evaluations per candidate must match all normal costs, rates and losses exactly
and preserve every byte outside live outputs. The exhaustive extension covers
all block widths/heights 1 through 19 in four CfL/norm modes: 8,804 cases and
539,236 descriptors per execution. The broad contract fixture checks all three
paths, including invalid input/range cases, with 329 submissions per family.

Focused tests pass Release, scoped ASAN and all four CUDA sanitizers. Exhaustive
and contract tests pass Release and ASAN. The six focused and two exhaustive
runs execute 19,336 baseline batches and 58,008 batches per selective candidate;
repeats are not independent images. CUDA memory/init/synchronization checks
report zero errors, race checking zero hazards, and memcheck zero leaked bytes.

Complete-encode inputs retain frozen S133/S138 oracles: Flower 500, padded HD
from 1919×1079, padded 4K from 3839×2159, and Flower 2000 (fourfold nearest-
neighbor replication, not a native large photo). Distance 1.2, effort 7, fully
resident, wide and opt-in compact. Six-round Williams blocks balance positions
and ordered predecessors. Each timed process has one reference, six warm rounds
and eighteen measured rounds; a second process pass reverses image/width order.
Every encode checks bytes, summary, coefficient width/storage and seven branch
selections. Instrumented runs additionally check the actual per-shape selectors.

Ordinary and instrumented campaigns each have separate Release/ASAN preflights
and two timed passes, with prespecified independent seeds. All observations,
cross-label pairs and duplicate-label controls are retained. Per-shape GPU
intervals are analyzed with the same within-round pair-mean statistic. Clock
and throttle readings are descriptive evidence, not post-hoc filters used to
discard unfavorable timing observations.

This does not replace a fresh normal CMake/CTest campaign, broad quality/effort
promotion qualification, independent decoder testing or new Metal/Linux/
second-GPU validation.

## Results

Both timed campaigns and all preflights complete. There are 3,456 measured
encodes across 32 timed processes, plus warmup/reference calls. Together with
the initial probe and ten trace processes, 4,941 frozen-oracle encodes pass,
including 112 scoped host-ASAN encodes. No other recorded build or diagnostic
job overlaps a timed process. The instrumented callers verify 17,171 GPU-stage
intervals and 4,906 telemetry endpoints across preflights, timing, probe and
traces. Enforced-power-limit endpoints are separately retained for every encode.

The primary statistic is median within-round mean candidate-label time minus
mean baseline-label time, not the difference of separate medians. Negative is
faster. Quantization is nested inside whole-call time; per-stage medians and
phase deltas are not additive.

### Ordinary whole-encode timing

#### 32×32 only

| Case / width | Whole-call delta r0 / r1 (ms) | Quantization delta r0 / r1 (ms) |
| --- | ---: | ---: |
| Flower 500 / wide | −0.638 / +0.003 | −0.332 / −0.038 |
| Flower 500 / compact | −0.150 / +0.175 | −0.041 / −0.056 |
| HD / wide | +0.343 / −2.610 | +0.534 / −0.327 |
| HD / compact | −0.855 / +1.006 | −0.593 / +0.247 |
| 4K / wide | −1.669 / −3.434 | −1.235 / −0.837 |
| 4K / compact | −2.816 / +11.260 | +3.504 / +5.278 |
| Flower 2000 / wide | +0.906 / +3.761 | +1.815 / +2.224 |
| Flower 2000 / compact | −1.478 / +1.233 | −1.248 / −0.645 |

Favorable primary comparisons: 8/16 whole-call and 10/16
quantization; cross-label pairs: 29/64 and 34/64 respectively.

#### 32×32 plus 16×16

| Case / width | Whole-call delta r0 / r1 (ms) | Quantization delta r0 / r1 (ms) |
| --- | ---: | ---: |
| Flower 500 / wide | −0.400 / −0.346 | −0.334 / −0.072 |
| Flower 500 / compact | +0.032 / +0.113 | −0.177 / +0.026 |
| HD / wide | +1.004 / −0.831 | −0.170 / +0.863 |
| HD / compact | −0.226 / −0.083 | −0.069 / +0.140 |
| 4K / wide | −3.586 / −7.584 | −2.507 / −3.423 |
| 4K / compact | −0.259 / +3.291 | +0.293 / +3.120 |
| Flower 2000 / wide | +3.254 / +1.156 | +0.995 / +1.195 |
| Flower 2000 / compact | −2.220 / −0.060 | +0.276 / +0.280 |

Favorable primary comparisons: 10/16 whole-call and 7/16
quantization; cross-label pairs: 40/64 and 34/64 respectively.

The largest absolute duplicate-label whole-call control is −10.502 ms
(4K / compact, r0, labels 5 minus 2). Both candidates are
favorable in three of four 4K whole-call primary comparisons, including both
wide runs, but neither establishes a uniform encoder-wide gain.

### Repeated GPU-stage intervals

#### 32×32 only

| Case / width | 32×32 delta r0 / r1 (ms) | Sum of all seven stages delta r0 / r1 (ms) |
| --- | ---: | ---: |
| Flower 500 / wide | −0.028 / −0.028 | −0.028 / −0.023 |
| Flower 500 / compact | −0.024 / −0.025 | −0.023 / −0.034 |
| HD / wide | −0.131 / −0.145 | −0.122 / −0.155 |
| HD / compact | −0.135 / −0.144 | −0.141 / −0.159 |
| 4K / wide | −0.626 / −0.802 | −1.175 / −0.598 |
| 4K / compact | −0.698 / −1.170 | +1.962 / −0.058 |
| Flower 2000 / wide | −0.301 / −0.260 | −0.220 / −0.208 |
| Flower 2000 / compact | −0.243 / −0.280 | −0.212 / −0.269 |

32×32 improves in 16/16 primary comparisons and
64/64 cross-label pairs. Summed evaluator intervals improve in
15/16 and 54/64 respectively.

#### 32×32 plus 16×16

| Case / width | 16×16 delta r0 / r1 (ms) | 32×32 delta r0 / r1 (ms) | Sum of all seven stages delta r0 / r1 (ms) |
| --- | ---: | ---: | ---: |
| Flower 500 / wide | −0.005 / −0.005 | −0.028 / −0.027 | −0.032 / −0.027 |
| Flower 500 / compact | −0.004 / −0.004 | −0.025 / −0.023 | −0.030 / −0.029 |
| HD / wide | −0.033 / −0.031 | −0.132 / −0.142 | −0.154 / −0.158 |
| HD / compact | −0.037 / −0.034 | −0.140 / −0.142 | −0.168 / −0.167 |
| 4K / wide | −0.184 / +0.082 | −0.923 / −1.183 | −0.232 / −3.428 |
| 4K / compact | +0.128 / −0.625 | −0.346 / −1.038 | +1.523 / −3.614 |
| Flower 2000 / wide | −0.029 / −0.068 | −0.222 / −0.300 | −0.185 / −0.525 |
| Flower 2000 / compact | −0.056 / −0.057 | −0.295 / −0.296 | −0.143 / −0.316 |

32×32 improves in 16/16 primary comparisons and
64/64 cross-label pairs. Summed evaluator intervals improve in
15/16 and 57/64 respectively.

The added 16×16 stage improves in 14/16 primary comparisons and 55/64
cross-label pairs, with two unfavorable 4K stage comparisons. The unchanged
stages are retained in the raw analysis; their variation is not credited as
an algorithmic saving. Both variants have an unfavorable summed-GPU result
in the first compact-4K instrumented process despite a faster 32×32 stage.

### Telemetry and traces

The 3,456 endpoints associated with measured instrumented encodes all return
success for all six telemetry APIs:

| Reported field | Measured-endpoint range |
| --- | ---: |
| SM clock | 210–1,537 MHz |
| Memory clock | 5,500–5,500 MHz |
| Power draw | 23,052–59,687 mW |
| GPU temperature | 72–75 °C |
| Performance state | 3–3  |
| Throttle mask | 36–36  |

The reported enforced limit remains 40,000 mW at all 9,882 encode endpoints.
Clock readings nevertheless vary widely, while P3 and the `0x24` throttle
mask persist. Reported power draw and enforced limits are different telemetry
fields; neither is replaced or clipped to make them agree. Whole-encode
endpoints can occur in host-side gaps and may reflect sensor update intervals.
They do not establish the clocks during an evaluator or prove that thermal
throttling caused any specific latency delta. They do establish that a constant
enforced limit was insufficient evidence of a constant operating state.

Eight ordinary Nsight Systems CUDA/NVTX captures verify the exact one-shape and
two-shape dispatch, unchanged 192-thread grids, seven preparation/evaluation/
finalization launches each, and unchanged unrelated kernels and memcpy payloads.
Summed evaluator durations average the two labels per family:

| Case / width | Baseline / 32×32 only / both shapes (ms) |
| --- | ---: |
| HD / wide | 3.544 / 3.408 / 3.377 |
| HD / compact | 3.475 / 3.348 / 3.317 |
| 4K / wide | 21.698 / 18.501 / 20.469 |
| 4K / compact | 16.135 / 21.831 / 20.284 |
| Flower 2000 / wide | 6.898 / 6.449 / 5.905 |
| Flower 2000 / compact | 6.613 / 6.490 / 6.472 |
| Flower 500 / wide | 0.483 / 0.461 / 0.456 |
| Flower 500 / compact | 0.484 / 0.461 / 0.457 |

Both candidates are locally favorable in seven of eight ordinary captures,
with compact 4K unfavorable. Two additional instrumented 4K captures verify
identical kernels, launch dimensions/resources and copy payloads relative to
the corresponding ordinary captures. Each of their twelve labeled encodes
adds exactly fourteen `cudaEventRecord` calls and no other in-window CUDA API
count change. All in-window CUDA calls succeed; the caller also checks event
query, elapsed-time and destruction status outside those windows.

## Evidence and next action

The repeated 32×32 gain is the clearest result: 16/16 primary and 64/64
cross-label GPU-stage comparisons favor reuse for each variant. The two-shape
variant adds a smaller, less uniform 16×16 gain and is the next implementation
candidate, with 32×32-only retained as a diagnostic ablation. Integrate the
two-shape path into normal source, verify the resulting native bodies and
resource counts, then run broad quality/effort/width, contract and sanitizer
qualification with fresh normal builds and repeated performance controls.
No universal whole-encode speedup is claimed from the current 10/16 favorable
whole-call results. The current compact-4K reversal remains part of the record.

The new telemetry disproves the stronger assumption that a fixed enforced
power limit implied a fixed clock state. It does not settle the cause of all
whole-call variability: future clock observations should cover GPU-active
windows, not just encode endpoints, without changing device policy. Keep the
ordinary path separate from instrumentation. The encoder is not established
to be maxed out.

All 94 recorded jobs are terminal: 93 accepted and one preserved failed initial
module audit. Forty retained runtime files and all reused frozen inputs retain
their hashes. No prior artifact was removed or overwritten. All 18 linked GPU
executables are accounted for in the two audited module groups.

Evidence root: `build-cuda-ninja/profiles/s142-artifacts/`; scripts/prototypes:
`build-cuda-ninja/profiles/s142_*`, with the late report/verification/freezing
drivers named `report_s142.py`, `doc_results_s142.py`, `verify_s142.py` and
`freeze_s142.py`. The frozen S141 GPU object is an explicit
hashed input. Core records include `before.json`, `inputs.json`, `trace_inputs.json`,
`native_scan.json`, `linked_v2.json`, the failed `linked_job.json` and log,
`within_analysis.json`, `eventwithin_analysis.json`, `event_analysis.json`,
`trace_analysis.json`, `eventtrace_analysis.json`, `trace_parts.json`, logs,
`final_summary.json`, `source_snapshot_index.json` and `artifact_hashes.json`.
