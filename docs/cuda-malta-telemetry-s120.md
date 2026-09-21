# Resident Malta telemetry (S120)

Date: 2026-09-08. Starting revision: `645bf23`, branch `feat/cuda`.
Windows 11, RTX 3060 Laptop (sm86), driver 577.00, CUDA 11.8, MSVC 14.37.

## Question and measurement boundaries

[S119](cuda-malta-cycle-rate-s119.md) observed much lower local cycle rates
beside later resident 4K Malta calls than in saved-input replay. S120 asks
whether read-only management telemetry can identify the operating constraint.
[S113](cuda-quantization-attribution-s113.md) already found that sampled SM
clock values changed only about every 500 ms. Re-querying a cached value is
not a per-kernel clock measurement.

The device initially reports an enforced/requested 40 W power limit, a 60 W
default and an 80 W maximum. These are observations, not settings selected by
this study. No clock, power, scheduling, priority, affinity or security setting
is changed. No setter API is loaded by the telemetry probe.

The observer loads the installed NVML DLL, resolves the GPU by PCI bus ID and
records each getter's status and host query interval. It reads SM/memory clocks,
GPU temperature, power usage, the enforced limit and clock-event reasons.
A batched field query adds timestamped limit counters, energy, specifically
requested instantaneous/average power, and power-limit fields. Field IDs and
units follow [NVIDIA's NVML field reference](https://docs.nvidia.com/deploy/nvml-api/group__nvmlFieldValueEnums.html).
Unsupported results remain unsupported; they are not converted to zero.

Current [NVML power-usage documentation](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html)
describes averaging on non-GA100 Ampere devices, whereas the installed 11.8
header does not specify that detail. S120 therefore retains the generic getter
separately from the explicitly requested instantaneous/average fields. Neither
a field's name nor a recent query establishes its effective sensor cadence.

## Controlled harness

S119's aligned probe and S117's GPU objects are reused without recompilation.
Both new host executables and their scoped host-ASAN versions have complete
GPU native bodies identical to the retained aligned encoder/replay executables.
Every label uses production Malta, never the preload prototype.

Each encode or replay burst is bracketed, outside its timed interval, by two
guarded 20-microsecond clock-mapping probes. Host QueryPerformanceCounter bounds
around each synchronized mapping launch bound the offset to device globaltimer.
The intersection of before/after bounds is used only when consistent. A
management query is associated with a Malta bracket only if its entire host
interval lies inside that bracket under every offset in the intersection.
This mapping assumes a constant offset across one encode/burst, not across
the whole process. Mapping does not make a cached sensor value fresh.

Four-label encoder controls retain S119's two unprobed and two probed copies,
four warm and twelve measured balanced Williams rows, 20-microsecond local-rate
probes and exact codestream/summary/storage checks. HD and 4K each run twice
with background telemetry enabled and twice disabled, reversing process order.
The disabled observer initializes NVML and takes only before/after samples;
enabled observers additionally poll at a requested 20 ms interval. Both buffer
child output and telemetry; detailed observations are printed after completion.

Saved-input replay uses S117's current 4K full-response call 12 and original
strides, four launches per burst, three probed and three unprobed labels, six
warm and twelve measured Williams rows, and two opposite orders with telemetry
on/off. All outputs, untouched inputs and guards are checked against the
separate scalar GPU reference. Event intervals include enabled local probes,
but exclude clock mapping and management readback. Cross-process telemetry
comparisons are observational controls, not paired within-process effects.

## Results: cadence limits attribution

All 2,032 clock-mapping anchors form consistent before/after offset
intersections. Job-median widths are 16.5–27.6 microseconds for measured
encodes and 23.1–24.6 microseconds for replay. The complete observer campaign
retains 2,425 management samples; 49 SM-clock queries can be placed entirely
inside measured Malta brackets using the conservative offset bounds.

The following are ranges of enabled measurement-job medians, not requested
sampling rates or guarantees of fresh values:

| Observation | Observed cadence/cost |
|---|---:|
| Management poll start interval | 31.15–31.23 ms |
| Complete management query group | 2.52–2.80 ms |
| Changed SM-clock values | 498.8–499.7 ms |
| Changed instantaneous-power field | 93.2–94.3 ms |
| Changed power-limit counter | 498.5–500.2 ms |

The largest complete query group in the enabled measured jobs takes 15.3 ms.
No high-resolution timer or scheduling setting is changed to force the
requested 20 ms poll interval.

Clock mapping exposes the cached-clock problem directly. Across the two
enabled 4K jobs, the small subsets below have management queries confidently
inside the kernel brackets. Local rates are means of the two endpoint probes
for those same brackets; they are not readings from inside Malta itself.

| 4K call | Matched queries | NVML SM clock | Local endpoint cycles/ns |
|---|---:|---:|---:|
| Early full-response call 0 | 3 | 262–757 MHz | 0.673–0.781 |
| Late full-response call 12 | 6 | 270–1,282 MHz | 0.275–0.303 |

The NVML readings cannot safely normalize individual kernel timings, even
when the query is accurately placed in time. Across all four 4K measurement
jobs, ordinary local probes still show early medians around 0.767–0.818
cycles/ns and late medians around 0.285–0.304. The late slowdown is present
with management polling both enabled and disabled. Replay remains near
1.603–1.613 cycles/ns.

## Limit flags and counter caveats

Every reason query wholly inside a measured encode or replay burst reports
`0x24`: software power-cap and software thermal-limit flags according to
[NVIDIA's clock-event definitions](https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksEventReasons.html).
The enforced limit remains 40 W and reported memory clock 5,500 MHz. Eligible
GPU temperatures span 62–67 C; memory temperature is not measured. The same
flags occur during much faster replay, so they do not identify the cause of
the resident/replay difference or prove a particular sensor crossed a limit.

The timestamped power-limit and below-base-clock counters advance in roughly
500 ms batches, identically in these runs. Across the active observer window,
their net increments are approximately 0.98–1.05 times elapsed wall time;
cache-boundary phase can produce values slightly above one. Within individual
encodes, adjacent-query ratios have median zero and occasional values as high
as 22.3. These are delayed counter updates, not meaningful per-call duty-cycle
fractions. The old thermal-policy field is zero throughout; the newer software
thermal, hardware thermal and hardware power-brake counter fields return
`NVML_ERROR_NOT_SUPPORTED`. Zero in the old field does not cancel the reported
flag or supply the missing newer measurement.

The explicitly requested average-power field is unsupported. The generic
power getter and instantaneous-power field are retained separately. In the
two 4K jobs their eligible medians are respectively about 37.3–37.5 W and
39.5–40.1 W. Short outliers and differing update phases preclude treating these
as exact per-kernel power. The energy field is also retained but not used for
an energy claim: its net 4K deltas imply about 288–290 W over the observation
windows, inconsistent with these power readings. That counter needs separate
validation before use; this study does not diagnose the discrepancy.

## Observer effects and qualification

Using unprobed encoder labels, telemetry-on minus telemetry-off whole-encode
job medians are +4.35/−1.60 ms at HD and −0.35/−9.42 ms at 4K in the two
orders. Quantization changes are +2.45/+0.23 ms and +1.29/−4.16 ms respectively.
Replay's unprobed event differences are −5.50/+3.20 microseconds per kernel.
These cross-process results are mixed, and the within-process duplicate
controls also drift. They do not establish a speedup from telemetry, zero
observer effect or a correction to apply to production timings.

All 550 observer-wrapped encodes pass exact codestream, summary, call-count,
storage and guard checks, including 15 scoped host-ASAN checks. Replay passes
480 checked bursts, including 12 under host ASAN. The new host clock-mapping
path additionally passes an HD memcheck job with five exact encodes and zero
reported errors: 555 exact encodes overall. The observer jobs retain 11,424
local-rate samples and 2,032 mapping samples; memcheck adds 48 and 8. No GPU
body is changed, and the reused probe retains S119's four-sanitizer coverage.

The initial telemetry analysis used an integer parser for replay's `ms`
field. Correcting that analysis-only parser completed the audit using the
same logs; no measurement was rerun or discarded for that issue.
Two analysis result constructors also gained the telemetry-enabled metadata
bit after the preparation snapshot. `analysis_addendum.json` records that
exact insertion; the original source bytes are preserved and match their
preparation hashes. No timing formula, measured input or executable changed.

Evidence resides in `U:/gjxl-cuda-diagnostics/s120`: 31 recorded jobs, native
dumps, executables and host objects, inputs and library identities, raw logs,
encoder/replay/telemetry analyses, `summary.json`, source snapshots and artifact
hashes. All jobs are terminal. No admin, firewall or permission blocker was
observed, and user scratch files remain untouched.

## Decision and next test

Production kernels, pooling and dispatch remain unchanged, with no new
compatibility layer. The available management API reports a constrained
operating state but cannot resolve its cause at the early/late-call timescale.
Further polling of the same cached fields is not the next optimization step.

The next bounded experiment should test workload continuity: replay the
idempotent full-response input for progressively longer device-only bursts,
keeping input copies and checks outside each burst, and compare early/late
local rates against ordinary short replay. Short replay currently interleaves
four Malta launches with substantial copies and validation; resident encoding
has a different work history. Sustained replay can test whether that history
is sufficient to reproduce the rate drop without changing system settings.
Then evaluate promising instruction/scheduling reductions under the relevant
resident-like operating state, rather than assuming short-replay gains transfer.
