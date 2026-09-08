# Malta cycle-rate observation (S119)

Date: 2026-09-08. Starting revision: `b2cd301`, branch `feat/cuda`.
Windows 11, RTX 3060 Laptop (sm86), CUDA 11.8, MSVC 14.37, Release.

## Measurement question

[S118](cuda-malta-allocation-s118.md) showed that the resident encoder's
early/late Malta slowdown survives both allocation paths. Counter captures
execute identical instructions with nearly unchanged traffic, but collection
itself removes much of the late-call slowdown. S119 observes cycle rate near
normal-stream launches without a counter profiler or clock/power changes.

The diagnostic is local to sm86. NVIDIA defines `clock64` as a cycle counter
and `globaltimer` as a nanosecond timer, while explicitly making the latter
target-specific and intended for tooling. SM identifiers may change after
preemption. Samples therefore retain both SM identifiers and exclude migrated
samples from rate calculations. This is not a portable runtime clock API or
a production feature. See the [CUDA 11.8 PTX register reference](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi).

## Probe and controls

A separate one-thread, one-block kernel samples the timestamp, cycle counter
and SM identifier over a requested 5 or 20 microsecond window. It writes one
48-byte record to independent guarded storage. Bounded loops prevent a stalled
or unsupported timer from hanging the process. Both probe versions use
18 registers, no shared memory, no stack frame and no spills. Native audits
check the ordered register reads and stores.

S117's GPU object is reused unchanged. Host launch hooks enqueue a probe before
and after each selected paired 32x64 Malta launch. Labels 0/2 have no probes;
1/3 have probes. All labels run the production Malta kernel, not the preload
prototype. The existing GPU bodies, allocation policy and storage format do
not change. Buffers are allocated once; resets, synchronization and readback
are outside the encode timer. Both label groups perform that bookkeeping.
Sample text is emitted only after the timing loop.

Each process uses four warm and twelve measured balanced Williams rows.
HD and 4K run with both sampling windows and opposite process orders. Flower
500 supplies zero-target preflights. Every encode checks the retained
codestream, baseline summary, targeted-call count and AC storage width/size.
The two probe copies and two controls measure perturbation rather than
assuming the observation is free.

Saved-input replay uses current S117 full-response calls 0 and 12, original
row strides, four launches per burst, six warm and twelve measured six-label
Williams rows, and two opposite orders. Three labels have probes and three
do not; every label uses production Malta. Each burst is checked against the
separate scalar scale/response reference, including guards and untouched
inputs. Replay event intervals include the probes when enabled. Their time
is not subtracted to manufacture an unobserved kernel-only duration.

The gap from the end of a pre-probe to the start of its post-probe brackets
Malta but can include stores, dispatch gaps and scheduling effects. It is not
an exact kernel-duration measurement. Likewise, the observed local cycle
rate is not a measurement of uninterrupted useful SM execution or a direct
identification of a thermal/power governor.

## Timer-phase control

The initial probe starts at an arbitrary timer phase. All 13,632 encoder and
replay observations have timestamps divisible by 1,024 ns and elapsed windows
of exactly 5,120 or 20,480 ns. The initial replay rates differ by window:
5 microseconds gives 1.463–1.486 cycles/ns, versus 1.572–1.576 at 20 microseconds.
This exposed a short-window measurement bias, not a kernel optimization.

The aligned probe waits for an observed timer tick before starting and uses
the same ordered timestamp/cycle/SM reads at both endpoints. It reuses the
original host objects and all non-probe GPU objects. After native audits,
standalone tests, all four CUDA sanitizers and encoder/replay preflights, the
entire timing campaign is repeated. The aligned replay rates now agree:
1.601–1.601 cycles/ns at 5 microseconds and 1.602–1.611 at 20 microseconds
(rounded ranges of job medians). This phase-controlled version is the primary
result. It does not make the timer portable or eliminate sampling overhead.

The aligned campaign also has 13,632 observations, the same timestamp quantum
and window lengths, and no detected migration. Both versions always report
SM identifier 0 at both endpoints. No claim about simultaneous rates across
the other SMs follows from that observation.

## Resident execution differs from replay

The following ranges span aligned job medians across the two window lengths
and two process orders. Cycle-rate ranges include both before/after medians.
Replay duration is the unprobed event interval per kernel; resident duration
is the probe-bracket gap and must not be treated as the identical boundary.

| Context | Local cycles/ns | Duration and boundary |
|---|---:|---:|
| Saved 4K calls 0 and 12, replay | 1.601–1.611 | 0.537–0.541 ms, unprobed event |
| HD call 0, resident | 1.276–1.381 | 0.174–0.182 ms, bracket |
| 4K call 0, resident | 0.708–0.772 | 1.044–1.138 ms, bracket |
| 4K call 12, resident | 0.288–0.313 | 2.551–2.781 ms, bracket |
| 4K call 18, resident, half resolution | 0.272–0.297 | 0.708–0.770 ms, bracket |

The later full-resolution response remains much slower than the earlier one,
and the endpoint cycle rate falls correspondingly. This persists without
counter collection. It is consistent with S118's unchanged instruction/traffic
counts and with a substantial execution-context rate difference; it does not
identify whether the cause is power management, clock gating, scheduling or
another system effect.

As a consistency check, multiply each full-response bracket gap by the mean
of its two endpoint cycle rates, then take each call's median. Across aligned
4K calls 0/1/12/13, both windows and both orders, this yields approximately
0.798–0.811 million cycles despite the large wall-time change. The initial
5-microsecond probe gave 0.723–0.753 million, whereas its 20-microsecond version
gave 0.787–0.793 million. The aligned result reduces the window dependence.
These products are approximate bracket normalizations, **not** cycles measured
inside Malta: endpoint rates need not hold throughout the bracket.

## Observation cost and limits

Aligned replay probes add 0.0193–0.0214 ms per kernel at 5 microseconds and
0.0456–0.0486 ms at 20 microseconds (about 3.6–4.0% and 8.5–9.0%). Launch and
alignment overhead are included; no correction is applied to the measurements.

Whole-encode paired medians change by +0.260 to +1.044 ms at HD and −3.581 to
+1.811 ms at 4K. The 4K duplicate-control differences range from −0.921 to
+11.094 ms, and duplicate-probe differences from −7.060 to +4.426 ms. Those
controls preclude interpreting negative probe deltas as a speedup or claiming
the observation has no effect. Rate/bracket correlation, repeated across both
windows and orders, is the useful result; whole-encode improvement is not.

## Qualification and evidence

There are 1,160 exact encode checks, including 60 scoped host-ASAN checks,
and 1,920 checked replay bursts against the scalar GPU reference. The two
versions retain 27,264 encoder/replay probe observations. Six accepted
standalone release/host-ASAN jobs each check 48 samples plus invalid-argument
handling; eight accepted CUDA sanitizer jobs each check 48 samples, with zero
reported memory, initialization, synchronization or race errors.

Two setup issues are preserved. The initial memcheck process returned zero
and reported zero errors but lost the host PASS marker; it remains unqualified.
A host-only explicit stdout flush fixed the marker, followed by a clean
memcheck repeat and the other sanitizers. Separately, the initial replay
native audit expected 95 GPU bodies but found 107 because the linked executable
also contains 12 dependency kernels. Its corrected audit compares the entire
retained 106-body replay baseline plus the new probe, not merely a subset.
Neither issue was an admin/firewall blocker or a device arithmetic failure.

Evidence is retained under `U:/gjxl-cuda-diagnostics/s119`, with the second
campaign under `aligned/`: executables, objects, logs, native dumps, source
snapshots, input/library hashes, both encoder and replay analyses, the derived
`cycle_rate_summary.json` and artifact hashes. Validation checks 112 recorded
jobs (111 accepted plus the preserved marker failure), all 32 timed-job
isolation windows against other recorded jobs, input identities, full aligned
native equivalence, encode/sample counts, sanitizer results and all forty
retained runtime hashes. All recorded jobs are terminal. No system, power,
clock or security settings were changed; user scratch files remain untouched.

## Decision and next experiment

Keep production kernels, pooling and dispatch unchanged. This study adds no
compatibility layer and does not promote S117's preload prototype. Small
replay gains cannot be assumed to transfer to the substantially lower local
cycle rates seen inside the 4K encoder.

The next bounded test should correlate read-only clock/power/temperature and
performance-limit telemetry with the same early/late resident phases and
replay, while measuring telemetry overhead with controls. If this platform
does not expose those readings, retain that limitation explicitly. Do not
alter system settings to make a benchmark look faster. Kernel scheduling work
should subsequently be qualified in the actual resident execution context;
the backend is not established to be maxed out.
