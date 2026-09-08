# Sustained Malta replay (S121)

Date: 2026-09-08. Starting revision: `f3051e7`, branch `feat/cuda`.
Windows 11, RTX 3060 Laptop (sm86), driver 577.00, CUDA 11.8, MSVC 14.37.

## Hypothesis and controlled work

[S119](cuda-malta-cycle-rate-s119.md) observed much lower local cycle rates
beside late resident 4K Malta calls than in short replay.
[S120](cuda-malta-telemetry-s120.md) showed that management telemetry cannot
resolve that difference at the per-call timescale. S121 tests whether longer
device-only work is sufficient to reproduce a slowdown without the encoder's
other kernels, allocator or CPU policy work.

The input is S117's captured current 4K full-response call 12, with original
3839-element row strides. It initializes its output, rather than adding to it;
production source assigns each response directly in this mode. Repeating the
same kernel on unchanged inputs is therefore idempotent. The harness rejects
other geometry, low-frequency or accumulation modes. Its final output is
checked bitwise against one separate scalar scale/response reference, with
input and scratch/output guards checked as well.

Each process fixes a burst length of 4, 32, 128 or 512 launches. Input resets,
copies and result checks occur outside the burst; none are inserted between
its Malta launches. Five CUDA events delimit four equal launch-count quarters
and the whole burst. This describes submitted work, not a guarantee of
uninterrupted SM execution or freedom from OS/driver scheduling.

Four labels use the same production kernel. Labels 0/2 have no local probes;
1/3 use the unchanged S119 aligned 20-microsecond probe before and after
launches 0, 1, N/4, N/2, 3N/4 and N−1, with duplicate positions removed.
There are four selected launches for N=4 and six otherwise. Quarterly event
boundaries are identical in both label groups; enabled probes remain inside
their event intervals, and no overhead is subtracted.

Four qualification bursts are followed by four warm and eight measured
balanced Williams rows. Two process orders traverse the burst sizes in
opposite directions. New release/host-ASAN host executables link the retained
S117 GPU object and S119 aligned probe object without recompiling GPU code.
Their entire 107-body GPU native payloads match the retained replay baseline.

## An uncontrolled operating-state transition

The initial environment snapshot reports a 40 W current/requested limit.
After the campaign, it reports 71.14 W, while the default remains 60 W and
the maximum remains 80 W. No agent command changes power, clocks, priority,
affinity, driver or security settings. The user was informed and asked whether
charger or power-profile changes occurred; the snapshot difference alone does
not identify who or what caused the transition, or its precise time.

The original 512-launch first-order log contains a striking warmup transition.
Its first thirteen timed warm bursts average roughly 2.38–2.45 ms per launch.
The next warm burst changes from about 1.98 ms per launch in its first quarter
to 0.74–0.77 in the other quarters, followed by bursts around 0.77 ms overall.
These raw observations are retained, not discarded or relabeled as steady
measurements. Their coincidence with a before/after power-limit change is a
confound, not an identified causal intervention.

Accordingly, the first-order and reverse-order curves are not treated as a
single fixed-state comparison. A second host-only harness adds read-only
`nvmlDeviceGetEnforcedPowerLimit` calls immediately before and after each
burst, outside its event interval. Query status must succeed; values and host
QueryPerformanceCounter intervals are buffered and printed after the run.
The new host objects again have unchanged GPU native payloads. The complete
burst-size campaign is repeated with these explicit endpoint-state records.

## Endpoint-checked repeat

The added readings confirm variation, not a stable new limit. Across all
qualification, warm and measured bursts, there are 896 limit samples and 18
bursts whose before/after values differ. One 512-launch preflight reports
40→30 W, then 30→40 W, and later 40→69.484 W. During the measured population,
96 bursts have both endpoints at least 60 W and 160 have both at most 40 W;
five change their exact endpoint value while remaining in the higher band.
All measured lower-band endpoints are exactly 40 W. No middle/mixed-band
measured burst is omitted; none occurs in this particular measured population.

These are observational endpoint categories, not assigned experimental power
settings. Equal endpoints do not exclude a temporary change inside a burst.
The power getter's measured-job median cost is 5.85–7.6 microseconds, with a
largest call of 853.2 microseconds; all calls are outside the event interval.

The table uses medians of the sixteen measured **unprobed** bursts per job.
It does not estimate the causal effect of changing the power limit; the jobs
also differ in time and prior workload. Values are milliseconds per launch.

| Launches per burst | Both endpoints 40 W | Both endpoints at least 60 W |
|---|---:|---:|
| 4 | 0.549 | 0.535 |
| 32 | 0.718 | 0.538 |
| 128 | 2.221 | 0.693 |
| 512 | 2.476 / 2.600 | Not observed in the measured repeat |

Within the 128-launch, 40 W-endpoint job, unprobed first-quarter and
last-quarter medians are 0.710 and 3.130 ms per launch. The median paired
last/first ratio across those individual bursts is 4.32. The corresponding
higher-endpoint job gives 0.560 and 0.798 ms, with a 1.44 ratio. Both show a
within-burst change; the lower-endpoint run is substantially more severe.

Sparse local probes agree with this execution-history difference. At launch
127 of the lower-endpoint 128-launch job, before/after medians are both about
0.301 cycles/ns; the higher-endpoint job is about 1.024–1.025. At launch 511
of the two lower-endpoint long-burst jobs, medians are about 0.294–0.305.
Their brackets are roughly 2.64–2.73 ms. This approaches S119's late resident
rates and durations without the encoder's other kernels or allocation path.
Brackets still include dispatch/probe-boundary effects, and endpoint probes
do not measure uninterrupted useful execution inside Malta.

Probe deltas are not uniformly small or positive: some lower-endpoint
32/128-launch comparisons and duplicate controls vary substantially. Both
versions retain complete per-quarter event data, paired probe/control deltas,
duplicate differences and raw local samples. No probe subtraction or
cross-state timing correction is applied. Unprobed quarter timings are the
independent evidence that this is not merely time spent running local probes.

A separate post-campaign read-only Windows status check reports AC online,
83% battery and battery saver off on all three samples, alongside a 40 W
enforced GPU limit. These are after-the-fact observations, not evidence that
AC status was unchanged during earlier transitions. Status-field meanings
follow Microsoft's [system power status reference](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-system_power_status).

## Qualification and retained evidence

Both campaigns pass 448 checked replay bursts apiece. The scoped memcheck job
adds four, for 900 checked bursts and 151,552 production Malta launches in
total. Thirty-two checked bursts run under host ASAN. There are 4,952 guarded
local-probe records including memcheck, with no detected migration, and 896
power-limit records in the endpoint-checked campaign. Memcheck reports zero
errors. The GPU objects retain their earlier differential and four-sanitizer
qualification; no new full-encoder test run is claimed for this host-only study.

Evidence is retained under `U:/gjxl-cuda-diagnostics/s121`, with the repeated
campaign in `power_checked/`: executables, objects, source snapshots, complete
native dumps, capture/library identities, raw timing/probe/limit logs, both
`analysis.json` files, `power_analysis.json`, before/after environment snapshots
and artifact hashes. Validation checks all 46 recorded jobs, sixteen timing
isolation windows against other recorded jobs, native equivalence, guard/PASS
counts, source/input identities and all forty retained runtime hashes. All
jobs are terminal; no admin/firewall/permission blocker was observed. User
scratch files are untouched, and no system setting was changed by the agent.

## Decision and optimization consequence

Sustained replay can reproduce several-fold slowdown with unchanged inputs,
parameters and native kernel math. Extra encoder arithmetic or an
encoder-specific memory layout is not required for the phenomenon. The
observed operating-limit changes also invalidate a single context-free replay
throughput number. They do not identify the responsible software, firmware,
thermal sensor or power-source event.

Production remains unchanged, with no new compatibility layer or dispatch
heuristic. Future candidate screening needs sustained as well as short replay,
explicit operating-state records, and final in-encoder qualification. Artificial
idle or pacing would require complete-encode evaluation; none is introduced.

The next concrete kernel experiment is reducing the scaling tile loader's
integer address work: replace repeated quotient/remainder and row-address
reconstruction with an equivalent coordinate recurrence while preserving
contiguous lane accesses and floating-point order. Native instruction/resource
checks can establish whether that actually removes work before timing it under
the relevant resident-like conditions. The backend is not proved maxed out.
