# Malta full-batch tails and input preloading (S117)

Date: 2026-09-08. Starting revision: `dbca212`, branch `feat/cuda`.
Windows 11, RTX 3060 Laptop (sm86), CUDA 11.8, MSVC 14.37, Release.

## Experiment

[S116](cuda-malta-batching-s116.md) preserved contiguous lane accesses but
failed to reduce executed warp instructions. S117 removes per-item tile-tail
checks from guaranteed full batches, then tests input preloading separately.
All variants keep the 32x64 output tile, 256 threads and 11,520 static shared
bytes. Their scalar scaling helper, rounded divisions, paired-row response
arithmetic, zero-tile collective and accumulation semantics are unchanged.

The shared tile contains 2,880 values including its halo. Two-value batches
process a 2,560-value full prefix followed by a 320-value scalar tail;
four-value batches process 2,048 followed by 832. Each unrolled item retains
contiguous lane accesses. Only the full prefix loses per-item tail checks;
ordinary image/halo bounds checks remain. The launcher always uses 256 threads,
as required by this prefix calculation and the existing output mapping.

The preload variant loads both reference/distorted pairs into scalarized local
arrays before calculating either scaled result. Invalid halo values bypass
scaling rather than evaluating zero-valued inputs through potentially
exceptional divisions. The tail remains the ordinary scalar loader. Native
code confirms four predicated input loads precede the first scaling arithmetic;
the non-preload full-batch version starts arithmetic after its first pair.
Static instruction order is not a measurement of dynamic issue overlap.

| Labels | Scaling variant |
|---|---|
| 0 / 4 | Actual production launcher |
| 1 | Copied one-iteration control |
| 2 | Two-value full batches, separate scalar tail |
| 3 | Four-value full batches, separate scalar tail |
| 5 | Two-value full batches with input preloading, scalar tail |

Production selects 32x64 on the timed HD/4K cases. Fixed 32x64 candidates are
not identity controls for production's smaller tile selections on tiny images.
Labels 0/4 are duplicates; label 1 is native-identical for full/LF and 2D/flat
forms. The preload and non-preload two-value bodies differ in all four forms.

The diagnostic retains all 78 production GPU bodies unchanged and adds 16
experiment bodies. All four release/host-ASAN executables match the 94-body
object. The two-value variants keep baseline 2D register counts of 48 full /
40 LF; four values use 55 / 49. All 20 reported target/reference resource
records have zero stack frames and spills. Response shared-load occurrences
remain 70 scalar loads per paired body; static occurrences are not dynamic
per-output instruction counts.

## Protocol and scope

The timing protocol uses six retained S65 first-full-scale captures at each
of 1919x1079 and 3839x2159, with two opposite-order repetitions: 24 jobs.
Stages 0/1 are full/init, 2/4 dense LF/add and 3/5 zero LF/add. These are
historical replay inputs, not current-encoder captures or every AQ scale and
iteration. The experiments do not establish an integrated encode-time gain.

Each job checks six untimed qualification bursts, then six warm and twelve
measured six-label Williams rows. Each burst has four launches and is compared
bit-for-bit with the separate scalar scale/response GPU reference. Uploads,
resets and checks are outside CUDA-event intervals. The initial default-stream
upload is explicitly synchronized before using the nonblocking replay stream.
Timing jobs are isolated from other recorded agent jobs; clocks, cache policy,
power settings, priority and affinity are not modified.

Within each measured row, variants are compared with the mean of controls
0/1/4. The preload-versus-non-preload contrast is also measured directly as
5/2; subtracting their independently aggregated contrasts is not equivalent.
Reported ranges of paired medians are not confidence intervals.

The correctness protocol covers 10,752 three-stage fixtures per executable:
14 shapes, six labels, requested 2D/flat forms, full/LF, initialize/add and
16 patterns. It includes odd boundaries, thin/tall planes, independently
padded strides, nonzero offsets, signed zero, subnormals, thresholds, NaN/Inf,
sparse data and exceptional accumulation. Candidate flat routes are forced;
production labels retain their own dispatch. Inputs and guards are checked.
Host ASAN instruments the harness, not GPU code or every linked library.

## Historical replay results

Ranges below span the paired medians for stages and repetitions in each group.
Negative is faster. The preload variant beats the non-preload two-value batch
directly in all 24 jobs, whereas removing tail checks alone is not compelling.

| Replay group | Two-value prefix | Four-value prefix | Two-value preload |
|---|---:|---:|---:|
| HD full | -1.56 to +0.77% | +6.52 to +10.02% | -4.41 to -3.53% |
| HD dense LF | -0.26 to +0.86% | +9.19 to +10.32% | -3.60 to -2.48% |
| HD zero LF | -1.38 to -0.12% | +6.39 to +8.31% | -3.13 to -1.58% |
| 4K full | -0.42 to +0.20% | +15.21 to +18.38% | -4.004 to -3.938% |
| 4K dense LF | -0.302 to -0.090% | +11.39 to +12.16% | -3.324 to -2.789% |
| 4K zero LF | -0.135 to +0.058% | +4.64 to +5.19% | -1.579 to -1.163% |

Duplicate production controls span approximately -1.04 to +0.79% at HD and
-0.26 to +0.28% at 4K. These are same-process microkernel results, not encoder
speedups.

## Native execution and counters

Nsight Compute captures full, dense-LF and zero-LF 4K stages for production,
two-value prefix, four-value prefix and preload: two opposite orders, 24
reports, one matching kernel and four replay passes each. Clocks/cache controls
are `none`; counter collection is separate from event timing.

Executed warp instructions for preload increase 0.661% full, 0.678% dense LF
and 1.376% zero LF; FFMA thread counts are unchanged. Global-load sectors rise
0.25–0.27%, 0.13–0.14% and 0.77–0.80%, respectively. L2 traffic changes by
about +0.49%, -0.38% and -0.31 to -0.13%. DRAM traffic and shared wavefronts
remain approximately unchanged. Together with the native load ordering, this
supports an input-scheduling explanation for the historical replay gain, not
reduced arithmetic or data traffic. It does not prove dynamic load overlap.

Four-value prefixes reduce warp instructions only 0.88%, 0.53% and 1.08% but
raise 2D registers to 55/49 and lower the register-limited CTA count to four
from five/six. The observed LF shared carveout changes from 100 to 64 KiB.
This variant is consistently slower and is not an integration candidate.

## Current encoder integration

A diagnostic public-launch wrapper selects baseline or preload only where
production selects paired 32x64 tiles. All other dispatches remain unchanged.
Only the original launcher's definition is renamed; internal callers still
reach the wrapper. Object and four release/host-ASAN executables preserve all
94 original/prototype GPU bodies exactly. Linked libraries are the qualified
S114 production build, with hashes recorded in `within_inputs.json`.

Four labels provide two baseline and two preload copies. Four warm and twelve
measured balanced Williams rows run in two opposite case orders on Flower 500,
1919x1079 and 3839x2159. Every encode compares codestream bytes to a retained
oracle, summary to the baseline, and AC storage width/size. Separate executables
measure ordinary encode times and event intervals around every targeted Malta
launch. Event allocation and elapsed-time reads are outside the encode timer;
the event recordings themselves perturb execution. These interval sums can
include enclosed launch gaps and are not profiler kernel-duration sums.

| Case | Targeted calls | Malta interval baseline | Preload difference, repetitions |
|---|---:|---:|---:|
| Flower 500 | 0 | 0 ms | No targeted work |
| HD | 12 | 2.24 / 2.30 ms | -0.049 / -0.071 ms (-2.19 / -3.05%) |
| 4K | 24 | 26.32 / 27.40 ms | +0.069 / +0.277 ms (+0.27 / +1.06%) |

The uninstrumented whole-encode medians improve 0.84/0.95 ms at HD and
5.77/9.54 ms at 4K, but 4K baseline-duplicate differences are -10.95/-3.09 ms
and preload-duplicate differences +5.84/-5.66 ms. Whole-call variation cannot
be attributed entirely to Malta. In particular, a whole-encode gain does not
override the missing 4K stage gain. Production is not changed on this evidence.

## Input provenance and current-call capture

The S65 replay uses `s50_phase_probe.cpp::MakeSynthetic`; the current integrated
test reads the retained `cuda-padded-1080p.pfm` and `cuda-padded-4k.pfm`. Direct
comparison with that same compiled generator finds every R/G value different,
and all but one HD / two 4K B values different. Maximum absolute differences
are approximately 0.089, 0.111 and 0.123 per channel. These are different image
signals, not just a newer codec revision processing identical inputs.

A new baseline capture records every eligible call and verifies both final
codestreams exactly. HD has twelve full-resolution calls. 4K has twelve
3839x2159 and twelve 1920x1080 calls; the latter use reference/distorted/output
strides 1920/3839/3839. The old capture had only the first six full-resolution
calls. This establishes coverage gaps, not by itself the cause of the timing
discrepancy.

Capture synchronizes the existing stream and copies live rows to packed host
planes. Initialize calls do not read uninitialized accumulation; their stored
initial plane is zero. The existing capture header records packed width
strides, while capture logs retain actual strides. A new replay restores each
original stride from those logs, padding with sentinels and checking all
guards/gaps. Capture timings are deliberately not used as performance data.

Both packed and stride-restored replays pass 36 release and 36 host-ASAN
preflights each. The stride-restored host executables preserve all 94 GPU
bodies exactly. Current-call timing uses the same six-label, four-launch-burst
protocol as the historical screen: 36 calls in forward and reverse order,
72 jobs. Each restores the captured inputs and original row strides; it does
not reproduce the encoder's original allocation addresses, cache history or
host submission conditions. Independent replay call medians must not be
summed and presented as an observed encode duration.

| Current capture group | Preload paired-median range | Faster jobs / jobs |
|---|---:|---:|
| HD 1919x1079 full | -4.11 to -2.32% | 8 / 8 |
| HD 1919x1079 LF | -2.96 to -0.92% | 16 / 16 |
| 4K half-scale 1920x1080 full | -4.59 to -2.12% | 8 / 8 |
| 4K half-scale 1920x1080 LF | -2.35 to +0.36% | 15 / 16 |
| 4K full-scale 3839x2159 full | -3.23 to -2.40% | 8 / 8 |
| 4K full-scale 3839x2159 LF | -3.39 to -0.99% | 16 / 16 |

Preload wins 71/72 contrasts against the within-row control mean, but not all
direct contrasts against the non-preload two-value batch. The half-scale LF
case is least convincing; its production-duplicate medians span -1.19 to
+1.01%. Four-value prefixes regress every current-call job. Simple two-value
prefixes remain mixed and generally small relative to duplicate variation.

For scale only, summing independent call medians gives about 2.09/2.10 ms for
HD baseline and 9.83/9.82 ms for 4K baseline. Their preload sums are about
2.5% and 2.3% lower, respectively. These are constructed replay sums, not
measured encode durations. The much larger 26–27 ms in-encode 4K event sum
therefore remains unexplained. Current inputs and original strides do not
eliminate the replay/integration discrepancy; blaming the old input alone
would be unsupported.

## In-encode kernel trace

Two Nsight Systems captures use the existing uninstrumented four-label
`within.exe`, with twelve measured and four warm Williams rows each. CUDA
tracing is enabled, CPU sampling/context-switch tracing disabled, and system
clocks unchanged. No timing-event wrapper or new GPU body is introduced.
Both traces pass 65 exact encode checks. Each contains exactly 1,560 targeted
Malta kernels, grouped by encode and checked against all 24 current-call
geometries, full/LF specializations and baseline/preload labels. CUDA runtime
correlation IDs associate each kernel with its launch API interval.

| Trace | Baseline kernel sum | Paired preload difference | Baseline / preload duplicate differences |
|---|---:|---:|---:|
| Repetition 0 | 24.81 ms | -0.497 ms (-2.04%) | -0.324 / -0.624 ms |
| Repetition 1 | 26.03 ms | -0.132 ms (-0.51%) | -1.183 / -0.286 ms |

These are medians of paired per-encode kernel-duration sums. Differences of
independently reported medians need not equal the paired-median difference.
Launch-API interval sums are only about 0.32–0.37 ms per baseline encode.
The large in-encode total is therefore present in recorded kernel durations,
not merely an artifact of event intervals enclosing host submission gaps.
Those timestamps do not measure uninterrupted SM execution or isolate device
preemption, stalls and clock behavior.
It is not valid to subtract durations from separate trace/event campaigns
and assign their difference to a measured overhead component.

Execution context matters strongly within an encode. For example, baseline
full-response call 0 is about 0.96–0.97 ms in the two traces, while the later
same-size call 12 is about 2.63–2.75 ms. Their independently replayed medians
are both about 0.55 ms. Individual preload contrasts also change sign between
traces. These observations do not isolate clock behavior, allocation/cache
effects or another cause; they show that the replay gain alone is inadequate
to predict integrated performance.

## Decision, qualification and evidence

Do not promote the preload variant yet. Guaranteed full batches alone do not
produce a convincing improvement, four-value batches are consistently worse,
and two-value preloading is a reproducible replay improvement without a stable
4K integrated qualification. This does not establish that preloading cannot
help. It rejects an unconditional production replacement on the evidence
available. No compatibility shim, dispatch heuristic or runtime setting is added.

The next bounded investigation should explain the baseline's within-encode
kernel slowdown before further load-scheduling tuning: retain per-call
provenance and geometry, compare early/later execution contexts, and distinguish
device clock/cache/allocation or scheduling effects with direct evidence. Do not describe
the remaining gap as host submission overhead merely because it appears in
event timings; the kernel trace contradicts that simple explanation.

Qualification passes 10,752 three-stage fixtures in release and the same count
under scoped host ASAN, plus four CUDA sanitizer jobs with 240 fixtures each
(memcheck, racecheck, initcheck and synccheck), with no errors or race warnings.
There are 24 historical replay preflights; current captures add 72 packed and
72 stride-restored preflights, split equally between release and host ASAN.
All 96 timed replay jobs and 24 counter captures compare every burst with the
scalar GPU reference. Twelve whole-encode timing jobs and their preflights
pass 840 encode checks, including 30 scoped host-ASAN checks. Capture adds two
and kernel tracing adds 130 exact encode checks: 972 total in this study.
These are repeated checks, not distinct images. No new full CTest suite is
claimed; unchanged production retains S114's qualification.

Evidence is retained at `U:/gjxl-cuda-diagnostics/s117`: prototypes, current-call
captures, release/host-ASAN binaries, native dumps, raw timings, counter and
system traces, input/source/library hashes and source snapshots. Key results
are `screen.json`, `summary.json`, `mechanism_analysis.json`,
`within_analysis.json`, `events_analysis.json`, `fresh_summary.json` and
`trace_analysis.json`. Frozen replay/capture/analysis scripts include
`s117_fresh_pitched.py`, `s117_fresh_replay.cpp`, `s117_trace.py`,
`s117_validate_final.py` and `s117_freeze_final.py`.

The final validator checks all 368 job exit statuses and log hashes, 108
timed-job isolation windows, source/input/capture/library identities, native
and sanitizer reports, encode checks, trace kernel counts and all 40 retained
runtime hashes. All launched processes are terminal. No admin, firewall or
permission blocker was observed; no system/security settings were changed.
