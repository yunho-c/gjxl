# Post-selection cadence intervention (S177)

September 10, 2026. Starting revision: `235c1fe`, branch `feat/cuda`.
Windows, RTX 3060 Laptop (`sm_86`), CUDA 11.8, MSVC 14.37 Release;
scoped host ASAN uses clang-cl 22, traces use Nsight Systems 2023.2.3.

## Outcome

Changing only the host pause after successful GPU strategy selection changes
the execution time of the identical downstream GPU sequence at 4K. This is
controlled evidence of cadence sensitivity in this setup, beyond S175/S176's
observation that removing host work accompanies slower subsequent kernels.
It does not identify a clock, power, thermal, cache, preemption or driver
mechanism. No production pause, scheduler, compatibility layer or runtime
change is promoted, and the GPU selection prototype remains experimental.

An 8 ms request actually pauses about 16–17 ms in the unprofiled 4K group
medians. Versus GPU selection without a pause, quantization outside the
strategy frontend drops 20–21 ms, and complete calls improve 5–10 ms in two
reversed process repetitions. Separate traces confirm that the identical
230 downstream kernels execute 13–16 ms faster. At 1080p the downstream
interval barely changes, so pauses simply make complete calls slower.

## Intervention and controls

The S175 frontend source copy has exactly two edits: include a diagnostic
pause header, then call its helper after successful candidate grid export
and statistics assignment, immediately before returning. The source audit
reconstructs both edits. Original CPU selection does not call the helper.
The helper uses `std::this_thread::sleep_for`, measures its actual wall
duration and records its invocation count. A zero request returns without
sleeping or reading its local duration timer. No high-resolution timer,
clock, power, priority, affinity or security setting is changed.

The pause is inside the existing complete frontend and complete-call timers,
but outside merge/export, readback and submission-wait scopes. Its intended
or actual duration is never subtracted from complete-call time. Quantization
minus the complete frontend is computed within each encode before pairing;
that residual excludes the pause and is initially only a downstream proxy.

Four variants each have duplicate labels in one DLL:

| Labels | Selection | Requested pause |
| --- | --- | ---: |
| 0, 1 | Original CPU | None |
| 2, 3 | GPU candidate | 0 ms |
| 4, 5 | GPU candidate | 8 ms |
| 6, 7 | GPU candidate | 24 ms |

The diagnostic configuration resets per-thread pause state between calls
and rejects unsupported requests. Every candidate call records exactly one
pause-helper invocation; original calls record none. Zero requests must
record zero elapsed pause. Nonzero measured pauses must be at least their
requested duration and fit inside the frontend scope. Output, full summary,
native coefficient owner and S175's branch/transfer/host-capacity checks
remain mandatory on every encode.

The corrected S175 backend, cost/selection kernel and original CPU search
objects are reused without rebuilding. New normal/ASAN DLLs compile only
the frontend and pause helper; new harnesses provide the eight-label schedule.
All twelve CUDA module hashes match S175 exactly. Device cost computation,
selected grid, metadata, coefficient representation and GPU launch settings
are unchanged. The helper is diagnostic plumbing, not a production API.

## Unprofiled campaign

All eleven standard inputs, automatic/eight-thread budgets and normal/ASAN
builds pass 44 preflight processes, 396 checked encodes. The 22 ASAN processes
include 176 instrumented DLL calls and 22 normal reference calls, not 198
fully instrumented executions. No fresh CUDA sanitizer suite is run because
no GPU code changes; S175's sanitizer evidence is not recounted as new testing.

1080p and 4K then run at eight requested threads, twice in reversed case and
label-order schedules. Each process checks one reference, eight warm and
sixteen measured eight-label rounds: 772 encodes, 512 measured. Each cycle
uses the base order `01726354` and its seven modulo-eight rotations. Positions
and all 56 directed within-round predecessor pairs are balanced. The source
audit checks those properties. Inter-round transitions and longer history
are not claimed to be completely balanced.

Group statistics average the two identical labels within each round, take
variant differences, then the median across sixteen rounds. Percentages use
the corresponding within-round ratios. All four individual label-pair
medians and duplicate-control deltas remain available. Ranges below span two
process repetitions, not confidence intervals. Negative means faster.

| Input / requested pause | Actual pause group medians (ms) | Downstream proxy vs GPU-zero (ms) | Whole call vs GPU-zero (ms) |
| --- | ---: | ---: | ---: |
| 1080p / 8 ms | 15.598–17.619 | −0.044 to −0.027 | +15.810 to +16.818 |
| 1080p / 24 ms | 33.714–34.614 | −0.323 to −0.051 | +32.408 to +33.409 |
| 4K / 8 ms | 15.795–16.964 | −21.257 to −20.428 | −10.335 to −5.001 |
| 4K / 24 ms | 30.575–31.121 | −29.554 to −25.409 | −3.123 to +8.256 |

The 4K zero-pause candidate again has a slower downstream proxy than original
CPU selection: +14.443 to +15.786 ms. Its whole-call changes are mixed,
−0.812 to +7.327 ms. The nominal 8 ms candidate pause improves on GPU-zero
in both repetitions, but versus original CPU selection the complete-call
changes are −3.918 to +3.137 ms. The nominal 24 ms variant is +0.705 to
+11.252 ms slower than original. No result supports a production sleep or
an image-size/content threshold fitted to these two cases.

Duplicate controls remain substantial. At 4K, identical original-label
whole differences are −3.324/+10.808 ms; zero-pause GPU duplicates are
−6.854/+8.149 ms. The paused variants also vary. The repeat and within-round
controls constrain interpretation but cannot remove shared-machine variation
or all effects carried over from earlier labels. No sample is filtered or
rerun, and no power/clock correction is applied to its timing.

## Separate trace follow-up

After the timing campaign, a new normal/ASAN harness adds NVTX complete-call
markers without rebuilding either DLL. Two preflight processes check 18
encodes. Two 4K/eight-thread captures use eight warm and eight measured
eight-label rounds, reversing order in the second: 258 encodes, 128 measured.
The 256 labeled windows include warmups; reference calls remain outside them.
CUDA/NVTX capture has no CPU sampling or context-switch tracing. Embedded
stdout is recovered from SQLite and checked through terminal completion.

The analyzer verifies identical ordered standard kernel signatures and copy
sequences, including launch dimensions and resource counts. All three GPU
pause variants execute the same 296 kernels; original CPU selection executes
295. The only extra kernel is selection after 65 common prefix launches;
the remaining 230 standard kernels are identical. Every one of 89,408 GPU
activities across the windows correlates with a successful runtime call in
its own window and belongs to one measured device/context/stream/process per
trace. The reference context is excluded.

The copy difference remains solely S175's seven cost readbacks totaling
2,088,480 bytes replaced by one 137,760-byte grid/error readback. The 25-copy
prefix, 20-copy suffix and memset work match. There is no transfer or device
work change among the three GPU pause variants.

| Traced 4K metric vs GPU-zero (ms) | 8 ms request, r0 / r1 | 24 ms request, r0 / r1 |
| --- | ---: | ---: |
| Actual pause group median | 15.561 / 17.273 | 35.681 / 33.623 |
| Common prefix kernel execution | −0.014 / −0.096 | +0.337 / +1.526 |
| New selection kernel execution | −0.0001 / −0.090 | +0.007 / −0.055 |
| Common downstream kernel execution | −13.454 / −16.233 | −21.426 / −23.704 |
| Gaps within complete device span | +14.584 / +19.405 | +34.810 / +33.273 |
| Complete call | −1.142 / −4.701 | +6.624 / +10.961 |

The expected larger gap accompanies shorter actual downstream kernel
execution, confirming the unprofiled proxy's direction. The original suffix
starts at `AdjustQuantFieldKernel`, after the last `FinalizeCostKernel` in
the prefix. Groups share this matched split; all sums and residuals are
computed per encode before pairing. Independent table medians are not
additive. Traced effect sizes differ from unprofiled ones, and traced timings
do not replace unprofiled results or establish production throughput.

## Interpretation and next investigation

The intervention demonstrates that later submission timing can change the
observed cost of otherwise identical GPU work on this system. It does not
separate power/clock management, cache state, scheduling, preemption or other
hardware/driver effects. S113's slowly refreshed NVML SM-clock values are not
reused as per-kernel exposure data; a constant 40 W limit is not a constant
clock assertion. The experiment changes neither those settings nor the
quality/effort/output contract to obtain a favorable comparison.

A next diagnostic can compare device-side cycle and global-time spans inside
representative downstream kernels. NVIDIA documents `%clock64` as a cycle
counter, but marks `%globaltimer` as target-specific tooling functionality
whose behavior can change. Any such probe needs validation on this `sm_86`
target, instrumentation-off controls, retained resource/code checks and a
clear distinction between elapsed cycles and useful instruction work.
This is a proposed measurement, not a clock result from S177.
[NVIDIA PTX ISA 11.8 timer definitions](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi).

## Evidence and disposition

Root: `build-cuda-ninja/profiles/s177-artifacts` on C. Helpers and diagnostic
sources are `build-cuda-ninja/profiles/s177_*`. The stage checks 1,444 whole
encodes: 396 initial preflight, 772 ordinary campaign, 18 trace preflight
and 258 captured-process calls. There are 54 journaled jobs, all accepted,
with no rejected build, encode or capture and no restarted job. Native module
extraction commands/output are retained separately in build metadata.

All 1,536 ordinary and 512 traced warm/measured NVML endpoints report a
40,000 mW enforced limit. No task build, sanitizer, heavy archive/hash sweep
or unrelated GPU job overlaps either campaign; light source editing and
ordinary shared-machine activity remain limitations. No admin/firewall
blocker occurs. Nothing changes power, clocks, cooling, priority, affinity,
driver, firewall or security settings. The diagnostic sleeps remain outside
production, and no new compatibility layer is introduced.

| Phase (UTC, September 10) | Start | Finish |
| --- | --- | --- |
| Normal/ASAN build | 06:16:01.569 | 06:16:42.121 |
| All-input preflight | 06:17:35.291 | 06:18:59.929 |
| Unprofiled campaign | 06:19:16.464 | 06:21:28.176 |
| Trace-harness build | 06:24:42.542 | 06:25:00.576 |
| Trace-harness preflight | 06:25:00.648 | 06:25:09.790 |
| Two captures including export | 06:26:19.818 | 06:27:48.951 |

The S175 inventory of 869 artifacts and S176 inventory of 408 artifacts are
reverified. Pinned helpers, predecessor binaries and source snapshots are
not overwritten or rebuilt. No material file is deleted. Each capture checks
for at least 900 MB free on C; about 993 MB remains at the observed post-check.
The U drive is not used for new captures. Durable reports, SQLite, embedded
stdout, original measurements, decisions and source/build/protocol pins are
retained; transient `temp` scratch is excluded from the immutable inventory.

Verification recomputes ordinary and trace statistics from raw logs/SQLite,
checks schedules, scopes, actual pauses, transfer/owner counters, module and
input hashes, correlated activities, all job outcomes/nonoverlap and the
final source/artifact inventory:

```powershell
python build-cuda-ninja/profiles/s177_verify.py --frozen
```

No new CTest, install-consumer, independent-decoder, concurrent/batch encoder,
Metal or Linux qualification is claimed. No change is promoted. The goal
remains reducing actual fully resident encode time, not fitting a pause to
this laptop; the encoder is not established to be maxed out.
