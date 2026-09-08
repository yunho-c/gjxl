# Malta tile-loader recurrence screening (S122)

## Outcome

Neither tested recurrence improves the current paired Malta kernel. Coordinate
recurrence increases executed warp instructions by 2.86%; carrying input
offsets increases them by 3.27%. Both preserve exact output and resource use,
but short replay is slower and sustained replay offers no repeatable win.
Production is unchanged. No compatibility layer or dispatch rule is added.

This follows [S121's sustained-workload study](cuda-malta-sustained-s121.md).
The experiment returns to reducing kernel work, while retaining short and
sustained timing boundaries and read-only power-limit endpoint checks.

## Candidates and native checks

The production 32-by-64 output tile uses 256 threads to load a 40-by-72 shared
tile including its halo. Its linear loader computes quotient/remainder by 40
and two strided input addresses for each item.

Four treatments are tested:

- Mode 0: production launcher.
- Mode 1: an independent copy with the production loader.
- Mode 2: coordinates advance by the recurrence `256 = 6 * 40 + 16`; a
  column carry selects a six- or seven-row advance.
- Mode 3: the same recurrence also carries the two unsigned input offsets.
  Invalid halo offsets may wrap as integers; pointers are formed only after
  the original coordinate bounds checks establish a valid location.

All candidates retain the shared locations, lane ordering, halo checks,
collective, paired responses, floating-point order, and initialization/addition
semantics. The wrapper deliberately forces the same paired tile for both
2D and flat-grid fixture coverage. No production fallback is introduced.

All four control bodies are native-identical to production. All 78 original
Butteraugli bodies are unchanged. The object has 12 prototype bodies; each
host executable contains those 90 target bodies and 12 unchanged linked-library
bodies. Full native comparisons cover release and host-ASAN executables.

Resources are unchanged across all three prototype loaders: 48/40 registers
for full/low-frequency 2D kernels and 56/47 for flat-grid kernels, with 11,520
bytes shared memory and zero stack/spill storage. Static full-2D instruction
counts rise from 540 in the control to 552/560 in the two recurrences; these
counts include compiled helper paths and are not executed-work estimates.

## Executed-work evidence

Eight Nsight Compute captures cover production, its native-identical control,
and both candidates in forward/reverse order on the current 3839-by-2159
full-response input captured at resident call 12 in S117. Clock and cache
control are explicitly disabled. Each report selects one eligible launch.

| Treatment | Executed warp instructions, both orders | Change from production |
| --- | ---: | ---: |
| Production and copy | 71,963,205 | — |
| Coordinate recurrence | 74,023,965 | +2.86% |
| Input-offset recurrence | 74,318,565 | +3.27% |

All eight reports have exactly 139,303,116 predicated-on FFMA thread
instructions and 367,200 shared-store wavefronts. Shared-load wavefronts vary
by less than 0.002%; DRAM traffic stays approximately 66.4 MB read and 33.2 MB
written, with about 150 MB of L2 traffic. Neither candidate removes floating-
point work or materially reduces memory traffic.

The native code explains why a source-level recurrence is not cheaper here.
The original constant quotient/remainder is already a multiply/shift sequence.
The recurrence introduces carry predicates, selections, coordinate updates,
and loop-carried values. The offset version still multiplies its selected
row increment by each stride and adds 64-bit offsets. These are observations
of this CUDA 11.8/sm86 compilation, not a general claim about recurrence loops.
Counter-profiled duration is not used as resident performance evidence.

## Short and sustained replay

Replay uses the same captured input, output geometry and strides for every
treatment. The selected full-response call initializes accumulation, making
repeated launches idempotent. A separate scalar scale/response reference
defines the final expected output. Every burst checks bitwise output, guards,
the unused scaled plane, and unchanged inputs.

Each of four timing jobs performs four qualification bursts, four warmup
rounds and eight measured rounds in randomized four-treatment Williams
blocks. Both orders are used for four- and 128-launch bursts. Five CUDA events
delimit the whole burst and its quarters. Copies, synchronization, validation,
and two read-only enforced-power-limit queries are outside the timed region.
There are no local-cycle probes or artificial idle intervals in these bursts.

The baseline for each measured round is the mean of modes 0 and 1. The table
reports medians of eight within-round comparisons, without power correction
or overhead subtraction. Positive candidate percentages mean slower.

| Burst / order | Baseline ms per launch | Coordinate change | Offset change | Copy vs production |
| --- | ---: | ---: | ---: | ---: |
| 4 / forward | 0.5540 | +1.08% | +0.47% | +0.14% |
| 4 / reverse | 0.5518 | +1.13% | +0.63% | −0.44% |
| 128 / forward | 2.0815 | +1.05% | +2.91% | −2.25% |
| 128 / reverse | 2.1176 | +0.55% | +0.23% | −0.57% |

All 448 unprofiled endpoint queries, including preflight and warmup, report
40 W; no endpoint change is observed. Equal endpoint readings do not prove
constant state throughout a burst. Sustained timing remains much slower than
short replay. Duplicate-control variation is also larger in sustained runs,
so small differences there should not be treated as precise effect sizes.
The candidates nevertheless provide neither a work-reduction mechanism nor
a repeatable measured win. No whole-encoder speedup or low-frequency timing
claim is made, and neither candidate advances to in-encoder qualification.

## Qualification, evidence and decision

Release and host-ASAN each pass 7,168 differential fixtures. Four CUDA
sanitizer jobs each pass 160 scoped fixtures with no reported errors or
hazards, for 14,976 fixture invocations in total, each checking three stages.
Coverage includes both frequency modes, initialization/addition, flat/2D
grids, partial and tall shapes, signed zero, subnormal/extreme values,
infinity, NaN and exceptional accumulation values.

Unprofiled replay passes 224 checked bursts, eight under host ASAN, with
14,784 logical Malta launches. The eight counter jobs each add one checked
four-launch burst; profiler-internal replays are not included in that logical
count. There are 42 complete accepted job records, including four timing
jobs, eight counter captures and eight counter exports. Timing isolation is
checked against other recorded jobs, not unknown desktop activity. All forty
retained runtime binary hashes pass.

Evidence is retained under `U:/gjxl-cuda-diagnostics/s122`: sources,
executables/objects, native dumps, capture/library identities, raw fixture,
sanitizer, timing and counter logs, analyses, environment snapshots and hashes.
An initial audit wrapper hit a report-filename collision after its native
child had passed; a separately recorded recheck passed. An initial preparation
assertion omitted the 12 dependency kernels and stopped before pinning inputs;
the corrected gate verifies them against retained native evidence. These
startup issues are preserved in `audit_notes.json`, not counted as successful
qualification jobs. The notes also record the build script's replay-support
extension after the initial fixture builds.

All jobs are terminal. No admin/firewall/permission blocker was observed; no
power, clock, security, process-priority or affinity setting was changed.
User scratch files are untouched. This is a rejected prototype study, not a
claim that the backend is maxed out.

A narrower follow-up is supported by the disassembly: original input-address
construction retains signed-extension and high-word correction operations
even after nonnegative coordinates pass their bounds checks. Explicitly
converting those validated values through `uint32_t` before widening is worth
testing without changing tile scheduling or arithmetic. It is not implemented
or qualified by this study.
