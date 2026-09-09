# Per-encode serializer worker pool (S163)

## Experiment

Production remains S162 (`cd37bd4`). S163 tests whether reusing serializer
workers within one encode improves fully resident throughput. It changes no
production source, CUDA kernel, scheduling default, or compatibility layer.
The experimental executor and serializer overlay stay in the ignored
diagnostic artifact tree; this report is the committed deliverable.

Decision: do not promote the eager-capacity pool. Reuse is promising for
sparse 513 and flower 500, but this version regresses tiny retained work and
does not reliably improve larger real-image whole encodes. Keep the S162
production implementation and investigate on-demand capacity separately.

`RunParallelSections` currently starts and joins a new set of threads for
each parallel section phase. The diagnostic adds an alternative per-encode
pool. On its first parallel batch the pool eagerly creates its maximum
capacity: eight background workers for automatic budgeting, or seven plus
the caller for the explicit eight-thread budget on this machine. Each batch
activates only the required number of participants. Idle workers sleep on a
condition variable. An atomic index distributes independent tasks, statuses
are collected by task index, and all participants finish before the caller
consumes results. No worker persists into the next encode.

The caller's explicit budget and participant tracker propagate into workers.
Explicitly nested parallel work remains serial, as in production. Automatic
nested calls retain the legacy implementation; the experiment does not
silently serialize them or recursively reuse a busy pool. The diagnostic
context lives on the coordinator thread, not arbitrary worker threads.

Both policies execute from the same loaded DLL object. The harness checks
module-handle equality, sets each independent state's policy before timing,
and alternates those states. This removes cross-policy DLL/code-layout
differences, but not differences between the states' allocations or between
their execution histories. Legacy-only and pool-only two-state controls are
included rather than assuming those effects are absent.

Both policies include the diagnostic context, counters, and an executor
object; legacy leaves that object's worker list empty. This is a controlled
comparison of the two instrumented policies, not an uninstrumented
production-versus-experiment speed claim.

Pool construction, worker creation, and shutdown remain inside the outer
encode timer. Existing internal profile totals end before the bridge scope
destroys the pool, so outer time is the decision metric. Root-dispatch
counters record calls, worker creations, creation spans, and pool shutdown.
Creation spans may overlap useful worker execution; they are not additive
overhead that can simply be subtracted from total time. Counters do not claim
to count arbitrary nested threads or GPU-backend workers.

## Qualification

The standalone executor test passes in MSVC and host-ASAN builds. Each run
executes 3,608 dispatches in eight scopes, varying batch sizes from zero to
1,000 under automatic, one-, two-, and eight-thread budgets. It checks
exactly-once tasks, worker-index bounds, budget propagation/restoration,
worker reuse, deterministic first-error selection, exception translation,
recovery after task errors, nested context restoration, and four concurrent
encoder scopes. Fifteen injected partial-start failures check that started
workers are drained, no task runs, and the stopped pool cannot be reused.

Codec-level tests pass in normal and ASAN builds. Each uses 96 frames across
all six dense/sparse coefficient representations, eight transform strategies,
and two group shapes. Each frame compares legacy and pooled bytes under
automatic/eight-thread budgets and balanced/maximum-compression behavior:
384 comparisons, 768 serializations per build. Explicit tracked participation
never exceeds its budget. These tests exercise nested maximum-compression
work and per-call executor teardown. ASAN is not a data-race detector; no
ThreadSanitizer result is claimed.

Thirty-two initial whole/retained comparisons cover real flower 500 and 4K,
dense 65, sparse and two dense 513 patterns, and dense 1025 under both CPU
budgets, plus same-policy flower controls. Actual whole encodes pass CUDA
memcheck with full leak checking and initcheck: zero errors and zero leaked
bytes/allocations. Raw CUDA extraction from both diagnostic DLLs matches
S162's eleven modules exactly. No CUDA source is recompiled. The normal
90-test CTest suite from S162 is predecessor evidence, not a new S163 run.

Two rejected build journals remain intact: the first overlay lacked its
creation-timing variable declaration, and the first codec ASAN link omitted
the ASAN runtime libraries. Corrected source/build jobs use new output names
or link the existing objects after adding those runtime libraries. The
normal codec executable from the latter build had already linked; both codec
executables subsequently pass their own recorded tests. No firewall/admin
blocker or OS settings change occurs.

## Timing protocol

The predeclared matrix crosses two repeats with both state-creation orders,
shuffling jobs across whole and retained modes. Automatic and explicit
eight-thread budgets are separate. Whole comparisons cover six real images,
65/pattern 2, 513/patterns 0/1/2, and 1025/pattern 2. Retained comparisons cover
flower 500, 65/pattern 2, 513/patterns 0/2, and 1025/pattern 2. Both modes have
legacy-only and pool-only controls for flower 500, dense 513, and dense 1025.
This yields 136 whole jobs and 88 retained jobs, 224 total.

Each job uses four warmup and eight measured rotating duplicate-ABBA rounds.
The primary statistic is the median of within-round duplicate-label mean
differences; four cross-label medians are sensitivity checks, not confidence
intervals. All 41 saved phases use that same statistic. Worker-aggregate times
are not additive wall times, and no null subtraction is performed.
Percentage and millisecond medians are computed separately and may straddle
zero on a near-neutral job; faster counts below use the percentage sign.

Whole mode uses fresh fully resident effort-7/no-final-score workflows with
a retained backend per state. Retained mode deep-copies one GPU-produced
frame per state before timing and serializes it with the whole workflow's
resolved entropy/order options. Every output and public summary must match
the oracle; real inputs also match frozen S153 bytes. Retained AC FNV-1a
fingerprints/counts/nonzeros match, and GPU allocation/submission counters
must stay unchanged after capture. FNV is not collision-free proof.

Input creation, output clearing/destruction, comparisons, audit queries,
and NVML observations are outside timing. Pool shutdown is inside it.
No build or sanitizer overlaps the timing campaign. Power, clocks, affinity,
priority, firewall, and security settings are untouched.

## Performance results

All entries below are primary outer-time percentage changes; negative is
faster. Each cell lists R0/O0, R0/O1, R1/O0, R1/O1. The saved analysis also
contains every millisecond delta, every phase, and all cross-label checks.

### Fresh whole encodes

| Case | Automatic CPU budget | Eight-thread budget |
|---|---|---|
| 1025_p2 | -0.391, +1.379, -3.144, -2.299 | -2.622, -5.290, -1.741, -0.952 |
| 1080p | -1.233, -0.670, -1.729, -2.276 | +0.081, -0.936, -1.815, -3.557 |
| 4k | -1.541, +1.185, -1.568, -0.481 | -2.020, +1.963, +3.580, -2.472 |
| 513_p0 | -5.218, -11.321, -11.340, -7.753 | -6.353, -6.496, -9.621, -5.909 |
| 513_p1 | -0.049, +3.233, -3.942, +0.092 | -0.920, +1.775, -1.392, +1.673 |
| 513_p2 | -1.249, +0.519, +6.389, -5.225 | -6.169, +2.552, +4.412, -3.876 |
| 65_p2 | -1.938, +1.949, +1.444, -0.278 | +4.235, +1.016, +0.698, -1.451 |
| flower_2000 | -1.804, +0.249, -1.197, -5.180 | -2.567, +1.888, +0.169, +1.613 |
| flower_3200x2160 | +0.644, -1.504, -3.083, -3.149 | +1.392, -0.146, +1.822, +3.456 |
| flower_500 | -2.608, -2.100, -4.937, -1.135 | -0.868, -1.628, -3.497, -8.956 |
| keong_3839x2159 | -3.298, +0.878, -3.739, -3.209 | -1.760, -1.781, -1.125, +0.570 |

### Retained serialization

| Case | Automatic CPU budget | Eight-thread budget |
|---|---|---|
| 1025_p2 | -2.400, +1.424, -2.004, -0.055 | -7.599, -2.882, -7.693, +0.429 |
| 513_p0 | -22.168, -25.180, -23.312, -22.708 | -21.200, -21.737, -24.096, -22.624 |
| 513_p2 | +2.602, -2.750, +2.289, -7.206 | +2.354, +6.421, +2.403, -1.757 |
| 65_p2 | +1.885, +5.540, +4.612, +4.039 | +3.057, +3.903, +3.911, +1.334 |
| flower_500 | -7.186, -6.965, -3.948, -4.814 | -6.083, -9.041, -9.388, -3.601 |

### Same-policy, two-state null controls

| Mode / policy / case | Automatic CPU budget | Eight-thread budget |
|---|---|---|
| whole / legacy / 513_p2 | +4.306, -1.293, +3.278, +0.617 | -0.492, +0.370, -5.052, -2.996 |
| whole / legacy / 1025_p2 | +4.171, -1.306, -3.173, +0.966 | +3.749, +1.759, +0.312, -1.731 |
| whole / legacy / flower_500 | -1.735, +2.253, +2.025, +0.338 | -0.449, +2.698, +1.813, -0.344 |
| whole / pool / 513_p2 | +5.032, -1.633, -10.909, -0.471 | -3.418, -1.254, -0.122, +0.088 |
| whole / pool / 1025_p2 | -2.760, +4.414, -0.166, +2.917 | -3.444, +1.987, +2.453, +9.070 |
| whole / pool / flower_500 | -0.299, +3.147, +0.124, +2.140 | +0.217, +0.734, -3.592, -3.247 |
| retained / legacy / 513_p2 | +0.920, -3.355, +1.834, -3.100 | +10.351, +7.399, +2.553, -3.203 |
| retained / legacy / 1025_p2 | +2.719, -0.255, -2.240, -3.900 | +3.635, -2.058, -1.981, +2.543 |
| retained / legacy / flower_500 | -0.561, +2.661, -5.664, -0.162 | +1.854, -1.489, +0.929, +0.415 |
| retained / pool / 513_p2 | -0.510, -2.475, +5.329, -0.909 | -1.007, +1.866, +3.895, +3.979 |
| retained / pool / 1025_p2 | +1.326, -5.139, -4.956, -1.576 | -0.327, -2.848, -0.181, +2.042 |
| retained / pool / flower_500 | -2.461, -1.857, -1.893, -2.716 | +2.230, +0.387, -2.505, -1.900 |

## Interpretation and retention decision

Whole-encode primary outer time is faster in 60/88 comparisons, with 215/352
cross-label checks faster. Real-image results account for 34/48 primary and
120/192 cross-label improvements, ranging from -8.956% to +3.580%. That is a
promising tendency, not a general throughput win. In particular, explicit
eight-thread flower 2000 and flower 3200 improve in only one of four primary
trials each, while 4K improves in two. Dense 513/pattern 1 and pattern 2 each
split four improvements/four regressions across the two budgets.

Sparse 513/pattern 0 is the clearest positive result: all eight whole primary
and all 32 cross-label comparisons improve, by 5.218-11.340% primary outer
time. All eight retained primary and all 32 retained cross-label comparisons
also improve, by 21.200-25.180%. Flower 500 improves in all eight primary
trials in each mode: 0.868-8.956% whole and 3.601-9.388% retained, with 28/32
and 31/32 cross-label improvements respectively. There is no sparse-513
same-policy control in this matrix; the flower controls are nonzero.

The tiny 65/pattern 2 retained case is the clearest negative: all eight
primary outer results regress by 1.334-5.540%, with 27/32 cross-label checks
slower. Its internal stream total improves in all eight primary trials.
Excluding shutdown would therefore produce the wrong retention decision for
this case. The pool creates eight automatic or seven explicit-budget workers,
while legacy creates only four or two at the recorded root dispatches.
Eager capacity is visibly excessive;
the timing does not prove that excess creation alone explains the regression.

Dense 513/pattern 2 with eight threads remains unresolved: whole results split
two improvements/two regressions; retained results regress in three of four
primary trials. But retained legacy-only controls also regress in three of
four, by +10.351%, +7.399%, +2.553%, and -3.203%. The whole pool-only dense
513/automatic control reaches -10.909%, and dense 1025/eight reaches +9.070%.
Even within one loaded object, distinct states and execution history can
produce substantial apparent changes. These controls are not subtracted and
do not causally exonerate either policy. Retained totals overall improve in
25/40 primary and 105/160 cross-label comparisons.

Every measured comparison records five root serializer dispatches. On larger
real images legacy creates 25 automatic or 20 explicit-budget background
workers across those recorded root calls, versus eight or seven in the pool.
The following
retained/eight-thread diagnostics are descriptive medians over the 64
measured samples per policy across four trials, not paired speed statistics.
All times are milliseconds; section wall and token worker-sum columns are
legacy to pool.

| Case | Workers created, legacy / pool | Legacy creation span | Pool creation span | Pool shutdown | Section wall | Token worker sum |
|---|---|---|---|---|---|---|
| 65/p2 | 2 / 7 | 0.077 | 0.152 | 0.341 | 1.652 to 1.616 | 0.878 to 0.873 |
| Sparse 513/p0 | 16 / 7 | 0.385 | 0.152 | 0.318 | 1.495 to 0.695 | 0.379 to 0.389 |
| Flower 500 | 9 / 7 | 0.248 | 0.167 | 0.386 | 2.496 to 2.192 | 2.915 to 3.022 |
| Dense 513/p2 | 17 / 7 | 0.493 | 0.184 | 0.461 | 18.820 to 19.277 | 62.911 to 66.350 |

Sparse section wall time falls substantially without reduced token-worker
time. This supports pursuing orchestration costs for short tasks; it does not
show less coefficient or entropy work. Creation spans overlap useful work,
legacy join time is not separately isolated, and worker sums include elapsed
scheduling effects. Adding or subtracting these columns would not reconstruct
outer time. Dense section writing still consumes tens of milliseconds, so
reuse alone does not eliminate that bottleneck.

Keep production at S162. Do not promote this eager-capacity implementation or
choose an input-specific enablement threshold from this matrix. A useful next
experiment is an on-demand per-encode pool that starts only the workers a
batch requires, with the same explicit-budget and nested-work semantics.
Measure creation, wakeup, task-grain scheduling, and teardown together. This
does not rule out a persistent cross-frame executor, which has a different
ownership/lifetime contract and was not tested here. The dense AC section
writer also remains a separate optimization target. No GPU tile, compact
owner, or fused reduction policy changes in this stage.

## Evidence

Artifacts are under `build-cuda-ninja/profiles/s163-artifacts`. Preparation
verifies all 1,951 frozen S162 files and archives its 340 production sources.
The three protected untracked Markdown files remain unread and excluded.
The completed timing campaign ran on 2026-09-09 from 19:25:54 to 19:41:20
UTC: 7,168 measured calls, 3,584 warmups, 224 oracle calls, and 176 untimed
whole-encode captures for retained states. All 21,504 NVML observations
report a 40,000 mW enforced limit; this does not assert constant operating
power, clocks, or temperature.

The verifier recomputes the entire analysis, checks the exact diagnostic
source overlay against unchanged production, verifies predecessor and build
input hashes, and confirms every job log and timing order. There are 271
journals: 269 accepted and the two rejected build attempts described above.
No build or sanitizer job overlaps the campaign. The source snapshot and
artifact manifest retain helpers, binaries, native modules, logs, protocols,
and computed results. Production is unchanged and the optimization goal
remains open.
