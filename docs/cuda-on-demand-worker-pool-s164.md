# On-demand serializer worker capacity (S164)

## Experiment

Production remains the S162 implementation, at report-only S163 commit
`b6d73ec`. S163 found consistent sparse-513 and flower-500 benefits from
per-encode worker reuse, but eager maximum capacity regressed tiny retained
serialization and larger whole-encode results were mixed. S164 tests the
specific capacity problem without changing production or GPU kernels.

Decision: retain the experiment and evidence, not a production policy change.
On-demand capacity addresses the eager pool's tiny-work problem, but loses to
eager on retained sparse 513 and does not reliably improve larger whole
encodes. Keep production at S162 and next isolate the bit-writing work.

Three policies execute from the same loaded diagnostic DLL: legacy threads,
S163-style eager per-encode capacity, and on-demand per-encode capacity. The
new policy starts only the workers required by a parallel batch and retains
them until the encode ends. It grows to the largest requirement encountered,
never shrinks mid-encode, and activates only the current batch's allowance.
Automatic budgeting retains the legacy caller-waits behavior; explicit
budgeting includes the caller. Zero/single-participant work remains serial.
Explicit nested work remains serial, and automatic nested calls use the
legacy fallback rather than recursively entering a busy pool.

Growth occurs only after the previous batch's completion barrier. New workers
start with a snapshot of the current generation, captured under the mutex,
so they cannot treat an old generation as fresh work. Publishing a new batch
still uses the mutex/condition-variable protocol, an atomic task index, and
task-indexed statuses. A partial thread-start failure stops and drains all
workers, including workers retained from earlier successful batches, before
returning failure. No worker survives into the next encode.

Every policy includes the diagnostic context, executor object, and counters;
legacy never creates workers in that object. Module-handle equality is
checked, so policies share the same loaded code. Distinct state allocations
and execution histories are not assumed equivalent; same-policy controls are
measured. This is an instrumented-policy comparison, not a speed claim against
an uninstrumented production executable.

Root serializer counters record calls, worker creations, growth events,
creation spans, and shutdown. They exclude arbitrary nested and GPU-backend
worker creation. Creation spans can overlap useful work and are not additive
overhead. The primary outer timer includes pool construction, growth, wakeup,
and shutdown. Existing internal profile totals end before wrapper teardown.

## Qualification

The standalone executor passes normal MSVC and host-ASAN builds. Each runs
7,216 stress dispatches in 16 scopes, including four concurrent encoders
running both policies. Batch sizes range from zero to 1,000 under automatic,
one-, two-, and eight-thread budgets. Assertions cover exactly-once work,
worker-index bounds, capacity at every batch, reuse, budget/depth propagation
and restoration, first-error selection, exception handling, error recovery,
and nested diagnostic context restoration.

Each build also passes 15 eager partial-start injections, 240 on-demand
growth-failure injections, and 256 dispatches with changing budgets. Growth
failure tests increment the requested capacity across successful batches,
verify that the failing batch executes no tasks, and reject reuse of the
stopped pool. The changing-budget test checks parked workers do not violate
the next batch's allowance. These are additional to the 7,216 stress calls.
ASAN is not a race detector; no ThreadSanitizer result is claimed.

Codec tests pass in normal and ASAN builds. Each uses 96 frames across all
six dense/sparse coefficient representations, eight transform strategies,
and two group shapes. Under both CPU budgets and balanced/maximum-compression
behavior, legacy bytes are compared with eager and on-demand bytes: 768
comparisons and 1,152 serializations per build. Explicit tracked participation
stays within its budget. No new production CTest run is claimed.

Sixty-two whole/retained qualification comparisons cover flower 500 and 4K,
65/pattern 2, 513/patterns 0/1/2, and 1025/pattern 2 under both CPU budgets,
plus three same-policy flower controls in each mode. Actual whole encodes
pass CUDA memcheck with full leak checking and initcheck: zero errors and
zero leaked bytes/allocations. Both diagnostic DLLs retain exactly the eleven
raw CUDA modules from S162. Only diagnostic CPU sources are compiled; frozen
S162 normal and ASAN libraries are linked into new S164 outputs.

## Timing protocol

The predeclared campaign has 416 jobs: 232 whole and 184 retained. Legacy to
on-demand comparisons cover six real whole inputs and five synthetic cases;
retained comparisons cover flower 500 and four synthetic cases. Eager to
on-demand comparisons focus on tiny 65, sparse 513, and flower 500 in both
modes. All three same-policy controls cover tiny 65, sparse 513, dense 513,
dense 1025, and flower 500 in both modes. This adds the tiny and sparse control
coverage missing from S163. Automatic and explicit eight-thread budgets are
separate, with two repeats crossing both state-creation orders. Jobs are
shuffled across modes within each repeat.

Each job uses four warmup and eight measured rotating duplicate-ABBA rounds.
Primary changes are medians of within-round duplicate-label mean differences.
Percentage and millisecond medians are separate and can straddle zero near
neutrality. Four cross-label medians are sensitivity checks, not confidence
intervals. All 41 saved phases use this protocol; worker-aggregate times are
not additive wall times and controls are not subtracted.

Whole mode performs fresh fully resident effort-7/no-final-score workflows
with a retained backend per state. Retained mode captures and deep-copies a
GPU-produced frame once per state before timing, then serializes using the
whole workflow's resolved options. Every output and public summary must
match; real inputs also match frozen S153 bytes. Retained AC fingerprints,
counts, and nonzeros match, and GPU allocation/submission counters cannot
change after capture. FNV is not collision-free proof.

Input creation, output clearing/destruction, correctness/audit queries, and
NVML observations are outside timing; pool shutdown is inside. No build or
sanitizer overlaps the campaign. No power, clock, affinity, priority, firewall,
or security setting changes occur. No admin/firewall blocker is encountered.

## Performance results

Entries are primary outer-time percentage changes, negative faster. Each
cell lists R0/O0, R0/O1, R1/O0, R1/O1. All millisecond deltas, 41 phases,
and cross-label checks remain in the saved analysis.

### Legacy to on-demand: whole

| Case | Automatic CPU budget | Eight-thread budget |
|---|---|---|
| 1025_p2 | -4.537, -3.183, -1.094, -3.046 | +0.321, +0.030, -2.127, -3.089 |
| 1080p | -1.641, -2.775, -0.601, +0.256 | -1.600, -2.367, -3.806, -3.327 |
| 4k | +0.741, -0.891, +0.096, +0.193 | -0.729, +1.319, +3.607, +3.530 |
| 513_p0 | -7.273, -5.887, -6.656, -6.237 | -8.386, -4.959, -5.403, -5.911 |
| 513_p1 | +0.937, -5.884, -1.654, -1.220 | -2.366, -3.043, -3.695, +0.477 |
| 513_p2 | -6.138, -4.540, -0.586, -3.263 | -4.405, +0.114, +0.382, +1.107 |
| 65_p2 | -7.199, -1.498, -0.530, +1.696 | +1.864, -2.211, -1.706, +2.457 |
| flower_2000 | -4.045, +0.912, -5.224, -0.288 | -0.863, +2.285, -0.194, -2.149 |
| flower_3200x2160 | -0.855, -0.932, +3.314, +4.196 | -0.709, +3.140, +0.340, -1.800 |
| flower_500 | -3.921, -3.127, -4.093, -4.303 | -2.888, -5.833, -3.005, -0.803 |
| keong_3839x2159 | -2.079, +2.412, +2.188, -0.242 | -1.955, -2.014, -0.265, +4.143 |

### Legacy to on-demand: retained

| Case | Automatic CPU budget | Eight-thread budget |
|---|---|---|
| 1025_p2 | +0.783, +3.469, -3.516, -1.286 | -2.173, -2.221, +2.158, -2.382 |
| 513_p0 | -20.875, -22.594, -19.261, -21.728 | -18.826, -15.964, -18.528, -16.629 |
| 513_p2 | +2.259, -1.250, -2.025, -1.197 | -2.423, -2.203, -4.107, -6.341 |
| 65_p2 | -3.939, -1.474, -0.262, -3.424 | -2.222, -1.316, -1.564, -3.530 |
| flower_500 | -9.550, -7.943, -8.463, -9.178 | -6.411, -5.788, -6.718, -8.729 |

### Eager to on-demand: whole

| Case | Automatic CPU budget | Eight-thread budget |
|---|---|---|
| 513_p0 | +0.809, +0.131, +0.274, +8.994 | +0.283, +2.887, +1.508, -2.007 |
| 65_p2 | -4.416, -0.663, -3.482, -3.260 | -3.642, +0.614, -0.478, -0.097 |
| flower_500 | +1.380, +1.820, -1.419, -4.044 | +2.094, -3.191, -0.507, -1.639 |

### Eager to on-demand: retained

| Case | Automatic CPU budget | Eight-thread budget |
|---|---|---|
| 513_p0 | +2.483, +6.543, +5.208, +4.569 | +5.941, +5.834, +3.565, +3.643 |
| 65_p2 | -6.374, -9.703, -6.085, -1.924 | -0.578, -8.076, -3.584, -4.132 |
| flower_500 | +0.985, +4.812, -2.068, -3.834 | +0.784, +3.800, +4.463, -3.263 |

### Same-policy two-state controls

| Mode / policy / case | Automatic CPU budget | Eight-thread budget |
|---|---|---|
| whole / legacy / 65_p2 | -4.095, -1.654, -2.998, -0.290 | -0.562, -2.758, +0.305, +2.784 |
| whole / legacy / 513_p0 | -0.322, +1.387, +1.271, -1.262 | +0.265, +0.618, +0.283, -1.160 |
| whole / legacy / 513_p2 | +2.352, +5.210, +0.089, +0.195 | +3.429, +0.149, -2.492, -2.522 |
| whole / legacy / 1025_p2 | +0.215, -0.060, -7.753, -0.029 | +0.783, +5.687, -10.000, -0.622 |
| whole / legacy / flower_500 | +0.010, -1.902, +3.574, -0.712 | -1.968, +0.175, -0.007, +0.925 |
| whole / eager / 65_p2 | -2.467, -0.248, -0.447, +0.503 | +2.509, +0.178, +2.269, +1.724 |
| whole / eager / 513_p0 | +1.523, -1.685, +0.615, +0.448 | -2.018, -0.203, +1.360, -1.436 |
| whole / eager / 513_p2 | +2.314, -2.223, -0.062, -0.066 | +1.264, -3.258, +0.442, +1.722 |
| whole / eager / 1025_p2 | +0.655, +5.798, -1.014, -2.563 | -0.110, +1.685, +1.874, -3.031 |
| whole / eager / flower_500 | -0.119, -0.246, -0.154, +1.054 | -0.190, +0.878, -0.465, +3.546 |
| whole / lazy / 65_p2 | +0.564, +1.123, +1.584, +0.927 | +1.082, +0.606, +0.430, +0.945 |
| whole / lazy / 513_p0 | +2.156, +1.537, +0.237, -0.393 | +0.874, +3.341, -3.391, +2.765 |
| whole / lazy / 513_p2 | +0.861, +0.609, -3.591, -3.961 | -3.165, -4.681, +1.018, -2.706 |
| whole / lazy / 1025_p2 | -1.587, +2.115, +0.117, -1.626 | -0.588, +1.129, -0.080, -2.990 |
| whole / lazy / flower_500 | +1.741, -1.664, -1.321, +0.501 | +2.655, +1.224, +2.036, -1.120 |
| retained / legacy / 65_p2 | -2.492, -0.162, -0.633, -1.431 | +0.187, +1.097, -0.700, +0.010 |
| retained / legacy / 513_p0 | +2.695, -0.879, +5.735, +0.771 | -1.398, +1.164, +1.405, -1.608 |
| retained / legacy / 513_p2 | -0.971, +1.811, +7.767, +0.622 | +0.377, -0.574, +1.380, -2.033 |
| retained / legacy / 1025_p2 | -2.283, +6.143, +1.943, +1.783 | -3.903, +3.850, +2.761, +2.491 |
| retained / legacy / flower_500 | -2.531, -4.106, -3.169, +1.522 | +1.074, -1.548, -1.316, +2.986 |
| retained / eager / 65_p2 | +1.225, +1.567, -1.689, -2.430 | +1.867, -1.936, +0.103, +0.567 |
| retained / eager / 513_p0 | +1.123, +1.370, +3.408, -3.087 | -0.768, +0.173, -0.408, -2.842 |
| retained / eager / 513_p2 | -0.797, +4.768, +3.289, -3.763 | +3.835, +2.629, +0.083, -2.748 |
| retained / eager / 1025_p2 | -1.915, +2.895, -1.529, -2.225 | -4.548, +1.140, +5.089, +2.620 |
| retained / eager / flower_500 | -0.245, -1.045, -0.613, -0.679 | +1.556, -4.195, -0.125, +2.015 |
| retained / lazy / 65_p2 | -5.202, -1.309, -1.986, -1.264 | -2.295, -2.044, -3.344, +5.387 |
| retained / lazy / 513_p0 | +1.745, -0.211, -2.951, -1.275 | +2.442, -3.046, -0.149, +1.043 |
| retained / lazy / 513_p2 | -0.350, +0.430, +0.629, -3.638 | -5.900, +3.051, +0.741, -0.597 |
| retained / lazy / 1025_p2 | -3.098, -2.970, +2.232, -3.172 | -2.055, -3.641, -0.062, +0.272 |
| retained / lazy / flower_500 | -1.341, -0.225, -1.661, -2.597 | +1.155, +1.060, +4.360, -1.152 |

The control-table label `lazy` means the on-demand policy.

## Interpretation

Against legacy, on-demand improves 62/88 whole primary comparisons and
240/352 cross-label checks. The six real images account for 32/48 primary and
119/192 cross-label improvements, with primary changes from -5.833% to
+4.196%. Retained serialization improves in 36/40 primary and 125/160
cross-label comparisons. These counts use the percentage sign; they are not
pooled throughput estimates or statistical confidence claims.

Sparse 513 improves in all eight primary and all 32 cross-label comparisons
in each mode: 4.959-8.386% whole and 15.964-22.594% retained. Flower 500 also
improves in all eight primary comparisons in each mode, by 0.803-5.833%
whole and 5.788-9.550% retained; 29/32 whole and 32/32 retained cross-label
checks improve. The new sparse controls are materially smaller than these
legacy-to-on-demand changes, although not zero.

Tiny retained serialization improves versus legacy in all eight primary
trials, by 0.262-3.939%, with only 21/32 cross-label improvements. Against
eager it improves in all eight primary and all 32 cross-label checks, by
0.578-9.703%. Whole tiny results remain mixed versus legacy: 5/8 primary
and 19/32 cross-label improvements. Its on-demand-only whole controls are
positive in all eight trials; legacy-only automatic controls are negative in
all four. Thus the recorded tiny improvement is not a precise noise-free
estimate, even though the reduction in unnecessary workers is exact.

On-demand is not a universal successor to eager. Sparse retained outer time
regresses against eager in all eight primary trials by 2.483-6.543%, with
30/32 cross-label checks slower. Whole sparse comparisons against eager
regress in seven of eight primary trials, with cross-label checks split
15 faster/17 slower. Both policies eventually create the same seven/eight
workers there, but on-demand grows in two waves instead of one. Flower-500
eager comparisons are mixed in both modes despite fewer on-demand workers.
Do not select an image-specific policy threshold from these fixtures.

Large whole results remain unresolved. 4K regresses in six of eight primary
trials, even though its section-writing phase improves in all eight. Dense
513/eight regresses in three of four whole trials while retained improves in
all four. Controls can also be substantial: the whole legacy-only dense
1025/eight control reaches -10.000%, and retained legacy-only dense 513/
automatic reaches +7.767%. No control subtraction or causal exoneration is
performed. This evidence supports further work, not a general production
replacement or a claim that scheduling is now maxed out.

The following are descriptive medians over 64 measured samples per policy
from the retained/eight-thread eager-versus-on-demand comparison, not paired
speed estimates. Every measured encode records five root dispatches. Times are
milliseconds, shown eager to on-demand; worker and growth counts are exact.

| Case | Workers created | Growth waves | Creation span | Shutdown | AC-tokenization wall |
|---|---|---|---|---|---|
| Tiny 65 | 7 to 1 | 1 to 1 | 0.165 to 0.047 | 0.330 to 0.148 | 1.172 to 1.124 |
| Sparse 513 | 7 to 7 | 1 to 2 | 0.142 to 0.149 | 0.287 to 0.289 | 1.034 to 1.140 |
| Flower 500 | 7 to 3 | 1 to 2 | 0.147 to 0.090 | 0.353 to 0.211 | 2.091 to 2.165 |

These measurements are consistent with capacity and growth timing both
mattering, but do not isolate a pure thread-creation cost. Legacy joins are
not separately timed, creation can overlap useful work, and medians are not
additive. A future scheduling experiment could plan initial capacity from
known task-batch dimensions rather than merely choosing maximum or incremental
growth. It would still need complete-encode qualification.

The large-image warning also cannot be assigned solely to serializer time.
For 4K/eight in R1/O0, paired median outer time increases 9.821 ms while the
quantization-pipeline phase increases 10.075 ms and stream time increases
0.015 ms. R1/O1 increases outer time 9.495 ms, quantization-pipeline time
8.411 ms, and stream time 1.059 ms. These separately computed medians must
not be summed. The GPU modules are identical and the new pool starts in the
serializer, but this does not establish a cause for the phase variation.
There is no retained-4K timing or 4K same-policy control in this matrix.

Keep the pool implementations diagnostic. Prioritize isolating the actual
bit-writing and append work next, then revisit scheduling with its changed
task costs if warranted. No new compatibility layer, runtime default,
coefficient representation, fused-reduction policy, or GPU tile changes are
retained by S164.

## Separate source lead

Source inspection during the timing campaign identified work worth isolating
after this experiment. `WriteAnsTokenStream` in `src/codestream/ans.cpp`
reserves up to two reverse chunks per token, traverses them through individual
`WriteBits` calls into a temporary writer, then appends that temporary.
`BitWriter::Append` in `src/codestream/bit_writer.cpp` currently feeds each
full byte through `WriteBitsUnchecked`, including aligned destinations;
`AppendByteAligned` already uses bulk copies for its different section API.
These are source-level observations, not measured attribution or a retained
optimization. Separate probes should distinguish ANS recurrence, chunk
staging, bit emission, and append time before changing the writer.

## Evidence

Artifacts are under `build-cuda-ninja/profiles/s164-artifacts`. Preparation
verifies all 1,348 frozen S163 files and archives 340 unchanged production
sources. The three protected untracked Markdown files remain unread and
excluded. The timing campaign completed on 2026-09-09 from 19:58:14 to
20:18:19 UTC: 13,312 measured calls, 6,656 warmups, 416 oracle calls, and
368 untimed whole-encode captures for retained states. All 39,936 NVML
observations report a 40,000 mW enforced limit, not constant operating power,
clocks, or temperature.

The verifier recomputes every timing statistic, checks all 489 accepted job
journals and their logs, confirms qualification precedes timing and that no
build/sanitizer overlaps it, checks the exact serializer overlay against
unchanged production, and verifies source/library/native-module hashes.
There are no rejected S164 build or test journals. The artifact manifest and
source snapshot retain the protocols, binaries, helpers, inputs, and results.
Production is unchanged and the optimization goal remains open.
