# Immutable AC validation (S162)

## Outcome

Retain this as a targeted removal of redundant validation work, with
failure-atomic frame copy assignment. It is not a universal throughput win.
All 56 dense whole-workflow validation-phase comparisons are faster by
78.5–98.0%. Ordinary-density synthetic 513x519 whole encodes are faster in
all eight primary comparisons, by 1.1–5.7%; 28 of 32 cross-label sensitivity
comparisons are faster, not all 32.

Real-image whole-encode results split 24 faster/24 slower, with primary
differences ranging from -3.071% to +4.385%. There is no demonstrated general
real-image throughput gain. Retained dense 1025/automatic and flower
500/eight-thread serialization are slower in all four primary comparisons.
Those adverse results are retained below, not dismissed as noise. No kernel
or scheduling change is bundled with this CPU-side optimization.

## Change

S162 removes repeated dense edge-tail scans from `VarDctEncoderFrame::valid()`.
It does not change coefficient values, layouts, representation selection,
CUDA kernels, tile scheduling, entropy options, or defaults. The existing
private sparse-validation flag becomes AC-validation provenance for all six
native representations: dense and sparse signed 8-, 16-, and 32-bit storage.
There is no compatibility materialization, mutable query cache, runtime
experiment selector, or additional frame field.

Both direct producers establish the invariant before publishing the frame:

- Borrowed dense assembly zero-initializes owned storage and copies only
  checked active transform ranges. Group coverage must be complete.
- Owned dense assembly still exhaustively checks zero tails and, when
  requested, unwritten active coefficients before consuming the input owner.
- Sparse assembly still exhaustively validates masks, offsets, payloads,
  bounds, and coverage before consuming ownership.
- CPU coefficient coding zero-initializes dense storage and writes only
  checked active ranges. It sets provenance before its final structural
  `valid()` check and publishes only after that check succeeds.

Native queries require provenance. `valid()` retains geometry, strategy,
group coverage, storage shape, DC reconstruction/finite-value, raw-quant,
EPF, and sparse-range checks. It no longer reads dense coefficient tails that
the private, immutable owner cannot legitimately change after publication.
The proof assumes callers respect the const views and transferred ownership;
mutating through cast-away const or retained aliases is not supported.

Copy assignment now constructs a complete candidate and then moves it into
place. This is necessary for provenance integrity: allocation failure during
memberwise assignment could otherwise pair partially copied group metadata
with the old coefficient owner and a stale true validation bit. Copy
construction remains deep, moves remain nonthrowing, and copies may share
only the already-immutable coefficient-order population. Failed copy
assignment leaves the destination unchanged, at the cost of temporary storage
for a complete replacement frame. The ordinary fully resident handoff uses
move assignment.

## Qualification

The new `frame_validation` test exercises all 36 source/destination
representation pairs in both directions, using 36- and 60-block-square
frames. These sizes have the same group count and allocated capacity but
different active ranges. Across 72 cases it injects all 1,044 observed copy
allocation failures, checks unchanged destination ownership and coefficients,
source validity, exact successful-copy codestreams, deep copying, self-copy,
move construction/assignment, and copying/assigning invalid frames. It also
checks 144 allocation-free validations and rejects 81 malformed dense tails:
first, middle, and last tail positions in each partial group/channel at all
three widths. The normal run passes.

All 90 CTests pass, including the CPU producer, CUDA resident paths, compact
and sparse ownership, reconstruction, and codestream tests. Total CTest wall
time is 306.57 seconds. Independent ASAN and compact builds are in this
stage's artifact directory; predecessor build directories are not rebuilt.

Seven host-ASAN tests pass: frame validation, compact frame, compact allocation,
sparse frame, sparse allocation, CUDA sparse resident, and public codestream
workflow. The compact-width build passes frame validation, CUDA sparse
resident, and public codestream workflow. In each resident fixture invocation,
24 cases execute 672 comparisons and 384 control evaluations, including
150 sparse calls, 426 dense calls, and 24 representation transitions.

All eight CUDA sanitizer jobs pass: memcheck with full leak checking and
initcheck on wide/compact resident fixtures and on actual wide/compact
fully resident whole encodes. Every job reports zero errors; all four
memchecks report zero leaked bytes/allocations. CUDA device code is checked
by Compute Sanitizer, not host ASAN. The ASAN build instruments C/C++ with
clang-cl; nvcc uses the same MSVC host compiler as the prior stage. There are
no new racecheck/synccheck runs for this CPU-only change.

Native extraction checks eleven CUDA modules in each of four executables:
whole and retained production DLLs, compact DLL, and ASAN resident test.
All resource reports and instruction bodies match a qualified S160 module
after normalizing only source-derived anonymous-namespace IDs. All fourteen
production `.cu` sources are byte-identical to their pre-stage copies.

Link-map-derived CPU disassembly spans cover `valid()`, `GetNativeAcGroup`,
and `ValidateSimpleCodestreamFrame` in both whole and retained DLL pairs.
The baseline `valid()` spans call the out-of-line six-way tail visitor; both
candidate spans omit that call. The verifier checks this mapping. These are
spans up to the next higher code symbol, which can include padding or
compiler-generated code, not claims of exact function size or complete
call-graph equivalence. Objdump's nearest-export heading is not the function
identity; the map address is authoritative.

Two rejected job journals are retained. The CTest wrapper had an obsolete
expected summary string in its already-running Python invocation; CTest itself
returned zero with all 90 tests passing. A separate validation job checks the
original hashed log and all 90 passed rows. The first compact benchmark link
was missing `gjxl_pfm_io.lib`; a new job builds that dependency and links the
already-compiled bridge object successfully. A campaign preflight attempted
before that repair also stopped before starting qualification tests.

## Timing method

The frozen S160 production DLL is the whole-workflow baseline; the frozen
S161 current DLL, which contains that same production implementation plus
the capture callback, is the retained-serializer baseline. New DLLs link S162
production libraries with the unchanged frozen bridge sources. No private
frame or backend object crosses a DLL boundary. The public summary header is
unchanged. Baseline artifacts are run, not rebuilt or overwritten.

Whole measurements use fresh fully resident effort-7/no-final-score encodes
with a retained backend per implementation. Retained measurements deep-copy
one GPU-produced frame per state outside timing, then serialize it with the
resolved whole-workflow entropy/order options. Every output matches its own
and the cross-DLL oracle; real inputs additionally match frozen S153 bytes.
Retained logical AC FNV-1a fingerprints, counts, and nonzero counts must
match. The fingerprint is not a collision-free proof. GPU allocation and
submission counters must remain unchanged after capture.

The predeclared matrix has 152 whole jobs and 72 retained jobs. Whole cases
include six real images and all nine synthetic combinations of widths
65/513/1025 with low/ordinary/high-density patterns. Retained comparisons
cover dense 65, sparse 513, dense 513, dense 1025, and real flower 500. Each
mode includes baseline-versus-baseline and candidate-versus-candidate
two-state null controls for dense 513/1025. All use automatic and eight-thread
CPU budgets. Two campaign repeats each contain both DLL/backend creation
orders, shuffled across both modes; order is not confounded with repeat.
Synthetic patterns 0/1/2 respectively use amplitude/target pairs
0.01/1.2, 1/1.2, and 1/0.01, with the unchanged S160 source and seed.

Each job has four warmup and twelve measured rotating duplicate-ABBA rounds.
The primary statistic is the median of within-round duplicate-label mean
differences. All 41 phases use that statistic; four cross-label median
comparisons are sensitivity checks, not confidence intervals. Nested
worker-aggregate times are not additive wall times. Null controls are reported
without subtraction. Input creation, output destruction, correctness queries,
and NVML observations are outside timing. Builds and sanitizers do not overlap
the timing campaign. No power, clocks, affinity, priority, security, or
firewall settings are changed.

The campaign ran from 18:24:00 through 18:44:39 UTC on 2026-09-09:
10,752 measured calls, 3,584 warmups, 224 out-of-interval oracle calls, and
144 whole captures for retained measurements. All 14,336 warmup/measured
results match, with unchanged owner sizes. All 28,672 enforced-limit
observations are 40,000 mW; this does not imply fixed CPU/GPU clocks, actual
power, temperature, or system load.

## Performance results

All entries below are primary outer-time percentage changes; negative is
faster. Each cell lists R0/O0, R0/O1, R1/O0, R1/O1. The saved analysis also
contains every millisecond delta, every phase, and all cross-label checks.

### Fresh whole encodes

| Case | Automatic CPU budget | Eight-thread budget |
|---|---|---|
| 1025_p0 | +0.894, -0.575, +0.398, +1.242 | -0.616, +1.421, +0.605, -1.509 |
| 1025_p1 | -0.328, -2.102, +4.280, -1.376 | -1.430, -1.449, +0.277, +4.407 |
| 1025_p2 | -1.330, +1.039, +2.209, +0.739 | -0.095, -5.760, -0.582, -0.589 |
| 1080p | -0.798, +0.954, +2.685, -0.922 | +0.700, +0.793, -0.581, +1.515 |
| 4k | -3.071, -1.412, -1.311, +0.363 | +1.127, -2.739, -1.131, +0.171 |
| 513_p0 | +0.591, +0.769, -0.039, +1.208 | +0.687, +0.718, +2.655, -0.519 |
| 513_p1 | -5.696, -1.137, -2.882, -5.667 | -1.405, -2.240, -2.370, -3.338 |
| 513_p2 | -1.216, +1.541, +1.178, -2.022 | +0.922, +0.515, -0.681, -2.439 |
| 65_p0 | -3.874, -2.480, -2.479, +0.885 | -0.587, -1.577, -1.972, +0.468 |
| 65_p1 | -0.553, -1.394, +1.946, +0.663 | -0.824, -1.138, +2.005, -0.759 |
| 65_p2 | -7.937, -0.149, -1.962, -1.171 | -0.880, +1.299, +2.402, -3.249 |
| flower_2000 | +1.050, -1.930, +4.385, -0.255 | +0.879, -0.870, -0.104, +0.876 |
| flower_3200x2160 | +1.213, -1.525, +0.909, +1.003 | +0.213, -0.865, -1.747, -0.743 |
| flower_500 | -0.494, +0.331, +0.365, -1.245 | +0.309, -1.214, -2.166, +0.496 |
| keong_3839x2159 | +2.818, -1.396, +0.029, -0.959 | -0.283, -1.395, +0.901, +2.058 |

### Retained serialization

| Case | Automatic CPU budget | Eight-thread budget |
|---|---|---|
| 1025_p2 | +1.094, +5.331, +4.245, +1.858 | -0.043, +2.995, +4.087, -1.335 |
| 513_p0 | -1.097, +1.964, -0.679, -0.056 | +3.407, -1.493, -0.867, +1.221 |
| 513_p2 | +1.596, +0.231, -0.405, -0.257 | +1.066, +3.406, -0.567, +1.390 |
| 65_p2 | +2.620, -2.667, +1.543, -0.330 | -1.907, +1.879, -0.853, -1.131 |
| flower_500 | -0.301, -0.731, +2.325, -0.470 | +1.556, +4.184, +1.772, +2.482 |

### Same-build, two-state null controls

| Mode / build / case | Automatic CPU budget | Eight-thread budget |
|---|---|---|
| whole / baseline / 513_p2 | +4.238, +2.781, +4.899, -0.750 | -2.516, -0.528, +1.105, +5.313 |
| whole / baseline / 1025_p2 | -2.186, +2.538, -2.170, +1.514 | +0.352, -0.501, -0.914, -0.133 |
| whole / candidate / 513_p2 | +3.055, +2.173, +2.961, -1.139 | -5.418, +0.157, +0.518, -0.903 |
| whole / candidate / 1025_p2 | +0.122, +0.385, -5.854, +2.632 | +1.945, +0.287, -3.592, -1.698 |
| retained / baseline / 513_p2 | -4.691, -0.276, +2.022, +1.370 | -0.269, +1.717, -2.474, -0.094 |
| retained / baseline / 1025_p2 | +1.672, +1.201, +2.086, +2.838 | -3.087, -0.578, +3.015, +3.817 |
| retained / candidate / 513_p2 | -0.389, +2.544, -2.636, +2.072 | -1.379, +2.167, -2.621, +0.962 |
| retained / candidate / 1025_p2 | +1.130, +2.980, +0.249, -2.316 | +2.239, +0.522, -3.192, -2.987 |

## Interpretation and remaining work

The local mechanism is established by source ownership, fault tests, CPU
disassembly, and the measured validation phase. A wide dense `valid()` call
previously scanned 724,224 unused coefficient bytes at 65x71, 3,833,088 at
513x519, and 6,880,512 at 1025x1031. S162 avoids those reads; the initial owned
assembly scan remains. Storage sizes and coefficient contents are unchanged.

For dense 513/pattern 2/eight threads, whole validation saves
0.639–0.715 ms, or 91.0–91.6%, but whole outer changes are
+0.922%, +0.515%, -0.681%, and -2.439%. This does not establish a consistent
fix for S160's original high-density whole-workflow regression, and S162
does not directly remeasure S152. Across all whole comparisons, 68/120 primary
and 273/480 cross-label comparisons are faster; across retained comparisons,
18/40 primary and 73/160 cross-label comparisons are faster.

All 24 dense retained validation comparisons are faster, but retained dense
1025/automatic outer time is slower by 1.094–5.331% in all four primary
comparisons (11/16 cross-label checks slower). Flower 500/eight-thread retained
serialization is slower by 1.556–4.184% in all four primary comparisons
(14/16 cross-label checks slower). Baseline-only retained 1025/automatic null
controls are also positive in all four comparisons; this shows a same-build
state/label effect, not proof that it explains the candidate's slowdown.
No null subtraction or causal dismissal is used.

The retention decision is a scoped tradeoff for fresh fully resident work:
eliminate provably redundant reads, retain the consistent ordinary-density
513 benefit, and preserve failure-atomic ownership. It is not a claim of
Pareto improvement. Repeated retained-serializer costs remain open. The next
useful investigation is the remaining AC tokenization/section-writing work
and scheduling in a controlled diagnostic, with these adverse cases and both
builds' null controls preserved. The present data do not isolate cache state,
DLL layout, compiler decisions, allocation behavior, or worker scheduling as
the cause of those costs. The broader optimization goal remains active.

## Evidence

Stage artifacts and source snapshots are under
`build-cuda-ninja/profiles/s162-artifacts`. Preparation verified all 618 frozen
S161 files and archived all 339 unchanged pre-stage production sources.
The three protected untracked Markdown files remain unread and excluded.

There are 252 job journals: 250 accepted and the two explicitly explained
rejections. The full verifier recomputes all statistics, checks source and
artifact hashes, validates test/sanitizer markers, checks timing order and
non-overlap, and verifies the predecessor files again. The build-input index
contains 652 records for 340 unique source files because relative and
absolute paths entered the initial set before resolution; duplicate records
are identical and checked. This is an index duplication, not extra source
versions. Five production/test/build files change, plus this report.
