# Retained-frame serializer diagnosis (S161)

## Outcome

S160's dense 513x519/eight-thread whole-workflow regression is not reproduced
consistently when serializing retained GPU-produced frames. More importantly,
the validation phase changes direction: it is faster for the current native
implementation in every retained-frame primary comparison. This does not
exonerate the fresh whole-workflow cost or establish its cause. It prevents
treating a universal validation/dispatch slowdown as an established diagnosis.

Production remains S160 (`f2a77c9`). No library, CUDA kernel, scheduling policy,
or default is changed in this stage. The committed deliverable is this report;
the source-pinned harness and evidence remain under
`build-cuda-ninja/profiles/s161-artifacts`. All 2,878 S160 artifacts were checked
before preparation and again by the verifier, without modifying or rebuilding
predecessor artifacts. The three protected untracked Markdown files remain
unread and excluded.

## Method and coverage

Two new diagnostic DLLs link the frozen S152 baseline libraries and newly
archived copies of the already-qualified S160 production libraries. Each uses
its matching frame headers and owns all of its private objects. No private
frame crosses the DLL boundary. A diagnostic copy of `workflow.cpp` has only a
capture declaration and one callback added at the serializer handoff. The
callback deep-copies that implementation's GPU-produced frame during one
ordinary fully resident effort-7/no-final-score encode, outside measurement.

Each timed call serializes the retained immutable frame with the exact entropy
and coefficient-order options resolved by that full encode. The explicit
encode thread scope is restored for each serialization. Fresh result storage
is used; destruction, comparisons, audit queries, and NVML observations are
outside timing. Every output must match both its own full-encode codestream
and the cross-DLL baseline oracle. The public full-encode summaries also match.
The harness checks matching FNV-1a fingerprints over all logical active AC
coefficients, coefficient counts, and nonzero counts; a fingerprint is not a
collision-free proof. Real-image oracles additionally match frozen S153 bytes.
Backend allocation/submission counters must not change after capture. The
timed serializer itself makes no GPU calls.

Fourteen initial comparisons cover flower 500, padded 4K, 65x71 dense,
513x519 low/ordinary/high-density patterns, and 1025x1031 dense, under automatic
and eight-thread budgets. Both DLLs also run together under all-kernel CUDA
memcheck with full leak checking and initcheck on flower 500. Both pass with
zero errors, and memcheck reports zero leaked bytes/allocations. Their actual
GPU work occurs during frame capture; the later measured operation is CPU
serialization. These 16 jobs execute 80 serializer calls and 32 whole captures.
There are no new host-ASAN or CTest runs: production is unchanged and S160's
89 CTests/five ASAN fixtures remain predecessor evidence, not S161 runs.

Raw extraction shows the baseline DLL has exactly S160's baseline ten CUDA
modules, and the current DLL exactly its production eleven modules. No GPU
source is recompiled. All 91 recorded S161 jobs are accepted: three builds,
16 qualification jobs, and 72 timing jobs. There are no rejected journals.

## Predeclared timing matrix

Five comparisons use both CPU budgets: dense 65x71, sparse 513x519, dense
513x519, dense 1025x1031, and real flower 500. The synthetic source/seed and
targets match S160: sparse amplitude 0.01/target 1.2, dense amplitude 1/target
0.01. Dense 513/1025 also have both baseline-versus-baseline and
current-versus-current two-frame null controls under each budget, 18
specifications total.

Each of two repeats includes both DLL/backend creation orders, shuffled within
the repeat. Unlike S160, order is crossed with repeat, not tied to it. The
72 jobs each use four warmup and twelve measured rotating duplicate-ABBA rounds:
3,456 measured serializer calls, 1,152 warmups, 72 serializer oracles, and
144 out-of-interval whole-encode captures. All 4,608 warmup/measured results
match their oracles. All 9,216 enforced-limit observations are 40,000 mW; this
does not imply fixed CPU/GPU clocks, actual power, temperature, or system load.

The primary statistic is the median of twelve within-round differences between
duplicate-label means, with percentages computed from those same means. All
41 saved phases use that statistic; four cross-label medians are retained as
sensitivity checks, not confidence intervals. Nested aggregate worker times
are not additive wall time. No build or sanitizer overlaps timing. The campaign
runs from 17:39:18 through 17:43:06 UTC on 2026-09-09.

## Results

Cells below contain four primary percent changes in this order:
repeat 0/baseline-first, repeat 0/current-first, repeat 1/baseline-first,
repeat 1/current-first. Negative is faster. These are serializer-only results,
not new whole-encode speedups.

| Input | Automatic budget | Eight-thread budget |
| --- | ---: | ---: |
| Dense 65x71 | +1.007, +1.892, -4.432, -0.193 | -1.691, +1.932, -3.136, -0.562 |
| Sparse 513x519 | -13.037, -14.329, -13.116, -15.264 | -14.589, -15.466, -13.569, -10.700 |
| Dense 513x519 | -1.783, +3.820, -3.827, -0.603 | -3.001, -5.416, -0.413, +2.699 |
| Dense 1025x1031 | +2.098, -2.401, +2.503, -0.795 | -0.443, -3.238, +1.330, +1.003 |
| Flower 500x500 | -2.816, +0.091, +0.591, -0.617 | +0.033, +1.973, -0.350, +4.971 |

S160's dense 513/eight-thread whole encodes were +4.226%/+5.945%, with all
eight cross-label comparisons slower. Here three of four primary serializer
comparisons favor the current build and one is slower. Dense 1025/eight-thread
also changes sign across repeats. Flower 500 serialization is not consistently
improved, despite S160's consistently improved whole encodes. Sparse 513
serialization improves in all eight primary comparisons, by 10.700–15.466%.

### Validation reverses direction

An exploratory breakdown of all saved S160 phases, using its unchanged original
eight-round statistic, shows dense 513/eight-thread validation was
+82.468%/+104.271% (0.336/0.387 ms slower). Those complete 72-job/42-metric
derivations are preserved separately and do not replace S160's frozen analysis.

In S161 the same logical dense 513/eight-thread validation is
-37.709%, -35.686%, -38.546%, and -37.728%, saving 0.245–0.284 ms.
Across both budgets, dense 1025 validation improves 31.950–44.696%.
All 40 non-null retained-frame validation primary comparisons favor current
code, including small, sparse, and real inputs. Thus the earlier positive
validation phase cannot be treated as a context-independent cost of the
expanded native dispatch.

### Null controls remain substantial

| Null / input | Automatic budget | Eight-thread budget |
| --- | ---: | ---: |
| Baseline / dense 513 | +7.455, +2.716, -3.118, -1.674 | +4.411, -5.470, +3.380, -0.647 |
| Current / dense 513 | +0.773, +3.219, -1.554, +1.872 | -1.544, -5.693, -0.200, -0.377 |
| Baseline / dense 1025 | -1.597, -2.419, +2.158, +5.204 | +1.082, -1.109, -1.625, +1.296 |
| Current / dense 1025 | -1.741, -0.008, -1.873, -0.246 | +2.680, -1.808, +1.647, -0.442 |

Identical implementations still differ by several percent. These controls are
not subtracted from the implementation contrast, do not explain the variation,
and do not invalidate the original fresh-workflow regression. Deep-copy frame
allocation, retention, repeated cache use, absent interleaved GPU work, and
different DLL layout all distinguish this harness from the whole workflow.
The experiment does not isolate any one of those factors as the cause.

## Source/code evidence and next action

The CPU audit saves link-map-bounded instruction spans for `frame.valid()`,
`GetNativeAcGroup`, and `ValidateSimpleCodestreamFrame` from the two frozen S160
whole-encode DLLs and the two new serializer DLLs, twelve spans. These bounds
can include alignment padding; they are not exact function-size claims.
Disassembly export labels alone are not function identities in these stripped
DLLs: the link-map addresses are authoritative. This is an inspectable audit,
not a proof that the entire CPU call graph is instruction-identical.

Source inspection finds concrete redundant work: `VarDctEncoderFrame::valid()`
checks every unused dense group tail for zero on every query, even though the
published owner is private and read-only. The dense 513 fixture has 811,200
active coefficients but 1,769,472 dense slots, so this revisits 3,833,088 bytes
of padding per full tail scan. Sparse owners already validate payloads before
publication and cache that fact. The corresponding dense one-time validation
is a candidate to test, not an implemented change or established explanation
for S160's regression.

The next experiment should remove only repeated dense-tail validation while
preserving assembly rejection, private ownership, moved-from/copy behavior,
and failure atomicity. It must qualify both fresh whole encodes and retained
frames, including the adverse dense/eight-thread fixtures and null controls.
Retained-frame wins alone must not justify a production change. There is no
basis here for changing tile scheduling, CPU budgets, or compact-width defaults.

## Reproduction and limits

`s161_prepare.py` checks predecessor/source pins and archives link libraries;
`s161_build.py` records inputs before building the capture bridges and harness.
`s161_qualify.py` checks native CUDA identity, direct outputs, and sanitizer
coverage. `s161_timing.py` pins its source/binaries/analyzer and matrix before
execution. `s161_inspect_s160.py` preserves the exploratory predecessor phase
breakdown; `analyze_s161.py` derives the new paired results.
`s161_cpu_native.py` extracts the post-campaign CPU spans.
`verify_s161.py` checks source/provenance, capture-only workflow edits, every
journal, derived statistics, native evidence, and timing non-overlap.
`freeze_s161.py` snapshots sources/helpers and inventories artifacts;
`verify_s161.py --frozen` rechecks them.

No firewall/admin blocker was observed. No power, clock, thermal, affinity,
priority, security, or firewall setting was changed, and no remote push was
made. RDP is not assumed to explain any result. This is progress in diagnosis,
not a claim that CUDA VarDCT performance has been maxed out.
