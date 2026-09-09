# Same-object sparse policy and serializer scheduling (S159)

## Scope

S158's separate-context comparison exposed several-percent timing differences
between identical-mode controls. S159 removes prepared-object identity from the
comparison and qualifies a small-input eligibility gate. This is an experimental
source overlay, not a production default change or a compatibility layer.
Production still has the S152 implementation (`2118ccb`).

The overlay is `build-cuda-ninja/profiles/s159-artifacts/source`. It starts from
the frozen S158 source; all 2,683 S158 artifacts were verified before preparing
it and are checked again by the final verifier. No previous-stage source,
executable, helper, or report is rebuilt or modified.

Qualification and the 72-job timing campaign are complete. Sparse-input gains
survive same-object comparisons under both tested thread budgets. Dense/null
variation remains; neither a thread-budget change nor an independent fusion
speedup is supported. The next implementation step is a diagnostic-free native
sparse candidate compared directly with production S152, not a claim that the
current diagnostic overlay has already been promoted.

## Implementation

Selected-fused modes 104/105 extend S158's 40/41 by checking sparse size and
capacity eligibility **before** moving population readback into the earlier
batch. They require at least `3 * 256 * 256` active coefficients, a count that
fits `uint32_t`, and a packer grid that fits the backend's maximum X dimension.
Small or unsupported inputs leave population readback in the ordinary final
batch. Wide small inputs keep the original up-front host allocation behavior.
The existing density threshold remains at most one nonzero per eight slots.

A diagnostic-only setter switches a prepared object's AC policy between calls
without changing its compact-storage bit. It rejects busy, invalid, fault-armed,
unknown-mode, and compact-bit-changing requests. There is no public API or
production runtime selector proposal here. The actual early-count arithmetic,
native typed/sparse consumers, GPU packer, and fused readback are inherited from
S156–S158. No CUDA translation unit changes.

## Qualification

The transition fixture uses 24 cases: five shapes from 1x1 through 513x519 and a
mixed-transform frame, each with four amplitudes. Across 48 prepared contexts,
the sequence `104, 0, 40, 16, 8, 6, 104, 0, 104` (plus the fixed compact bit)
alternates selected, dense, movement-control, unfused, and forced-sparse paths.
Every step exercises scored policy evaluation, unscored policy evaluation, and
ordinary final evaluation, followed by an evaluation without final output.

It makes 1,728 full comparisons against dense-wide references, checks native
coefficients and codestream bytes, independently counts nonzeros, and checks
decision resets, sparse ownership, submissions, copy batches, lazy headers, and
invalid setter requests. Its counters are 216 small-input skips, 324 sparse
calls, and 146 setter guards. These are fixture counters, not timing samples.

The full-workflow fixture covers 52 flat, smooth, noisy, and finite-huge cases,
efforts 3/7, score on/off, and target-byte/error modes: 18 policy/compact modes
produce 936 full encodes. The CTest suite contains 96 tests.

All 96 tests, both host-ASAN fixtures, and all four CUDA checks passed. CUDA
reported zero errors and, for memcheck, zero leaked bytes/allocations. All 85
recorded job journals are accepted; there were no rejected build/test/timing
journals. The initial and V2 source build-input snapshots are both retained;
V2 adds the workflow test and its CMake registration without changing library
sources. The already-built first timing executable is not overwritten.

Both new fixtures are built with host AddressSanitizer on all C/C++ translation
units. CUDA translation units and their launch wrappers remain compiled by
nvcc, not ASAN-instrumented. Both fixtures also run under all-kernel CUDA
memcheck (including full leak checking) and initcheck, four CUDA jobs total.
The portable S157 Compute Sanitizer 2025.2.1 is reused without installation,
launch limits, suppressions, or blocking-launch flags. This stage does not
repeat S158's race/synchronization checks because GPU code is unchanged; those
checks remain frozen predecessor evidence, not additional S159 runs.

Native extraction covers the two timing executables, two ordinary fixtures,
and two ASAN fixtures. Each contains the same 11 raw CUDA modules. All 14 CUDA
source files are unchanged. Every function's instructions and every module's
resources match S158 after normalizing only the two source-derived eight-hex
anonymous-namespace IDs; no instruction or encoded-word normalization is used.

## Predeclared timing method

One benchmark prepared object serves **both** policy labels. A separate
dense-wide oracle is evaluated once and destroyed before measurement. Four
warmup and eight measured rotating duplicate-ABBA rounds use fresh caller
outputs. The timed interval includes unscored one-update resident evaluation
and codestream serialization. Policy switching, caller-output allocation,
power queries, correctness comparison, and output destruction are outside it.
The oracle codestream, quant fields, and score bits must match on every call.

The main policy comparisons use the existing explicit eight-thread encode scope.
Selected-vs-dense and null controls at 4K also use the default automatic budget. The
separate budget control keeps the same dense AC mode and prepared object while
switching automatic thread budget zero versus explicit eight. The participant
tracker counts active instrumented participant scopes, which can include
blocked parents or nested scopes on the same thread under automatic scheduling.
It is not an OS-thread census and does not measure runnable threads, CPU
utilization, or GPU occupancy.

The 36-spec matrix includes selected-vs-dense (65x67, 513x519, padded 4K; low and
dense inputs), dense population movement, fusion-only, small-input gate,
identical-policy null controls, and automatic-versus-eight budget controls,
each in wide and compact form. Two shuffled/reversed campaigns make 72 jobs:
2,304 measured calls, 1,152 warmups, and 72 out-of-interval oracle evaluations.
The low-amplitude fixture uses 0.01; the dense fixture uses 16. These DCT8
fixtures are not the real-image whole-workflow corpus from S158.

The primary statistic is the median of eight within-round differences between
duplicate-label means. Percentage differences use those same round means.
Four comparisons of label medians are also retained as a sensitivity check,
not confidence intervals. Negative changes mean faster. The benchmark records
the enforced GPU power limit before/after every timed interval; this does not
establish constant clocks, actual power, temperature, or system load.
Total and phase statistics are independently calculated medians; the phase
medians need not add to the total median, and median percentages need not have
the same sign as median millisecond differences close to zero.

## Results and decision

The campaign ran from 16:07:50 through 16:20:50 UTC on 2026-09-09. All 3,456
warmup/measured calls matched their 72 dense-wide oracle evaluations. All 6,912
recorded enforced-limit observations were 40,000 mW. No build or sanitizer job
overlapped the campaign.

### Sparse input: reproducible benefit, mostly in evaluation

Each cell is the two repeats' primary **evaluation-plus-serialization** change
against the matching dense mode (percent; negative is faster):

| Low-amplitude fixture / budget | Wide | Compact |
| --- | ---: | ---: |
| 513x519 / eight | -23.854, -23.889 | -14.823, -16.196 |
| 3839x2159 / eight | -19.207, -17.014 | -9.483, -4.497 |
| 3839x2159 / automatic | -17.919, -20.029 | -8.658, -13.105 |

All 12 primary changes favor sparse storage. Of 48 cross-label comparisons,
46 also favor it; two compact 4K/eight comparisons in the second repeat reverse
slightly (+0.075%, +0.891%). All evaluation-only cross-label comparisons favor
sparse storage. Wide 4K evaluation saves 22.77–24.65 ms across the four runs;
compact saves 5.84–8.82 ms. These unusually sparse fixtures contain only seven
nonzeros at 513x519 and two at 4K. They do not represent all real images or the
entire density range accepted by the selector. S158's real-image workflow
results remain separate predecessor evidence.

At 4K, native wide ownership falls from 106,168,320 to 4,665,608 bytes; compact
ownership falls from 26,542,080 to 4,665,602 bytes. The additional retained GPU
header allocation is 4,665,604 bytes. Selected sparse output uses one pack
attempt and one readback batch. Header storage remains allocated on the same
object when it switches back to dense; the logs deliberately expose that
retained state rather than assigning it exclusively to the candidate label.

### Dense and identical-policy controls remain variable

These are dense 4K total-percent changes, again listing both repeats:

| Contrast / budget | Wide | Compact |
| --- | ---: | ---: |
| Selected vs dense / eight | +0.977, -8.901 | +1.068, -6.157 |
| Selected vs dense / automatic | +2.408, -6.855 | -0.585, +1.274 |
| Identical dense vs dense / eight | -1.489, -1.374 | +5.124, +2.488 |
| Identical dense vs dense / automatic | +0.749, -2.995 | +1.691, -3.353 |

The consistently slower compact 4K retained total from S158 is not reproduced
across both repeats here. But identical policies still show several-percent
differences on the same object, so this does **not** establish that the early
population movement is free, identify the source of the variation, or explain
it as separate-context placement or RDP. The compact automatic evaluation-only
primary changes are +5.045% and +3.825%; the second run also has all four
evaluation cross-label comparisons slower. This phase-level adverse result is
retained, not hidden by total-time medians.

At 513x519, dense selected evaluation is slower in all four primary comparisons
(+0.919% to +2.450%, approximately 0.041–0.112 ms). Total changes are mixed for
wide (+1.093%, -1.097%) and slightly positive for compact (+0.502%, +0.658%).
No new sparse pack, readback batch, or header allocation occurs on these dense
inputs. At 4K the selected exact count is 21,924,397 nonzeros out of 24,883,200,
and both labels retain identical dense widths and owner sizes. The pure
population-movement control also changes sign at 4K, preventing a simple
end-to-end causal attribution.

### Fusion, small-input gate, and scheduling

Fusion-only changes at 4K are +1.196%/+1.953% wide and -4.956%/+1.302% compact;
513x519 is -0.278%/-1.833% wide and +0.672%/+0.075% compact. It removes one
readback batch for nonempty sparse payloads, but has no consistent independent
end-to-end win across these sizes and widths.

The new small-input gate is functionally verified to avoid early population
readiness and sparse packing on ineligible inputs. Its isolated 65x67 total
changes versus old selected-fused mode are +1.714%/-0.008% wide and
+11.085%/+0.937% compact. This run does not establish a gate speedup. Small
selected-vs-dense and null controls are also variable. The gate is an
eligibility simplification, not a measured standalone optimization claim.

Source inspection shows that automatic `RunParallelSections` permits nested
parallel sections, whereas an explicit budget inhibits nested parallel work.
However, **no timing sample exceeds eight active tracked participant scopes**,
including automatic-budget samples. This experiment therefore does not support
the proposed excess-nesting explanation for the retained timing variation.
It does not prove that other inputs cannot trigger more nesting.

Switching automatic to eight in the same dense mode gives 4K total changes of
+0.066%/-0.117% wide and +6.165%/+0.074% compact. The 513x519 changes are
-2.088%/-0.085% wide and +0.251%/-0.631% compact. There is no basis here for
changing production thread-budget defaults. Sparse gains survive both settings;
the gain is not dependent on the proposed scheduling explanation.

The diagnostic policy setter and fault instrumentation must not become
production compatibility/runtime layers. Native sparse consumption is still
the useful candidate. Its next direct comparison must include dense controls
and ordinary fully resident workflows against the actual S152 source, rather
than treating the overlay's dense label as a complete production baseline.

## Reproduction and limits

`s159_prepare.py`, `s159_build.py`, `s159_qualify.py`, `s159_native.py`, and
`s159_timing.py` record source pins, build inputs, exclusive job logs/journals,
qualification, native comparisons, and the pre-execution timing protocol.
`analyze_s159.py` derives paired results from raw logs. `verify_s159.py` checks
source pins, predecessor hashes, qualification markers, native evidence,
derived results, timing order/non-overlap, and unchanged production sources.
`freeze_s159.py` archives sources and helpers and inventories local artifacts;
`verify_s159.py --frozen` checks that inventory again.

No firewall/admin blocker was observed. No clock, power, thermal, affinity,
priority, security, or firewall setting was changed; no remote push was made.
RDP is not assumed to explain any result. The three protected untracked
Markdown files are excluded from all source inventories and commits.
