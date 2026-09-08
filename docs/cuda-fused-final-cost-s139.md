# Fusing CUDA AC final-cost composition (S139)

Disposition: unpromoted. Keep the [S138](cuda-fused-ac-integration-s138.md)
production evaluator and allocation layout. This experiment proves that final
cost can consume shared channel reductions with exact tested output, but does
not establish a dependable complete-encode improvement. Whole-call primary
results favor the candidate in 8/16 comparisons, with three of four 4K results
unfavorable. The encoder is not established to be maxed out.

## Hypothesis and isolated implementation

S138 places all three channels of each AC candidate in one CUDA block. S139
retains its 192-thread schedule, transform pass order, coefficient layout,
quantization, inverse transform and halving reductions. The diagnostic kernel
writes each channel's rate and loss into small shared arrays. After one final
block barrier, contiguous lanes compose the candidates' costs using the exact
existing channel order, weighting, explicit FMAs, normalization and `powf`.
The seven separate final-cost launches are bypassed for candidate labels.
Quant-norm preparation remains separate and reuses the costs range as before.

The local-output structs and final-cost body are generated from the current
production source, with narrowly checked substitutions. Final-cost lanes use
global candidate indices for descriptors/norms and block-local indices for
channel reductions. Inactive tail candidates reach every barrier but do not
load descriptors or consume uninitialized local reductions. No fast-math
relaxation is introduced.

Only ignored diagnostic sources contain the selector. Baseline labels use the
normal S138 fused evaluator plus final-cost kernel; candidate labels use the
new combined kernel. Both have the same host validator, prepared owner,
resident pipeline and allocations. The candidate leaves global loss/rate
scratch untouched, but that scratch is still allocated: this experiment does
not claim a device-memory capacity reduction. Removing it would require a
separately qualified contract/owner change.

The frozen S138 traces put seven final-cost kernels at approximately 0.021 ms
for Flower 500, 0.049–0.051 ms for HD, 0.158–0.167 ms for padded 4K and
0.083–0.087 ms for Flower 2000. At 4K the enclosing fused evaluator itself took
15.85–22.16 ms. These are short instrumented GPU durations, not whole-call
latency or a bound on all launch/traffic effects, but they show how little
standalone final-cost work is available to save.

## Focused correctness and native code

Starting revision `1506809`, branch `feat/cuda`, September 8, 2026. Windows,
RTX 3060 Laptop (`sm_86`), CUDA 11.8, MSVC 14.37, Release; scoped host ASAN uses
clang-cl. Normal production sources are unchanged. Diagnostic CUDA objects and
callers are freshly built against frozen S138 libraries; scoped ASAN also
reuses the hash-checked S138 host validator/search, pipeline and resident
objects because those sources have not changed. This is not a new full CMake
build or CTest run, nor a Metal, Linux, independent decoder or second-GPU claim.

The first focused grid has 144 cases and 29,876 descriptors. The extended grid
adds both descriptor and device-derived quant norms to both host and signed
device CfL: 288 cases and 59,752 descriptors per execution. It covers fifteen
geometries including thin images, partial tiles and axes through 257 blocks.
Each case obtains final-cost bits from the normal public S138 batch, resets a
guarded strided-input arena, and requires three candidate evaluations to match
those bits exactly while preserving every byte outside final costs, including
the unused global loss/rate scratch. Quant fields and all input guards are
preserved. Baseline costs must also be finite and nonnegative.

The extended grid passes Release, scoped ASAN and all four CUDA sanitizers.
Memory/init/synchronization checking reports zero errors; race checking reports
zero hazards, and memcheck reports zero leaked bytes. An exhaustive extension
covers every block width/height pair from 1 through 19 in all four CfL/norm
combinations: 8,804 cases and 539,236 descriptors in each of Release and ASAN.
Together these grids execute 19,480 baseline and 58,440 candidate batches.
Repeated evaluations are not independent images. The broader AC contract test
also passes both labels in Release and ASAN, including invalid inputs and
foreign/overlapping range rejection; each execution observes 329 submissions
per label family.

The native audit finds 228 kernel bodies: 220 S138 bodies are exactly unchanged,
one source-unchanged quant-norm body matches the historical S134 variant, and
seven new combined bodies have zero stack/local storage and zero static
local-memory loads/stores. Both timing labels share the same quant-norm body.
Seventeen GPU executables, including traces and ASAN callers, have identical
ten-module sets; eight modules match the frozen S138 production encoder.

| Shape | Registers | Shared bytes |
| --- | ---: | ---: |
| 8×8 | 40 | 9,248 |
| 16×8 | 39 | 8,720 |
| 8×16 | 56 | 9,104 |
| 16×16 | 44 | 17,296 |
| 32×16 | 55 | 16,848 |
| 16×32 | 88 | 17,232 |
| 32×32 | 73 | 33,616 |

## Balanced complete-encode timing

Inputs match S133/S138: Flower 500, padded HD from 1919×1079, padded 4K from
3839×2159, and Flower 2000 (fourfold nearest-neighbor replication, not a native
2000-square photograph). Distance is 1.2, effort 7, fully resident, with both
wide and opt-in compact coefficients; no coefficient-width default changes.

Labels 0/2 duplicate the baseline, and 1/3 duplicate the candidate. Four-round
Williams blocks balance positions and ordered predecessors. Each process has
one reference encode, four warm rounds and sixteen measured rounds. Two passes
reverse case/width order: sixteen timed processes and 1,024 measured encodes.
Every encode checks frozen bytes, summary, coefficient width/storage and seven
expected AC-batch selections. No other recorded build or diagnostic job
overlaps a measured process.

The statistic below is the median across rounds of mean candidate-label time
minus mean baseline-label time, not a difference of independently computed
medians. Negative is faster. Whole-call time includes preparation, quantization
and serialization; nested savings must not be added together.

| Case / width | Whole-call delta r0 / r1 (ms) | Quantization delta r0 / r1 (ms) |
| --- | ---: | ---: |
| Flower 500 / wide | −0.257 / −0.012 | −0.086 / −0.155 |
| Flower 500 / compact | −0.105 / −0.375 | −0.042 / −0.090 |
| HD / wide | +0.069 / −2.478 | −0.126 / −0.635 |
| HD / compact | +0.350 / +1.317 | +0.327 / +1.078 |
| 4K / wide | +0.147 / +3.315 | +2.846 / +2.611 |
| 4K / compact | −1.264 / +0.809 | −1.921 / −0.872 |
| Flower 2000 / wide | −0.355 / +6.296 | −1.654 / +0.495 |
| Flower 2000 / compact | −1.150 / +2.486 | +0.597 / +1.324 |

Whole-call comparisons favor fusion in 8/16 primary results and 31/64
cross-label pairs; quantization is 9/16 and 26/64. Serialization is 9/16 and
38/64 despite unchanged serialization code. Equivalent baseline labels differ
by −9.654 ms in the first wide-4K process, much larger than its +0.147 ms
primary effect. This does not support a precise universal regression estimate,
a universal speedup, or a post-hoc small-image selector.

## Broader qualification, traces and power-state caveat

The broader candidate-path campaign passes eighty quality, rate-search/reuse,
stress and one-/two-request batch configurations in both coefficient widths,
plus fourteen writer rejections matching frozen errors. Sixteen scoped ASAN
configurations, four production memchecks and four production initchecks also
pass. These campaigns provide 1,296 frozen-oracle checks; controlled preflight
and timing add 1,376, and traces add forty: 2,712 checks in total, including
262 scoped host-ASAN checks. All twelve CUDA sanitizer jobs qualify with zero
errors/hazards, and every memcheck reports zero leaked bytes.

Eight complete-process Nsight Systems traces contain 32 labeled windows plus
eight reference encodes. Every CUDA call succeeds. Each candidate removes
seven final-cost launches, with the same seven quant-norm launches, all other
kernel-name/count multisets unchanged, and identical memcpy counts/payloads.
Wide/compact launch counts fall from 294/295 to 287/288 at 4K, 320/321 to
313/314 at HD, and 359/360 to 352/353 for both Flower sizes.

However, all 2,752 timed/preflight power endpoints report 40,000 mW, while the
eighty subsequent trace endpoints report 72,150–73,977 mW, sometimes changing
inside an encode. The original fixed-40-W trace audit correctly fails. A
separate structural analysis preserves the power records and explicitly marks
the traces as not controlled latency evidence. A read-only post-trace
`nvidia-smi` query also reports a 71.05 W current/requested limit. The origin of
this power-state change is not established. No power, clock, priority, affinity
or firewall policy was changed by this experiment; no elevation/firewall
blocker was encountered.

For completeness, raw traced summed evaluator-plus-final-cost duration versus
the combined kernel is listed below. Values average the two labels in each
family. They must not be compared directly to the prior 40-W traces or treated
as a controlled causal latency result.

| Case / width | Raw higher-power trace, baseline → candidate (ms) |
| --- | ---: |
| Flower 500 / wide | 0.501 → 0.500 |
| Flower 500 / compact | 0.501 → 0.499 |
| HD / wide | 3.419 → 3.584 |
| HD / compact | 3.575 → 3.661 |
| 4K / wide | 12.167 → 12.358 |
| 4K / compact | 11.932 → 12.672 |
| Flower 2000 / wide | 6.380 → 6.768 |
| Flower 2000 / compact | 6.410 → 6.598 |

Even these descriptive totals favor the candidate only for Flower 500. The
per-shape record associates each baseline final-cost launch with its preceding
evaluator, then compares that sum with the corresponding combined shape.
The 16×16 and 32×32 stages account for the largest baseline 4K stage totals
in these captures. Their timing deltas inherit the same power-state caveat;
they do not establish register pressure or occupancy as the root cause.

## Disposition and next action

Keep S138 in production. Next isolate complete-candidate scheduling and tile
packing on that qualified baseline; do not combine this unproven final-cost
fusion with a scheduling change and then attribute the result to either one.
In particular, channel grouping and candidate packing can alter matrix-read
sharing and shared/register pressure without changing the numerical algorithm.
Those are hypotheses to test, not established bottleneck explanations.

Evidence root: `build-cuda-ninja/profiles/s139-artifacts/`. New artifacts are on
the workspace drive because the prior U: evidence drive was nearly full; no
frozen artifacts were removed or overwritten. Drivers and generated prototypes
are under `build-cuda-ninja/profiles/s139_*`.

One native-scan wrapper failed because its metadata filename collided with the
child's analysis JSON. The child outputs and success marker were preserved;
the separately recorded native verifier checked their hashes and conclusions.
No scan was restarted, and the original wrapper is not counted as an accepted
job. This was a diagnostic reporting failure, not a GPU or permission failure.
The failed fixed-power trace analysis is also preserved, and no runtime trace
was repeated to replace it. All processes are terminal. There are 180 recorded
jobs: 165 accepted, fourteen expected writer rejections and one failed
fixed-power analysis, plus the native wrapper failure without terminal job
metadata. Forty retained runtime files and all reused frozen S138 inputs keep
their hashes.

Key records include `before.json`, `inputs.json`, `production_inputs.json`,
`trace_inputs.json`, `native_scan.json`, `linked.json`, `within_analysis.json`,
the four `production_*.json` campaign manifests, `trace_analysis.json`,
`shape_analysis.json`, `prior_cost_budget.json`, `timing_summary.json`,
`power_readonly.log`, the preserved failure logs, `final_summary.json`,
`source_snapshot_index.json` and `artifact_hashes.json`.
