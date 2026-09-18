# Fused CUDA AC evaluation and zero forward scratch (S138)

Disposition: retained. The normal CUDA candidate batch now uses the fused
three-channel evaluator qualified in [S137](cuda-fused-ac-evaluation-s137.md).
Forward coefficients remain in shared memory through residual quantization,
inverse transform and rate/loss reduction. The unused global forward scratch
is removed from CUDA validation, the internal launch interface and the prepared
owner's allocation plan. No experiment selector or compatibility path is
enabled in production.

At padded 4K, the actual AC-search device arena shrinks by 304,496,640 bytes
(290.39 MiB). Controlled whole-encode paired deltas improve in 14/16 runs,
including all four 4K comparisons at −15.7 to −25.8 ms. Quantization and enclosing
AC search improve in all sixteen primary comparisons. Two small-image
whole-call deltas are unfavorable; this is not a universal speedup claim.
The encoder is not established to be maxed out.

## Implementation and contract

The seven supported shapes use the S137 scheduling rule: 192 threads per block,
`max(width, height)` lanes per transform, and `64 / max(width, height)` complete
three-channel candidates per block. Each channel has a padded shared tile, and
each candidate has an immutable shared Y-coefficient copy for cross-channel
CfL reads. Inactive tail groups still participate in every barrier. Transform
pass order, normalization, coefficient layout and tall-pixel redistribution are
unchanged from the qualified prototype.

The implementation reuses `AcStrategyResidualDctSource` and
`AcStrategyDctLossOutput`, adding compile-time shared-input/scalarized-reduction
specializations. The quantization and loss formulas are not duplicated.
Compile-time recursive halving preserves the FP32 reduction tree and avoids
the local arrays observed in S137 V1. Separate forward, residual/inverse and
inverse-loss entry points remain internal numerical references exercised by
tests, not production fallbacks. Quant-norm preparation and final cost remain
separate kernels. No fast-math relaxation is introduced.

CUDA reports zero B bytes. A zero scratch requirement means its pointer and
offset are unused, so B is not resolved, alignment-checked or included in the
input/output overlap graph. A, rates and final costs remain disjoint checked
outputs. Existing conservative count/index/grid limits are unchanged. The
common prepared owner skips zero-byte B planning/allocation and clears its B
view on every preparation, including reuse. Metal's positive B requirement
still follows the original allocation path; no Metal runtime qualification is
claimed here.

The broad contract test now allocates only queried ranges. Its compact arena
packs the three live outputs directly together, preserving short-range,
misalignment, partial-overlap, input-alias and foreign-output rejections. It
also verifies that unused B fields can name foreign storage with a maximal
offset without being accessed. The old A/B alias test becomes an A/cost alias
test, rather than expecting an unused range to reject a request.

## Actual prepared-memory reduction

The frozen S134 and new production prepared-search tests both run at
3840×2160. The new test independently reconstructs the owning layout, explicitly
requires zero B, and checks repeated preparation plus shrink/restore reuse with
no further device allocations. The same check passes under scoped host ASAN.

| Prepared AC-search storage | S134 bytes | S138 bytes |
| --- | ---: | ---: |
| Owning resource arena | 323,845,888 | 19,349,248 |
| Channel-loss scratch A | 1,555,200 | 1,555,200 |
| Forward-coefficient scratch B | 304,496,640 | 0 |
| Channel-rate scratch | 3,110,400 | 3,110,400 |

The arena difference equals the removed B range exactly. Descriptor, matrix
and cost storage remain present. Every controlled encode also checks the
queried B size and actual arena capacity, with the candidate capacity exactly
one old B range below its baseline. This is device-buffer capacity, not process
VRAM, allocator reserve or RSS. The logical coefficient-traffic opportunity in
S137 is not relabeled as measured DRAM bandwidth.

## Qualification

Starting revision `c805915`, branch `feat/cuda`, September 8, 2026. Windows,
RTX 3060 Laptop (`sm_86`), CUDA 11.8, MSVC 14.37, Release; scoped ASAN uses
clang-cl. A fresh CMake/Ninja CUDA build passes 82/82 CTests, none skipped.
The initial build's warnings for two template variables are resolved with
`[[maybe_unused]]`; the recorded rebuild is warning-free.

The new permanent `cuda_fused_ac_strategy` fixture covers fifteen block
geometries, including thin images, partial tiles and axes through 257 blocks.
Both host and signed device CfL are tested: 144 nonempty geometry/CfL cases and
29,876 descriptors per execution. Strided RGB/mask inputs, matrices, descriptors,
reference norms/coefficients and outputs occupy a guarded arena. An independently
launched forward-plus-residual/inverse path supplies exact channel-loss and
channel-rate bytes. Three normal public fused batches must match those bytes,
preserve all inputs/guards and unused reference storage, produce finite costs,
and repeat final-cost bits exactly. This fixture uses descriptor quant norms;
device quant-norm behavior is covered by the broad contract and encode tests.
It does not claim a separate old-final-cost oracle inside this new fixture.

The fixture passes CTest, standalone, scoped ASAN and four CUDA sanitizers:
seven executions. The exhaustive extension covers all block width/height pairs
from 1 through 19 and both CfL forms: 4,402 cases and 269,618 descriptors in
each of Release and ASAN. Together the focused/exhaustive runs execute 9,812
reference and 29,436 fused batches. Repeated evaluations are not independent
images. The unchanged final-cost source/native body and frozen whole-encode
oracles supply additional coverage beyond repeatability.

The broader production matrix passes eighty configurations across quality,
rate search/prepared reuse, stress and one-/two-request batches in both
coefficient widths, plus fourteen writer rejections matching frozen errors.
Sixteen production ASAN configurations, four production memchecks and four
production initchecks also pass. Production checks total 1,296 frozen-oracle
encodes. Controlled preflights/timing add 1,396, and traces add forty:
2,732 frozen-oracle encodes, including 262 scoped host-ASAN encodes.

Fourteen CUDA sanitizer jobs qualify: four focused grid jobs, eight production
encode jobs and two broad contract jobs. Memory/init/synchronization checks
report zero errors, race checking reports zero hazards, and all qualified
memchecks report zero leaked bytes. Broad contract and prepared-memory/reuse
checks also pass under ASAN. ASAN covers the modified host validator/search,
resident owner, diagnostic callers and the two pipeline units, not every
library. No new independent decoder, Linux or second-GPU qualification is
claimed.

## Native-code audit and benchmark isolation

Production has 221 kernel bodies. All seven fused bodies match S137 V2
instruction/control bytes exactly and retain its resources: zero stack/local
storage and zero static local-memory operations; shared storage ranges from
8,576 to 33,536 bytes. Of the 214 original S134 bodies, 213 are unchanged.
The source-unchanged quant-norm kernel has the historical code-generation
variant already recorded in S136 and matched there to S132/S133. Its source
formula, 35 registers and zero stack/local/shared resources are unchanged.

Nineteen GPU executables are module-audited, including production, focused and
ASAN tests, controlled encoders, traces and the explicitly flushed contract
fixture. Each has ten modules. Eight production modules match S134; nine match
between production and diagnostic variants. The diagnostic batch's quant-norm
body matches S134 instead of production's historical variant. Final cost and
all other bodies match their corresponding references. There is no identified
compiler root cause for the harmless norm variant.

The controlled harness restores old-size B allocation and the two original
transform launches only for baseline labels. Candidate labels use the normal
fused kernel with zero B. Both use the new common host validator; obsolete
per-batch B validation is not restored to the baseline. A diagnostic callback
checks the restored B owner's type/range and supplies its pointer to the
baseline launches. Both labels use the same diagnostic quant-norm variant.
These are instrumented public-encode comparisons, not byte-identical copies
of two complete release executables. The broader production matrix uses the
actual new production GPU modules. No diagnostic selector/callback is linked
into the normal CMake encoder.

## Balanced complete-encode timing

Inputs match S133/S137: Flower 500, padded HD from 1919×1079, padded 4K from
3839×2159 and Flower 2000 (fourfold nearest-neighbor replication, not a native
2000-square photograph). Distance is 1.2, effort 7, fully resident, with wide
and opt-in compact coefficients. No coefficient-width default changes.

Labels 0/2 are duplicate baselines; 1/3 duplicate candidates. Four-round Williams
blocks balance label positions and ordered predecessors. Each process has one
reference encode, four warm rounds and sixteen measured rounds. Two process
passes reverse case/width order: sixteen timed processes, 1,024 measured
encodes. No other recorded task build, qualification or trace job overlaps
these timed processes. Every encode checks frozen bytes, summary, coefficient
width/storage, dispatch counts and exact scratch/arena accounting.

The primary statistic is the median across rounds of mean candidate-label time
minus mean baseline-label time. Negative is faster. Whole-call time includes
input preparation, quantization and serialization; search is nested inside
quantization, so these savings must not be added together.

| Case / width | Whole-call delta r0 / r1 (ms) | Quantization delta r0 / r1 (ms) | Search delta r0 / r1 (ms) |
| --- | ---: | ---: | ---: |
| Flower 500 / wide | +0.192 / −0.089 | −0.142 / −0.284 | −0.158 / −0.161 |
| Flower 500 / compact | −0.294 / +0.082 | −0.065 / −0.180 | −0.153 / −0.156 |
| HD / wide | −1.226 / −1.035 | −1.362 / −0.317 | −1.189 / −0.971 |
| HD / compact | −1.405 / −1.178 | −1.654 / −1.117 | −1.342 / −1.146 |
| 4K / wide | −25.367 / −19.281 | −18.311 / −18.814 | −5.839 / −7.317 |
| 4K / compact | −15.700 / −25.788 | −7.980 / −25.793 | −0.210 / −5.800 |
| Flower 2000 / wide | −3.015 / −10.571 | −5.626 / −9.316 | −2.957 / −2.082 |
| Flower 2000 / compact | −5.698 / −6.348 | −5.614 / −4.687 | −1.847 / −1.809 |

Whole-call comparisons favor the candidate in 14/16 primary results and 59/64
cross-label pairs; quantization is 16/16 and 63/64; search is 16/16 and 61/64.
Preparation is only 7/16 and 38/64: removing B is not established as a reliable
host-preparation timing gain. The two unfavorable whole-call results are
Flower 500 at +0.192 and +0.082 ms. Four-K whole-call percentage deltas are
approximately −7.68%, −5.98%, −5.00% and −8.10%, but duplicate controls remain
substantial: equivalent candidate labels differ by −12.302 ms in the second
wide-4K process. No universal percentage or post-hoc size threshold is claimed.
S137 and S138 gains are not additive; both compare fusion against decomposed
evaluation, with different allocation layouts and campaigns.

## Traces, disposition and next work

Eight complete-process Nsight Systems captures contain 32 labeled windows
plus eight reference encodes. All CUDA calls succeed. The candidate replaces
fourteen AC forward/residual kernels with seven fused kernels, leaving seven
quant-norm and seven final-cost kernels. All other kernel-name/count multisets
and memcpy counts/payload bytes match. Four-K launches fall from 301/302 to
294/295 for wide/compact; HD falls from 327/328 to 320/321, and both Flower
sizes from 366/367 to 359/360.

In this campaign, traced summed AC work is favorable in both widths for all
four inputs. At 4K it changes from 28.23 to 20.00 ms wide and 24.59 to 17.03 ms
compact. These short instrumented captures support the changed GPU-work shape,
not a universal latency estimate or a complete explanation of whole-call
variation. All 2,872 power endpoints report a 40 W enforced limit, which does
not prove stable clocks, cache/placement or absent external activity. No power,
clock, priority, affinity or firewall policy was changed, and no elevation or
firewall blocker was encountered.

Retain the normal fused path and smaller owner. Next investigate the remaining
quant-norm/final-cost boundaries and tile packing using this qualified baseline.
Complete candidates now share a block, so final channel composition/reduction
may be able to consume local results without global loss/rate scratch. Test
register/shared-memory tradeoffs and complete-encode behavior before combining
that hypothesis with further scheduling changes. The rejected S112, S135 and
S136 experiments are not re-enabled by this result.

Evidence root: `U:/gjxl-cuda-diagnostics/s138/`. Key records are `before.json`,
`within_inputs.json`, `production_inputs.json`, `trace_inputs.json`,
`within_analysis.json`, `timing_summary.json`, `memory_analysis.json`, the four
`production_*.json` manifests, `native_scan.json`, `linked.json`,
`trace_linked.json`, `contract_flushed_linked.json`, `trace_analysis.json`, job
logs, `source_snapshot_index.json`, `final_summary.json` and
`artifact_hashes.json`. Drivers and failed attempts are archived with the
`build-cuda-ninja/profiles/s138_*` sources.

There are 193 terminal job records: 177 successful, fourteen expected writer
rejections, one initial diagnostic compilation failure and one unqualified
sanitizer capture. The diagnostic failure was an exception thrown from an
`extern "C"` checking callback under `/EHsc` and `/WX`; the V2 harness reports
fatal invariant failures without throwing across that interface. The first
contract memcheck exited zero with zero CUDA errors/leaks but omitted buffered
application stdout, so it is not counted as qualified. A diagnostic-only
unit-buffered wrapper repeats both contract sanitizers with complete success
markers and the same production GPU modules. Original records are preserved;
none is a live stalled process. Forty retained runtime files keep their hashes.
