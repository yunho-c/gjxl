# Early sparse selection and fused readback (S158)

## Status

The early-selection and fused-readback candidate is qualified and its primary
paired campaign is complete. It is not promoted: sparse-input gains are clear,
but dense retained-context regressions and inconsistent incremental fusion
results need explanation. A completed context-order diagnostic also exposes
several-percent shifts between identical-mode controls. Production remains
at the S152 implementation; the native sparse frame work from S156/S157 has not
been enabled by default. No compatibility adapter or dense expansion cache is
introduced.

## Existing statistics replace an additional density kernel

S157 showed that choosing sparse storage after packing penalizes dense inputs.
The encoder already calculates exact per-position **zero counts** for coefficient
ordering. S158 sums the first 5,952 full-population bins with a 64-bit accumulator
and subtracts that sum from the active coefficient count. The final 192 sampled
DCT8 bins duplicate a subset and must not participate in this sum.

These populations describe the final quantized integers before the existing
active-group packing step. Packing preserves those integers. The resulting
nonzero count is exact, not a sample or an image-content heuristic. Invalid
totals are rejected before any caller output is published.

The existing 24,576-byte population readback moves into an already-required
quantizer/policy readback batch. Final assembly omits its former copy. This adds
neither a density kernel, a device allocation, a submission, nor population
transfer bytes. Tests independently compare the count with dense coefficients
and validate the cached bins against the CPU population implementation.

Experimental selected modes require at least `3 * 256 * 256` active slots and
at most one nonzero per eight slots. Dense inputs skip sparse packing and its
header allocation entirely. This threshold is a candidate policy, not a claim
that every density below it benefits equally.

## Exact-size allocation enables fused readback

Knowing the payload count before packing lets the host allocate its exact typed
payload immediately. Masks, offsets, packed count, frame metadata, and payload
are then copied in one batch. The former second payload-copy synchronization is
removed. The returned packed count must match the early count before assembly.

The sparse compute submission still has an explicit completion check. Both
enqueue-error and completion-error paths drain queued copies before local host
owners can be destroyed. Injected under/over-counts exercise bounded copies and
subsequent mismatch rejection, while preserving the caller's previous frame.

Tiny forced-sparse diagnostic cases without ordering populations retain the
two-stage readback. Selected small cases stay dense. The low selector bit chooses
compact coefficients; each prepared context captures its own selector.

| Diagnostic modes | Purpose |
| --- | --- |
| 0/1 | Dense wide/compact control |
| 2/3 | Prior late automatic selection |
| 6/7 | Prior forced sparse |
| 8/9 | Early selected sparse, separate payload copy |
| 14/15 | Early statistics, forced sparse, separate payload copy |
| 16/17 | Early statistics but dense output; movement-cost control |
| 40/41 | Early selected sparse with fused readback |
| 46/47 | Early statistics and forced sparse with fused readback |

## Qualification

The first unfused version passed 2,016 policy evaluations and the inherited
217-case sparse failure fixture. Its sources, libraries, and two executables
are archived separately from the final candidate.

The expanded normal-build fixtures passed:

- 24 policy cases, 384 contexts, 2,688 evaluations, and 2,520 dense-oracle
  comparisons, including mixed transforms, changed quantization, repeated
  contexts, absent final output, and population-cache validation;
- 480 early-readback failures plus 480 invalidated retries and 360 successful
  controls;
- 245 fused-boundary failures plus 245 invalidated retries and 133 successful
  controls, covering int8, int16, and int32 payloads;
- 52 complete workflow cases across 16 modes: 832 public encoding calls;
- 160 contexts and 960 concurrent evaluations, with shared/independent backends
  and pooled/nonpooled allocation, compared with 24 dense controls;
- all 94 CTests, and a repeated pass of the inherited 217-case failure fixture.

All five new fixtures also passed with the linked project C/C++ translation
units built with clang-cl 22 AddressSanitizer. CUDA translation units, including
their host launch wrappers, remain built by nvcc with CUDA 11.8/MSVC and are not
AddressSanitizer-instrumented. All 14 CUDA sanitizer runs passed: memory and
initialization checks cover all kernels in all five fixtures; race and
synchronization checks cover every SparseAc launch in the fused-failure and
concurrency fixtures. There were zero errors or reported race hazards, and all
five memory checks reported zero leaked allocations. No launch-count limit,
blocking-launch override, or hazard suppression is used.

Device checks use the already-frozen portable Compute Sanitizer 2025.2.1 from
S157. No installation, firewall/security change, power/clock adjustment,
affinity change, or priority change is made for this campaign.

CTest exited zero and reported all 94 tests passed. The initial runner expected
an older summary string and recorded a rejected journal. That journal and log
are retained unchanged. A separate checker verifies all 94 individual pass rows,
the actual summary, and the original log hash; no test rerun was needed.

All 14 CUDA source files are byte-identical to S157. Across 13 ordinary and
AddressSanitizer executables, the extracted device modules have the same
resources and instructions as S157 after normalization of only the two
source-derived anonymous-namespace symbol IDs. No instructions are normalized
away, and no new device kernel is involved.

## Measurement protocol

The predeclared campaign uses same-executable rotating duplicate-control ABBA,
four warmup rounds, eight measured rounds, and two shuffled/reversed repeats.
Every call checks its codestream; the real corpus uses six frozen S153 oracles.
No benchmark overlaps another recorded GPU job or a build. Read-only NVML observations
record the enforced power limit, not actual power or constant clocks.

Synthetic whole workflows use eight CPU workers and fresh fully resident effort-7
pipelines on a retained backend. Cases include low detail, ordinary noise, and
near-dense coefficients. Retained-context tests separately time one unscored
policy update and serialization with fresh caller owners; preparation, output
allocation, validation, and destruction are outside those intervals. Their DCT8
fixtures are not a substitute for the real-image whole-workflow measurements.

The primary statistic is the median of within-round candidate-minus-control
means and corresponding within-round percentages. The four cross-label pairs
are also retained to expose sensitivity to duplicate-control variation.

## Primary results

All 136 primary jobs completed: 6,528 paired/warmup calls plus 112 independent
in-process oracle calls, or 6,640 calls in total. Of the paired/warmup calls,
4,352 were measured and 2,176 were warmups. The six-image corpus accounts for
1,152 calls. All codestream comparisons passed. All 13,056 enforced-limit
observations were 40,000 mW; this does not establish constant actual clocks,
temperature, or power.

Percentages below are candidate time changes against the matched dense control;
negative means faster. Each range contains the two primary repeat statistics,
not a confidence interval. These are same-overlay comparisons, not a direct
cross-build comparison with production S152.

| Real-image whole workflow | Wide control | Compact control |
| --- | --- | --- |
| Flower 500 | -6.15% to -5.74% | -5.29% to -4.13% |
| Padded HD | -13.46% to -12.73% | -8.00% to -6.45% |
| Flower 2000 | -14.02% to -8.73% | -5.03% to -3.29% |
| Padded 4K | -10.38% to -10.24% | -4.29% to -2.88% |
| Flower 3200x2160 | -11.40% to -11.36% | -5.53% to -5.24% |
| Keong 3839x2159 | -13.62% to -10.67% | -9.41% to -5.48% |

All 24 real-image primary statistics favor the candidate. Two of the eight
cross-label comparisons for compact padded 4K reverse direction; the other
real-image pairs all favor the candidate. Wide padded 4K saves 30.46-31.94 ms;
wide Keong saves 38.83-48.17 ms. These gains include the native sparse
representation and consumption, not just the readback fusion added in S158.

Low-detail synthetic whole workflows at sizes 257, 513, and 1025 improve by
13.72-18.55% wide and 6.23-11.12% compact, with all cross-label pairs favorable.
Size 65 stays dense; its low-detail wide result ranges from -0.75% to +0.11%,
and compact from +0.93% to +1.47%.

| Low-detail retained evaluation + serialization | Wide | Compact |
| --- | --- | --- |
| 513x519 | -25.00% to -20.87% | -13.80% to -12.94% |
| Padded HD | -22.75% to -20.90% | -13.44% to -12.55% |
| Padded 4K | -21.28% to -16.94% | -7.43% to -6.68% |

Retained wide 4K saves 23.48-31.27 ms. Its native AC owner contains 4,665,608
bytes instead of 106,168,320; compact contains 4,665,602 instead of 26,542,080.
The reused device payload workspace and the extra 4,665,604-byte device header
are separate from those host-owner figures. Compact retained 4K has one
cross-label reversal, and serialization alone regresses in one repeat.

## Limits exposed by the controls

Dense selection really skips sparse packing: all dense retained cases report
zero pack attempts and zero sparse-readback batches, with identical dense
owner sizes. That does **not** imply zero latency cost. Dense 4K retained cases
regress 1.59-2.94% wide and 1.60-7.95% compact; all eight compact cross-label
pairs are slower. The compact regressions include serialization increases of
10.38% and 2.15%, despite identical coefficient bytes and serialization code.
Their cause is not established by this experiment.

Wide near-dense 1025 whole workflows regress 2.76-3.40%, with all eight
cross-label pairs slower. Ordinary-noise 1025 wide regresses 2.62-4.66%.
Smaller/dense results and the population-movement-only controls show substantial
sign reversals. Moving statistics has not been established as free, and early
selection is not uniformly faster than the prior late-fallback policy.

The fused path demonstrably removes one batch when the sparse payload is
nonempty. However, **fusion alone has no consistent measured speedup** against
the already-early unfused modes. Every synthetic whole-workflow fusion contrast
changes direction between repeats. Retained 4K fusion ranges from -3.02% to
+4.32% wide and -2.04% to +3.17% compact. Retained 513 wide has a small favorable
total result in both repeats, but its evaluation-only result changes direction.
Zero-payload cases already need only one batch without fusion.

The primary retained harness constructs the dense context first and candidate
second, so mode and context-construction order are confounded. The follow-up
uses a new executable with unchanged libraries and byte-identical device
modules: identical-mode dense controls, population-only controls, and the
selected-fused candidate, each with both construction orders and two repeats.
It is a post-primary diagnostic, not silently pooled into the primary campaign.

## Context-order diagnostic and decision

All 12 follow-up jobs passed: 588 calls including 384 measured samples, 192
warmups, and 12 independent dense-wide oracle calls. Every case remains dense,
with 21,924,397 nonzeros among 24,883,200 slots, a 53,084,160-byte int16 owner,
and zero sparse-pack attempts/readback batches. All 1,152 additional enforced
limit readings are 40,000 mW.

| Compact dense 4K contrast | Baseline constructed first | Candidate constructed first |
| --- | --- | --- |
| Identical dense mode 1 vs mode 1 | +1.15% to +2.48% | -4.92% to -1.50% |
| Population-only mode 17 vs mode 1 | -0.77% to -0.01% | -0.26% to +4.37% |
| Selected-fused mode 41 vs mode 1 | -0.73% to +2.38% | -0.84% to +2.07% |

These controls expose measurement/context sensitivity of the same order as
several reported regressions. They do not identify a fixed allocation penalty
or prove that the population movement is free. In one identical-mode run the
within-round primary statistic favors the candidate while all four cross-label
median comparisons favor the baseline. Both calculations are retained; they
measure differently under the observed variation and must not be substituted
selectively to obtain a favorable answer.

The follow-up selected-fused results do not reproduce the primary compact
dense 4K penalty as a consistent fixed effect. Nonetheless, neither the primary
regressions nor the small dense-control differences can be dismissed as an RDP
effect or a confirmed timing artifact: no such cause has been established.

S158 therefore establishes a correct exact-count handoff, one fewer readback
batch for nonempty sparse payloads, and useful sparse-input whole-workflow and
retained-context gains. It does **not** establish universal dense-path neutrality
or an incremental end-to-end fusion win. The next implementation/promotion step
should first gate out unnecessary population movement for ineligible small
frames, separate context/host-allocation/serializer effects in dense timing,
and compare the stripped production candidate directly with the production
baseline. A new GPU density-reduction kernel is unnecessary for this design.

## Evidence and reproducibility

Evidence resides in `build-cuda-ninja/profiles/s158-artifacts`. It includes the
first qualified unfused source/binary archive, final build-input snapshot,
ordinary and instrumented builds, native device-code comparisons, all job logs
and journals, both predeclared protocols, derived analyses, and final source
snapshots. The post-primary protocol was declared before that diagnostic's
samples, separately from the primary protocol.

`analyze_s158.py` and `analyze_s158_crossover.py` independently rederive the
reported paired statistics. `verify_s158.py` checks source/binary/log hashes,
all 2,605 predecessor artifacts, completed qualifications, protocol ordering,
and nonoverlap of recorded timing jobs with other recorded work. It retains the
single CTest summary-checker rejection explicitly: 183 journals in total, 182
accepted, with no actual test failure. `freeze_s158.py` snapshots the final
sources and hashes the artifact inventory; `verify_s158.py --frozen` validates
that inventory after the report commit. No push is performed.
