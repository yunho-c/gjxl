# Metal precision study

Status: completed. The targeted FP32 Malta reciprocal is selected for production.
The global math flags, FP16 storage, and matrix operand precision are unchanged.

The accepted change computes one approximate reciprocal for two Malta scale
factors. It meets the preset timing gate on padded 4K (-1.64%) and Planter 4K
(-2.74%), with six of seven independent pairs favorable on each. Other workloads
are less consistent, so this is a modest, workload-dependent improvement.
Numerical qualification found no decoded Butteraugli regression across 184
image/target cases. The complete ledger is
[metal-precision-results.json](metal-precision-results.json).

The study starts from `1576fde` in `gjxl-metal-kernel-dataflow`. Existing dirty
geometry-probe/CMake work and compact-coefficient reports are preserved. Baseline
source and executable identities, generated shaders, commands, and raw results
are retained under `build/precision-study/`.

## Scope

1. Selective FP32 contraction and fast math functions in strict Butteraugli/AQ
   stages, measured independently before combining winners.
2. FP16 intermediate storage with FP32 sensitive arithmetic, including a compact
   eroded masking map and a separately measured filter scratch experiment.
3. Mixed-precision matrix operations in AC candidate evaluation, with FP32
   refinement if the first pass is useful but decision error requires it.
4. Reduced-precision final transforms after the earlier screens establish their
   numerical behavior. Final quantization and decoder-like reconstruction stay
   unchanged unless separately qualified.

The existing matrix implementation and recent geometry studies are the baseline;
their historical improvements are not counted as results of this study.

## Acceptance budget, fixed before screening

These are engineering acceptance thresholds, not codec guarantees or predictions.
Bitstream changes are permitted. Every output must decode using the pinned djxl,
with finite pixels and unchanged dimensions. The unchanged CPU Butteraugli
implementation evaluates independently decoded linear-sRGB pixels. The candidate's
own approximate metric cannot certify its quality.

- Per image, decoded Butteraugli may increase by at most
  `max(0.002, 0.01 * baseline_score)` at the same nominal target.
- At matched decoded quality, geometric-mean size regression must be <= 0.5%,
  with no individual regression > 2%. Report same-target sizes separately.
- For approximate metric screening on identical image pairs, score error must
  be <= `max(0.002, 0.005 * reference_score)` and map maximum error <=
  `max(0.01, 0.01 * reference_map_max)`. Report underestimation separately.
- Local arithmetic errors and changed search decisions are diagnostics; they do
  not substitute for decoded quality checks. No NaN/Inf, overflow, invalid access,
  changed read-only input, or poisoned padding corruption is acceptable.
- Qualify natural photographs and deterministic smooth/dark/near-neutral
  gradients, texture, impulses, and contrast boundaries. Include targets 0.1,
  0.3, 1.2, and 3.0, odd dimensions, and small/1080p/4K timing workloads.
- A speed candidate must show repeatable complete-encode improvement in an
  independent confirmation: seven alternating process pairs, three warmups and
  seven samples, profiling disabled. Require >= 1% median improvement on at least
  two representative large workloads, >= 6/7 favorable pairs there, and no
  repeatable regression > 1% on the remaining workloads. Memory-only benefits
  are reported separately and are not promoted as speed improvements.

A rejected screen is a completed experiment when its implementation, input and
artifact identities, numerical results, and reason for rejection are preserved.
Further refinement is required when the data identifies a plausible repair with
enough potential benefit to pay for its additional work.

## Experimental representation details

The eroded-mask storage screen changes that private map's producer and every
shader consumer to packed half elements. Blurred reference/distorted masks remain
FP32, preserving subtraction before any narrowing. The screen retains the host's
original allocation capacity and offsets; it measures narrower accesses, not a
reduction in allocated memory. A production candidate would require explicit
typed layouts, matching diagnostic bindings, and pipeline capability selection.

The half blur-scratch screen narrows horizontal-pass intermediates only. Sums,
products, differences, and final outputs remain FP32. Dynamic scratch reservation
is initially unchanged, so this experiment cannot establish occupancy savings.

The first matrix screen converts SIMD-group matrix operands to half and keeps
FP32 accumulators, intermediates, and outputs. The AC storage refinement also
packs the gathered threadgroup pixels as half, retaining the FP32 scratch
capacity needed by the later residual consumer. Final coefficient rounding and
integer storage remain unchanged in every matrix screen.

## What was tested

The retained ledger covers 23 arithmetic/storage variants and one unchanged
recompile control. Seventeen variants, including the control, received the
initial three-workload screen. Seven follow-ups isolated the arithmetic behind
the most promising result. Contract failures and absent stage headroom stopped
unsuitable follow-ups before expensive confirmation.

| Family | Implemented experiments | Outcome |
| --- | --- | --- |
| Butteraugli FP32 | Contraction, explicit FMA, fast functions, relaxed arithmetic with fast or precise functions, reciprocal compiler flag | Broad relaxed arithmetic was faster and passed decoded-quality checks, but changed three existing contract tests. |
| AQ FP32 | Reconstruction contraction; EPF contraction/fast functions; reduction fast functions | No convincing broad speedup; EPF contraction exceeded the decoded-quality budget on the flat image. |
| FP16 storage | Eroded mask; horizontal blur scratch | Mask narrowing passed its screen but showed no compelling whole-encode gain. Blur scratch failed the quality budget. Neither establishes an allocation/RSS reduction. |
| AC matrix precision | Half forward operands; half forward/inverse operands; half gathered input storage, all with FP32 accumulation | Screens passed, but AC-stage savings were negligible. FP32 rescoring would add work to a first pass that was already not faster. |
| Final DCT precision | Half matrix operands with FP32 accumulation and unchanged integer rounding | Failed high-quality neutral/Kodak cases and did not improve speed. |
| Targeted FP32 repairs | Relaxed math without contraction; local Malta/filter reassociation; filter reassociation without contraction; their combination; precise and fast shared Malta reciprocals | Local reassociation preserved contracts but recovered little speed. The explicit Malta reciprocal retained useful stage savings and passed full numerical qualification. |

All prototypes are generated by `tools/metal_dataflow/precision.py`; no FP16
prototype changes production allocation layouts. The matrix experiments use the
existing SIMD-group implementation, with `__builtin_convertvector` for half
operands and FP32 accumulators. They do not measure hypothetical matrix hardware
throughput or substitute dense DCTs for the existing factored/matrix paths.

Representative rejection evidence:

- EPF contraction: flat image, target 1.2, decoded Butteraugli
  `0.2871216 -> 0.3586436` despite unchanged file size.
- Half blur scratch: neutral image, target 0.1, score increase `0.00203385`
  and size increase about 67.5%; flat/1.2 also failed quality.
- Half final DCT: neutral/0.1 score increase `0.192897` and size ratio
  `2.78874`; neutral/1.2 and Kodak17/0.1 also failed quality.

## Broad relaxed arithmetic: measured benefit, rejected integration

`ba_relaxed_precise` retains precise FP32 functions while allowing relaxed
arithmetic across the Butteraugli translation unit. Its independent seven-pair
whole-encode confirmation measured median paired changes of -3.88% on padded
1080p, -2.77% on padded 4K, -2.74% on Planter 4K, -2.85% on Sun Forest 4K, and
-3.54% on the 4672x5584 Doughnuts image. All seven pairs favored the candidate
on each of these large workloads. Tiny-image timings were noisy.

Its 184 unique image/target cases passed the original 1% decoded-quality budget.
Three target-0.3 cases needed additional samples for a tighter rate comparison.
Matching uses an observed score no worse than baseline plus
`max(0.0002, 0.001 * baseline_score)`; better-quality points are conservative
rate evidence. Seven fixed lower target offsets (0.01%, 0.05%, 0.1%, 0.2%, 0.5%,
1%, 2%) were evaluated independently for each unmatched case. The smallest
eligible observed codestream was retained, without assuming monotonicity or
interpolating a BD-rate curve. The resulting geometric-mean size increase was
0.009923%, and the worst individual increase was 1.864275%.

That corpus result did not qualify a global compiler-policy change. The full
integration suite introduced failures in `metal_butteraugli_operation_contract`,
`metal_aq_evaluation`, and `adaptive_quantization_gpu_policy`. The latter's
bounded policy fixture changed an iteration score from `0.9429513812` to
`0.962431`, with block-map errors up to about `0.127`. Disabling contraction
globally did not repair that policy failure. These are distinct from decoded
quality on the fully-resident corpus. Existing test tolerances were not widened.

The raw failed build, validation logs, and passing/failing contract comparisons
remain available. The initial full build had 123/124 tests passing; only
`quantization_pipeline` failed. The broad candidate had 120/124 passing.

## Targeted Malta reciprocal

Malta's two scale factors divide by the same positive normalization denominator.
`ba_malta_fastdivide` calculates `fast::divide(1.0f, norm + absolute)` once and
multiplies both numerators by it. All storage and arithmetic remain FP32, with
the surrounding safe/precise compiler flags and contraction policy unchanged.

At padded 4K, its three-pair stage screen reduced main Malta GPU time by 14.03%
and subimage Malta by 16.41%; the sum of all disjoint GPU stages improved 1.35%.
A precise shared reciprocal improved those Malta stages by 10.63% and 11.68%,
respectively. Local reassociation alone recovered only about 0.2–0.3% of all
GPU stage time. The fast reciprocal's initial whole-encode screen was -1.65%
at 1080p and -0.68% at 4K.

Independent, profiling-disabled confirmation:

| Workload | Median paired whole-encode change | Favorable pairs |
| --- | ---: | ---: |
| Synthetic 128x96 | -0.03% | 4/7 |
| Padded 1080p | -1.25% | 4/7 |
| Padded 4K | **-1.64%** | **6/7** |
| Planter 4K | **-2.74%** | **6/7** |
| Sun Forest 4K | -0.76% | 5/7 |
| Doughnuts 4672x5584 | -1.82% | 5/7 |

The two bold rows meet the preset >=1% / >=6-of-7 criterion. No remaining
workload has a repeatable >1% regression (using the same >=6-of-7 direction
criterion). Variability is substantial: Planter pairs range from -11.79% to
+6.96%. The medians are useful confirmation of a small benefit, not evidence
for a universal 2–3% gain. GPU-stage savings are reported separately above.

Numerical qualification of the fast reciprocal:

- 46 stress/natural cases plus the pinned 38-image corpus at four targets
  (152 cases), with overlaps removed: **184 unique cases**.
- **183/184 byte-identical**. Sun Forest 1080p at target 0.1 changed decoded
  pixels but retained exactly the same reported CPU Butteraugli score and used
  one fewer byte (1,758,350 bytes). No measured decoded-score regression and no
  size regression; no target refinement was necessary.
- 59 fixed-pair metric comparisons, including identity pairs: maximum score
  error `4.76837158203125e-7`, maximum underestimation
  `1.1920928955078125e-7`, maximum map error `7.152557373046875e-7`.
- All 18 focused fixed-pair comparisons also passed Metal API/shader validation.
  Odd strides, poisoned padding, prefix/suffix guards, and read-only inputs were
  checked independently from the numerical thresholds.
- Fresh Release integration: **123/124 tests pass**, with only the inherited
  `quantization_pipeline` failure. All six focused contract/AQ/Butteraugli tests
  pass with Metal API and shader validation enabled.
- All seven compiled shader translation units match the screened candidate's
  complete textual AIR after removing only ModuleID and exact source path
  strings. Binary AIR differs because source filenames are embedded. Three
  encodes using the final linked integration artifacts match the screened
  codestreams exactly.

## Measurement and reproduction

Measurements use the 20-core Apple M4 Pro, Release builds, effort 7, target 1.2,
SIMD-group transforms, fused-tuned AC candidates, and explicitly forced
fully-resident Metal AQ. They do not widen the automatic resident admission
policy. Numerical coverage additionally includes targets 0.1, 0.3, and 3.0.

The whole-encode boundary is the benchmark's `metal-public-workflow`: input
preparation, backend selection, quantization/AQ, codestream assembly/serialization,
and summary assembly. File I/O, cold process startup, and final quality scoring
are outside that timed boundary. Stage-instrumented runs are separate from
profiling-disabled wall runs. Aggregate worker time is not reported as latency.
Each pair consists of two independent processes in alternating baseline/candidate
order, with three warmups and seven samples each. Percent changes summarize
paired process medians. Identical host binaries and frozen, explicitly selected
metallibs are used for each comparison. Known competing build/measurement jobs
are rejected by the quiet guard.

The scripts retain commands, exit status, environment overrides, raw samples,
source/library/binary/input hashes, independently decoded images, and CPU metric
results in `build/precision-study/`. The frozen initial budget is in
`build/precision-study/src/docs/metal-precision-study.md`; later refinements do
not overwrite it. Failed attempts are retained. A matching successful command
can resume; changed identities require a new output directory.

The main entry points are:

```sh
# From the recorded starting source, with this harness present:
# one-time snapshot and Release baseline in an empty study directory.
python3 tools/metal_dataflow/precision.py prepare
python3 tools/metal_dataflow/precision.py build
python3 tools/metal_dataflow/precision_quality.py inputs
python3 tools/metal_dataflow/precision_quality.py baseline
python3 tools/metal_dataflow/precision_quality.py baseline --corpus

# Generate and qualify an isolated candidate:
python3 tools/metal_dataflow/precision.py variants ba_malta_fastdivide
python3 tools/metal_dataflow/precision_contracts.py baseline ba_malta_fastdivide
python3 tools/metal_dataflow/precision_quality.py ba_malta_fastdivide
python3 tools/metal_dataflow/precision_quality.py ba_malta_fastdivide --corpus
python3 tools/metal_dataflow/precision.py metric-probe
python3 tools/metal_dataflow/precision_metrics.py ba_malta_fastdivide --full
python3 tools/metal_dataflow/precision_metrics.py ba_malta_fastdivide --validation
python3 tools/metal_dataflow/precision_matched.py ba_malta_fastdivide
```

The quality scripts use NumPy and the pinned local corpus/decoder paths declared
in `precision_quality.py`. `precision_candidate.py` freezes a separate integration
source/build and verifies shader equivalence. `compare.py` records alternating
timings; the exact original and confirmation invocations are retained in the
command records. `precision_report.py --publish` produces the compact repository
ledger after a completed decision is recorded. This study is specific to this
checkout, toolchain, device, and corpus; neither FP16 results nor small timing
effects are universal hardware claims.
