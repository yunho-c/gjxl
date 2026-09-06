# Resident AQ profile counts and storage

This milestone-4 checkpoint follows `1b1ef91`. It replaces the AQ test's
observed-graph allowance with a geometry/policy-derived bound for recording,
snapshot resolution and frame-output wall stages. It also plans reference
preparation, initial quantization and quant-field adjustment profiles. These
are complete **individual profile** bounds, not whole-workflow admission.

The implementation is
[`metal_aq_profile_storage_plan.h`](../src/gpu/metal/metal_aq_profile_storage_plan.h)
and its companion source. Successful planning does not allocate managed backing
or inspect pixels. Checked geometry and policy validation precede arithmetic;
failure leaves the output unchanged. Caller inputs, evaluator/device storage,
score histories, frames, earlier output graphs and parent-session aggregation
remain separate owners.

## Dispatch counts are not stage counts

A timestamp stage can encompass a substantial dispatch sequence. The bound
counts every dispatch, including stage-only profiling, and does not substitute
the 4096 timestamp-sample limit for an allocation bound.

Let `R(N)` be the number of successive ceiling divisions by 256 needed to reach
one, **including one pass when N is already one**. Production inputs are
nonzero. Thus `R(256)=1`, `R(257)=2`, `R(65536)=2`, and `R(65537)=3`.
`NextButteraugliReductionCount` is shared by the planner and both executing
reduction loops. A resident score reduction operates on anchor count, not just
one fixed final dispatch; the complete-map path operates on source pixel count.

The production Butteraugli call graph gives these component counts:

- Psycho image: 13 dispatches—tiled Opsin, tiled low/medium frequencies, four
  high-frequency passes, two medium-B blur passes, suppression, and four
  ultra-frequency passes. Mask precomputation is fused into the last pass.
- Malta: six accumulation sites, each one fused dispatch or two fallback
  dispatches when the pipeline's threadgroup capacity is insufficient. The
  bound uses 12, sharing the unchanged accumulation-order array with execution.
- Mask blur: two dispatches. Reference erosion: one. Expansion/subsampling:
  three each. L2, final composition and crop/scale composition: one each where
  those calls occur. Resident sinks fuse the otherwise separate L2 work.

With `P` source pixels, `A` anchors and `F` nonempty AQ families:

| Prepared Butteraugli path | Reference preparation | Complete-map comparison | Resident-sink comparison |
| --- | ---: | ---: | ---: |
| Expanded small image | 19 | `33 + R(P)` | Unavailable |
| One unexpanded scale | 16 | `29 + R(P)` | Unavailable |
| Two scales | 35 | `62 + R(P)` | `58 + F + R(A)` |

Expansion means either source dimension is below eight; two scales require
both dimensions at least 15 and no expansion. Those predicates come from the
shared device-storage plan. The resident bound is for AQ's seven-family
descriptor. Test-only stage capture, including its additional copy/uncached
reference work, is explicitly outside this production plan. The two-dispatch
Malta hardware fallback is source-bounded; the M4 Pro runtime checks exercise
the fused implementation, not a forced fallback.

## Complete resident policy profile

Before AC placement, use padded block count `B` for `A` and `min(7, B)` for `F`.
This is an upper bound even when the geometry cannot actually use every family.
It requires no selected grid or image-dependent decision. The sink flag must
agree with the source geometry.

For score count `Q = iterations + evaluate_final_field`, Gaborish flag `G` and
EPF pass count `E`, define perceptual work `D` as the resident-sink comparison
above, or the complete-map comparison plus `F` block-reduction dispatches.
The full resident policy dispatch bound is:

```text
Q * (23 + 4*F + G + E + D)
  + 2*F + 2
  + (!evaluate_final_field) * (20 + 2*F)
```

The 23 is reset, 20-dispatch resident radix quantizer, Opsin-to-linear and policy
update. Each family contributes adjustment, coefficient production, inverse DCT
and scatter. First use additionally gathers/transforms each family, computes
final CfL and initializes the policy (`2F + 2`). Cached-forward runs need no more
work. An unscored terminal frame requires another quantizer and coefficient-only
pass, but no inverse transform or perceptual evaluation.

`ComputeResidentAqProfileStoragePlan` combines this count with the previous
shared context-array reserve recipe. Stage/group lengths include every fixed
policy label and the shared transform-specific ID functions, whose strings are
unchanged. Kernel labels include the registry limit and generated fallback IDs.
The plan uses the larger of callback-input/original-graph recording and
original/resolved-copy overlap, then adds wall-stage backing. No frame output
adds zero wall stages; owned output adds two; independent completed output adds
three. Actual frame/score storage is not disguised as part of the profile.

The shared shader geometry limit bounds this policy at 636 dispatches, including
the Malta fallback. This is a consequence of the reviewed call graph and
iteration limits, not a new execution cap or a permission to truncate profiling.

## Auxiliary submissions and initial sorting

`ComputeAqAuxiliaryProfileStoragePlan` returns three separately selectable graph
plans. Each uses one stack stage/context descriptor rather than heap callback
arrays. Reference preparation uses the count above. Quant-field adjustment uses
one reset plus one dispatch per nonempty family, bounded by `1 + min(7, B)`.

Initial quantization starts with five dispatches: reset, gradient, erosion,
modulation and mask convolution. Resident AC inputs add mask validation and,
with Gaborish, three inverse-filter dispatches. Resident initial CfL adds one.
The optional frame-only quantizer adds two bitonic sorts plus five fixed
prepare/median/deviation/finalizer/raw-quant dispatches.

`ComputeInitialQuantSortPlan` extracts the executing power-of-two capacity
calculation. If the reserved sort count is `2^L`, the two sorts together record
`L(L+1)` dispatches. Preparation uses this same capacity; ordering and sort-step
arithmetic are unchanged. The complete initial bound is therefore:

```text
5 + resident_ac + 3*(resident_ac && gaborish) + initial_cfl
  + frame_only_quantizer * (5 + L*(L+1))
```

At the shared coefficient-storage geometry limit, `L` is at most 25 and this is
at most 665 dispatches. The standalone sort helper additionally checks its
32-bit sort-index boundary. These count options describe selected encode flags;
full evaluator-preparation validation remains in the existing API.

The three auxiliary graphs can survive simultaneously in a parent session.
They must not be replaced by their maximum when composing retained ownership.
Only their serial transient recording/resolution work can use a maximum, with
earlier retained graphs added separately.

## Qualification

Fresh trees are `build/resident-aq-profile-plans` and
`build/resident-aq-profile-plans-asan`. The frozen parent is `1b1ef91` in
`build/resident-submission-plans`; it is not rebuilt against new source.

The permanent unit test checks 576 resident policy shapes, 128 auxiliary flag
shapes, nine reduction-depth boundaries and 64 sort-boundary cases through
`2^31`. Invalid geometry, policies, sink counts and output pointers preserve
their prior plans. Zero-credit/armed-failure-hook checks include a geometry near
the coefficient limit, reaching four reduction passes, 636 policy dispatches
and 665 initial dispatches without allocating an image.

The expanded Metal test exercises 434 resident profile cases. Six small/medium
geometries cover every iteration/final-score/Gaborish/EPF combination, including
expanded, single-scale and multiscale inputs. Two additional shapes (2049x2049
and 14x4682) verify three-pass anchor and complete-map reductions respectively.
Quantizer stages have exactly 20 dispatches; score-reduction stages match actual
anchor/family or pixel counts. The complete diagnostic envelope is planned
before recording, not fitted from the first observed graph. Owned and completed
outputs preserve exact frames and scores against the reference and ordinary
cached execution. Existing metadata-failure/recovery and AQ-host lifetime tests
remain active.

Sixty initial-quantization cases vary resident AC, initial CfL, optional quantizer
and inverse filtering across six geometries. Recorded initial counts equal the
formula, and fields/masks/quantizers match ordinary execution exactly. Six
reference and adjustment pairs also fit their separate bounds; reference counts
are exact. Reference backing can enter the idle cache after owner destruction;
the test explicitly trims it before asserting the domain is empty. An initial
test failure came from omitting that trim, not an allocation escaping accounting.

Stage-boundary timestamps are supported on this M4 Pro. Dispatch-boundary
timestamp sampling is unavailable and is not newly qualified. The dispatch
**graph** remains recorded and checked in stage mode.

The new behavioral tests also pass against the frozen parent runtime/metallib,
compiled with the new planner and test source. The header additions introduce
no class members or base-class changes. This checks the bounds against the
parent's recording recipes as well as the candidate's shared calculations.

Full Release results are parent **86/87** and candidate **87/88**, with only the
same inherited CPU `quantization_pipeline` golden mismatch: actual
`0.24919039011001587`, expected `0.24914586544036865`. Four selected tests (AQ
profile plans, AQ host/lifetime, submission metadata and diagnostic storage)
pass three repetitions each in Release and ASan/UBSan: 12 runs per configuration.
Strict warnings also pass for the new planner/test and the expanded AQ test.

The unsuppressed sanitizer run reproduces the metal-cpp Objective-C nil-reference
idiom at `Foundation/NSObject.hpp:112:49`. Successful runs disable leak detection
and suppress only `null:*/third_party/metal-cpp/*`, with halt-on-error enabled.
No GJXL code is suppressed; this is not an unsuppressed sanitizer-clean claim.

All 56 corpus/policy codestream pairs match exactly. Kodak17, planter 4K and
padded-stress 4K outputs are independently decoded with the pinned decoder and
have identical decoded PFM hashes. Both builds pass all 22 pinned conformance
fixtures. Sixty-one common runtime objects and the complete metallib remain
byte-identical; existing AQ evaluation and Butteraugli host units change, plus
the new planner unit. No numerical shader or serializer object changes.

Local evidence, exact commands and output hashes are in
`build/aq-profile-plan-qualification/`, driven by `run.py` and sealed in
`validation.json`. Earlier evidence and frozen builds are preserved. No new
performance or physical-footprint measurement is asserted by this checkpoint.

## Remaining whole-workflow work

These plans close the policy/geometry-to-count gap for the covered Metal AQ
submissions. They still need composition with host policy fields, retained
preparation across attempts, completed/owned outputs, parent diagnostic sessions,
CPU serialization, and retained batch results. Public execution-domain admission,
the remaining lifetime/reuse dispositions and aggregate CPU scheduling remain
unfinished. No latency, throughput, physical-footprint or whole-admission claim
is implied by this profile checkpoint.
