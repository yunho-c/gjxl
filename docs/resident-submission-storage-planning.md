# Metal submission metadata storage plans

This milestone-4 checkpoint follows `e1abff9`. It connects AC and resident AQ
callback-input capacities to the actual recording code, and adds an AC
policy-to-dispatch bound. It does not implement whole-workflow admission or an
AQ/Butteraugli dispatch-count estimator.

## AC validation, recording and resolution

[`ComputeAcSubmissionStoragePlan`](../src/gpu/metal/metal_submission_storage_plan.h)
uses the actual private `ValidatedAcStrategyBatch` and `AcStrategyProfileContext`
types, plus `MetalProfiledComputeStage`. Friend declarations allow `sizeof` the
executing types without making the implementation records public or copying ABI
byte constants into a planner.

Let `B` be the input batch count and `N <= B` the nonempty count. Validation
reserves `B` records even if all batches turn out empty. Profiling reserves `N`
contexts and `N` stage records. The unprofiled path needs only validation backing.
All three arrays coexist with the original profile graph during synchronous
command recording. They are gone before the completed submission's graph is
deep-copied during resolution. The plan's `working` bound covers the larger of
recording and resolution, including their different overlapping owners.

Execution uses the returned capacities. Its first call plans the validation
array; after filtering empty batches, its second call uses the actual `N`.
Upfront workflow planning can conservatively use `N = B`. This preserves the
old allocation recipe rather than allocating contexts for empty batches. Public
submission spans can contain more than seven batches, so the planner does not
silently assume the normal workflow's seven-family input.

Each nonempty AC batch records at most five dispatches:

1. Gather and forward DCT, or one fused forward dispatch.
2. Residual and inverse DCT, or one fused residual/inverse dispatch.
3. Final cost reduction.

`EncodeTransformBatch` records exactly one dispatch for each current transform
implementation. Thus the bound is `5N`, not a geometry-dependent DCT loop count.
Fused variants reduce it to three or four. This bounds dispatch graph allocations
in stage mode as well as dispatch-timestamp mode, including callbacks that finish
recording before a timestamp-capacity overflow is reported.

Stage/group strings come from shared production labels. The kernel-label bound
covers both the shared registered-name limit and the fallback stage label plus
`.dispatch_` and a decimal invocation. The default submission label is the
workflow's `frontend.ac_strategy`; another resolver label must supply its own
maximum length. Profile graph capacities and original/snapshot overlap reuse the
[existing graph planner](resident-profile-storage-planning.md).

The plan excludes device backing, candidate/matrix arrays, caller inputs,
session aggregation, old result graphs, and the already documented driver and
small-control-object exclusions. `working.retained_bytes` is a conservative
inventory bound, not the size of the final returned graph; that is separately
bounded by `profile.resolved_output`.

## Resident AQ context capacity and pointer stability

`ComputeResidentAqProfileInputStoragePlan` extracts the existing reserve recipe
without tightening or enlarging it. With seven supported families, score count
`Q = iterations + evaluate_final_field`, Butteraugli-sink flag `S`, Gaborish flag
`G`, and EPF pass count `E`, each of the two input arrays reserves:

```text
Q * (40 + 4*S + G + E) + 8*(!evaluate_final_field) + 1
```

Iterations are zero through four, EPF passes zero through three, and `Q` must be
nonzero. This allows the terminal unscored frame pass only after an iteration.
Other resident API validation, including required output descriptors, remains
in the evaluator. The actual private resident context type determines its bytes.
The strategy list is shared with preparation/reconfiguration, with unchanged
contents and ordering.

Stages hold pointers into the context array. Reallocation would invalidate
previously stored pointers even if the vectors' final contents looked correct.
`AppendMetalProfileStage` therefore checks equal logical lengths and spare
capacity in both arrays before either push. Exhaustion returns through the
existing typed `resource_plan_exceeded()` exception boundary, invalidates the
evaluator, and prevents submission. There is no fallback growth or retry that
escapes the reservation. The helper only accepts nothrow-copyable contexts;
neither append can allocate after the checks.

This is a capacity invariant, not a report of a reproduced existing dangling
pointer: the current formula has slack and covered recording succeeds. It
prevents a future underestimated recipe from silently turning into a lifetime
bug. Existing successful allocation sizes, command order and numerical kernels
are unchanged.

The AQ input plan does **not** include graph records/labels, wall stages,
snapshot overlap, evaluator/readback backing, score histories or frame output.
One profile stage may dispatch many kernels. Its stage capacity must not be used
as a dispatch or timestamp bound.

## Qualification

Fresh Release and ASan/UBSan trees are `build/resident-submission-plans` and
`build/resident-submission-plans-asan`. The immediate parent is the sealed
`e1abff9` Release build in `build/resident-aq-host-plans`, not an earlier mixed
checkpoint. Tests are enabled, libjxl reference fixtures and compile-time Metal
profiling are disabled. The Release tree includes the encoding benchmarks but
no new performance measurement is claimed here.

The new permanent unit test checks 1,190 AC shapes (including zero/empty batches
and spans larger than seven), 144 resident reserve-policy shapes, overflow/null
and invalid-policy atomicity, and no managed allocation while planning with
zero credit and an armed physical-failure hook. Nine pairs of context/stage
capacities exercise the append guard, checking stable earlier pointers, no
allocation attempt on exhaustion, unchanged arrays, and mismatched-length
rejection.

The existing Metal AC test now also checks bounded recording/resolution on all
seven transform families across six DCT/fusion configurations: 42 cases with
two nonempty batches separated by an empty batch. Profiled GPU costs equal
ordinary GPU costs exactly. Each of the six configurations additionally checks
three physical metadata-allocation failures with recovery and three terminal
underplans. These failures do not commit a submission.

The AQ host test adds 144 real-Metal profile cases spanning 8x8 single-scale and
89x57 padded multiscale inputs, all seven transform families in the larger
fixture, iterations zero through four, final-score on/off, Gaborish on/off and
EPF zero through three. First-use forward/final-CfL stages fit the shared reserve;
bounded repetition and ordinary cached execution preserve exact frames and
scores. Two physical callback-array failures and one typed underplan preserve
outputs, commit no GPU work, and recover after preparing a fresh owner. The
preceding host-plan lifetime and failure suite remains enabled.

Only the AQ **input-array** count is policy-derived in this test. The test sizes
its other diagnostic owners from a first observed graph, using the existing
graph bounds, and adds their envelope separately. This is controlled fixture
validation, not an AQ whole-workflow preflight bound. Stage timestamps are
supported on this M4 Pro; dispatch-boundary timestamp cases are explicitly
skipped because the hardware does not expose that capability.

Both new behavioral test variants also pass against the verified frozen parent
runtime and metallib, compiled with the new test/planner source. The executing
private record layouts are unchanged; header edits add shared definitions,
friend declarations and a free helper, not fields. This confirms the formulas
describe the parent's existing allocation recipe as well as the candidate.

The full Release suites report parent **85/86** and candidate **86/87**. Both
retain exactly the inherited CPU `quantization_pipeline` golden mismatch:
actual `0.24919039011001587`, expected `0.24914586544036865`. It is not fixed or
hidden by this work. Four selected tests—submission plans, Metal AC, AQ host
plans and diagnostic storage—pass three repetitions each in Release and in
ASan/UBSan (12 runs per configuration). Strict warnings pass for the new planner,
new unit test and both expanded Metal tests.

The unsuppressed sanitizer run reproduces the existing metal-cpp Objective-C
nil-reference idiom at `Foundation/NSObject.hpp:112:49`. The successful runs
disable leak detection and suppress only `null:*/third_party/metal-cpp/*`, with
halt-on-error enabled. No GJXL source is suppressed; this is not a clean
unsuppressed sanitizer claim.

All 56 corpus/policy parent-candidate codestream pairs are byte-identical. Three
representative outputs (Kodak17, planter 4K and padded-stress 4K) are independently
decoded with the pinned decoder, with identical decoded PFM hashes. Both builds
pass all 22 pinned conformance fixtures. Sixty common runtime objects and the
entire metallib are byte-identical; only the existing AC and AQ submission units
change code generation, alongside the new planner object. No numerical kernel
or serializer object changes.

Local commands, logs, outputs, build/test/corpus records and artifact hashes are
in `build/submission-plan-qualification/`, driven by its `run.py` and sealed in
`validation.json`. No new physical-footprint, latency, throughput or completed
admission claim follows from these checks. The separate runtime study was not
restarted or changed, and earlier qualification evidence was preserved.

## Remaining composition

The next required count audit is AQ/Butteraugli and initial-quantization
recording: resident quantizer selection, geometry-dependent bitonic sorting,
multiscale perceptual/filter dispatches, and preparation versus policy versus
final-output submissions. The 4096 timestamp-sample limit is not a substitute
for these counts. Remaining host policy arrays, diagnostics/session aggregation,
attempt counts and retained batch results must join the existing device, host,
serializer and submission plans before public whole-workflow admission can make
a hard managed-memory promise. Milestones 4, 5 and 6 remain unfinished.
