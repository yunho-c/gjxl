# Whole resident workflow storage planning

This checkpoint composes the previously qualified components across a complete
production Metal resident Butteraugli encode, from borrowed linear RGB through
retained codestream output. It also covers forced-Metal target-byte and
target-bpp searches. It is an internal estimator and qualification checkpoint,
**not public admission or completion of milestone 4**.

The original checkpoint is `f33fe0b`, with parent `b67b45e`. Implementation is in
[`resident_workflow_storage_plan.h`](../src/codestream/resident_workflow_storage_plan.h)
and its companion source. Effort-to-AQ-iteration resolution is shared with the
executing workflow; the policy itself is unchanged. Search-interval bounds use
the actual private `SearchInterval` type beside the search implementation.

The subsequent [last-use release checkpoint](resident-last-use.md) changes
`working` from a conservative sum to the maximum of AC-search and
completion/serialization envelopes. It retains AC in earlier retryable attempts
and continues charging idle AQ/input/metric buffers. The qualification and
numerical bounds below remain the original `f33fe0b` record; current measurements
and phase semantics are in the linked checkpoint.

## Supported boundary

The plan starts with a fresh `PreparedWorkflow`, at fixed source/padded geometry,
and spans every attempt until outer publication. It includes resident input
backing, AC search, prepared AQ/Butteraugli, orchestration fields, completed-frame
backing, CPU serialization, selected-result retention and requested diagnostics.
At the original checkpoint the prepared evaluator and AC scratch survived into
the CPU tail; its bounds accounted for that lifetime without shortening it.

Fully-resident and throughput Butteraugli policies, all current efforts, high
density where supported, and both serializer compression modes are covered.
Automatic single-target encoding requires Metal to have already been selected;
the estimator does not select or qualify the backend. Automatic resident
target-size search runs on the CPU and is explicitly rejected by this planner.
So are CPU, exact coefficients, maximum error and maximum throughput. Their
remaining plans are requirements of the broader admission effort, not removed
from scope. GPU diagnostics retain their existing restriction to forced-Metal
single-target resident Butteraugli encoding.
The existing GPU-profiled workflow also requires its CPU profile output;
requesting GPU diagnostics therefore implies the serializer's profiling bound.

Caller image backing and old/published results are excluded. Input conversion
adapters, retained batch results, idle capacity belonging to other jobs and
public-domain configuration require separate composition. Immutable backend/code,
driver internals, allocator headers, thread stacks and small runtime control
objects retain the existing managed-boundary exclusions. The bound is not RSS.
Successful planning performs no backing allocation and no backend/image access.
It does not replace ordinary request validation. Invalid/unsupported shapes and
overflow leave the supplied plan unchanged.

## Ownership composition

The six device contributions are resident input, AQ persistent arena, AQ staging
arena, Butteraugli arena, AC-search buffers and independent completed output.
AQ borrows the resident source/coding planes. Butteraugli borrows its nine
transient planes when the executing sink/filter geometry supports that sharing;
those planes are not charged again. Bounds use all supported transform families
and block count before CPU placement has selected actual anchors.

Host composition adds:

- The complete AQ prepared/reconfiguration/operation bound, including lazy final
  metadata, invariant-CfL replacement and adjustment/readback work.
- Prepared sharpness and initial/strategy fields, the retained selected grid,
  the provisional preparation grid, and adjusted policy input. Initial CfL output
  is separate from evaluator-owned readback; searches include both generations
  during replacement.
- AC prepared candidates/matrices/cost tables and CPU placement/export, with
  reused-owner bounds for multi-attempt searches. Its atomic new grid is counted
  alongside the earlier selected grid.
- AC submission input metadata, the completed host snapshot and destination
  upload temporary, and the complete serializer working envelope.
- The fresh score history, optional attempt-timing array, and GPU diagnostics.

The original `working` conservatively sums these phase peaks, including the serializer's own
output bound. `output` separately exposes codestream, score, timing and diagnostic
backing before publication; it must not be added to `working` again. The
`retained_bytes` fields are owner-capacity bounds, not a claim that all listed
owners remain live at one common phase boundary.

One reservation must span the whole job. It is invalid to close an attempt's
reservation and subtract its still-owned preparation bytes from the next job's
credit. Existing cache rules discard incompatible/oversized backing before
reuse, and same-job cache transfers retain their allocation-owned charges.
Automatic eviction of other jobs' idle capacity before FIFO admission remains
public-domain integration work.

## Parent diagnostic session

A fresh successful resident workflow emits five child submissions: reference
preparation, initial quantization, AC search, field adjustment and resident AQ.
Resident input preparation does not produce a GPU profile graph. Six
orchestration wall records, four AC wall records and three completed-output wall
records give a 13-record upper bound. Stage and dispatch counts come from the
individual submission planners, including geometry-dependent reductions and
all work recorded in stage mode.

The full parent graph bound covers all retained child graphs and outer-vector
growth. One additional maximum child working bound covers current callback
inputs, original/snapshot overlap and child/parent wrapper overlap. The parent
does not discard earlier graphs to make room for a later child. Failure before
append is also covered by the current-child term. Label bounds include generated
fallback IDs, not only the registered kernel-ID limit. No observed graph or
timestamp-buffer limit is used to size the reservation.

## Target-size attempts

Search is serial and keeps at most one earlier best codestream and summary
while evaluating the next candidate. Add that retained result once, not once per
maximum attempt. Score arrays and fixed-geometry preparation also do not grow
with the number of attempts. Attempts can change matrix scales, replacing the
evaluator; the prior owner is reset before preparing its successor.

The timing vector reserves the maximum attempt count. The search starts with
one interval even if its first endpoint succeeds. Endpoints do not split it;
each later midpoint erases one interval and appends two. Therefore maximum
logical interval storage is `max(1, maximum_attempts - 1)` records, using the
reviewed growing-vector backing bound. Underplan errors remain terminal; normal
candidate failures retain the existing search policy. Memory pressure never
changes effort, backend choice, candidates or result selection.

## Qualification

Fresh Release trees are `build/resident-workflow-plans` and
`build/resident-workflow-plans-asan`; the frozen parent is
`build/resident-aq-profile-plans`. The qualification harness and evidence are in
`build/workflow-plan-qualification/`. `run.py` records commands, exact output
hashes and test results and seals them with source/build hashes in
`validation.json`. Earlier evidence and builds are preserved.

The permanent whole-workflow test checks 2,240 policy/geometry/profile shapes
under zero-credit and armed-allocation-failure conditions. It checks all 64
attempt limits, including actual interval growth through all requested attempts,
and atomic invalid/unsupported shape handling. Runtime coverage comprises 95
complete-workflow cases, each with a separate unprofiled oracle and two
plan-bounded cold/warm executions. These include tiny/expanded, single-scale,
multiscale, padded and 3839x2159 inputs; all iteration counts; optional final
scores; no/CPU/GPU diagnostics; explicit and automatic thread counts; throughput,
high density and maximum compression; both size metrics and selection policies.
Complete bytes and summaries match, and recorded graphs fit preflight counts.

Additional tests reject an oversized reservation before work, distinguish typed
underplans from physical backing failures, preserve old outputs and recover.
An owned size-search result stays charged after the producer reservation closes
and becomes uncharged only at explicit publication. These are harness-driven
reservation tests; public entry points do not yet automatically reserve this plan.
The existing diagnostic test now uses this bound instead of a manual 64 MiB
allowance; it covers all 406 resident diagnostic allocation-failure positions,
plus its separate primitive tests and recovery checks.
An initial expanded test incorrectly supplied a null CPU-profile output when
requesting GPU diagnostics. The wrapper rejected it before allocation. The test
now follows the existing interface and the planner explicitly includes the
implied CPU profiling; the failed log is retained with the qualification.

Qualification platform: Apple M4 Pro, 48 GiB, 14 logical CPUs, Mac16,7;
macOS 15.6 (24G84), Apple Clang 17 and SDK 26.2. Both trees use Release,
libjxl-reference builds disabled and compile-time Metal profiling disabled;
the Release tree also builds benchmarks. The sanitizer tree uses
`-fsanitize=address,undefined -fno-omit-frame-pointer` and matching linker flags.
Stage-boundary timestamps are available. Dispatch-boundary sampling is not and
is explicitly skipped, not newly qualified.

Full Release suites finish **87/88 parent** and **88/89 candidate**, with only the
same inherited CPU `quantization_pipeline` golden mismatch: actual
`0.24919039011001587`, expected `0.24914586544036865`. Whole-workflow, Metal
diagnostic, Metal resource-budget and rate-control-search tests pass three
repetitions each in Release and ASan/UBSan: twelve runs per configuration.
Strict warnings pass for the new planner/test, rate-control implementation and
expanded diagnostic test.

The unsuppressed sanitizer run reproduces metal-cpp's Objective-C nil-reference
idiom at `Foundation/NSObject.hpp:112:49`. Successful runs use
`ASAN_OPTIONS=detect_leaks=0:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:suppressions=.../ubsan.supp`, where the only
suppression is `null:*/third_party/metal-cpp/*`. No GJXL source is suppressed;
this is not an unsuppressed sanitizer-clean claim.

All 56 parent/candidate corpus/policy codestream pairs match exact hashes.
Kodak17, planter 4K and padded-stress 4K are separately decoded with the pinned
decoder and match decoded PFM hashes. Both builds pass all 22 pinned conformance
fixtures. Sixty-two common runtime objects and the complete metallib remain
byte-identical. Only existing `workflow.cpp` and `rate_control.cpp` objects change,
plus the new planner object; no numerical shader, GPU runtime or serializer
algorithm object changes. These checks do not establish a latency or footprint
improvement.

### Capacity bounds versus observed use

A separate fresh-domain probe uses the same deterministic synthetic input recipe,
fully-resident effort 7, target 1.0, four CPU participants and no diagnostics:

| Source | Complete planned bytes | Observed peak managed backing bytes |
| --- | ---: | ---: |
| 89x57 | 27,482,700 | 15,367,607 |
| 3839x2159 | 7,952,975,062 | 2,933,527,656 |

The larger plan comprises 2,680,353,820 device bytes, 67,347,528 host frontend
peak-bound bytes, 5,205,273,698 serializer peak-bound bytes and 16 score bytes.
These are conservative, content-independent capacity bounds, not predicted
allocations. The observation is the ledger's peak of live plus idle backing,
not physical memory or a timing result. A configured limit can reject a job whose
actual content-dependent run would fit. Admission documentation must expose
that conservatism; it must not relabel the bound as expected memory usage.
The probe source, binary and JSONL output are retained with the qualification.

## Remaining work

Complete the other workflow/backend policy envelopes, including CPU host-policy
state and compatibility outputs, and combine input adapters and retained batch
results. Then wire shared C++/C execution domains, full upfront reservations,
cache shedding and atomic failure paths through all entry points. The estimator
alone is not admission. Milestone 5's explicit last-use/reuse dispositions and
milestone 6's aggregate CPU coordination, fairness and throughput gates remain
required. No new speedup, physical-footprint reduction or whole-domain scheduling
claim is made here.
