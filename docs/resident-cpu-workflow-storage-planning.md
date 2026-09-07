# Native CPU workflow storage bounds

This milestone-4 checkpoint supplies the CPU half of whole-workflow planning.
It does not wire public admission or complete the remaining Metal compatibility,
mixed-backend, input-adapter or retained-batch envelopes. Parent: `eea326c`.

Implementation is in
[`adaptive_quantization_storage_plan.h`](../src/codec/adaptive_quantization_storage_plan.h)
and [`cpu_workflow_storage_plan.h`](../src/codestream/cpu_workflow_storage_plan.h)
and their companion sources. The maximum-error update constant moves from a
local literal into the shared internal policy header. Its value and executing
policy do not change.

## Shared CPU-side AQ policy

The policy bound covers `RunAdaptiveQuantizationPolicy` and its adjusted-input
variant. It includes the mutable field, adjusted-initial or best-feasible field,
the prior evaluator's block map, adjustment/update temporaries, returned fields
and score history, and optional per-evaluation profiles. The current evaluator's
incoming map and other scratch are separate. Maximum-error policy always uses
five updates and six evaluations, independent of the Butteraugli iteration
setting. Butteraugli accepts zero through four updates and always has a final
evaluation on this CPU-side policy path.

Maximum-error updates simultaneously own the current field, a new destination
and the update routine's atomic temporary. Best-feasible copy replacement has
its reviewed reused-vector bound. This is not the device-resident loop's
optional-final-evaluation policy. Current/old caller result and profile owners
are not silently discarded or counted as the same allocation.

## Native CPU evaluator

`CpuAdaptiveQuantizationEvaluator` retains the previous detailed frame and RGB
until the next evaluation succeeds. The new detailed result has a fresh owned
frame, reconstructed RGB and block distances. The policy separately retains the
previous block distances. The complete bound includes that old/new overlap.

Current evaluation work additionally includes raw quantization and inverse
sigma, padded reconstruction, cropped/filtered source-sized images, quantizer
selection, CfL output/scratch, coefficient reconstruction, loop filtering,
atomic color conversion and reduction. The prepared native Butteraugli envelope
includes preparation and comparison, including distorted-psychoimage replacement.
The one-shot variant uses its separate multiscale/reused-scratch bound.
Maximum-error mode has no Butteraugli reference or pixel-distance map, but still
constructs its normal reconstructed RGB output. Child scratch is composed from
the existing reviewed component planners, not a fitted bytes-per-pixel factor.

The evaluator planner also covers diagnostic filter combinations. Those do not
become supported codestream profiles: the serializer keeps its existing initial
profile restrictions. Caller images, strategy grid, sharpness, initial field
and destination planes are separate owners. The returned frame and score/profile
backing remain included until transfer/publication or destruction.

## Complete CPU workflow

The workflow bound adds two padded images (workflow Opsin and preprocessing),
the separate compatibility RGB destination, prepared and compatibility masks and
block fields, sharpness and prior strategy grid, preprocessing/CfL, initial
quantization and CPU AC placement/export. It then composes the complete CPU AQ
and serializer envelopes, optional attempt timings and target-size control.
The native reference is already in the AQ envelope; it is not added twice.

CPU workflows always collect the final AQ score. The CPU workflow profile does
not request the nested per-evaluation AQ profile, so its serializer-profile
bound is separate from the optional profile in the standalone AQ planner.
The current score and codestream output are already in AQ/serializer working
bounds. A target-size search adds at most one earlier best codestream/score
owner, not one per attempt. A single reservation must span the full search.

Forced-CPU Butteraugli, maximum-error and byte/bpp searches are covered, including
current effort, density, compression and thread-count settings. Automatic
requests require proof that **every attempt stays on CPU**. The planner does
not select a backend, and cannot alone bound automatic exact-coefficient
searches whose attempts may cross between CPU and Metal.

All plans are checked, O(1), allocation-free on success and atomic on failure.
They do not replace request validation. Caller inputs, old/published outputs,
input adapters, retained batch output, immutable backend/code, driver internals,
stacks and small control/allocator overhead remain outside these envelopes.
The conservative sum is not expected usage, physical memory or public admission.

## Qualification

Fresh builds: `build/cpu-workflow-plans` and `build/cpu-workflow-plans-asan`.
Frozen parent: `build/resident-last-use`. Evidence and scripts are retained in
`build/cpu-workflow-plan-qualification/`; `validation.json` seals source, build,
decoder, test and output artifacts. Prior builds/evidence are preserved.

The permanent test covers:

- 4,480 allocation-free workflow planning checks, all 64 search limits, invalid/null/
  overflow atomicity, and maximum-error's iteration-independent count.
- 40 isolated shared-policy cases with incoming evaluator output accounted
  separately. Results/profiles remain charged after producer reservation close
  and become uncharged on destruction.
- 80 isolated native evaluator cases, with no serializer allowance in the AQ
  reservation. They compare exact frame coefficient/metadata snapshots, fields,
  scores, maximum-error results and reconstructed pixels against an ample-budget
  oracle. Tiny/expanded/multiscale/padded geometry, mixed strategies, all update
  counts, prepared/one-shot metrics, diagnostic filters and worker thresholds
  are represented.
- 70 complete CPU workflow cases, each with an unprofiled oracle and two bounded
  runs. These compare exact codestreams and summaries, including maximum error,
  size metrics/selection, one/four attempts, automatic CPU routing, high density,
  maximum compression and explicit/automatic CPU participation. Previous public
  output remains alive during the next run without retaining managed charges.
- Pre-admission too-large rejection, typed underplan versus physical allocation
  failure, atomic public output and recovery. The ordinary suite checks that no
  backing escapes into the shared default domain.

The same new test, including the padded-4K mode, is independently compiled
against frozen parent runtime libraries plus the new pure planners. All **65
common runtime object files and the metallib are byte-identical** between parent
and candidate. Only two new planning objects are added. This is not an encoding
algorithm, scheduling, latency or physical-memory improvement claim.

An initial isolated test tried to serialize non-default diagnostic filter
profiles; the serializer correctly rejected them. That test now compares their
full frame representation directly, while complete production workflows retain
exact codestream comparisons. The failed log is preserved. The CPU CLI parity
harness initially omitted its required rate-control argument; its usage failure
is retained, and the corrected runs explicitly select distance or maximum error.

### Large CPU capacity check

The separate 3839x2159 synthetic CPU check uses distance 1.0, effort 1, four CPU
participants and no diagnostics. A fresh-domain oracle plus two bounded calls
produce the same 6,012,490-byte output. The complete plan is 9,897,494,030 bytes;
observed peak managed backing is 3,912,779,204 bytes. These are capacity-ledger
figures, not RSS. The conservative reservation can reject an image whose actual
content-dependent encode would fit a smaller limit. Public admission must expose
that distinction, not present the plan as predicted memory use.

### Regression and decoder gates

Qualification ran on Apple M4 Pro (`Mac16,7`), 48 GiB, 14 logical CPUs,
macOS 15.6 (`24G84`), Apple Clang 17.0.0 and SDK 26.2. The complete frozen-parent
suite passes 88/89 and the candidate passes 89/90. Both have only the inherited
`quantization_pipeline` score mismatch: actual `0.24919039011001587`, expected
`0.24914586544036865`. Neither full suite is entirely green.

The new CPU planner test, `adaptive_quantization_loop`, `maximum_error` and
`rate_control_search` each pass three repetitions in Release and three under
ASan/UBSan. These CPU-only executions need no sanitizer suppressions; leak
detection is disabled (`ASAN_OPTIONS=detect_leaks=0:halt_on_error=1`,
`UBSAN_OPTIONS=halt_on_error=1`). Both new planners and the permanent test also
pass `-Wall -Wextra -Wpedantic -Werror` syntax checks.

All 56 existing corpus/policy parent-candidate codestream pairs and nine
additional forced-CPU pairs match exact hashes. The latter cover efforts
1/4/7/9/10, maximum error, high density, maximum compression and padded 4K.
Three existing Metal outputs and three CPU outputs (effort 7, maximum error and
padded 4K) are independently decoded for both builds with the pinned decoder;
each pair matches decoded PFM hashes. Both builds pass all 22 pinned conformance
fixtures. These output checks supplement the isolated capacity tests; they do
not turn an internal plan into public admission or establish a performance gain.

## Remaining integration

Combine this CPU coverage with Metal exact-coefficient, maximum-error and
maximum-throughput plans and explicit mixed-backend automatic-attempt lifetimes.
Then compose input adapters and aggregate retained batch results, and install
the immutable shared C++/C execution domain with complete upfront reservations,
cache shedding and terminal underplan handling. The resident Butteraugli and CPU
estimators are ingredients for that boundary, not substitutes for it. Milestone
5's remaining reuse/fusion audit and milestone 6's aggregate CPU scheduling and
fairness gates remain unchanged.
