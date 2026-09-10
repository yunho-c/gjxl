# Metal compatibility and mixed-search storage bounds

This milestone-4 checkpoint completes the remaining encoding-policy recipes for
the existing workflow entry points. Parent: `1dd8dbb`. The implementation is
[`compatibility_workflow_storage_plan.h`](../src/codestream/compatibility_workflow_storage_plan.h)
and its companion source. It is internal planning, not public admission.
No existing runtime source or encoding policy changes.

## Forced Metal compatibility paths

The planner covers exact-coefficient Butteraugli and maximum error, resident
maximum error (fully-resident/throughput), and maximum-throughput encoding.
Exact-coefficient and maximum-throughput byte/bpp searches keep one reservation
through all attempts and retain at most one earlier best result. Current public
validation rejects maximum-throughput maximum error and high-density maximum
error/throughput; the planner does not make these combinations supported.

The common host frontend is the conservative native-workflow preparation bound.
It covers padded Opsin/preprocessing, initial fields/CfL, masks, strategy/sharpness
and compatibility destinations. Some Metal modes omit some of these owners;
including the complete shared bound is deliberate conservatism. **The native
CPU AQ evaluator is not included in the forced-Metal bound.** Metal evaluator
storage is instead composed from the executing device layout and separately
reviewed host, policy, transform and owned-frame bounds.

Exact coefficients include:

- Owned original/coding device planes, complete evaluator arenas and prepared
  device Butteraugli, with the same scratch borrowing rules as execution.
- Fixed-geometry host metadata, reconfiguration overlap, quantized/exact
  reconstruction staging, and exact-upload group offsets.
- Packed prepared CPU forward coefficients and their construction scratch,
  raw quantization, inverse sigma, quantizer/CfL/reduction scratch, shared AQ
  policy and the incoming evaluator block-distance map. The prior map is
  already in the shared policy bound.
- Both the current CPU-authoritative coefficient frame and final Metal frame
  export: they coexist during the final evaluation. Prior per-evaluation CPU
  frames are not retained after their evaluation returns.
- Fresh nonresident AC-search device and host staging, placement/export and
  submission input metadata. Exact AC search does not reuse the resident
  prepared-search object.

Resident maximum error uses borrowed resident input, resident AC preparation,
complete reconstruction/filter arenas and the six-evaluation CPU-side policy.
Its adjusted initial field and provisional grid are additional owners. It does
not allocate a Butteraugli reference, execute the fused resident Butteraugli
loop, or return the independent completed-frame lease: final owned-frame
assembly remains part of this compatibility path.

Maximum throughput uses a frame-only evaluator, including its initial-quantizer
sort capacity. Its DCT8 grid, sharpness, fresh initial/strategy/final fields and
pixel mask coexist with compatibility destinations. The plan includes the owned
output frame but no AQ score history, AC candidate search, reconstruction metric
or Butteraugli reference. Device scratch can remain charged in idle pools after
the evaluator is destroyed.

All paths include complete serializer working/output capacity and optional
attempt timings/CPU-side profiles. The public GPU-profiled entry point excludes
these modes; its existing resident Butteraugli graph remains covered by the
separate [resident workflow plan](resident-workflow-storage-planning.md).

## Automatic exact-coefficient searches

Automatic exact-coefficient search preserves the existing per-attempt backend
selection. Eligible targets may use Metal, while other attempts use CPU. Native
reference/preparation and GPU evaluator or idle capacity may coexist across the
transition. The correct simple conservative composition is therefore the **sum
of the complete CPU and Metal bounds, not their maximum**.

This budgets shared owners and the single retained best result twice. Actual
ledger accounting still charges each backing allocation once; this is extra
reservation credit, not duplicate ownership. The plan does not multiply results
or evaluator capacity by the attempt count. The final output bound is the larger
output envelope. A tighter phase-aware mixed bound can be considered separately;
no fitted observed-memory multiplier is used here.

Automatic fully-resident size searches remain all-CPU by existing policy and use
the [CPU workflow plan](resident-cpu-workflow-storage-planning.md). Other
single-target automatic requests use the plan for the selected backend. Planning
does not initialize a backend, change qualification or force a selection.

## Qualification

Fresh candidate: `build/compatibility-workflow-plans`; frozen parent:
`build/cpu-workflow-plans`. Scripts, commands, hashes and retained outputs are in
`build/compatibility-plan-qualification/`; `validation.json` seals the final
source/build/test/decoder artifacts. The new test is also compiled independently
with the new pure planner against frozen parent libraries, including its 4K mode.
All **67 common runtime object files and the metallib are byte-identical**;
only the new planning object is added.

Qualification ran on Apple M4 Pro (`Mac16,7`), 48 GiB, 14 logical CPUs,
macOS 15.6 (`24G84`), Apple Clang 17.0.0 and SDK 26.2. The full parent suite
passes 89/90 and the candidate passes 90/91. Their sole failure is the inherited
`quantization_pipeline` score mismatch: actual `0.24919039011001587`, expected
`0.24914586544036865`. Neither full suite is entirely green.

The final permanent test passes three repeated Release runs and three ASan/UBSan
runs. The unsuppressed Metal invocation reproduces the existing
`third_party/metal-cpp/Foundation/NSObject.hpp:112:49` null-wrapper diagnostic.
The repeated sanitizer runs use only the existing
`null:*/third_party/metal-cpp/*` suppression, with leak detection disabled and
halt-on-error enabled. This is not an unsuppressed sanitizer-clean claim.
The new source/test pass `-Wall -Wextra -Wpedantic -Werror` syntax checks.

Ten parent/candidate codestream pairs match exact hashes, and each output is
independently decoded with the pinned decoder to matching PFM hashes. These
cover Kodak17 exact, maximum-error, maximum-throughput, high-density and maximum
compression paths, plus padded-4K exact, resident maximum-error and maximum-
throughput encodes. Both builds pass all 22 pinned conformance fixtures.
An initial CLI harness command incorrectly combined explicit effort with
`--high-density`; its usage-failure log and command traceback are retained.
Corrected CLI qualification omits explicit effort for that case; the C++ test
separately covers its independently configurable effort/density options.

The permanent test includes 2,800 allocation-free planning checks, all 64 mixed
search limits, invalid/null/overflow atomicity, and 57 complete compatibility
workflow cases. Each runtime case has an ample-budget oracle and two bounded
calls with cold/warm capacity and the previous public result still alive.
It compares exact codestreams, summaries, score counts and requested attempt
timings. Five policy combinations cover tiny, expanded, multiscale and padded
geometry; exact high-density and maximum compression are included.

The 64-attempt automatic search on 128x96 crosses the qualified target interval
(one Metal-eligible attempt, 63 CPU attempts). Its before-serialization trace
verifies the increased Metal AQ backing survives into the following CPU attempt.
All attempts succeed; retained native/GPU state fits the combined reservation.
The test respects actual hardware qualification: on a non-qualified backend it
checks the automatic all-CPU fallback instead of falsely requiring Metal.
First managed-allocation failure and a one-byte underplan are distinguished in
each of the five policy combinations, preserve public output, and permit
recovery. Too-large reservations reject without opening admission. Trimming and
destruction leave no charges, and no tested backing escapes to the default domain.

The separate padded-4K mode uses the deterministic 3839x2159 synthetic input,
effort 1, target 1.0 (or maximum error 0.05 per channel), four CPU participants
and no diagnostics. It compares an oracle and two bounded calls for exact
coefficients, resident maximum error and maximum throughput. Managed-capacity
observations are not physical memory, expected usage or latency measurements.

| Path | Complete planned bytes | Peak managed backing bytes | Codestream bytes |
| --- | ---: | ---: | ---: |
| Exact coefficients | 9,116,576,802 | 3,017,448,868 | 6,012,490 |
| Resident maximum error | 7,622,201,366 | 1,732,440,040 | 11,780,735 |
| Maximum throughput | 6,523,380,690 | 1,161,605,396 | 6,024,785 |

The parent and candidate report the same values. A public hard limit below the
plan can reject content that would actually fit; admission must make this
conservative rejection explicit. No physical-footprint or speedup claim follows
from these ledger measurements.

## Remaining integration

The policy-specific plans still need a single entry-point plan selected from
validated options and actual backend routing. Input adapters and aggregate
retained batch results must be composed before public-domain enforcement.
Whole-domain configuration, upfront admission, cache eviction, terminal
underplan propagation and end-to-end concurrent failure/progress qualification
remain milestone-4 requirements. The remaining last-use/reuse dispositions and
aggregate CPU scheduling remain milestones 5 and 6. These internal estimators
do not complete any of those milestones or establish a speedup.
