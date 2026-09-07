# Shared planning recipes

Execution and admission share allocation recipes where their counts describe
the same operation. Component planners still own the lifetime composition of
those allocations: retained output, reusable scratch, phase maxima, and pooled
backing are not interchangeable.

## Token counts

The AC/DC group tokenizers and group storage planners use
`ComputeAcGroupTokenCounts` and `ComputeDcGroupTokenCounts`. These checked
recipes describe existing reservations, not expected emitted token counts.
Execution passes the validated anchor count; preflight uses the block count
as its anchor upper bound.

`ComputeTokenizationStoragePlan` sums these counts during its existing traversal
of at most four rectangular group classes: interior, right strip, bottom strip,
and corner. It returns aggregate AC/DC token bounds and the largest individual
AC group/DC stream. The serializer consumes those results for entropy tasks and
emission scratch instead of maintaining independent whole-frame formulas.

Aggregate token counts describe one encoding candidate. Map/order variants
multiply retained owners where those variants coexist; they do not multiply
the token count of an individual entropy task. Group scratch, joined template
and context phases, serializer tasks, and writer lifetimes remain separate.

## Frontend dispatch

`codec/frontend_dispatch_internal.h` owns the participant policies for color
conversion, initial quantization, and prepared forward transforms. Runtime
supplies task count, work size, explicit CPU limit, and hardware availability.
Planners use the same policy with the stage cap as a hardware upper bound;
they do not sample hardware or assume domain worker availability.

The shared thread-vector recipe uses the final admitted participant count at
runtime. Explicit CPU limits and managed participation include the caller;
unmanaged automatic execution can spawn every participant. Serial execution
allocates no thread vector. Planning therefore reserves `participants - 1`
thread objects for explicit limits and conservatively reserves `participants`
for automatic execution. Per-participant row arrays and per-task status arrays
retain their own counts.

Color conversion owns its atomic temporary image and dispatch storage.
Initial quantization joins and destroys row-phase workers before allocating
finish-phase fields; its envelope remains the maximum of those two phases.
Prepared forward transforms retain their coefficient/index output alongside
per-tile lists and dispatch storage. Scheduling, nested serial fallback,
participant admission, launch failure handling and joining are unchanged.

## Workflow publication

`codestream/workflow_publication_storage_plan.h` returns the common score,
optional timing, search-control, retained-best and output owners. It performs
checked, allocation-free planning on success and publishes the plan only after
all checks pass. A one-attempt search still has search-control storage; only
multi-attempt searches retain a previous best result. Frame-only compatibility
passes zero score count. GPU-profile output remains a resident-route addition.

| Route | Composition retained by the caller |
| --- | --- |
| CPU | AQ already includes current scores; serialization already includes current codestream output. Timing, search control and the previous best remain separate additions. |
| Compatibility Metal | Preserve the complete evaluator/AC/serializer envelope and frame-only score behavior. |
| Resident Metal | Add common owners to the existing search/completion phases. Keep pooled backing and retry overlap explicit; take the existing phase maximum. |

The shared helper does not produce a complete working envelope. Moving an owner
between phases requires a separate lifetime review, even when its byte count
comes from a shared recipe. All new helpers remain outside the installed header
manifest; public interfaces and allocation-owner definitions are unchanged.

## Validation and reservation changes

The cleanup is compared with `3b758f8` in an isolated checkout, excluding
concurrent input-handoff work. A 1,440-record plan matrix covers small, threshold,
padded, thin, 1080p and 4K extents; CPU limits 0/1/2/8/12; serializer policies;
CPU, resident and compatibility routes; timing, profiles and search attempts.
Token/serializer/publication bounds and resident phase envelopes match exactly.
Affected color/forward bounds decrease by one `std::thread` object (8 bytes on
the audited toolchain); complete-workflow bounds decrease by 8 or 16 bytes in
this matrix. Initial quantization's unchanged finish-phase bound dominates its
smaller row phase. These are reservation changes, not measured latency or
physical-memory improvements.

`planning_recipes` checks dispatch thresholds and caps, caller participation,
publication/search ownership, overflow atomicity, execution within the tighter
color reservation and admission rejection below that reservation. Existing
group, frontend, workflow, worker-launch, concurrency and installed-interface
tests remain the integration gates. Frozen group-formula tests and retained
pre-change frontend/serializer oracles provide independent comparisons.

Both C++20 and C++23 Release suites pass 118 of 119 tests, including the new
recipe test, allocation-failure/worker-launch cases, admission, batch lifecycle
and installed consumers in both language modes. The sole failure is the existing
`quantization_pipeline` score mismatch, reproduced on the retained baseline
with the same actual/expected values. All 56 baseline/candidate encodes are
byte-identical, all three pinned decoded-pixel comparisons match, and all
22 pinned conformance fixtures pass. The independent frontend and serializer
binary oracles also match their pre-change captures.
