# Resident resource accounting and admission

This is the implementation record for milestone 4 of
[resident execution](resident-execution.md), based on the integrated checkpoint
`ec4d4c5`. **Milestone 4 is not complete.** The tested
[capacity-domain primitive](../src/core/resource_budget.h) now has
[real Metal allocation attachments and all-pool trim](resident-metal-accounting.md).
The [host attachment checkpoint](resident-host-accounting.md) adds writer/image-plane
backing and joined-worker propagation; host coverage is still partial and
whole-workflow admission is not implemented yet.
The inventory and decisions below retain the CPU tail and batch-result requirements.

## Source-backed ownership inventory

Capacity means allocation backing, not logical image size or physical footprint.
A borrowed view is not another allocation. The current last-use boundary and
the current destruction boundary are often different.

| Owner / storage | Backing and aliases | Last consumer / current retention |
| --- | --- | --- |
| Caller source and previous caller output | Borrowed source planes; caller's previous codestream can coexist with the candidate | Caller owns these; neither is a new encoder allocation. Exclude them explicitly from the managed limit. |
| [Prepared resident input](../src/gpu/ops/resident_input.h) | One resident original-RGB/padded-Opsin preparation; AQ and AC search borrow device views | `PreparedWorkflow` owns it before its borrowers so it is destroyed after them. Retained through attempts and CPU serialization. |
| [Prepared host quantization](../src/codec/quantization_pipeline_internal.h) | Block/tile quantization, sharpness, strategy and CfL arrays; optional preprocessed Opsin and reference storage | Retained through prepared attempts. Ordinary resident preparation leaves the full-resolution host pixel mask empty; explicit diagnostics materialize it. |
| [AC strategy search](../src/gpu/ops/ac_strategy_search.cpp) | Per-family candidates, matrices, costs and their device buffers; CPU cost arrays; two maximum-sized packed-transform buffers plus rate scratch | GPU completion and CPU placement finish using scratch before AQ. `PreparedAcStrategySearch` currently retains it through the workflow for possible reuse. Resident input/mask views are borrowed. |
| [AQ persistent/staging arenas](../src/gpu/metal/metal_aq_evaluation.cpp) | Geometry-planned planes, metadata, retained coefficients, reconstruction/filter images and small reductions | Prepared evaluator owns arenas and waits before returning them. Repeated evaluations reuse them. Two filter images and gathered pixels can be borrowed by Butteraugli; charge their physical AQ backing once. |
| [Butteraugli arena](../src/gpu/metal/metal_butteraugli.cpp) | Owning scratch plus borrowed AQ planes; reference psychoimages/masks have image-specific contents | Per-image preparation rebuilds references. Destruction precedes returning borrowed AQ backing. A successful operation may return only its owning capacity to the backend cache. |
| [Completed frame](../src/gpu/metal/metal_aq_evaluation.cpp) | Independent final AC/destination allocation and host block metadata; serializer borrows a read-only frame view | Survives evaluator/backend destruction. Final AC capacity is `3 * AC_group_count * 65536 * sizeof(int32_t)`, plus the destination table. Lifetime ends with the last owning encoding artifact, not GPU completion. |
| [CPU serializer](../src/codestream/encoder.cpp) | DC tokens; AC values/contexts/templates and populations; coefficient-order scratch; entropy models/search temporaries; section writers and candidate codestreams | Different effort/entropy paths retain different candidate sets. Worker-local scratch multiplies by participating workers. Final sections and candidate bytes can coexist with input frames and prepared GPU storage. |
| [Idle Metal pools](../src/gpu/metal/metal_backend.cpp) | One AQ arena per persistent/staging/resident-input class, plus one Butteraugli arena per backend | Idle capacity is volatile, not necessarily physically absent. Existing per-arena AQ caps and the process-wide Butteraugli idle cap are independent; active/completed/CPU storage is not covered. |
| [Batch results](../src/codestream/batch_workflow.cpp) | Result vector retains every successful codestream until the entire batch returns | In-flight worker count does not bound accumulated result bytes. Previous caller results remain caller-owned until atomic publication. |

The lowest Metal allocation choke point is `MetalBackend::Allocate`; its current
`MetalBuffer` owns the Metal handle independently of backend lifetime. The static
DCT basis buffer is created during backend construction outside this entry
point. `DeviceScratchArena` owns one buffer and creates checked non-owning slices;
summing its planes as additional allocations would double-count shared storage.

## Integration decisions

These decisions define the intended workflow integration; the public API and
allocation adapters are not wired by the foundation commit.

- **Domain/configuration:** an immutable, shared execution-domain handle carries
  managed-capacity and aggregate CPU limits. A C++ workflow option and equivalent
  opaque C handle will permit explicit domains. A null handle selects one shared
  production domain, not a fresh budget per encode, batch, context, or explicit
  backend. Do not reconfigure a live domain in place.
- **Defaults:** zero managed bytes means no configured hard memory cap, while
  still collecting accounting. Do not invent a fixed fraction of available RAM.
  The eventual aggregate automatic CPU limit is at least one and based on hardware
  concurrency; the existing per-image CPU setting remains an upper bound, not
  another independent domain allowance. CPU enforcement belongs to milestone 6.
- **Managed boundary:** encoder-owned input staging, prepared host/device arrays,
  AC/AQ/Butteraugli scratch, completed frames, CPU serializer temporaries, candidate
  bytes and retained batch results are in scope. Caller input/previous output,
  results after publication, code/pipeline objects, immutable backend setup,
  command/driver internals, thread stacks and small control-object/allocator
  overhead are explicit exclusions. This is not a process-RSS guarantee.
- **Reservation estimates:** use checked geometry/option planners before heavy
  preparation. Extract the existing plane-layout planners so planning and actual
  allocation share formulas. Combine the maximum simultaneously live planned
  device storage with conservative host/token/model/writer capacity bounds for
  the selected policy and worker count. Include replacement-before-release,
  completed output, and retained retry candidates. Do not fit a bytes-per-pixel
  coefficient to benchmark RSS and call it a bound.
- **Progress:** reserve the complete managed work envelope before admitting an
  image. An allocation within it must never wait for another reservation. An
  underestimated plan fails atomically with an explicit error; it neither grows
  silently nor waits while holding resources another job needs. A plan above the
  hard limit fails before preparation. FIFO waiting prevents repeated small jobs
  from starving a larger admissible request.
- **Caches:** backings keep their original domain while idle. Transfer a reusable
  backing into an admitted job in the same domain; account capacity, not just the
  smaller requested layout. Shed an oversized or cross-domain cache entry if it
  cannot fit the target plan. The controller must evict idle storage before
  waiting when cache charges obstruct admission; the primitive cannot free memory
  it does not own. Trim must cover all managed idle pools, not just Butteraugli,
  without allowing pre-trim active leases to repopulate them.
- **Retained batch results:** reserve a conservative aggregate result-capacity
  envelope before starting a batch, separate from worker working-set envelopes.
  Charge each accumulating result against it. If that envelope plus the required
  largest-image working set cannot fit, fail the batch atomically before starting,
  rather than filling the budget with results and deadlocking later jobs. Report
  this conservative rejection distinctly; no hidden spilling, dropped requests
  or new streaming-output API. Transfer results out of managed accounting only
  at the existing public ownership handoff.

Concrete plane/token/model bounds, the API adapters, and allocation coverage
still require implementation and tests before workflow enforcement can land.
In particular, excluding all CPU allocations or retained batch results to make
a GPU-only limiter pass would not satisfy this design or the parent roadmap.

## Primitive and invariants

`ResourceBudget` copies share one immutable allowance. It provides FIFO,
cancellable `Reserve` and nonblocking `TryReserve`. A try request cannot bypass
an existing waiter. Too-large plans return `OutOfMemory`; temporary occupancy
and cancellation return `Unavailable`. Invalid outputs remain unchanged.

`ResourceReservation` is a move-only complete-work envelope. Before creating a
backing allocation, `PrepareAllocation` consumes credit within that envelope and
returns a pending `ResourceAllocation` ticket. Only successful physical allocation
may `Commit` it. A failed allocation destroys the pending ticket and restores
credit. A backing owner must free its allocation before releasing the ticket;
borrowers and subviews hold no extra tickets.

The synchronized snapshot partitions protected capacity as:

`committed = live backing capacity + idle backing capacity + reserved unbacked capacity`

Pending allocation capacity is a subset of reserved unbacked capacity. Requested
live bytes are at most live capacity; they describe an allocation request, not
the sum of every alias's logical image extent. Per-owner-class counters reconcile
to the total. Peaks describe protected/owned capacity, not OS physical pages.

At a proven last-use boundary, `ReduceCapacity` can release unneeded plan capacity
but cannot grow the envelope or undercut outstanding tickets. Closing a
reservation releases all its unused capacity. Its live, idle, or pending
tickets remain charged independently until freed, even after the domain wrapper
is destroyed. Cache transfer into a new open reservation consumes that job's
credit without duplicating the physical backing. Transferring from another open
job returns credit to the source; transferring from a closed producer removes
the redundant protection of the destination's previously unused reservation.
No reservation grows after admission, so there is no allocation-time wait cycle
inside this primitive.

The primitive does not own buffers, evict caches, propagate worker contexts,
estimate memory, cap CPU participation, or alter encoding. It cannot establish
whole-workflow limits until its integration coverage is complete.

## Initial reuse/lifetime audit for milestone 5

| Candidate | Current disposition and required evidence |
| --- | --- |
| Shared AQ/Butteraugli planes, smaller final staging, omitted host mask and deferred metadata | Implemented in milestone 3; preserve its parity/lifetime gates and count the shared physical backing only once. |
| AC-search packed/rate scratch retained after placement | Implement a source-aware last-use release opportunity after checking prepared-run reuse/retries. Measure complete-call and peak effects; do not discard reusable source state indiscriminately. |
| Full evaluator/reference storage retained during CPU serialization | Audit retry and diagnostic consumers, then release what has no future consumer. Independent completed-frame ownership permits this but does not itself release the evaluator. |
| Three existing AQ pools plus Butteraugli cache | Common Metal accounting and generation-aware all-pool trim are implemented and measured in the Metal attachment checkpoint. Domain-wide automatic eviction/admission is still required. |
| Intermediate AQ coefficients versus serializer layout | Keep distinct where reconstruction/AQ consumers require their current layout. The final-output destination optimization is already implemented; another layout change needs measured end-to-end benefit. |
| Small copied completed-frame metadata | Intentionally retained: separates output from much larger evaluator lifetimes. Do not make metadata zero-copy by keeping the evaluator alive. |
| Exact shared candidate calculations / further kernel fusion | Inventory against the integrated profile. No unqualified arithmetic reordering or candidate pruning. A documented measured rejection or reasoned deferral is valid; this row is not yet a completed audit. |
| CPU token/model/writer overlap and storage growth | Inventory by balanced/high-density/exhaustive policy while adding allocation coverage. Preserve model search, tie rules, and exact output; resource pressure cannot silently reduce search. |

## Foundation validation and remaining gates

The permanent `resource_budget` test covers requested-versus-capacity counters,
pending/committed/idle transitions, domain sharing and isolation, cache transfers,
producer/domain destruction, moves, malformed requests, undersized plans,
exception unwinding, maximum-size accounting, FIFO order, head/interior/tail
cancellation, retained-output admission pressure, and eight concurrent callers.
An independent state model recomputes per-class/total protection after 20,000
deterministic operations, including moves, shrinking and cross-job transfers.
This is primitive testing, not Metal pressure testing or throughput qualification.

The final primitive passes 50 Release repetitions, 25 ASan/UBSan repetitions,
and 25 separate ThreadSanitizer repetitions. Sanitizer runs use no suppressions;
this CPU-only target does not include Metal's vendor wrappers. These checks do
not establish physical-footprint behavior or platform leak-detector coverage.
The source also passes AppleClang 17 `-Wall -Wextra -Wpedantic -Werror` checks.
The fresh final full Release suite passes 64/65: the additional resource test is
green, and the sole failure remains the reproduced CPU `quantization_pipeline`
golden mismatch (`0.24919039011001587` versus `0.24914586544036865`).

Release artifacts are in `build/resident-resources`, using the same build options
as milestone 3. The final full log is `final-ctest.log`; repeated primitive
results are in `resource-budget-repeat.log`. Separate RelWithDebInfo builds
`build/resource-sanitize` and `build/resource-tsan` use, respectively,
`-fsanitize=address,undefined -fno-omit-frame-pointer` and
`-fsanitize=thread -fno-omit-frame-pointer`. Their final logs are
`resource-budget-final.log`. The invocation is:

```sh
ctest --test-dir build/resident-resources -R '^resource_budget$' --repeat until-fail:50 --output-on-failure
UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir build/resource-sanitize -R '^resource_budget$' --repeat until-fail:25 --output-on-failure
TSAN_OPTIONS=halt_on_error=1 ctest --test-dir build/resource-tsan -R '^resource_budget$' --repeat until-fail:25 --output-on-failure
```

At the `b6bc044` foundation checkpoint, the header was consumed only by its
permanent test and production encoders were unchanged. The subsequent Metal
attachment has its own [qualification record](resident-metal-accounting.md).
Neither checkpoint claims a whole-encoder managed-memory bound; milestone 3's
frozen combined baseline remains intact.

Milestone 4 still requires the remaining host allocation attachments and shared planners,
CPU serializer coverage, public domain configuration/propagation, automatic cache eviction,
retry and batch-result integration, end-to-end failure/progress tests, and
physical peak/idle/post-trim measurements. Milestone 5 still requires the audited
opportunities' final dispositions and measured gates. Milestone 6 still requires
aggregate CPU scheduling and actual latency/throughput qualification. None is
marked complete by the foundation's unit tests.
