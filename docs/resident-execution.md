# Resident execution architecture

## Charter and motivation

Preserve GJXL's current encoding decisions while making data representation,
storage lifetimes, resource admission, and CPU/GPU scheduling coherent across
the complete Metal resident workflow.

The architectural problem is broader than slow kernels. Independently designed
stages can require owned results, incompatible layouts, long-lived scratch, and
separate resource decisions. Those boundaries make allocation, conversion, and
retention compulsory even when the consumer only needs access to existing data.
Unified memory does not eliminate those traversals or establish safe lifetimes.

The organizing questions are: what data is needed, in which representation,
until when, and by whom? The goal is a GPU-centric heterogeneous encoder, not a
GPU-only encoder. CPU AC placement, tokenization, entropy coding, and emission
can remain appropriate parts of the design.

This is the policy-preserving architecture roadmap under the broader
[Metal encoding performance roadmap](metal-encoding-performance.md).
[Metal AQ](metal-aq.md) remains authoritative for numerical/residency contracts,
and [codestream documentation](codestream.md) for the supported bitstream profile.
[Resident frame handoff](resident-frame-handoff.md) retains the detailed design
and qualification record for completed milestones 1 and 2; its measurements are
not evidence for the unimplemented milestones below.

Development branch: `refactor/resident-execution` (originally
`refactor/resident-frame-handoff`). The worktree remains
`../gjxl-resident-frame-handoff`.

## Scope and current status

The proposal numbers below refer to the original architecture discussion, not
the implementation milestone numbers used later in this document.

| Proposal | Scope | Status after preparation integration (milestone 3) |
| --- | --- | --- |
| #3: Stable coefficients and frame views | Included | Principal handoff complete in `ca440d1` and `dabe129`: ownership-independent consumers, direct final AC destinations, independent completed-output lease. |
| #4A: Reuse, fusion, shorter intermediate lifetimes | Included, subject to numerical and end-to-end gates | Resident fusion is in the `4ea12ab` base. Shared scratch, deferred preparation, and volatile capacity caching from `perf/metal-preparation` are integrated and jointly qualified. Remaining opportunities need an explicit inventory. |
| #4B: Screening, pruning, selective refinement | Separate policy track | Deferred; not an unfinished requirement of this structural refactor. |
| #5: Ownership, resource budgets, scheduling | Included | Output ownership is established. Existing AQ leases and batch overlap are foundations, not coordinated whole-workflow admission. |

The integrated preparation branch contributes two commits:

- `e1010fe`: share nine transient F32 planes between AQ and Butteraugli, reduce
  staging capacity, defer provisional metadata, and avoid unrequested host masks.
- `306f153`: bounded volatile Butteraugli capacity caching and explicit trimming.
  Its per-backend and process-wide idle-cache limits do not cover active storage,
  existing AQ pools, or completed output.

Its original design records are [shared scratch and preparation](metal-preparation-integration.md)
and [volatile caching](metal-volatile-preparation-cache.md). Their historical
measurements remain separate. The [joint integration record](resident-execution-integration.md)
qualifies the combination against `a747fca` with preparation at `306f153`.

### Invariants and exclusions

- Preserve AQ iteration/stopping policy, AC candidate sets and tie rules,
  quantization decisions, coefficient-order policy, entropy behavior, and
  rate-control/backend-selection behavior. Memory pressure must not silently
  lower effort, omit candidates, or select a different encoding policy.
- Preserve the public owned-frame and diagnostic paths, existing validation,
  deterministic output, and atomic failure behavior. Existing nonresident paths
  remain supported; this effort does not promise to migrate every mode to leases.
- Storage, ownership, and scheduling changes require exact parent/candidate
  decisions and codestream bytes for the same backend/options. Arithmetic reuse
  and fusion must also pass existing numerical gates and downstream exactness
  checks. A changed decision requires a separately qualified proposal, not a
  widened tolerance hidden in this refactor.
- Do not require zero copies everywhere. Small independently owned metadata can
  release much larger scratch sooner. Intermediate AQ coefficients need not use
  the final serializer layout when they have different consumers.
- #1/#2 quality-time policy redesign, predictive AQ, #4B search reduction, GPU
  entropy coding, a new general computation-graph framework, and cross-image
  kernel fusion are not prerequisites or completion criteria.

## Milestones and dependencies

Milestones 1 and 2 retain the numbering in the handoff record. The expanded
roadmap adds integration before coordinated resources and scheduling. Each
milestone should remain independently reviewable and record its actual baseline.

### 1. Ownership-independent consumers — complete

`ca440d1` introduced the read-only frame interface and adapted owned frames.
Serialization, coefficient orders, block contexts, and tokenizers consume the
view without requiring a particular allocation owner.

### 2. Completed Metal output — complete

`dabe129` writes final AC coefficients directly into serializer-compatible shared
storage and publishes an exclusive completed-frame owner after GPU completion
and validation. It retains neither backend nor evaluator/scratch. Small metadata
snapshots intentionally separate output lifetime from preparation lifetime.

Completion here does not mean minimum peak memory: the workflow can retain a
prepared evaluator for reuse independently of the completed frame. The recorded
milestone-2 footprint is effectively unchanged from milestone 1.

### 3. Integrate preparation and handoff — complete

The [integration record](resident-execution-integration.md) covers semantic
reconciliation, joint lifetime tests, exact corpus/policy bytes, pinned decoder
and conformance checks, and fresh complete-call/physical-footprint measurements.
Both parent and candidate retain only the documented CPU golden mismatch
(63/64 Release tests); the targeted sanitizer suites pass 3/3 with the recorded
vendor-header limitation. The measured peak falls, but one-second idle footprint
rises with caching; this is not a whole-encoder memory budget.

Deliverables:

- Reconcile `e1010fe` and `306f153` with completed-output generation. Preserve
  distinct ownership of borrowed scratch, cached capacity, and completed frames.
- Bring their design/qualification records into the integrated branch and record
  exact integrated revisions. Preserve historical results as historical.
- Establish a fresh combined correctness, latency, and memory baseline before
  attributing further improvements to resource or scheduling changes.

Acceptance:

- Exact coefficients, scores, and bytes across the existing corpus/policy cases;
  pinned decoder/conformance coverage and inherited failures reported explicitly.
- Joint lifetime tests: changed images and dimensions, evaluator reconfiguration,
  serialization during evaluator reuse, backend/evaluator destruction, forced
  idle reclamation, trim during active work, and failed preparation/submission.
- Complete-call timing including teardown; peak and idle footprint with the
  backend alive. Concurrent correctness stress is not throughput qualification.

### 4. Coordinated resource accounting and admission — pending

Depends on milestone 3's integrated ownership model. Start with a small explicit
set of resource classes and reservations, not a general graph runtime.

The [resource implementation record](resident-resources.md) contains the
source-backed ownership inventory, integration decisions, and tested primitive
for reservation/allocation lifetimes and FIFO admission. The
[Metal attachment checkpoint](resident-metal-accounting.md) adds real backing
charges, domain-aware cache transitions, and all-pool trim with physical-memory
and complete-call measurements. The [host attachment checkpoint](resident-host-accounting.md)
adds writer/image-plane backing and joined CPU worker propagation. The
[serializer attachment](resident-serializer-accounting.md) extends this to owned
token/model/candidate containers, with exact parity and measured overhead. The
[frontend attachment](resident-frontend-accounting.md) covers preparation/evaluation
arrays and completed-frame metadata, with exact parity and measured costs. The
[publication attachment](resident-publication-accounting.md) adds candidate and
retained codestream-byte ownership through C/C++ and batch publication. The
[diagnostic attachment](resident-diagnostic-accounting.md) extends retained
ownership to score histories, summaries, attempt timings and GPU profile graphs.
The [shared device plans](resident-storage-planning.md) extract checked layouts
and AC-search capacities for reuse by upfront planning and actual allocation.
The [host/token bounds](resident-token-storage-planning.md) add reviewed vector
growth/replacement bounds, shared AC/DC token reservations, and aggregate token
owners plus worker scratch across order/context-map candidates.
The [entropy bounds](resident-entropy-storage-planning.md) cover aggregation,
Prefix/ANS model search and retained candidates, and model/token writer scratch;
they also close the missed high-density clustering-queue accounting attachment.
The [representation bounds](resident-representation-storage-planning.md) cover
coefficient-order counts, sampling, permutations, worker reduction and retained
cleared scans, plus ordinary/exhaustive context-map selection and replacement.
The [whole-serializer plan](resident-serializer-storage-planning.md) composes
these bounds with candidate/dispatcher storage, current automatic nesting,
header and section writers, assembly and retained output publication.
The [frontend representation bounds](resident-frontend-storage-planning.md)
cover image planes, owned CPU frames, packed prepared forward coefficients and
atomic reconstruction scratch, including tile lists and dispatch backing.
The [field and color-correlation bounds](resident-field-storage-planning.md)
cover initial-quantization row workers/atomic fields, adjustment, quantizer
selection and per-tile CfL scratch with independently retained map output.
The [preprocessing/perceptual bounds](resident-perceptual-storage-planning.md)
cover color dispatch/atomic images, nested filter scratch, block reductions and
native Butteraugli prepared/one-shot ownership and multiscale replacement. They
also preserve typed underplan errors across six previously generic wrappers.
The [AC-search host bounds](resident-ac-search-storage-planning.md) cover CPU
placement/export and fresh/reused candidate, matrix, cost-table and staging
owners. Their isolated host and mixed-library Metal checks do not replace
whole-workflow qualification or include backend submission/profile metadata.
The [profile-graph bounds](resident-profile-storage-planning.md) add char-string
growth and nested diagnostics, including original/snapshot overlap during Metal
profile resolution. Backend stage/context arrays and policy-to-count planning
remain separate.
Whole-workflow planning/admission is still pending; these partial attachments
do not satisfy the milestone.

Deliverables:

- A source-backed inventory of allocation owners, capacities, aliases, last
  consumers, and release/reuse boundaries: input staging, reference/prepared
  state, AQ/Butteraugli scratch, completed frames, CPU serializer temporaries,
  and idle caches. Classify retained batch codestream results explicitly too.
- A common ledger that counts each backing allocation once. Separate requested
  live bytes, reserved capacity, and retained idle capacity; distinguish all
  three from measured physical footprint and OS-reclaimable pages.
- Define the admission domain and its relationship to the shared production
  backend, explicit backends, concurrent single-image calls, and batch drivers.
  Calls sharing a budget must not each receive an independent full allowance.
- Reserve before allocation/work admission, transfer accounting on reuse or
  ownership changes, and release reservations on every failure and teardown path.
  Include prepared state retained across attempts and completed outputs awaiting
  consumers, rather than admitting only the nominal GPU scratch size.
- Document the configured managed-memory boundary and exclusions. Caller-owned
  inputs, returned results, driver allocations, and process RSS must not be
  implicitly promised as bounded by a scratch-capacity limit.
- Define exhaustion behavior, cache shedding, and trimming. Reject an image
  that cannot fit the configured hard managed limit with a clear failure before
  expensive work, rather than waiting forever or silently exceeding that limit.
  Admitted work must be able to drain without another job holding its required
  capacity. Any staged reservation growth needs a demonstrated progress rule.

The resource record specifies the configuration surface, defaults, domain
ownership, reservation strategy and treatment of retained batch results. Remaining
frontend bounds, combined whole-workflow estimators and API integration remain
to be implemented; those recorded decisions are not existing API promises.

Acceptance:

- Deterministic accounting tests for aliasing, cache hits/misses/growth, trim,
  reclamation, retries, exceptions, and destruction; no leaked reservations.
- Concurrent small/large jobs and multiple callers/backends exercise the declared
  domain, including under-budget rejection and waiting-job progress. Failed
  requests preserve outputs and leave subsequent work usable.
- Counters reconcile with owned allocation capacities. Physical peak/idle
  measurements are reported separately, including any excluded memory.

### 5. Targeted reuse and lifetime reductions — pending

The inventory from milestone 4 defines the opportunities. Individual #4A changes
can proceed alongside its accounting implementation, after milestone 3.

Deliverables:

- For each material temporary or repeated calculation, identify producer,
  consumers, lifetime, and whether elimination/reuse changes arithmetic order.
  Record candidates as implemented, rejected after measurement, or explicitly
  deferred with a reason; do not leave an open-ended requirement to fuse more.
- Prioritize measured costs: unnecessary materialization, repeated metadata
  preparation, scratch surviving its last consumer, and exact shared calculations
  across current candidates. Reusing transform arithmetic requires a valid
  factorization; pruning candidates is still outside this milestone.
- Feed revised capacities and lifetime transitions back into milestone 4's
  accounting. Preserve reference data and completed output across scratch reuse.

Acceptance:

- Numerical and downstream exactness gates, including boundary/padded images and
  diagnostics. Explicit lifetime/failure coverage for each new alias or reuse.
- Complete-encode or peak/idle-memory benefit under stated workloads, with
  downstream costs and regressions reported. Fewer passes or allocations alone
  do not establish a performance win; unsuccessful experiments need not ship.
- The audited set has a recorded disposition. Intentional remaining copies and
  materializations are documented rather than treated as unbounded follow-up.

### 6. Coordinated CPU/GPU scheduling — pending

Depends on stable output lifetimes and working admission from milestone 4;
milestone 5 changes require updated resource estimates and requalification.

The [batch driver](../src/codestream/batch_workflow.h) already permits one image's
CPU work to overlap another's Metal work. It invokes independent single-image
workflows, whose [CPU budgets](../src/core/thread_budget.h) are per encode. The
goal is coordinated use of that capability, not a claim of introducing overlap
for the first time.

Deliverables:

- Coordinate CPU participation across admitted jobs, including caller threads
  and nested serializer work; per-image thread limits must not multiply into
  an unbounded aggregate allocation of workers.
- Make admission and scheduling aware of preparation, GPU completion, and CPU
  consumption. Release reusable scratch after its last actual consumer while
  preserving references needed for retries and output needed by serialization.
- Preserve each image's dependency order and deterministic decisions, request
  ordering, individual failure reporting, and driver shutdown behavior. Define
  fairness so an otherwise admissible large request cannot starve indefinitely.
- Use the smallest explicit scheduling mechanism that achieves these goals.
  Replacing the batch driver or adding asynchronous public APIs is not required.

Acceptance:

- Single-image/batch byte parity across thread and in-flight settings; mixed-size
  and changed-image workloads; simultaneous callers; failures and shutdown while
  work is active or waiting. No deadlock, stale borrowed data, or starvation.
- Instrumented aggregate CPU participation and managed capacity respect their
  declared limits. Measure queued and service latency as well as batch throughput.
- Repeated independent-process comparisons against the integrated baseline,
  including single-image regressions, memory-pressure cases, and a range of
  in-flight counts. Concurrency correctness alone is insufficient.

## Qualification and completion contract

Use fresh Release parent/candidate builds and record revisions, build flags,
hardware, input metadata/hashes, effort, AQ mode, distance, CPU budget, memory
budget, and in-flight count. Warm up, alternate independent process order, and
retain multiple samples on small, natural, padded 1080p/4K, and mixed-size inputs.

Measure the complete in-memory encode including evaluator teardown. Report cold
backend setup separately. For batches, include admission/queueing and completion
of all results in makespan, and report per-image latency separately. GPU-stage
timings and aggregate worker counters explain attribution, not encoder latency.
Record peak, backend-alive idle, and post-trim memory with measurement boundaries.

Each milestone reruns relevant permanent tests and the corpus/policy, pinned
decoder, and conformance gates. Preserve output atomicity and existing numerical
thresholds. The handoff checkpoint has a known CPU `quantization_pipeline` golden
mismatch; reproduce it on each fresh comparison baseline rather than assuming a
future failure is inherited. Record sanitizer limitations explicitly.

The structural effort is complete when milestones 3–6 satisfy their gates:
integrated ownership is qualified, declared resources are coherently accounted
and admitted, the audited reuse opportunities have dispositions, and cross-image
scheduling obeys resource limits with qualified correctness/performance. Maintain
reviewable milestone commits and durable summaries identifying raw artifacts.

Completion does not require every intermediate to disappear, every workload to
speed up, or all encoding work to move to Metal. It does require explicit costs,
lifetimes, limits, and evidence for the choices retained. No fixed speedup is
promised by this roadmap.

## Separate policy track: #4B and #1/#2

Candidate screening/pruning, selective regional refinement, predictive AQ, and
different iteration/stopping policies may use this infrastructure later. They
need independent natural-image and difficult-image size/quality curves,
decoded-pixel and Butteraugli measurements, determinism, and rate-control checks.
Changing policy is not a way to pass the structural branch's performance gates.
This track is deferred and is not a dependency of milestones 3–6.
