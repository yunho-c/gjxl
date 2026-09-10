# Unified admission preflight and cache progress

This checkpoint follows `e30caaf` on `refactor/resident-execution`. It composes
the existing policy-specific storage recipes with outer input/publication and
batch ownership, and supplies safe domain-wide idle eviction. **It does not yet
enable public whole-workflow admission or complete milestone 4.**

## Recipe selection and outer adapters

`ComputeWorkflowStoragePlan` accepts an already selected execution route:
CPU-only, Metal-only, or potentially mixed automatic exact-coefficient search.
It delegates to the qualified CPU, resident or compatibility recipe without
changing backend selection, AQ policy, candidates or serializer behavior.
Normal entry-point options and buffer validation must precede it. In particular,
the CPU-only route requires proof that every attempt stays on CPU; the planner
does not initialize a backend or decide whether automatic Metal is qualified.

Borrowed linear RGB adds no input owner. The packed-sRGB C adapter adds the
three converted source-sized F32 planes and a fresh maximum-output-sized byte
array for C publication. The internal codestream and converted input remain
alive during that copy. Only the C array is published; internal score history
is not an additional retained C result. These are managed backing bounds, not
RSS, and retain the component plans' documented exclusions. The adapter recipe
does not expand the public C API's available encoding policies.

## Batch ownership and concurrency

`BatchWorkflowStorageAccumulator` streams over requests without allocating a
request-sized plan array before admission. An ordinarily invalid request still
requires a result slot, but no encoding work or retained-output allowance.
Each valid request contributes its maximum retained result capacity, including
codestream, summary score history and attempt timings. The three actual
request-sized arrays are separately bounded: public result records, internal
owned results and publication-escrow tickets.

Let `B` be those arrays plus all retained results, `W` the largest per-image work
bound, and `I` the sum of componentwise maximum idle capacities across the four
production Metal pools. A plan retaining caches reserves `B + I + n * W` for
`n` in-flight workers. Componentwise maxima matter: different completed images
can leave different pools behind while later requests occupy the work slots.
Each active image's own pooled capacity is already included in its `W`.

A finite limit below `B + W` rejects the batch before work; there is no spilling,
result dropping or effort reduction. If `B + W` fits but `B + I + W` does not,
the planner omits `I` and requires trimming **before every completed worker
retires or reuses its work slot**. An active job can return buffers to a pool
before its CPU tail finishes: those buffers remain covered by that job's work
slot until it trims. This condition is part of the plan's executor contract,
not a behavior already installed in the current batch driver. Empty and
all-invalid batches require no work slots.

## Two cache correctness requirements

### Capacity reuse must respect the allocation's plan

Legacy Butteraugli caching may reuse an allocation up to twice the requested
size. Checking only whether that capacity fits the job's *currently unused*
reservation does not prove that later allocations will fit. For explicitly
reserved work, Butteraugli now requires an exact-capacity cache hit, as the
other three AQ pools already do. Unadmitted calls retain the existing bounded
hysteresis and purged-cache recovery.

The regression fixture caches 512 bytes, then admits a 1,024-byte job needing
256 bytes of Butteraugli capacity plus a later 700-byte allocation. Reusing
512 would fit initially but incorrectly exhaust the otherwise sufficient plan.

### Eviction must close the cache-return race

Evicting once and then entering FIFO admission is insufficient: an active job
can return a buffer after eviction and close its reservation, leaving a waiter
blocked indefinitely by idle storage. `ResourceBudget::Reserve` now accepts an
optional eviction callback and enqueues **before** invoking it. The callback
runs without the budget mutex. While any request is queued, `MakeIdle` refuses
cache retention without changing the live ticket; existing cache-return paths
then destroy the backing. Failure, exception and cancellation remove the
waiter and notify followers. Invalid, oversized, already-cancelled and
immediately admissible requests do not invoke eviction.

`TrimMetalPreparationCachesForDomain` visits all live Metal backends, including
explicit backends, and drops only matching-domain idle pools. It neither
creates a backend nor touches active leases. The registry lock protects visits
against concurrent unregister/destruction; each backend retains the registry
through static teardown. Lock order is registry, backend cache, then budget.
Reservation callbacks must therefore release the budget lock before eviction.
Small registry/control metadata remains outside the managed backing boundary.

Targeted eviction does not advance the backend-wide manual-trim epoch or
invalidate another domain's active leases. The queued-admission check prevents
refill while admission waits. Existing explicit trim keeps its stronger
generation-based semantics.

## Qualification

The retained harness is `build/workflow-admission-qualification/run.py`.
It uses the frozen `build/compatibility-workflow-plans` parent and fresh
`build/workflow-admission` Release and `build/workflow-admission-asan` builds.
New pure-plan record layouts are compiled together when testing the old encoder
runtime; the old encoder never observes those records. No mixed-layout backend
or prepared-object ABI is used. Parent-side test/driver compilation uses the
parent's original inline resource-budget header to avoid mixing different class
definitions across translation units. Final timing drivers are linked against
the exact archived library hashes recorded in `build.json`. Initial unsealed
driver evidence is retained separately; the final record supersedes it.

Permanent tests cover:

- 270 allocation-free route/diagnostic/effort/geometry compositions, including
  padded 4K; adapter sums, invalid routes and atomic overflow/error handling.
- Streaming retained-result and per-pool maxima, 1–8 requested worker slots,
  finite-limit slot selection, the exact minimum bound, impossible batches,
  rejected additions, empty batches and all-invalid batches.
- Eight complete C cases: CPU/Metal, RGB/RGBA, padded rows and two image sizes.
  Each runs an ample-reservation oracle and bounded cold/warm encodes with
  previous published output alive. Input, publication and undersized-plan
  failures are atomic; physical-failure recovery reuses the reservation.
- A heterogeneous four-image batch plus an invalid request at 1, 2 and 4
  workers, three calls each; exact bytes/summaries and publication lifetimes.
  Tight-plan trimming is tested as a planning contract, not claimed as an
  integrated scheduler feature. Metadata underplans preserve caller output.
- All four real Metal pools across three backends/two domains, active leases,
  queue/refill progress, oversized Butteraugli reuse, concurrent registry visits
  during twelve backend lifetimes, callback failures/exceptions, cancellation,
  existing FIFO/state-model and concurrent-accounting tests.

On Apple M4 Pro (14 CPU threads, 48 GiB), macOS 15.6, AppleClang 17 Release:

- Parent: 90/91 tests; candidate: 92/93. The sole failure on both is the
  inherited `quantization_pipeline` golden mismatch:
  `actual=0.24919039011001587`, `expected=0.24914586544036865`.
  The initial candidate run had benchmarks disabled (91/92); the final run
  matches the parent's benchmark-smoke configuration and includes that gate.
- The four targeted resource/cache/workflow tests each pass three consecutive
  Release and ASan/UBSan runs. The CPU budget test also passes three unsuppressed
  sanitizer runs. Metal runs retain the existing vendor-header null-call
  suppression: an unsuppressed run reproduces
  `third_party/metal-cpp/Foundation/NSObject.hpp:112:49`. Leak detection is
  disabled. This is not an unsuppressed-clean Metal sanitizer claim.
- Strict warning compilation passes. Ten parent/candidate codestream pairs
  match byte-for-byte, and all ten independently decoded PFM pairs match. Cases
  include exact coefficients, maximum-error variants, maximum throughput,
  high density, maximum compression and padded 4K. Both builds pass all 22
  pinned conformance fixtures. The compiled Metal library is byte-identical.

Complete-call timing uses seven alternating independent-process pairs per case,
nine encodes per process and two discarded warmups. The retained driver measures
the in-memory call through preparation/evaluator teardown; backend creation,
input loading and output hashing are outside the timer. Settings are forced
Metal fully-resident, SIMD DCT, fused-tuned AC, effort 7, distance 1.2, automatic
per-image CPU count, one in-flight image and no public memory admission. Each
case alternates the source and a deterministic `0.91 * pixel + 0.017` variant.

| Case | Median paired change in process-median complete-call time | Pair range |
| --- | ---: | ---: |
| Kodak 17 | +1.48% | -1.73% to +3.10% |
| Synthetic 3839x2159, padded to 4K | +0.48% | -0.94% to +1.10% |

All 252 encoded size/hash pairs agree between builds. The final sample shows
small latency regressions, not a speedup. Pair scatter and the differing signs
in earlier unsealed captures do not isolate a stable causal effect; keep these
costs visible and requalify with public admission. Neither physical-footprint
improvement nor admitted-batch throughput is claimed. The
`validation.json` manifest seals source, build, test, decoder and timing evidence;
`run.py verify` rechecks the retained hashes.

## Remaining integration

Milestone 4 still requires shared entry-point validation/actual route selection,
immutable public C++/C execution-domain handles with one shared default,
admission before C conversion and other allocations, nested C/workflow/batch
context propagation, batch enforcement of chosen slots/trim obligations,
terminal whole-operation handling of planner violations, and final concurrency,
failure and physical-memory qualification. Milestones 5's remaining reuse/fusion
audit and 6's aggregate participating-CPU scheduling/throughput gates remain
open. Nothing here changes the roadmap's policy-preservation or completion bar.
