# Public workflow admission

This checkpoint connects the [unified plans](resident-admission-preflight.md)
to real C++, C and batch entry points. The comparison parent is `156b3b2`.
It completes milestone 4's memory-admission part of resident execution, not aggregate CPU
scheduling, a codec-policy change, or a process-RSS limiter.

Later work adds [shared CPU participation](resident-cpu-coordination.md); the
measurements below describe this earlier memory-admission checkpoint only.

## Behavior and ownership

`ExecutionDomain` is an immutable shared handle containing a managed-capacity
allowance. `VarDctEncodingOptions::execution_domain` selects it. Every null
option/context uses the same default, including the existing fallback ledger;
null does not create a fresh allowance per call or backend. Zero bytes means
unlimited but accounted. An explicit domain with a zero limit is independent
of the default. Domain identity, not equality of numeric limits, defines sharing.

The C API exposes the equivalent `GJXLExecutionDomain` handle and appends it to
`GJXLContextOptions`. Context creation retains the underlying domain; destroying
the opaque handle does not invalidate the context. The installed C++ bridge,
`gjxl/execution_domain.hpp`, exports or retains that same shared object across
the two APIs. It does not copy its allowance. Link `gjxl::c` as well as the C++
codec target when using the bridge. The public C++ options layout changed:
recompile C++ consumers and do not mix old libraries with current headers.
The sized C interface continues accepting the original 8-byte and 12-byte
context prefixes without reading the appended pointer.

For example, after checking `ExecutionDomain::Create` for failure:

```cpp
std::shared_ptr<const gjxl::ExecutionDomain> domain;
auto status = gjxl::ExecutionDomain::Create(
    {.managed_memory_bytes = budget_bytes}, &domain);
// On success, share domain with every participating request/context.
options.execution_domain = domain;
```

Snapshots distinguish live requested/capacity bytes, idle capacity, unbacked
reserved capacity, backing/committed peaks, active reservations and waiters.
The allowance protects `live capacity + idle capacity + unbacked reservation`.
Peaks are cumulative for that domain; they are not sampled GPU usage or RSS.
An allocation's ticket retains the accounting even after its producing call or
wrapper has gone away. Published caller results are outside the managed boundary.

## Complete-call admission

Normal option, geometry and backend qualification select a checked policy plan.
Admission then reserves its entire work envelope before managed input,
preparation or encoder allocations. The C envelope also contains converted
linear input and simultaneous internal/output-copy storage. C packed-input
layout and opaque-alpha validation precede admission and perform no conversion.
Nested completed-output, C and batch calls inherit the outer reservation;
an explicitly different domain is rejected rather than bypassed.

The planner uses the existing backend selector. Automatic fully-resident size
search stays on CPU. Automatic exact-coefficient search plans both backends
only if its bounded existing candidate order can visit the Metal-eligible
interval. A fixed-size stack simulation shares interval tie selection with the
real search; it does not encode candidates, prune them, or change early success.
Qualification can initialize the immutable backend earlier than before, even
when a subsequent memory admission fails. This setup is outside managed backing
and must not be confused with starting image preparation.

FIFO admission evicts matching-domain idle storage across live Metal backends
before waiting. Admitted jobs never wait for more allocation credit or grow
their reservation. Reuse consumes the destination job's existing credit;
oversized/cross-domain cache entries cannot bypass its plan. Tight allowances
may evict a warm cache that cannot coexist with a new complete reservation.
This is intentionally conservative and can affect warm latency.

- A plan above the hard limit fails with `OutOfMemory` before managed work.
- A physical allocation failure remains `OutOfMemory`.
- An admitted plan violation remains typed `resource_plan_exceeded()` in C++
  and is `GJXL_ERROR_RESOURCE_PLAN_EXCEEDED` (7) in C. It is terminal: no fallback,
  retry, silent growth or partial caller-output publication.
- Invalid ordinary input/options and unavailable backends keep their normal
  status categories. When an input has multiple simultaneous errors, earlier
  preflight can expose a different one first.

## Batch contract

All requests in one batch, including invalid ones, must select the same domain.
The batch streams preflight without an image-count-sized plan vector, then
reserves result metadata, conservative retained-result capacity and complete
largest-image work slots. The minimum is the retained envelope plus one full
work slot. A smaller cap rejects the entire batch before result arrays or image
work; there is no spill, dropped request, reduced effort or streaming API.

The configured worker count is a ceiling. A finite allowance can admit fewer
slots; when idle storage cannot coexist even with one slot, each worker trims
matching-domain idle caches before reusing its slot. Work in other active slots
retains its own full credit. Results remain charged through publication of the
entire result array, not merely until each image completes.

Ordinary invalid/unavailable requests retain ordered per-image failures. A
preflight allocation/representation failure is a whole-batch admission failure.
An admitted planner violation or required-trim failure stops new work, drains
already active workers and leaves the caller's previous result array unchanged.
Physical failures inside an ordinary image encode remain per-image failures.
The driver can be reused after failure. Its existing CPU/GPU overlap is preserved;
this checkpoint does not impose an aggregate CPU-participation limit.

## Boundary and qualification

Included are encoder-owned converted input, prepared host/device arrays,
AC/AQ/Butteraugli storage, completed frames, serializer/model/candidate backing,
diagnostics and retained batch output through atomic publication. Excluded are
caller input and prior/published output, immutable backend/pipeline setup,
command/driver internals, thread stacks, bounded control records/error-message
strings and allocator overhead. The component ownership/planning records linked
from [resident resources](resident-resources.md) remain the source of the bounds.

The final source scan rechecked ordinary containers, heap factories, strings,
callbacks and raw Metal allocation sites across `src/`. Its two retained audit
listings distinguish real owners from declarations and compatibility overloads:

- `PublicationVector`'s ordinary-vector backing and the C byte-array copy have
  explicit tickets before allocation and retain them through publication.
- Serializer `Storage`/nested `Storage` and the ANS queue/map use managed
  allocators. Legacy ordinary-vector adapters and test-only Metal snapshots are
  not used by the admitted public encoding path.
- Image/evaluator/prepared/completed-frame factories allocate small control
  objects whose image-sized members use managed host/device storage. Their
  controls, fixed callbacks and status strings remain declared exclusions.
- Backend registration, pipeline labels, the immutable DCT basis and driver
  objects are setup/control exclusions. The image-buffer Metal allocation
  choke point carries a backing ticket. Shared arena slices are not extra owners.
- Batch worker/thread-control storage belongs to the excluded driver setup;
  result records, bytes and diagnostic backing are admitted. PFM loader buffers
  belong to input I/O outside the in-memory encode boundary.

This audit preserves the stated boundary; it does not reinterpret all process
allocations as managed or infer coverage merely because measured peaks fit.

The new public-path tests cover all policy families, cold/warm finite domains,
terminal underplans and same-domain recovery, impossible-plan preallocation
rejection, FIFO small/large calls, shared C/C++ context lifetime, and tight batch
slot/trim enforcement. They observe actual worker participation and retained
publication capacity. Shared C/C++ tests run before default-domain oracles so
even a transient default-ledger allocation would increase its pristine peak.
The installed consumer separately compiles the historical C layouts and checks
domain sizing, output atomicity, linkage, handle identity and retained contexts.

Qualification artifacts live in `build/public-admission-qualification/`.
The parent is built from a complete `git archive 156b3b2`, including its own
headers and installed-consumer sources, with dependencies linked from the same
unchanged directories. Metal-cpp is `27c4382`; the libjxl source submodule is not
initialized and the optional libjxl reference implementation is disabled in
both builds. Independent decoding uses the separately pinned existing `djxl`.
This avoids the C++ layout mismatch that would make a current-header/old-library
comparison invalid.

### Final validation

Platform: Apple M4 Pro / Mac16,7, 48 GiB RAM, 14 logical CPUs; macOS 15.6
(24G84), Apple Clang 17.0.0. Both fresh builds use Release, tests and benchmarks
enabled, native Butteraugli, and Metal profiling disabled. The production Metal
policy is SIMD-group transforms plus fused-tuned residual inverse.

- Parent: **92/93 CTests pass**. Candidate: **94/95 pass**. Both retain only the
  inherited `quantization_pipeline` golden mismatch, actual
  `0.24919039011001587` versus expected `0.24914586544036865`. It was not changed
  or reclassified as a pass.
- Six targeted Release tests pass three times each: public admission, its CPU
  subset, batch workflow, publication, rate-control search and unified plans.
  The installed consumer passes, including the separately compiled old C layouts.
- ASan/UBSan: CPU admission and rate-control tests pass three times without
  suppression. Full public admission, batch and publication tests pass three
  times with the established Metal-cpp null-member-call suppression. The
  unsuppressed vendor failure at `Foundation/NSObject.hpp:112:49` reproduces;
  leak detection is disabled. This is not an unsuppressed Metal sanitizer claim.
- Strict warning-as-error compilation passes for changed C adapter/workflow/
  batch/search units and the new public test; `git diff --check` passes.
- Ten parent/candidate codestream cases match exactly, including compatibility,
  maximum-error, high-density, maximum-compression and padded-4K policies. All
  ten independently decoded PFM pairs match. Both revisions pass all **22**
  pinned-decoder conformance fixtures.

### Default-domain complete-call cost

Seven alternating independent-process pairs per case, nine calls per process,
discarding the first two complete calls and comparing process medians. Inputs
alternate with a changed-image variant. Backend setup is measured separately,
outside the encode timer; planning, admission, execution, teardown and result
publication are inside. Effort 7, target 1.2, forced fully-resident Metal, CPU
automatic, default unlimited shared domain. No GPU/worker aggregate counter is
used as wall time. All **378** call-level byte-size/hash comparisons agree.

| Public workload | Median paired wall-time change | Pair range |
| --- | ---: | ---: |
| Kodak 17, single | -0.17% | -4.02% to +5.75% |
| Synthetic 3839x2159, single | -0.39% | -2.84% to +1.93% |
| Four-image Kodak 17 batch, four slots | +0.32% | -1.65% to +1.24% |

These are effectively flat median results, not evidence of a speedup. Raw
process observations, commands, input hashes and driver/build provenance are
retained. They compare this checkpoint to `156b3b2`, not the original pre-refactor
architecture; final scheduling still needs its own broader baseline comparison.

### Finite-domain cost and footprint

Three independent processes per configuration, three complete calls each;
configuration order reverses in the middle round. Synthetic 257x193 and
3839x2159 inputs alternate with changed-image content, effort 7, target 1.2,
fully-resident Metal, **one CPU participant per image**, single or four-image
batch. Unlimited explicit, minimum-fit and full-concurrency finite domains are
checked. All **108** calls (**270** images) succeed with matching bytes across
allowances; actual domain peaks respect every finite cap. Tight batches trim
before slot reuse; every domain is empty after final explicit trim.

The following padded-4K figures separate reserved allowances, actual backing
and process physical footprint. GiB means `2^30` bytes. The physical ranges are
the three independent-process peaks, not a guarantee enforced by the API.

| Workload/domain | Cap GiB | Peak managed backing GiB | Physical peak GiB | Post-trim physical GiB |
| --- | ---: | ---: | ---: | ---: |
| Single, unlimited | None | 2.423 | 2.749–2.753 | 2.045–2.140 |
| Single, minimum fit | 6.670 | 2.423 | 2.747–2.754 | 2.047–2.132 |
| Batch, unlimited / four slots | None | 9.693 | 8.860–9.251 | 2.115–2.147 |
| Batch, minimum fit / one slot | 7.241 | 2.428 | 2.762–2.764 | 1.416–1.782 |
| Batch, finite / four slots | 29.064 | 9.693 | 8.799–9.017 | 2.060–2.144 |

The single complete plan is 7,161,788,400 bytes; the minimum batch plan is
7,774,762,280 bytes. These conservative bounds can reject work whose actual
content-dependent execution would fit. They must not be advertised as expected
memory use. Warm idle capacity is 1,946,881,084 bytes for these single/four-slot
cases, and zero after each tight batch. Final managed capacity is zero after
trim, while physical footprint remains nonzero because caller inputs, allocator
retention and driver/system storage are outside the managed contract.

Pressure has a measured cost. Across the last two calls per process, median
complete-call time is 353 ms for unlimited single versus 408 ms at the minimum
cap; existing idle capacity cannot coexist with a new full reservation and is
evicted. Four-image batch completion is 896 ms unlimited/four-slot, 1,630 ms
minimum-fit/one-slot, and 971 ms finite/four-slot. These short repeated pressure
runs expose the concurrency/cache tradeoff; they are not the seven-pair default
timing study above. No effort or candidate reduction compensates for that cost.

Milestone 4's declared memory boundary and progress rules are now implemented
and qualified. Milestone 5 still needs the remaining reuse/fusion dispositions;
milestone 6 still needs aggregate CPU scheduling, queue/service instrumentation
and its end-to-end qualification. Neither is marked complete by this result.
