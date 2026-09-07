# Shared CPU participation checkpoint

This is the first scheduling checkpoint after `f30e92a`, not completion of
[milestone 6](resident-execution.md#6-coordinated-cpugpu-scheduling--in-progress).
It bounds shared CPU participation without changing codec decisions or replacing
the synchronous batch driver. Remaining work is listed below.

## Ownership and limits

Each `ExecutionDomain` now owns immutable managed-memory and CPU allowances.
`cpu_participant_limit == 0` resolves once to hardware concurrency, clamped to
1..256. Explicit values are 1..256. Null C/C++ options share the default domain;
explicit domains are independent, even when their numeric limits match.
Per-image `cpu_thread_count` remains an additional ceiling, not a multiplier
of the shared allowance. Automatic mode retains stage-specific desired
parallelism but now uses the caller as a participant too.

The domain enforces this invariant under one mutex:

```text
active participants + pre-launch worker reservations + suspended workers <= limit
```

These are GJXL scheduling counts, not measured hardware occupancy. A preempted
participant still owns its ticket. Idle persistent driver threads, external
caller threads, thread startup/exit machinery, short control/locking operations,
backend initialization and OS/driver internals are not an OS-thread pool governed
by this budget. Image I/O outside the in-memory encoding calls is outside scope.

`CpuBudget` queues new and resumed image callers FIFO for one slot. Additional
workers only reserve currently free slots, and cannot bypass queued callers.
There is no blocking acquisition of extra parallelism while holding a caller
slot: a denied worker grant simply leaves tasks for existing participants.
Move-only tickets retain their budget state independently of public handles.

`CpuWorkerGroup` reserves local per-image capacity and then global capacity
before constructing threads, returning any excess local reservation. Unlaunched
slots are released during cleanup; launched workers transfer pending capacity
to active participation. A loop's caller never takes a second worker ticket.
Explicit per-image budgets continue suppressing nested fan-out. Automatic nested
loops share both quotas and cannot create an independent allowance.

## Blocking and lifecycle boundaries

An image caller yields its global slot during a GPU wait or worker join and
queues FIFO to resume. Its per-image caller credit stays reserved, so resumption
never depends on descendants returning local capacity. Created workers instead
retain global protected capacity while suspended: otherwise nested joins could
accumulate dormant OS worker threads across admitted images. Those workers
resume their existing credit without acquiring another slot or waiting on a
caller that is joining them. Active and suspended counters remain distinct.

The Metal suspension encloses the whole `call_once`, not only the underlying
`waitUntilCompleted`. Otherwise a second waiter can hold the sole CPU slot while
waiting for the first waiter to resume inside `call_once`. Batch completion
similarly releases its work mutex before reacquiring CPU capacity.

The five existing parallel-loop implementations use pregranted worker groups:
color rows, initial AQ, forward transforms, coefficient-order groups and
serializer sections. Both normal joins and partial-launch cleanup yield caller
participation; serial fallback runs only after resumption. Direct component
calls outside an admitted workflow preserve their legacy behavior.

The main CPU scope starts after whole-workflow memory admission. C RGBA alpha
validation and batch preflight use temporary CPU scopes, ended before memory
admission. C conversion and the final output copy are covered by the main scope;
nested codec calls borrow that same participation. C++ publication retains its
scope through prepared-state teardown and ownership transfer. Each batch image
retains its scope through result retention and optional cache trimming. The
batch coordinator has its own short preflight/publication participation and
yields while image workers run.

The participant observer follows actual scope entry, suspension and resumption,
including sequential caller work; nested parallel scopes do not count that
caller twice. Memory and CPU snapshot sets are individually consistent but
sampled separately, not atomically together.

## API and measurement boundaries

The C domain option and snapshot structs append CPU fields. The old 16-byte
options and 72-byte snapshots remain supported; intermediate-sized fields are
read/written only when wholly covered. A separately compiled C fixture uses the
old declarations without including the current header and checks canaries.
The existing legacy context fixtures remain. C++ consumers must rebuild for the
larger domain/snapshot/timing layouts. The Rust wrapper does not yet expose
explicit domain selection; its older C declarations remain compatible.

`VarDctEncodingTiming` adds initial CPU queue time, caller resume queue time and
yielded GPU/join wall spans. These end at the existing internal successful-result
commit, before prepared teardown and outer publication. They do not include
batch-driver queueing or memory-admission wait and are not worker CPU time or GPU
execution time. They must not be relabeled complete service latency; surround
the public call for complete latency. The follow-up
[batch timing checkpoint](resident-batch-timing.md) adds separate arrival queue,
service and internal-readiness spans without changing these older boundaries.

## Qualification record

The comparison uses the complete archived `f30e92a` sources in
`build/cpu-coordination-parent-source`, a fresh Release baseline in
`build/cpu-coordination-parent`, and candidate `build/cpu-coordination`.
No current C++ headers are linked against old archives. Runtime shaders are
unchanged. Both builds enable tests/benchmarks, disable the optional libjxl
reference and Metal profiling, and use Apple Clang 17 on the Apple M4 Pro
(Mac16,7, 48 GiB, 14 logical CPUs, macOS 15.6 / 24G84).

Raw commands, sources, logs, retained codestreams and decoded images are in
`build/cpu-coordination-qualification/`; `run.py` exposes build, tests,
sanitized, parity, conformance, performance, pressure, seal and verify actions.
The decoder is the separately retained pinned `djxl` executable in the primary
checkout, not a claim that this worktree's libjxl source submodule is initialized.

The frozen Release parent passes 94/95 tests and candidate 98/99. Both reproduce
the same `quantization_pipeline` mismatch: actual `0.24919039011001587`, expected
`0.24914586544036865`. Nine selected tests pass three repetitions each. The new
public tests cover CPU and Metal, per-image limits 0/1/2/8 under domain limits
1/2/4, initial queue timing, physical-allocation and underplan recovery, and
simultaneous C/C++ calls plus two four-worker batch drivers sharing limits 1/3.
Mixed dimensions and changed images retain exact result order, bytes and summaries.

The budget and execution-scope tests also pass ten ThreadSanitizer repetitions
each. They cover FIFO/cancellation, shared-state lifetime, pending/active/dormant
transitions, nested observation without duplicate caller counts, one-slot
resumption and `call_once`, per-image nested limits and partial-launch cleanup.
ASan/UBSan passes three repetitions each of four CPU-only tests without
suppression, and three repetitions each of three Metal/batch tests with only
the existing `null:*/third_party/metal-cpp/*` UBSan suppression. The unsuppressed
Metal run separately reproduces the vendor `Foundation/NSObject.hpp:112:49`
null-member-call report. ASan leak detection is disabled for these platform runs;
the result is not a claim of unsuppressed Metal or leak-sanitizer cleanliness.

Ten retained policy/corpus pairs have identical parent/candidate SHA-256 hashes
for both codestreams and independently decoded PFM pixels. Both revisions pass
all 22 pinned-decoder conformance cases. Shaders have identical SHA-256 hashes.

### Complete-call regression measurements

Each row uses seven alternating independent parent/candidate process pairs,
nine public calls per process, with the first two calls discarded as warmup.
Inputs alternate original and deterministically changed pixels. Backend setup
is outside the timer and reported separately; planning, admission, teardown and
publication are inside. Effort 7, target 1.2, forced fully-resident Metal with the
production SIMD/fused-tuned backend, automatic per-image CPU policy, shared
default domain (candidate CPU cap 14), unlimited managed memory.

| Complete public call | Median paired change | Paired range |
| --- | ---: | ---: |
| Kodak 17 single | -2.47% | -4.10% to +3.02% |
| Synthetic padded 3839x2159 single | -1.12% | -2.61% to +4.38% |
| Four-image Kodak 17 batch | -0.39% | -3.10% to +0.80% |

All 378 call-level size/FNV64 checks agree across revisions. The small timing
changes overlap process-to-process variation: this establishes no consistent
regression in these cases, not a demonstrated general speedup or final M6
throughput qualification. Batch values are whole-batch makespan, not per-image
service time.

### Combined CPU and managed-memory pressure

Twenty-four independent processes cover 257x193 and 3839x2159, single/four-image
batch, CPU cap 1/3, three process repetitions and three calls per process. All
72 calls / 180 images respect CPU and memory limits, match bytes across caps,
and leave zero managed capacity after trim. Tight batch admission selects one
in-flight image and trims idle capacity between images.

The 4K single-image managed cap is 7,369,550,030 bytes; the batch cap is
7,982,524,102 bytes. Plans retain conservative automatic-parallelism bounds even
when the independent CPU cap allows less work; they are not physical-RAM limits.
Peak managed backing is 2,602,024,508 bytes single and 2,606,649,170 bytes batch.

| Tight 4K call | CPU cap / observed peak | Warmed median wall | Physical peak range |
| --- | ---: | ---: | ---: |
| Single | 1 / 1 | 414 ms | 2.750–2.762 GiB |
| Single | 3 / 3 | 335 ms | 2.749–2.757 GiB |
| Four-image batch, one slot | 1 / 1 | 1681 ms | 2.758–2.849 GiB |
| Four-image batch, one slot | 3 / 3 | 1341 ms | 2.754–2.763 GiB |

These are deliberate-cap costs, not parent/candidate speedups. Each warmed
median pools the last two calls from three independent processes. Post-trim
physical footprint remains 1.443–2.126 GiB across these configurations despite
zero managed capacity: caller images, allocator retention and driver state are
outside the managed ledger. Tight-cap cache eviction costs remain explicit.

### Integrated-baseline preparation and embedding validation

The next milestone's integrated baseline is freshly archived from
`ec4d4c5317d983b5f6df29d2474923443c9cfe63` into
`build/resident-scheduling-integrated-source` and built in
`build/resident-scheduling-integrated`. It passes 63/64 tests after the generated
artifact repair below, with the same quantization golden mismatch. This is
baseline preparation, not a final scheduler comparison against that revision.

Its initial parallel Unix Makefiles build generated a truncated embedded Metal
payload (151,552 of 1,489,228 bytes), causing seven additional test failures.
The log contains overlapping `Linking gjxl.metallib` rules, and the truncated
payload matches the complete file's prefix, pointing to an embedding/link race.
The bad include and a linked diagnostic executable remain in the qualification
directory. Regenerating the embedding after shader completion and rebuilding
repairs all seven failures without changing baseline source.

`verify_embedded.cpp` independently compares the linked `EmbeddedMetalLibrary()`
span with the complete library byte-for-byte for the immediate parent, candidate,
sanitized candidate and repaired integrated baseline; all match. The unchanged
shader SHA-256 is `9dcbc4dff15fd81c0793a1df3d68fd06ac821d0f34929a0e397f1de4e9cafae4`.
The follow-up [build-ordering checkpoint](metal-build-ordering.md) resolves the
Makefile defect and adds graph and linked-payload regression tests. Explicit
Ninja remains consistent with the repository's Justfile. Do not attribute this
generated-artifact failure to the CPU scheduler.

## Remaining milestone 6 work

- The [batch lifecycle checkpoint](resident-batch-lifecycle.md) now defines and
  tests explicit shutdown: reject new/queued calls, drain the active call and
  join workers while the object stays alive. Keep this coverage current with
  subsequent scheduler changes; external callers must return before destruction.
- The [batch timing checkpoint](resident-batch-timing.md) now records driver,
  memory and CPU queueing separately from image service through internally
  retained-result readiness. Use those metrics alongside complete public-call
  makespan in the remaining final qualification, preserving the distinct later
  whole-array publication boundary.
- The [worker-launch checkpoint](resident-worker-launch.md) now exercises actual
  loop construction failures, partial-worker cleanup, fallback/recovery and
  exceptional shutdown/queue transitions. Its sanitizer matrix supplements the
  primitive partial-launch model.
- Qualify the complete final scheduler against the integrated preparation baseline,
  not just this immediate parent: small/natural/padded 1080p/4K and mixed-size
  inputs, several in-flight counts, simultaneous callers, single-image regressions,
  throughput, per-image latency and memory pressure. Keep exact bytes/independent
  decoded pixels, conformance and sanitizer qualifications current.

The scope is still a shared synchronous execution domain. No asynchronous public
API, codec-policy change, GPU entropy implementation or thread-pool replacement
is required merely to close these gates.
