# Batch queue, service and internal readiness

This checkpoint follows `806830e` and adds the measurements needed for the final
[scheduling qualification](resident-execution.md#6-coordinated-cpugpu-scheduling--in-progress).
It does not mark that milestone complete or claim a performance improvement.

## Measurement contract

Each `VarDctBatchEncodingResult` now has a separate `scheduling` record. All
images in a batch share the arrival boundary: entry to the batch `Encode`
implementation, before the driver mutex, preflight or resource admission.

- `queue_nanoseconds`: arrival through the worker's observation of successful
  initial image CPU admission. Includes driver-mutex queueing, preflight,
  memory admission, waiting for an in-flight image slot and initial CPU wait.
- `service_nanoseconds`: initial image CPU admission through internally retained
  result readiness, after prepared-state teardown, ownership transfer and any
  per-image cache trim. Includes GPU/join waits and CPU resume queueing; it is
  calling-thread wall time, not active CPU time or GPU execution time.
- `ready_nanoseconds`: arrival through that internal readiness boundary.
  Exactly queue plus service for each image, measured on the steady clock.
- `cpu_admitted`: distinguishes work that reached initial image CPU admission
  from errors resolved before it. For the latter, service is zero and queue
  equals ready. Preflight validation itself can use coordinator CPU participation;
  this flag specifically describes the image workflow's CPU scope.

Readiness precedes release of the image's CPU scope and the worker-completion
bookkeeping. It also precedes whole-array publication: this synchronous API
does not publish an image as soon as it is ready. Surround the public `Encode`
call to measure complete batch latency and use that makespan for throughput.
The later publication/return gap is not part of per-image service. Image service
spans overlap and must not be summed into an encoder latency estimate.

An error after CPU admission still records its real service/readiness spans,
even when the older successful-workflow `timing` result remains zero. An early
invalid request records readiness when preflight resolves it. Whole-batch
failures, including closed-driver rejection and terminal planner violations,
leave all caller-visible records unchanged.

`VarDctEncodingTiming` retains its older internal successful-commit boundary;
its `total_nanoseconds` is not relabeled as complete image service. CPU scopes
collect their admission timestamp only when timing is requested. No image
arithmetic, decisions or shader dispatch is changed.

## Storage and benchmark reporting

The public C++ result layout grows; consumers must rebuild against matching
headers and libraries. The existing whole-batch planner derives result-record
capacity from `sizeof(VarDctBatchEncodingResult)`, so it includes the new fields
without a hard-coded size adjustment. C domain/context ABI layouts are unchanged.

`gjxl_image_batch_benchmark` appends median/maximum image queue, service and ready
milliseconds to its CSV. These distributions pool all images from measured
batched calls, excluding warmups. Individual queue/service/ready values add as
above, but independently computed distribution medians need not add. Existing
`batch_ms_per_image` remains makespan divided by image count, a throughput-derived
quantity, not observed service latency. Sample collection/validation is outside
the public-call timer, as are driver construction and source generation.

## Qualification

The permanent `batch_scheduling` / `batch_scheduling_cpu` tests use controlled
serialization, worker-exit and publication boundaries. They verify same-driver
queueing, one-slot work admission, service waits, internal readiness before
public availability, exact bytes, per-image failures before/after CPU admission,
and successful reuse after a failed image. Existing lifecycle tests now check
that CPU/memory admission waits appear in queue time and shutdown/planner
rejection preserves every scheduling field. The installed C++ consumer checks
the appended record and its identity through closed-driver rejection.

Fresh Ninja Release, ThreadSanitizer and ASan/UBSan builds use
`build/resident-batch-timing{,-tsan,-asan}`, with corresponding
`build/resident-batch-timing-*.log` outputs. Configuration matches the preceding
lifecycle checkpoint: Apple Clang 17 on the recorded M4 Pro, optional libjxl
reference disabled, Release tests/benchmarks enabled. Sanitized builds target
the CPU scope, lifecycle and scheduling tests.

The full Release suite passes 104/105 tests, including the installed consumer.
Its only failure is the same `quantization_pipeline` golden mismatch reproduced
by the fresh `806830e` build (102/103): actual `0.24919039011001587`, expected
`0.24914586544036865`. Changed source/executable hashes are retained in
`build/resident-batch-timing-artifacts.sha256`; the shader bytes are unchanged.

Seven selected Release tests pass three repetitions each. The CPU scope and
CPU-only lifecycle/scheduling tests pass ten ThreadSanitizer repetitions each
and three unsuppressed ASan/UBSan repetitions each. The two Metal-inclusive tests
pass three ASan/UBSan repetitions each using only the existing
`null:*/third_party/metal-cpp/*` UBSan suppression; the
separate unsuppressed scheduling run reproduces
`Foundation/NSObject.hpp:112:49`. Leak detection is disabled, so these results
do not establish leak-sanitizer or unsuppressed Metal cleanliness.

CPU and fully-resident Metal benchmark smoke runs cover one/two/four/eight
images, two warmups and three samples on the 64x64 fixture. Every call checks
exact single-image bytes/summaries and bounds per-image readiness by complete
public-call wall time. Both emit the new CSV distributions. These are reporting
smoke checks, not independent-process performance qualification. The changed
runtime, benchmark and test units also pass strict
`-Wall -Wextra -Wpedantic -Werror` syntax checks.

Real parallel-loop launch-failure coverage and the final independent-process
comparison against the integrated preparation baseline remain outstanding.
The new benchmark columns are observability, not by themselves performance
evidence or a reason to declare the scheduling effort finished.
