# Shared worker orchestration

Color conversion, initial quantization, prepared forward transforms, coefficient
ordering and serialization share `core/parallel_work_internal.h`. Their wrappers
still select participants, reserve a `CpuWorkerGroup`, and handle serial and
nested execution. The runner owns the repeated parallel task loop, context
propagation, launch boundary, joining and ordered status collection.

## Stage policies

| Stage | Thread-launch `system_error` | Task `length_error` |
| --- | --- | --- |
| Color conversion | Join, then retry serially | Internal worker error |
| Initial quantization | Join, then retry serially | Internal worker error |
| Prepared forward transforms | Join, then retry serially | Invalid argument |
| Coefficient ordering | Join, then return internal error | Out of memory |
| Serialization | Join, then return internal error | Out of memory |

Each wrapper supplies literal error descriptions, preserving its existing
messages without allocating policy storage on success. Managed allocation
failures preserve their original status and resource-plan classification;
ordinary allocation failures remain out-of-memory errors. Setup allocations
remain outside the launch exception boundary and reach the existing component
handlers.

## Ownership and execution invariants

- The runner borrows an admitted group until every started worker has joined.
  It uses the existing `ParallelScope`, `LaunchWorker` and `JoinCpuWorkers`
  primitives, including their resource context and CPU participation behavior.
- Spawned workers keep indices `[0, spawned_worker_count)`. A participating
  caller uses the next index. Automatic unmanaged calls can launch every
  participant; explicit limits and managed calls include the caller.
- Per-task status storage and the reserved thread vector keep their existing
  container types and allocation owners. Component planners retain their
  capacities and lifetime composition. No task queue or persistent pool is added.
- Task failures do not cancel other tasks. After joining, the first error in
  task-index order wins, independently of completion order.
- Launch failure stops further task acquisition and joins all started workers
  before returning or retrying. Joining suspends CPU participation and resumes
  it before serial retry. Retry executes the entire range with the original
  serial exception boundaries, while the status/thread storage remains alive.
- Unexpected launch exceptions also join before propagating. Typed allocation
  failures at launch are handled before the broader `bad_alloc` catch.

The helper is private and excluded from the installed-header manifest. Batch
driver orchestration, GPU submission, stage thresholds and numerical kernels
remain separate.

## Qualification

The extraction uses `4a1e63c` as its pre-change baseline. Build artifacts and
the source-hash record live under `build/worker-orchestration/`.

The focused `parallel_work` test exercises worker/caller indices, inherited
context, admitted participant limits, both allocation-owner types, ordered
errors, typed allocation failures and synchronized partial-launch cleanup. It
also verifies that an unexpected launch exception cannot abandon started
workers. Existing launch-failure integration tests retain their fallback
expectations and cover actual CPU/Metal encoding, output atomicity, domain reuse
and shutdown/drain behavior.

Both C++20 and C++23 Release suites pass 119 of 120 tests, including every
launch-failure case and the installed consumers. The sole failure is the
existing `quantization_pipeline` score mismatch, reproduced with the untouched
baseline binary and identical actual/expected values. The focused runner test
also passes ASan/UBSan.

All 1,440 planning records match the baseline exactly. All 56 baseline/candidate
encodes are byte-identical, all three pinned decoded-pixel comparisons match,
and all 22 pinned conformance fixtures pass. These checks establish the tested
behavior and reservation bounds; no latency or physical-memory improvement is
claimed for the extraction.
