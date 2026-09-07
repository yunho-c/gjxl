# Partial worker-launch failure qualification

This checkpoint follows `0683407` and closes the actual loop launch-failure gate
in [milestone 6](resident-execution.md#6-coordinated-cpugpu-scheduling--in-progress).
The final comparison against the integrated preparation baseline is still required.

## Scope and preserved behavior

`worker_launch_internal.h` provides a caller-thread-local, one-shot fault at the
existing `std::thread` construction boundary. A selected loop throws immediately
before constructing worker 0, 1, 2 or 3; earlier workers in that same loop are
genuinely constructed. This is deterministic exception injection, not exhaustion
of operating-system resources. No fault is automatically inherited by workers.
With no installed fault, the wrapper forwards directly to `emplace_back` after
the thread-local check. It does not change worker reservation or fallback policy.

| Construction site | Public route exercised | Existing failure behavior |
| --- | --- | --- |
| Color rows | Native CPU | Join partial workers; serially recompute on `system_error`, return OOM on `bad_alloc` |
| Initial quantization | Native CPU | Same serial fallback / OOM split |
| Prepared forward transforms | Exact-coefficient Metal; also direct CPU-domain component test | Same serial fallback / OOM split |
| Coefficient orders | Native CPU and fully-resident Metal | Join partial workers; return internal error or OOM |
| Serializer sections | Native CPU and fully-resident Metal | Join partial workers; return internal error or OOM |
| Batch driver startup | Four-worker driver creation | Stop and join partially created workers; return internal error or OOM |

The prepared-forward helper is used by the **exact-coefficient Metal** workflow,
not the native CPU workflow. The direct component test supplies its own real
storage admission, execution-domain scope and per-image CPU budget. Fully-resident
Metal launch tests cover its actual CPU coefficient-order and serializer tail.

## Permanent coverage

`worker_launch_failure_test.cpp` has 107 injected-failure cases, plus recovery
encodes. Its deterministic 320x272 fixture crosses the frontend parallel threshold
and contains four AC groups. Effort 7 reaches coefficient-order construction.
The reference uses one CPU participant; tested per-image settings are automatic
and four, with an aggregate domain limit of four.

- Six batch-startup cases cover both exception types after zero, one or three
  successfully constructed workers. The caller holds the domain's sole CPU slot;
  cleanup must yield and recover that slot, and a subsequent creation must work.
- Twelve direct prepared-forward cases cover both exception types and zero, one
  or two earlier workers at both thread settings. Fallback/recovery coefficients
  are byte-exact and transform metadata matches. Failure preserves the old owner.
- Forty-eight native CPU and 36 Metal public-call cases cover the applicable
  sites with the same exception/partial-launch/thread-setting matrix. They check
  exact bytes and summaries after fallback and recovery, and unchanged public
  bytes, summary and timing records on failure.
- Five gated batch cases close the driver while an image has partially launched
  workers and another caller is queued. Shutdown drains both active images,
  isolates an image error where appropriate, rejects the queued call without
  changing its result, and retains the admitted image's service timing.

Every completed case trims idle storage and checks zero live reservations,
waiting requests, managed capacity, and active/reserved/suspended/waiting CPU
participants. The peak protected CPU count must remain within the domain limit.
Recovery uses the same domain; successful error handling cannot merely leave it
unusable for subsequent work.

CTest partitions the complete matrix into seven `worker-launch-cpu` groups and
four `worker-launch-metal` groups. Each group has a 600-second instrumentation
allowance, while synchronization gates retain independent ten-second deadlines.

## Qualification

Fresh Ninja Release builds are retained at
`build/resident-launch-failure{,-tsan,-asan}`. The ordinary build enables tests
and benchmarks; sanitizer builds enable tests. Optional libjxl reference and
Metal profiling are disabled. Hardware is the Apple M4 Pro / Mac16,7, 48 GiB,
14 logical CPUs, macOS 15.6 (24G84), Apple Clang 17 (`clang-1700.6.4.2`).

- Final Release: **115/116 tests pass**, including all eleven launch groups and
  the installed consumer. The only failure is the previously reproduced
  `quantization_pipeline` golden: actual `0.24919039011001587`, expected
  `0.24914586544036865`. No numerical tolerance is changed.
- ThreadSanitizer: all seven CPU groups pass, 864.37 seconds total. C++ and
  Objective-C++ use `-fsanitize=thread -fno-omit-frame-pointer`.
- ASan/UBSan: all seven CPU groups pass unsuppressed, 342.56 seconds total.
  All four Metal groups pass with the existing
  `null:*/third_party/metal-cpp/*` UBSan suppression, 6.67 seconds total.
  These builds instrument C++ with
  `-fsanitize=address,undefined -fno-omit-frame-pointer`; Objective-C++ flags
  are unchanged. Leak detection is disabled. The separate unsuppressed Metal
  run reproduces the null-member-call report at `Foundation/NSObject.hpp:112:49`.
  This is not a claim of leak-sanitizer or unsuppressed Metal cleanliness.
- All changed runtime units and the new test pass
  `-Wall -Wextra -Wpedantic -Werror` syntax checks.

An initial monolithic CPU test exceeded its 180-second allowance under both
sanitizers; a later monolithic TSan retry was explicitly stopped to partition
the matrix. Neither attempt counts as a pass. The completed split runs above
exercise every case and are the qualification evidence.

Raw logs are `build/resident-launch-failure-final-ctest.log`,
`build/resident-launch-failure-final-strict.log`,
`build/resident-launch-failure-tsan-groups.log`, and
`build/resident-launch-failure-asan-{cpu-groups,metal-groups,metal-unsuppressed}.log`.
Source, executable, library, build-cache and log identities are retained in
`build/resident-launch-failure-artifacts.sha256`. The shader is unchanged at
SHA-256 `9dcbc4dff15fd81c0793a1df3d68fd06ac821d0f34929a0e397f1de4e9cafae4`.

These checks qualify exception cleanup, ownership and shutdown transitions.
They do not substitute for the remaining whole-call latency, throughput and
memory-pressure comparison against the integrated baseline.
