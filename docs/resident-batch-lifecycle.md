# Explicit synchronous batch shutdown

This scheduling checkpoint follows `9bc7f24`. It closes the explicit
shutdown/drain gate in [milestone 6](resident-execution.md#6-coordinated-cpugpu-scheduling--in-progress).
The subsequent [timing](resident-batch-timing.md) and
[worker-launch](resident-worker-launch.md) checkpoints close those respective
gates; final integrated-baseline qualification remains outstanding.

## Public contract

`VarDctBatchEncoder::Shutdown()` permanently closes a driver and synchronously
joins its workers after draining the active `Encode` call. Activation is the
open-state check under the driver's encode mutex, before planning/admission.
A call already active there finishes normally, including its admission waits,
all images, individual errors, resource cleanup and atomic result publication.

Calls queued behind it, and calls arriving after closure, return `kUnavailable`
without changing their result arrays. A null output remains an invalid argument.
Repeated and concurrent shutdown calls are supported. Even an empty batch is
rejected after closure; `max_in_flight()` retains the configured value.

Shutdown is a drain, not cancellation. A resource-admission wait must still
become satisfiable; shutdown cannot revoke another caller's held memory or CPU
resources. Do not call it from work whose completion that same driver awaits.
In particular, do not hold a reservation needed by the active batch and expect
shutdown to free that reservation on the holder's behalf.

The driver object and all borrowed request/input storage must remain alive
through their applicable calls. Join all external API-calling threads before
destroying the object. Shutdown joins the driver's workers, not those external
threads; a queued caller may still be unwinding when another shutdown returns.

## Ordering and CPU participation

An atomic closing flag rejects incoming work promptly and is checked again
under the encode mutex. The separate worker-stopping flag is only set after
that mutex proves the active call has drained. Stopping workers sooner could
abandon the active call's images or leave its completion wait unsatisfied.

The shutdown caller yields any existing CPU participation for the entire
mutex wait and worker join, then unlocks before resuming participation. This
lets an active image finish even if the shutdown caller initially owns the
domain's sole CPU slot. Normal calls also yield participation while waiting
for the driver mutex. Startup-failure worker joins use the same yielding rule.
No image arithmetic, output format, admission plan or GPU dispatch is changed.

## Qualification

Permanent `batch_lifecycle` / `batch_lifecycle_cpu` tests cover:

- One/four driver workers sharing a one-slot CPU domain, CPU and fully-resident
  Metal; active-image drain with exact single-image bytes and summaries.
- A queued call and a new call rejected without replacing or changing sentinel
  result arrays; preserved individual invalid-image errors and result order.
- Two simultaneous shutdown calls, one initially holding the sole CPU slot;
  idempotent shutdown and rejection after the workers have stopped.
- Active calls waiting for CPU or finite managed-memory admission, released by
  the resource holder while shutdown waits.
- A whole-batch planner violation after activation and concurrent closure;
  unchanged caller output and complete resource cleanup on failure.

Tests use calling-thread/worker boundary hooks and bounded gates, not concurrent
object destruction. The installed downstream C++ consumer also links and calls
the new method, checks repeated shutdown and verifies retained results survive
a rejected encode.

Fresh Release, ThreadSanitizer and ASan/UBSan builds use the
`build/resident-batch-lifecycle{,-tsan,-asan}` directories; corresponding command
outputs are retained as `build/resident-batch-lifecycle-*.log`. They use Ninja,
Apple Clang 17, the M4 Pro configuration from the CPU checkpoint, and disable
the optional libjxl reference. The Release build enables tests and benchmarks;
sanitized builds target the lifecycle executable.

The fresh Release suite passes 102/103 tests, including the installed consumer.
The sole failure is the same `quantization_pipeline` mismatch reproduced in the
fresh `9bc7f24` build (100/101): actual `0.24919039011001587`, expected
`0.24914586544036865`. Neither suite is all-green. The changed sources and linked
lifecycle executables have a retained `build/resident-batch-lifecycle-artifacts.sha256`
inventory; the shader bytes remain unchanged from the build-ordering checkpoint.

Eight selected Release tests pass three repetitions each. The CPU-only
lifecycle test passes ten ThreadSanitizer repetitions and three unsuppressed
ASan/UBSan repetitions. The Metal-inclusive lifecycle test passes three
ASan/UBSan repetitions with only `null:*/third_party/metal-cpp/*` suppressed.
Its separate unsuppressed run reproduces the vendor
`Foundation/NSObject.hpp:112:49` null-member-call report. ASan leak detection is
disabled; this is not a claim of leak-sanitizer or unsuppressed Metal cleanliness.
The changed runtime and test translation units also pass strict
`-Wall -Wextra -Wpedantic -Werror` syntax checks.

The follow-up [timing checkpoint](resident-batch-timing.md) adds queue/service
measurement. Real loop launch-failure coverage and a repeated integrated-baseline
performance comparison remain required for final scheduling completion. These
functional checks alone are not a throughput or latency claim.
