# Bounded volatile preparation cache

This change builds on `e1010fe` (shared Butteraugli/AQ scratch and deferred host
preparation). It retains allocation capacity between images. Every preparation
still rebuilds the plane layout, Gaussian kernels, reference psychoimage and
reference masks; image data, options, plane views and borrowed AQ pointers are
never reused as prepared state.

## Ownership and limits

Each Metal backend owns at most one idle Butteraugli arena. The default
`MetalBackendOptions::butteraugli_cache_bytes` is 1 GiB; zero disables caching and
smaller values lower that backend's admission limit. An additional atomic budget
caps the sum of idle Butteraugli capacity across all live backends at 1 GiB.
Larger active encodes remain supported; arenas exceeding either limit are freed
at return. This is an additional-cache limit, not a total encoder memory budget:
active allocations and the existing AQ arena pools are outside this accounting.

An idle allocation is `MTL::PurgeableStateVolatile`. Acquisition takes exclusive
ownership and makes it nonvolatile before any CPU or GPU access. If Metal reports
`Empty`, the allocation is discarded and preparation allocates afresh. Capacity
may be reused for a smaller layout up to twice the requested size. Larger
high-water capacity is shed, and insufficient capacity is freed before growing.
Concurrent returns favor the smaller arena; budget exhaustion simply bypasses
caching. There is no global LRU or cross-backend buffer transfer. An idle backend
can occupy the budget until it reuses, trims or destroys its allocation, even
when the OS has reclaimed that allocation's pages.

Only successfully prepared, valid operations may return capacity. Failed
preparation, submission, completion or operational readback invalidates reuse.
AQ waits for outstanding submissions, forwards its invalidation to Butteraugli,
then destroys the borrower before returning the backing AQ arenas. The cached
object contains only its owning allocation and scratch-layout counters.

## Trimming persistent encoders

`GpuBackend::TrimPreparationCache()` releases a particular backend's idle
Butteraugli capacity. `TrimVarDctPreparationCache()` and the additive C API
`gjxl_trim_preparation_cache()` reach the process-wide production Metal backend
shared by single-image, batch and C API encoders. They are safe alongside
independent encodes and do not initialize Metal or wait for active submissions.
The backend publishes its pointer only after successful initialization.

Trimming advances a generation and frees idle capacity under the cache mutex.
Active leases remain nonvolatile and valid, but leases acquired before the trim
cannot repopulate the cache on return. New leases can cache again. Applications
can call the trim API when becoming idle or responding to their own memory
pressure notification. This does not trim the existing AQ pools, pipeline states
or application-owned frames. Destroying an explicit backend also releases its
cache and process-budget accounting. Destroying a C API context or batch driver
does not destroy the shared production backend.

## Validation and evidence

Evidence is retained in `build/volatile-cache-qualification` in the
`gjxl-metal-preparation` worktree. The comparison baseline is the frozen,
previously qualified Release driver built from the source committed as `e1010fe`.
The candidate uses the same complete-encode boundary, distance 1.2, effort 7,
fully resident AQ, tuned SIMD Metal kernels and automatic CPU thread budget.
Backend construction, image loading, output hashing and file I/O are outside the
timed boundary; evaluator teardown is included.

Tests cover changed-image/options equivalence to an uncached backend, grow and
shrink behavior across expanded/single-scale/multiscale layouts, disabled and
undersized budgets, forced Metal reclamation, preparation/comparison failures,
AQ poison recovery and destruction with an outstanding submission. Trimming
active BA and AQ leases preserves results and prevents their return to cache.
Three simultaneously live backends race to return real reference allocations;
only the two fitting the global budget are retained, and destruction returns
the accounting to zero. The public workflow trim is checked before initialization
and between two byte-identical encodes. Both C and C++ entry points are exercised.

The implementation introduces no kernel or encoder-policy change. Existing AQ
resource tests now allow zero fresh allocations when all three arenas are warm;
no numerical tolerances or golden values are changed.

The final Release suite passes 61/62 tests. The sole failure is the same
pre-existing CPU `quantization_pipeline` golden mismatch as #1: actual
`0.24919039011001587`, expected `0.24914586544036865`. Targeted ASan/UBSan runs
pass 3/3 suites. Leak detection is disabled; UBSan null checks are suppressed
only in `third_party/metal-cpp` headers, whose nil retain/release wrappers trip
those checks before encoding. The unsuppressed failure and suppression are
retained. No suppression applies to GJXL source.

An additional stress driver runs 16 concurrent padded-4K encodes on shared and
independent backends, including continuous trimming during active encodes and
repeated one-second idle/resume cycles. Every output equals its uncached oracle;
observed idle Butteraugli capacity stays within the process budget. These are
correctness/lifetime stress results, not a concurrent-throughput benchmark.
Actual OS-induced memory pressure was not forced; reclamation recovery was
exercised through Metal's real `Empty` state.

## Measured performance and memory

The primary run contains 200 accepted independent processes and 2,614 encodes.
Every retained JXL is hashed and checked for equality across variants and
repetitions. Pinned `djxl` v0.13.0 at `e8ff0976` accepted 72 retained outputs;
the decoded linear-RGB PFM hashes match for all 36 natural images. Decode logs,
commands and hashes are retained; temporary PFMs are removed to bound disk usage.

Each natural-image process alternates two different images for nine encodes;
indices 0–2 are warmup and indices 3–8 are retained. Three independent process
pairs alternate baseline/candidate order. Reported milliseconds are medians of
per-image medians; percentage reductions are medians of paired per-image/process
reductions, so they need not equal the ratio of the two displayed medians.
Hardware is Apple M4 Pro, 48 GiB, macOS 15.6. These are Metal-only comparisons.

| Natural corpus | #1 baseline | With cache | Paired reduction | Improved pairs |
| --- | ---: | ---: | ---: | ---: |
| 24 Kodak images | 24.48 ms | 23.48 ms | 4.50% | 71/72 |
| Six 1080p photos | 80.61 ms | 74.02 ms | 8.06% | 18/18 |
| Six 4K photos | 261.13 ms | 237.08 ms | 9.86% | 18/18 |

These gains are incremental to #1; they are not a fresh comparison with the
older fusion-only branch. The 4K allocation retained by these AQ workloads is
about 831 MiB. Cache-disabled candidate runs differ from #1 by a median paired
0.20%, supporting allocation reuse as the source of the warm-stream gain.
Efforts 4 and 8 at 1080p improve by 8.90% and 8.01%; distance 1.0 improves 6.95%.
No shader changes are included.

Benefits depend on reuse. Fresh-backend 4K encodes are 2.06% slower in this run.
Alternating 1080p/4K geometry is 3.82% slower: the downsizing rule deliberately
sheds the larger arena and misses on the next growth. This bounds high-water
retention relative to current work, at the cost of this resizing pattern.
After a five-second idle interval, the next 4K encode improves only 3.00%
across three process pairs, despite retaining the allocation. Cache-hit latency
after idle is not the same as continuously warm latency.

The explicit-trim and forced-Empty runs recover correctly with fresh allocations
and unchanged output. Their injection calls occur before the encode timer;
their timing rows are recovery probes, not total-lifecycle speedup evidence.

At five seconds idle, with the backend alive, median reported physical footprint
is 268.0 MiB for #1 and 266.9 MiB with the cache. At one second idle after resuming,
those medians are 1,194.0 and 2,025.6 MiB: an additional 831.5 MiB transiently.
Median process peak footprint is effectively unchanged (2,971.0/2,971.5 MiB).
These are task physical-footprint counters at explicit boundaries, not claims
that volatile capacity consumes no physical RAM. It remains reclaimable and can
be explicitly trimmed; the existing AQ pools are independent of this cache.

Very small forced-Metal inputs are noisy and include regressions. The primary
127x129 case is 5.72% slower, while 767x511 improves 4.87%. Below that range,
individual process pairs swing by tens of percent in both directions. A
same-binary on/off follow-up uses seven pairs and 101 encodes per process,
discarding the first 33. For 3x7, 8x8 and 17x19, median paired reductions are
-8.26%, -17.58% and +15.26%, with wide sign-changing ranges. This does not establish
a general small-image win or a reliable size threshold. The three tiniest cases
are below the existing automatic-Metal geometry gate; forced Metal still permits
them. A further seven-pair on/off run at 127x129 improves by 0.84% (six of seven
pairs; range -0.06% to +3.77%), so that primary slowdown did not reproduce as a
persistent cache penalty. No new size gate or encoder policy change was
introduced.

Another worktree was intermittently building and benchmarking during this
session. The harness rejects trials when it observes another GJXL test/benchmark,
compiler or decoder and waits for a quiet interval before retrying. Rejected
attempts are recorded. This reduces observed overlap, but is not an exclusive
machine or controlled power-state experiment; tiny-image variability remains a
qualification limit. An initial interrupted pilot is excluded from the results.
