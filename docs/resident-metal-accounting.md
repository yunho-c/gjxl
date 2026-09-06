# Metal allocation accounting and all-pool trim

This is the first production attachment of the [resource primitive](resident-resources.md),
following `b6bc044`. It is a partial implementation of resident-execution milestone 4,
not whole-workflow admission. The frozen runtime comparison is the integrated
preparation/handoff checkpoint `ec4d4c5`. Encoding policy and shaders are unchanged.

## Ownership and accounting boundary

`MetalBackend::Allocate` prepares a capacity ticket before requesting a shared
`MTL::Buffer`, commits it only after backing and its C++ owner exist, and publishes
the caller's output last. Allocation failure restores reservation credit and
preserves the previous output. `MetalBuffer` owns the ticket, with member order
that frees the Metal handle before releasing its charge. Buffers and their charges
can outlive the evaluator, backend, reservation, and domain wrapper.

Accounting covers buffers created through this entry point, including resident
input, AC search, AQ arenas, Butteraugli arenas, and completed AC output. Borrowed
planes and frame views do not carry additional charges. Class scopes label the
backing owner; unlabelled low-level allocations remain explicitly unclassified.
The immutable DCT basis created during backend construction is excluded setup
storage, as are pipeline objects and Metal command/driver internals.

Capacity here is the requested Metal buffer length / `DeviceBuffer::size_bytes`,
not Metal page rounding or physical footprint. Idle volatile buffers retain their
full capacity charge even when macOS reclaims their pages. `Empty` is detected on
reacquisition and that buffer is released before fresh allocation.

The internal `ResourceContext` propagates a borrowed reservation and owner class
through explicit synchronous scopes. Tests also install it in joined workers;
the owner must keep the reservation stable until those workers finish. With an
installed reservation, an insufficient plan fails before backing allocation and
cannot fall back to another budget. Without one, each allocation gets a ticket
in the same process-wide unlimited default domain. This default collects real
Metal capacity but does **not** admit or limit a complete encode. Production
workflow worker propagation and public domain configuration remain unfinished.

## Cache transitions and trim

An idle backing keeps its original domain. Same-domain acquisition transfers it
into the new reservation and marks it live without double-counting capacity.
Cross-domain or over-capacity acquisition drops it before requesting a fresh
backing. The actual retained capacity, not a smaller requested layout, must fit
the reservation. Successful return marks the backing idle; failure and teardown
paths release it normally. Image-specific reference state and borrowed AQ views
are never cached as reusable contents.

One backend cache mutex and one generation now cover all three AQ/resident-input
pools and the Butteraugli pool. `TrimPreparationCache`, the shared-workflow trim,
and the C trim API release all four idle pools. They do not initialize Metal or
wait for active submissions. An active lease acquired before trim cannot refill
the pools afterward; a later acquisition may cache again. Active work and
independently owned completed output remain valid. Trim itself may take time to
release buffers; "does not wait for active submissions" is not a latency bound.

Cache transitions may acquire the ledger lock while holding the cache lock.
The ledger invokes no cache callbacks. The future admission controller must
request eviction outside the ledger lock, preserving this lock order.

The pre-existing AQ per-arena caps and Butteraugli per-backend/process idle caps
are unchanged and still independent. This checkpoint adds their common accounting
and all-pool trim, not automatic domain-wide eviction or a combined memory cap.

## Correctness and failure qualification

The new permanent `metal_resource_budget` test exercises:

- Two backends sharing default accounting; aliases counted once; buffers retaining
  charges after backend destruction; nested class/context restoration.
- Explicit reservation exhaustion before allocation, deterministic backing failure
  after ticket preparation, unchanged outputs, and restored credit.
- Real resident-input and Butteraugli cache hits, cross-domain misses, forced
  `Empty` reclamation, successful recovery, active-lease trim, and backend teardown.
- Four simultaneous resident-input preparations under one allowance, with one
  distinct backing per active lease and no pre-trim return repopulation.
- A complete forced-Metal resident encode, classified retained idle capacity,
  all-pool trim, failure before submission, and byte-identical subsequent recovery.

The completed-frame test checks that retained device outputs keep their charges
after evaluator/backend destruction and while serializing. Its small host metadata
is not yet attached. Existing AQ/completed-frame lifetime tests now assert that
trim clears all pools, not just Butteraugli.

All 56 corpus/policy JXL SHA-256 comparisons match the frozen integrated runtime.
Separate pinned `djxl` decoding of Kodak17, planter 4K and padded stress 4K yields
matching linear-RGB PFM hashes. Both builds pass all 22 pinned conformance fixtures;
decoder revision is `e8ff09762481785938d8e4e01333ed3917571161`. Successful conformance
scratch files are removed by that test; logs and the manifest are retained.

The candidate full Release suite passes 65/66. The only failure is the same CPU
`quantization_pipeline` golden mismatch: actual `0.24919039011001587`, expected
`0.24914586544036865`; no numerical tolerance or golden changed. The frozen parent
was rerun and passes 63/64 with that same failure. AQ, completed-frame and Metal-resource tests
pass ten Release repetitions each. Separate ThreadSanitizer resource and
Metal-resource targets pass ten repetitions each with no suppressions. Five
ASan/UBSan targets pass: Metal resources, AQ, completed frame, Butteraugli and GPU
quantization pipeline. Leak detection is disabled; UBSan suppresses null checks
only in the vendor `third_party/metal-cpp` nil retain/release wrappers, as in the
integration qualification. No GJXL source checks are suppressed.

## Complete-call timing

Apple M4 Pro, 48 GiB, macOS 15.6; Release, SIMD/fused-tuned Metal, fully resident,
effort 7, distance 1.2, automatic CPU threads. Tests/benchmarks are enabled;
libjxl-reference fixtures and compile-time Metal profiling are disabled. The timer
includes the synchronous encode and evaluator teardown, excluding backend creation,
input loading, hashing and output writes. Each of seven alternating independent
process pairs runs 15 alternating original/changed-image encodes; three are warmup
and six observations per image remain. All 840 output sizes/hashes match.

Times are medians of process medians; percentage changes are medians of paired
ratios, not ratios of the displayed medians. Negative means lower latency.

| Input | Integrated parent ms | Accounted candidate ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 23.572 | 23.688 | +0.29% | 2/7 |
| Padded 1080p, 1919x1079 | 73.454 | 73.511 | +0.10% | 3/7 |
| Padded 4K, 3839x2159 | 246.729 | 246.720 | -0.13% | 4/7 |
| Planter 4K, 3840x2160 | 251.756 | 251.535 | -0.36% | 5/7 |

Changed-image companions range from -0.25% to +0.19%. This cohort does not show a
material complete-call timing effect; it establishes neither a speedup nor zero
overhead for all workloads. No other GJXL encoder/test was observed at process
boundaries; the machine was not exclusively reserved or power/thermally fixed.

## Physical footprint, trim cost and recovery

Separate three-pair padded-4K probes retain two source images. Counters are process
physical footprint in MiB, not managed capacity or free system RAM. Without trim,
median peaks are 2973.611/2971.283 MiB (parent/candidate), and one-second backend-alive
idle values are 2209.986/2210.673 MiB. Accounting alone does not reduce live storage.

The explicit-trim probe encodes A/B/A, waits one second, trims, observes immediately
and after another second, then encodes A again. All 24 output sizes/hashes match.

| Boundary or operation | Integrated parent | Accounted candidate |
| --- | ---: | ---: |
| Physical footprint before trim, MiB | 2021.751 | 2020.267 |
| Physical footprint immediately after trim, MiB | 2021.751 | 1899.204 |
| Physical footprint one second after trim, MiB | 264.626 | 264.798 |
| Trim call, ms | 0.122 | 15.060 |
| First encode after trim, ms | 328.907 | 350.771 |

Candidate accounted live capacity is zero after each completed encode; its idle
capacity is 1856.690 MiB. Trim makes idle capacity exactly zero, without pending or
unbacked reservations. Resuming repopulates it; backend destruction releases it.
The observed managed-backing peak is 2555.773 MiB, a different quantity from
physical peak. CPU allocations are not in that managed counter yet.

All-pool trim releases more owned storage, costs more, and makes the next encode
slower in this small probe. Both builds reach essentially the same one-second
post-trim physical footprint. Immediate capacity release must not be sold as an
equal immediate physical-footprint reduction. These costs support explicit idle
or pressure-triggered trim, not unconditional trimming between images.

## Artifacts and remaining work

Local artifacts are in `build/metal-resource-qualification`: `run.py`, `summarize.py`,
`trim_driver.cpp`, `driver-build.json`, `parity.json`, `conformance.json`,
`full-workflow.json`, `memory.json`, `trim.json`, `summary.json`, conformance logs,
and `baseline-full-ctest.log`. The unchanged complete-call driver and shared harness
are in `build/handoff-qualification`. The manifest pins all 14 static libraries,
four driver binaries and both driver sources; the summarizer rechecks their hashes,
completion status and timed output hashes. No older baseline is rebuilt in place.

Candidate Release artifacts and logs are in `build/resident-managed-metal`
(`full-ctest.log`, `repeat.log`). Sanitizer builds are `build/managed-metal-sanitize`
and `build/managed-metal-tsan`, each with `qualification.log`. The sanitizer flags
are the corresponding `-fsanitize=address,undefined` or `-fsanitize=thread`, plus
`-fno-omit-frame-pointer`, with RelWithDebInfo. The ASan/UBSan invocation uses
`ASAN_OPTIONS=detect_leaks=0` and `UBSAN_OPTIONS=halt_on_error=1:suppressions=.../build/handoff-qualification/ubsan.supp`.

Milestone 4 still requires checked shared planners, managed host allocation
adapters including serializer/candidate bytes, public domains, production worker
propagation, complete-work admission and idle eviction, retries, retained batch
results, and end-to-end exhaustion/fair-progress tests. Milestones 5 and 6 retain
their lifetime/reuse audit and aggregate CPU/GPU scheduling requirements. This
Metal-only attachment does not discharge those requirements.
