# Managed host backing and CPU worker propagation

This checkpoint extends [Metal allocation accounting](resident-metal-accounting.md)
with a standard-container allocator, real bit-writer/image-plane attachments, and
propagation through the existing joined CPU workers. Its comparison parent is
`97bb46a`. It remains a partial implementation of milestone 4: token/model arrays,
candidate codestreams, batch retention and complete-work admission are not covered
merely by introducing an allocator.

## Allocation-owned tickets

`ManagedAllocator<T>` selects a domain at allocation time, not at construction of
the container. A new backing prepares a ticket before allocation, commits after
successful backing allocation, then places the ticket in an aligned private
header. The payload is correctly aligned even for over-aligned element types.
Multiplication, header addition and representable pointer differences are checked.
No global `new`/`delete` override or application-wide allocation interception is used.

The charge is the container allocator's `n * sizeof(T)` backing request. Logical
vector size, requested image extent and physical footprint are different quantities.
`clear` and logical rollback do not release retained capacity. During growth the
old and new allocations are both charged until the old backing is actually freed.
An element-copy exception destroys the new backing while retaining the original
vector and charge. An insufficient reservation cannot grow or use the fallback
domain. `ManagedAllocationFailure`, derived from `bad_alloc`, carries its original
status; bit writers preserve that status through their existing atomic operations.

Deallocation moves the ticket to the freeing thread's stack, destroys the header,
frees the backing, and only then releases the ticket. No producer thread, resource
scope or reservation wrapper is needed at that point. Container moves and swaps
retain the original allocation/domain; copies and growth allocate in the current
domain. Rebound node allocators use the same ownership rule. These are allocation
semantics, not an admission controller or permission to lend another job's storage.

The aligned ticket header, allocator bookkeeping and allocation-library rounding
remain explicit small-overhead exclusions. They are not represented as image or
writer payload capacity. C++ consumers must rebuild against changed internal
container implementations; C ABI option/result layouts are unchanged.

## Scope and actual production coverage

`ResourceContext` now includes host-accounting enablement. Top-level linear-RGB
encoding and standalone frame serialization install `ManagedHostScope`; an explicit
reservation also enables host accounting. An unconfigured encode uses the shared
unlimited default domain. Caller-created images outside such a scope remain
uncharged, and entering a scope does not retroactively charge borrowed input.

The covered production containers are:

| Owner | Coverage and lifetime |
| --- | --- |
| `BitWriter::storage_` | Every backing allocation under the encoder scope, including section writers, entropy-model writers and assembly writers. Fixed serializer classification happens only when allocating, not on individual bit writes. Allotments still reserve before callbacks and preserve logical bytes on failure. |
| `Image3FBuffer::planes_` | Each of its three owning plane backings when allocated inside an encoder/admitted scope. This includes prepared/reconstructed images using this type, not every raw float vector in the codebase. Replacement remains atomic even if allocation fails after creating only some new planes. |

The existing CPU helpers for color conversion, forward transforms, initial
quantization, coefficient ordering and codestream sections explicitly capture the
calling resource context and install it through `ParallelScope`. The constructor
requires that context, so worker calls cannot accidentally use a two-argument form
that reads only the fresh thread's defaults. All workers join before the borrowed
reservation can be reset. Nested serial work retains its calling context.
Participation counts and work scheduling policy are unchanged; this is not an
aggregate CPU admission limit. Batch workers still need per-job domain/admission
integration rather than inheriting one context at thread-pool construction.

## Permanent tests and failure boundaries

`managed_allocator` covers uncharged caller inputs, scoped default accounting,
capacity versus size, retained capacity after `clear`, over-alignment, allocator
rebind, copy/move/swap, cross-domain growth, overflow/zero allocation, injected
backing failure, and element-copy exceptions. Four concurrent workers use a shared
reservation and retain their allocations after joining; another thread then frees
them after the producer reservation is closed. Writer growth/allotment failures
preserve bytes. An image replacement deliberately exhausts its plan after one of
three new planes, verifies the old image and charge, then succeeds at a smaller size.

The serializer test compares single-thread reference bytes with four-participant
serialization for balanced, high-density and maximum-compression policies. It
checks managed writer activity, no surviving charges, one-byte-plan rejection,
unchanged caller output and subsequent recovery. The fixture's ample writer budget
is not a bound or estimator for the complete serializer's unconverted storage.

The initial full suite found a device-only resource fixture that constructed its
caller image inside the installed reservation. With real host-plane coverage this
correctly exhausted its device-only allowance. The fixture now creates that source
before entering its resource scope; no allowance was increased or numerical test
weakened. The final implementation also avoids a preliminary per-bit-write class
scope by using an allocator owner tag, and is rebuilt/requalified separately.

## Remaining allocation and admission work

The next attachments must cover owned storage, not just add labels around it:

- `ac_group.h` values, contexts, token templates, populations, anchors and nonzero
  maps; `dc_group.h` DC/metadata tokens and their owner arrays.
- `entropy.h`, `entropy.cpp` and `ans.cpp` model tables, reverse maps, populations,
  candidate/search storage and retained prepared entropy candidates.
- Coefficient-order counts, decisions, permutations and code/token scratch;
  serializer vectors of sections/models and candidate codestream bytes.
- Quantization preparation's raw quant/sharpness/mask arrays; CPU AC-search costs
  and metadata; completed-frame host metadata and other image-shaped vectors not
  represented by `Image3FBuffer`.
- C input conversion before the current workflow entry, and accumulating batch
  results before public ownership handoff. Ordinary public result vectors are
  not intercepted by the managed allocator.

Owner-independent input spans/views can preserve borrowing across converted
containers. Preserve deterministic model search and candidate retention; memory
pressure must not implicitly reduce encoding policy. Shared checked planners must
include simultaneous replacement allocations and policy/worker-dependent storage.
The public domain, cache-eviction/admission controller, retry and retained-result
reservations, and full workflow pressure/progress qualification remain mandatory.

## Qualification

The frozen `build/resident-managed-metal` parent was rerun and passes 65/66 Release
tests; the fresh `build/resident-managed-host` candidate passes 66/67. The sole
failure in each is CPU `quantization_pipeline`: actual `0.24919039011001587`, expected
`0.24914586544036865`. No golden or numerical tolerance is changed. The new allocator
and expanded serializer tests pass 25 Release repetitions and ten ThreadSanitizer
repetitions each. Four CPU-only ASan/UBSan targets pass without suppressions:
allocator, bit writer, serializer and core geometry. Five additional targets pass:
workflow, Metal AQ, GPU quantization pipeline, completed frame and Metal resources.
Those five use the existing vendor-header-only UBSan null suppression; all ASan
runs disable leak detection. These results do not establish leak-detector coverage.
The final domain-wrapper-lifetime case is included in the final full Release run
and 25 additional allocator-only repetitions in each Release, ASan/UBSan and TSan.

All 56 corpus/policy JXL SHA-256 comparisons match. Three separate pinned `djxl`
linear-RGB PFM comparisons match, and both builds pass all 22 conformance fixtures.
The decoder remains `e8ff09762481785938d8e4e01333ed3917571161`; successful conformance
scratch is automatically deleted, while logs/manifests are retained.

Measurements use Apple M4 Pro, 48 GiB, macOS 15.6; Release, SIMD/fused-tuned Metal,
fully resident, distance 1.2, effort 7 and automatic CPU threads. Tests/benchmarks
are enabled; libjxl-reference fixtures and compile-time Metal profiling are off.
The complete synchronous-call timer includes teardown, not backend creation,
input loading, hashing or output writing. The warm matrix has seven alternating
independent-process pairs per workload, each with 15 alternating original/changed
encodes and the first three discarded. Physical-memory and trim probes each have
three independent-process pairs and retain both source images. Builds and tests
finish before timing starts. Process checks do not establish exclusive machine
ownership or fixed thermal/power conditions.

### Complete-call results

Times are medians of seven process medians. Changes are medians of seven paired
ratios, not ratios of the displayed times; the signs can therefore differ. All
840 original/changed-image outputs retain identical sizes and hashes.

| Input | Metal-only accounting ms | Host attachments ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 22.857 | 23.091 | +0.87% | 3/7 |
| Padded 1080p, 1919x1079 | 70.603 | 70.998 | +0.01% | 3/7 |
| Padded 4K, 3839x2159 | 239.843 | 238.404 | +0.49% | 3/7 |
| Planter 4K, 3840x2160 | 255.415 | 256.239 | -0.06% | 4/7 |

Changed-image companions are +0.29%, +0.18%, +2.18% and +0.91%, respectively. The
padded-4K companion's seven paired changes range from -1.63% to +4.95%; its
quantization-stage paired median is +2.11%. Serializer-stage medians across all
eight images increase by 0.31% to 3.88%; the original padded-1080p tail increases
1.13% and is slower in all seven pairs. These are recorded overhead/variation, not
a speedup claim or proof of zero cost. The change is retained for actual resource
ownership and enforcement infrastructure, not as a latency optimization.

### Footprint and trim

In the no-trim cohort, median process peaks are 2975.517/2969.923 MiB
(parent/candidate). One-second backend-alive idle readings are 2209.673/1311.095 MiB,
but the separate trim cohort reads 2021.079/2022.189 MiB at the corresponding idle
boundary. Neither implementation changes the idle-cache capacity or reclamation
policy in this checkpoint. This variability does not establish a causal physical
memory saving from host accounting.

Both trim probes retain 1856.690 MiB of accounted idle capacity after encoding,
with zero live/unbacked capacity, and reduce idle capacity to zero on trim. The
candidate observed managed-backing peak is 2559.230 MiB versus 2555.773 MiB for the
parent: that comparison has expanded coverage, not 3.457 MiB of proven physical
growth. Many CPU allocations remain outside these counters.

| Trim-cohort boundary / operation | Parent | Candidate |
| --- | ---: | ---: |
| Physical footprint immediately after trim, MiB | 1847.017 | 1882.626 |
| Physical footprint one second after trim, MiB | 265.626 | 266.517 |
| Trim call, ms | 7.443 | 9.978 |
| First encode after trim, ms | 306.195 | 334.619 |

The first post-trim encode is slower in this small three-pair cohort. A separate
three-pair confirmation is retained rather than replacing these observations.
Its parent/candidate post-trim medians are 316.802/332.185 ms: the median paired
change is +3.56%, versus +9.64% in the initial cohort. Candidate resume is slower
in all three confirmation pairs and two of the original three. Trim-call medians
reverse direction in confirmation (18.154/8.958 ms), demonstrating why neither
an immediate trim-cost improvement nor a fixed regression should be inferred from
one small cohort. The repeated post-trim slowdown is retained as a measured cost;
these probes do not isolate its cause. All 48 outputs across the two cohorts match.
No whole-encoder footprint reduction or bounded memory under pressure is claimed.

### Artifacts and reproduction

`build/host-resource-qualification` retains `run.py`, `driver-build.json`,
`parity.json`, `conformance.json`, `full-workflow.json`, `memory.json`, `trim.json`,
`summary.json`, per-process warm samples, conformance logs and
`baseline-full-ctest.log`. `trim-confirmation/trim.json` and
`trim-confirmation-summary.json` retain the separate repeated trim cohort;
`summarize_trim_confirmation.py` checks its copied binaries and all 48 hashes.
Its driver sources are the unchanged
`build/handoff-qualification/driver.cpp` and
`build/metal-resource-qualification/trim_driver.cpp`. The manifest pins 14 static
libraries, four driver binaries and both sources; parent binaries are copied from
their previously qualified artifacts, not compiled using changed headers.

The summarizer recomputes paired statistics, verifies complete manifests and
timed output hashes, and checks all pinned library/driver/source hashes:

```sh
python3 build/metal-resource-qualification/summarize.py build/host-resource-qualification resident-managed-metal resident-managed-host
```

The candidate full/repeated logs are in `build/resident-managed-host`:
`final-ctest.log`, `host-final-repeat.log` and `allocator-final-repeat.log`.
RelWithDebInfo sanitizer builds are `build/managed-host-sanitize`
(`cpu-final.log`, `metal-final.log`, `allocator-final-repeat.log`) and
`build/managed-host-tsan` (`final-qualification.log`, `allocator-final-repeat.log`).
They use `-fno-omit-frame-pointer` with the respective `-fsanitize=address,undefined`
or `-fsanitize=thread`. Metal ASan/UBSan invocations use
`ASAN_OPTIONS=detect_leaks=0` and
`UBSAN_OPTIONS=halt_on_error=1:suppressions=.../build/handoff-qualification/ubsan.supp`;
CPU-only checks omit that suppression. TSan uses `TSAN_OPTIONS=halt_on_error=1`.
