# Shared resident device-storage plans

This is a further milestone-4 checkpoint, based on `dd5cd54`. It extracts the
device-capacity recipes needed by [whole-workflow admission](resident-resources.md)
and makes actual allocation consume those same plans. It does **not** introduce
public execution domains, a whole-workflow memory bound, or CPU/GPU scheduling.
Host/serializer bounds and admission remain required, not optional follow-up.

## Why the extraction precedes admission

Previously AQ and Butteraugli calculated arena sizes and then independently
repeated their plane geometry while slicing the allocation. AC search counted
candidates by traversing tiles, built host candidates, and only then derived
device scratch sizes. Completed-output sizing was embedded in output creation.
An admission estimator that copied those recipes would become another place
where capacities, aliases, and the actual implementation could disagree.

The new planners use geometry and resolved options only. Successful planning
requires no pixel reads, backend initialization, or heap backing. The normal
`Status` type can allocate a small diagnostic string on invalid input; this is
not a no-allocation guarantee for error reporting. All failed plans preserve
their output, and scratch append failure preserves both the output slice and
the previous layout end.

## Shared calculations and their boundaries

| Planner | What it determines | What it deliberately does not determine |
| --- | --- | --- |
| [Device plane/layout](../src/gpu/scratch.h) | Checked element/stride size, aligned offsets and arena end; minimal containing bytes exclude padding after the last row | Backend ownership and actual backing availability; alias lifetime safety |
| [Resident input](../src/gpu/metal/metal_storage_plan.h) | Original RGB, padded Opsin and preparation-result slices in one arena | Borrowed caller input or host staging outside that arena |
| AQ | Persistent/staging slices selected by borrowing, metric, frame-only, quantizer and filter flags | Resolve encoding policy or perform initial quantization/strategy search |
| Butteraugli | Owning full/subscale planes, reductions, Gaussian kernels and existing scratch/reference metrics | Charge borrowed AQ planes twice or validate their runtime backend/alias relationships |
| Completed frame | Independent group-major AC coefficients and the immediately following destination table | Construct final destinations or include host block metadata |
| [AC search](../src/gpu/ops/ac_strategy_storage_plan.h) | Per-family counts and device requests, the two maximum packed-transform buffers and maximum rate scratch | Existing larger retained capacities, replacement overlap, or CPU placement/model-search storage |

`DeviceScratchArena::BindPlane` binds a checked planned slice without allocating
or recomputing its geometry. It verifies the recorded byte span against the real
buffer. Ordering/non-overlap belongs to the plan; this is not an alias ownership
checker. AQ, resident input and Butteraugli also verify that their final bound
layout end equals their planned end, including when a larger cached arena is
reused. The independent tests check every named owning slice for aligned,
contiguous coverage without overlap.

AQ keeps its existing 32-bit coefficient-index limit. Resident-input planning
keeps its own less restrictive geometry checks rather than inheriting AQ's
coefficient-storage constraint. Completed output retains the existing bound on
`3 * group_count * 65536` coefficients. Its allocation size and destination
offset are shared with final output creation. Actual destinations still come
from authoritative post-search anchors, and completed output stays independent
of evaluator, scratch, cache and backend lifetime.

AC candidate counts use an exact integer factorization of the previous sum:

`sum_y sum_x (positions_x * positions_y) = sum_x positions_x * sum_y positions_y`.

Each axis consists of full eight-block tiles plus at most one partial tile.
This makes count planning constant work per family even for very large accepted
geometries. It does not reorder candidate generation, floating-point arithmetic,
cost evaluation, placement, or ties. Generation keeps the original tile/anchor
traversal, reserves the planned count, rejects an unexpected extra candidate
before vector growth, and checks the final count. Uploads and allocations use
the planned sizes. Host matrices still exist for empty families, while their
device buffers are not allocated, matching the previous behavior.

The device plans describe fresh minimum requests, not the actual capacity of
every already-retained allocation. Admission must still shed or explicitly
cover oversized cache/prepared entries and replacement-before-release. In
particular, summing AC `device_bytes` with arena requests is **not** a complete
encode envelope. Pre-search AQ/output planning can bound anchor count by block
count; actual allocation continues to use the existing resolved counts.

The later [AC-search host plan](resident-ac-search-storage-planning.md) adds
candidate/matrix/readback/dense-table owners, nonresident packing and CPU
placement/export, including vector-growth bounds for prepared reuse. Backend
submission/profile metadata and whole-workflow composition remain separate.

## Correctness qualification

The frozen parent is `build/resident-diagnostics` at `dd5cd54`; the fresh Release
candidate is `build/resident-storage-plans`. Both enable tests/benchmarks and
disable libjxl-reference fixtures and compile-time Metal profiling. The parent's
standalone drivers are copied from its verified historical qualification, not
recompiled against candidate headers.

Permanent CPU-only checks cover:

- 20,000 deterministic plane-layout cases against independent 128-bit arithmetic,
  including invalid types/strides, alignment and addition overflow, forged byte
  spans, and atomic plan/binding failure. The fake backend represents capacities
  without actually allocating huge buffers.
- 18,432 AQ flag/geometry/filter combinations against the frozen `dd5cd54`
  capacity recipe, plus all named slice coverage; six resident-input geometries
  and tiny/expanded/multiscale/borrowed Butteraugli layouts with independent
  capacity and metric formulas.
- 8,200 AC resident/nonresident geometries against the old tile-by-tile recipe
  and frozen family dimensions. This includes every 1–64 block width/height
  pair, empty families and larger thin/padded geometries. A near-32-bit-limit
  geometry is planned without allocating its very large backing.
- 147 completed-frame layouts across pixel/block/group boundaries, single and
  maximum anchor bounds, the accepted maximum group-index range, and atomic
  rejection immediately beyond it.
- Explicit one-byte reservations and armed managed-host failure hooks verify
  that successful planning consumes no managed backing or fallback-domain
  allocation. This complements source inspection; it is not an interception
  test for every global `operator new`.

During development, the independent AC fixture initially transposed rectangular
family dimensions. The frozen parent confirms that strategy names are
rows-by-columns while extents are width-by-height; the fixture was corrected
from that source. No production strategy, golden value or tolerance changed.

The full Release suites pass 72/73 parent and 74/75 candidate. The sole failure
is reproduced on both: CPU `quantization_pipeline`, actual
`0.24919039011001587`, expected `0.24914586544036865`.

All 56 corpus/policy cases preserve codestream SHA-256, including efforts 1–10,
resident/exact/throughput modes, final-score diagnostics, target-size retries
and maximum-error policy. Both builds pass all 22 pinned conformance fixtures.
Independent decoding of Kodak17, planter 4K and padded stress 4K preserves all
three linear-RGB PFM hashes. The pinned decoder revision is
`e8ff09762481785938d8e4e01333ed3917571161`.

Fresh ASan/UBSan builds pass the three CPU planner/range suites twenty times each
without suppressions, and six real Metal suites: AQ, Butteraugli, AC search,
completed output, resource accounting and diagnostic accounting. Those retain
borrow/reuse/trim/backend-destruction and injected-failure coverage. Only the
six Metal suites use the existing `null:*/third_party/metal-cpp/*` suppression
for Objective-C nil retain/release wrappers. Leak detection is disabled and both
sanitizers halt on errors. This checkpoint changes no synchronization mechanism;
it does not claim a new ThreadSanitizer qualification.

## Performance, capacity and physical-footprint qualification

Apple M4 Pro, 48 GiB, macOS 15.6 (24G84), AppleClang 17, arm64. Fully-resident
Metal uses SIMD/fused-tuned kernels, effort 7, distance 1.2 and automatic CPU
threads. No hard managed limit or new admission mechanism is enabled. The timer
covers the complete synchronous in-memory encode, including evaluator teardown,
but excludes input loading, backend creation, hashing and output writes.

Seven alternating independent parent/candidate process pairs cover each of four
workloads. Each process alternates the named source with a deterministically
changed companion, performs 15 encodes, discards the first three, and contributes
six observations per image. All 840 outputs preserve per-image byte count/hash.
The table reports medians of process medians and the median paired percentage
change, not a ratio of the displayed medians.

| Input | Parent ms | Candidate ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 23.455 | 23.197 | -0.23% | 4/7 |
| Padded 1080p, 1919x1079 | 70.888 | 70.763 | -0.02% | 4/7 |
| Padded 4K, 3839x2159 | 244.557 | 244.682 | +0.19% | 3/7 |
| Planter 4K, 3840x2160 | 257.633 | 258.329 | +0.26% | 2/7 |

Changed companions have paired changes +0.10%, -0.18%, -0.42%, and -0.38%,
respectively. These small mixed observations establish neither a consistent
speedup nor a guarantee of zero overhead.

A separate three-pair padded-4K physical-footprint probe retains two caller
images. Its 18 encoded outputs also match the warm workload hashes. These are
process physical counters, not allocation-capacity counters or a hard RSS cap.

| Boundary | Parent MiB | Candidate MiB |
| --- | ---: | ---: |
| Peak after encoding | 2974.689 | 2971.876 |
| One second idle, backend alive | 2206.001 | 2205.220 |
| One second after backend destruction | 254.017 | 249.907 |

Two separately retained three-pair trim cohorts verify all 48 output hashes and
the same exact managed counters in every process: peak backing **2,922,164,054
bytes** (2786.793 MiB), idle capacity **1,946,881,084 bytes** (1856.690 MiB) before
trim, and zero idle capacity afterward. Live and reserved-unbacked capacity are
zero at the measured idle boundaries. The one-second post-trim physical
footprints are 263.157/262.376 MiB parent/candidate in the original cohort and
261.439/262.173 MiB in confirmation. Freed managed capacity does not imply an
immediate equal reduction in physical footprint.

Explicit trimming was slower for the candidate in both small cohorts; retain
that cost rather than dismissing it because capacity is identical. Original
parent/candidate trim medians are **6.014/11.700 ms**, with individual values
`[6.014, 19.222, 0.048]` / `[11.700, 18.551, 5.871]`. Confirmation medians are
**12.532/18.942 ms**, values `[6.288, 12.532, 18.029]` /
`[19.191, 18.942, 16.862]`. Resume-encode medians are 314.836/318.105 ms and
318.821/332.636 ms, respectively. The backend trim/cache implementation is
unchanged, and both builds have identical compiled Metal-library hashes; this
experiment does not establish the cause of those differences.
It is not evidence of zero trim overhead. Admission/pressure qualification must
include eviction and resumed work, not substitute the warm-stream results.

The existing batch benchmark supplies a separate three-pair regression probe,
with three warmups and five samples, two changed inputs, batch sizes 2/4 and
the same Metal policy. Within each build it verifies exact batch/sequential
bytes and summaries; the cross-build probe compares reference byte counts, not
retained full batch-output hashes.

| Workload / batch size | Parent ms | Candidate ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| 256x192 / 2 | 14.774 | 14.662 | -0.76% | 3/3 |
| 256x192 / 4 | 24.267 | 24.640 | +1.58% | 1/3 |
| 1080p / 2 | 116.362 | 114.636 | -1.48% | 2/3 |
| 1080p / 4 | 204.550 | 204.788 | -0.31% | 2/3 |

These mixed small-cohort results are measured costs, not milestone-6 throughput
qualification. The benefit of this checkpoint is having checked shared planning
inputs for admission; it does not change the existing scheduling policy.

## Artifacts and reproduction

Local evidence is under `build/storage-plan-qualification`: parent/candidate
test logs, `planners.log`, `driver-build.json`, retained JXL/PFM outputs and input
hashes, conformance commands/logs, `full-workflow.json`, `memory.json`, `trim.json`,
`trim-confirmation/trim.json`, `batch.json`, recomputed summaries and verifiers.
These are local qualification artifacts, not installed tools or durable corpus
fixtures. Permanent planner tests and the frozen AQ capacity oracle live under
`tests/`.

`summarize.py` verifies sixteen library hashes, four drivers, both driver-source
hashes, both encoder binaries, all retained corpus/decoded/input files,
conformance logs and pinned decoder/info hashes, all timed hashes and the eighteen
physical-probe hashes. `verify_probes.py` additionally verifies both trim cohorts'
drivers, all 48 probe outputs, exact managed counters and batch binary/source
manifests. `validation.sha256` pins relevant test executables and logs.

Sanitizers use the separate `build/storage-plan-sanitize` RelWithDebInfo tree,
`-fsanitize=address,undefined -fno-omit-frame-pointer` and
`-O2 -gline-tables-only -DNDEBUG`. Their logs are `sanitize-planners.log` and
`sanitize-metal.log`. Changed core/planner/AC-search sources and planner tests
pass AppleClang `-Wall -Wextra -Wpedantic -Werror` syntax checks (`warnings.log`).

After building the candidate and retaining the frozen parent:

```sh
python3 build/storage-plan-qualification/run.py build
python3 build/storage-plan-qualification/run.py parity
python3 build/storage-plan-qualification/run.py conformance
python3 build/storage-plan-qualification/run.py performance
python3 build/storage-plan-qualification/run.py memory
python3 build/storage-plan-qualification/run.py trim
python3 build/storage-plan-qualification/run.py trim-confirmation
python3 build/storage-plan-qualification/batch.py
python3 build/storage-plan-qualification/summarize.py
python3 build/storage-plan-qualification/verify_probes.py
```

Never rebuild the frozen parent from the now-refactored source or against its
new headers. No executable-source logic changed after the qualified libraries
were built; subsequent source edits only clarified planner comments.

## Remaining milestone-4 work

Combine these shared device recipes with source-backed host/frontend,
token/model/writer, diagnostics, replacement and retained-candidate capacity
bounds. Resolve options before work admission, propagate a shared execution
domain through C/C++ callers and workers, reserve the full image envelope,
evict obstructing idle capacity before waiting, and pre-reserve aggregate batch
results with a progress rule. Qualify under-budget rejection, concurrent callers,
mixed geometry, retries, fairness and failure recovery against real allocations.

Milestone 5 still needs its source-aware last-use reductions and the final
reuse/fusion inventory dispositions. Milestone 6 still needs aggregate CPU
participation and admission-aware scheduling, with measured queue/service
latency and throughput. Pure device plans complete neither milestone.
