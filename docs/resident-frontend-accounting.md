# Preparation, evaluation and completed-frame host backing

This checkpoint extends the [serializer attachment](resident-serializer-accounting.md)
at `e368154`. It is part of milestone 4 of [resident execution](resident-execution.md),
not completion of whole-workflow admission or scheduling.

## Ownership boundary

The existing managed allocator now covers the following additional backing:

| Owner | Attached storage |
| --- | --- |
| Strategy and CfL objects | Encoded strategy cells and the two signed-byte color-correlation planes; their construction/search scratch. |
| Prepared host pipeline | Initial quantization, strategy/pixel masks, sharpness, cached forward coefficients, transform records and tile indices. |
| AC search | Packed image/mask/matrix staging, candidate records, family costs and retained CPU cost arrays. |
| AQ evaluation | CPU fields and reconstruction scratch; Metal host metadata, reconfiguration candidates, readback planes, exact-coefficient staging and destination tables. |
| Completed frames | Owned CPU frame AC/DC and block metadata; independent Metal completed-frame host metadata, in addition to its already-accounted device allocation. |
| Supporting CPU/GPU operations | Quantizer selection, CfL fitting, EPF, maximum-error and Gaborish staging; native Butteraugli image/psychoimage/difference/kernel storage; Metal reference-kernel uploads and capture staging. |
| Joined CPU work | Initial-quantization, transform and color-conversion status/worker vectors, using the existing propagated resource context. |

Views still borrow without charging another allocation. Capacity is charged
before backing allocation, includes replacement-before-release overlap, and
stays with the backing across moves and destruction on another thread. An
explicit reservation cannot fall back to the unlimited default domain. The
existing encoder-entry opt-in remains: constructing caller inputs outside a
managed scope does not charge them to an encoder job.

Generic host containers inherit the allocating scope's resource class. The
workflow's host scope is preparation; AC-search preparation and completed-frame
construction select their own classes. In particular, both completed-frame
construction and its later CfL snapshot use the completed-frame class. Moving a
backing does not relabel or duplicate its charge. These are accounting categories,
not a promise that every host readback has a separate scratch category.

The frame/plane view contracts and C ABI are unchanged. Internal owning C++
containers use a different allocator type and consumers must rebuild against
the matching headers. `CopyContiguousPlane` now accepts any vector allocator;
Metal test-only snapshots retain their ordinary vectors through explicit copies.
No numerical algorithm, sorting rule, shader, AQ policy, AC candidate set, or
entropy policy changes in this checkpoint.

## Failure and lifetime checks

Managed failures retain their precise status through the affected status-returning
boundaries. Metal operation completion/invalidation still occurs on these paths.
Quantizer selection now catches allocation failures before it publishes either
the quantizer or raw field. CPU worker launch failure stops and joins already
launched workers before returning; the existing OS thread-creation fallback is
unchanged. This is not aggregate CPU admission.

Permanent tests cover:

- A fresh CPU completed frame's exact 13 backings and byte capacity, correct
  class, allocation-free move, closed producer reservation, and destruction on
  another thread.
- Prepared forward coefficients' exact retained capacities, including all three
  planes and metadata, with temporary tile lists released after preparation.
- Exhaustive failure positions: 13 for frame construction, 14 for forward
  preparation, two for quantizer selection and 23 for native Butteraugli. Failed
  operations preserve outputs and release pending/live charges; successful
  completion must leave the failure hook unconsumed.
- An undersized frame reservation preserves both the previous frame and the
  precise underplan status, without default-domain allocation.
- All 724 managed-host allocation positions of a small resident effort-1 encode,
  with a deliberately ample explicit test reservation. Every injected failure
  preserves caller bytes and is followed immediately by successful, byte-identical
  encoding on the same backend, before cache trimming. Every iteration drains
  its domain after trim and reservation closure. This fixture allowance is not
  a production planner or proof of coverage for every encoding policy.
- Metal completed frames retain exactly nine backings each: final device output,
  five host vectors, strategy cells and two CfL planes. Tests reconcile their
  capacities after source/evaluator destruction, changed images/dimensions,
  reconfiguration, reclamation, profiling, concurrent serialization and trimming.

The old native Butteraugli test intercepted only unaligned global `new`, so it
stopped reaching image backing after the allocator change. It now injects at the
managed allocator and checks domain drainage as well as output atomicity. The
Metal Butteraugli cache fixture now reserves the device arena plus the actual
maximum temporary upload kernel (33 floats), instead of assuming preparation
allocates only device storage.

## Qualification

The frozen parent is `build/resident-serializer-storage`; the fresh Release
candidate is `build/resident-frontend-storage`. Tests and benchmarks are enabled;
libjxl-reference fixtures and compile-time Metal profiling are disabled. The
parent drivers are copied from their verified previous manifest, never rebuilt
with changed owning-container headers.

All 56 corpus/policy codestream SHA-256 comparisons match. Separate pinned
`djxl` decoding of Kodak17, planter 4K and padded stress 4K produces identical
linear-RGB PFM hashes. Both builds pass all 22 pinned conformance fixtures. The
decoder revision remains `e8ff09762481785938d8e4e01333ed3917571161`.

The full Release parent passes 67/68 and candidate 68/69. Both reproduce the
same CPU `quantization_pipeline` golden mismatch: actual
`0.24919039011001587`, expected `0.24914586544036865`. No golden or tolerance was
changed. The initial candidate additionally failed the obsolete Butteraugli
allocation-hook test described above; it is not an inherited failure.

Seven CPU ASan/UBSan targets pass without suppressions. Four Metal/workflow
targets pass with only the existing metal-cpp-header null suppression. Leak
detection is disabled and both sanitizers halt on errors. These builds use
RelWithDebInfo and frame pointers. Ten Release repetitions each of frontend
storage, native Butteraugli, completed Metal frames and Metal accounting pass.
Ten TSan repetitions each of frontend storage, native Butteraugli and parallel
codestream encoding also pass with halt-on-error and no suppressions.

## Complete-call and physical-memory measurements

Apple M4 Pro, 48 GiB, macOS 15.6, Apple Clang 17; Release, SIMD/fused-tuned Metal,
fully resident, effort 7, distance 1.2, automatic CPU threads. As in the parent
qualification, the timer includes synchronous encoding and evaluator teardown,
but excludes backend creation, loading, hashing and output writes. Each of seven
alternating independent parent/candidate process pairs encodes a primary input
and its deterministically changed companion 15 times, discards the first three,
and retains six observations per image. All 840 output sizes/hashes match.

| Primary input | Parent / candidate median ms | Median paired change | Candidate faster pairs |
| --- | ---: | ---: | ---: |
| Kodak17 | 23.284 / 23.182 | -0.76% | 4/7 |
| Padded 1080p | 72.541 / 72.487 | +0.04% | 3/7 |
| Padded 4K | 251.273 / 246.145 | -0.12% | 5/7 |
| Planter 4K | 257.842 / 257.574 | +0.03% | 3/7 |

Percentages are medians of paired ratios, not ratios of the displayed medians.
Companion-image changes are +1.64%, +0.33%, -0.33%, and +0.46%, respectively.
Across all eight images, quantization-stage changes range from -0.37% to +0.74%
and serializer-tail changes from -3.39% to +2.05%. Directions are mixed across
pairs. This cohort does not establish a consistent complete-call speedup or
regression, nor does it prove zero accounting cost. Allocator alignment,
bookkeeping and ordinary run-to-run effects were not isolated experimentally.

Separate three-pair physical-memory probes report the following medians in MiB:

| Boundary | Parent | Candidate |
| --- | ---: | ---: |
| Process physical peak | 2968.517 | 2969.486 |
| One second idle, backend alive | 2015.298 | 2018.533 |
| One second after backend destruction | 252.470 | 251.720 |

There is no material physical-memory reduction demonstrated here. The separate
three-pair A/B/A, idle, trim, idle, resume-A probe matches all 24 outputs. Both
versions retain 1856.690 MiB managed idle capacity before trim and zero afterward;
live and unbacked charges are zero after encoding and trimming. Accounted peak
increases from 2755.063 to 2786.793 MiB because of expanded host coverage, not
because physical memory grew by that amount. One-second post-trim physical
footprints are 261.939 / 261.564 MiB.

Trim-call medians are 16.060 / 11.240 ms. Resume medians are 324.525 / 321.333 ms,
with a median paired change of -0.98% and two of three candidate runs faster.
These small diagnostic cohorts do not establish a trim/resume improvement.
In particular, pre-trim one-second physical footprint in the trim cohort is
2017.392 / 2206.798 MiB, unlike the nearly equal independent memory cohort.
OS reclamation and sampling boundaries remain important; retained managed
capacity is not a physical-footprint prediction.

All builds, sanitizer suites and repeated tests finished before timed runs.
The harness checks for overlapping GJXL processes, but does not establish
exclusive machine ownership or fixed thermal/power conditions.

## Artifacts and remaining work

`build/frontend-resource-qualification/run.py` reuses the retained comparison
harness with frozen parent drivers and freshly linked candidate drivers. Its
build manifest identifies both drivers, libraries and driver sources; parity
outputs, decoder logs, conformance manifests, timing/process observations,
physical-memory and trim probes are retained there. `summary.json` is recomputed
only after verifying all fourteen library hashes, four driver hashes, both
driver sources, completed manifests and timed output hashes. Full-suite
and repeated-test logs are in `build/resident-frontend-storage`; sanitizer logs
are in `build/frontend-storage-sanitize` and `build/frontend-storage-tsan`.

After building the candidate and preserving the frozen parent:

```sh
python3 build/frontend-resource-qualification/run.py build
python3 build/frontend-resource-qualification/run.py parity
python3 build/frontend-resource-qualification/run.py conformance
python3 build/frontend-resource-qualification/run.py performance
python3 build/frontend-resource-qualification/run.py memory
python3 build/frontend-resource-qualification/run.py trim
python3 build/metal-resource-qualification/summarize.py \
  build/frontend-resource-qualification resident-serializer-storage resident-frontend-storage
```

These are retained local qualification artifacts, not installed tools; the
regression tests are committed. Do not rebuild the frozen parent with candidate
headers. Only tests and documentation changed after the timed libraries were built.

Milestone 4 still requires publication-aware accounting for candidate codestreams,
score histories and retained batch results, shared checked planners, public
execution-domain handles, and whole-workflow admission with cache eviction and
progress guarantees. Public diagnostic result construction needs an explicit
boundary too; it must not be silently excluded as a way to pass a hard budget.
Test-only snapshots and small profile/control-object overhead are distinct from
production candidate/result payloads. File decoding is outside the in-memory
encode boundary.

Milestone 5's last-use/reuse audit and milestone 6's aggregate CPU/GPU scheduling
and pressure/throughput qualification remain required. More attached vectors do
not satisfy those milestones or establish a whole-encoder memory limit.
