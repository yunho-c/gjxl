# Frontend image, frame and transform storage bounds

This milestone-4 component follows `e040e7f`, the
[whole-serializer storage plan](resident-serializer-storage-planning.md). It adds
checked host bounds for image backing, owned coefficient frames, prepared
forward transforms and coefficient reconstruction. **It is not the complete
frontend or whole-workflow plan.** AC search, quantization fields, preprocessing,
perceptual evaluation, device/host coexistence, retries, diagnostics and retained
batch outputs still need their remaining bounds and composition before admission.
The subsequent [field/CfL checkpoint](resident-field-storage-planning.md) supplies
the initial quantization, field adjustment, quantizer selection and color-map
bounds; it does not complete the other remaining frontend/workflow work.

The interfaces are in
[`frontend_storage_plan.h`](../src/codec/frontend_storage_plan.h). Each successful
plan allocates no backing, takes O(1) work in image size and preserves its output
on invalid input or overflow. Bounds follow the reviewed libc++ C++20
`HostStorageBound` capacity/replacement contract, not process RSS. Thread stacks,
small runtime control objects and allocator headers remain excluded.

The only existing runtime change shares the unchanged forward-transform
parallel threshold and worker cap with the planner. No transforms, numerical
operations, encoding decisions, allocation recipes or scheduling policies change.

## Distinct ownership boundaries

Let `B` be padded 8x8 block count, `P = 64B` padded pixels, `T` the count of
64x64-pixel color tiles, and `G` the count of 256x256-pixel AC groups.

| Component | Retained output | Additional working storage |
| --- | --- | --- |
| `Image3FBuffer` creation/replacement | Three fresh contiguous F32 planes at the exact requested extent | No other backing; any old image is a separate overlapping owner. |
| Owned coefficient frame | Thirteen fresh backings: control fields, DC, CfL and group-major AC | CPU coefficient coding and assembly use only stack scratch beyond this output. Borrowed input/prepared coefficients and old output are separate. |
| Prepared forward coefficients | Three packed F32 coefficient arrays, growing transform records and flattened tile indices, exact tile offsets | Temporary per-tile lists and possible worker status/thread arrays remain alive during preparation. |
| Coefficient reconstruction | Writes the caller's provided image; no returned backing owner | Full atomic temporary image, group offsets and one transform's coefficient/DC/pixel arrays. |

These are different representations and lifetimes, not interchangeable estimates.
An encoder-owned input or old output is not exempt from the **workflow** limit
merely because one operation borrows it. Add its live backing to the new working
bound when they overlap. In particular, `Image3FBuffer::resize` constructs all
three replacement planes before releasing the previous image.

## Owned frames

Both `ComputeQuantizedCoefficients` (direct or prepared) and
`AssembleVarDctEncoderFrame` create the same fresh ownership shape:

- Two byte arrays of `B` elements: encoded strategy cells and EPF sharpness.
- Four I32 arrays of `B` elements: raw quantization and three authoritative DC
  planes; three F32 arrays hold decoder-equivalent reconstructed DC.
- Two signed-byte arrays of `T` elements hold color correlation.
- One `size_t[G]` used-coefficient table and one I32 array of
  `3 * G * 65536` elements hold all group/channel AC storage.

Thus retained backing and construction peak are both
`30B + 2T + sizeof(size_t)*G + sizeof(int32_t)*3*65536*G` bytes. Every allocation
is fresh exact-count/copy construction. The count is **thirteen**, not one charge
per borrowed plane view. A tiny or boundary group still owns its full unused
65536-element channel capacity; counting only used coefficients underestimates
the existing representation.

The quantizer and profile are inline. Forward DCT, quantization, DC conversion,
and the frame's validation use stack scratch within these entry points. A
prepared coefficient input is separately owned and remains live during coding;
this frame plan does not include preparing it. Tests exercise both adjusted and
fixed-raw AC decisions without changing either policy.

## Prepared forward coefficients

The geometry-only maximum is `B` transform anchors. Three coefficient arrays
retain exactly `P` floats each, without AC-group padding. The transform-record
vector grows by pushes, the flattened tile-index vector grows by forward-range
insertion, and tile offsets reserve exactly `T + 1` elements. Their respective
capacity/replacement policies are growing, growing and fresh exact.

During preparation, an outer exact `T`-element vector holds per-tile index vectors.
Every anchor belongs to exactly one tile. Summing each list's reviewed growth
bound is at most the growth bound on the sum of anchor counts, including
replacement peaks. These lists survive transform execution and flattened-index
construction; they cannot be charged only until the DCT begins.

The existing parallel threshold is 65536 **per-channel** coefficients, not three
times that value. Below it there is one participant; otherwise the upper bound
is `min(B, 8, requested_threads)`, with zero requested threads meaning automatic
and bounded by eight. Hardware and nested explicit scopes can reduce actual
participation. Possible status entries are bounded by `B`; thread objects by
the participant bound. Automatic mode launches all participants, while explicit
mode also uses the calling thread. Forward-transform scratch itself is on stacks.

`output` describes the retained vectors and their construction peaks.
`working` is the **complete** conservative operation bound, including `output`,
tile lists and dispatch arrays. Do not add output to working again. The planner
does not install an `EncodeScope` or establish aggregate CPU scheduling.

## Reconstruction

`ReconstructQuantizedCoefficients` preserves caller output until the entire
operation succeeds. Its scratch therefore includes a full three-plane padded
temporary image, even though its API accepts a preallocated destination view.
The caller's image, if encoder-owned, must be counted separately by composition.

The remaining scratch is `size_t[G]` offsets, three transform coefficient arrays,
one DC array and one inverse-DCT pixel array. DC and pixel arrays overlap inside
the channel loop; different anchors do not retain those arrays concurrently.
The plan scans the fixed strategy table and current CPU support predicate for
the largest strategy that fits the block geometry, including both rectangular
orientations. It does not multiply transform scratch by every image block.

## Qualification

The frozen parent is `build/resident-serializer-plan` at `e040e7f`; the fresh
Release candidate is `build/resident-frontend-representation-plans`. Both enable
tests/benchmarks and disable libjxl-reference fixtures and compile-time Metal
profiling. The environment is Apple M4 Pro, 48 GiB, macOS 15.6, Apple Clang 17.

The permanent [frontend-plan test](../tests/frontend_storage_plan_test.cpp) covers
allocation-free planning, overflow and invalid-output atomicity, nondecreasing
thread bounds, and twelve tiny/padded/rectangular/mixed-strategy fixtures. They
include multi-group images, the forward parallel threshold and a second DC group.
Inside calculated reservations it runs 12 image creations, 48 preparations
(automatic/1/2/8 threads), 72 frames (direct/prepared/assembly, two decision modes),
and 24 reconstructions. All 156 outputs match their reference snapshots exactly.
Fresh frames reconcile to exactly thirteen completed-frame backings and the
calculated byte count. Retained image/frame/preparation backing survives closing
the producer reservation and disappears on owner destruction.

Serial physical-allocation sweeps cover 3 image, 14 preparation, 13 direct-frame,
58 reconstruction and 13 assembly failure positions (101 total). Failed
operations preserve all previous output values and drain pending/live charges.
Each operation also rejects zero admitted credit with the typed
`ResourcePlanExceeded` status and leaves the physical failure hook unconsumed;
the default domain's peak remains zero. The hook is thread-local, so these are
not exhaustive injections into parallel workers.

The first fault-test run exposed an adapter bug: `Image3FBuffer` throws
`std::bad_alloc` for physical failure, whereas the other entry points return a
`Status`. The test now catches that throwing API's allocation exception while
preserving the distinct managed underplan status. The initial aborted log is
retained. No production exception behavior was changed. A subsequent comment-only
formatting rebuild preserves the exact Release test-executable hash.

Separate oracle drivers are compiled with each revision's own headers and
libraries; extracted parent headers are checked against Git blob IDs. Their 60
length-framed records contain prepared F32 coefficients, transform/tile order,
frame metadata/DC/full AC capacity, and reconstructed F32 pixels. Both byte streams
are 48,100,194 bytes with SHA-256
`360ab3ebfe0551d8d844039bb7d03986d166207d9d9e02a61fe8946113551d62`.
This compares exact floating-point bits, not a numerical tolerance or only sizes.

Full Release suites pass 78/79 on the parent and 79/80 on the candidate. Both
reproduce only the CPU `quantization_pipeline` golden mismatch, actual
`0.24919039011001587` versus expected `0.24914586544036865`. No golden or tolerance
was changed. Frontend planning, frontend ownership and owned-frame tests each
pass three ASan/UBSan repetitions (nine runs, RelWithDebInfo/frame pointers,
leak detection disabled, UBSan halt-on-error, no source/vendor suppressions).
The new planner, affected reconstruction implementation and new test pass
`-Wall -Wextra -Wpedantic -Werror` syntax checks.

All 56 whole-workflow corpus/policy codestream pairs match SHA-256, and separately
decoded Kodak17, planter 4K and padded stress 4K match linear-RGB PFM hashes.
Both builds pass all 22 pinned conformance fixtures with `djxl` v0.13.0 at
`e8ff09762481785938d8e4e01333ed3917571161`. The compiled Metal shader payload is
unchanged.

## Evidence and remaining work

`build/frontend-plan-qualification/` retains the own-header oracle build commands,
source/library/executable hashes, raw canonical streams, corpus JXL/PFM outputs,
decoder/conformance records, Release/sanitizer logs, strict compile and failure
evidence. `verify.py` re-reads the actual output files and binary hashes as well
as manifest counts; `validation.sha256` seals the final inputs/logs. The initial
test-adapter failure remains separate from the passing final logs. Do not rebuild
the frozen parent against later sources.

After building the identified candidate and preserving the frozen parent:

```sh
python3 build/frontend-plan-qualification/run.py build
python3 build/frontend-plan-qualification/run.py oracle
python3 build/frontend-plan-qualification/run.py parity
python3 build/frontend-plan-qualification/run.py conformance
python3 build/frontend-plan-qualification/verify.py
```

The local qualification harness is not an installed tool; permanent regression
tests are committed. No quiet timing or physical-footprint cohort is claimed.
The separate runtime-characterization controller (PID 3298) and an encoder child
were verified live and left untouched. These new planning functions are not yet
called by workflow admission, and no latency or physical-memory improvement is
claimed. Remaining frontend plans, whole-workflow composition and public domain
integration must still cover fields, AC search, CPU/Metal perceptual state,
retries, diagnostics and accumulated batch results. Milestones 5 and 6's audited
reuse/lifetime reductions and aggregate scheduling remain separate requirements.
