# Metal AQ host storage bounds

This milestone-4 checkpoint follows `1781c4f`. It bounds the remaining production
host arrays owned by a prepared Metal AQ evaluator and the host part of an
independent completed frame. It also fixes an exception escape discovered while
testing exact-input staging. **Whole-workflow planning and admission remain
unfinished.** These component plans do not yet admit public encoder jobs.

## Contract and boundaries

[`ComputeAqHostStoragePlan`](../src/gpu/metal/metal_aq_host_storage_plan.h)
describes a fresh production evaluator at fixed source/padded coding geometry.
Its options cover the owner's entire operation history, including optional
diagnostics and exact-input staging, not merely its next call. `Reconfigure`
changes strategies and sharpness, not image dimensions. A different geometry
requires a different evaluator; both owners must be counted if they overlap.

The five bounds separate immediate prepared capacity, complete preparation,
capacity retained after selected operations, a complete serial operation, and
the maximum working peak. `retained_bytes` is a conservative inventory bound,
not a promise that all of those arrays are simultaneously live. Allocation and
replacement capacities use the reviewed libc++ C++20
[`HostStorageBound`](../src/core/host_storage_bound.h) contract. Plans perform no
image access, backend initialization or successful-path heap allocation and
leave output unchanged on validation/overflow failure.

The planner includes the temporary Gaussian upload kernel used while preparing
the evaluator's device Butteraugli reference. Device arenas, borrowed planes and
idle capacity belong to the [shared device plans](resident-storage-planning.md).
Profiling graphs **and their callback-input arrays**, policy score histories,
returned initial-CfL maps, caller image/field destinations, previous outputs,
and fresh owned/completed frames are separate owners. A parent workflow envelope
must include their overlap; they are not excluded from the overall resource
milestone. Small control objects, inline arrays, stacks, allocator headers and
opaque driver allocations retain the established accounting exclusions.

Private reconstruction/probe/stage-snapshot testing methods are not part of this
production contract. They populate additional buffers and can leave different
capacity histories: for example, a 1024-coefficient probe can exceed a tiny
image's entire three-channel coefficient count. Do not reuse a probe-populated
evaluator under this production bound.

## Source-backed inventory

Let `P` be padded pixel count, `N` source pixel count, `B = P / 64` blocks,
`T = ceil(width / 64) * ceil(height / 64)` color tiles, and
`G = ceil(width / 256) * ceil(height / 256)` AC groups. Anchor count is at most
`B`, including before final strategy selection. Counts and derived products are
validated against the existing AQ geometry and shader limits.

| Owner | Capacity recipe |
| --- | --- |
| Strategy grid and sharpness | Two fresh `B`-byte arrays. |
| CfL readback/state | Two fresh `T`-byte arrays; invariant replacement temporarily overlaps two new arrays with the old pair. |
| Row-major anchors | Initially push-grown, at most `B` records: retained at most `2B`, replacement peak at most `3B` records. Reconfiguration moves in a separately reserved `B`-record array. |
| Final transform layouts | Fresh reserve of `B` records unless deferred; subsequent reconfiguration moves in a fresh array of at most `B` records. |
| Ordinary evaluator metric readback | One `B`-float map plus `3B` floats for transform maximum error, regardless of which metric is selected. Absent for frame-only preparation. |
| Initial fields | Two `B`-float arrays when initial quantization is prepared. A `P`-float host mask is eager without resident AC inputs, otherwise lazy only if a diagnostic requests it. |
| Raw quant readback | One lazy fixed-size `B`-element I32 array, bounded across resident and nonresident calls. |
| Exact input | `3P` I32 AC values and `3B` I32 DC values; another `3P` floats only when reconstructing exact coefficients rather than accepting an exact linear image. |
| RGB/policy diagnostics | Three `N`-float RGB arrays and one `B`-float policy-field array when selected. |

The planner uses the actual `sizeof(AqAnchor)` and
`sizeof(QuantizedAcTransformLayout)`, not duplicated ABI-size constants.
Reconfiguration retains the old arrays while building new strategy records,
flattened/grouped anchors, sharpness and transform layouts. The seven grouped
anchor lists have a **combined** logical count of at most `B`; their individual
growth bounds can therefore be summed using that aggregate count.

Preparation scratch additionally includes `2B` I32 strategy records, a growing
`2B` I32 flattened anchor list, growing grouped XY pairs, and the 11,904-float
quantization-table upload. Resident CfL packing uses `6B` I32 records, two
`T+1` I32 offset/position arrays and one `T`-element `size_t` array. The position
and value-offset arrays are temporary inside the metadata helper; summing their
peaks with the enclosing scope is conservative. Device Butteraugli preparation
also uses a serial maximum of 33 host floats for Gaussian uploads.

Beyond replacement metadata, serial-operation scratch is the maximum of the
invariant CfL replacement, a `B`-float atomic quant-field adjustment, and a
`G`-element `size_t` exact-input cursor array. Fixed-size readbacks never grow
past their geometry-derived capacity during production reuse.

### Completed output is not exact-input staging

Normal owned-frame export maps the completed shared Metal AC/DC buffers and
passes borrowed spans to frame assembly. It does **not** require the evaluator's
`3P` host AC staging array. The [owned-frame plan](resident-frontend-storage-planning.md)
accounts for the returned CPU frame separately.

The new `ComputeCompletedFrameHostStoragePlan` instead covers the independent
device-backed frame's host snapshot. Its retained host capacity is exactly
`30B + 2T + sizeof(size_t) * G` bytes: strategy/sharpness, raw quant, quantized
and floating DC, CfL, and used-coefficient counts. Working capacity additionally
bounds a temporary U32 destination upload array of `anchor_count` elements.
The existing completed-frame device plan owns group-major AC and destination
backing; no duplicate host AC plane is charged here. Old outputs, the evaluator,
and a fresh replacement output must still be composed by their caller.

## Exact group-offset exception boundary

`UploadInput` previously constructed its exact-input group-offset vector outside
an exception handler. After the larger fixed-size exact buffers had been
populated, either physical allocation failure or an exhausted explicit
reservation could throw through the Status-returning API. The ordinary
resident-path failure sweep in the earlier frontend checkpoint did not exercise
this optional exact-prefix allocation.

The new narrow handler surrounds only that allocation. It preserves
`ManagedAllocationFailure::status()`, translates physical `bad_alloc` to OOM,
and translates excessive vector length to invalid input. The caller then uses
its existing invalidation path. Counts, zero initialization, traversal, upload,
arithmetic, outputs and successful-path policy are unchanged. An invalidated
evaluator is not reused as though upload had succeeded: recovery prepares a
fresh evaluator on the same backend without enlarging the job reservation.

## Qualification

The fresh Release tree is `build/resident-aq-host-plans`; the fresh targeted
ASan/UBSan tree is `build/resident-aq-host-plans-asan`. The frozen full Metal
comparison is `build/resident-perceptual-plans` at `d70ab11`. Its Metal/AQ source
and interfaces are identical to the immediate parent `1781c4f`; the intervening
checkpoints added storage plans/tests. Parent headers are exported from Git and
verified, and frozen libraries are checked against their previous manifest.
No frozen parent library is rebuilt against current source.

The permanent new test covers:

- 2,208 valid option/geometry formula cases, alongside invalid combinations,
  null/oversized geometry, allocation-free planning and unchanged output on
  failure; completed-snapshot formulas at minimum and block-count anchors.
- 42 real-Metal lifetime cases across tiny, padded, thin, mixed-transform and
  multiple-group geometries. Modes include full/frame-only, eager/deferred,
  resident/nonresident quantization, maximum error, both exact prefixes,
  diagnostic masks/RGB, and maskless reuse after lazy materialization.
- Repeated all-DCT8/mixed/all-DCT8 reconfiguration, retained-capacity checks,
  exact owned/completed frame comparison, and independent completed output after
  evaluator destruction and all-pool trimming.
- 43 preparation allocation failures, plus 42 later failures: 22 reconfiguration,
  two invariant CfL, one lazy mask, one field adjustment, four exact staging,
  three RGB readback, and nine completed snapshots. Every injected failure is
  followed by successful recovery and zero pending/backing charges after teardown.
- Typed zero-credit preparation/reconfiguration rejection, and focused physical
  OOM/zero-credit checks at the exact group-offset allocation. The latter preserve
  a previous frame and caller map/score without submitting GPU work. The same
  regression binary linked with frozen parent libraries reproduces both exception
  escapes; the candidate returns the expected statuses.

The full Release parent passes 81/82 and candidate 85/86. Both reproduce only the
CPU `quantization_pipeline` golden mismatch: actual `0.24919039011001587`, expected
`0.24914586544036865`. No golden or tolerance changed. The AQ-host, completed-frame
and shared-device-plan sanitizer tests pass three repetitions each (9 runs).
Leak detection is disabled. The unsuppressed run reproduces metal-cpp's null
member-call issue before encoding; only `null:*/third_party/metal-cpp/*` is
suppressed in the passing run, with no GJXL-source suppression.

All 56 corpus/policy codestream hashes match. Separate pinned `djxl` decoding
of Kodak17, planter 4K and padded stress 4K produces identical PFM hashes.
Both builds pass all 22 pinned conformance fixtures. The decoder remains
`e8ff09762481785938d8e4e01333ed3917571161`.

Of the comparison's common codec/GPU/Metal/serializer objects, 57 are byte-identical.
The changed AQ object contains the exception fix. The other changed object is
the earlier AC host-plan addition; it matches the sealed immediate-parent object.
The complete metallib is byte-identical. These are correctness/source-continuity
checks, not a new latency or physical-memory measurement.

Exact commands, hashes, failures, repeated tests, JXL/PFM outputs and decoder logs
are retained in `build/aq-host-plan-qualification/` and `build/aq-host-plan-*.log`.
`run.py` provides `build`, `parity`, `conformance`, `tests`, `seal` and `verify` commands;
`validation.json` seals the evidence and participating sources/binaries.

Backend validated-batch/stage/context bounds, workflow-derived profiling and
attempt/result counts, combined admission, the explicit reuse/lifetime inventory,
and aggregate CPU scheduling still remain. No performance or whole-workflow
completion claim follows from this component checkpoint.
