# Preprocessing and native perceptual storage bounds

This milestone-4 component follows `07831b4`, the
[quantization-field/CfL checkpoint](resident-field-storage-planning.md). It adds
checked bounds for color conversion, nested loop filters, block reductions, and
native CPU Butteraugli preparation/comparison/one-shot scratch. **Whole-workflow
planning and admission remain incomplete.** AC search, evaluation/policy state,
remaining Metal host metadata and attempt/result lifetimes still need their
remaining bounds and composition with the existing device/serializer plans.

The interfaces are in [`frontend_storage_plan.h`](../src/codec/frontend_storage_plan.h)
and [`perceptual_storage_plan.h`](../src/codec/perceptual_storage_plan.h). They are
allocation-free on success, checked, O(1) in image size and atomic on invalid
input/overflow. They follow the reviewed libc++ C++20 `HostStorageBound` contract,
not process RSS. Caller-provided source/destination planes and previous output
owners are separate; encoder-owned versions must be included by composition.
Thread stacks, allocator headers and small runtime/pimpl control objects remain
the documented exclusions. Planning does not install a domain or CPU scope.

## Color conversion and filters

Public `LinearRgbToOpsin` and `OpsinToLinearRgb` build a fresh three-plane image
before publishing to the caller. Their complete scratch is `12P` bytes plus
possible dispatcher arrays. The internal `LinearRgbToPaddedOpsin` instead writes
directly into private caller scratch; it has no temporary image. Its existing
contract permits partial scratch modification on failure and requires nonoverlap
with the source. This is not a new atomicity promise for that internal primitive.

The color dispatcher processes `source.height` rows, each `destination.width`
pixels wide. The parallel threshold uses their product, not the unpadded source
area or padded destination area. Below 65536 dispatch pixels it is serial;
otherwise the bound is `min(source.height, 12, requested_threads)`, with zero
meaning automatic and bounded by twelve. Actual hardware/nested scopes can reduce
participation. A parallel invocation owns one status per dispatched row and at
most one thread object per participant; callback arithmetic has no heap scratch.
The unchanged threshold/cap are now shared with the planner. Its thread count
must match the executing `EncodeScope`; it is not aggregate CPU enforcement.

`ApplyLoopFilters` composes the existing nested lifetimes at the exact pixel
extent. Each temporary image owns three fresh F32 planes:

| Gaborish | EPF iterations | Maximum simultaneous temporary images |
| --- | ---: | ---: |
| Off | 0 | 0 |
| Off | 1 or 2 | 1 |
| Off | 3 | 2 |
| On | 0 | 1 |
| On | 1 or 2 | 2 |
| On | 3 | 3 |

When both filters run, the outer destination scratch survives Gaborish's own
temporary, then EPF's one or two temporaries. The two inner stages do not overlap.
Standalone forward/inverse Gaborish each need one temporary image. No numerical
operations, filter options, allocation recipes or pass ordering change.

`ComputeEpfInverseSigma`, `ReduceButteraugliDistanceMap` and `ReduceMaximumError`
each own one fresh atomic F32 array at block resolution (`4B` bytes). Their output
planes and inline scalar/reduction state are separate from that temporary.

## Preserve the underplan error across wrappers

Six boundaries previously caught `ManagedAllocationFailure` through its
`std::bad_alloc` base and returned a generic out-of-memory status:

- Public forward and inverse color conversion, plus the internal padded path.
- Forward and inverse Gaborish.
- The outer combined loop-filter scratch allocation.

They now return the original typed status first, preserving
`resource_plan_exceeded()`. Genuine physical-allocation failure still returns
ordinary out-of-memory. This is required for the planned terminal-underestimate
contract; it changes neither successful output nor ordinary allocation failure.
Existing EPF and native Butteraugli allocation boundaries already preserve this
distinction and were not modified.

Own-header parent/candidate drivers reproduce all six erased reasons on `07831b4`
and all six preserved reasons on this candidate. Both reject before consuming
the physical failure hook and preserve output. The permanent test asserts the
typed result too; the padded case needs two hardware threads because its serial
path allocates no backing. It ran on the fourteen-thread qualification machine.

## Native Butteraugli ownership inventory

This is **CPU** storage, separate from the existing Metal plane-layout plans.
Let `N` be requested pixels, `M` pixels after expanding each dimension to at
least eight, and `S` half-resolution pixels (ceil-divided dimensions) only when
there is no expansion and both requested dimensions are at least fifteen.

One prepared scale owns the following F32 backing, all fresh exact-count arrays:

| Owner | Planes at that scale |
| --- | ---: |
| XYB image | 3 |
| Opsin blurred/result images | 6 |
| Frequency blurred/transposed scratch | 2 |
| Reference and distorted psychoimages | 20 |
| Seven difference planes plus AC/DC images | 13 |
| Difference mask-blur transpose | 1 |
| Observable difference stages and atomic swap staging | 18 |
| Total | 63 |

Three small kernel vectors retain 5, 33 and 13 floats. Frequency blur starts with
its largest (33-tap) kernel and subsequently shrinks its size without dropping
capacity; Opsin's five-tap path needs no transpose. Compile-time checks tie this
recipe to the pinned sigmas and call order. Contiguous multi-plane owners are
checked at their full element count, not merely as independently valid planes.

Prepared storage includes both scales' independent scratch, a final requested-size
map, an expanded three-plane main input when needed, and a three-plane subscale
input when needed. With `E` equal to one for expansion, the retained byte formula is:

```
prepared = 4 * (63*M + 51 + N + E*3*M + (S ? 66*S + 51 : 0))
```

Fresh preparation constructs this owner without additional peak backing beyond
the formula. Re-preparation must separately account an old owner still live
until the new one succeeds. A comparison replaces the distorted psychoimage by
building a fresh ten-plane candidate before releasing the old one. Main/subscale
replacement is serial, so complete comparison backing is `prepared + 40*M`.
The prepared reference, masks, stage swap and scratch remain independently owned
after the producer reservation closes and release their charges on destruction.

These are complete operation/owner bounds, **not incremental cross-reservation
credits**. A new producer comparing a reference charged to an earlier reservation
must account that live owner and its new reservation consistently. Subtracting
retained bytes from a peak does not automatically transfer credit: newly replaced
psychoimages can remain charged to the new producer. The lifetime test uses a
complete new envelope while the old owner remains charged; normal workflow retries
should retain the original whole-job reservation.

### One-shot and reusable multiscale scratch

The public one-shot API creates fresh native scratch. Internally, the same scratch
type can be reused at the **same requested geometry**. It shares XYB, Opsin,
frequency and difference scratch between scales, replacing rather than retaining
capacity on an extent change. Its stage-swap owner can retain capacity after
repeated calls or failures even though it is empty after an initial successful
fresh call. Both observable scale outputs remain owned.

A conservative sum of each owner's maximum retained capacity is:

```
retained-capacity bound = 4 * (66*M + 15*S + N + E*6*M + 51)
complete working bound = retained-capacity bound + 44*M
```

The extra eleven planes cover the largest actual overlap: a ten-plane frequency
candidate coexisting with a one-plane scratch replacement. Other resizes happen
serially and need no larger extra backing; the nine-plane difference candidate
already belongs to the counted stage-swap owner. Summing every vector's independent
replacement peak would unnecessarily assume simultaneous resizes. This remains a
conservative owner-capacity bound, not an exact prediction of live storage at
return. Previously larger/different geometry requires a separately composed
history/old-owner bound; this API does not cover arbitrary external scratch history.

## Qualification

Frozen parent: `build/resident-field-plans` at `07831b4`. Fresh Release candidate:
`build/resident-perceptual-plans`. Both enable tests/benchmarks and disable libjxl
reference fixtures and compile-time Metal profiling. Platform: Apple M4 Pro,
48 GiB, macOS 15.6 (24G84), Apple Clang 17.

The [permanent test](../tests/perceptual_storage_plan_test.cpp) exercises 351
reservation-backed calls across thirteen tiny, thin, threshold, padded and
multiscale fixtures. Color conversions run at automatic/1/2/12 threads. Public
conversion and filters also run in place; outputs have sentinel row padding.
Checks cover all eight combined filter configurations, standalone Gaborish,
three block reductions, one-shot and prepared native maps/scores, exact eager
prepared backing, real comparison replacement peaks, changed-image reuse and
producer teardown. Internal multiscale scratch is reused across three calls.

Physical fault sweeps cover 152 positions across allocating entry-point variants,
plus 46 re-preparation and two comparison failures against an existing reference:
200 total. Failed calls preserve caller outputs and keep the old reference usable;
all charges drain when their owning prepared/scratch state is destroyed. Allocating
variants also reject zero admitted credit without escaping to the default domain.
The physical hook is thread-local; these serial sweeps are not exhaustive injection
into thread-launch/runtime allocations. The six wrapper underplan checks separately
exercise the parallel padded dispatch-allocation boundary.

The first test run exposed a harness mistake: initial `Reserve(0)` is invalid,
whereas a live reservation reduced to zero is the supported way to test a
zero-allocation path. The harness was corrected; reservation semantics and
production no-op paths were not changed. The initial log is retained.

Own-header drivers compare 234 length-framed records of exact F32/scalar bits,
maps, scores, image outputs and sentinel padding. The 59,939,628-byte streams are
identical, SHA-256
`38f399b8d4a11825ca83f428b4dcc53d5fbfe68c472069a47d3a14f240eaaa74`.
They are retained losslessly gzip-compressed because disk space is low; canonical
hashes and framing refer to decompressed bytes. Parent headers are checked against
their Git blob IDs, and frozen libraries against the preceding qualification.

Full Release suites pass 80/81 on the parent and 81/82 on the candidate. Both
reproduce only the CPU `quantization_pipeline` golden mismatch, actual
`0.24919039011001587` versus expected `0.24914586544036865`. No golden or numerical
tolerance changes. The planner, affected runtime files and new test pass strict
`-Wall -Wextra -Wpedantic -Werror` syntax checks.

Perceptual planning, native Butteraugli distance and field planning each pass
three ASan/UBSan repetitions (nine runs total, RelWithDebInfo/frame pointers,
leak detection disabled, UBSan halt-on-error, no source/vendor suppressions).
All 56 whole-workflow corpus/policy codestream pairs match SHA-256; separately
decoded Kodak17, planter 4K and padded stress 4K match linear-RGB PFM hashes.
Both builds pass all 22 pinned conformance fixtures with `djxl` v0.13.0 at
`e8ff09762481785938d8e4e01333ed3917571161`. The three native Butteraugli object
files and compiled Metal shader payload are byte-identical to the parent.

## Evidence and remaining work

`build/perceptual-plan-qualification/` retains own-header build commands, source
and library hashes, compressed exact streams, error-reason comparisons, test logs,
corpus outputs and artifact verification. `run.py`, `verify.py`, `seal.py`,
`validation-commands.json` and `validation.sha256` identify the completed evidence.
Do not rebuild frozen parents against newer source.

Reproduce using these identified frozen builds:

```sh
python3 build/perceptual-plan-qualification/run.py build
python3 build/perceptual-plan-qualification/run.py oracle
python3 build/perceptual-plan-qualification/run.py underplan
python3 build/perceptual-plan-qualification/run.py parity
python3 build/perceptual-plan-qualification/run.py conformance
python3 build/perceptual-plan-qualification/verify.py
```

The verifier checks actual compressed/decompressed bytes, framing/counts, input
and binary hashes, specific test failures and fixed error reasons. It does not
treat a manifest's completion field alone as proof.

The separate runtime study remained active during functional qualification, so
this component claims no new timing or physical-footprint cohort. Allocation
recipes and numerical work are unchanged; the new bound functions are not yet
installed in production admission. Error-reason preservation is the intended
runtime change, alongside sharing the unchanged color-dispatch constants.

Remaining AC/search, evaluation/policy, Metal-host and attempt/result bounds must
be composed into the complete workflow plan before exposing the domain/admission
API. The full milestone-5 lifetime/reuse audit and milestone-6 aggregate CPU/GPU
scheduling gates remain required; these component tests do not complete them.
