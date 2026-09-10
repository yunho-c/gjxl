# Quantization-field and color-correlation storage bounds

This milestone-4 component follows the frontend representation checkpoint
`1ac4eae`. It adds checked host backing bounds for initial quantization, field
adjustment, quantizer selection and color correlation. **It does not complete
frontend planning or whole-workflow admission.** Search, preprocessing/perceptual
state, evaluation/retry/result storage and their overlapping host/device lifetimes
still need their remaining bounds and composition.
The subsequent [preprocessing/perceptual checkpoint](resident-perceptual-storage-planning.md)
supplies color/filter, block-reduction and native Butteraugli bounds; remaining
search/evaluation/metadata and whole-workflow composition are still pending.

The internal interfaces extend
[`frontend_storage_plan.h`](../src/codec/frontend_storage_plan.h). Successful
planning allocates no backing and takes O(1) work in image size. Invalid inputs
and checked overflow preserve the previous plan. These use the reviewed libc++
C++20 `HostStorageBound` contract; allocator headers, small runtime control
objects and thread stacks remain excluded. They bound managed backing, not RSS.
Neither a budget nor a CPU scope is installed by these functions.

The only existing runtime change shares the **unchanged** initial-quantization
parallel threshold and worker cap with the planner. Numerical operations,
allocation recipes, candidate policy and scheduling are unchanged. The Release
`adaptive_quantization.cpp.o` is byte-identical to the frozen parent's object.
The new planner functions are not yet called by production admission.

## Initial quantization: separate overlapping phases

For padded pixel dimensions `W,H`, let `P = W*H`, `B = P/64`, and `R = H/4`.
`ComputeInitialQuantField` retains a `P`-float pixel mask and a `P/16`-float
quarter-resolution erosion input through both phases below.

| Phase | Additional simultaneously live backing |
| --- | --- |
| Four-row reduction | At most `p` fresh `W`-float row arrays. If parallel, `R` status entries and at most `p` thread objects. |
| Erosion, modulation, blur and publication | Two fresh `B`-float arrays (quant field and strategy mask) plus a fresh `P`-float blurred mask. |

The worker bound is one below 65536 pixels; otherwise
`p = min(R, 12, requested_threads)`, with zero requested threads bounded by twelve.
Hardware availability and an enclosing explicit parallel scope may reduce actual
participation. Automatic mode launches all participants; explicit mode includes
the caller. Bounding thread objects by `p` covers both without changing either.
The planner's thread count must match the executing `EncodeScope`.

Each worker's row array is destroyed before it starts another four-row group.
All workers are joined and dispatcher vectors destroyed before field creation
and convolution. The thread-launch fallback joins workers before serial work;
its surviving dispatcher and single row array also fit the row-phase bound.
Fuzzy erosion, per-block modulation and `ConvolveSymmetric5` have no other heap
scratch. Therefore the complete temporary bound is the maximum of:

```
rows   = 4*(P + P/16 + p*W)
         + (p > 1 ? R*sizeof(Status) + p*sizeof(std::thread) : 0)
finish = 4*(2*P + P/16 + 2*B)
```

The caller supplies three destination planes. They are **not** included above;
if encoder-owned, their live backing must be added by workflow composition.
All three are updated only after validation and all fallible work succeeds.
No temporary backing survives the operation. As elsewhere, a scratch bound's
`retained_bytes` describes container capacities within the operation, not a
returned storage owner.

## Adjustment and quantizer selection

These two planners accept **block extents**, not pixels:

- `AdjustQuantField` creates one fresh `B`-float atomic temporary. This is needed
  even when input and output alias. Anchor traversal uses stack scratch.
- `CreateQuantizerFromField` reserves `B` floats for values and constructs `B`
  floats for absolute deviations. Both arrays coexist during in-place median
  selection. Their bound is `8B` bytes; the resulting `Quantizer` is inline.

Neither bound includes borrowed input or caller raw/adjusted destination planes.
No change was made to medians, modulation, thresholds or quantizer selection.

## Color correlation: output ownership versus tile scratch

Let `T = ceil(W/64)*ceil(H/64)` and
`C = min(W,64)*min(H,64)`. All map outputs are two fresh signed-byte planes of
`T` elements each. Output publication moves this independent owner atomically.

| Mode | Complete working backing, including the `2T`-byte output |
| --- | --- |
| Copy (`CreateColorCorrelationMap`) | `2T` bytes |
| Fast pixel-domain initial map | `4T` bytes: temporary factor planes coexist with the copied output |
| DCT initial, final or prepared-final map | `2T + 16C` bytes: output plus four fresh F32 coefficient vectors for one tile |

Both fast and ordinary final multiplier searches use the same backing shape.
The initial DCT path partitions a tile into 8x8 transforms. Final transforms
partition their tile without crossing its boundary. Each of the four vectors
reserves exactly the tile's total coefficient count before appending; no growth
beyond that reserve occurs. Tile processing is serial, and transform/multiplier
arithmetic uses stack arrays. Scratch scales with the largest tile, not every
tile in the image.

Prepared-final input must be an **unmodified producer result** from
`PrepareForwardDctCoefficients`. Its current shallow `valid()` checks do not
establish unique tile indices or physical tile membership for arbitrary manually
constructed structures. This geometry bound does not make that stronger claim.
The prepared coefficient owner itself remains separate from CfL scratch.

An old output remains live while replacement is built, including self-copy.
Composition must add its backing to `working`; tests verify a charged old/new
map peak of `4T` bytes for replacement by copying. Closing the producer reservation
does not uncharge the returned map; destruction of its owner releases the charge.

## Qualification

The frozen Release parent is `build/resident-frontend-representation-plans` at
`1ac4eae`; the fresh candidate is `build/resident-field-plans`. Both enable tests
and benchmarks and disable reference fixtures and compile-time Metal profiling.
Platform: Apple M4 Pro, 48 GiB, macOS 15.6 (24G84), Apple Clang 17.

The permanent [field-plan test](../tests/frontend_field_storage_plan_test.cpp)
checks allocation-free planning, invalid/overflow atomicity, formulas and worker
bounds. Thirteen fixtures include all seven supported transform families, mixed
strategies, color-tile edges, the parallel threshold, and wide/tall thin images.
At targets 1.2 and 4.0, 338 reservation-backed operations cover automatic/1/2/12
initial-quant threads; field adjustment including in-place aliasing; quantizer
selection; map self-copy; pixel/DCT initial maps; and ordinary/fast direct and
prepared final maps. Strided caller outputs include sentinel padding. Exact
snapshots compare all output values and padding; prepared and direct final maps
also match each other. Retained map backing reconciles to exactly two allocations.

Serial physical-failure sweeps cover 68 allocation positions across ten entry
point variants. All failures preserve previous outputs and drain live/pending
charges. Each variant also rejects zero admitted credit with the typed
`ResourcePlanExceeded` status before consuming the physical failure hook.
Separate tests verify charged output replacement overlap and a parallel initial
quantization job admitted only enough for common/dispatcher arrays: worker row
allocation fails through the inherited reservation, without escaping to the
default domain or publishing output. The latter check requires at least two
hardware threads and ran on the fourteen-thread qualification machine. Physical
fault injection is thread-local; the serial sweep is not an exhaustive parallel
physical-failure campaign.

Own-header oracle drivers are compiled against each revision's library. Extracted
parent source is verified against Git blob IDs. Their 260 length-framed scalar
records are byte-identical, including exact F32 bits, fields, quantizer state,
map factors and sentinel padding: 32,010,120 bytes per stream, SHA-256
`29534ef69d8f0e55d98ab95ee4a6ca2db97bb49360cc7d4bd246f5a1c1e0ff1b`.

Full Release suites pass 79/80 on the parent and 80/81 on the candidate. Both
reproduce only the CPU `quantization_pipeline` golden mismatch, actual
`0.24919039011001587` versus expected `0.24914586544036865`. No golden/tolerance
was changed. Field-plan, representation-plan and frontend-ownership tests each
pass three ASan/UBSan runs (nine total, RelWithDebInfo/frame pointers, leak
detection disabled, UBSan halt-on-error, no source/vendor suppressions).
The planner, changed runtime implementation and new test pass
`-Wall -Wextra -Wpedantic -Werror` syntax checks.

All 56 whole-workflow corpus/policy codestream pairs match SHA-256. Separately
decoded Kodak17, planter 4K and padded stress 4K also match linear-RGB PFM hashes.
Both builds pass all 22 pinned conformance fixtures using `djxl` v0.13.0 at
`e8ff09762481785938d8e4e01333ed3917571161`. The compiled Metal shader payload is
unchanged. Corpus/policy coverage includes efforts 1–10, density/compression,
diagnostics, exact/resident/throughput modes, target-size retries and maximum
error; these are functional checks, not a new timing experiment.

## Evidence and remaining work

`build/field-plan-qualification/` retains build/oracle manifests and source hashes,
canonical streams, Release/sanitizer logs, qualification commands and artifact
verification (`run.py`, `verify.py`, `seal.py`, `validation-commands.json` and
`validation.sha256`). The verifier rehashes libraries and actual retained JXL/PFM
outputs, checks record framing/counts, and reads the specific test results rather
than relying only on JSON completion markers. Frozen parent libraries are verified against their previous
qualification manifest and must not be rebuilt against current source.

Reproduction with these frozen builds:

```sh
python3 build/field-plan-qualification/run.py build
python3 build/field-plan-qualification/run.py oracle
python3 build/field-plan-qualification/run.py parity
python3 build/field-plan-qualification/run.py conformance
python3 build/field-plan-qualification/verify.py
```

No timing or physical-footprint cohort is claimed here: the separate runtime
study remained live during functional qualification. Runtime arithmetic and the
existing initial-quantization object are unchanged, and production does not yet
invoke these new bounds. Whole-workflow admission still needs the remaining
frontend/search/perceptual/evaluation/result bounds and composition. Milestones
5 and 6 still require their complete lifetime/reuse audit and aggregate scheduling
with the recorded correctness, resource and performance gates.
