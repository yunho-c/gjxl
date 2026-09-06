# Coefficient-order and block-context representation storage bounds

This milestone-4 component follows `53178f9` (entropy search/emission bounds).
It adds count-only plans for coefficient-order selection/tokenization and block
context-map selection, beside the implementation's private types and policy
constants. It does not change selection, sorting, sampling, tokenization,
parallelism, allocation recipes, or encoded output. The planners are not yet
called by production workflow admission. **Milestone 4 remains incomplete.**

The interfaces are in
[representation_storage_plan.h](../src/codestream/representation_storage_plan.h).
Successful planning performs no managed backing allocation and does not traverse
image groups. Inputs are valid-frame block geometry and an upper bound of 1–8
coefficient-order participants. Automatic runtime threading requires a bound
that covers its possible participants; passing `1` does not make runtime serial.
Maximum compression must pass its normalized full-order behavior, even if the
original workflow option requested effort-7 sampling.

## Coefficient orders

Seven supported transform orientations share five coefficient-order families:

| Family | Representative / transpose | Coefficients | LLF prefix |
| --- | --- | ---: | ---: |
| 0 | DCT8 | 64 | 1 |
| 2 | DCT16x16 | 256 | 4 |
| 3 | DCT32x32 | 1024 | 16 |
| 4 | DCT16x8 / DCT8x16 | 128 | 2 |
| 6 | DCT32x16 / DCT16x32 | 512 | 8 |

The plan includes a family when either orientation can fit the block geometry.
It follows the existing early return when both block dimensions are below five.
Let `B` be block count and `G = ceil(width/32) * ceil(height/32)` AC groups.
Outside the early return, the participant bound is one when `64*B < 65536`,
otherwise `min(workers, G)`. Hardware/nested-scope limits may reduce participation.

For every eligible family of `N` coefficients and `L` LLF coefficients:

- Three fresh `uint32_t[N]` output scans: at most 5952 elements across all families.
- At most `3 * (1 + N - L)` permutation tokens: an end token and no more than
  one token per non-LLF position. The all-family bound is 5874 `EntropyToken`s.
  The output grows by pushes, so its capacity/replacement bounds use the
  [reviewed vector growth contract](resident-token-storage-planning.md).
- Global `uint64_t` zero counts, plus one set per possible participant when
  parallel. Worker counts remain allocated while the global reduction is built.
  Within a family, the first assign allocates once; subsequent groups only
  modify counts. A transposed strategy does not require another family array.

The group-view table, dispatch-status and thread-object vectors are included.
Thread stacks and thread-runtime control allocations remain excluded, as in the
declared managed boundary. Sampling additionally retains an outer vector per
group and at most `B` selection bytes. The sampled plan also covers the existing
full-count fallback for non-DCT8/mixed frames; it does not assume sampling always
executes or reduces the output/model sizes.

Selection's natural scan, inverse rank and validation are bounded by the larger
serial permutation-tokenization envelope: natural scan, inverse LUT, permutation,
Lehmer code, and an `N+1` Fenwick tree. Two `uint8_t[N]` seen arrays cover the
overlap between custom-order validation and natural-order generation. Local
selection and tokenization scratch are not multiplied by the group count.

Important retained-capacity detail: when a derived scan equals natural order,
the runtime clears its output vector without releasing its backing. Thus an
empty `used_order_mask` does not imply zero output storage. `orders` covers these
cleared vectors too; the whole-serializer plan must retain that charge through
the owner's destruction, not just through order-token generation.

`working` is a complete conservative envelope including both `orders` and
`tokens`. It sums these named scratch bounds even where their maxima occur in
different phases. Do not add those outputs to `working` a second time.

## Context maps

The plans use the same adaptive/split thresholds as the selector:

| Block count | Maximum exhaustive candidates | Maximum quant thresholds | Maximum map entries |
| --- | ---: | ---: | ---: |
| Below 1024 | 1 | 0 | 39 |
| 1024–8191 | 5 | 0 | 39 |
| At least 8192 | 6 | 1 | 78 |

Ordinary selection returns only one map. Exhaustive candidates can be removed by
deduplication; the plan does not rely on that removal. It includes all nested map
backings and the growing outer candidate vector. A map under the adaptive
threshold has four block contexts (1980 AC contexts); above it the plan uses
the validated 16-context cap (7920 AC contexts). The format's 15-context default
is not the upper bound for the general map representation.

Let `cells = 13 * (threshold_count + 1)`. Adaptive construction retains
`size_t[cells]` occurrence counts and byte arrays for remap, clusters and labels.
Copying the remap into the map initially allocates `cells` bytes, then resizing
to `3*cells` allocates that exact larger count under the reviewed libc++ growth
recipe. The old map backing can coexist with its replacement and is counted
separately. The median-threshold vector remains alive while its value is copied
into the output map. Sorting uses inline ranks and no heap-allocating stable sort.

`map` bounds one fresh nested-map copy, excluding the inline map object.
`output` bounds the selected map, or all exhaustive maps plus the outer vector.
`working` includes `output` and construction/replacement scratch. Encoder-level
candidate copies and the ordinary path's outer wrapper vector are deliberately
not included; their owners belong in the whole-serializer composition.

## Verification

The fresh Release candidate is `build/resident-representation-plans`; the frozen
parent is `build/resident-entropy-plans` at `53178f9`. The parent was not rebuilt
against current headers. Artifacts and the functional runner are retained under
`build/representation-plan-qualification/`.

Permanent [representation tests](../tests/representation_storage_plan_test.cpp)
check independent count formulas over 137 by 67 block geometries, one/two/eight
worker bounds and both order behaviors; invalid/overflow cases preserve output.
A large count-only case does not allocate the image or token stream. An armed
physical-allocation failure hook remains armed throughout successful planning.

Real operations cover 64 fixtures spanning tiny frames, thin transposes, group
edges, all seven transform orientations and mixed strategies, both quantization
split regimes, and zero/dense/sparse coefficients. Each order behavior runs with
one, two and eight participants under its computed reservation. Additional
DCT8 fixtures exercise nondefault sampled orders and empty-but-capacity-retaining
natural orders. All returned vector capacities and allocation counts reconcile
with the serializer class and remain charged after the producer reservation
closes. These are managed capacities, not process physical-footprint samples.

Serial physical-allocation sweeps cover 19 full-DCT8 order, 35 sampled-DCT8 order,
91 mixed-order, 145 order-tokenization, 22 exhaustive-map and 8 ordinary-map
positions. These thread-local hook sweeps do not claim to inject every parallel
interleaving. Separate underplanned tests cover all four operations before their
first physical allocation, plus an order worker's failure after caller-owned
dispatch arrays fit. Typed resource-plan errors preserve previous outputs,
release partial backing, join workers and do not escape into the default domain.

The independent frozen-parent oracle emits logical values and capacities in
explicit little-endian fields, not object padding or a rolling hash. Its 128
order/token and 128 map results compare byte-for-byte across 547072 bytes:
SHA-256 `94102079867dc091b391e73979d65e088697871be5dacbde740cef1a126fcd8b`.
Each oracle is linked against its corresponding libraries; parent headers are
extracted from the pinned commit and every extracted file is checked against
its Git object ID before compilation.

All 56 corpus/policy cases preserve codestream SHA-256 against the frozen parent.
Separate pinned `djxl` decoding of Kodak17, planter 4K and padded stress 4K
preserves linear-RGB PFM hashes; both builds pass all 22 pinned conformance
fixtures. The decoder pin remains
`e8ff09762481785938d8e4e01333ed3917571161`.

The complete Release candidate passes 77/78; its only failure is the reproduced
CPU `quantization_pipeline` golden mismatch: actual `0.24919039011001587`,
expected `0.24914586544036865`. The frozen parent passes 75/77: it reproduces
that same mismatch and additionally reports `Butteraugli capacity reuse failed
at 2`. Both parent and candidate then pass ten isolated repetitions of the
Butteraugli test. The original failure remains recorded, not replaced by the
successful repeats. Metal source is unchanged and the metallib hashes match.

That cache assertion combines preparation status and exact allocation count
into one message, so the observed failure cannot establish which condition
failed. Source permits a fresh allocation if volatile capacity was reclaimed;
the test expects none in that case's same-size reuse. This is a concrete
test-robustness concern, not proof that OS reclamation caused this instance.
Its diagnostics and treatment of legitimate reclamation still need an explicit
audit; no allocation assertion or numerical tolerance was weakened here.

The new representation, existing token-plan and block-map tests pass three
repetitions each under ASan/UBSan (RelWithDebInfo, leak detection disabled,
halt-on-UB, no source or vendor suppressions). Both affected implementation
files and both storage-plan tests pass `-Wall -Wextra -Wpedantic -Werror` syntax
checks. The shared synthetic-frame fixture is extracted unchanged from the
token-plan test and used by both suites.

No latency or physical-footprint improvement is claimed for these unused
planners. The independent libjxl characterization
controller (PID 3298) and its live encoder child were observed running during
this checkpoint; functional checks do not constitute a quiet timing window.

Reproduction after building the fresh candidate and retaining the frozen parent:

```sh
python3 build/representation-plan-qualification/run.py build
python3 build/representation-plan-qualification/run.py oracle
python3 build/representation-plan-qualification/run.py parity
python3 build/representation-plan-qualification/run.py conformance
python3 build/representation-plan-qualification/verify.py
shasum -a 256 -c build/representation-plan-qualification/validation.sha256
```

The runner's parent-source archive, source/library/executable manifest, binary
representation outputs, JXL/PFM outputs, full and repeated CTest logs, strict
compiler log and sanitizer logs remain local qualification artifacts. Do not
rebuild a frozen baseline against later source or headers.

## Remaining whole-serializer composition

The next combined plan must add these representation bounds to token and entropy
plans without omitting or double-counting their returned owners. In particular:

- Actual `AcEncodingCandidate` objects, copied block maps, DC stream adapters,
  candidate stream views, and phase-specific dispatch/status/profile arrays.
- Concurrent DC/order/AC optimization scratch, retained Prefix fallbacks and
  deferred ANS candidates, and section-by-width measurements whose `clear()`
  does not necessarily release capacity. Bound workers, not all possible tasks.
- Header internals: `WriteSimpleDcGlobal` runs another Prefix optimization over
  the fixed 313-token context tree, and block-map writing constructs a separate
  temporary entropy model. These are not the main DC/AC stream optimizers.
- Header/section bit bounds, every writer's largest allotment (not just emitted
  bits), parallel section writers and measurement arrays, single-group collapse,
  TOC reservation, final assembly, and independently retained codestream bytes.

Only then can frontend/device/attempt/result lifetimes be combined into an
upfront whole-job reservation. API/domain admission, failure-progress tests,
the lifetime/reuse inventory and aggregate CPU/GPU scheduling remain required;
passing these component tests does not finish the resident-execution effort.
