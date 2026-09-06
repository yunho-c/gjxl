# Host-vector and token storage bounds

This continues milestone 4 of [resident execution](resident-execution.md) from
`ccfaac9`, the shared device-layout checkpoint. **Whole-workflow admission is
still pending.** The new component bounds cover tokenization, not every serializer
allocation and not the public encoder's complete managed-memory envelope.

## Contracts and production use

[HostStorageBound](../src/core/host_storage_bound.h) performs checked sums of
vector backing bounds. Each limit must cover all sizes and explicit reserve
requests since that owner was initially empty, including previous larger groups
when worker scratch is reused. It does not substitute the current logical size
for retained capacity. Nested elements' allocations require their own entries;
inline owner objects count only when they themselves occupy allocated backing.

For a maximum count `N`, in units of `sizeof(T)`:

| Allocation contract | Current backing bound | Including replacement overlap |
| --- | ---: | ---: |
| Fresh exact allocation, bounded writes | N | N |
| Reused exact reserve/assign, bounded writes | N | 2N |
| Implicit vector growth also allowed | 2N | 3N |

These are reviewed **libc++ C++20** contracts, not portable ISO guarantees. The
SDK's `__vector/vector.h` uses exact requested capacity for fresh count and
forward-range constructors and `reserve`; growth uses
`max(2 * old_capacity, new_size)`, capped at `max_size()`. For a growth request
bounded by N, the replaced capacity is less than N and the new capacity is at
most 2N. A shrinking replacement is also covered: the old capacity is at most
2N and the new capacity at most N. The managed allocator rejects impossible
element counts before multiplication; byte sums, factors and multiplicities
are checked before publishing a result.

Single-pass input ranges, `vector<bool>`, moves from a larger unaccounted owner,
and allocations inside elements are not covered by that growth contract. Both
this helper and the existing publication owner reject unreviewed standard
libraries/language modes at compile time. The version gate does not prove a
future libc++ implementation retains these algorithms: source review and the
real-allocation tests remain necessary when upgrading the toolchain.

`retained_bytes` is a conservative sum of current backing capacities, distinct
from replacement overlap. It is **not** a claim that all those temporaries remain
live at phase exit. Summing individual peaks is intentionally conservative;
these are neither allocator-header/RSS estimates nor measured working sets.

[Token storage plans](../src/codestream/token_storage_plan.h) share reservation
counts with the production AC-template/direct and DC/metadata tokenizers:

- An AC group has at most 32x32 blocks. For B covered blocks and A transform
  anchors, validated coverage implies 64B coefficients per channel. Existing
  reservations remain `3 * (64B + A)` values/descriptors or values/contexts,
  with `3A` template block keys. Before placement, use A <= B. This preserves
  the parent's capacity requests; it does not change scan order or emit more
  tokens to fill the reservation.
- Direct balanced tokenization has at most `min(context_count, token_capacity)`
  sparse context populations and at most
  `min(token_capacity, context_count * 128)` distinct context/symbol entries.
  The latter vector grows implicitly, so its bound includes both spare capacity
  and replacement overlap. Reused anchors/maps/accumulators are separate from
  newly retained group outputs.
- The seven supported natural-order tables retain 2624 `uint32_t` entries;
  one permutation-validation bitmap, at most 1024 bytes, can coexist. The direct
  path shares one prepared table; template construction can have one per worker.
- A DC group has at most 256x256 blocks. Its DC stream has 3B tokens; metadata
  has `2M + 2A + B`, where M is its color-tile count. Coverage/anchor validation
  scratch is included independently. Runtime strategy and value validation is
  unchanged and is not replaced by a geometry-only plan.

`ComputeTokenizationStoragePlan` sums interior, right-strip, bottom-strip and
corner group classes without walking all image groups. It includes every
retained token owner, allocated outer token containers, direct fixed-population
reduction capacity, and worker scratch. Exhaustive encoding retains each order's
values/templates and each order/map pair's context array. Template construction
and context materialization are joined phases, so their worker scratch peaks
are combined with a maximum, not concurrent addition. The retained output sum
conservatively includes all templates and materialized contexts together.

The caller supplies policy-derived upper bounds on context count, map/order
variants and workers. This component does not independently choose or prune
those policies. Only the shared group-count functions currently feed production
allocation sites; the aggregate bound awaits the complete upfront planner.

## Permanent tests

[token_storage_plan_test.cpp](../tests/token_storage_plan_test.cpp) checks:

- Every 1..32 by 1..32 AC geometry, three anchor bounds, and four context limits
  with/without fixed populations; independent frozen count/backing recipes.
- Every 1..256 by 1..256 DC geometry with one and maximum anchors; 144 ordinary/
  exhaustive aggregate cases compared against explicit group traversal and a
  separate global DC-count formula. Large geometry planning does no image work.
- Real vector growth, exact reservation, assignment, shrinking, forward-range
  insertion, retained smaller sizes, overflow, and failure-atomic bound sums.
- 144 real tokenizer configurations: seven transform families plus mixed
  multi-group frames; zero, dense and sparse signed-extreme coefficients;
  ordinary direct with/without populations and two order variants with twelve
  retained context arrays per group (six map slots); 1/8 worker participants.
  Inputs/maps/custom orders and
  independent comparison-oracle temporaries are outside the tested component's
  reservation. The aggregate harness allocates the reduction's exact-sized
  population destination but does not invoke the private reduction routine.
- Direct/template token parity, alternating larger/smaller groups using one
  scratch owner, rejection under a one-byte reservation before physical
  allocation, and 50 individually injected allocation failures across direct,
  template, context materialization and DC-group tokenization. Outputs remain
  unchanged on failure, charges drain, and no allocation escapes to the default
  domain. Pure planning is checked with an armed managed-allocation hook, not
  global `operator new` interception.

## Qualification

Fresh Release candidate: **75/76**; frozen `ccfaac9` parent: **74/75**. The sole
failure in both is the inherited CPU `quantization_pipeline` golden:
`0.24919039011001587` actual versus `0.24914586544036865` expected. No golden or
tolerance was changed. The new test and AC/DC tokenizer tests also pass twenty
repetitions each with AddressSanitizer and UBSan (leak detection off, no source
or vendor-header suppression needed by these CPU tests).

All 56 corpus/policy cases retain identical codestream SHA-256, including 38
natural/padded images, efforts 1-10, density/compression modes, diagnostics,
exact coefficients, throughput modes, target-size retries and maximum error.
Both builds pass all 22 pinned conformance fixtures. Independent `djxl` decoding
of Kodak17, planter 4K and padded stress 4K retains identical PFM hashes; decoder
revision `e8ff09762481785938d8e4e01333ed3917571161`. Metal libraries are identical.
The two modified tokenizer sources, aggregate planner and new test also pass
`-Wall -Wextra -Wpedantic -Werror` syntax checks.

### Complete-call measurements

Apple M4 Pro, 48 GiB, macOS 15.6 (24G84), AppleClang 17, Release; fully-resident
Metal, fused-tuned AC, SIMD CPU, distance 1.2, effort 7, automatic CPU threads.
Seven alternating independent process pairs per workload each encode 15 times,
alternating the named input and a changed-image companion. The first three
encodes are warmups; each image then has six retained observations per process.
The timer includes synchronous workflow teardown, excluding backend creation,
input loading/generation, process startup and output writes. All 840 encoded
outputs retain their expected per-image hash and size. Process snapshots check
for other GJXL jobs, not exclusive machine ownership or controlled thermals.

| Named input | Parent ms | Candidate ms | Median paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 23.353 | 23.288 | -0.46% | 5/7 |
| Padded 1080p | 71.063 | 71.193 | +0.02% | 3/7 |
| Padded 4K | 244.455 | 249.734 | +1.05% | 3/7 |
| Planter 4K | 258.181 | 258.200 | +0.21% | 3/7 |

The millisecond columns are medians of process medians; paired changes are
medians of pairwise ratios, not ratios of those columns. Companion changes are
+1.01%, -0.18%, +0.94%, and -0.17%, respectively. In the padded-4K cohort the
named input's tail changes -0.89%, but the companion's tail changes +2.80%.

A separate seven-pair padded-4K confirmation, with the same protocol and 210
hash-checked encodes, gives 244.217 -> 244.652 ms for the named input (-0.18%
paired, 4/7 faster) and 245.234 -> 240.808 ms for its companion (-1.01% paired,
6/7 faster). Their tail changes are -0.03% and -2.05%. Both cohorts are retained;
the initial slower result is not discarded. These small, direction-changing
observations establish neither a repeatable speedup nor zero overhead.

### Physical memory and reclamation

Three alternating padded-4K pairs retain two source images throughout each probe.
Median process peak is 2970.579 -> 2972.611 MiB; one-second backend-alive footprint
is 2018.111 -> 2019.142 MiB, and one second after backend destruction is
252.142 -> 253.204 MiB. These are physical counters, not managed admission limits.

Six separate trim-probe processes each record exactly 2,922,164,054 bytes of
managed backing peak and 1,946,881,084 bytes of idle capacity after encoding.
Every process reaches zero live, idle and unbacked managed bytes after trim;
resume restores the same idle capacity and output hashes. Physical footprint
one second after trim is 263.407 -> 261.454 MiB. Median trim time is
6.105 -> 4.522 ms and first-resume encoding is 329.716 -> 333.348 ms. This small
cohort is not a reclamation-speed improvement claim and does not resolve the
earlier device-plan checkpoint's slower trim observations. Cache/trim code is
unchanged; eventual admission-pressure qualification must still include eviction
and resume cost. All 24 trim-probe outputs are hash-stable.

### Reproduction and artifacts

Local ignored artifacts are under `build/token-plan-qualification/`. Frozen
parent builds/drivers are reused only after checking their archived library and
executable hashes; candidate drivers are compiled against candidate headers and
libraries. The runner records commands, executable/library/source hashes, input
hashes, retained output files and process snapshots. The verifier checks all
retained corpus/decoded files, decoder/conformance evidence, timed output hashes
and physical-probe hashes; the extra verifier checks the confirmation cohort
and every trim process's exact capacity counters.

```sh
ctest --test-dir build/resident-token-plans --output-on-failure
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/resident-token-plans-asan \
  -R '^(token_storage_plan|ac_group|dc_group)$' --repeat until-fail:20 --output-on-failure
python3 build/storage-plan-qualification/summarize.py \
  build/token-plan-qualification resident-storage-plans resident-token-plans
python3 build/token-plan-qualification/verify_extra.py
```

The full Release command retains the documented inherited failure; it is not a
fully green command. Qualification scripts/artifacts are local records, while
the source tests above are durable repository coverage. No aggregate scheduling
or batch-throughput benefit is asserted by this token-planning checkpoint.

## Remaining whole-workflow planning

The aggregate token bound explicitly excludes frame/input backing, coefficient-
order search and order tokens, block-map/candidate objects, stream-view tables,
dispatch/status/profile arrays, entropy models and candidates, ANS reverse
chunks, and writers/publication buffers. Those owners already have accounting
attachments; their defensible upfront bounds still need to be combined with
these token and device plans, including simultaneous owners and retry/batch
retention. Then implement public execution-domain admission, cache shedding and
progress/failure qualification as specified in [resident resources](resident-resources.md).
Milestones 4, 5 and 6 remain required; this checkpoint does not complete them.
