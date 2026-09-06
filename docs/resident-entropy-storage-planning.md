# Entropy storage planning

This continues milestone 4 of [resident execution](resident-execution.md), from
the frozen `d9134db` host/token checkpoint. **Whole-workflow admission is still
pending.** These are count-only bounds for real entropy operations, not a claim
that the serializer or encoder has an upfront reservation yet.

## Contract and production changes

[`entropy_storage_plan.h`](../src/codestream/entropy_storage_plan.h) describes
aggregation, model ownership/model writing, token emission, and policy-dependent
optimization envelopes. Successful planning allocates nothing and does not
traverse token/context arrays. Checked failures preserve the previous plan.

An optimizer plan's `working` includes its output; do not add `output` to it
again. `output` bounds capacity remaining at optimizer return. Deferred ANS
includes **all** retained width candidates, even those marked `!survives`.
`HostStorageBound::retained_bytes` inside a working bound is a sum of backing
upper bounds, not a claim that all of them survive until phase exit. Individual
replacement peaks are conservatively summed, including some sequential phases.

Inputs, previous outputs, borrowed prefix/prepared models, fixed populations,
stream-view tables, and the caller's deferred section/width measurement array
are excluded from these component bounds. The serializer must account for their
owners separately. A borrowed ANS partition must have been built within the
supplied original context/initial-histogram limits. Models must be freshly
generated within those limits, not externally supplied containers with arbitrary
historical capacity. Prefix preparation requires a cost output, as its API does.
Stack arrays, immutable tables, small control objects and allocator headers
retain the previously declared managed-memory exclusions.

The only runtime changes are:

- The high-density ANS refinement priority queue now uses `Storage<Pair>`.
  The earlier serializer attachment missed its default `std::vector` backing.
  The pair fields, comparison tuple, insertion order and merge decisions are
  unchanged; this corrects coverage rather than changing clustering policy.
- ANS emission and planning share the checked two-chunks-per-token reservation.
  This replaces an unchecked `2 * tokens.size()` multiplication.
- Aggregation and planning share the existing 4096-input/65536-dense-value
  thresholds. Private search record types move to file scope so their actual
  `sizeof` is used; their layout and algorithms are unchanged.

## Capacity derivation

The reviewed libc++ C++20 vector rules come from
[host/token planning](resident-token-storage-planning.md): fresh exact arrays
use N/N retained/peak elements; growing arrays use 2N/3N; reused exact arrays
use N/2N. Nested allocations are counted separately. Model profile constants
are asserted so an alphabet/table expansion requires revisiting the proof.

### Aggregation

Small inputs sort in place and grow the weighted destination. For maximum input
N below 4096, its bound is 2N/3N `WeightedValue` entries. At or above 4096, the
counting path reserves at most `min(N, 2^32)` output entries and owns 65536
uint64 dense counts. This output bound also covers smaller inputs taking the
sort path: their largest doubled capacity is 4096, no more than this N.

For sparse unique values, let `U = min(N, 2^32 - 65536)`. The map uses the actual
libc++ rebound node type's `sizeof`, not just the key/value pair size. Its bucket
bound follows the SDK's `__hash_table` implementation:

1. The map starts empty, never erases, and retains the default load factor one.
   `operator[]` checks duplicates before constructing a node. At most U nodes
   exist, including the newly constructed node during rehash.
2. If C old buckets trigger insertion rehash, the new distinct count is greater
   than C. The floating-point trigger preserves this implication by monotonic
   rounding; the special empty-map case is handled separately.
3. The requested bucket count is the maximum of `2*C + !power2(C)` and the
   rounded-up floating-point new size. Both are at most 2U. Next-prime rounding
   is less than twice a request greater than one (Bertrand's bound); the initial
   allocation is two. Thus a current bucket array has at most 4U pointer slots.
4. The replacement is allocated before the old bucket array is freed. Old C is
   less than U, so the overlap is at most 5U slots, not merely the new 4U.

The local SDK audit covers node layout, unique-key insertion and `__do_rehash`.
The next-prime function's contract is also explicit in the
[LLVM libc++ implementation](https://raw.githubusercontent.com/llvm/llvm-project/release/17.x/libcxx/src/hash.cpp).
This is a reviewed implementation bound, not an ISO unordered-map guarantee;
changing the map, load factor, operations or standard library requires review.

Across K clusters, raw and weighted logical counts sum to N. Their growing
vector bounds therefore sum to 2N/3N, rather than K times the whole input.
Only one cluster's dense/hash scratch exists at a time. The optimizer bounds
conservatively include that maximum scratch alongside retained weighted outputs.

### Model and emission writers

A Prefix cluster uses at most 12 configuration bits, 20 alphabet-size bits,
and a tree header of `2 + 18*4` bits plus at most `2*128` RLE entries, each at
most five code bits and three extra bits: 2154 bits per cluster. Simple trees
fit inside this bound. A nontrivial context map adds three header bits, one
single-cluster Prefix model, and at most `46*C` token bits.

An ANS histogram uses at most 20 header bits and, for each of 256 symbols,
seven depth bits, seven repeat-marker bits, eleven repeat-count bits and twelve
population bits. These deliberately loose per-symbol sums also cover flat and
one/two-symbol encodings. Each ANS configuration is at most twelve bits.

Model writing counts the outer temporary and both context-map candidate/best
writers. ANS adds the public context-map wrapper's temporary. The two Huffman
RLE arrays are serial scratch. The destination writer is separate, since it
may already contain preceding fields or sections.

Prefix tokens emit at most 15 depth bits plus 31 extra bits. ANS emits at most
31 extra bits plus one 16-bit renormalization chunk per token, and a 32-bit final
state per section. Thus the respective payload bounds are `46*N` and
`32 + 47*N`. ANS reserves exactly `2*N` reverse-chunk slots even if fewer are
emitted. Temporary payload and destination backing can coexist. BitWriter bounds
cover byte rounding, doubling and replacement; callers must include the largest
allotment ever reserved, not just the bits finally written.

### Optimizer ownership inventory

Here H is the original histogram count, K is at most `min(H,32)`, C is context
count, N is total tokens, and S is sections. All private record sizes come from
the same translation unit as their producers.

| Operation | Backing included in its complete working bound |
| --- | --- |
| Fast Prefix | H original histograms; clustered/reordered K histograms; map/symbol/distance/index scratch; one model; final partition's N exact raw values, offsets, weighted values and aggregation scratch; cluster-code configuration scratch; model writers and optional S costs. |
| Full Prefix | Original H histograms; legacy, maximum-seeded, refined, current and helper/replacement K histogram owners; retained maps/seeds/symbols; refinement codes; a conservative allowance for six named model owners; one final configured-partition raw/weighted pass, not one per screened cluster cap; optional retained prepared map/weighted vectors and S costs. |
| Direct balanced ANS | Original and mutable-copy H histograms; clustered/reordered K histograms and canonicalization scratch; two C maps; K fixed populations; one final width model and all configuration/table/writer scratch. Borrowed fixed context populations are not copied or charged again. |
| Direct high-density ANS | Direct clustering storage plus cost/version/renumbering/reverse arrays and the managed pair queue; K raw/weighted cluster owners; one aggregation scratch set; 28 configuration candidates per cluster; one final width model. |
| ANS from Prefix | Eight configuration candidates per cluster and up to four retained width models; validation writer; raw/weighted aggregation only when prepared input is not borrowed; model/table/writer and section-cost scratch. |
| Deferred ANS | Same prepared-input search, retaining four candidate slots and all generated models. The bound also covers later finalization's section-cost scratch; the caller-owned S-by-width measurement table remains separate. |

Each configuration candidate retains only its histogram frequencies. Incumbent
and current histogram-search frequencies plus remainder/rebalancing arrays are
serial scratch, not retained once per precision shift. Final ANS models add the
outer reverse-map vector, reciprocal vector, and frequency-sized reverse maps
whose total length is 4096 per nonempty histogram. Alias-table construction has
one table, distribution, cutoffs and two push/pop stacks; an active index occurs
in at most one stack. Copies into final width models coexist with configuration
frequencies.

The clustering queue needs a historical bound: its initial pairs number at most
`K*(K-1)/2`; after successive merges, at most `(K-1)*(K-2)/2` additional pairs
are enqueued. Therefore all enqueues total at most `(K-1)^2` (961 for K=32).
Stale entries and retained vector capacity make the current active-pair count
insufficient. Normal vector doubling/replacement bounds apply to this total.

Exact ANS cost measurement can retain four section arrays, the best-cost copy,
and a singleton measurement replacement. The planner explicitly accounts for
best-cost assignment overlap; this also exceeds finalization's two-array need.

## Validation and remaining work

The permanent `entropy_storage_plan` test covers:

- Fifty aggregation cases from empty input through 262144 values: dense,
  sparse-unique, repeated, random and uint32-boundary values. A separate sorted
  occurrence oracle checks values and counts. Outputs outlive the reservation;
  their real capacities and owner-class charges reconcile exactly.
- A 192-case optimizer matrix over 1/7/33/257 contexts, empty and 4097-token
  streams, random/full-width and structured populations, with/without an initial
  17-histogram map, and eight ordinary/prepared/deferred variants. Additional
  eight-variant fixtures cover empty sections, small-policy failures, and actual
  high-density refinement. Both split and interleaved emission are exercised.
  Balanced cases with initial maps borrow independently constructed fixed
  populations. Retained models, prepared values and costs reconcile to the
  ledger; deferred finalization agrees with immediate ANS selection.
- All 2929 allocation-failure positions across the small-policy variants,
  plus all 10150 positions in the twelve-context fixture that produces nine
  fast clusters and eight high-density clusters. This exercises a real queue
  push, not merely a refinement invocation with no beneficial merge. Separate
  finalization fault sweeps preserve prepared candidates on failure.
- Synthetic fresh models with all 32 clusters, a 7425-context map, deep Prefix
  trees, and full 256-symbol/4096-entry ANS tables. Repeated unaligned model/token
  writing fits bit/backing bounds. All 229 writer allocation-failure positions
  preserve the previous destination bits.
- Exhausted-reservation failures retain the typed `ResourcePlanExceeded` reason,
  preserve outputs, and leave the physical-allocation fault hook unconsumed.
  No explicit job escapes to the default domain. Invalid/overflow plans are
  atomic; large count-only cases use `2^32` tokens and `UINT32_MAX` contexts
  without traversing or allocating those inputs.

Fresh Release parent/candidate builds are `build/resident-token-plans` and
`build/resident-entropy-plans`. Final candidate CTest passes 76/77, and the frozen
parent passes 75/76. Both reproduce only the CPU `quantization_pipeline` mismatch:
actual `0.24919039011001587`, expected `0.24914586544036865`. No golden or tolerance
was changed. The entropy, entropy-storage, token-storage and serializer-storage
tests pass three repetitions each under ASan/UBSan in
`build/resident-entropy-plans-asan`, with leak detection disabled and **no source
or vendor suppressions**. The new plan/test and Prefix source pass
`-Wall -Wextra -Wpedantic -Werror`; the ANS translation unit retains two signed
comparison warnings reproduced on the exact parent source. Its qualified check
keeps those two warnings visible with `-Wno-error=sign-compare`; no arithmetic was
changed to silence them.

All 56 parent/candidate corpus/policy codestream SHA-256 values agree, all 22
pinned conformance fixtures pass on each, and separate pinned `djxl` decoding
of Kodak17, planter 4K and padded stress 4K produces identical linear-RGB PFM
hashes. Decoder revision: `e8ff09762481785938d8e4e01333ed3917571161`.

### Complete-call and memory qualification

Apple M4 Pro, 48 GiB, macOS 15.6 (24G84), AppleClang 17, fresh Release,
SIMD/fused-tuned Metal, fully resident, effort 7, distance 1.2, automatic CPU
threads. The unchanged complete-call driver includes evaluator teardown but
excludes backend creation, input loading, hashing and output writes. Seven
alternating independent process pairs per workload run 15 encodes each,
discarding the first three and retaining six measurements each for the named
image and its deterministic changed-image companion. All 840 outputs preserve
their image's size/hash. Percent changes are medians of paired process-median
ratios, not ratios of the displayed medians.

| Named workload | Parent ms | Candidate ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 23.108 | 23.257 | +0.81% | 0/7 |
| Padded 1080p | 70.964 | 70.736 | -0.24% | 5/7 |
| Padded 4K | 248.341 | 245.650 | -0.39% | 4/7 |
| Planter 4K | 258.655 | 259.624 | +0.51% | 2/7 |

Companion changes are respectively +0.50%, -0.22%, -0.87% and +0.21%.
Kodak's measured CPU-tail change is +1.89% for the named image and +5.56% for
its companion; this is not a measured zero-overhead change. This accounting
checkpoint accepts the observed small complete-call costs, without claiming a
speedup or attributing them to an unmeasured microarchitectural cause.

A separate Kodak/planter confirmation runner is retained, but its quiet-process
check stopped **before collecting any samples** because another libjxl study was
active. No other job was stopped. The completed timing cohort ended at
2026-09-06 07:54:59 UTC; memory/trim records completed by 07:55:43 UTC. The other
study's observed controller start was 08:00:03 UTC, after those measurements.
The initial slower results remain the evidence; confirmation awaits a quiet
window and must remain a separate cohort, not replace them.

Three independent alternating padded-4K memory pairs preserve all 18 encode
hashes. Median process physical peak is 2973.376 -> 2973.392 MiB; one-second idle
with the backend alive is 2003.798 -> 2016.423 MiB; one second after destruction
is 250.423 -> 249.720 MiB. These physical counters are not managed capacities.

All six separate trim processes preserve all 24 encode hashes. Each records
managed peak 2922164054 bytes, idle capacity 1946881084 bytes before trimming,
zero idle after trim, and the same idle capacity after resuming. Live/unbacked
charges are zero at these completed-call checkpoints. Median one-second
post-trim physical footprint is 260.595 -> 261.251 MiB; trim time is
21.557 -> 16.525 ms and first-resume time 330.122 -> 336.592 ms. This does not
establish a reclamation or resume-latency improvement; no Metal/cache code
changed, and these small cohorts do not resolve earlier trim-time variability.

### Artifacts and reproduction

Local artifacts are in `build/entropy-plan-qualification`: frozen library/driver
hashes, raw corpus outputs and decoded PFM files, conformance manifests/logs,
all 840 timing observations, memory/trim process snapshots, initial/final CTest
logs, sanitizer and warning logs, and the stopped confirmation's diagnostic.
The original parent drivers are copied only after validating their previous
qualification hashes; candidate drivers use the candidate's headers/libraries.
`validation.sha256` identifies the final sources/docs, manifests, test binaries,
logs, runner scripts and unchanged parent/candidate Metal libraries.

```sh
python3 build/entropy-plan-qualification/run.py build
python3 build/entropy-plan-qualification/run.py parity
python3 build/entropy-plan-qualification/run.py conformance
python3 build/entropy-plan-qualification/run.py performance
python3 build/entropy-plan-qualification/run.py memory
python3 build/entropy-plan-qualification/run.py trim
python3 build/storage-plan-qualification/summarize.py build/entropy-plan-qualification resident-token-plans resident-entropy-plans
python3 build/entropy-plan-qualification/verify_extra.py
```

Do not rebuild the frozen parent against current source. The component planners
are not yet invoked by whole-serializer admission; only the shared chunk-count
and aggregation constants feed current allocation directly.

The next integration step is the whole serializer envelope: coefficient-order
and map search, stream tables, candidate objects, policy/worker overlap, section
assembly, writer allotments and publication. Combine it with frontend/device and
retry/result lifetimes before adding public execution-domain admission. Milestone
5's last-use reductions and milestone 6's aggregate CPU scheduling remain required.
