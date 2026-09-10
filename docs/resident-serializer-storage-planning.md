# Whole-serializer storage planning

This is a component checkpoint toward milestone 4 of
[resident execution](resident-execution.md), based on `48644ab`. It composes the
[token](resident-token-storage-planning.md),
[entropy](resident-entropy-storage-planning.md), and
[representation](resident-representation-storage-planning.md) bounds into one
checked envelope for the complete CPU serializer. **Whole-workflow planning and
admission are still incomplete.** The frontend, device, attempt, diagnostic and
retained-batch-result lifetimes must be combined with this plan before imposing
the public managed-memory limit.

## Contract and boundary

[`ComputeSerializerStoragePlan`](../src/codestream/serializer_storage_plan.h)
takes pixel dimensions, codestream policy, the encode's CPU thread setting, and
whether a serializer profile is requested. It bounds
`EncodeVarDctCodestreamToBuffer` on a valid borrowed simple frame, including
validation, order/context-map selection, tokenization, entropy search, candidate
measurement, section writing, assembly, and the fresh managed output copy.

- `working.peak_bytes` is the complete conservative backing envelope, **including
  output**. The separately exposed `output`/`maximum_output_bytes` bounds the
  retained result for future batch admission; it is not another allocation to
  add to this working envelope.
- Backing includes vector capacity and replacement-before-release, not merely
  logical sizes. It follows the reviewed libc++ C++20 contract in
  `HostStorageBound`; this is neither a portable ISO-vector guarantee nor RSS.
- The input frame, preexisting caller output/profile, thread stacks, allocator
  headers and small runtime control objects are outside this serializer bound.
  An encoder-owned completed frame is still inside the **workflow** boundary;
  borrowing it here does not exempt it from that later combined plan.
- Successful planning allocates no backing and is O(1) in image size. Invalid
  options, unsupported dimensions and overflow leave the destination unchanged.
  Maximum compression normalizes sampled coefficient order to full order, just
  as the serializer does.
- The requested CPU setting must match the executing `EncodeScope`. The planner
  does not install a scope, admit a job or change production scheduling. Zero
  bounds current automatic behavior rather than assuming automatic means one
  worker or that aggregate scheduling has already been implemented.

This is deliberately an upper bound, not expected usage or the smallest possible
reservation. Named phase peaks are summed conservatively. In particular, all
task outputs plus active complete-task envelopes can count active outputs twice.
There is no measured bytes-per-pixel fit or unaccounted fallback allowance.

## Composition and concurrency proof

Let `B` be padded 8x8 blocks, `G` AC groups of at most 32x32 blocks, `D` DC groups
of at most 256x256 blocks, and `T` 8x8-block color-correlation tiles. The checked
component planners establish at most `195B` AC tokens and `6B + 2T` combined DC
and metadata tokens. Rectangular boundary groups are included.

Ordinary coding has one AC representation. Maximum compression has at most six
context maps and two order variants, hence twelve AC candidates. The order
variant is omitted from that maximum when geometry cannot generate custom-order
tokens. The representation plan includes retained cleared scans; the token plan
includes natural orders, all value/context owners, exhaustive templates, fixed
populations and worker-local tokenization/reduction scratch. Candidate map copies
and the ordinary outer map container are added by this composition.

The entropy phase has one DC task, optionally one order task, and one task per
AC candidate. Its envelope includes all task outputs plus the largest `P`
complete task-working bounds, where `P <= 8`. Sorting at most fourteen scalar
entries avoids multiplying AC-sized scratch by every candidate or treating the
much smaller DC/order tasks as AC-sized work.

Ordinary selection takes the larger FastPrefix/direct-ANS bounds, including
deferred model-cost writing. Exhaustive selection retains Prefix while preparing
ANS; prepared weighted populations survive the ANS builder but not the completed
AC task. Deferred AC candidates can retain all four alphabet-width models,
Prefix, selected/fallback copies and costs. Finalization is included. Width
measurement arrays remain charged after `clear()` because capacity is retained.

[`ComputeSerializerControlStorageBound`](../src/codestream/encoder.cpp) lives
beside the private candidate/task types, so their sizes cannot silently diverge
from a duplicated stand-in type. It includes:

- Candidate containers, stream adapters, moved-DC-token outer containers, section
  writer objects, section-size tables and the compact default context map.
- Top-level dispatcher status/thread arrays, bounded by the largest task list
  across tokenization, entropy, measurement and writing.
- Optional tokenization, entropy and section-work profiling arrays, as well as
  measurement-work arrays that production allocates even without a profile.
- Exhaustive ANS section tasks and width measurements, common DC section tables,
  per-worker AC section tables and candidate TOC-size arrays.

**Automatic nesting is part of the current implementation.**
`InExplicitParallelScope` suppresses nesting only for a nonzero per-encode CPU
setting. The two automatic DC-cost measurements can each launch a DC-group
dispatcher. Their two additional status/thread arrays and simultaneous small
header writers are included. This is storage coverage, not proof of an aggregate
CPU limit; milestone 6 must still coordinate those participants.

The writing envelope takes the largest applicable concurrency envelope among AC
emission, DC emission, common DC-header measurement, selected AC-global writing,
candidate-prefix measurement, and nested DC-header measurement. Independent
phases do not each receive a full multiplied writer envelope. Destination section
storage and retained models remain separately included throughout.

## Headers, writer reservations and output bound

[`ComputeSerializerHeaderStoragePlan`](../src/codestream/headers.cpp) lives beside
the actual format writers and fixed context-tree tokens. It combines:

- At most 120 padded file-header bits and 33 frame-header bits.
- At most 36 quantizer bits; context-map flags/counts, at most one 10-bit
  threshold, map serialization and the 313-token fixed-tree search/emission.
- Complete DC/AC/order model bounds, optional coefficient-order tokens, and all
  corresponding temporary model/header writer scratch.

The prefix/ANS model union takes componentwise maxima; it does not assume the
larger encoded bit count also implies the larger in-memory model. Generated AC
maps have at most 16 block contexts, and custom-order/DC context counts come from
their serializer definitions.

The shared token-emission limits are 46 bits per Prefix token and 47 bits per ANS
token plus a 32-bit ANS stream state. The payload bound therefore includes:

`DC_global + AC_global + 47*(AC_tokens + DC_tokens) + 90D + 32G + 7*(G + D + 2)`

Here `90D` is two ANS states plus at most 26 modular-header bits per DC group.
Each logical section receives up to seven alignment bits. The payload is rounded
up to bytes. The frame/TOC bound then adds `153 + 15 + 32S` bits, where `S` is one
for the collapsed single-AC-group case and otherwise `G + D + 2`.

The TOC term deliberately covers its `WithMaxBits(8 + 7 + 32S)` reservation,
not only bits eventually written. Likewise, entropy-model bounds include writer
reservation history. A writer's largest capacity request can exceed its final
logical size. The sum of growing section backing peaks is bounded by three times
the total padded payload capacity; final assembly and the single-group collapsed
writer are separate overlapping owners. Publication uses a fresh exact-size
byte vector while assembly is still live.

## Qualification

The frozen parent is `build/resident-representation-plans` at `48644ab`; the new
Release candidate is `build/resident-serializer-plan`. Both enable tests and
benchmarks, disable libjxl-reference fixtures and compile-time Metal profiling,
and use Apple Clang 17 on the Apple M4 Pro/macOS 15.6 qualification machine.
Do not rebuild a frozen comparison tree against the later source.

The permanent [serializer-plan test](../tests/serializer_storage_plan_test.cpp)
covers:

- Allocation-free planning, invalid inputs, overflow/output atomicity, policy
  normalization, nondecreasing thread bounds and profile/no-profile bounds.
- 128 file/frame-header combinations across every dimension selector transition,
  maximum format dimensions and endpoint matrix scales; 56 quantizer selector
  combinations through their maximum values. These write actual headers inside
  their calculated backing allowances without allocating giant images.
- Twelve valid synthetic frames: tiny, padded, transposed, mixed-strategy,
  multi-AC/DC-group and context-map threshold geometry; zero, dense and sparse
  `INT32_MIN`/`INT32_MAX` AC values. All 576 complete encodes fit their reservations
  across three entropy policies, two order settings, four thread settings
  (automatic/1/2/8), and profiling on/off. Every output matches its serial oracle.
- Exactly one managed serializer backing remains at successful return. Closing
  the producer reservation does not uncharge it; public vector publication does.
- Exhaustive serial physical-allocation fault sweeps for the small dense frame:
  632 balanced, 1370 high-density and 2630 maximum-compression positions (4632
  total). Each failure preserves caller output and the complete profile, releases
  partial backing, and retains the ordinary allocation-failure status. The final
  unconsumed-hook success matches the oracle. The physical hook is thread-local;
  these sweeps do not claim injection into every parallel worker.
- A zero-credit admitted reservation fails with `ResourcePlanExceeded` before
  physical allocation, without growing, publishing or escaping to the default
  domain.

Separate parent/candidate oracle drivers use the same fixture source and each
revision's own headers and libraries. All extracted parent headers are checked
against Git blob IDs before compilation. Their 72 synthetic codestreams match
exactly: 2,184,980 bytes including little-endian length framing, SHA-256
`25ebafe4c03e5a5d355e0b080160ad25655ba38224232d1d0fa81d5ea167073f`.

All 56 whole-workflow corpus/policy codestream pairs match byte-for-byte hashes.
Independent pinned `djxl` decoding of Kodak17, planter 4K and padded stress 4K
matches linear-RGB PFM hashes. Both builds pass all 22 pinned conformance fixtures;
the decoder is v0.13.0 at `e8ff09762481785938d8e4e01333ed3917571161`.
The Metal library's shader payload is unchanged.

Full Release CTest passes 77/78 on the parent and 78/79 on the candidate. Both
reproduce only the CPU `quantization_pipeline` golden mismatch, actual
`0.24919039011001587` versus expected `0.24914586544036865`. No golden or tolerance
was changed. The earlier checkpoint's intermittent Butteraugli capacity-cache
test failure does not reproduce in these two runs; that is not proof of a fix
or an OS-reclamation root cause.

The serializer, entropy and representation plan tests each pass three repetitions
with AddressSanitizer and UBSan (nine runs, RelWithDebInfo and frame pointers).
Leak detection is disabled, UBSan halts on errors, and no source/vendor checks
are suppressed. The new planner/test and affected encoder/header implementations
pass `-Wall -Wextra -Wpedantic -Werror` syntax checks. `ans.cpp` passes with only
`-Wno-error=sign-compare`; its two warnings reproduce identically in the frozen
parent at lines 469 and 520. They are not silently counted as a warning-free run.

## Evidence and remaining work

`build/serializer-plan-qualification/` retains runners, own-header oracle build
commands, library/executable/source hashes, raw JXL/PFM outputs, pinned decoder
logs, conformance manifests, full Release logs, verbose failure counts and
sanitizer logs. `verify.py` re-reads the binary hashes and actual outputs as well
as the manifest counts and known failures; `validation.sha256` seals the final
qualification inputs/logs. Permanent regression coverage is committed; the local
qualification harness and generated artifacts are not installed tools.

After building and freezing the identified configurations:

```sh
python3 build/serializer-plan-qualification/run.py build
python3 build/serializer-plan-qualification/run.py oracle
python3 build/serializer-plan-qualification/run.py parity
python3 build/serializer-plan-qualification/run.py conformance
python3 build/serializer-plan-qualification/verify.py
```

No quiet timing or physical-footprint cohort is claimed. The separate runtime
characterization controller (PID 3298) and an active encoder child were verified
live during qualification and were not altered. This checkpoint adds planning
and shared constants/private-type placement, not a production admission call,
numerical-policy change or memory/latency reduction. Whole-workflow composition,
public execution domains, cache-aware admission, retained batch results, audited
last-use reductions and aggregate CPU scheduling remain required by the roadmap.
