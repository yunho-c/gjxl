# S183: pack ANS reverse chunks into 56-bit words

The private packed ANS writer passes primitive and integrated host tests in
normal and AddressSanitizer builds. It remains unpromoted: no whole-image
qualification or performance campaign has run, and storage reliability is
still unresolved. Production and GPU code remain unchanged.

## Why this experiment

The [S182 campaign](cuda-bounded-appender-qualification-s182.md) ended with
an unresolved `ENOSPC` error and mixed whole-encoder results for S181's
bounded token appender. This stage does not resume that interrupted campaign
or combine the two candidates. Its parent is
`0a1c1399afbb0df7f482c9cde28775cbd508e8f3` on `feat/cuda`; production remains
the qualified S168 implementation.

C: was rechecked at 1,098,903,552 bytes free, U: at 30,883,840. A read-only
`fsutil volume diskfree C:` query returned error 5, access denied. No
elevation was requested, and no storage/security/settings change was made.
The mounted Windows Recovery volume was not used for artifacts. Long GPU
timing remains paused; the small CPU-only work below is separately scoped.

[S165](cuda-ans-writer-decomposition-s165.md) found material ANS staging and
emission costs. S167 batched forward bit writes, and S168 improved aligned
append, but neither removed reverse-chunk storage. The current writer still
reserves two eight-byte `ReverseBitChunk` records per token, pushes one record
for each nonempty extra-bit or renormalization chunk, then reads the records
in reverse order to form the final bitstream. The recurrence and chunk
staging were not separately timed in S165, and those old phase timings do
not quantify an achievable saving in current production.

The new private prototype packs chunks into fixed 56-bit words as they are
generated. The final emission traverses these words instead of individual
chunk records. This removes the per-record width field and moves bit packing
into the recurrence callback. It does not remove the final word traversal,
temporary BitWriter, output copy, ANS recurrence, or any coefficient scan.
Packing adds work on the recurrence path and may not improve whole time.

Evidence is under `build-cuda-ninja/profiles/s183-artifacts`, with versioned
`s183_*` helpers beside it. The 343 unchanged production sources are pinned
through the existing S182 source archives, avoiding another duplicate copy.

## Bit order and bounds

Let `P` contain `n` pending bits, with `0 <= n < 56`, and let an incoming
chunk contain `b <= 31` valid low bits `V`. The required prepend is the
conceptual integer `(P << b) | V`; it preserves each chunk's internal
LSB-first order while reversing chunk order. Merely reversing complete
forward-packed words would not generally preserve that order.

If `n + b < 56`, the combined value fits a native word and becomes pending.
Otherwise let `s = n + b - 56`. The oldest/high 56 bits are stored as
`(P << (56 - n)) | (V >> s)`, and the newest/low `s` bits remain pending as
`V & ((1 << s) - 1)`. Here `0 <= s <= 30`; both stored expressions fit
56 bits even when the conceptual combined value needs 86 bits. One incoming
chunk completes at most one full word. The full-word array is eventually
read backwards, after the remaining newest pending bits.

Emission writes the final 32-bit ANS state, pending bits, then reversed full
words. State and pending are combined into one checked write when their
total is at most 56 bits; otherwise they use two writes. Every later word
uses the existing checked `WriteBits(56, ...)`. The temporary writer is
appended once to the destination; there are no unchecked stores, byte-order
reinterpretations, padded logical lengths, or new public APIs.

ANS emits at most 31 HybridUint extra bits and one 16-bit renormalization
chunk per token. The integrated overlay checks `N <= SIZE_MAX / 47`, then
requests `floor(47N / 56)` uint64 slots. Any remaining partial word lives in
the packer itself. This is a token-bound reservation, not an extra prepass
over tokens or chunks. For an AC group at the existing 199,680-token bound,
the old requested chunk reservation is 3,194,880 bytes and the new requested
full-word reservation is 1,340,704 bytes, about 58.0% lower. These exclude
object/allocator overhead and temporary/destination output buffers, and
are not resident-memory or peak-process measurements. The bound is specific
to that AC-group size, not every possible DC/order stream.

For a valid payload of `B` bits, the stored full-word count is exactly
`floor(B / 56)` and pending length is `B % 56`. Since each original chunk
has at most 31 bits, this reduces logical array storage for sufficiently
large nonempty streams; tiny/empty streams have different metadata tradeoffs.
No timing or memory-census result is inferred from the bound alone.

## Primitive qualification

The algebra audit checks all 56 pending positions and 32 incoming widths
with endpoint and alternating patterns for both operands: 28,672 comparisons
against an arbitrary-precision conceptual prepend, including 496 exact
boundary and 7,440 split-pattern cases. This supplements the invariant
argument; it is not exhaustive enumeration of all 86-bit combined values.

Normal MSVC and clang-cl AddressSanitizer builds compile only the primitive
fixture and unchanged production BitWriter. Each passes 102,552 exact-output
cases and 102,552 insufficient-allotment rollback checks. Expected bytes
come from an independent bit-at-a-time oracle, not BitWriter or the packing
formula. Every initial byte alignment and output tail is covered.

Coverage includes every pending position/incoming width with six value
patterns; every stream length 0..257 with eight chunk patterns; and lengths
4,096, 65,536 and 399,360 with zero, maximum-width, alternating 31/16 and
random-width chunks. The largest maximum-width primitive case intentionally
exceeds ANS's 47-bit-per-token bound. Null destinations are also rejected.
Each build observes 6,081,088 input chunks, 1,710,194 full packed words,
98,515,702 payload bits, 113,670 exact boundaries and 1,596,524 split
boundaries across the fixtures. These are test totals, not encoder workload
counts. No allocation-fault injection or new malformed-chunk API is claimed.

The first launcher omitted the hyphen on `-ExecutionPolicy`. PowerShell
rejected it before invoking the build script, and the build directory was
verified empty. Its log/journal and pinned launcher are retained. A new
`s183_recover_v2.py` corrects only the invocation. No candidate or build-script
correction was required. The successful primitive build ran 09:55:24.687–
09:55:45.221 UTC on 2026-09-10; normal and ASAN fixtures finished at
09:55:46.222 and 09:55:49.778 UTC. The six primitive build artifacts total
580,173 bytes. Their run durations are not performance measurements.

## Private ANS integration

`s183_ans.cpp` is an isolated overlay of current production `ans.cpp`.
The source audit reconstructs it exactly by replacing only the split-view
`WriteAnsTokenStream` body and adding the packer include. ANS model
construction, context/config validation, HybridUint conversion, reciprocal
division, renormalization, reverse maps, public entry points, and count-only
paths are unchanged. The original `ProcessAnsTokenStream` supplies the
packer callback; its return-status checks and allocation/length catches are
retained. Allocation failure can discard the local packer without publishing
the destination, but allocator-failure behavior is not fault-injected here.

The overlay is tested with the existing bit-emission, entropy and private
AC-section validation suites in normal and ASAN builds. Frozen support
libraries are reused read-only; the newly compiled BitWriter objects override
older archive members so both builds use current S168 append behavior.
There is no new device compilation or whole-image/GPU benchmark campaign.

The integrated build ran 09:59:26.108–10:00:34.678 UTC, and all six test
processes passed between 10:00:35.287 and 10:00:56.994 UTC. There was no
integrated source, build or test correction. Each build passes:

- The existing independent-reference ANS bit-emission test: all 816 valid
  HybridUint configurations, 30,240 exact comparisons and 30,240 allotment
  failures, mapped/unmapped contexts, interleaved/offset-split storage,
  every extra-bit width 0..31 and every output tail. The reference uses
  ordinary division/modulo and unbatched writes, not candidate packing.
- The existing entropy suite, including 443,904 value encodings, scanned
  ANS configuration/layout checks, model policies, and 40 late section
  failures after populated and empty sections.
- The private AC-section suite: 192 valid batches, 1,408 invalid models,
  96 invalid-token cases and 64 shared-model concurrent checks. Its legacy
  versus private-section comparison shares the candidate writer; the
  independent byte oracle is supplied by the bit-emission test above.

The unchanged bit-emission test still prints 1,651,280 reference chunk writes
and 808,688 predicted S167 batched writes. The latter is a baseline simulation
in that test, **not** a measured or predicted count for this new packer.
No allocation-fault campaign, ThreadSanitizer run, whole-encoder corpus test,
CUDA sanitizer run or full CTest run is claimed.

The fourteen integrated objects/executables total 12,067,339 bytes. C: still
had 1,083,367,424 bytes free afterwards. Small artifact writes succeeded,
but this does not explain or clear the S182 disk error.

## Decision and remaining work

No performance or production-retention decision is made here. The prototype
changes both staging representation and its reservation bound; a later
performance experiment must attribute the combined change accordingly or
split those policies into explicitly declared variants. Required follow-up
includes full whole-encode byte/summary checks on the established corpus,
resource and sanitizer qualification appropriate to the integrated encoder,
and repeated unfiltered whole-encoder timing with duplicate controls.

Storage reliability/headroom remains unresolved. Passing these small CPU
tests does not establish that another long campaign is safe. This stage
does not complete the optimization goal or demonstrate that the fully
resident path is maxed out.

## Preservation and verification

Eleven jobs are journaled: ten accepted and the initial launcher rejection.
No prior artifact, pinned helper or failed journal is overwritten. There is
no deletion, recovery-volume use, privilege elevation, or power/clock/thermal/
affinity/priority/security change. The frozen S182 870-file inventory and
all production sources remain unchanged. Source audits, primitive algebra,
test markers, commands, nonoverlapping jobs, support inputs and output hashes
are checked before freezing the new evidence.

Read-only frozen verification from the repository root:

```powershell
python build-cuda-ninja/profiles/s183_verify.py --frozen
```

The report is the only tracked change. The goal remains active; the next
performance decision must be based on the integrated fully resident encoder,
not this primitive's reservation arithmetic or host correctness results.
