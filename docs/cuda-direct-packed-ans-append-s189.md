# Direct checked packed-ANS append: S189

Date: 2026-09-10. Parent: `d04bb6a51508e6cb66aa92833e92a36f1f76230d`.
Production is unchanged. The private candidate passes its CPU diagnostic gate
and is ready for separately qualified resident whole-encoder evaluation.

## Result and decision

Removing the packed owner's temporary BitWriter and final copy improved the
primary metric in **48/48 synthetic actual-ANS cells**, including **16/16**
large-stream cells. Their geometric-mean direct/growable ratio was
**0.8051657462**, a **19.4834%** reduction.

| Prospective diagnostic condition | Required | Observed |
|---|---:|---:|
| Wins across all 48 cells | at least 36 | 48 |
| Wins across 16 large-stream cells | at least 12 | 16 |
| Large-stream geometric-mean ratio | at most 0.99 | 0.8051657462 |

All conditions pass. This permits a new whole-encoder qualification against
current production, not immediate promotion or a claimed image-encoding speedup.
Neither the [S187](cuda-fixed-reverse-bit-owner-s187.md) nor
[S188](cuda-integrated-ans-owner-screen-s188.md) failed owner-only gate is changed.

Large dense-low streams improve 20.42–21.62%, mixed streams 15.26–16.29%, and
wide streams 33.09–37.75%. Large zero-heavy streams improve only 0.40–2.82%.
These are synthetic distributions; they do not predict photographic gains.

## Isolated implementation change

The candidate starts from S183's **growable** reverse-word owner. It does not
combine S187's fixed-capacity storage change.

The entire ANS source is exactly S183 except for the private helper include and
type name. The helper's reservation, storage, push arithmetic and fields are
unchanged. Only its append implementation changes:

1. Reject a null destination, and reject payload lengths greater than
   `SIZE_MAX - 32` before adding the final ANS state.
2. Call the existing `destination->WithMaxBits(payload + 32, operation)`.
   This checks the destination's accumulated size and any parent limit, and
   reserves the complete final output capacity before emission.
3. Inside that transaction, write the same state, pending tail and reverse-order
   56-bit words directly with existing checked `WriteBits` calls.
4. Return the transaction status. No temporary BitWriter is constructed and no
   final `BitWriter::Append` is performed.

When pending width is at most 24, state and tail share one at-most-56-bit call;
otherwise state and tail use separate calls. This order and arithmetic are
unchanged from S183. All full-word calls remain checked, including ordinary
width/value validation and bytewise emission. This is not a raw-store or
unchecked-output prototype.

The operation object is stack-resident; its pointer-sized lambda target is used
synchronously by WithMaxBits. Explicit outer catches cover `bad_alloc` and
`length_error` during std::function construction as well as any escaping
allocation path. The implementation does not assume that std::function
construction is allocation-free on every implementation.

There is **no BitWriter implementation, header, layout or ABI change**.
No compatibility layer, new public API, fixed owner, selector, counter,
forced-inline change or GPU change is added.

## Bounds, lifetime and atomicity

The private packer's existing contract keeps cumulative payload representable,
all full words within 56 bits, and pending width below 56. The new 32-bit-state
addition is guarded explicitly. Integer models check that boundary for 32- and
64-bit size_t; they are not 32-bit binary qualification.

WithMaxBits retains its existing checks for addition overflow, padded-byte
overflow, maximum vector size and parent allotments. Its reservation happens
before the callback. The callback reads the const owner's independent word
vector while mutating only the destination, so output reservation cannot
invalidate source iterators. The stack operation and owner both outlive the
synchronous callback.

On a callback error or caught exception, WithMaxBits restores the previous
logical bit count, vector size and previous last byte. Nested calls restore the
parent allotment pointer. The new primitive tests explicitly cover a successful
direct append followed by a later enclosing failure, not just early rejection.

No allocator-failure injection was performed. Allocation-failure behavior is
supported by the unchanged transaction implementation and the new construction
catches, not by a claim that every allocation site was fault-injected.
MSAN, TSAN and GPU memory checks are outside this CPU stage.

## Correctness qualification

The S183 bit-at-a-time oracle and deterministic seed are retained. All pending
positions 0..55, chunk widths 0..31, eight destination alignments, multiple
flushes, zero-width chunks, and large streams remain covered. Added checks
perform a second append into the same writer and force both a late limit error
and an exception after successful direct append.

Each normal/ASAN primitive build passes:

| Check or input-work count | Per build |
|---|---:|
| Primitive cases | 102,552 |
| First exact output comparisons | 102,552 |
| Repeated exact output comparisons | 102,552 |
| Insufficient-allotment rollback | 102,552 |
| Post-append outer-limit rollback | 102,552 |
| Post-append outer-exception rollback | 102,552 |
| Input chunks packed | 6,081,088 |
| Full words packed | 1,710,194 |
| Input payload bits | 98,515,702 |

Combined normal/ASAN totals are **410,208 exact output checks** and
**615,312 rollback checks**. Null destinations are also rejected in every case.
The three input-work rows count packing once per case, not every repeated
emission performed by the extended fixture.

Four existing host fixtures pass in each build:

- ANS bit emission: 816 configurations, 30,240 comparisons and 30,240
  atomic-failure checks. Its `predicted_batch_writes` remains an existing
  reference calculation, not a measured candidate call count.
- Entropy: 443,904 value encodings across 816 configurations, both storage
  layouts and mapped/unmapped scans, plus 40 late section failures.
- AC section validation: 192 valid, 1,408 invalid-model, 96 invalid-token
  and 64 concurrent cases.
- BitWriter: cross-byte writes, append modes, nested allotments and atomic errors.

The exact sealed S188 normal/ASAN harnesses then qualify the new candidate DLL.
All 1,152 DLL-boundary calls pass: ordinary and exact-limit emission,
insufficient-limit rollback, early/late invalid contexts and null destination.
The three harness processes (normal check, ASAN check and timing setup) also
pass 72 production-reference outputs and 72 count-only checks.

Normal builds use MSVC 14.37; ASAN uses LLVM 22 clang-cl. Current S187 BitWriter
objects are reused unchanged. Normal host fixtures use S168 codestream/S166
support, and ASAN uses S162 support with the appropriate S166 encoder object.
The private AC fixture includes encoder source itself. All support inputs are
hash-verified; this stage executes no GPU work.

## Actual native boundary

The new normal/ASAN DLLs export the actual ANS writer directly under the
existing sealed harness's `S188Write` export name. Labels 0/1 still share the
unchanged S188 growable DLL; labels 2/3 now share the **direct-append** candidate.
The old harness's local variable names and S188 log tags are retained; they do
not mean fixed-capacity ownership is being tested.

The complete raw COFF/PE audit verifies linked code against object bytes after
masking only REL32-family fields, and checks the actual export RVA and
writer-to-recurrence / recurrence-to-callback destinations.

| Selected normal native entry section | Growable | Direct |
|---|---:|---:|
| Writer instructions / bytes | 192 / 812 | 192 / 812 |
| Recurrence instructions / bytes | 289 / 1140 | 289 / 1140 |
| Packing callback instructions / bytes | 51 / 170 | 51 / 170 |

The recurrence and callback bytes and normalized relocations match exactly.
This is not merely a count comparison. Matching writer counts alone do not
assert writer relocation equality: its output-helper target intentionally changes.

A postmeasurement output-method audit verifies the old helper's final
BitWriter::Append target and the new helper's WithMaxBits target, with no final
BitWriter::Append target in the new helper. The selected owner Append sections
are 280 instructions / 1,078 bytes versus 90 / 401. The new method delegates
emission to the transaction lambda; these counts exclude callees and separately
outlined handlers and do not represent total executed work or explain speedups.

## Fixed synthetic CPU screen

The unchanged S188 harness and source-defined inputs are reused. There are
24 datasets: 1,024 / 16,384 / 199,680 tokens, four distributions, and two layouts.
The deterministic generator, 458 contexts, initial eight-histogram partition,
balanced ANS optimization and independent ordinary-division/bit-at-a-time oracle
are described in S188. Actual optimized models have one to three clusters.
All dataset metadata and oracle results match between the qualification and
timing processes. Token fingerprints are independently reconstructed by the
verifier; artifact integrity is checked with SHA-256.

The four distribution rules are zero-heavy (95% zero, otherwise 1..15),
dense-low (0..511), mixed (80% 0..7 / 15% 0..255 / 5% 0..65535),
and full-range 32-bit wide values. They are not captured coefficient data.
Layout pairs share tokens and models but have different destination prefixes
(`case % 8`); layout and alignment effects cannot be separated from this matrix.

All datasets are preconstructed before timing. Forward and reverse passes
produce 48 cells, each with four warm and 32 measured four-label Williams rounds
(0132 / 1203 / 2310 / 3021, order index reversed on pass one).
There are **6,912 exact checked calls**, including **6,144 measured**.
No timing rerun, sample filtering, performance stop or concurrent build,
sanitizer or heavy hash verification occurs.

The timer includes token-view construction, actual ANS writer/callback,
allocation, packing, checked append and staging destruction. Input/model/oracle
construction, prefix setup, output/status checking, logging and final output
destruction are outside. No GPU/NVML or clock locking is involved.

Primary delta is the median within-round direct duplicate mean minus baseline
duplicate mean. Percentage is the median within-round ratio minus one; it is
not a ratio of independently computed medians. The large-stream aggregate is
the geometric mean of those per-cell ratios. All four cross-pair statistics and
both same-path duplicate deltas are retained. The gate is a screening rule,
not a claim of independent samples or statistical significance for every cell.

Complete cell table; negative means direct is faster. Layout is followed by
destination prefix width in bits.

| Case | Tokens | Distribution | Layout / prefix | R0 change % | R0 delta ns | R1 change % | R1 delta ns |
|---:|---:|---|---|---:|---:|---:|---:|
| 0 | 1024 | zero-heavy | interleaved / 0 | -3.0000 | -300 | -2.8259 | -275 |
| 1 | 1024 | zero-heavy | split / 1 | -4.7962 | -500 | -4.9505 | -500 |
| 2 | 1024 | dense-low | interleaved / 2 | -26.5152 | -5200 | -26.3568 | -5225 |
| 3 | 1024 | dense-low | split / 3 | -25.0296 | -5300 | -23.9669 | -6800 |
| 4 | 1024 | mixed | interleaved / 4 | -22.4359 | -3500 | -22.6481 | -3250 |
| 5 | 1024 | mixed | split / 5 | -21.9814 | -3550 | -22.0316 | -3200 |
| 6 | 1024 | wide | interleaved / 6 | -43.2658 | -18100 | -43.7926 | -16150 |
| 7 | 1024 | wide | split / 7 | -40.2192 | -17450 | -40.9154 | -15650 |
| 8 | 16384 | zero-heavy | interleaved / 0 | -0.6618 | -1025 | -0.9002 | -1425 |
| 9 | 16384 | zero-heavy | split / 1 | -2.7761 | -4650 | -2.8528 | -4700 |
| 10 | 16384 | dense-low | interleaved / 2 | -21.4198 | -75250 | -20.9084 | -69400 |
| 11 | 16384 | dense-low | split / 3 | -20.4728 | -72725 | -20.3589 | -68850 |
| 12 | 16384 | mixed | interleaved / 4 | -16.9411 | -46525 | -16.8261 | -58300 |
| 13 | 16384 | mixed | split / 5 | -16.5141 | -45025 | -16.7587 | -61575 |
| 14 | 16384 | wide | interleaved / 6 | -36.9245 | -263125 | -37.1909 | -267700 |
| 15 | 16384 | wide | split / 7 | -39.0203 | -277550 | -36.8266 | -330625 |
| 16 | 199680 | zero-heavy | interleaved / 0 | -0.4035 | -7725 | -0.5616 | -10625 |
| 17 | 199680 | zero-heavy | split / 1 | -2.0569 | -45175 | -2.8230 | -59650 |
| 18 | 199680 | dense-low | interleaved / 2 | -20.4220 | -1021325 | -21.6165 | -973975 |
| 19 | 199680 | dense-low | split / 3 | -21.3150 | -996775 | -20.6555 | -1086050 |
| 20 | 199680 | mixed | interleaved / 4 | -15.8020 | -544875 | -16.1015 | -534925 |
| 21 | 199680 | mixed | split / 5 | -16.2872 | -669900 | -15.2588 | -520450 |
| 22 | 199680 | wide | interleaved / 6 | -33.0930 | -4340275 | -36.0551 | -4229675 |
| 23 | 199680 | wide | split / 7 | -37.7529 | -4070875 | -36.3112 | -3621500 |

All primary deltas are negative, ranging from a 0.4035% to a 43.7926% reduction.
Same-path discrepancies remain visible: large wide case 22 R1 has
+203,250 / +7,300 ns duplicate deltas versus primary -4,229,675 ns.
Large mixed split case 21 R1 has -17,500 / +51,350 ns duplicates versus
primary -520,450 ns. No controls or samples are discarded.

The measured gain belongs to the direct-output transaction as a whole.
Removing allocation/growth work, removing the final copy, and changing emission
alignment are not separately timed causes. The smaller zero-heavy gains are
particularly important when considering future sparse photographic inputs.

## Execution, correction and preservation

| Job group | UTC interval | Outcome |
|---|---|---|
| Initial build launcher | 12:15:05.822–12:15:17.348 | exit 1 at export lookup |
| Corrected build completion | 12:17:25.764–12:18:55.339 | exit 0 |
| Normal primitive | 12:19:18.142–12:19:19.682 | passed |
| ASAN primitive | 12:19:19.706–12:19:25.769 | passed |
| Eight host fixtures | 12:19:25.777–12:19:48.720 | all passed |
| Two sealed DLL checks | 12:19:48.738–12:19:51.296 | both passed |
| Fixed CPU timing | 12:20:48.785–12:21:01.306 | passed |

The first launcher lost regex escapes in its symbol selector and stopped after
normal ANS compilation, before linking any DLL or fixture. The new version uses
literal matching, preserves and reuses that exact object, and creates only the
missing outputs. The original launcher and failure journals remain intact.
No C++ correction, object overwrite or timing rerun occurred.

All 15 primary job records reconcile intent, launch/PID, terminal exit and log
hash: 14 accepted, one known launcher failure. The build produced 29 files
totaling 13,271,368 bytes. Every launch required at least 3 GB free on C.
Initial free space was 4,070,211,584 bytes; final space is in final_summary.json.
Nothing was deleted or moved; no recovery volume, elevation, admin/firewall
intervention, security change, profiler or push was used. Power, clocks,
cooling, priority, affinity, driver and production settings are unchanged.

Artifacts are in `build-cuda-ninja/profiles/s189-artifacts`. The S188 frozen
107-file inventory is unchanged, and 343 production-source archives from S182
are reused and rehashed. Source overlays, fixtures, sealed-input references,
all samples, corrected launcher, complete native audits and journals are retained.
After freezing, verify read-only with:

```powershell
python build-cuda-ninja/profiles/s189_verify.py --frozen
```

## Required next qualification

Build new fixed-policy original and direct-packed whole-encoder DLLs against
current production. Audit complete host boundaries and unchanged CUDA modules,
qualify every resident input in normal/ASAN builds plus CUDA memory/init checks,
then run a prospectively fixed broad campaign with duplicate controls.

Compare end-to-end and retained CPU stages against production, not only S183.
Do not combine fixed-capacity ownership or another unqualified change.
S189 has not yet demonstrated a whole-image speedup or justified a production
change. GPU composition/reduction fusion and tile scheduling remain separate
mechanisms outside this stage.
