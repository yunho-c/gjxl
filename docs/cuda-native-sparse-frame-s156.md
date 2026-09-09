# Native sparse frame ownership and resident integration (S156)

## Status

S156 integrates S155's typed sparse handoff into an actual frame owner and
the fully resident CUDA encoder, without reconstructing a dense host array.
It remains an isolated experiment, not an enabled production policy.
Across the six-image corpus, both repetitions favor sparse storage in all
twelve input/width combinations. Padded 4K paired whole-workflow savings are
22–29 ms with int32 storage and 6–21 ms with int8 storage. Compact results
vary substantially, and the weakest compact case is near the noise floor.
Production stays at S152 (`2118ccb`); the investigation started from
`4bc4839`. The backend is not established to be maxed out.

The source overlay is
`build-cuda-ninja/profiles/s156-artifacts/source/`. It was copied from 330
tracked inputs. Its `third_party` junction points to the repository's
existing dependencies, which were treated as read-only and excluded from
recursive snapshots. Main-tree implementation files are unchanged.

## Representation and consumers

The frame's native AC view has six alternatives: dense and sparse forms of
signed 8-, 16-, and 32-bit coefficients. Sparse storage owns one 64-bit
mask and one absolute 32-bit payload offset per 64 logical coefficients,
plus typed nonzero values. Active coefficients follow packed group/channel
order; empty edge capacity is not allocated. Payload intervals may arrive
in arbitrary tile order, but values within a mask retain coefficient order.

`SparseCoefficientSpan` provides indexed rank lookup, forward iteration,
subspans, and mask-based nonzero counts. It deliberately has no `data()`
operation. The old typed int32 group query rejects narrow or sparse owners;
there is no lazy expansion cache or compatibility adapter.

Assembly validates shape, logical coverage, tail bits, interval bounds,
overlap, payload coverage, nonzero values, optional unwritten sentinels,
transform metadata, quantization/DC data, and supplied order populations.
It moves the caller's arrays only after all potentially failing work.
Completed owners are private and immutable through their views. Const group
queries and structural validation allocate nothing; they do not repeat the
full payload scan. Copies deep-copy coefficient arrays. The immutable order
population cache can retain its existing shared ownership.

The original AC tokenizer consumes the native sparse view, with a specialized
mask-count operation excluding LLF coefficients. Uncached coefficient-order
selection reads sparse values directly; the resident population cache avoids
that recount when available. CPU reconstruction, CUDA's exact-frame consumer,
and the Metal coefficient-copy visitor accept the new view. Metal was not
built or run on this Windows machine.

## Resident handoff and lifetimes

Only the qualified 256-thread packer is integrated. Each warp ballots its
nonzeros, the block computes warp prefixes, and each nonempty block reserves
one payload interval atomically. Headers use absolute offsets, so consumers
do not depend on block completion order. The three widths use 30/32/28
registers, 68 shared bytes, and zero local storage or stack/spills.

At the final frame boundary, upstream population and inverse-transform
consumers have finished. The packer reuses the other coefficient allocation:

| Selected width | Typed input | Sparse payload output |
|---|---|---|
| int8 | quantized workspace, byte offset 0 | reconstruction-coefficient workspace |
| int16 | quantized workspace, byte offset N | reconstruction-coefficient workspace |
| int32 | reconstruction-coefficient workspace | quantized workspace |

`N` is the active logical coefficient count. The input remains intact until
the representation decision, permitting a direct dense readback on fallback.
There is no extra full-sized GPU payload allocation. A separate header/count
buffer is allocated on first use and retained by the prepared context; its
bytes are included in staging and peak scratch accounting. At padded 4K,
this adds 4,665,604 device bytes, not another 99.5 MB coefficient workspace.

The first readback includes masks, offsets, nonzero count, and ordinary
frame metadata/populations. After checking the count, the host allocates
exact-sized typed values and reads them back. The regular checked assembler
takes ownership. All-zero payloads retain their sparse type even though
their values array is empty.

Diagnostic-only, thread-local controls are captured during preparation;
they permit same-executable dense/sparse and wide/compact comparisons.
Changing that selector afterward does not change an existing context.
No such selector is proposed as a public API.

The exploratory automatic policy skips counts below `3 * 512 * 512` and
keeps sparse storage only when header-plus-payload bytes are below 80% of
the active dense wire bytes. The forced mode bypasses those two policy
gates, but not launch/range safety checks. These thresholds are experimental,
not a qualified production default. Packing and the first readback still
cost time when the density gate chooses dense storage.

## Correctness and memory qualification

- The original owner-integrated build passed 85 CTests. The final integrated
  build passed the expanded 87-test suite.
- The native frame fixture covers 384 combinations of two group layouts,
  eight transform layouts, eight coefficient patterns, and three widths.
  Each successful run checks 1,536 sparse codestream comparisons, 384
  bitwise CPU reconstructions, and 1,488 rejected assemblies. Both cached
  and uncached order paths and full/sampled order policies are exercised.
- The span fixture checks 405 storage cases, 335,205 subranges, and 3,840
  malformed/range/sentinel rejections, including partial 64-value words,
  arbitrary payload interval order, and signed extrema.
- The allocation fixture checks 12 width/cache/zero combinations, injects
  failure at all 168 observed assembly allocations, and verifies 48
  allocation-free native queries. Failed assembly preserves source array
  identity/content and output identity/codestream bytes. Successful transfer
  is zero-copy; frame copies do not alias coefficient storage.
- The full-workflow fixture performs 312 public encode calls over 52 cases
  and six representation modes. It compares complete codestreams and every
  deterministic summary field. Cases include tiny/edge/multi-group inputs,
  noise and large finite samples, effort 3/7, final score on/off, three-attempt
  target-size control, and maximum-error mode. A public target-size call may
  contain several internal encode attempts; those are not counted separately.
- The prepared fixture uses 32 contexts and 160 evaluations. It alternates
  scored policy output, ordinary evaluation, unscored frame output, bounded
  evaluation, and repeated scored output. Dense/sparse coefficients, bytes,
  fields, quantizers, scores, and diagnostic RGB are compared exactly. The
  header allocation is reused, and its memory accounting is checked.
- The isolated GPU fixture checks 273 width/count/pattern cases and three
  invalid/empty dispatches. It verifies source preservation, header/counter/
  payload guards, unwritten payload tails, owner validation, and every
  logical coefficient, including signed extrema and partial tiles.
- A high-density 513×519 noise fixture performs 18 public encodes at
  Butteraugli targets 0.0001, 0.01, and 0.1. All six automatic-mode calls
  choose dense fallback, matching forced dense/sparse codestreams and
  summaries exactly. This qualifies fallback correctness, not its speed.

All linked project host code in the ASan workflow/prepared builds is
instrumented, including frame assembly and consumers; this is not only an
instrumented outer harness. The frame, span, allocation, workflow, and
prepared tests pass under ASan.

There are twelve completed CUDA sanitizer runs: workflow and prepared
fixtures under full-kernel memcheck/initcheck; the same complete fixtures
with racecheck/synccheck restricted to the new `SparseAc` kernels; and the
isolated pack fixture under all four tools. They report zero errors/hazards,
and memcheck reports zero leaks. The earlier broad workflow racecheck was
deliberately stopped after about five minutes of slow unrelated-kernel
instrumentation. Its partial log is retained and excluded from completed
qualification counts; it is not reported as a passed full race check.

## Native-code comparison

Rebuilding in the source overlay changes CUDA source-derived anonymous
symbol identifiers, so raw cubin equality with S152 does not hold. Only
those two eight-hex identifiers are normalized in the comparison; other
instruction text, encoded words, function names, and resources are retained.

All ten existing modules have identical resource reports. In the ordinary
build, nine have identical instructions; `PrepareQuantNormsKernel` in the
AC-strategy module has different register moves/scheduling. In the ASan
build, all ten existing modules match the baseline instructions after
normalization. The precise cause of this rebuild difference is not asserted.
Both paths in each differential/timing process use exactly the same GPU
modules. Ordinary workflow, prepared, and timing executables share all eleven
modules; the two ASan executables share all eleven of their modules. The
new sparse kernel instructions match between the ordinary and ASan builds.

## Whole-workflow measurement

The measurement boundary includes a fresh profiled encoder workflow and its
prepared-context lifecycle per call, using one retained CUDA backend per
process. Input/oracle loading occurs before timing. The returned codestream
is checked against a frozen prior oracle after timing; returned-result
destruction is outside this boundary. No GPU event/dispatch instrumentation
is inserted into these timing calls.

Each process uses four warmup and eight measured rounds, with four rotating
ABBA labels per round: two dense controls and two sparse controls. All four
labels use one executable. Six existing inputs, wide/compact mode, and two
shuffled/reversed repetitions produce 24 processes and 1,152 checked calls,
of which 768 are measured. The primary statistic is the median, across
rounds, of the sparse-pair mean minus the dense-pair mean; percentages are
computed per round before taking their median. Duplicate-label cross-checks
are retained. Power-limit queries are read-only, outside timed intervals;
they do not establish clock or power constancy.

The following are improvements in the paired whole-workflow statistic;
each range spans the two process repetitions, not a confidence interval.

| Input | Sparse int32 improvement | Sparse compact improvement |
|---|---:|---:|
| flower 500 | 6.30–7.73% | 4.32–5.47% (int16) |
| padded 1080p | 13.93–14.65% | 3.99–8.10% (int8) |
| flower 2000 | 9.48–13.23% | 0.79–5.01% (int8) |
| padded 4K | 8.24–10.11% | 2.28–7.72% (int8) |
| flower 3200×2160 | 9.29–12.14% | 1.23–5.27% (int8) |
| Keong 3839×2159 | 4.57–7.82% | 6.67–7.82% (int16) |

The paired median saves 22.21–28.74 ms for padded 4K int32 and
6.07–21.15 ms for padded 4K int8. Keong int16 saves 19.36–22.03 ms.
All four cross-label median comparisons favor sparse storage in 23 of 24
processes. In the first flower-2000 compact process, one sparse control is
1.00–1.94% slower than the two dense controls while the other is faster;
its primary paired gain is only 0.79%. Keep that instability in the result.

All 1,152 calls match the frozen prior codestream oracles and deterministic
within-process summaries. Large sparse frames retain the expected native
sizes: padded 4K int8 uses 4,944,133 coefficient-storage bytes instead of
26,542,080 fixed-capacity dense host bytes; Keong int16 uses 5,964,348
instead of 53,084,160 bytes. These are host owner sizes, not wire sizes or
total process memory. There is no dense expansion cache.

These whole-encoder observations are not interchangeable with S155's
handoff-only reductions. In particular, variable pipeline/serializer times
mean that the complete measured gain cannot be assigned solely to the
packing kernel or D2H traffic. There is no GPU trace in this campaign and
no claim that clocks or operating power stayed constant. Additional repeated
measurements and density-policy work are required before default promotion.

## Remaining gate and evidence

This step establishes a native representation and integrated resident
candidate. Before promotion: choose a defensible small/dense-input policy,
measure dense/high-density cases and retained-context performance, qualify
sparse-enabled concurrent contexts and failure injection at the new GPU
boundary, and remove diagnostic-only controls. Do not infer an unconditional
default or cross-platform qualification from the sparse corpus.

Evidence is in `build-cuda-ninja/profiles/s156-artifacts/`: original and
corrected source/build snapshots, binaries, native dumps, per-job journals,
sanitizer protocols, ordinary timing protocol/results, and derived analysis.
Failed fixture/build attempts are retained: an incorrect tiny-cache fixture
expectation, a missing compiler environment, missing `<numeric>`, a mutable
image-view call, a missing required quantizer output, and mixed `auto`
declarations. These were fixture/launch issues, not accepted production runs.

The first CUDA memory-check journal rejected a zero-exit run because its
final buffered fixture PASS line was absent. It is retained unchanged and
revalidated using every expected per-case row, zero exit status, and the
sanitizer's zero-error/zero-leak summaries. Subsequent sanitizer validation
uses complete case sequences and the tool's summaries. Strict raw-module
and instruction-equality postprocessing attempts are also preserved; the
final native report explicitly retains the rebuild differences above. The
first final verifier also compared Python integer-keyed label maps directly
to JSON's string-keyed maps; its source is preserved and the comparison now
uses canonical JSON types without changing any timing samples.

No firewall/admin blocker appeared. No clock, power, thermal, affinity,
priority, firewall, or persistent security setting was changed. No external
push was performed. The three protected untracked Markdown files were not
read, edited, or staged.

After freezing the report and source/artifact inventory, verify with:

```powershell
python -X utf8 build-cuda-ninja/profiles/verify_s156.py --frozen
```

`freeze_s156.py` is an exclusive one-time finalizer, not a command to rerun
over the frozen directory. Further implementation should use a new overlay
so this evidence and its source pins remain intact.
