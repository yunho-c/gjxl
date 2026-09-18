# Compact consumption, composition fusion and tile scheduling (S106)

Starting revision: `b6d490d`, following the
[S104–S105 disposition and counter survey](cuda-candidate-qualification.md).
Tests ran 2026-09-07 evening local time / 2026-09-08 UTC on the same RTX 3060
Laptop and CUDA 11.8 toolchain. Production remains S79. This investigation
builds three separate prototypes; it does not reopen the rejected S84/S95
default changes or claim that optimization is exhausted.

## Compact coefficient consumption

The actual token-template and direct-tokenization implementations now have a
diagnostic typed-input version. A group supplies three signed int8, int16 or
int32 spans. Type selection occurs outside the coefficient loop; conversion
to int32 happens on load, with no dense expansion array. The contiguous
nonzero count remains vectorizable, and LLF subtraction, channel order,
custom scan order, nonzero prediction, contexts, signed values and fixed
symbol populations retain their original operations.

Release and scoped-ASAN each pass 126 cases and 756 checks: all seven
transform shapes, natural/custom orders, zero/sparse/dense inputs, signed
8/16-bit limits and wide int32 values. Both token-template and direct-token
outputs match int32 results, including contexts and symbol populations;
short spans leave token-template output unchanged. The ordinary AC-group
regression suite also passes with the generalized int32 implementation,
including its pinned tokens, signed extremes, malformed groups and frame
traversal. These are consumer-level tests, not a newly qualified compact frame.

The remaining obstacle is ownership/API integration, not the scalar value
conversion. `VarDctEncoderFrame` owns fixed-capacity int32 rows;
`GetAcGroup`, frame assembly/validation, coefficient-order calculation and
reconstruction still expose or assume int32 storage. The prototype does not
change that frame or feed compact CUDA readback through a full encode.
Accordingly, no end-to-end compact speedup or measured host-memory saving is
claimed. Mixed-strategy/edge compact frames, transactional transfer of typed
ownership, copied frames and concurrent readers still need qualification.

The next compact candidate should move a narrow owner directly from CUDA
readback into the frame and dispatch typed consumers, retaining int32 fallback.
It must avoid both S84's second owner and its dense expansion. Keeping narrow
fixed-capacity rows initially would preserve existing group offsets and tail
semantics; active-only offsets are a separate optimization. The legacy dense
span getter needs an explicit compatibility design, not a reinterpret cast
or an implicit expansion in every serializer access. Do not claim saved memory
until complete single/batch encodes demonstrate the new owner's actual lifetime.

## Fused composition and maximum reduction

A CUDA prototype combines `ComposeKernel` with the first 256-wide maximum
reduction pass. It preserves the materialized distance map for the existing
AQ per-transform reduction. The subsequent maximum passes are unchanged.
Composition arithmetic and the maximum reduction tree are retained, including
nonfinite/negative-value handling. It uses 18 registers and 1,024 shared bytes,
versus 16 registers for separate composition and 10 registers/1,024 shared
bytes for separate reduction; the compiler reports no spills.

Qualification covers 192 reference/fused map-and-score pairs over twelve
geometries from 1x1 through odd padded 4K, two row-padding choices, in-place
and separate output, finite inputs, NaNs, infinities and negative results.
Maps are compared bitwise, finite scores bitwise, invalid scores as NaNs,
and row guards remain intact. Memcheck, racecheck, initcheck and synccheck
all pass the same cases; memcheck reports zero leaked bytes. This qualifies
the standalone composition/reduction pair, not integration into the complete
prepared Butteraugli or encoder lifecycle.

Each timing row contains sixteen repeated stage chains. Two warm rounds
precede twelve measured alternating-order rounds per geometry/padding.
The table contains median CUDA-event milliseconds per chain and the median
within-round percentage change. Percentages are paired statistics, so they
need not equal ratios of the two independent medians.

| Input | Row padding | Separate ms | Fused ms | Paired change |
| --- | ---: | ---: | ---: | ---: |
| 500x500 | 0 | 0.0396 | 0.0262 | -33.8% |
| 500x500 | 7 | 0.0386 | 0.0279 | -28.4% |
| 1919x1079 | 0 | 0.1910 | 0.1526 | -30.1% |
| 1919x1079 | 7 | 0.2442 | 0.1768 | -27.7% |
| 3839x2159 | 0 | 1.1828 | 0.8356 | -28.8% |
| 3839x2159 | 7 | 1.2959 | 0.9316 | -28.6% |

At 4K this removes one logical 33,153,604-byte map read and one launch per
comparison, while retaining the map write and AQ reread. These bytes come
from source access counts, not new hardware traffic counters. The roughly
0.35 ms stage reduction is not an encoder speedup or a guaranteed saving
when surrounded by the real perceptual pipeline. Repetition/cache state,
host launch supply and natural clock variation differ from a full encode.

A broader encoder-only sink could compose values directly inside the existing
per-transform L16 accumulation and emit score partials. That could also avoid
the composed-map write and AQ reread, but must preserve each thread's pixel
assignment, four squarings, accumulation/reduction order, odd-edge coverage
and error bits. Generic distance-map consumers must retain their output.
Score partial capacity, signed-zero/invalid semantics, scratch aliasing and
diagnostic map lifetime need explicit tests before this larger fusion.

## CPU strategy-tile scheduling

The diagnostic scheduler operates only on already-computed candidate costs.
Independent color tiles contain at most 8x8 base blocks (64x64 pixels).
Each tile keeps its original internal merge order and writes disjoint byte
cells. Export remains serial and commits only after every tile succeeds.
Statuses are checked in tile order so scheduling does not select the reported
failure. Partial worker-creation failure joins started workers and leaves
caller output unchanged; it does not rerun partially processed tiles.

Two families use up to four CPU participants including the caller: contiguous
tile partitions, and dynamic eight-tile chunks. Both obey explicit CPU thread
budgets and serialize within the existing explicit nested-parallel contract.
Worker creation/join is inside measured time; no persistent pool or system
scheduling change is introduced. Duplicate labels share each implementation.

Release and ASAN each pass twelve geometry cases, 480 exact-grid comparisons
and 156 atomic-failure checks. Instrumented cell accesses detect no cross-tile
read/write, including odd tile edges. Tests cover budgets 1/2/4/automatic,
explicit nested scopes, invalid costs and injected worker-creation failures.
Fourteen full-encode preflight jobs pass frozen bytes and fresh summaries,
including 4K budgets one/two. ASAN is scoped to the new host sources, not all
retained libraries. Independent encoder contexts and concurrent batches are
not yet qualified for this scheduler.

Main timing uses twelve six-label Williams rounds after six warm rounds,
with a fresh reference per process. Five inputs run twice in reversed case
order; 4K budgets one/two have one additional run each. Default fully resident
distance 1.2/effort 7, automatic CPU policy except those explicit controls,
and no final score request are unchanged. Timings include complete Encode
helper calls and internal owner cleanup, but exclude backend lifetime,
file I/O, returned-result destruction and comparisons. No NVML sampling or
profiler capture runs during this campaign. Natural machine-state drift and
light concurrent editing remain limits.

Below are ranges across candidate/control duplicate-label paired medians in
both repetitions. The descriptive gate requires every value negative; it is
not a significance test. The merge deltas are separate paired measurements,
not values subtracted from outer timing to manufacture a gain.

| Input, automatic CPU policy | Static whole-call change | Dynamic whole-call change | Static merge reduction ms | Dynamic merge reduction ms |
| --- | ---: | ---: | ---: | ---: |
| Flower 500 | -0.53% to +3.36% | -1.17% to +0.15% | **slower** 0.10–0.13 | **slower** 0.01–0.05 |
| Keong 500 | -3.10% to +0.40% | -2.42% to +5.67% | **slower** 0.07–0.10 | -0.10 to +0.02 |
| Keong 2000 derivative | -2.10% to +2.53% | -5.94% to +1.55% | 2.70–3.25 | 2.86–3.67 |
| 1919x1079 | **-4.12% to -1.00%** | -3.97% to +0.45% | 0.78–1.02 | 0.84–1.02 |
| 3839x2159 | -4.82% to +3.74% | -2.14% to +1.87% | 4.61–6.41 | 5.11–6.51 |

Only static 1080p passes the complete-call gate across both repetitions.
Large-image merge savings are real in the scoped measurements, but do not
establish a general full-encode improvement. Small images can lose merge time
to worker overhead. No geometry cutoff is promoted from these cases.

The one-thread 4K control is particularly important: static labels appear
2.36–3.79% faster in whole-call paired medians even though no parallel work
occurs and merge deltas span -0.16 to +0.89 ms. This is not a parallel speedup;
it warns against attributing all whole-call differences to saved merge time.
Budget-two static/dynamic whole-call comparisons are mixed despite merge
reductions of 3.98–4.50 / 2.87–3.33 ms. Those budget checks have one replication,
not the main cohort's two.

## Disposition and evidence

Continue with compact frame ownership and an integrated composition/reduction
control. Keep the tile scheduler as a candidate: its scoped savings justify
further full-call/concurrent qualification, not an unconditional default.
Do not add the three mechanisms' apparent savings together.

The 26 encoder jobs contain 1,406 calls, 1,380 exact byte/summary comparisons
and 26 frozen reference checks. Main timing is 02:22:01.246285–02:25:40.094676
UTC, 218.85 seconds. All GPU jobs are serial and terminal. Racecheck takes
about two minutes with active processes; no permission prompt/block is
observed and no power, clock, priority, security or driver setting changes.

Two wrapper-marker mistakes are preserved: the prior validator printed
`S104-S105 VERIFIED`, not the requested slash spelling; racecheck printed
`RACECHECK SUMMARY`, not memcheck's `ERROR SUMMARY`. Both processes returned
zero. Existing logs/hashes and their actual success summaries were verified;
neither successful underlying check was repeated to fix report wording.

Scripts and diagnostic sources are ignored `build-cuda-ninja/profiles/s106_*`;
new binaries, build/test logs and measurements are in
`U:/gjxl-cuda-diagnostics/s106`. A recomputing validator and manifest anchor
sources, outputs, references and unchanged retained runtime files. Mutable
temporary files are retained but excluded from freezing. Only documentation
is committed. Production code, forty retained runtime files and the user's
three untracked Markdown files are unchanged.

Recheck the local evidence with:

```powershell
python build-cuda-ninja/profiles/s106_validate.py --frozen
```
