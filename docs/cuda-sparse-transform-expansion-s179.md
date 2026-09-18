# S179: selective transform-local sparse expansion

The candidate is correct but rejected at its precommitted performance screen:
coefficient-token worker work improves in five of six cells, and tokenization
wall time in three of six, short of the required six and four. Keong shows a
useful local worker reduction, but whole-encode changes remain mixed. No
production change is promoted, and the optimization goal remains active.

## Question and scope

S178 demonstrated that scheduling changes can alter later device elapsed
rates even with unchanged kernel code. This experiment instead targets a
host coefficient-consumption operation, with unchanged device modules and
without the S175 selector, S177 pauses, or S178 recorder. The parent is
`cee46dbc3c415ce7a910e18d30e169b792702171` on `feat/cuda`; production remains
the qualified S168 implementation.

The direct sparse tokenizer reads each ordered coefficient through a mask,
payload offset, and rank calculation. A candidate expands only transforms
with nonzero AC into one bounded, native-width local array, then reads that
array using the original coefficient order. It tests whether avoiding those
repeated point lookups outweighs extra clearing, enumeration, and stores.
It is not another reservation-factor experiment.

GPU tokenization was considered first. Current coefficient-order and block-
context choices are made in `encoder.cpp` after frame handoff. Naively moving
the coefficient walk back to the GPU would therefore require another handoff
or a broader resident-frame/serializer integration. This stage makes no GPU
tokenization implementation or speedup claim.

Evidence is under `build-cuda-ninja/profiles/s179-artifacts`, with versioned
`s179_*` helpers beside it. All 343 production files are unchanged; no public
ABI, compatibility layer, sparse representation, or default is changed.

## Candidate and controls

The original direct-token template remains exact. Its clone retains the
original anchor validation, LLF-excluding count, token reservation, predictor
maps, coefficient order, signed packing, contexts, population collection,
failure-atomic output publication, and allocation handling. Only the source
of coefficient reads during token emission changes.

After a nonempty transform's original count token, the accessor clears its
active local array and enumerates masked payload entries into their physical
coefficient positions; emission still uses the original scan order. It
respects arbitrary per-word payload offsets, partial first/last
words, and nonzero LLF entries. Zero-AC transforms skip expansion, including
when their LLF rectangle is nonzero. The array has 1,024 entries: at most
4,096 logical bytes for int32, not a measured compiler stack-frame size.
There is no whole-frame dense reconstruction or retained expansion cache.

Modes 0/1 call the original template; modes 2/3 call the candidate for sparse
groups. All dense whole routes still call the original. A common diagnostic
selector is configured only between joined encodes. Per-group audit slots
record capacities and expansion counts; their overhead is inside Encode.
Concurrent reconfiguration of this diagnostic selector is unsupported.

Both whole DLLs contain eleven CUDA modules identical to the S169/S168
standard build. Source auditing reconstructs the full original translation
unit and verifies the bounded clone/dispatch edits. It does not claim that
separately instantiated original/candidate host functions have identical
machine code.

## Qualification and fixed screen

Normal and host-ASAN fixtures each pass 122,880 checks across 128 frames /
320 groups, plus 1,296 focused sparse-range cases. The combined totals are
245,760 checks and 2,592 range cases. Coverage includes all seven supported
strategies and mixed layouts, int8/int16/int32 extrema, eight coefficient
patterns, natural/custom orders, three context maps, populations on/off,
unaligned spans, reversed payload-word order, nonzero LLF and suffix values,
and partial-word ranges through 1,024 values. Every token, context, and
population field matches original dense and sparse oracles. Capacities and
independently counted expansion work also match. Short spans and null
scratch/output/audit are rejected without changing the sentinel output.

Whole preflight passes all eleven inputs, automatic/eight requested CPU
threads, and normal/ASAN builds, then CUDA memcheck on 4K and initcheck on
Keong. These 46 processes exercise 230 whole encodes. The 22 ASAN processes
include 88 instrumented Encode calls and 22 normal reference calls.
CUDA reports zero errors, and memcheck reports zero leaked bytes/allocations.
The instrumented whole-GPU runs are slow but show advancing Encode records;
their durations are not treated as performance evidence.

Each whole process first obtains a reference from frozen normal S168; image
references also match the frozen codestream. Every subsequent call checks
bytes, complete summary, coefficient width, native owner, token count,
original vector capacities, and expansion bounds. The frozen S172 group
census independently supplies anchor/token/capacity totals; native payload
counts are checked against S169's unchanged coefficient census.

The precommitted screen uses 1080p, 4K, and Keong with eight threads, two
reversed process-order passes, eight warm-up rounds and sixteen measured
rounds. Each round follows one of four Williams orders (`0132`, `1203`,
`2310`, `3021`), balancing positions and directed within-round predecessors.
All samples remain. The primary comparison is the median within-round
difference of the candidate duplicate mean and original duplicate mean;
individual pairs and both duplicate disagreements are retained.

Setup, reference encoding, prior-result clearing, validation, and read-only
NVML endpoints are outside the timed complete Encode call. Expansion and its
audit are inside. Worker work is nested and must not be added to wall time.
No build, sanitizer, profiler, compression, or heavy hashing runs during the
screen; light report editing and ordinary shared-machine activity remain
limitations. No clock, power, cooling, priority, affinity, driver, firewall,
or security setting changes, and no permission prompt blocks the stage.

The screen cannot authorize promotion. Its precommitted broader-test gate is
lower coefficient-token worker work in all six cells and lower AC-tokenize
wall time in at least four. Otherwise this general expansion candidate is
rejected; a local improvement alone is not a whole-encoder gain.

## Results and decision

The six screen processes complete 582 whole encodes, including 384 measured
calls. All 1,152 warm/measured NVML endpoints report the unchanged 40,000 mW
power limit. The following deltas are medians of within-round differences,
not differences of independent medians. Negative is faster; worker work is
an aggregate across workers, not a wall-clock saving.

| Case / pass | Whole ms (%) | AC-tokenize wall ms | Coefficient-token worker ms (%) | Earlier quantization ms |
|---|---:|---:|---:|---:|
| 1080p / 0 | +1.604 (+2.66%) | +0.115 | -0.034 (-0.43%) | +0.104 |
| 4K / 0 | -8.719 (-3.22%) | -0.501 | -1.039 (-4.17%) | -7.652 |
| Keong / 0 | -6.620 (-2.08%) | +0.270 | -4.925 (-7.46%) | -6.451 |
| Keong / 1 | +4.011 (+1.27%) | -1.192 | -8.320 (-11.27%) | +9.141 |
| 4K / 1 | -7.833 (-2.80%) | +0.368 | +0.151 (+0.59%) | -3.973 |
| 1080p / 1 | -1.532 (-2.23%) | -0.184 | -0.461 (-5.11%) | -1.091 |

Keong's worker reduction repeats, but its tokenization wall and whole-encode
results change sign. Both 4K whole comparisons are faster, yet substantial
changes occur in quantization, before the modified consumer; one 4K worker
result is slightly slower. The data do not isolate carryover or shared-machine
variation, so those whole gains cannot be assigned directly to expansion.
The 1080p whole result also changes sign.

Same-path duplicates remain important. Their paired whole deltas (original
1 minus 0; candidate 3 minus 2) are retained below, in milliseconds. Full
individual-pair and per-phase results remain in `analysis.json`.

| Case / pass | Original duplicate | Candidate duplicate |
|---|---:|---:|
| 1080p / 0 | -1.328 | +0.040 |
| 4K / 0 | -1.455 | +0.921 |
| Keong / 0 | -1.028 | -3.567 |
| Keong / 1 | -5.386 | -0.294 |
| 4K / 1 | +0.483 | +0.007 |
| 1080p / 1 | +0.756 | -1.284 |

The failed gate ends this candidate's screen: no broader timing campaign is
run, no subset is promoted, and no rerun or sample filtering is used to rescue
the result. These observations do not establish that compact consumption is
maxed out.

## Expansion cost census

Counts are identical across candidate labels, warm/measured calls, and both
passes. A transform/channel is expanded only if it has nonzero AC. Copied
payload includes its LLF values; coefficient tokens below exclude the three
per-anchor nonzero-count tokens.

| Case | Expanded transforms/channels | Cleared logical values | Copied payload values | Emitted coefficient tokens | Cleared values / coefficient token |
|---|---:|---:|---:|---:|---:|
| 1080p | 2,184 | 2,073,600 | 70,907 | 196,665 | 10.54 |
| 4K | 8,160 | 8,294,400 | 278,533 | 568,099 | 14.60 |
| Keong | 53,200 | 9,980,160 | 649,374 | 1,420,455 | 7.03 |

For the int32 whole inputs, clearing accounts for 8,294,400 / 33,177,600 /
39,920,640 logical store bytes, respectively, plus payload overwrites of
283,628 / 1,114,132 / 2,597,496 bytes. These are operation counts, not measured
DRAM traffic, cache misses, or RSS. The local buffer is reused; these totals
are not simultaneously allocated memory.

Original value and context vector capacities remain exactly 6,227,352 /
24,907,680 / 24,978,183 elements each for the three cases. Native Sparse32
ownership is unchanged, including 5,779,732 bytes for 4K and 7,263,096 bytes
for Keong. This experiment neither reduces reservation nor changes the
device-to-host representation.

Expansion clears roughly seven to fifteen values for every emitted
coefficient token in these cases. A future candidate should therefore be
defined from transform-size and expansion-to-token costs, qualified against
the original consumer, and screened independently. This is a hypothesis for
new work, not a post-hoc claim that a subset of this candidate is proven fast.
GPU tokenization remains a separate order/context and resident-handoff
integration question. Blanket reservation/expansion sweeps are not the next
step suggested by this evidence.

## Reproducibility and preservation

All times below are UTC on 2026-09-10:

| Activity | Start | Finish |
|---|---|---|
| Normal and ASAN build | 07:47:16.197 | 07:48:17.781 |
| Normal fixture | 07:48:29.423 | 07:49:08.622 |
| ASAN fixture | 07:49:08.624 | 07:50:28.220 |
| Normal/ASAN whole preflight | 07:52:56.141 | 07:53:55.826 |
| CUDA memcheck | 07:53:55.830 | 07:56:09.443 |
| CUDA initcheck | 07:56:09.457 | 07:57:48.552 |
| Fixed performance screen | 07:58:00.408 | 08:00:10.146 |

All 55 journaled subprocesses are accepted with exit code zero and do not
overlap. There are no failed builds or qualification jobs in this stage.
Combined whole coverage is 812 encodes: 230 preflight and 582 screen calls.
ASAN and CUDA-sanitizer runs are correctness evidence only. Complete commands,
start/finish records, exit status, stdout/stderr, binary/module hashes, source
snapshots, protocols, counters, and unfiltered results are retained.

`s179_verify.py` rehashes the immutable S178 998-file inventory, nineteen
inherited support inputs and SDK pins, eleven CUDA modules per build, all
input/oracle pins, and source/protocol records. It reconstructs the original
source, reparses every whole log, checks the exact schedules and journals,
and independently re-derives the analysis and gate. `s179_freeze.py` archives
the final report and helpers and emits `final_sources.json`,
`final_summary.json`, and `artifact_hashes.json`; frozen verification checks
the exact artifact set and every recorded hash.

From the repository root, the retained verification command is:

```powershell
python build-cuda-ninja/profiles/s179_verify.py --frozen
```

The frozen helpers are evidence, not mutable production infrastructure: do
not rebuild or edit this artifact set in place. A new experiment requires
new versioned files. No predecessor/helper is overwritten, no material file
is deleted, no GPU module or production source is changed, and no machine
setting is adjusted. No firewall or administrator prompt blocks this stage.
Only this report is committed. The full optimization objective remains open.
