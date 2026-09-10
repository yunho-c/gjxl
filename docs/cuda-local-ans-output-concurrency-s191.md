# S191: local versus adjacent ANS output ownership

Date: 2026-09-10. Parent: `43e28b068d459814ee1d867945a865a7e8dd6bd9`.
This CPU diagnostic investigates the concurrency lead from
[S190](cuda-direct-packed-ans-whole-s190.md). Production is unchanged.
The screen passes: local ownership improves all sixteen large direct-packed
eight-worker cells, with a 37.814% aggregate batch-time reduction versus
0.632% at one worker. This justifies a separate resident experiment, not
promotion or a causal hardware-counter claim.

## Question and isolated mechanism

S189's single-threaded synthetic actual-ANS screen favored direct packed
append, but S190's resident whole-encoder qualification rejected it, with
repeated dense-input CPU-stage regressions. Source review showed that
WriteAcSections concurrently writes into a contiguous vector of BitWriter
objects. Direct append repeatedly changes those destination objects; the old
temporary-writer paths perform most emission on worker-local metadata.

S191 compares adjacent destinations with worker-local output followed by one
move publication. It fixes the ANS implementation and input/model data within
each comparison. Original, growable packed, and direct packed writers are
separate controls, not three changes combined into one candidate. One/eight
workers test whether an ownership effect changes with concurrency.

This can test a concurrency interaction; it does not directly measure
cache-line invalidations or establish false sharing as the cause of S190.
The synthetic inputs also do not reproduce every real image's ANS model.
No production scheduling, coefficient ownership, GPU composition/reduction,
tile policy, BitWriter ABI or compatibility layer changes are made.

## Harness and ownership contract

The exact S188 dataset generator, model builder, independent ordinary-division
ANS/bit-at-a-time oracle, production-reference writer and count-only checks
are reused. The source prefix containing those functions is mechanically
verified unchanged. There are 24 datasets: 1,024/16,384/199,680 tokens,
zero-heavy/dense-low/mixed/full-range-wide distributions, and two layouts.
Models start with eight histograms and optimize to one to three clusters.

Layout pairs share tokens/models but differ in destination prefix width
(`case % 8`), as in S188/S189. The prefixes are retained, not conflated with a
layout-only effect. All metadata, oracle results and independently regenerated
token fingerprints match the frozen S189 dataset records.

Each batch has sixteen identical-data sections and a seventeen-element output
vector; slot zero is an untouched header sentinel. Adjacent mode writes into
slot `1 + index`. Local mode writes into a stack BitWriter and, only on success,
moves its completed ownership into that slot. Prefix setup occurs inside the
worker in both modes. The move does not copy the byte buffer.

Failed local output is not published; its destination remains empty. Failed
adjacent output retains its original prefix. Both modes verify the writer's
prefix rollback before local storage dies, and every successful section must
match the complete independent oracle bytes and logical bit count. This is
a batch diagnostic contract, not a new general-purpose publication API.

BitWriter is 40 bytes with 8-byte alignment in both tested builds. Adjacent
objects can therefore share a 64-byte region. Every timed batch records its
first section's address modulo 64, after timing. This is an address-layout
observation, not a hardware cache-line-size query or contention counter.

The same diagnostic pool apparatus serves both ownership modes. It has one
or eight background workers plus a coordinating thread, start/end barriers,
and a fixed atomic next-section queue. Pool construction, thread creation,
joining and destruction occur outside batch timing. Each batch includes both
barriers, queue dispatch, prefix setup, ANS calls, allocation, emission, local
staging destruction and move publication. Data/model/output-vector construction,
checking, logging and final output destruction are outside timing.

This apparatus is not a proposal to add a production pool. Per-encode pool
experiments S163/S164 remain rejected; their gates are not changed. No timing
claim about removing production thread startup is made here. Source review
checks publication across barrier phases, immutable shared data/models,
exclusive per-index objects, joining, and startup-failure barrier release.
TSAN and thread-creation-failure injection are not claimed.

## Build, native boundary and correctness

No ANS or BitWriter source is recompiled. The normal/ASAN growable DLLs from
S188 and direct DLLs from S189 are reused. New original DLLs link the frozen
S186 production ANS and S187 current BitWriter objects. New normal MSVC 14.37
and clang-cl 22 ASAN harnesses use those same objects for reference and model
construction, with hash-pinned support libraries. Compile/link are separate.

The complete normal COFF/PE audit covers all three actual exported writers
and the harness's original reference. Writer, recurrence and callback code
matches the frozen objects after masking only REL32 fields; real call targets
are resolved. Twenty-nine complete native entries include every present
selected BitWriter method. Unused WithMaxBits is absent from original/growable
DLLs and unused Append from the direct DLL; the reference harness has all five
selected methods. Full normal/ASAN harness object disassemblies are retained.
This is not linked-layout neutrality or ASAN native equivalence.

Each normal/ASAN qualification process covers all 24 datasets, three writers,
one/eight workers and both ownership modes, with sixteen sections per batch.
Additional batches on cases 0/7/16/23 alternate successful and failing sections
for early/late invalid contexts, insufficient output limits, and an enclosing
error or exception after successful emission. All participants complete before
checking; failures do not silently cancel the rest of the batch.

Per build: 528 batches, 8,448 ANS calls, 6,528 exact successful outputs and
1,920 rollback checks. Combined: **16,896 calls, 13,056 exact outputs and
3,840 rollback checks**. Both builds pass. All three harness processes
(normal check, ASAN check, timing setup) also construct and check 24 production
reference outputs and count-only results each. This CPU stage runs no GPU work.

## Prospectively fixed timing protocol

The schedule is fixed before build and correctness checks. There are 144
case/worker/policy combinations, shuffled with seed 19120260910, followed by
their reverse order: 288 cells. Within a cell, labels 0/1 are adjacent and
2/3 local. Four warm and sixteen measured rounds use the Williams orders
0132/1203/2310/3021, reversing the order index on pass one.

There are 23,040 sixteen-section batches and 368,640 exact-checked ANS calls,
including 18,432 measured batches and 294,912 measured calls. All samples and
same-path duplicate controls are retained. No filtering, performance stopping
or campaign rerun is allowed. No build, sanitizer, profiler or heavy artifact
verification overlaps timing. Each child has a 3 GB C-space prelaunch guard
and separate durable intent, PID launch, terminal exit and final report.

The primary delta is the median within-round local duplicate mean minus
adjacent duplicate mean; percentage uses the median within-round ratio.
All four cross-pairs and both same-path duplicate differences are kept.
Geometric ratios weight cells equally. The interaction ratio divides each
large direct eight-worker ownership ratio by its separately measured one-worker
ratio. It is not cross-worker ordinal sample pairing or causal hardware proof.

All prospective criteria must pass before a separate resident ownership
experiment is justified:

| Criterion | Requirement |
|---|---:|
| Large direct/eight-worker ownership wins | at least 12/16 |
| Large direct/eight-worker geometric local/adjacent ratio | at most 0.95 |
| Large direct geometric parallel/serial ownership interaction | at most 0.97 |

These are practical diagnostic criteria, not statistical significance,
universal speedups or automatic production promotion. Original/growable
controls and all smaller inputs remain visible even though they do not define
this prospective mechanism-screen gate.

## Results and decision

**All three prospective criteria pass.** Local output is eligible for a
separately qualified resident experiment; production is not changed or promoted.

| Criterion | Observed | Result |
|---|---:|---|
| Large direct/eight-worker wins | 16/16 | pass |
| Large direct/eight-worker geometric ratio | 0.6218597471 | pass |
| Large direct parallel/serial interaction ratio | 0.6258119952 | pass |

Large direct batches improve **37.814%** with eight workers, versus **0.632%**
with one worker (geometric ratio 0.9936846078). The ownership effect therefore
has a substantial concurrency interaction in this apparatus. The interaction
ratio is not a speedup from one to eight workers; it compares the two ownership
ratios at their respective worker counts.

All 288 cells, 23,040 batches and 368,640 checked timing calls are retained;
294,912 calls are measured. All four journaled children are accepted and the
independent verifier reconstructs the complete result. No sample filtering,
timing rerun or changed gate occurs.

The full implementation/worker/size aggregates follow. Negative favors local
ownership; each row has sixteen dataset/pass cells.

| ANS implementation | Workers | Tokens | Local wins / 16 | Geometric change (%) |
|---|---:|---:|---:|---:|
| original | 1 | 1024 | 7 | +0.563 |
| original | 1 | 16384 | 11 | -0.509 |
| original | 1 | 199680 | 3 | +1.329 |
| original | 8 | 1024 | 16 | -11.527 |
| original | 8 | 16384 | 16 | -16.906 |
| original | 8 | 199680 | 14 | -21.670 |
| growable packed | 1 | 1024 | 8 | +0.089 |
| growable packed | 1 | 16384 | 8 | -0.482 |
| growable packed | 1 | 199680 | 12 | -0.997 |
| growable packed | 8 | 1024 | 16 | -14.246 |
| growable packed | 8 | 16384 | 14 | -21.008 |
| growable packed | 8 | 199680 | 15 | -26.131 |
| direct packed | 1 | 1024 | 11 | -0.396 |
| direct packed | 1 | 16384 | 9 | -0.360 |
| direct packed | 1 | 199680 | 10 | -0.632 |
| direct packed | 8 | 1024 | 14 | -24.076 |
| direct packed | 8 | 16384 | 16 | -33.276 |
| direct packed | 8 | 199680 | 16 | -37.814 |

Direct output wins in 46/48 eight-worker cells. The two losses are tiny
zero-heavy cases 0/1 in pass zero: +0.658%/+0.428%, with primary deltas
+400/+300 ns. Large zero-heavy cells improve only 1.088-5.579%; dense-low
improves 37.002-41.732%, mixed 32.782-36.413%, and wide 57.921-61.846%.
These are synthetic batch results, not image-encoding gains.

Every direct-packed cell is shown below. The full original/growable cells and
their four cross-pairs/duplicate controls remain in analysis.json.

| Case | Tokens | Distribution | Layout / prefix | One worker R0 (%) | One worker R1 (%) | Eight workers R0 (%) | Eight workers R1 (%) |
|---:|---:|---|---|---:|---:|---:|---:|
| 0 | 1024 | zero-heavy | interleaved / 0 | -0.084 | +0.201 | +0.658 | -0.905 |
| 1 | 1024 | zero-heavy | split / 1 | -3.131 | -0.782 | +0.428 | -0.352 |
| 2 | 1024 | dense-low | interleaved / 2 | +1.590 | -1.592 | -21.585 | -20.289 |
| 3 | 1024 | dense-low | split / 3 | -0.517 | +0.239 | -23.424 | -20.295 |
| 4 | 1024 | mixed | interleaved / 4 | -0.363 | -1.435 | -16.055 | -16.817 |
| 5 | 1024 | mixed | split / 5 | -2.633 | +0.591 | -16.789 | -18.956 |
| 6 | 1024 | wide | interleaved / 6 | +8.097 | -0.042 | -48.819 | -42.982 |
| 7 | 1024 | wide | split / 7 | -4.309 | -1.633 | -53.142 | -50.282 |
| 8 | 16384 | zero-heavy | interleaved / 0 | -0.660 | +0.839 | -2.733 | -2.100 |
| 9 | 16384 | zero-heavy | split / 1 | -0.410 | +1.091 | -6.140 | -2.744 |
| 10 | 16384 | dense-low | interleaved / 2 | +3.315 | -2.623 | -34.417 | -31.287 |
| 11 | 16384 | dense-low | split / 3 | -1.214 | +1.319 | -35.622 | -32.058 |
| 12 | 16384 | mixed | interleaved / 4 | -3.759 | +0.933 | -28.697 | -28.498 |
| 13 | 16384 | mixed | split / 5 | -1.176 | -2.323 | -28.184 | -25.062 |
| 14 | 16384 | wide | interleaved / 6 | +1.095 | +2.484 | -56.824 | -56.569 |
| 15 | 16384 | wide | split / 7 | -3.448 | -0.903 | -58.743 | -57.556 |
| 16 | 199680 | zero-heavy | interleaved / 0 | -0.487 | -1.243 | -1.088 | -5.579 |
| 17 | 199680 | zero-heavy | split / 1 | -3.691 | -0.217 | -3.068 | -2.514 |
| 18 | 199680 | dense-low | interleaved / 2 | +0.819 | -1.851 | -40.849 | -40.925 |
| 19 | 199680 | dense-low | split / 3 | +0.208 | +0.916 | -41.732 | -37.002 |
| 20 | 199680 | mixed | interleaved / 4 | -1.347 | -2.292 | -36.413 | -35.453 |
| 21 | 199680 | mixed | split / 5 | +0.439 | -1.233 | -32.782 | -33.030 |
| 22 | 199680 | wide | interleaved / 6 | -1.366 | +2.019 | -57.921 | -61.106 |
| 23 | 199680 | wide | split / 7 | -1.161 | +0.536 | -61.707 | -61.846 |

## Interpretation and next resident experiment

Original and growable controls also benefit at eight workers: their large
geometric changes are -21.670% and -26.131%. This prevents claiming that only
the direct writer suffers a concurrency-sensitive output cost.

Alignment is important here. Seven of eight large datasets have nonzero
destination prefixes. Production's original/growable temporary-writer paths
then perform a bytewise unaligned Append into the shared destination, updating
its counter repeatedly too. The sole aligned large case, zero-heavy case 16,
has original ownership changes of +1.029%/+1.107%; growable is +0.829%/-1.146%.
Distribution and prefix remain coupled in this inherited matrix, so these
observations do not isolate an alignment-only effect. Actual AC destinations
in WriteAcSections are initially empty; the generic synthetic-control gains
must not be extrapolated to that production boundary.

The local/adjacent comparison holds each case's prefix, ANS implementation,
tokens and models fixed, and the one/eight-worker results support testing
output ownership further. But sixteen sections reuse one dataset/model, so
cache reuse and task balance differ from distinct resident image groups.
No coherence counters, actual-token capture, allocation-fault injection,
TSAN/MSAN or whole-encoder timing is supplied by this stage. Pool synchronization,
dispatch, prefix setup, allocator behavior and move publication are included;
they are not separately attributed causes.

Duplicate controls are retained even for strong improvements. Large direct
case 23/eight/pass 0 has adjacent/local duplicate deltas
+10,923,050/-490,500 ns versus primary -38,967,300 ns. Original case 23 at the
same worker/pass setting has +12,575,850/-126,450 ns duplicates versus
-25,535,525 ns. Direct case 16/eight/pass 0 has -92,700/+36,650 ns duplicates
versus only -66,900 ns primary. None is subtracted or discarded.

The bounded follow-up is worker-local AC section emission followed by one
successful move into the final ordered vector, using the exact S189 direct
ANS objects. Include unchanged adjacent-direct and current-production controls.
Preserve failed-section publication, section ordering, immutable model lifetime,
joining, allocation/status behavior and timer coverage. Check the exact source
overlay and linked code, all resident bytes/summaries/storage in normal/ASAN
builds, and CUDA memory checks before prospectively fixed whole timing.

No BitWriter padding, layout/ABI change, new pool or image-specific selection
is justified by S191. S190 remains a failed campaign and will not be rerun or
reclassified as passing because this separate CPU mechanism screen succeeds.

## Execution, corrections and preservation

| Job | UTC interval on 2026-09-10 | Outcome |
|---|---|---|
| Original DLL/harness build | 13:13:16.318-13:13:39.351 | exit 0 |
| Normal concurrent checks | 13:17:14.825-13:17:28.882 | passed |
| ASAN concurrent checks | 13:17:31.432-13:17:53.527 | passed |
| Fixed timing campaign | 13:19:34.540-13:27:14.867 | passed |

The build produces thirteen files totaling 3,021,618 bytes. Its child exits
zero with the PASS marker, but the preparation wrapper then asserts because
it expected fourteen outputs. Clang did not emit a separate ASAN .exp file.
A new completion helper verifies the exact existing thirteen-file set and
durable terminal journal and writes only the missing build summary. It does
not rebuild, replace a binary or change C++ source.

The first native auditor also stops before writing outputs: it assumes all
five selected BitWriter functions survive linking in every DLL. A new version
records the exact three unused-function omissions described above and audits
all twenty-nine present complete entries. Both original helpers remain intact.
These are diagnostic bookkeeping corrections, not encoder failures or timing
reruns. All four actual job records remain accepted with exit zero.

Artifacts are under `build-cuda-ninja/profiles/s191-artifacts`, with versioned
`s191_*` helpers beside them. The verifier reconstructs all sample/cell orders,
data fingerprints, oracle metadata, per-cell comparisons, aggregate criteria,
native code checks and process journals. Intent, PID launch, terminal exit,
command, log hash and nonoverlapping intervals reconcile for every child.

S190's 1,468, S189's 190 and S188's 107 frozen artifact files remain unchanged.
The 343 production-source archives are reused from S182, not recopied. Source,
object, DLL, support and SDK inputs are hash-verified. The original and corrected
helpers, all raw timing samples, modulo-64 layouts, cross-pairs, duplicate
controls, source archives and final inventory are retained.

C had 3,921,657,856 bytes free at the postmeasurement check; U had 30,883,840.
The frozen summary records final space and artifact/source counts. Nothing is
deleted, moved or overwritten, and the recovery volume is not used. No machine
power, clock, cooling, affinity, priority, driver, firewall or security settings
are changed. No elevation/admin prompt, profiler, GPU run or push is involved.
Protected untracked notes are not read, edited or staged.

After freezing, verify read-only with:

```powershell
python build-cuda-ninja/profiles/s191_verify.py --frozen
```

This stage does not claim full CTest, cross-platform/installed-consumer
qualification, GPU sanitizer results, measured memory reduction, real-image
speedups or completion of the broad optimization goal. Production remains
unchanged; the resident-path objective stays active.
