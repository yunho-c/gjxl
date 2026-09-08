# Bounded candidate-cost tile scheduling (S112)

Date: 2026-09-08. Starting revision: `64ee875`, branch `feat/cuda`.
Windows 11, RTX 3060 Laptop (sm86), CUDA 11.8, MSVC 14.37, Release.

## Scope

This follows S106's CPU strategy-tile prototype after S111's resident
composition/AQ fusion. Candidate costs are already computed and resident in
host spans; only the hierarchical merge across independent 8x8-base-block
color tiles is scheduled here. Per-tile math, merge order, priority checks
and tie-breaking are unchanged. CPU cost computation remains serial.
The common prepared-cost entry also serves other GPU backends; Metal
execution and performance are not newly tested on this Windows machine.
Compact coefficient storage remains OFF in these experiments, and S111's
GPU fusion remains enabled in both timing routes.

**Disposition: not promoted.** The clean candidate reduces the odd-4K CPU
merge by 4.35–4.67 ms, but the enclosing quantization phase is slower in all
eight automatic-thread paired medians, and complete-call/batch results are
mixed. Runtime source and CMake routing are restored to `64ee875`. The tested
candidate, its test and build integration are archived with the evidence;
no unused helper, runtime selector or compatibility layer is committed.

## Candidate design and contracts

The scheduler uses at most four participants including the calling thread,
bounded by hardware concurrency, the explicit per-encode CPU budget, tile
count, and available base-block work. The participant-count limit uses one
participant per 4,096 base blocks of work: 64 full color tiles. This bounds
thread creation; dynamic claiming does not guarantee equal work per
participant. Thin/partial tiles contribute their actual block count.
Consequently fewer than 8,192 base blocks run
serially. This is a work-grain policy to amortize thread creation, not an
image-content rule. It is a tested candidate, not a claim of a universally
optimal threshold.

Participants claim eight consecutive tiles at a time with a relaxed atomic
counter. There is no persistent pool, global admission controller or new
runtime option. Budgets are per encode, not a total cap across independent
encodes. Existing explicit nested parallel scopes remain serial, and all
temporary budget/tracker state is restored when the work completes.

Four fixed outcome records and three fixed thread slots replace the earlier
per-tile status vector and thread vector. Each participant records its first
failure. Since claimed chunks and indices within chunks increase, checking
the minimum failed tile after joining preserves serial error precedence.
A participant can stop after its first error without skipping an earlier
unclaimed tile. All started workers are joined, including after partial
thread-start failures. There is no retry of partially modified tile state.
Resource failures return an error; caller output is unchanged.

Worker exceptions record a code without constructing an allocating error
string on the worker. The integrated candidate additionally distinguishes
length errors and unexpected thread-factory failures. The default factory creates ordinary
threads; a template factory parameter permits deterministic failure injection
in the unit test and is not a production selector.

Tiles read immutable cost spans and write disjoint byte cells. The entire
search grid is private until every tile succeeds. Export remains serial and
commits the caller's grid only after success. The search-grid allocation and
exported-grid allocation remain; there is no memory-capacity saving claim.

## Qualification

The guarded diagnostic compares serial, contiguous four-way, dynamic
eight-tile, and work-grain-limited dynamic schedules. Release and scoped ASAN
each pass 672 exact-grid comparisons over 12 geometries, 192 unchanged-output
failure checks, 192 independent concurrent comparisons, and 63 scheduler
contract checks. Cell-access guards detect no reads/writes outside a tile.
Explicit budgets include one/two/four/automatic and nested scopes.

The integrated candidate's regression test expands this coverage (the test
is archived, not retained in the final runtime checkout):

- 840 exact-grid comparisons across 21 geometries and four cost patterns,
  including thin images, odd edges, and two/three/four-participant grain
  boundaries. Uniform, random, zero and exponent-range costs are included.
- 588 unchanged-output checks for nonfinite/negative costs and incomplete
  tables.
- 32 concurrent independent comparisons with mixed one/two/four/automatic
  budgets.
- 54 contract checks covering exactly-once work, participant peaks with a
  barrier, earliest-error precedence, empty/serial work, worker exceptions,
  and every partial-start position for resource/allocation/unexpected factory
  failures. Started workers must finish and leave no active participant.

Both release and scoped ASAN pass all of these checks. ASAN instruments the
new scheduler, actual search implementation and test, but not all linked
libraries. The integrated Release build passes **79/79 CTest tests** in
**176.49 seconds**, including existing pinned strategy/search tests and the
install-consumer check. This count qualifies the candidate build, not a new
test added to the restored S111 runtime.

The prototype and the actual production-linked scheduler each repeat S108's
40 successful quality/rate/stress/batch jobs: 480 checked encodes per route,
with frozen codestream bytes and complete summary records. Each also rejects
the same seven extreme inputs with the same errors. Five additional
prototype-scoped ASAN jobs cover independent/public-driver content and quality
batches plus rate search, adding 60 checked encodes. Those small-image batches
remain serial under the work-grain policy, so four additional actual-candidate
ASAN jobs exercise HD/4K two-image independent and public-driver batches,
adding 24 checked encodes with parallel tile work. Per-encode participant
limits are asserted on profiled independent requests; public-driver results
are byte/summary checked, and its per-request budget propagation is not
exposed as a public timing field.

Across all preflights, qualification and timing campaigns there are **6,684
checked encodes**, including **3,968 measured encodes**, plus **14 matching
expected rejections** counted separately. There are 160 serial GPU jobs
(including CTest); concurrency occurs only inside the intentional batch
requests. The four standalone host-test jobs are additional to these counts.

## Exploratory schedule timing

Twelve single-image jobs compare eight labels: 0/1 duplicate serial;
2/3 contiguous partition; 4/5 dynamic chunks; 6/7 grain-limited dynamic.
Five inputs run twice in reversed order, with additional one/two-thread 4K
controls. Each job checks one reference and eight warm plus 16 measured
Williams-design rounds of eight labels: 2,316 checked encodes and 1,536
measured encodes. Complete calls retain returned results until after timing.
Input I/O, backend lifetime and comparisons are outside the outer boundary.

Per-cell boundary checking is compiled out for timing. The prototype still
maintains per-tile TLS coordinates, so its small-image overhead and absolute
merge deltas are not clean production measurements. This limitation was
identified before the final production-source comparison below; the data
is retained, not silently replaced.

For the grain-limited candidate, automatic-thread paired merge medians save
5.75–7.26 ms on odd 4K, 3.09–4.97 ms on Keong 2000, and 0.76–2.27 ms on HD.
Small-image merge deltas range around zero, with several regressions.
Whole-encode paired medians are mixed on every input; the second 4K
repetition has all four candidate paired medians slower.

Eight two-image batch jobs add 912 checked / 640 measured encodes,
using duplicate serial and grain-limited labels with two CPU threads per
request. Every batch checks two merge entries and exact bytes/summaries.
All candidate paired merge-sum medians are favorable, but whole-call medians
are predominantly slower. Merge-sum time can overlap between requests and
must not be subtracted from batch wall time to manufacture a speedup.
This is important counterevidence for unconditional scheduling promotion.

## Final production-source measurements

The final controlled source is a copy of the production search implementation
with only an entry-level mode condition and merge timer/counter. It uses the
actual production scheduling header and has no per-cell checks or per-tile
TLS bookkeeping. The final single-image probe reports quantization and
serialization phases as well as merge and complete-call time. Duplicate
labels and one-thread controls remain in the design.

The clean single-image comparison uses four labels: 0/1 duplicate serial,
2/3 duplicate candidate. Twelve jobs each check one reference plus four warm
and 24 measured balanced rounds: **1,356 checked / 1,152 measured encodes**.
Every label occupies every position six times in measured rounds. Five
inputs run twice in reversed order, followed by one/two-thread 4K controls.

The table gives the range of the four candidate-versus-serial within-round
paired medians for each repetition. Negative means faster; these ranges are
not confidence intervals. Complete-call percentages are computed per pair
before taking medians, not from the ratio of separately aggregated times.

| Input / CPU budget | Repetition | Merge delta (ms) | Complete-call delta (%) |
| --- | ---: | ---: | ---: |
| Flower 500 / auto | 0 | −0.020 to +0.004 | −1.51 to +0.94 |
| Flower 500 / auto | 1 | −0.004 to +0.003 | −0.11 to +1.62 |
| Keong 500 / auto | 0 | +0.001 to +0.011 | −1.74 to −0.09 |
| Keong 500 / auto | 1 | +0.009 to +0.017 | −0.70 to +0.84 |
| Keong 2000 / auto | 0 | −2.48 to −2.25 | −3.66 to +0.56 |
| Keong 2000 / auto | 1 | −3.18 to −3.03 | +0.08 to +1.33 |
| HD / auto | 0 | −0.86 to −0.75 | −2.36 to +0.96 |
| HD / auto | 1 | −0.89 to −0.74 | −2.61 to −0.38 |
| Odd 4K / auto | 0 | −4.67 to −4.44 | +1.11 to +6.32 |
| Odd 4K / auto | 1 | −4.65 to −4.35 | −1.80 to +0.04 |
| Odd 4K / one | 0 | −0.67 to +0.28 | −1.02 to +2.90 |
| Odd 4K / two | 0 | −3.64 to −2.56 | −1.83 to +1.83 |

Odd-4K automatic-thread quantization-phase paired medians are all slower:
+2.73 to +14.13 ms in repetition 0, and +0.43 to +3.24 ms in repetition 1.
The quantization timer encloses candidate-cost readback and the CPU merge
(`workflow.cpp` → `quantization_pipeline.cpp` → `ac_strategy_search.cpp`).
Subtracting merge time **from each encode's quantization time before pairing**
leaves eight positive residual paired medians, +4.35 to +18.93 ms. This is
not a subtraction of separate medians. Serialization deltas are smaller and
mixed. The residual does not distinguish other CPU work, GPU execution,
launch/wait time, or shared-machine drift; it does not prove a cache, clock,
threading or GPU-kernel cause. Duplicate-control residuals are also noisy.
The one-thread complete-call control is mixed despite no new tile workers.

The final clean batch comparison adds **912 checked / 640 measured encodes**
across HD/4K two-image batches, independent requests and the public driver,
each repeated in reversed order. Each request has a two-thread CPU budget.
Four warm rounds precede 12 measured HD or eight measured 4K rounds; every
label/position combination is balanced. All 32 candidate-versus-control
paired merge-sum medians are lower: HD saves 2.51–3.69 ms and 4K saves
9.60–15.93 ms summed across the two requests. These overlapping scoped sums
are not batch latency. Complete-call results are mixed across repetitions
for every input/driver combination, ranging from −3.94% to +4.70%. All four
paired medians of the second independent 4K batch are slower (+1.09% to
+1.69%). The batch data does not support a general throughput improvement.

The bounded scheduling mechanism is byte-exact and locally effective, but
the measured end-to-end tradeoff does not justify enabling it. The final
tracked runtime is the existing serial candidate-cost merge from S111. The
candidate header/test were removed from the active checkout after archiving
their exact contents; the archive is recoverable. S111 GPU fusion and the
opt-in compact-storage policy are unchanged.

## Evidence and reproduction

Root: `U:/gjxl-cuda-diagnostics/s112`; scripts and diagnostic sources:
`build-cuda-ninja/profiles/s112_*`. Outputs use exclusive names. The first
extra-harness build failed because object/library arguments preceded
`/link` while `/TP` was active, and the renamed main needed an explicit
return value. No executable was produced. The failed log and original
wrapper/source are preserved; the corrected build succeeds.

Archived integration files are `integrated_ac_strategy.cpp`,
`ac_strategy_schedule_internal.h`, `ac_strategy_schedule_test.cpp`, and
`integrated_CMakeLists.txt`. `clean_strategy.cpp` preserves the final timing
instrumentation. Prototype sources, original/corrected failed-build inputs,
release/ASAN binaries, fresh candidate `build-cuda`, logs, input/oracle
references and recomputable analyses are retained. Reproducing the candidate
requires restoring those archived source files in an isolated checkout;
the current serial runtime checkout must not be mistaken for that build's
source. Do not overwrite the frozen evidence outputs with reruns.

The timing windows on 2026-09-08 (UTC) are:

| Campaign | Start | Finish |
| --- | --- | --- |
| Exploratory single | 06:21:47.174 | 06:28:21.747 |
| Exploratory batch | 06:28:50.075 | 06:32:08.915 |
| Clean single | 06:43:39.778 | 06:47:24.023 |
| Clean batch | 06:49:03.812 | 06:52:15.623 |

`s112_validate.py` checks per-job log/executable/input/oracle hashes, exact
success/error markers, counts, all five recomputed analyses, label-position
balance, nonoverlapping GPU jobs and build/host-test exclusion from timing
windows. It also verifies the restored runtime source against `64ee875`,
the retained binaries, and the S111 baseline libraries. `s112_freeze.py`
records the evidence and documentation hashes in `manifest.json`. The
post-freeze check is:

```powershell
python build-cuda-ninja/profiles/s112_validate.py --frozen
```

The 40-file retained runtime and S111 libraries are not rebuilt in place.
No firewall/admin prompt or permission blocker has appeared. No security,
driver, clock, power, priority or process-affinity settings changed.
No task build or sanitizer overlaps a timed campaign; light editing and
ordinary shared-machine state remain measurement limitations.

The next investigation should locate the quantization-phase time outside
the merge, before attributing its regression to any particular mechanism or
repeating broad timing sweeps. Future work should also examine the serial
search-grid export: it allocates a second byte-cell owner and reconstructs
anchors through validated setters
although the private search grid already owns a complete layout. Any direct
ownership handoff must retain coverage validation and failure atomicity.
This is an unimplemented hypothesis, not a measured saving. The encoder has
not been established to be maxed out.
