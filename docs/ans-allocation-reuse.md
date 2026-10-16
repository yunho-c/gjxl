# ANS search allocation reuse

This work starts from `ebfa2d508d6defbe6446b193716d345bffbf6410` on
`feat/modular`, including its histogram-fitter scan optimization. The isolated
branch is `perf/ans-allocation-reuse`. The separate parallel-workflow profiling
worktree and its uncommitted changes are not part of this implementation.

## Implementation

ANS search now stores each candidate's at-most-256 normalized frequencies
inline with its syntax fields. A lightweight histogram view lets the existing
cost estimator and histogram writer consume either this search storage or an
owning final model. The estimator, precision/configuration search order,
clustering, tie comparisons and model selection are unchanged.

`BuildBestAnsHistogram` uses fixed local arrays for trials, including the
population costs used by cluster refinement. Configuration candidates retain
their inline populations in one managed allocation per cluster, reserved once
for the policy's maximum candidate count. Copying an equivalent population or
discarding a trial no longer acquires an allocation ticket. Workers own distinct
cluster storage; no mutable search storage is shared and no candidate view
escapes the search. Only selected models allocate owning frequency buffers and
encoder lookup tables, retaining the existing external model representation.

Direct ANS raw-value vectors also reserve their already-known cluster counts.
Raw-value collection and aggregation remain separate passes. In particular,
large sparse 32-bit value populations can still allocate hash-map nodes during
aggregation; this change does not eliminate every ANS allocation.

The optimizer's storage planner accounts for the larger inline candidate type
and its exact reserved capacity, and removes the former separate frequency
backings. Scratch arrays are bounded independently of token count. The global
managed allocator, synchronization, resource limits, output ownership and
failure-atomic publication are unchanged.

## Correctness and resource qualification

Fifteen focused Release suites passed: `entropy`, `entropy_storage_plan`,
`ans_selection`, `ans_selection_faults`, `modular_parallel_entropy`,
`modular_api`, `modular_prepared_stream`, `ans_bit_emission`, `ans_reverse_bits`,
`modular_reference_hybrid`, `modular_reference_hybrid_palette`,
`modular_prepared_stream_reference`, `metal_modular_reference_workflow`,
`context_map` and the new `ans_search_storage` regression.

The storage suites cover exact managed admission, retained output accounting,
empty/full-alphabet populations, up to 257 contexts and 64 clusters, prepared
and deferred models, and allocation-failure unwinding. Parallel entropy tests
check shared/local/mixed models, concurrent calls, domain limits, launch and
allocation failures, byte parity and independent decoding.

The new regression freezes 42 model-plus-ordered-payload outputs from the
unmodified baseline. Empty, single-value, binary, dense, wide and repeated-value
fixtures exercise prefix-derived ANS and balanced/high-density/rate-optimized
direct ANS, with exact and estimated selection where supported. Every output
and candidate count matches. Serialization runs after search scratch has been
destroyed, checking retained model ownership. The probe uses the existing
thread-local fault-injection hook to count successful allocations in serial
diagnostics; production allocation tracking is unchanged.

For the rate-optimized estimated cases with an exact cost requested, the probe
counts the following managed backing allocations. Nonempty fixtures contain
8,193 tokens over 12 contexts and three sections, one of which is empty.

| Value fixture | Baseline | Reuse |
| --- | ---: | ---: |
| Empty | 32 | 27 |
| Single value | 86 | 38 |
| Two values | 208 | 115 |
| Dense 8-bit | 109 | 39 |
| Spread 16-bit | 105 | 49 |
| Sparse 32-bit | 8,306 | 8,256 |
| Repeated runs | 98 | 50 |

Across all 42 cases the counts are 46,422 versus 44,998. The sparse 32-bit
fixtures dominate that total through unchanged aggregation hash-map nodes.
These are diagnostic fixture counts, not allocation counts for the photos.

CPU-only ThreadSanitizer builds passed `entropy_storage_plan`,
`modular_parallel_entropy` and `ans_search_storage`, using `-O1 -g -DNDEBUG`,
`-fsanitize=thread -fno-omit-frame-pointer` and
`TSAN_OPTIONS=halt_on_error=1`. These checks exclude Metal and the external
decoder; the Release suites cover both. No races were reported.

Logs: `build/reuse-focused.log`, `build/reuse-regression.log`,
`build/reuse-context-map.log` and `build/reuse-tsan.log`.

## Performance capture

The paired capture uses the public learned encoder with default estimated
direct ANS on the M4 Pro, with explicit CPU and Metal routes. Both executables
use the same base revision, input files, policies and Release `-O3 -g -DNDEBUG`
configuration. Backend setup, file I/O and independent decoder validation are
outside the complete encoder interval.

Each condition uses one discarded fresh process and three measured fresh
processes, alternating baseline/change order. The three 4000x3000 RGB8 photos
1029, 1039 and 1049 use 14 participants; 1029 also has a one-participant CPU
control. At 14 participants the stream interval is joined local modeling plus
emission wall time. The serial control instead sums its disjoint model and
emission wall stages. Neither interval is a sum of overlapping worker timers.

All-thread 1 kHz Samply captures are separate diagnostic calls. Attribution
uses the integrated profiling study's analyzer, frozen alongside the new raw
captures, including both current- and previous-stack CPU-delta assignments.
Mutex attribution is sampled CPU evidence, not recoverable elapsed time.

AC power and active-desktop load are recorded before and after every call.
Raw observations, including background activity, are retained. This is a
paired three-image study, not a general codec speed or libjxl comparison.

All **68 encodes** (14 discarded calls, 42 measured calls and 12 Samply calls)
matched the qualified bytes and independently decoded exactly. All parallel
calls used 14 entropy participants for 192 populated streams. Source, frozen
executable, input and linked-library hashes remained unchanged during capture.

| Image/backend, 14 participants | Complete seconds, baseline -> reuse | Joined stream seconds, baseline -> reuse |
| --- | ---: | ---: |
| 1029 CPU | 5.385 -> 4.962 | 1.078 -> 0.388 |
| 1039 CPU | 4.076 -> 3.518 | 1.001 -> 0.324 |
| 1049 CPU | 3.433 -> 2.938 | 0.811 -> 0.309 |
| 1029 Metal | 6.443 -> 4.988 | 1.272 -> 0.404 |
| 1039 Metal | 5.099 -> 4.009 | 1.010 -> 0.328 |
| 1049 Metal | 4.066 -> 3.220 | 0.876 -> 0.318 |

Values are medians of three calls, computed separately for each interval.
Joined stream modeling/emission improved **2.62-3.15x**. Complete-call median
ratios were 1.09-1.29x, but the complete-call variation is substantial: for
example, CPU 1029 ranged from 4.683-7.338 seconds before and 4.080-6.263 seconds
after. Another GJXL profiling workload appeared in 19 of the 84 before/after
snapshots for measured calls. The archive retains these observations, desktop
load and every individual result; this is not an isolated-machine benchmark.

For the one-participant CPU 1029 control, serial model plus emission medians
were **2.706 -> 2.648 seconds**, and complete medians were 35.130 -> 34.059
seconds. The small serial change and much larger parallel stage change are
consistent with reducing contention between workers. They are not an exact
decomposition of the complete speedup.

| Image/backend | Managed-mutex sampled CPU seconds, baseline -> reuse | Share of stream-encoding sampled CPU, baseline -> reuse |
| --- | ---: | ---: |
| 1029 CPU | 7.125 -> 1.196 | 60.8% -> 28.2% |
| 1039 CPU | 6.107 -> 1.599 | 62.7% -> 42.9% |
| 1049 CPU | 5.910 -> 1.616 | 62.5% -> 46.4% |
| 1029 Metal | 7.531 -> 1.389 | 58.6% -> 30.6% |
| 1039 Metal | 6.697 -> 1.522 | 63.8% -> 40.8% |
| 1049 Metal | 6.607 -> 1.598 | 63.8% -> 46.8% |

This uses current-stack CPU-delta attribution. Previous-stack attribution
gives 57.6-63.5% before and 25.0-44.9% after, with the same conclusion. The
absolute managed-mutex attribution fell **72.7-83.2%**; it is not a recoverable
wall-time estimate. Remaining managed allocations still contend: the change
substantially reduces this hotspot without eliminating it.

The CPU route's maximum committed managed capacity fell by 14,766,080 bytes
per image; Metal's complete-call maximum was unchanged because another stage
sets that envelope. Peak backing did not consistently fall: observed increases
reached about 8.3 MB. Inline candidates trade bounded storage for fewer small
allocations, and scheduling also affects overlapping backing lifetimes. These
are managed capacity metrics, not RSS or physical memory measurements.

[Structured results](ans-allocation-reuse-results.json) bind the frozen
executables, input identities, allocation probes, all timing ranges and Samply
attribution to `build/ans-allocation/qualification/manifest.json`. That manifest
and its raw artifacts retain both the baseline and candidate evidence. Initial
smoke calls under `build/ans-allocation/smoke-*` are excluded from the tables.

## Reproduction

Build `gjxl_ans_allocation_profile` with the pinned Modular oracle and Metal,
plus `gjxl_ans_search_probe`. Freeze a baseline executable/probe and source
manifest before changing ANS; `build/ans-allocation/baseline` retains this
study's originals. The baseline source archive includes the profiling tools
added to the base revision before the codec change.

```sh
python3 tools/modular_reference_learning/run_ans_allocation.py \
  --baseline build/ans-allocation/baseline --build build/release \
  --input /path/to/photo-large-1029.pam \
  --input /path/to/photo-large-1039.pam \
  --input /path/to/photo-large-1049.pam \
  --out build/ans-allocation/qualification \
  --samply-analyzer /path/to/analyze_parallel_samply.py

python3 tools/modular_reference_learning/report_ans_allocation.py \
  build/ans-allocation/qualification \
  --out docs/ans-allocation-reuse-results.json
```

The runner freezes the candidate executables and dSYM, source archive and
patch, build configuration, dependency/input hashes, analyzer sources and raw
results. Every complete output must independently decode and match the prior
qualified compressed hash. Verified duplicate outputs use APFS clones. The
new worktree reuses the pinned libjxl build and APFS-cloned Metal headers.
