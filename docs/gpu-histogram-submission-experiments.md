# Histogram submission: driver timing, buffer reuse and encoder boundaries

Reusing transient Metal buffers reduced the reported CPU-driver scheduling
interval by **90–97%** in fixed batches, and reduced batch evaluation time by
**5.9–21.3%**. Splitting the same dispatches across multiple compute encoders
showed no consistent benefit. The complete encoder's exact-size cache hits too
rarely to reproduce the fixed-batch benefit: its nominal 2.1% improvement does
**not establish an end-to-end speedup** in this variable session.

This follows [the command-wait study](gpu-histogram-collection-and-waits.md).
Measurements used an Apple M4 Pro, 14 CPU cores, 48 GiB RAM, macOS 27.0.1,
and **AC power** on 2026-10-16. The initial battery queue collected no samples;
the machine returned to AC while waiting, and the AC cohort was announced
before measurement. All competing codec studies finished before the accepted
captures. Power and power settings remained consistent, but Apple background
services were active. These results are a separate matched comparison and do
not replace earlier absolute performance baselines.

The [machine-readable results](gpu-histogram-submission-experiments-results.json)
include medians, ranges, hashes, cache counts and artifact locations.

## Fixed batches: the resource-reuse effect

The following are median evaluation-call times in milliseconds over twenty
measured rounds per configuration. Input preparation, the native CPU oracle
and result destruction are outside this interval; result destruction is
recorded separately. All rows use synthetic inputs held fixed across variants.

| Batch | Fresh / single | Reuse / single | Fresh / multiple | Reuse / multiple | Reuse / single latency reduction |
|---|---:|---:|---:|---:|---:|
| 1 node × 16,384 rows | 1.931 | 1.604 | 1.924 | 1.631 | 16.9% |
| 10 nodes × 32,768 rows | 4.344 | 3.421 | 4.359 | 3.449 | 21.3% |
| 32 nodes × 131,072 rows | 34.583 | 32.537 | 34.538 | 32.521 | 5.9% |

Every measured reuse evaluation reused all three buffers. Multiple mode used
5, 13 and 33 encoders respectively, versus one, with one command buffer in
every case. Changing encoder count changed the median by approximately
−0.3% to +1.7%, without a consistent improvement.

The scheduling timestamps locate the largest saving inside the driver's
reported scheduling interval. The following also uses milliseconds and one
compute encoder:

| Batch | Driver interval, fresh | Driver interval, reuse | GPU span, fresh | GPU span, reuse |
|---|---:|---:|---:|---:|
| 1 × 16K | 0.284 | 0.027 | 1.445 | 1.427 |
| 10 × 32K | 0.857 | 0.025 | 3.223 | 3.209 |
| 32 × 128K | 1.161 | 0.044 | 31.928 | 31.877 |

The GPU span changed by less than 1.4% in these comparisons. Host allocation
and packing also became cheaper. This is evidence that repeatedly introducing
fresh buffers causes substantial scheduling overhead. The timestamps expose
elapsed scheduling intervals; they do not identify the underlying VM,
residency or zeroing mechanism, or measure pure CPU busy time.

## Full fitter: startup attribution and actual reuse

Across the nine measured fresh/single image encodes, the delay from commit
entry to GPU start totaled 9.986 seconds over 4,501 batches. Its exclusive
partition, clipped at GPU start, was:

| Portion of startup delay | Total | Share |
|---|---:|---:|
| Before driver scheduling begins | 2.791 s | 28.0% |
| Inside driver scheduling, before GPU start | 6.804 s | 68.1% |
| After scheduling ends, before GPU start | 0.391 s | 3.9% |

In 249 batches, GPU execution began before the scheduling interval ended.
Clipping those intervals avoids double-counting their overlap. These totals
are accumulated operation timings across nine encodes, not a partition of
whole-encoder wall time: CPU fits execute concurrently.

Real batch node counts and bin/symbol spans change the required buffer sizes.
The single-slot cache requires an exact capacity match, producing the following
hit rates across measured reuse runs:

| Reuse mode | Output hit | Dispatch hit | Scratch hit | All three hit |
|---|---:|---:|---:|---:|
| Single encoder | 31.61% | 0.67% | 14.45% | 0.35% |
| Multiple encoders | 30.52% | 1.25% | 14.03% | 0.50% |

Only about 20% of cumulative requested scratch bytes used cached backing.
Adaptive scheduling also changed batch counts between variants, despite
identical logical fitting work. Consequently, the complete encoder does not
receive the fixed probe's consistently warm-buffer behavior. Idle managed
capacity after encoding ranged from 4.8 to 13.9 MiB in reuse runs; explicit
trimming then released every tracked owner. This is capacity accounting, not
a measurement of physical resident RAM.

## Uninstrumented complete encodes

Each cell is a per-image median in seconds over three measured repetitions;
the total is the sum of those three image medians. All configurations have one
excluded warmup. The previous binary is a control for the ordinary fresh/single
path; the other four rows use the new binary with both controls crossed.

| Configuration | 1029 | 1039 | 1049 | Sum of medians |
|---|---:|---:|---:|---:|
| Previous binary, fresh / single | 9.585 | 7.007 | 6.692 | 23.285 |
| New binary, fresh / single | 9.852 | 7.001 | 6.817 | 23.670 |
| New binary, fresh / multiple | 9.507 | 6.842 | 6.716 | 23.066 |
| New binary, reuse / single | 9.738 | 6.744 | 6.684 | 23.166 |
| New binary, reuse / multiple | 9.580 | 6.943 | 6.682 | 23.205 |

Relative to new fresh/single, the nominal reductions are 2.55%, 2.13% and
1.97% for fresh/multiple, reuse/single and reuse/multiple. The previous binary
is itself 1.63% lower than new fresh/single. These small differences do not
establish either an optimization win or a disabled-path regression.

The ranges overlap and runtimes drifted during capture. For example, on 1029,
new fresh/single ranged from 9.332–10.251 s and reuse/single from
9.300–10.083 s. Snapshots record active `mediaanalysisd`, `mds_stores` and other
system services; their contribution to the variability was not isolated.
All observations are retained. Three measured repetitions, with five variants,
do not fully balance every position, so small order/drift effects remain
possible. Traced and untraced absolute times also varied substantially; do not
subtract their times to estimate instrumentation overhead.

The evidence supports a **bounded cache that can reuse suitably larger
backing**, with admission charged for the actual retained capacity and logical
buffer bounds kept explicit. That is the next useful experiment: improve and
measure hit coverage, then repeat the full comparison. Keep a single compute
encoder and keep reuse opt-in for now. A resident frontier fitter could also
reduce submission and CPU handoff counts, but these measurements do not
qualify that larger redesign or a speedup over CPU fitting/libjxl.

## Controls and timing contract

The private profiler accepts `--fit-buffers=fresh|reuse` and
`--fit-encoders=single|multiple`. Defaults remain fresh buffers and a single
compute encoder. For a given batch, both routes retain the same kernels,
dispatch grids, resource barriers, scratch layout and arithmetic, with one
command buffer and one wait. The complete fitter's adaptive scheduling can
change batch membership and therefore its total number of dispatches.
Multiple mode ends the compute encoder after each dispatch, including validation
and the four operations in each scratch wave. It inserts no timestamp counters.
Existing `gpu-stage` profiling continues to use its instrumented encoder layout;
the factorial comparison instead uses `gpu-wall` with wait tracing.

Reuse uses three exact-capacity slots in the backend's existing bounded,
purgeable preparation cache: output, dispatch and scratch. An idle slot can be
used only after domain admission succeeds; another size/domain or reclaimed
backing causes fresh allocation. Result backing is returned only when its last
owning result object is destroyed. Failures drain submitted work and discard
transient leases. Explicit trimming and domain eviction include the new slots.
Profiles report actual reuse bits and bytes, so a requested reuse policy is not
mistaken for a guaranteed hit. No high-water over-allocation is introduced.
The complete-call profiler records idle cached capacity after encoding, trims
its owned cache outside the timed interval, and then requires zero committed
storage, backing owners, pending allocations and open reservations. This
retains the strict cleanup check while accounting for intentional caching.

Wait traces additionally read `MTLCommandBuffer.kernelStartTime` and
`kernelEndTime` after completion. These are host timestamps of **CPU-driver
scheduling**, not shader-kernel execution. See Apple's
[start-time contract](https://developer.apple.com/documentation/metal/mtlcommandbuffer/kernelstarttime)
and [end-time contract](https://developer.apple.com/documentation/metal/mtlcommandbuffer/kernelendtime).
The analysis retains commit-to-scheduling-start, scheduling interval, and
scheduling-end-to-GPU-start. The last difference is signed: CPU scheduling and
GPU execution may overlap, so these are not assumed exclusive CPU busy times.
There are no extra callbacks or submission boundaries. New trace/control
invocations use profiler schema 10; older calls retain their prior schemas.

## Experiment design

The complete-call experiment crosses both controls on the three 4000 x 3000
RGB8 photographs 1029/1039/1049. The policy remains learned Best, Metal unique
sample rows, a 16,384-row GPU cutoff, 64-fit cohorts, four histogram shards,
64 active properties and a 64 MiB scratch ceiling. CPU limits remain 14 with
four participants for CPU weighted-feature work and an 8 GiB execution domain.

Attribution uses all four variants with ordinary command-buffer timing and
wait tracing. A separate uninstrumented comparison includes the four variants
and the frozen prior binary as a disabled-path control. Each configuration has
one warmup and three interleaved measured repetitions. Complete bytes, logical
fitting work and independent pinned-libjxl decoding must match. Captures record
power, thermal state, competing processes and source/binary/dependency hashes;
detected overlapping runs are retained and excluded.

The fixed-batch probe removes the adaptive CPU scheduling/batch-composition
variable. It uses **synthetic** fixtures, not captured photographic nodes: one
16K-row node, ten 32K-row nodes and thirty-two 128K-row nodes, with two predictors
and sixteen properties. Samples, memberships and parent summaries remain
identical across all four variants. Native CPU evaluation checks every property
result and cost bit. Each case uses three warmup rounds and twenty interleaved
measured rounds. Input generation, CPU oracle and backend creation are outside
operation timing. Output destruction is measured separately. This probe gives
causal evidence for these fixed shapes, not an encoder speedup estimate.
The fixed-batch order balances all four positions and within-round ordered pairs
over four rounds; the twenty measured rounds cover five complete cycles. The
four-variant image study uses the same order, and the five-variant qualification
rotates its starting variant. Each image configuration still has three measured
repetitions, so its timing ranges are retained rather than treating the medians
as precise estimates of small differences.

Local artifacts: `build/gpu-histogram-submission-study-20261016/`.

## Reproduction

Build `gjxl_modular_hybrid_profile`, `gjxl_modular_histogram_submission_probe`
and the dedicated Metal histogram evaluator/workflow tests in the Release build.
Use the existing wait-study runner with a four-variant JSON list and
`--power=ac`; each variant adds the two controls above. Attribution also
adds `--gpu-profile=gpu-wall --gpu-wait-trace=on`; qualification uses
`--gpu-profile=gpu-off`.

```sh
python3 tools/modular_reference_learning/run_histogram_submission_probe.py \
  --binary build/release/gjxl_modular_histogram_submission_probe \
  --out NEW_PROBE_DIRECTORY --repeats 20 --power=ac
```

## Correctness validation

The Release evaluator, complete-tree workflow and resource-budget suites passed.
The final expanded evaluator suite covers 308 exact batches, 6,161 property
results, 15 complete trees and 27 failure checks. It exercises all four control
combinations, concurrent requests, a held result that must not be overwritten,
reuse after result release, cache trimming generations, reclaimed scratch
backing, and submission/completion failures. Multiple encoders still produce one
command-buffer submission. Native CPU candidates, costs and tree order remain
the oracle. These are focused tests; no new sanitizer or shader-validation run
is claimed.

Evidence is retained in `tests.log`, `evaluator-final.log`, and the final CTest
log under the artifact directory/build tree. An intermediate test assertion
incorrectly required even tiny buffers to lose their backing after requesting
purging; it was corrected to require scratch reclamation and exact results.
The final expanded suite passed after that correction.

The accepted cohort contains **276 exact fixed-batch evaluations and 108 exact
image encodes**: 48 traced and 60 uninstrumented, including warmups. For each
image, outputs are byte-identical across configurations and both phases, independently
decode exactly with the pinned libjxl libraries, retain identical logical
learning/GPU work, and have zero GPU admission fallbacks. Every recorded result
and output hash was audited. No accepted run overlapped a detected competing
codec/build/capture job. This does not imply absence of system background work.

An initial image attempt stopped when the profiler's old zero-storage assertion
encountered intentional idle cache backing. Independent decode had already
passed. The profiler now trims outside timing before the assertion; a complete
1029 reuse/multiple check passed, and all image configurations were restarted.
The partial attempt and recovery check are retained and excluded from timing
summaries. GPU implementation and probe binary did not change during that fix.

## Provenance and artifacts

The worktree is `gjxl-parallel-tree-fitting`, branch `feat/gpu-histogram-fitting`,
base revision `a2b6ff1eda55a7534dee1919e63c65932cd022c6` plus the recorded working
diff. Frozen binaries and dependency/source hashes are retained in
`build/gpu-histogram-submission-study-20261016/`:

- `fixed-batches-ac/manifest.json`: accepted fixed probe, one isolated attempt.
- `attribution-qualified/results.json`: 48 accepted encodes, no rejected overlaps.
- `qualification/results.json`: 60 accepted encodes, no rejected overlaps.
- `analysis.json`, `trace-audit.json`: audited aggregates and exclusive startup
  partition; `analyze.py` and `finalize_report.py` generate the summaries.
- `provenance.json`, `candidate.patch`, `new_sources/`, `candidate-CMakeCache.txt`:
  source/build identity; `baseline-provenance.json` identifies the prior binary.
- `attribution/` and `cleanup-check/`: excluded partial attempt and cleanup check.
- `power-settings-before.json`, `power-settings-after.json`: unchanged power
  settings and recorded AC condition. The initial battery queue produced no data.

Binary SHA-256 prefixes are `3503b1c5a6f3` (previous binary), `c9f94796e9bb`
(complete-call candidate), and `20652abd02d9` (balanced fixed probe). Full hashes
and accepted-manifest hashes are in the machine-readable results. The fixed
probe and image captures differ only in the profiler source's post-timing
cleanup/reporting fix; GPU/runtime sources are identical.
