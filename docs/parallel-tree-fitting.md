# Parallel local CPU preparation and fitting

Branch: `perf/parallel-tree-fitting`, based on the committed attribution work
at `e3db25a`. This implements step 2 of the tree-fitting acceleration plan.

The subsequent [fitting scheduling update](modular-fitting-scheduling.md)
replaces the bounded waves described here with a bounded ready queue and
shared feature cohorts. The measurements below describe the original wave
scheduler, with its recorded revisions and routing.

The [fitting profile](modular-histogram-fitting-profile.md) found 192 independent
local fits in each 12 MP photograph. Fitting consumed 69–72% of the encoder
interval. This slice runs existing local sample preparation and fitting across
CPU participants. Each individual fit retains its sampling, quantization,
histogram arithmetic, split choices, ties, partition order and tree layout.

## Execution and ownership

`ReferenceWorkflowOptions::learning_cpu_participants` controls local jobs:
zero uses the caller's allowance; one selects the serial control. The public
learned policy now honors `ModularEncodingOptions::cpu_thread_count` for this
work. The shared execution domain and per-image CPU budget both constrain
worker grants. Private calls without CPU participation or within an explicit
parallel scope stay serial. Global/buffered learning still performs one serial fit.

One job prepares a stream's samples, fits its tree, releases its sample columns,
and compacts the tree. Jobs run in bounded waves, with at most one admitted job
per requested participant. Each owns its mutable sample columns, histogram
workspace, tree and diagnostic counters. The caller also executes work. Worker
launch uses the existing CPU reservation, join and exception machinery.

CPU and hybrid routes share the scheduler. In the hybrid route, a wave stays
inside the existing feature batch. Read-only feature maps remain owned until
all jobs join; GPU token generation then sees the completed trees in canonical
stream order. Only one local feature batch remains live, with the existing
8 Mi-sample ceiling. This slice does not overlap GPU submission with CPU fits.

## Memory admission and failures

Before launch, each job reserves a conservative complete-work envelope:

- The existing geometry/RNG-based sample-preparation bound.
- Fitter workspace and maximum tree backing, using a capacity-only planner.
- Replacement tree backing during compaction.

The workspace planner shares the same allocation formula as the existing
sample-shape planner. The local workflow has no multiplier boxes; its property
cut limit bounds both static and dynamic histogram axes. Temporary preparation
storage is conservatively included alongside fitting workspace.

Reservations include other retained image, feature, token and tree backing
through the same managed-memory domain. Scheduler metadata, worker handles,
status slots and enabled per-job diagnostics are also admitted. A capacity
shortage ends the current wave early. If even one complete-job envelope cannot
be admitted, scheduler metadata is released and the remaining work uses the
original, tighter serial per-stage admission. No rows or trees are truncated to
fit a budget. Admission does not wait while retaining prerequisite resources.

Workers always join before their reservations or feature owners are released.
Allocation and launch failures propagate; partial launches are joined without
retrying completed jobs. The prepared frame, complete output and published
workflow statistics retain the existing publish-on-success contract.

## Timing and qualification contract

`learning_seconds` measures caller elapsed time across admission, preparation,
fitting, joins and counter reduction. `sample_preparation_seconds` and
`fitting_seconds` are sums of individual job wall intervals. They overlap in
parallel execution and must not be added to elapsed stages. Top-level Modular
wall/process-CPU attribution uses only the caller's joined interval.

Detailed fitter instrumentation gets independent per-job sinks. Counters and
fit records merge after joins in stream order. Fitter phase durations are work
sums; the disabled path allocates no fitter diagnostic sinks.

The profiling driver accepts:

```text
gjxl_modular_hybrid_profile INPUT OUT ROUTE auto best fit-off LEARNING_PARTICIPANTS CPU_CEILING
```

For example, compare learning participants 1, 4 and 14 with the same CPU ceiling
of 14 and route `auto`. CPU feature preparation retains its existing maximum of
four participants. Passing an explicit learning count selects result schema 2,
which separates `detail_wall` from `detail_work`. Older invocations preserve the
serial control and schema 1 for the existing profiling scripts.

Qualification gates:

- Exact trees, learning work and complete prefix/ANS bytes across participant
  counts, repeated calls, and CPU versus hybrid feature/token routes.
- Independent libjxl decoding, including RGB/RGBA, 8/16-bit, Best/All predictor
  cases and multi-batch GPU execution.
- Domain/per-image ceilings, absent participation, nested work, shared-domain
  callers, memory-driven concurrency reduction, and partial-launch unwinding.
- Matched AC-power measurements on the original 4000×3000 images 1029, 1039 and
  1049, preserving raw results, output hashes, input/build provenance and timing
  boundaries. Performance claims use caller elapsed time, not worker sums.

## Validation

Release, Apple Clang 21, Metal enabled, pinned libjxl decoder `e8ff0976`:

| Check | Result |
| --- | --- |
| `modular_parallel_learning` | Pass: trees, prefix/ANS bytes, exact decode, Best/All, ordered profile reduction, CPU/memory limits, partial launches and shared domains |
| `modular_reference_workflow` | Pass |
| `modular_histogram_learning`, `modular_histogram_preparation` | Pass |
| `modular_reference_hybrid`, `modular_reference_hybrid_palette` | Pass: serial CPU versus parallel hybrid, multi-batch ownership, All predictors, wide palette fallback and failed worker launch |
| `modular_api` | Pass: C/C++ parity, public default concurrency, global serial fitting, memory limits and failure atomicity |
| ThreadSanitizer, `gjxl_modular_parallel_learning_test --preparation-only` | Pass, no reported races |

The ThreadSanitizer build disables Metal and uses `-O1 -g -DNDEBUG
-fsanitize=thread -fno-omit-frame-pointer`. Its focused mode retains tree/profile
parity, CPU ceilings, memory fallback, partial launches and concurrent callers;
it omits entropy encoding and uses a smaller memory-pressure fixture. The full
instrumented test exceeded its 180-second timeout and is **not** counted as a
pass. Complete encoding and independent decoding passed in Release. The focused
sanitizer log is `build/tsan/parallel-learning-preparation.log`.

## Performance qualification

2026-10-15, Apple M4 Pro (10 performance + 4 efficiency cores), 48 GiB,
macOS 27.0.1, Apple Clang 21, Release `-O3 -g -DNDEBUG`. Automatic hybrid
features/tokens, automatic local trees, Best predictors, estimated direct ANS
with 64 clusters. All three original inputs are 4000×3000 RGB8 photographs.

The same executable and a CPU ceiling of 14 were used for every configuration;
only `learning_cpu_participants` changed. CPU feature preparation retained its
four-participant maximum. Automatic feature routing and Metal token generation
were unchanged across participant counts:

| Image | Feature/token batches | GPU feature batches | CPU feature batches |
| --- | ---: | ---: | ---: |
| 1029 | 5 | 4 | 1 |
| 1039 | 20 | 3 | 17 |
| 1049 | 8 | 3 | 5 |

Metal generated tokens for every batch, including those with CPU-produced
features. Learning reused all completed maps, computing no replacement weighted
features. Every configuration performed the same 192 fits per image. Detailed
fitter probes were disabled; top-level profiling and lightweight workflow
timers remained enabled in all configurations.

There were **36 complete encodes**: one warmup and three measured repetitions
for each image/participant count. Fresh processes ran in rotating participant
order. Both boundaries of every call observed AC power. Every output decoded
exactly with pinned libjxl and matched the earlier qualification's SHA-256.
All observations, including warmups and timing ranges, are retained in
[the results](parallel-tree-fitting-results.json).

### Complete encoder time

Seconds, medians of three measured calls. The final row sums per-image medians.

| Image | Serial learning | 4 participants | 14 participants | Speedup, 14 vs 1 |
| --- | ---: | ---: | ---: | ---: |
| 1029 | 36.631 | 15.580 | 10.697 | 3.42× |
| 1039 | 27.109 | 12.079 | 8.892 | 3.05× |
| 1049 | 24.673 | 11.005 | 7.712 | 3.20× |
| **Total** | **88.413** | **38.664** | **27.301** | **3.24×** |

The measured 14-participant ranges were 10.501–10.701 s, 8.848–8.995 s and
7.613–7.730 s respectively. End-to-end here is the encoder API interval:
admission, selection, preparation/fitting, GPU handoffs, entropy and publication
plus frame teardown. Input/output file I/O, backend creation and independent
decoding occur outside the timer.

Files remain **10,507,251 / 7,138,537 / 7,015,756 bytes**: 24,661,544 bytes total.
This is a latency improvement at exactly the same rate and reconstruction.
The measurements qualify the private hybrid route; the public CPU route's
parallel behavior is covered by correctness/resource tests. No new libjxl
timing comparison was performed in this slice.

### Preparation and fitting

These are joined caller-wall intervals, including the scheduler, admission,
sample preparation, fits, compaction and diagnostic reduction. They exclude
feature preparation and GPU token generation.

| Image | Serial learning | 4 participants | 14 participants | Speedup, 14 vs 1 |
| --- | ---: | ---: | ---: | ---: |
| 1029 | 30.086 | 8.987 | 4.148 | 7.25× |
| 1039 | 22.006 | 6.953 | 3.737 | 5.89× |
| 1049 | 20.006 | 6.340 | 3.060 | 6.54× |
| **Total** | **72.098** | **22.280** | **10.946** | **6.59×** |

The 14-participant route used **18 / 29 / 20** bounded learning waves for
1029/1039/1049, with no memory-driven reductions. Serial learning made one
sequential pass per feature batch: **5 / 20 / 8** passes. These counts include
the empty global learning scope and partial final waves. Feature-batch boundaries
remain synchronization boundaries for local fitting.

Worker-time sums remain in `detail_work`; they cannot be substituted for this
table. Total process CPU medians increased from 36.49/27.08/24.58 s with serial
learning to 42.35/30.49/27.80 s with 14 participants. Work counts and encoded
decisions stayed identical. The measurements establish latency scaling;
they do not attribute the CPU-time increase to a particular hardware effect.

### Remaining stages and memory

Median stage seconds with 14 learning participants. Learning is the disjoint
sub-interval above; feature preparation is the remainder of the training stage
shown separately. Independently computed medians need not sum exactly.

| Image | Transform/predictor selection | Sample preparation + fitting | Features | Token stage | Entropy modeling | Emission |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1029 | 3.711 | 4.148 | 0.079 | 0.092 | 2.241 | 0.351 |
| 1039 | 3.277 | 3.737 | 0.104 | 0.086 | 1.346 | 0.289 |
| 1049 | 3.051 | 3.060 | 0.073 | 0.073 | 1.108 | 0.291 |

Learning and transform/predictor selection are now comparable bottlenecks.
Selection still runs serially. Its 3.05–3.71 seconds per image make parallel
local selection a useful next CPU candidate, alongside further fitter work.
The current wave barrier also leaves room for a future scheduler that keeps
workers busy across uneven job durations. Neither change is included here.

The complete encoder's peak managed backing was unchanged across all three
participant counts: **717.892 / 638.827 / 615.689 MiB** for 1029/1039/1049.
Peak committed capacity, including unused reservations, was likewise unchanged
at **1814.371 / 1594.945 / 1528.747 MiB**. Learning uses extra concurrent
scratch; the later entropy stage still sets the overall peak. These are managed
allocation counters, excluding caller input, decoder storage, stacks, allocator
metadata and backend initialization; they are not process RSS measurements.

### Reproduction and evidence

The capture is retained in
`build/parallel-learning-qualification-20261015`, including all complete files,
per-call JSON, stdout/stderr, environment snapshots, a JSONL ledger, summary,
build flags, frozen executable, source archive and manifest. The manifest
verified source, binary, input, dependency and submodule hashes after the run.
Sources were an uncommitted working snapshot based on `e3db25a`.

- Executable SHA-256: `01698d3d27140a484ec55eddcdb610eddd7e07d70a7647be961c49103cd97331`.
- Source archive SHA-256: `0cdbc817ff66bffc75fab96e7a3ec097c078b6f8ba892b561f7e9ecbfc2d9dae`.

```sh
python3 tools/modular_reference_learning/run_parallel_learning.py \
  --binary build/release/gjxl_modular_hybrid_profile \
  --input /path/to/photo-large-1029.pam \
  --input /path/to/photo-large-1039.pam \
  --input /path/to/photo-large-1049.pam \
  --out build/parallel-learning-qualification \
  --reference docs/gpu-histogram-learner-integration-results.json \
  --participants 1 4 14 --cpu-ceiling 14
```
