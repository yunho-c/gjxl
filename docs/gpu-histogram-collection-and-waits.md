# GPU histogram collection and command waits

Most of the gap between GPU execution and caller waiting occurs before GPU
commands start. CPU participation restoration is negligible in these isolated
encodes. Shard and queue tuning does not show a robust speedup; retain the
defaults and test encoder boundaries without timestamp sampling next.

This follow-up starts from the qualified Modular consolidation at `a2b6ff1`.
It measures the GPU fitter with Metal unique sample rows, a 16,384-row cutoff,
64-fit cohorts, 14 CPU participants and an 8 GiB execution domain. Weighted
feature work on CPU retains its four-participant cap. The default
histogram layout has four shards, 64 active properties and a 64 MiB scratch
ceiling. The default queue target is 32 ready fits with a 500 microsecond flush
deadline. Public routing and these private defaults remain unchanged.

## Attribution contract

`--gpu-wait-trace=on` requires `--gpu-profile=gpu-wall` or `gpu-stage`.
It adds bounded, caller-owned records to the existing batch profile. There are
no completion callbacks, extra submissions or additional encoder boundaries in
wall mode. Stage mode retains the existing instrumented encoder boundaries.
Disabled tracing takes no new clock samples. Both successful results and their
operation trace publish together; failures drain submitted work before releasing
borrowed backing. The two commit timestamps are inline submission metadata.

The recorded host endpoints are commit entry/return, wait entry, blocking
entry/return and CPU participation restored. GPU start/end come from the
completed command buffer. Host timestamps use `mach_absolute_time` converted
with the system timebase. Apple's [GPU timestamp contract](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)
places GPU endpoints on the same system mach clock. The runner checks endpoint
ordering and the conservation of the blocking-window partition for every batch.

The blocking window is partitioned into time before GPU execution, GPU execution
overlapping the wait, and time after GPU completion. The last term includes
completion delivery, host scheduling and return/status/pool-cleanup overhead;
it does not separately identify those mechanisms. CPU resumption includes the
participation guard's bookkeeping and any permit reacquisition wait. Neither
term is a measurement of CPU busy time. Commit-to-GPU-start includes driver
submission and scheduling, and may overlap the commit call itself.

Queue records contain the flush reason, ready/live fits, queued/running CPU
jobs, oldest-ready age, time inside `Next`, and queued/running/ready counts when
results resume. These are state snapshots, not a CPU utilization trace. The
ready-set wait is attributed once if scratch limits split a set into sub-batches.
Result publication time covers copying results, releasing transient backing and
resuming workers after the evaluator returns. Queue waits overlap independent
CPU work and must not be added to whole-encode latency.

`--fit-ready-target=N` and `--fit-flush-us=N` expose private scheduling controls.
The accepted target is 1–256; the deadline is 0–1,000,000 microseconds. They do
not change tree policy, source sampling, predictor ordering or tie breaking.
New named controls select JSON schema 9; old invocations retain their schemas.

## Experiment design

The original binary is frozen before the diagnostic changes. Compare it with
the new binary's disabled, wall-only, wall-plus-trace and stage-plus-trace modes.
The three inputs are the existing 4000 × 3000 RGB8 photographs 1029/1039/1049.
Each configuration receives one warmup followed by interleaved measured runs.

A screening experiment on 1029 varies one control at a time: 1/2/4/8/16 shards,
16/64/128 active properties, 0/500/2000 microsecond deadlines, and ready targets
8/32. Promising choices receive a separate comparison across all three photos;
screening results alone do not establish an encoder speedup. Stage measurements
are attribution evidence, while performance comparisons disable instrumentation.

`tools/modular_reference_learning/run_gpu_histogram_wait_study.py` records exact
commands, source/binary/dependency/input hashes, raw per-batch JSON, AC power and
competing-process observations. It checks complete output hashes, logical learning
work and pinned-libjxl exact decoding. Contended attempts are retained and
excluded. Newly created identical output artifacts use APFS clones to bound
private disk usage. Backend/domain creation, file IO, hashing and independent
decoding are outside the complete private encoder call.

Local artifacts: `build/gpu-histogram-wait-study-20261016/`.

## Wait attribution

The attribution pass has 60 accepted encodes (five variants, three images,
one warmup and three measurements), all preserving complete bytes and exact
decode. Three overlapping attempts were retained and replaced: a peer Metal
test, a weighted-preset capture, and a compiler process.

Ordinary-command trace totals, median milliseconds per image:

| Image | Waiting before GPU starts | GPU execution inside wait | After GPU completion | CPU participation restored |
| --- | ---: | ---: | ---: | ---: |
| 1029 | 589.1 | 703.3 | 218.6 | 0.422 |
| 1039 | 427.2 | 576.1 | 90.4 | 0.208 |
| 1049 | 399.0 | 509.4 | 53.4 | 0.236 |

The first three intervals partition each individual blocking window exactly;
medians of their per-image totals need not add to the median total. They are
overlapping host/device attribution, not additive whole-encode stages. CPU
permit restoration is negligible in this isolated-encode study. Most of the
non-execution interval precedes GPU start, with a smaller but material interval
between GPU completion and the blocking call returning.

The median fraction of deadline-triggered batches is 99.0% / 98.4% / 96.6%.
At result resumption, independent fits are already GPU-ready in 79.9% / 89.8% /
86.1% of batches. This establishes available independent work at those
snapshots; it does not establish how much an asynchronous scheduler would save.
The derived gap between consecutive GPU commands also crosses cohort boundaries
and must not be treated as exclusive fitter scheduling overhead.

Stage-mode collection totals remain 404.1 / 314.2 / 280.5 ms. Validation totals
are 197.2 / 164.8 / 154.8 ms. Stage profiling cuts the pre-execution wait to
295.9 / 193.5 / 170.3 ms while increasing the command-buffer GPU span. Because
stage mode changes encoder boundaries and adds timestamp sampling, this is a
reason to investigate submission topology, not evidence that timestamp sampling
itself is an optimization.

## Validation

Five focused Release suites pass: combined Modular workflows, histogram queue,
Metal histogram evaluator, Metal histogram workflow and Metal resource budget.
The queue check now exercises alternate targets/deadlines and trace snapshots
while preserving exact trees and partition order across worker migration. The
Metal evaluator checks traced and untraced modes, GPU/host clock ordering,
constant-only nodes, exact results, bounded storage, and unchanged publication
on allocation/submission/completion failures.
Four CLI checks reject disabled-profile tracing, zero/oversized ready targets
and an oversized flush deadline before submission.

## Shard and batching screening

The screening pass uses 1029, one warmup and two measured runs per setting.
Everything except the named control stays at its default. Values are medians
in seconds; these are screening observations, not established speedups.

| Setting | Complete encode | GPU batches |
| --- | ---: | ---: |
| 1 shard | 3.102 | 368 |
| 2 shards | 3.141 | 358.5 |
| 4 shards (default) | 3.159 | 322 |
| 8 shards | 3.232 | 287 |
| 16 shards | 3.466 | 271 |
| 16 active properties | 3.178 | 338 |
| 128 active properties | 3.239 | 294.5 |
| Immediate flush | 3.169 | 342 |
| 2,000 microsecond flush | 3.175 | 287 |
| 8-fit ready target | 3.154 | 336 |

More shards and a longer deadline reduce the number of batches without
improving elapsed time here. Conversely, one shard creates more batches but
looks slightly faster in the screen. Batch count alone is therefore an
insufficient optimization objective. Additional shards expand partial
histograms and reduction work and can reduce concurrent property capacity
under the fixed scratch ceiling; they do not reduce total membership visits.

## Fresh complete-call qualification

The two strongest shard settings receive a separate three-image comparison,
with one warmup and three measured runs per variant: 60 accepted encodes.
All use the same learned policy and preserve exact output bytes. The GPU
variants use Metal unique sample rows; the whole-CPU control uses the matching
CPU uniqueness shortcut. This is a private combined-route comparison, not
attribution of a speedup to GPU tree fitting alone.

| Route | 1029 | 1039 | 1049 | Sum of medians |
| --- | ---: | ---: | ---: | ---: |
| Frozen original, 4 shards | 3.164 | 2.352 | 2.233 | 7.748 |
| Candidate, 4 shards | 3.150 | 2.336 | 2.206 | 7.693 |
| Candidate, 1 shard | 3.141 | 2.309 | 2.207 | 7.657 |
| Candidate, 2 shards | 3.141 | 2.320 | 2.210 | 7.671 |
| Whole CPU, unique rows | 3.277 | 2.407 | 2.270 | 7.955 |

One and two shards reduce the sum of medians by only 0.47% and 0.29% against
four shards. Each wins five of nine paired comparisons. One of the one-shard
1039 calls is 23.8% slower than its paired control despite no detected competing
benchmark; it is retained. This small, variable effect does not justify changing
the four-shard default. The queue target, deadline and active-property default
also stay unchanged. GPU fitting remains private and opt-in.

The disabled-mode instrumentation control is similarly variable: the earlier
mode comparison had 1029 medians of 3.250 s for the original and 3.464 s for the
candidate; the fresh comparison above reverses that ordering. These samples do
not establish a stable tracing overhead estimate. The stage-profiled route
changes command encoding and cannot substitute for uninstrumented qualification.

All observations retain ranges, paired ratios and raw artifact references in
the [results manifest](gpu-histogram-collection-and-waits-results.json).

## What the one-shard probe changes

A final 1029 probe interleaves one/four shards with ordinary-command tracing
and stage tracing (one warmup, two measured runs each; 12 accepted encodes).
The table deliberately labels which mode supplies each measurement.

| Measurement, milliseconds per image | 4 shards | 1 shard |
| --- | ---: | ---: |
| Collection, stage mode | 414.3 | 472.9 |
| Prefix scans, stage mode | 52.4 | 36.4 |
| Waiting before GPU starts, ordinary-command mode | 574.8 | 503.2 |
| GPU execution inside wait, ordinary-command mode | 697.1 | 745.0 |

Reducing shards makes collection about 14% slower in this probe while reducing
prefixing and pre-execution delay. It changes the tradeoff between device work,
scratch use and scheduling rather than removing the dominant collection work.
Concurrent CPU progress changes batch composition, so these runs are not a
fixed-batch kernel microbenchmark. Stage and ordinary-command intervals must
not be combined into an additive timing model.

## Next implementation effort

Keep the current defaults. First test multiple compute encoders per command
buffer **without timestamp sampling**, preserving kernels, scratch layout and
one submit/wait per batch. This isolates the encoder-boundary effect suggested
by stage profiling without simultaneously redesigning asynchronous ownership.
Qualify complete bytes, failure draining and uninstrumented complete-call time.

If launch delay persists, test bounded reuse of the dispatch/scratch arenas and
trace the resulting pre-execution interval. Small CPU allocation time alone
does not rule out deferred resource work before GPU start. Batch scratch size
has a moderate positive correlation with launch delay in these captures; that
correlation is a hypothesis generator, not proof of an allocation bottleneck.

For collection, capture representative batches and use targeted kernel
measurements/counters to distinguish local atomic contention from repeated
index/property scans. The shard sweep does not identify either as the root
cause. SIMD aggregation or a reusable bin-partition pass should follow that
evidence. Multiple in-flight batches remain a possible later change: ready work
exists, but the current experiment does not demonstrate its recoverable latency.

## Evidence and reproduction

Across the four passes, **162 accepted encodes** preserve complete bytes and
independent exact decoding. Five overlapping attempts are retained and excluded.
The final audit rechecks source, binary, dependency, input, output and result
hashes. All accepted observations pass the AC and competing-process checks;
ordinary desktop activity remains uncontrolled. Hardware is Apple M4 Pro,
14 CPU cores and 48 GiB RAM. Builds use Apple Clang Release `-O3 -g -DNDEBUG`.

The artifact directory contains the frozen `baseline` and `candidate` binaries,
their CMake caches, source patch/new sources, test logs, variant JSON files,
per-run output/trace records, an analysis script and the final audit script/log.
To reproduce a pass with the current profiler, create a variants JSON containing
objects with `name`, an absolute `binary` path, and an `args` array beginning
with route/buffering/predictor controls. Then run:

```sh
python3 tools/modular_reference_learning/run_gpu_histogram_wait_study.py \
  --variants VARIANTS.json --input INPUT.pam --out NEW_OUTPUT_DIRECTORY --repeats 3
```

Example private profiler invocation for the ordinary-command trace:

```sh
build/release/gjxl_modular_hybrid_profile INPUT.pam NEW_OUTPUT_DIRECTORY \
  auto auto best fit-off 14 14 14 14 14 fitter-gpu 16384 64 \
  --sample-rows=metal-unique --gpu-profile=gpu-wall --gpu-wait-trace=on \
  --histogram-shards=4 --active-properties=64 --fit-ready-target=32 --fit-flush-us=500
```

Use `--gpu-profile=gpu-off` and omit wait tracing for performance qualification.
