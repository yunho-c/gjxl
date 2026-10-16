# GPU histogram fitting: phase attribution and parallel prefix scan

Histogram collection and prefix sums dominate the timestamped GPU fitter. The floating-point split-cost estimator is a small part of its device time. Its results reproduce the native CPU bit patterns. The first optimization replaces serial integer prefix scans with SIMD scans, keeping the histogram layout, floating-point arithmetic, tree policy and compressed bytes unchanged. GPU fitting remains an opt-in private route.

The complete private encoder call improves **3.42%** against the original GPU
fitter and **4.37%** against the hybrid pipeline's CPU fitter on the three
qualified photographs. It remains **2.71% longer** than the whole CPU pipeline.
Keep GPU fitting opt-in; this study does not justify a general routing change.

## What the profiler measures

The private `gjxl_modular_hybrid_profile` tool accepts a final `gpu-off`, `gpu-wall` or `gpu-stage` argument after the fitting cutoff and cohort size. Optional shard and active-property counts follow it. Supplying these controls selects JSON schema 8; existing invocations retain their schemas.

- `gpu-off` retains ordinary submissions and does not read the profiling clock or allocate a diagnostic graph.
- `gpu-wall` reads the completed command-buffer GPU duration without counter sampling or extra encoder boundaries, and partitions the synchronous evaluator's interior into validation/planning, allocation, packing, recording/submission, waiting, readback, profile resolution and transient release. Final fixed-size publication and the caller's bookkeeping remain outside these intervals.
- `gpu-stage` adds a compute encoder and timestamp pair for each validation, collection, prefix, cost and selection dispatch. All scratch waves still use one command buffer and one submit/wait. It also records the wall phases.

Each batch records its node/row/task counts, scratch waves, active slots, largest bin/symbol counts, scratch bytes, dispatch counts, GPU stage durations and command-buffer GPU span. Records are written into a caller-owned bounded span, with explicit dropped-record detection. The qualification tool allocates that span before timing, outside codec admission. Backend stage/context arrays and recorded/resolved profile graphs use managed diagnostic backing, with an additional backend-specific bound composed into the evaluation reservation. Opaque Metal counter and driver allocations retain the existing driver exclusion.

Publication is failure-atomic for both results and the batch profile. A submission that has started is drained before backing is released; a diagnostic failure does not silently retry on CPU. GPU stage profiling changes encoder boundaries and has measurable overhead. Its timestamps are attribution evidence, not an additive decomposition of uninstrumented encoder wall time. In particular, GPU work overlaps CPU fitting in other jobs, and wall waiting includes scheduling/synchronization as well as execution.

## Initial attribution

The three inputs are `photo-large-1029`, `photo-large-1039` and `photo-large-1049`, each 4000 × 3000 RGB8. The initial experiment alternates off/wall/stage modes, with one warmup and three measurements per image/mode: 36 accepted encodes. All complete codestreams match within each image and decode exactly. The corrected process monitor detected no competing builds or encoders in these accepted observations. Desktop activity remains uncontrolled; the raw ranges are retained.

Median timestamped stage totals, in seconds:

| Image | Validation | Collection | Prefix sums | Split-cost scoring | Selection |
|---|---:|---:|---:|---:|---:|
| 1029 | 0.222 | 0.419 | 0.382 | 0.028 | 0.049 |
| 1039 | 0.178 | 0.330 | 0.315 | 0.020 | 0.040 |
| 1049 | 0.162 | 0.271 | 0.283 | 0.020 | 0.036 |

Collection plus prefixing account for approximately 72.6% of these stage totals; split-cost scoring accounts for approximately 2.5%. In wall-only captures, waiting accounts for 92–94% of the evaluator's instrumented interior. Packing takes about 52–88 ms per image, allocation 14–23 ms and recording/submission 11–33 ms. Immutable-row preparation, measured separately, takes about 48–55 ms per image. Buffer reuse may help, but these measurements favor kernel work first.

In stage mode, median command-buffer GPU spans are 1.164 / 0.929 / 0.810 s, versus caller waits of 1.687 / 1.237 / 1.116 s. That gap includes queueing, completion delivery and host rescheduling; this experiment does not isolate each cause. The final wall-mode captures add ordinary command-buffer timing so this gap can be observed without extra encoder boundaries.

Instrumentation controls also show variability: sums of per-image medians are 10.101 s uninstrumented, 10.671 s with wall instrumentation and 10.345 s with GPU stages. Individual stage-mode changes range from roughly -6% to +13%; this does not imply that collecting timestamps intrinsically accelerates a kernel. Encoder boundaries, concurrent CPU progress and ambient load all affect the complete call. The optimization therefore requires a separate interleaved comparison with instrumentation disabled.

## Implementation

Here, a prefix sum is the cumulative histogram below each candidate split threshold. The original prefix kernel assigned a histogram column to one GPU thread. That thread visited every bin and shard serially; a few of those same threads subsequently scanned extra-bit and occupancy columns. The replacement assigns each column to a SIMD group. Lanes sum shards independently for their bins, perform an inclusive integer prefix scan, and broadcast the tile total as the next tile's carry. Extra-bit and occupancy columns follow the same mechanism.

The dispatch uses the pipeline's actual SIMD width, validated as a divisor of the 128-thread group. Padding lanes contribute zero but participate in every scan and broadcast, so nonmultiples of the SIMD width and wide static properties are covered. Integer sums remain exact under the validated population bounds. The change adds no scratch backing and leaves the later floating-point scorer and ordered winner selection untouched.

## Qualification

The final comparison has one warmup and three measured repetitions for seven variants on each photograph, with alternating order: **84 accepted encodes**. It includes uninstrumented before/after, timestamped before/after, the whole CPU route, the hybrid CPU-fitter route and the candidate wall-only profiler. Together with attribution and the pilot, **128 accepted encodes** preserve complete bytes and decode exactly. Eight overlapping attempts were retained and excluded (six pilot, two final); one pilot overlap was our final profiler rebuild, the others were peer activity. Separate benchmark controllers were allowed to finish before measurements resumed. No competing build/encode/controller was detected in accepted calls, but ambient desktop activity and changing system load remain uncontrolled. All observations and ranges are retained.

The [results manifest](gpu-histogram-fitting-profile-results.json) records per-run observations, paired ratios, hashes and test logs. The reference binary is frozen before the SIMD change; the candidate uses the same policy, cutoff of 16,384 rows, four histogram shards, at most 64 active property slots, 64 MiB scratch ceiling and 64-fit cohorts.

Measurements use the M4 Pro, 14 CPU cores, 48 GiB RAM, AC power, an 8 GiB execution domain and requested limits of 14 for learning, selection, entropy modeling and CPU tokenization. Weighted-feature CPU production retains its four-participant limit. The complete-call boundary includes worker launch/join, uploads, allocations, submissions, waits and teardown. Backend/domain creation, file I/O, exact independent decoding and hashing are outside it. Every encode uses a fresh process/backend.

Uninstrumented complete-call medians, in seconds:

| Image | Original GPU fitter | SIMD GPU fitter | Hybrid / CPU fitter | Whole CPU | GPU change |
|---|---:|---:|---:|---:|---:|
| 1029 | 3.699 | 3.584 | 3.817 | 3.556 | -3.12% |
| 1039 | 2.742 | 2.634 | 2.786 | 2.533 | -3.94% |
| 1049 | 2.550 | 2.465 | 2.476 | 2.364 | -3.31% |
| **Sum of medians** | **8.990** | **8.682** | **9.079** | **8.453** | **-3.42%** |

Every one of the nine paired uninstrumented comparisons improves, by 2.22–5.37%.
These are three-repeat, single-device observations, not a general speed guarantee.
For example, 1029's original GPU measurements span 3.541–3.758 s and the
candidate spans 3.462–3.585 s. The interleaved comparison, rather than differences
between earlier experiments, is the basis for the reported improvement.

Timestamped prefix-sum totals:

| Image | Original | SIMD scan | Reduction |
|---|---:|---:|---:|
| 1029 | 349.1 ms | 51.8 ms | 85.2% |
| 1039 | 279.5 ms | 33.8 ms | 87.9% |
| 1049 | 262.9 ms | 29.6 ms | 88.7% |

The new GPU stage totals are approximately 55% collection, 29% validation,
6% prefixing, 3% split-cost scoring and 6% selection (rounding prevents an exact
100% sum). Ordinary wall-mode captures expose a further substantial wait gap:

| Image | Command-buffer GPU span | Caller wait | GPU batches, uninstrumented |
|---|---:|---:|---:|
| 1029 | 0.684 s | 1.306 s | 317 |
| 1039 | 0.561 s | 1.009 s | 279 |
| 1049 | 0.501 s | 0.917 s | 259 |

The wait and GPU columns are overlapping measurements, not additive stages.
Batch counts can change with concurrent CPU progress. GPU node counts and total
indexed rows remain identical across the before/after variants; all final
runs have zero fitting admission fallbacks. Complete file sizes remain
10,507,251 / 7,138,537 / 7,015,756 bytes.

The next targets are histogram collection and the command wait gap. A focused
CPU scheduling/completion trace and batch-occupancy experiment should establish
how much of that gap can be removed. Validation is also now a material device
cost. Allocation and split-cost scoring are lower priorities. Reassess
these boundaries after integrating the concurrently developed CPU/sample and
weighted-feature improvements; they were not merged into this pinned study.

The dedicated operator test passes 146 exact batches, 3,121 exact property results, seven complete trees and 19 failure checks. The focused tests cover exact CPU/GPU property results and trees, complete workflow byte parity and independent decoding, scratch limits, all-constant nodes, a partially populated final SIMD tile, wide static bins, extreme weighted populations, failure-atomic profile publication, managed diagnostic bounds and resource release. Dedicated operator and histogram workflow tests also run with Metal shader validation.

The implementation is `6bfe02f`. The measurement source base is `d9a9850`, including `feat/modular` through `9d8a011`; runtime modifications are preserved in each frozen binary's source patch. The before binary SHA-256 is `f1d0d3865bbcf4358cfded7804a7a81e425f50898f4f90e7a68898f71747842e`; the final candidate is `bba9ff8e436d327d2db48bc0bbfbe7065b7e880d732f7e681e0fc1bba1c38038`.

Local raw evidence is retained under `build/gpu-histogram-fitting-profile-20261016/`, including frozen binaries, source patches, commands, per-encode results/files, process activity, power observations and test logs. Competing builds in other worktrees are allowed to finish; overlapping measurements are retained as rejected attempts and replaced.

## Reproduction

Build `gjxl_modular_hybrid_profile` in the existing Release/Metal build. For one
stage-attribution capture, supply a new output directory:

```sh
build/release/gjxl_modular_hybrid_profile INPUT.pam NEW_OUTPUT_DIRECTORY \
  auto auto best fit-off 14 14 14 14 14 \
  fitter-gpu 16384 64 gpu-stage 4 64
```

Use `gpu-off` for performance comparisons, or `gpu-wall` for ordinary-command
GPU duration and host intervals. Use route `cpu` plus `fitter-cpu` for the whole
CPU control, or route `auto` plus `fitter-cpu` for the hybrid CPU-fitter control.
The frozen before/after runners, source patches and full per-batch JSON are in
the local evidence directory; the tracked results manifest records their hashes.
