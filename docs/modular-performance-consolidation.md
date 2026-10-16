# Modular performance consolidation

2026-10-16. Consolidation starts from local `feat/modular` at `9d8a011`.
The candidate is assembled and qualified in `integrate/modular-performance`
before staging is advanced. The existing CPU learner remains the byte/rate
oracle and the public automatic backend remains CPU.

## Gathered work

| Source tip | Integrated work | Default behavior |
| --- | --- | --- |
| `feat/gpu-histogram-fitting` / `7728f95` | Continuous local CPU scheduling, resumable exact fitter, batched Metal node evaluation, SIMD prefix scans and profiling | Ordinary CPU fitting; GPU fitting remains private and opt-in |
| `feat/modular-sample-rows` / `e5bf689` | Lower native sample materialization overhead; selected CPU/Metal rows, deferred materialization and guarded uniqueness proof | Native materialization improvements apply; raster rows and ordinary dedup remain default |
| `perf/modular-cohort-pipeline` / `317c7ff` | Persistent fitting queue, overlap across feature owners, protected completion admission, 8/16 Mi-sample windows | Cohort schedule and 8 Mi-sample limit remain default |

The known-population experiment was checkpointed as `d692041`, with its ANS
phase exporter/evidence in `d4b761d`. The weighted-preset experiment was
checkpointed as `90628ce`. These implementation changes stay on their own
branches. Their old measurements describe the checkpoint snapshots, not this
consolidation. Population-call fusion and historical research are left parked.

## Integration decisions

The continuous scheduler can retain several feature owners in one fitting
cohort. A single sample-row source for that cohort would be incorrect. Each
stream now carries its own borrowed owner alongside its feature views, and
each resumable fitting request carries that source through admission and
preparation. Owners remain immutable and survive every consuming fit and token
operation. Views and source pointers are cleared before their owner is released.
The metadata storage bound includes the added pointer.

The pipeline uses the same per-stream source and sample options. GPU fitting
also accumulates sample-row diagnostics, including uniqueness and deferred
materialization, instead of losing those counters at its private worker boundary.

Pipeline plus resumable/GPU fitting still returns `Unsupported`; combining those
schedulers is a separate implementation effort. No experimental public routing
or automatic speed crossover is introduced by this merge.

The profiler accepts composable named controls while retaining the previous
sample-row, GPU-profile and pipeline positional tails. For example:

```sh
build/release/gjxl_modular_hybrid_profile INPUT.pam OUT auto auto best \
  fit-off 14 14 14 14 14 fitter-gpu 16384 64 \
  --sample-rows=metal-unique --gpu-profile=gpu-stage

build/release/gjxl_modular_hybrid_profile INPUT.pam OUT auto auto best \
  fit-off 14 14 14 14 14 \
  --sample-rows=metal-unique --learning-schedule=pipeline --feature-window=16
```

The standalone reference bridge now links the required Metal archives/frameworks
when its production build enables Metal. A fitter test also explicitly names
the CPU profile type to avoid the new GPU profile's ambiguous name.

## Qualification

Nineteen focused Release suites passed. They cover histogram fitting/preparation,
both queues, resource/worker failures, parallel selection/learning/entropy/tokens,
public API, ordinary and palette hybrid workflows, pipeline admission, Metal
sample rows, and Metal fitting operators/workflows.

The new combined suite compares full stream selections, sampling, exact trees,
complete bytes and pinned-libjxl decoding. It crosses selected CPU/Metal rows
with ordinary/resumable/GPU fitting, then exercises pipeline fitting, mixed
CPU/Metal feature owners, and a multi-cohort image. It checks GPU row population,
retained-row diagnostics, uniqueness controls, actual GPU fitter execution and
explicit rejection of pipeline plus GPU fitting.

The pinned-libjxl bridge passes 257 native tree fixtures and 121 preparation
fixtures with uniqueness proof both disabled and enabled. Four focused checks
also pass with Metal API validation after rebuilding with the matched timing
flags: the combined suite, public API, Metal node evaluator, and Metal sample
rows. This is not a new exhaustive allocation sweep, ThreadSanitizer run or
Metal shader-instrumentation qualification; earlier branch-specific limitations
remain documented in their reports.

## Complete-call comparison

The matched capture compares staging `9d8a011` with the gathered candidate.
Both use Apple Clang 21 Release `-O3 -g -DNDEBUG`, the same pinned libjxl decoder,
14 participants, an 8 GiB managed domain, and AC power on Apple M4 Pro. Three
4000x3000 RGB8 photos (1029/1039/1049) receive one discarded warmup and three
interleaved measured calls per configuration. Five smaller policy cases cover
photo, alpha, palette, 16-bit/all-predictor, and buffered shared-tree encoding.

Timing includes the complete private encoder call and cleanup, excluding backend
and domain creation, file IO, and independent decoding. Detailed fitter/row/GPU
stage probes are disabled. Worker intervals can overlap and are not additive
latency or on-CPU measurements. The driver checks AC power and competing codec
or compiler work, retains excluded attempts, and validates sources, binaries,
inputs, dependencies, complete output hashes and learning-work counts.

The configurations distinguish unchanged-default CPU/Metal routes from the
explicit CPU uniqueness, GPU-fitting/Metal-row, and pipeline/Metal-row experiments.
Their combined results do not attribute savings to individual optimizations.

All **119 encodes** preserve complete codestream hashes and logical learning
work, and decode exactly with pinned libjxl. This comprises 21 warmups, 63
measured photo calls and 35 small correctness calls. No competing-work attempt
was excluded. Values below are seconds, with a median of three per image.

| Configuration | 1029 | 1039 | 1049 | Sum of medians | Max managed backing, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline-cpu | 4.057 | 2.967 | 2.960 | 9.985 | 816.0 |
| integrated-cpu | 3.479 | 2.496 | 2.365 | 8.339 | 810.5 |
| baseline-metal | 4.427 | 3.804 | 3.114 | 11.345 | 743.3 |
| integrated-metal | 3.692 | 2.682 | 2.356 | 8.730 | 741.1 |
| cpu-unique | 3.414 | 2.351 | 2.231 | 7.997 | 810.0 |
| gpu-fit-rows | 3.161 | 2.278 | 2.153 | 7.592 | 955.9 |
| pipeline-rows | 3.038 | 2.137 | 1.980 | 7.155 | 782.9 |

The unchanged-default CPU route decreases **16.5%** (9.985 to 8.339 s);
the unchanged-default Metal route decreases **23.0%** (11.345 to 8.730 s).
These are elapsed reductions on this corpus/session, not additive branch
speedups or a general CPU/GPU crossover. The combined opt-ins have different
memory costs and remain disabled by default. In particular, the GPU-fit/row
combination raises maximum managed backing compared with ordinary Metal.

[Structured results](modular-performance-consolidation-results.json) retain
all observations and per-image ranges. The final provenance check initially
stopped because its Python hash helper received a string instead of a `Path`,
after all 119 encodes had completed. The original error manifest is retained.
The archived independent finalizer rechecked every source, binary, dependency,
input and output hash before marking the capture passed; the runner is fixed.

Eight additional CLI encodes check old positional tails, composable named
controls, resumable fitting and actual GPU stage-profile output; two invalid
combinations are rejected. The final profiler-only fixes do not change any
production source or the frozen binary used for the timing comparison.

The qualified staging base is then merged back into the GPU fitting, sample
row and cohort branches, and into the two separately checkpointed experiments.
The local `branch-synchronization.json` records their resulting refs and clean
states. The experiment branches remain ahead solely by their own work and
integration fixes; neither experiment is included in staging.

## Reproduction and local artifacts

`tools/modular_reference_learning/qualify_performance_consolidation.py` freezes
both profilers and source archives, records commands and power/activity snapshots,
and retains every raw result. Verified duplicate outputs use APFS clones.

The local artifact root is
`/Users/yunhocho/GitHub/gjxl-modular/build/modular-performance-consolidation-20261016/`.
It contains the preflight branch/dirty-file manifest, build and test logs, merge
previews, and the matched capture. The ten pre-existing untracked staging files
are preserved separately from all integration commits.
