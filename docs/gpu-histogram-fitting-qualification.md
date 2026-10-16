# GPU histogram fitting: integration and qualification

The exact Metal fitter is integrated and available through private qualification controls. **Keep ordinary CPU fitting as the public default.** The complete-call results below do not establish a consistent speed advantage for enabling it automatically.

This is an exact-policy implementation: all accepted encodes preserve complete compressed bytes and decode exactly. Sampling, quantization, predictor selection, split thresholds, tie order, transforms and entropy policy are unchanged.

## Measurement conditions

The qualified integration is `bac7ad5` on `feat/gpu-histogram-fitting`, based on CPU consolidation `3aca01c` and `feat/modular` through `9d8a011`. Measurements used Apple M4 Pro (14 CPU cores, 48 GiB), macOS 27.0.1, AC power, an 8 GiB execution-domain budget, a 14-participant CPU ceiling and requested limits of 14 for learning, entropy modeling, selection and CPU tokenization. Weighted-feature CPU production retains its four-participant limit.

Times cover the complete private encoder call, including sample preparation, GPU uploads/allocations, dispatch/waits, CPU fitting/partitioning, fallback, entropy work and teardown. They exclude backend and execution-domain creation, file I/O, independent decoding and hashing. Worker launch/join inside the encoder remains included. Each encode uses a new process/backend; prepared data is not reused across encodes. These are complete encoder-call measurements, not CLI process runtime or kernel timings.

Each of eight input/policy cases has one warmup and three measured repetitions for five variants, with alternating order: **160 accepted encodes**. Warmups count toward correctness but not medians. All accepted activity logs contain no detected competing encoder/build/controller; any overlapping run is retained and excluded. Ambient desktop activity is not controlled, and every measured observation is retained. There are 0 rejected attempts in this final matrix.

The corrected process monitor uses `ps -ww -axo pid=,ppid=,args=` and follows Python ancestors through gaps between child jobs. Earlier pilot timings used a truncated `comm` field and are rejected. The CPU reference replay was repeated with the corrected monitor: its summed median overhead is -0.04%, with exact trees and bounds unchanged.

## Three 12 MP photographs

Each image is RGB8, 4000 × 3000. The hybrid route uses automatic existing Metal feature/token preparation. Its three fitter columns isolate ordinary CPU fitting, resumable CPU fitting with the new cohort boundary, and GPU evaluation for nodes with at least 16,384 distinct rows. The frozen CPU control is the baseline `after` binary; both it and the new CPU column use the same private call boundary.

| Image | Frozen CPU | Current CPU | Hybrid / CPU fitter | Hybrid / resumable CPU | Hybrid / GPU fitter | Bytes |
|---|---:|---:|---:|---:|---:|---:|
| 1029 | 3.544 s | 3.555 s | 3.710 s | 3.836 s | 3.754 s | 10,507,251 |
| 1039 | 2.594 s | 2.574 s | 2.984 s | 2.809 s | 2.907 s | 7,138,537 |
| 1049 | 2.383 s | 2.373 s | 2.474 s | 2.517 s | 2.567 s | 7,015,756 |
| **Sum of medians** | **8.521 s** | **8.501 s** | **9.168 s** | **9.163 s** | **9.228 s** | **24,661,544** |

The ordinary CPU route changes by -0.23% against the frozen control. GPU fitting changes the hybrid pipeline by +0.66%, and by +0.72% against its resumable CPU control. The GPU-fitting hybrid remains +8.55% relative to the entire current CPU pipeline. Per-image ranges and all observations are retained in the results JSON; small differences should not be read as robust speedups.

## Smaller, shared and 16-bit cases

| Case / policy | Dimensions / format | Current CPU | Hybrid / CPU fitter | Hybrid / GPU fitter | Bytes |
|---|---|---:|---:|---:|---:|
| photo-macan, auto/best | 500 × 500, 3ch/8bit | 0.28421 s | 0.28039 s | 0.39013 s | 252,179 |
| alpha-ria, auto/best | 500 × 500, 4ch/8bit | 0.34154 s | 0.34190 s | 0.49944 s | 228,118 |
| palette-358, auto/best | 72 × 72, 4ch/8bit | 0.00961 s | 0.01163 s | 0.01252 s | 1,241 |
| alpha16-ramp, auto/all | 513 × 257, 4ch/16bit | 0.37850 s | 0.37082 s | 0.57857 s | 21,392 |
| photo-macan, buffered/all | 500 × 500, 3ch/8bit | 0.77010 s | 0.75364 s | 1.50991 s | 248,954 |

Explicit GPU mode can exercise shared fits, whose CPU remainder currently uses serial resumable evaluation. Ordinary shared CPU fits can evaluate properties in parallel. The private automatic experiment therefore retains ordinary CPU fitting for shared/small cohorts; forcing GPU mode on these cases is diagnostic and is not a recommended policy.

## What remains expensive

| Image | GPU-route learning wall | GPU operation wall | GPU batches | GPU nodes | Indexed rows | Peak committed MiB |
|---|---:|---:|---:|---:|---:|---:|
| 1029 | 2.680 s | 1.692 s | 302 | 3024 | 139,467,918 | 1814.4 |
| 1039 | 1.951 s | 1.377 s | 267 | 2662 | 118,275,800 | 1594.9 |
| 1049 | 1.705 s | 1.171 s | 249 | 2374 | 99,973,026 | 1528.7 |

GPU operation time includes host packing, allocation, submission and waiting. It overlaps CPU work in other fits, so it must not be added to summed CPU fitting time. Indexed rows include repeated visits in descendant nodes. Committed memory includes backing plus unbacked reservations; it is not RSS or driver memory.

The isolated 1029 pilot tried 1,024 / 4,096 / 16,384 / 65,536-row cutoffs. The single measured call after each warmup took 11.118 / 4.690 / 3.830 / 3.937 seconds. The smallest cutoff issued over 2,300 batches. These pilot observations selected the 16K experimental setting; they do not establish a universal crossover or an optimal configuration.

The implementation already uses bounded threadgroup-local integer histogram tiles and lets CPU tails progress during GPU evaluation. The next useful investigation is dispatch-level GPU attribution: separate validation, histogram collection, prefixing, exact floating-point scoring and ordered selection; then measure batch occupancy and allocation/encoding overhead. Use that evidence to tune shard/slot counts or reduce dispatch boundaries. Preserve exact costs and complete bytes. Moving additional tree-control/partition work to GPU should follow a demonstrated bottleneck, rather than being assumed to fix this result.

## Correctness and failure qualification

- Eight focused Release tests pass, covering the CPU contract/reference/queue, parallel learning, Metal operator/integration, complete workflow and resource budget. Final routing tests additionally cover the private automatic eligibility rules.
- The Metal operator retains 140 exact batches, 3,025 exact property results and seven complete tree comparisons, including Best/All, weights, ties, wide static bins, extreme populations and malformed requests.
- Integrated tests compare exact trees/tokens, complete ANS and prefix files, and independent decoding across RGB/RGBA, 8/16-bit, local/shared and buffered cases. CPU cutoffs, scratch/cohort admission, concurrent execution domains and partial launches are covered.
- Fitting-specific device allocation, submission/completion and host allocation errors propagate without CPU retry. A CPU tree-capacity error after GPU evaluation drains both sides and releases resources.
- ASan/UBSan passes for the parallel learning and migrating-fit queue tests. Focused TSan runs pass for resumable shared/local fitting, profiles/admission, migrating row/result ownership, failures and concurrent domains.
- Dedicated Metal operator and histogram workflow tests pass with shader validation enabled.

The broader TSan parallel-learning CTest timed out at 180 seconds; it is not counted as passing. The shader-instrumented full Metal workflow also failed its feature-routing/submission-count assertion; only the ordinary full workflow and dedicated instrumented fitter tests are claimed as passing. The complete logs, including these limits and superseded attempts, are retained.

## Reproduction and artifacts

Build `gjxl_modular_hybrid_profile`, then use an output directory that does not exist:

```sh
build/release/gjxl_modular_hybrid_profile INPUT.pam OUT \
  auto auto best fit-off 14 14 14 14 14 fitter-gpu 16384 64
```

Replace `fitter-gpu` with `fitter-cpu` or `fitter-resumable` for the two hybrid controls. Replace the first `auto` with `cpu` and use `fitter-cpu` for the entire CPU pipeline. `fit-on` enables native diagnostic envelopes, which include paused time for migrating fits and must not be interpreted as additive CPU stage times.

The measured binary is `ad63078b3edfea744cda8fcd9b4eb09431c94f6c24381e4dc636ffbeff4659f6` at integration `bac7ad5`. The later private-default change from 1K to 16K uses the same explicit 16K configuration measured here; the public default remains CPU. The frozen baseline binary is `4e191d5ffb22dc2e5416284b7c244589c9058d22971ae9d02663b119a33af2ed`.

The local archive is `build/gpu-histogram-fitting-implementation-20261016/`: `workflow-qualification/` contains every file/result/activity log, `workflow-queue-pilot-isolated/` contains the accepted cutoff pilot, and `step2-overhead-isolated/` contains the corrected CPU reference replay. `qualify_workflow.py` records commands, power state, source snapshots, binary/input/output hashes, alternating order and rejection decisions.

Committed measurements, medians, raw-artifact hashes and test logs are in [gpu-histogram-fitting-workflow-results.json](gpu-histogram-fitting-workflow-results.json). The operation and ownership contract is in [gpu-histogram-fitting.md](gpu-histogram-fitting.md). No new libjxl throughput comparison is claimed here.
