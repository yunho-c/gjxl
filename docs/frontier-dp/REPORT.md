# Frontier DP Metal experiment

This report preserves the first captured-cost measurement. The subsequent
[selector comparison](SELECTORS.md) adds complete encodes, rectangle DP,
GPU greedy, and the current integration decision.

The original-bank exact solver works on the 20-core Apple M4 Pro. At 512 threads per tile, the complete device sequence takes 1.21 ms for a 3 MP photograph, 3.22 ms for padded 4K synthetic input, and 9.28–9.88 ms for three 24 MP photographs. Its elapsed time is comparable to the existing CPU selector. This establishes standalone selection feasibility; it does not establish a complete-encoder speedup or a rate-quality gain.

The experiment is on `experiment/gpu-frontier-dp`, based on `bb7b714e858ffe7668a84c8588745a99b7015f39`, in an isolated worktree. The primary checkout was not modified. These first measurements preceded the experimental checkpoint commit.

**Measured scope.** Costs came from ordinary effort-7, distance-1.2, fully resident Metal encoding with SIMD transforms and fused-tuned residual inverse transforms. The capture hook observes the actual GPU scores and the existing selected map. It records raw scores and the CPU's frozen binary32 DCT8 policy costs. Replay prepares the same policy costs on Metal and verifies every bit before solving.

The measured sequence is GPU policy-cost preparation and checked integer normalization, frontier DP, traceback, and an on-device wide-fallback dispatch. Candidate inputs are already packed by tile and resident. Scoring, capture/repacking, uploads, graph construction, pipeline compilation, CPU oracle work, downstream AQ metadata preparation, and codestream encoding are excluded. The original seven family buffers will need a GPU gather or direct-indexing implementation during integration.

**Results.** Each GPU entry is the median of three independent-process medians, each containing seven retained samples after three warmups. Image order rotates between processes; thread-count order rotates between samples. GPU durations use completed command-buffer timestamps. Probe wall times include command creation/encoding, submission, and completion wait, without output copying or validation.

| Input | Coding blocks | Tiles | GPU sequence, 512 threads | Probe wall time | Existing CPU selector | Proxy-cost reduction |
|---|---:|---:|---:|---:|---:|---:|
| Alpine lake, about 3 MP | 266 × 177 | 782 | 1.211 ms | 1.387 ms | 1.164 ms | 0.1345% |
| Synthetic padded 4K | 480 × 270 | 2,040 | 3.223 ms | 3.347 ms | 2.902 ms | 0.9046% |
| Alpine lake, 24 MP | 750 × 500 | 5,922 | 9.883 ms | 10.047 ms | 9.119 ms | 0.1436% |
| Forest stream, about 24 MP | 715 × 525 | 5,940 | 9.799 ms | 9.950 ms | 11.376 ms | 0.0306% |
| Campus interior, about 24 MP | 752 × 500 | 5,922 | 9.282 ms | 9.454 ms | 10.418 ms | 0.2901% |

The CPU control is the actual `FindAcStrategyGridFromResidentCandidateCosts` implementation on the same cost table, including grid allocation and export. It uses two warmups and seven samples in the capture process. It excludes scoring, wait/readback, and dense cost-table construction. Its grid is checked against the encoder's selected grid after each sample. CPU and GPU timings therefore represent different execution boundaries and are not a complete-pipeline A/B speedup.

The photographic proxy improvements are 0.03–0.29%. The larger synthetic improvement should not be presented as typical photographic compression improvement. No images were encoded using exact-selected maps in this experiment, and no matched-quality or visual comparison was performed.

**Thread-count sensitivity.** Complete device-sequence times:

| Input | 64 threads | 128 threads | 256 threads | 512 threads |
|---|---:|---:|---:|---:|
| Alpine 3 MP | 2.906 ms | 2.065 ms | 1.396 ms | 1.211 ms |
| Synthetic 4K | 7.955 ms | 5.023 ms | 3.738 ms | 3.223 ms |
| Alpine 24 MP | 22.374 ms | 14.614 ms | 11.172 ms | 9.883 ms |
| Forest 24 MP | 23.028 ms | 15.267 ms | 11.693 ms | 9.799 ms |
| Campus 24 MP | 22.053 ms | 14.363 ms | 10.992 ms | 9.282 ms |

Normalization uses at most 256 threads; the table varies the DP threadgroup size. The 512-thread configuration was fastest among those tested. This is an empirical result for this shader and device, not an occupancy attribution. The pipeline reports a 32-thread SIMD width, 1,024 maximum threads per group, and 24,528 bytes of static threadgroup storage including alignment.

**Stage diagnostics.** For Alpine 24 MP at 128 DP threads, policy preparation took 0.094 ms, DP with device-resident saved decisions took 14.830 ms, separate traceback took 0.116 ms, and the no-work wide-fallback dispatch took 0.059 ms. DP dominates. These diagnostic kernels write an additional device decision array, and separate commands have their own scheduling overhead; their times must not be added to decompose the fused complete-kernel result.

**Correctness.** Every one of the 20,606 captured tiles matched an independent full-state CPU DP evaluator in exact cost and deterministic strategy/anchor bytes at all four thread counts, in both fused and split-trace modes. Metal-generated policy bits and normalized integers matched the CPU reference. Buffer guards remained intact. All 20,606 captured tiles fit the checked uint64 path. There were 6,087 strict proxy improvements; the remaining 14,519 tiles tied the existing layout, and their maps also agreed on these inputs.

The synthetic GPU suite covers all 64 tile dimensions with 12 cases each: zero/tied costs, area-proportional ties, randomized costs, near ties, subnormals, largest finite binary32 values, wide exponent ranges, negative zero, NaN, negative costs, and infinity. Of 768 cases, 126 exercise the 320-bit device fallback and 192 are rejected as invalid. All valid cases match the CPU oracle. The suite passed with Metal API and shader validation enabled. Invalid scores are distinguished from finite range exceptions.

The CPU oracle retains all state values and does not reuse the GPU's slot schedule. Geometry is derived from the live candidate bank and checks the full-tile counts: 258 candidates, 12,200 states, 35,511 edges, and 1,282 reusable values. The earlier independent Python review also checked all partial dimensions against a separate bitmask cover oracle on small tiles; this experiment's GPU suite adds implementation validation.

Four existing checks passed after the experiment: `ac_strategy_search`, `metal_ac_strategy_search`, `ac_strategy_storage_plan`, and `ac_strategy_host_storage_plan`. `git diff --check` also passed.

**Numerical contract.** Costs are frozen binary32 policy scores. The DCT8 discount is applied once, using the same host-computed binary32 multiplier as the existing selector; entropy multipliers are already present in the scores. The GPU factors out the tile's common power of two and admits uint64 only when every possible 64-rectangle path is bounded below 2^64. Zero has an explicit case. A separate 320-bit solver handles finite exceptional ranges on device, using a fixed pool of 32 workers and device scratch. Invalid scores are flagged and rejected by the host validation interface.

Ties use fixed candidate order. Preserving the production greedy layout on objective ties is not implemented as a general policy; it happened naturally for the captured ties. Wide-fallback performance is unqualified because no photographic tile needed it.

**Scope limits.** This is the original 258-candidate bank. Current ordinary effort 10's dense bank is deliberately rejected by capture; its larger graph needs a different storage design. The standalone output is a tile-local strategy/anchor map. It does not build the grouped anchors, coefficient offsets, and CfL records currently prepared on the CPU by AQ reconfiguration. That host boundary remains in the production encoder. CUDA was not implemented or measured.

The host was an interactive macOS 15.6 system with a 20-core M4 Pro GPU and 48 GiB RAM. A process snapshot is retained; desktop activity was not disabled. No other experiment GPU jobs or builds were run concurrently with the retained capture/timing protocol. Per-process GPU medians varied, for example, from 9.245 to 9.828 ms for Campus 24 MP. These are repeated engineering measurements, not an isolated-hardware production qualification.

**Interpretation and next gate.** The original M4 Pro millisecond estimate was plausible, toward its upper end at 256 threads. Exact selection alone is not consistently faster than the current CPU selector. Avoiding its handoff may still help, but device metadata preparation and resource scheduling must be included before claiming that.

The next useful experiment is to feed the exact maps into complete encodes through a diagnostic integration, compare identical-map CPU/GPU results, and measure rate at matched SSIMULACRA2 and Butteraugli quality. The small photographic proxy gains make that evidence necessary before investing in a production residency refactor or changing an effort default.

**Artifacts and reproduction.**

- [Summary](../../build/frontier/results/kernel-probe-20260917/summary.json)
- [Manifest, source/binary hashes, machine details and exact commands](../../build/frontier/results/kernel-probe-20260917/manifest.json)
- [Frozen measured sources](../../build/frontier/results/kernel-probe-20260917/source/)
- [Metal validation result](../../build/frontier/results/kernel-probe-20260917/selftest-metal-validation.json)
- [Capture implementation](../../tools/frontier_dp/capture.cpp)
- [CPU graph and exact oracle](../../tools/frontier_dp/graph.h)
- [Metal kernels](../../tools/frontier_dp/frontier.metal)
- [Probe](../../tools/frontier_dp/probe.cpp)
- [Sequential runner](../../tools/frontier_dp/run_experiment.py)

The run directory also retains both byte-identical captures per input, every process's raw timings, input hashes, and logs. Capture-run whole-encoder timings are contaminated by diagnostic repeated CPU selection and file I/O and must not be used as encoder benchmarks. Source and binary hashes were checked again after measurement.

Build and run:

```sh
cmake -S . -B build/frontier -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_BUILD_TESTS=OFF -DGJXL_BUILD_BENCHMARKS=ON \
  -DGJXL_ENABLE_LIBJXL_REFERENCE=OFF \
  -DGJXL_BUILD_FRONTIER_EXPERIMENT=ON
cmake --build build/frontier \
  --target gjxl_encoding_benchmark gjxl_frontier_probe -j 8
build/frontier/gjxl_frontier_probe --self-test
python3 tools/frontier_dp/run_experiment.py \
  --output build/frontier/results/NEW-UNUSED-DIRECTORY
```

The runner expects the existing local photographic corpus and ImageMagick; `--corpus` overrides its location. It refuses an existing output directory, runs only the bounded five-input protocol, and freezes the measured source content. The CMake option defaults to OFF.
