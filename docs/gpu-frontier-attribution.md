The frontier fitter's slowdown is primarily GPU execution inside the node kernel. The new measurements support changing how that kernel distributes work, rather than prioritizing CPU command-wait overhead. A controlled metadata-copy experiment removes most reported spills but leaves almost all of the CPU/GPU performance gap.

The subsequent [occupancy and admission study](gpu-frontier-occupancy.md) tests this diagnosis on AC: admitting 512 rather than 128 nodes reduces complete fitting of the same prepared cohort by 32.95%, at a 99.94 MiB increase in managed backing. It also identifies increased launch backpressure at wider admission. Those matched measurements are separate from this battery study.

This investigation used `feat/gpu-resident-fitting` at `4d799089`, with the private diagnostic additions described below, on the M4 Pro (20 GPU cores, 48 GiB), on battery. It follows the [repeat measurements](gpu-frontier-repeat.md) and [continuous-pipeline comparison](newer-hybrid-pipeline.md). It investigates **1029, a 4000 × 3000 RGB8 photograph**. It is not a new three-image performance qualification. [Structured results](gpu-frontier-attribution-results.json) contain the observation summaries and artifact hashes.

We saved the actual prepared samples from the first frontier cohort of the full-resolution encode: **42 fits, 5,285,712 rows, and 35,878 resulting tree nodes**. The CPU and GPU then fitted those identical samples, including all nodes, partitioning, scratch management, and final tree publication. The GPU timing includes its input upload. Backend/domain/worker-pool creation and CPU input restoration are outside the timer; CPU restoration had a 2.95 ms median. Tree equality is checked after timing.

| Complete fitting of that cohort | Median | Measured range |
|---|---:|---:|
| CPU, 14 participants | 0.409 s | 0.392–0.449 s |
| GPU frontier | 1.207 s | 1.197–1.213 s |

These are six observations per route after one warmup, with alternating order, in one process. **The GPU took 2.95× the CPU time.** These times cover one cohort, not one image or the sum of multiple images. This comparison removes feature preparation, input-readiness scheduling, and tokenization as explanations for this particular gap. GPU command spans accounted for about 97.7% of the GPU fitting median.

Measured peak managed backing inside the fitting domain was 162.45 MiB for CPU and 259.90 MiB for GPU. Both use a common immutable archive outside that domain; these figures exclude that archive, backend/driver resources, and profiler memory. The CPU figure includes its mutable input copy. They are not process RSS or total-machine memory measurements.

The full-image Metal System Trace independently locates the bottleneck. Of the captured encode's 7.708 s of complete frontier fitting, command-buffer GPU spans occupied 7.536 s and the union of the fitter's **Active GPU intervals was 7.437 s**. Feature preparation, sample rows, and token generation accounted for approximately 21.6, 16.7, and 16.5 ms of their own active GPU intervals. Intervals were joined to this process's command-buffer IDs and overlapping events were counted once. The encode produced the established 10,507,251-byte output and decoded exactly.

Desktop GPU work overlapped 1.313 s of the fitter's active intervals, chiefly WindowServer. That is temporal overlap, not a measurement of how much it slowed the fitter; it cannot be subtracted from the timing. No competing codec benchmark was detected during the uninstrumented fitting comparisons. The captured encode's 8.853 s total is diagnostic and is not substituted for the earlier uninstrumented end-to-end measurements.

We also replayed individual command-buffer captures through Apple's GPU profiler. The early capture comes from the full encode; the middle and late captures replay the same archived cohort. “Node cost” below is the profiler's sampled cost attribution, **not an additive wall-time breakdown**. The profiler sampled one of the 20 GPU cores. Timeline utilization counters can include desktop activity.

| Captured submission, zero-based | Node-kernel sampled cost | Reported compute occupancy | Reported ALU utilization |
|---|---:|---:|---:|
| 0: roots and early descendants | 99.88% | 7.50% | 3.46% |
| 5: middle of the cohort | 99.54% | 10.17% | 6.43% |
| 17: final submission | 99.37% | 11.18% | 6.11% |

Queue setup accounts for under 0.7% of sampled cost in these captures. The GPU has work, but the node kernel achieves low occupancy and arithmetic utilization. The early capture's external-memory bandwidth was 42.41 GiB/s, with the tool reporting 8.97% of peak bandwidth. These observations do not suggest saturated external memory bandwidth; they do not rule out local memory latency, atomic contention, or synchronization bottlenecks.

The compiler reports **96 temporary registers and 1,360 spilled bytes** for `gjxl_histogram_frontier_nodes`. Those are static shader statistics, not bytes transferred during a fit. The full-image trace also recorded 1,936 spill events of 1,360 bytes, matching its 121 submissions × 16 node dispatches, plus the same number of 272-byte events for the queue-setup kernel. The separate replay identifies which shader owns each value. The profiler's zero-valued threadgroup-memory field is not used: the source explicitly declares roughly 14 KiB of histogram counters, plus other shared state.

A diagnostic alternate metallib changed the immutable `FitDescriptor` from a copied value to a `device const` reference, including the property and partition helpers. It changed no arithmetic, traversal policy, thresholds, or tie decisions. Reported node spills fell **1,360 → 144 bytes** and static instruction count **2,136 → 1,903**. Early-capture stack L1 write bandwidth fell from 197.25 to 0.43 GiB/s, while temporary registers remained 96 and reported occupancy remained low (7.68%). These counter windows are diagnostic comparisons, not a controlled whole-program bandwidth benchmark.

| Uninstrumented metadata experiment | Complete-fitting median | Range |
|---|---:|---:|
| Original copied descriptors | 1.244 s | 1.195–1.272 s |
| Device references | 1.199 s | 1.174–1.283 s |

Six process pairs alternated order; each process warmed both CPU and GPU before its measured calls. The ratio of medians improves by **3.54%**, but the median paired reduction is only **1.70%**, and two pairs regress. This is a promising small optimization, not a robust full-encoder speedup. Most of the gap remains even after eliminating most descriptor-related spills. The alternate shader is retained only as an experiment under `build`; production kernels and routing are unchanged.

Source inspection explains why the larger problem remains:

- [The node kernel](../src/gpu/metal/kernels/modular_frontier_histogram_fitting.metal) assigns one 128-thread group to an entire node, then processes properties, predictors, and histogram tiles in nested loops. It rereads the node's row indices for every histogram tile. A large node therefore cannot use multiple threadgroups for that work.
- Histogram increments use shared atomics, including a single extra-bit counter per predictor during parent collection. Histogram construction, dense bin-prefix scans, exact entropy scoring, single-lane selection, and partitioning share one long-lived kernel and its resource requirements.
- [Partition discovery](../src/gpu/metal/kernels/modular_resident_histogram_fitting.metal) uses three threadgroup barriers per 128-row chunk. Each dispatch must finish before the next set of ready nodes runs. Having about 125 nodes admitted per round does not mean all their groups remain busy for the entire round: small nodes can finish while larger ones hold up the boundary.
- [The CPU fitter](../src/codec/modular/histogram_tree_learning.cpp) physically partitions its columns for contiguous reads and skips unused property bins in its candidate scans. The GPU gathers through a row permutation and still performs dense histogram initialization/prefix work. Both skip inherited constant properties and invalid split thresholds; the issue is not that the GPU scores every empty threshold.

The measurements establish the node kernel as the target and confirm one source of unnecessary work. They **do not yet give trustworthy percentages for histogram collection versus entropy scoring versus partitioning**, or separate the costs of atomics, barriers, register pressure, and unequal node lengths. Low occupancy alone does not prove that concurrent unrelated GPU work would provide additive throughput.

The next implementation should split node processing into GPU-resident phases, starting with parent summaries and joint-histogram collection, followed by exact scoring/selection and partition/publication. This makes the internal costs directly measurable and permits large nodes to spread row tiles and properties across multiple groups. Keep a bounded scratch pool, canonical arithmetic and tie order, and GPU-owned queue state; phase boundaries need no CPU decision or wait. Small nodes should subsequently use compact bin work and size-aware batches. Increasing buffer reuse or tuning CPU waits is a lower priority for this executor.

The descriptor-reference change is a small candidate to qualify alongside that work, but its current evidence does not justify presenting it as the solution. Any promotion still needs exact trees, unchanged full codestream bytes and decode, bounded memory, and uninstrumented complete-encode comparisons on 1029/1039/1049, including the continuous CPU-fitting pipeline control.

The added private tooling is reusable: `GJXL_BUILD_MODULAR_FRONTIER_DIAGNOSTICS=ON` enables selected-cohort sample archives and bounded submission captures. `GJXL_FRONTIER_DIAGNOSTIC_DIR`, `GJXL_FRONTIER_DIAGNOSTIC_COHORT`, `GJXL_FRONTIER_DUMP_ROWS=1`, and `GJXL_FRONTIER_CAPTURE_SUBMISSIONS=0,5,17` select the output; Metal captures also require `MTL_CAPTURE_ENABLED=1`. The ordinary build excludes the hooks. The archive is a bounded, native-endian research format, not a public interchange format.

`gjxl_modular_resident_fitting_probe samples.gjfs result.json 16 6 128 cpu-frontier` replays the archive; its fit count comes from the archive. `GJXL_FRONTIER_PROBE_METALLIB` optionally selects an experimental library. All **70 complete fitting invocations / 2,940 tree comparisons** passed, both full-image diagnostic encodes retained the established bytes and exact decode, and the focused `metal_modular_frontier_histogram` CTest passed after the tooling changes. The descriptor variant's validation covers this one real cohort; it has not undergone full-image qualification.

Raw manifests, captures, logs, frozen executables, alternate shader sources, and scripts are under [`build/frontier-attribution-20261017`](../build/frontier-attribution-20261017/). `build-metadata-variant.py`, `run-replay-study.py`, `profile-capture.py`, `analyze-timeline.py`, and `summarize.py` preserve the experiment and analysis. The bulk GPU-counter XML export timed out and is partial, so it was excluded. Initial gpudebug queries returned an empty cached profile tree despite successful collection; reopening the embedded profiles produced the accepted extractions. Failed attempts are retained separately. No line-level phase attribution is claimed from them. Apple's [Metal symbol-file documentation](https://developer.apple.com/documentation/metal/generating-and-loading-a-metal-library-symbol-file) and the installed `gpudebug(1)` manual describe the profiling inputs and controls.
