Increasing frontier admission beyond 512 still helps, but much less than the earlier 128-to-512 change. On the same prepared cohort, 512 → 1024 reduces complete fitting by **5.54%** with the original shader and **3.77%** with descriptor references, while adding **133.25 MiB** of managed backing. The counter experiment also gives a stronger explanation for the occupancy-manager target: changing copied descriptors to device references nearly removes stack-write traffic and restores the target, yet actual occupancy remains low. Descriptor handling contributes substantially to the cache pressure; it does not explain the entire CPU/GPU fitting gap.

This follow-up to the [occupancy study](gpu-frontier-occupancy.md) used `feat/gpu-resident-fitting` at `f04a7b55` plus private diagnostic changes, on the **M4 Pro, 20 GPU cores, 48 GiB, AC power**. The input is photograph **1029's first archived prepared cohort: 42 fits, 5,285,712 rows, and 35,878 resulting tree nodes**. These times cover complete fitting of one cohort, not a full image encode or a sum across images. [Structured results](gpu-frontier-high-admission-results.json) preserve the summaries and raw-artifact hashes; [raw artifacts](../build/frontier-high-admission-20261017/) retain binaries, libraries, sources, captures, counters, manifests, and analysis scripts.

The boundary includes validation, scratch management, GPU input upload, node setup, split selection, partitioning, queue management, canonicalization, and publication. Sample preparation, CPU input restoration, backend/domain/worker-pool creation, and equality checks are outside the timer. The CPU control fits identical samples with 14 participants.

| Admitted nodes per round | Original shader: median (range) | Descriptor references: median (range) | Scratch | Peak managed backing |
|---|---:|---:|---:|---:|
| 512 | 0.827 s (0.814–0.849) | 0.773 s (0.759–0.778) | 133.25 MiB | 359.84 MiB |
| 768 | 0.786 s (0.773–0.806) | 0.765 s (0.742–0.782) | 199.88 MiB | 426.46 MiB |
| 1024 | 0.782 s (0.761–0.799) | 0.744 s (0.728–0.770) | 266.50 MiB | 493.09 MiB |

Each cell has six measured calls across three process blocks, with one warmup and two measured calls per CPU/GPU route in each process. Admission order varies across blocks; shader-variant order alternates, and CPU/GPU call order alternates within each process. All observations are retained. AC power was checked throughout, and the existing competitor guard detected no other codec benchmark, build, or profiler. The CPU median across 36 measured calls is **0.421 s**, with a 0.390–0.509 s range. The fastest GPU configuration still takes **1.77×** that median time.

The original shader's 512 → 1024 reductions are 4.08%, 5.77%, and 7.26% by process block; the reference variant's are 4.22%, 2.67%, and 3.35%. The additional original-shader improvement from 768 → 1024 is only **0.52%** by ratio of overall medians, smaller than the observed ranges. Descriptor references improve the same-cap medians by **6.61%, 2.67%, and 4.87%** at 512, 768, and 1024, respectively. Those comparisons all use fresh controls from this experiment, not the previous study's timings.

Rounds decline from 78 to 56 to 45; submissions decline from five to four to three. The median difference between complete-call time and summed command-buffer GPU spans is approximately 30–37 ms, depending on configuration. Most time remains in GPU execution. Increasing admission reduces batch boundaries but does not increase the 128 threads assigned to any one node. The 42 roots and early small frontiers remain particularly underfilled.

Peak managed backing excludes the immutable archive, driver/backend/profiler allocations, and unrelated process memory; it is not RSS. The operation's ordinary scratch ceiling remains 64 MiB, and the backend's ordinary ceiling remains 256 MiB. A diagnostic-build-only environment switch raises the latter to a bounded 320 MiB for this experiment. The 1024 configuration actually allocates 266.5 MiB of scratch. A separate check without the switch requested 1024 and correctly planned/executed **983**, demonstrating the existing 256 MiB clipping. The replay probe now plans with the actual device limits and the runner verifies both planned and observed maximum admission.

Four fresh captures replayed the first submission at 512 and 1024, with both shader variants. Each submission contains 16 frontier rounds. The command requested `--gpu-state default --exec overlapping`; embedded profile metadata resolves the replay GPU state to **medium** for all four captures. These instrumented replays are separate from the uninstrumented timing table. Node mixtures and append order can differ across captures and during replay, so these are not matched per-node throughput comparisons. Shader cost sampling covers one GPU core, while timeline counters can include other GPU activity.

| Counter average over the captured submission | 512 original | 512 references | 1024 original | 1024 references |
|---|---:|---:|---:|---:|
| Occupancy-manager target | 82.14% | 98.18% | 73.52% | 99.92% |
| Actual compute occupancy | 12.84% | 13.16% | 14.28% | 14.23% |
| L1 eviction measure | 5.24 | 0.68 | 7.18 | 0.33 |
| Stack L1 writes | 230.54 GiB/s | 0.28 GiB/s | 205.87 GiB/s | 0.35 GiB/s |
| Target samples below 50% | 19.85% | 1.07% | 32.78% | 0.09% |

The profiler labels the eviction counter “Measure of L1 Cache Line Evictions”; it is not reported here as a percentage or a cache-miss rate. The target and actual occupancy are different counters, not interchangeable measures of useful computation. Improving the target does not guarantee that the application supplies enough runnable work or that other execution constraints disappear.

The source change in the reference variant is limited to passing/reading immutable `FitDescriptor` values through device references rather than copying them. Arithmetic, policies, and split decisions are unchanged. The compiler again reports **1,360 → 144 spilled bytes** and **2,136 → 1,903 instructions**, with **96 temporary registers** in both variants. At 512, reported L1 register residency falls from 22.12% to 3.26%; buffer residency rises from 43.03% to 61.27%. Static register count alone would miss this change in runtime cache use. The experiment changes descriptor representation, instructions, and accesses together; it does not isolate spilling as the only mechanism.

Time-aligned counters narrow down the target reduction further. In the original shader's 512-node capture:

| Counter within the selected target windows | Target below 50% | Target at least 90% |
|---|---:|---:|
| L1 total residency | 99.51% | 73.38% |
| L1 eviction measure | 17.00 | 0.84 |
| Compute launch limiter | 87.12% | 2.33% |
| L1 register residency | 27.50% | 18.89% |
| L1 threadgroup residency | 29.72% | 11.12% |
| L1 buffer residency | 42.20% | 43.19% |
| Last-level-cache limiter / utilization | 16.36% / 16.31% | 4.12% / 4.09% |
| MMU limiter / utilization | 11.98% / 11.65% | 3.63% / 3.59% |

Low-target windows coincide with nearly full L1 residency, increased evictions, and launch limiting. Register and threadgroup allocations grow on top of a substantial buffer working set. Last-level-cache and MMU limiters remain close to their utilization counters; these windows do not show large additional stalls in those blocks. The 1024 original-shader capture shows the same pattern: 99.56% L1 residency and a 16.20 eviction measure below a 50% target, with small last-level-cache/MMU limiter–utilization gaps. This is consistent with on-chip working-set pressure, rather than evidence that external bandwidth or address translation is the principal cause of the target reduction. Apple's [occupancy diagnostic workflow](https://developer.apple.com/videos/play/tech-talks/111374/) motivates these comparisons.

Residency alone is insufficient: some initial shader intervals also have nearly full L1 residency while the target remains 100% and evictions are low. Traffic and eviction behavior must be considered alongside allocation. Similarly, bandwidth alone does not identify the offending allocation: the original shader has substantial stack writes even in high-target windows. The descriptor-reference experiment is useful because it changes a specific source-level representation and then demonstrates lower traffic/evictions and a higher target.

The core counters use approximately 31 µs samples; last-level-cache, MMU, and external-bandwidth counters use approximately 129 µs samples. The analysis classifies each counter sample at its own timestamp using an interpolated target, restricting results to their common time span. These are conditional sample averages, not additive shader costs or per-node timings. Repeating the L1 comparison using only samples with combined vertex/fragment occupancy below 1% retains the pattern: at 512 original, low-target residency is 99.61% and the eviction measure is 16.13, versus 73.62% and 0.55 at high target. Background GPU activity therefore does not explain away the association, although these counters cannot establish exclusive process ownership. The 1024 reference capture has only five samples below a 50% target; that rare subset is not used to infer a remaining bottleneck.

The pipeline API resolves the threadgroup-memory uncertainty. Both variants return **14,592 bytes** from `staticThreadgroupMemoryLength()`. The node dispatch adds no dynamic threadgroup allocation. Both report a maximum of 1024 threads per group and a 32-thread execution width; the kernel actually dispatches 128 threads per group. The device's 32,768-byte maximum is a **per-threadgroup API limit**, not the amount of storage available per GPU core. Dividing it by 14,592 would not establish a number of simultaneously resident groups. The shader profiler still reports zero threadgroup bytes, so its field remains unsuitable for this analysis.

The resulting diagnosis is more specific than before:

- Copied descriptors create substantial avoidable on-chip traffic/cache pressure. Replacing the copies largely removes the observed occupancy-target depression on these captures, and improves complete fitting without increasing managed backing.
- The occupancy manager is **not the sole explanation for low overall occupancy**. With references at 1024, its target is effectively 100%, but actual compute occupancy remains about 14% and complete fitting remains slower than the CPU control.
- Wider admission has diminishing returns and an expensive storage slope. This sweep does not establish a universal optimum, nor a precise hard register/threadgroup residency limit. Node size imbalance, small early frontiers, and the long monolithic per-node execution remain relevant; this study does not assign percentages to histogram collection, scoring, atomics/barriers, or partitioning.

The next implementation should carry the descriptor-reference change into correctness and full-pipeline qualification, then proceed with tiled histogram work and compact scratch. Tiling can distribute a large node across groups; compact scratch can support useful concurrency at lower memory cost. The 1024 result is a reference point for that work, not a new default. Simply raising admission further would require another storage/contract change and is a lower priority than improving work distribution and per-node storage. No end-to-end speedup, CPU-fitter win, or additive CPU/GPU throughput claim follows from this study.

The subsequent [tiled-frontier implementation and qualification](gpu-frontier-tiled-histogram-fitting.md) applies the descriptor-reference change and retains this study as its pre-implementation baseline.

All **128 complete fitting calls / 5,376 tree comparisons** across the sweep, clipping check, and diagnostic captures passed exact equality. The focused `metal_modular_frontier_histogram` CTest passed. At the end of this admission study, production shaders and routing defaults were unchanged. Full-image codestream/decode qualification was not repeated within this study because no experimental shader or admission policy was promoted; the subsequent qualification is linked above.

Reproduction uses `run_frontier_occupancy_study.py PROBE SAMPLES.gjfs OUTPUT --power AC --caps 512 768 1024 --scratch-mib 320 --diagnostic-large-scratch --variant control=CONTROL.metallib --variant reference=REFERENCE.metallib`. The private build must enable `GJXL_BUILD_MODULAR_FRONTIER_DIAGNOSTICS`; `GJXL_FRONTIER_DIAGNOSTIC_LARGE_SCRATCH=1` selects the 320 MiB backend ceiling. Diagnostic outputs now include pipeline properties in `pipeline.json`. The artifact directory retains the executed runner, changed sources, sample/library hashes, the capture driver, counter exporter, profiler driver, and `analyze.py`. An initial counter query during collection was rejected; the accepted exports came from reopening the embedded profile after collection completed. All owned GPU-debug sessions were terminated.
