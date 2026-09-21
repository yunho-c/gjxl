# Full paper ablation measurements — 21 September 2026

The completed study supports AC candidate fusion, SIMD DCT, and Malta fusion/locality as substantial contributors to encoding performance. Direct-image DCT I/O, EPF/RGB fusion, and reduced AQ synchronization have smaller effects. The resident ACS/AQ handoff does **not** show a consistent latency benefit in this experiment.

The original collection is complete and retained unchanged. A transient timing disturbance in one image's first round affects some small estimates. A separate balanced repeat and two sensitivity analyses preserve the main conclusions; they are reported explicitly below.

## Scope and validation

- Apple M4 Pro, 20 GPU cores, 14 CPU cores, 48 GiB unified memory, macOS 15.6. The encoder used eight participating CPU threads.
- 65 images: 32 CLIC, 24 Kodak, and three Unsplash scenes at each of 12, 24, and 48 MP. The large-image sizes share source scenes.
- Efforts 5 and 8; distances 0.5, 1, and 2; nine arms; three process rounds; seven timed calls per process.
- **10,530 jobs, 73,710 timed calls, and 8,190 intended paired comparisons**, with complete coverage and no unexpected job directories.
- Every timed encode matched its process's prepared output byte for byte. All 8,190 retained paired comparisons were byte-identical, including scalar/SIMD DCT. Retained outputs were also identical across process rounds. This is an observed corpus result, not a proof of numerical equivalence for every possible input.
- External decoding, finite-pixel validation, and SSIMULACRA2 scoring passed. Validation of identical input/codestream pairs was cached outside timing; retained artifact hashes and validation identities were checked in the final audit.
- Kernel-path audits verified the intended variants and AQ counts. Additional checks covered all 1,170 case/round blocks: the handoff used GPU selection with host-known dispatch and additional copied bytes, and split EPF performed the separate conversion once per AQ evaluation.

The original collection ran from 03:35:06 to 10:12:48 UTC, approximately **6 h 38 min**. AC power was recorded at both boundaries of every job. All 392 periodic telemetry samples reported AC power and no recorded thermal/performance warnings. Periodic telemetry began about six minutes after collection started; it is not a GPU-utilization trace and does not establish the cause of the timing disturbance.

Encoder revision: `56a730428428c71a5dea1c9b800d7218fe523e35`. Frozen executable SHA-256: `08f7f410da56315879cd010485cf6e0d84acb6cb83cf39b7f7fb6124c25ba877`. The manifest also records source, build, input, decoder, metric, and environment identities; collector support changes are preserved in the source archive and patch.

## Timing and interpretation

The measured boundary is the complete `EncodeLinearRgbVarDctCodestream` call. Input loading, output writing, decoding, quality scoring, and output comparison are outside it. Each process performs preparation, three warmups, and one untimed warm path audit before the seven timed calls. GPU stage profiling is disabled.

For each comparison, the analysis takes the median of the seven calls, forms a disabled/optimized ratio within each process round, takes the geometric mean across rounds within each image, then the equal-image geometric mean. Ratios above one favor the optimization. The plots show 95% percentile intervals from 2,000 source-scene cluster bootstrap draws; the three resolutions of each Unsplash scene share a cluster. Per-round ranges are retained separately. These intervals describe scene variation conditional on the measured rounds.

The seven comparisons are **conditional**, as specified in [the protocol](paper-ablation.md#comparisons). AC candidate fusion uses the common CPU-greedy reference; DCT locality and arithmetic use the specified split-AC/host-selector references. The residency comparison keeps GPU greedy selection in both arms. The gains cannot be added or multiplied into a combined production speedup. This study does not disable every encoder optimization or measure GPU versus CPU AQ.

## Ablation table: relative encoding time

Each cell is **encoding time with the optimization disabled / encoding time with it enabled**, summarized by the paired geometric-mean method above across all 65 images. Each row's matched reference is normalized to **1.000×**. Values above one mean that disabling the optimization increases latency; values below one mean that the ablated arm is faster. `e` denotes effort and `d` denotes distance.

| Ablation | Reference | e5, d0.5 | e5, d1 | e5, d2 | e8, d0.5† | e8, d1† | e8, d2 |
|---|:---:|---:|---:|---:|---:|---:|---:|
| Restore AQ completion waits | P | 1.012× | 1.010× | 1.011× | 1.011× | 1.020× | 1.013× |
| Restore ACS/AQ host handoff | P | 0.995× | 0.991× | 0.993× | 0.994× | 1.004× | 0.993× |
| Split Malta kernels | P | 1.083× | 1.081× | 1.087× | 1.111× | 1.136× | 1.136× |
| Split EPF/RGB conversion | P | 1.008× | 1.005× | 1.007× | 1.003× | 1.016× | 1.008× |
| Split AC candidate stages | H | 1.429× | 1.438× | 1.460× | 1.215× | 1.222× | 1.246× |
| Replace direct-image DCT with packed I/O | S | 1.016× | 1.018× | 1.021× | 1.019× | 1.016× | 1.023× |
| Replace SIMD DCT with scalar DCT | B | 1.159× | 1.162× | 1.166× | 1.095× | 1.105× | 1.115× |

Matched references, using the arm names in [the protocol](paper-ablation.md#comparisons):

- **P — `production`:** production GPU-greedy, resident ACS/AQ path with fused kernels and direct-image SIMD DCT.
- **H — `host-fused`:** CPU-greedy selection, fused AC candidate evaluation, and direct-image SIMD DCT.
- **S — `host-split-ac`:** CPU-greedy selection, split AC candidate evaluation, and direct-image SIMD DCT.
- **B — `packed-simd`:** CPU-greedy selection, split AC candidate evaluation, and packed SIMD DCT with gather/scatter.

The table uses the **original, all-collected-round estimates**, rounded to three decimal places, from the `cohort=all` rows of [aggregate.csv](../ablation-runs/full-20260921T033431Z/analysis/aggregate.csv). The [detailed ratio table](../ablation-runs/full-20260921T033431Z/analysis/REPORT.md) supplies the corresponding 95% source-scene cluster bootstrap intervals. For example, splitting AC candidate stages at effort 5, distance 1 gives **1.438× [1.402, 1.474]** relative encoding time.

**† Timing disturbance retained:** the effort-8, distance-0.5 and distance-1 columns include the disturbed first round for image 22. Consult the [sensitivity results below](#results-and-sensitivity) and [all-setting sensitivity estimates and intervals](../ablation-runs/full-20260921T033431Z/analysis/sensitivity.csv), especially before interpreting small effects. The post hoc diagnostics do not replace the primary values in this table.

**Suggested caption:** Relative encoding time with each optimization disabled, normalized to its matched optimized reference (1.000×). Entries are equal-image geometric means of paired latency ratios over 65 images; values above one indicate increased latency. Comparisons use the indicated conditional references, so effects cannot be added or multiplied into a combined speedup. † Original estimates retain a timing disturbance; see the sensitivity analysis.

## Results and sensitivity

The following table gives **percentage encode-time reduction at distance 1, as effort 5 / effort 8**. A negative value means the designated optimized arm was slower. Time reduction is `100 × (1 − 1 / ratio)`; it differs from the percentage extra time in the size figure.

| Optimization | Original, all collected rounds | Median-round sensitivity | Whole-image repeat sensitivity |
|---|---:|---:|---:|
| AQ synchronization | 0.95 / 2.01 | 0.95 / 1.33 | 0.94 / 1.29 |
| ACS/AQ residency | −0.92 / 0.38 | −0.85 / −0.51 | −0.92 / −0.51 |
| Malta fusion/locality | 7.52 / 12.00 | 7.52 / 11.20 | 7.53 / 11.21 |
| EPF/RGB fusion | 0.47 / 1.58 | 0.55 / 0.82 | 0.46 / 0.90 |
| AC candidate fusion | 30.48 / 18.19 | 30.47 / 18.34 | 30.49 / 18.42 |
| Direct-image DCT I/O | 1.75 / 1.58 | 1.73 / 1.97 | 1.75 / 2.08 |
| SIMD DCT arithmetic | 13.94 / 9.54 | 14.00 / 9.58 | 13.94 / 9.55 |

Across the three distances, AC candidate fusion reduces latency by approximately 30–31% at effort 5 and 17–20% at effort 8, with its conditional baseline. SIMD DCT contributes approximately 14% and 9–10%, respectively. Malta fusion/locality contributes approximately 8% and 10–12%. These findings survive both sensitivity analyses.

The smaller effects merit more restrained wording. Synchronization removal is about a one-percent effect in the sensitivity analyses; direct-image DCT I/O is about 1.5–2.3%; EPF/RGB fusion is generally below one percent. The resident ACS/AQ path has no measured latency advantage over the GPU-greedy host handoff here. The apparent positive 0.38% effort-8/distance-1 aggregate becomes a roughly 0.5% regression in both sensitivity checks. This is not evidence against every form of device residency, nor a comparison with CPU strategy selection.

At distance 1, AC candidate fusion has a larger relative effect on the larger-image cohorts: disabling it adds roughly 55–62% time at effort 5 and 29–31% at effort 8 for the 12/24/48 MP images. Each such cohort contains only three source scenes, so those estimates have limited generality.

### Timing disturbance and diagnostic repeat

Thirteen job medians exceeded twice the median of their three matching process-round medians. All occurred in round zero, image 22 (`clic2024_test/a36713f1943dac6bc74dea50cadaee6f`), effort 8, distances 0.5 or 1. Their record completion times span approximately 04:01:31–04:03:49 UTC. The affected medians were 2.7–7.4 times their three-round references. Nearby jobs also show smaller variation. AC and thermal telemetry do not identify a cause; no causal explanation for this disturbance is established.

A fresh diagnostic repeated **the entire image**, all six effort/distance settings and all nine arms, over three rounds: **162 jobs, 1,134 timed calls, and 126 paired comparisons**. It used the same frozen executable, source hashes, tools, protocol, and environment. All outputs matched the originals. The large spikes did not recur; the repeated three-round configuration medians were 0.936–1.084 times the corresponding original three-round medians.

The two **post hoc diagnostic** columns above are separate from the original aggregate:

1. Median-round sensitivity uses the median of the three paired round ratios within each image, then the equal-image geometric mean. It uses all three rounds and removes no records.
2. Whole-image repeat sensitivity recomputes the estimate using the complete balanced repeat for image 22 and the original measurements for the other 64 images. It replaces no files and does not select particular arms, samples, or favorable rounds.

The original estimates and plots retain the disturbance. The table and sensitivity intervals should accompany claims about the small effects. Strong fusion/DCT findings are supported; the unqualified raw small-effect point estimates should not be quoted alone as precise benefits.

## Retained artifacts

Original run: `ablation-runs/full-20260921T033431Z/`.

- [Main figure, PDF](../ablation-runs/full-20260921T033431Z/analysis/ablation-effects.pdf) and [size figure, PDF](../ablation-runs/full-20260921T033431Z/analysis/ablation-by-size.pdf). SVG and PNG versions are alongside them; both figures were visually checked after correcting the size figure's colorbar margin.
- [Full aggregate report](../ablation-runs/full-20260921T033431Z/analysis/REPORT.md), [all aggregate estimates and intervals](../ablation-runs/full-20260921T033431Z/analysis/aggregate.csv), and [all 8,190 paired observations](../ablation-runs/full-20260921T033431Z/analysis/paired.csv).
- [Coverage and artifact audit](../ablation-runs/full-20260921T033431Z/analysis/audit.json), [timing review](../ablation-runs/full-20260921T033431Z/analysis/timing-review.json), and [sensitivity estimates and intervals](../ablation-runs/full-20260921T033431Z/analysis/sensitivity.csv).
- [Original manifest](../ablation-runs/full-20260921T033431Z/manifest.json), frozen encoder, source archive/patch, CMake cache, raw job records, codestreams, and append-only ledger.
- [Balanced repeat audit](../ablation-runs/verify-image22-20260921T101806Z/analysis/audit.json) and [repeat manifest](../ablation-runs/verify-image22-20260921T101806Z/manifest.json).

The controller directory retains exact commands, hashed analysis code, power/thermal telemetry, and the original plotting code before layout/notice changes. The figure manifest hashes the final exports and their source data. Data, infrastructure, and reports remain in the isolated `experiment/paper-ablation` worktree. The frozen run manifests and source snapshots identify the exact code used for collection, independently of later commits recording the collection support, analysis tools, and report.
