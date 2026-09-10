# Historical Metal kernel tuning study

This report preserves the launch-bound, rectangular AC fusion and psycho
attribution study against `473ad7a482305d091bad8e01420876139e4c9237`. Its numbers
describe the recorded source and build snapshots, not a fresh measurement of
the current branch. Do not combine these ratios with the previous study's
`62d6175` baseline.

The accepted implementation was committed separately as
[rectangular AC fusion, dc44e4f](https://github.com/yunho-c/gjxl/commit/dc44e4f586c9d0c23e42409d42e1c1775e7f1103)
and [fine Butteraugli profiling, fee24d1](https://github.com/yunho-c/gjxl/commit/fee24d17376cdecc249b2d150da3acec7f3ac886).
Their focused notes are [AC fusion](metal-rectangular-ac-fusion.md) and
[Butteraugli profiling](metal-butteraugli-profiling.md). The rejected launch-bound
implementation and builds were discarded; their results remain here to explain
that decision.

Raw evidence is in `build/kernel-tuning/evidence`, with frozen source exports
and Release builds under `build/kernel-tuning`. The device is an Apple M4 Pro,
with 20 GPU cores and 48 GiB memory, running macOS 15.6 and Xcode 26.3.
The queried threadgroup memory limit is 32,768 bytes.

The [result ledger](metal-kernel-tuning-results.json) preserves paired changes,
fine-stage medians, source/runtime identities, and hashes of the raw evidence.
Its source hashes include historical tools that were subsequently discarded.
Absolute paths identify the original local run; the raw build and evidence
directories are not bundled with this report.

## Final complete-workflow results

Fully-resident Metal, distance 1.2, effort 7; seven alternating process pairs.
Negative changes mean less time. These compare the complete final implementation
to `473ad7a`, including the ordinary host-code changes for shared profiling bodies.

| Workload | Parent ms | Candidate ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Padded 4K | 223.317 | 221.282 | -0.61% | 7/7 |
| Padded 1080p | 68.310 | 67.665 | -0.94% | 4/7 |
| Kodak 17 | 22.203 | 22.168 | -0.79% | 5/7 |
| Planter 4K | 227.767 | 226.366 | -0.76% | 6/7 |
| Synthetic 128×96 | 9.546 | 9.763 | +3.46% | 2/7 |

The larger workloads show a modest improvement. The tiny-image cohort is noisy
(-23.52% to +24.74% across pairs) and trends slower. A separate focused cohort,
with nine pairs, five warmups and fifteen samples, still gives +1.08% paired
whole-call change, with mixed directions; its quantization phase changes -0.77%.
No small-image speedup is claimed, and a small residual regression cannot be
excluded. This focused result does not replace the final table.

The 4K candidate spends 176.412 ms in quantization and 40.357 ms in CPU codestream
encoding. AC tokenization is 14.027 ms, entropy optimization 8.022 ms, and section
writing 9.800 ms. These remain substantial CPU costs, separate from the shader
work. The ordinary balanced path already uses the direct tokenization/model
selection path; this is not evidence of exhaustive repeated entropy search.

## Launch-bound screen

Apple describes lowering the maximum threadgroup size as a possible way to
improve compiler register allocation and occupancy. That is a hypothesis to
measure, not evidence that these GJXL kernels spill registers.
[Apple's Metal compute guidance](https://developer.apple.com/videos/play/tech-talks/10580/)
provides the motivation; the following results come from this checkout.

The screen changed only MSL compiler bounds. One variant supplies the actual
dispatch size; another permits twice that size. It covers the tuned AC loss
kernels, DCT16 candidate kernel, EPF variants, resident L2 reduction variants,
and fixed Malta variants. Generic Malta stays at 1024; the already-1024-thread
low/medium blur is unchanged. No arithmetic or host launch geometry changes.
Queried pipeline limits confirmed that the attributes took effect.

Five alternating pairs, three warmups and seven measured calls per process:

| Compiler bound | 4K whole call | 1080p whole call | 128×96 whole call |
| --- | ---: | ---: | ---: |
| Actual dispatch size | -0.27% | +1.04% | +0.99% |
| Twice dispatch size | -0.23% | -1.87% | -10.92% |

The whole-call screen does not establish a consistent kernel improvement.
In the separate three-pair GPU-stage screen, the looser variant's total stages
change +0.89% at 4K and -0.19% at 1080p. Small-image samples vary substantially,
including stages whose shaders were unchanged. The production compiler bounds
therefore remain at their defaults; the tiny-image percentage is not a claimed
optimization. Both variants pass three targeted Metal tests and guarded
reduction/Malta/EPF/EPF-linear comparisons under Metal validation.

The rejected launch-bound implementation, generator and build exports were
discarded at the user's request. Measurement logs and shader hashes remain as
historical evidence. The rejected initial MSL attribute placement and its
compiler errors are also recorded in those logs.

## Small rectangular AC fusion

The 16×8 and 8×16 kernels retain each candidate's X/Y/B coefficients in
threadgroup memory through forward DCT, residual generation, inverse DCT, and
weighted loss. Each channel retains the split path's matrix arithmetic and
reduction tree. A threadgroup/device barrier publishes Y coefficients and the
quant norm before chroma residuals consume them. The scalar cross-channel cost
finalizer and CPU merge/tie rules are unchanged.

Shapes here follow the shader's rows×columns convention. The 16×8 kernel uses
192 threads; 8×16 uses 96. Both report 9,984 static threadgroup bytes. Each removes
one dispatch and a device coefficient round trip. They are selected independently
on devices supporting Apple family 9, with SIMD width, launch-size and memory
checks. Other shapes and inverse modes retain their existing paths.

The isolated fusion screen precedes the profiling change. Three alternating
process pairs, two warmups and three measured samples per process:

| AC stage | 4K parent → fused ms | Paired change | 1080p paired change |
| --- | ---: | ---: | ---: |
| 16×8 | 5.404 → 4.341 | -19.67% | -20.37% |
| 8×16 | 5.491 → 5.046 | -8.29% | -4.13% |
| All AC families | 37.632 → 36.066 | -4.25% | -3.65% |

The initial five-pair whole-call screen changes -1.38% at 4K, -0.23% at 1080p,
and +0.65% at 128×96. These are separate measurements from the GPU-stage table.
The scratch allocation plan is unchanged because the larger transform families
still need the shared device scratch. This is not a peak-memory reduction.

## Profiling boundaries

Psycho construction now has eight dependency-ordered stages at each scale:
`opsin`, `low_medium`, `high_x`, `high_y`, `medium_b`, `suppress_x`, `ultra_x`,
and `ultra_y`. Their ordinary dispatch counts are 1, 1, 2, 2, 2, 1, 2, and 2.
The Opsin stage also includes any required three-dispatch expansion or
subsampling. Ultra Y includes the existing fused raw-mask producer.

Distorted stages use `butteraugli.psycho.main.*` / `butteraugli.psycho.sub.*`,
with their former coarse names as group IDs. Reference preparation uses
`frontend.prepare_aq.reference.main.*` / `.sub.*`; each scale additionally has
a `mask` stage containing two blur dispatches and erosion. Its submission/group
ID remains `frontend.prepare_aq.reference`.

Ordinary and profiled encoding share the same phase bodies. The profile storage
plans include the extra stages and longer IDs. The comparison tool builds coarse
aggregates after the all-stage sum, preventing double counting, and compares
coarse/fine builds at their common boundary. Raw JSON retains every fine stage.
Separate encoder boundaries can affect timings, so these are instrumented
attribution measurements, not a decomposition of ordinary wall time.

At padded 4K, the final three-pair cohort gives the following phase medians.
The AQ columns sum the two feedback iterations within each raw sample; reference
preparation runs once. Units are milliseconds.

| Phase | Reference main | Reference sub | AQ main | AQ sub |
| --- | ---: | ---: | ---: | ---: |
| Opsin plus input preparation | 1.360 | 1.043 | 2.676 | 2.128 |
| Low/medium | 4.232 | 0.918 | 7.965 | 2.080 |
| High X | 1.185 | 0.272 | 2.395 | 0.676 |
| High Y | 1.194 | 0.264 | 2.551 | 0.538 |
| Medium B | 0.989 | 0.251 | 2.071 | 0.507 |
| Suppress X | 0.433 | 0.132 | 0.871 | 0.264 |
| Ultra X | 1.225 | 0.215 | 2.647 | 0.485 |
| Ultra Y plus raw mask | 2.137 | 0.366 | 3.976 | 0.860 |
| Reference mask blur/erosion | 1.711 | 0.334 | — | — |

The 33-tap low/medium phase accounts for about 15.3 ms across preparation and
feedback, making it the largest individual psycho-construction phase. Ultra Y
plus raw-mask production contributes about 7.4 ms. These combined costs sum
within each raw sample before taking medians, so they can differ from sums of
the table's individual medians. These identify the next
places to investigate; they do not establish a bandwidth or register-spill cause.
The studied 16×64 low/medium tile computes horizontal blur for 96 rows to cover
its halo. Reducing that duplicated work trades against threadgroup storage,
occupancy and global intermediate traffic. Previous coarsening/halo experiments
in the earlier study did not establish a further win.

A separate coarse-versus-fine comparison holds the rectangular shader fusion
constant. At 4K, the sum of instrumented GPU stages changes -0.02%; the resident
command-buffer span changes +0.30% and its host operation span +0.36%. At 128×96,
those latter spans increase 11.60% and 12.14%, respectively: fine instrumentation
has a meaningful cost on tiny workloads. The corresponding unprofiled whole-call
comparison is +0.06% at 4K and +0.62% at 128×96, with mixed pair directions.

## Qualification and costs

- Full Release suites: 121/122 for both revisions. The sole inherited CPU
  `quantization_pipeline` failure is unchanged: actual `0.24919039011001587`,
  expected `0.24914586544036865`. Initial stale one-stage/three-dispatch test
  assumptions were corrected; the final candidate suite is complete. The baseline
  suite was reused only after matching every baseline executable/metallib hash.
  This historical run used the subsequently discarded `--reuse-baseline-tests`
  helper; the committed qualification workflow runs both suites.
- The new probe passes 320 bitwise cases normally and 320 under Metal API/shader
  validation. It checks padded strides, 1/2/7/33/257 candidates, both quant-norm
  sources, eight data/error patterns, per-channel rates, losses, final costs,
  guards, and the absence of writes to suppressed coefficient scratch.
- Existing probes pass 1,344 coefficient cases, 56 image-DCT cases and 432 EPF
  cases. Eight selected Metal tests pass with validation enabled. Both revisions
  pass all 22 pinned decoder conformance fixtures.
- All 56 corpus/policy encode pairs are byte-identical. Three decoded pairs and
  24 retained-output decoded pairs match. Eight resource scenarios cover changed
  images, mixed batches, concurrent callers, admission limits and trimming.
- Seven ASan/UBSan tests pass. Leak detection is disabled, with the existing
  narrow metal-cpp null-call suppression; these host sanitizers complement the
  separate Metal shader validation. Ordinary and profiled resident AQ scores,
  quant fields and block distances now pass exact equality checks. Profile tests
  also cover reference stage counts and charged storage on six geometries,
  including expanded and single-scale images.

There are two additional candidate pipelines. The metallib grows by 38,160 bytes,
from 2,208,188 to 2,246,348 bytes. The resource driver's single-4K peak footprint
is 2,963,228,136 bytes for the parent and 2,966,488,432 for the candidate; this
diagnostic is consistent with essentially unchanged memory requirements.

Seven process-cold setup pairs give +1.07% setup time on the small workload and
+1.18% on 4K. First-call changes are +4.07% small and -1.64% at 4K. Backend setup
is outside the warm public-call table. These runs do not reset Metal's disk or
driver caches; cold and low-reuse behavior can offset the modest warm-call gain.

## Reproduction

Use the frozen `baseline-src` / `baseline` and `profile-src` / `profile` exports
to inspect the final comparison's original artifacts. The isolated fusion build
is `rect`; the rejected bound-only exports have been removed. The retained builds
use Release, tests and benchmarks enabled, with the optional libjxl reference
integration disabled. The source and runtime hashes accompany each experiment.

The main reusable commands are `compare.py`, `qualify.py --ac-candidate`,
`parity.py`, `resources.py`, and `setup.py` under
`tools/metal_dataflow`. Exact commands, environments, timestamps and log hashes
are retained in `bounds-commands.json`, `rect-commands.json`,
`qualification-final/commands.json`, and `final-commands.json` in the evidence
directory. `run-final.py` records the final sequence. The decoder is pinned to
libjxl `e8ff0976`.

The study-specific report generator regenerates the ledger from the retained
local evidence without rerunning encodes:

```sh
python3 tools/metal_dataflow/kernel_tuning_report.py \
  --evidence build/kernel-tuning/evidence \
  --output docs/metal-kernel-tuning-results.json
```

The final wall cohort uses seven alternating process pairs with three warmups
and seven measured calls. The launch-bound, isolated fusion and final attribution
stage cohorts use three pairs, two warmups and three samples. The separate
profiling-overhead stage cohort uses three pairs, three warmups and five samples;
its whole-call cohort uses five pairs, three warmups and nine samples. Warmup
and sample counts are per process. Timed runs executed after builds and correctness jobs
finished, with competing GJXL/build/study process checks before and after every
process.
The public-call boundary includes CPU codestream work; backend setup and input
loading are outside it. Setup/first-call diagnostics have their own cohort.
Times are medians of process medians; percentage changes are medians of paired
changes, so displayed time ratios need not equal the reported percentages.
