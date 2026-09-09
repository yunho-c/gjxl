# Selective Butteraugli plane geometry

**GPU-positive, not promoted to production.** The narrower per-kernel candidate consistently improves its targeted GPU stages, but complete-encode results do not meet the requested repeatability bar across image sizes. Production keeps its existing 8×8 plane launches. The exact tested patch and reproducible evidence are retained.

This follows an earlier six-kernel geometry screen, whose candidate was deferred.
Its aggregate result did not settle which individual launch changes helped.
This round measured the narrower components separately before measuring their
combination.

Baseline is `8570b3c5ebc1c68f2af6c68dce920cb8cd5be573`, with the current serializer
and shared forward-DCT work. Device: Apple M4 Pro, Mac16,7, 20 GPU cores,
48 GiB, macOS 15.6. Builds are Release, C++20/libc++. No shader arithmetic,
compiler limits, storage layout or CPU encoding algorithm changes are included
in the performance candidate. Its metallib is byte-identical to the baseline.

## Selection and attribution

| Operation | Selected shape | Scope |
| --- | --- | --- |
| Transposing convolution | 16×16 | 7 and 13 taps only |
| Fused ultra-Y and raw-mask production | 16×8 | Fused kernel only |
| Subsampling | 16×8 | Each RGB plane |

High-frequency kernels, unfused ultra, 15-tap filtering, suppression, erosion,
final composition and diagnostic lengths keep their existing 8×8 launches.
Opsin/low-medium tiles, Malta, reductions, EPF, coefficients and DCT are unchanged.
The general `MetalBackend::DispatchPlane` remains unchanged.

The selected implementation caches three shapes during pipeline creation.
Apple-family-9 support, 32-lane SIMD, pipeline maximum threads, and device
dimension limits gate each selection; otherwise that pipeline uses 8×8.
The convolution dispatcher additionally checks the tap count. Ordinary and
profiled execution share these bodies. There is no runtime experiment selector
in the selected source. These are M4 Pro measurements, not qualification of
every device supporting Apple family 9.

Three independent component arms used one experiment binary, selecting only
transpose, ultra/mask, or subsampling at backend creation. The baseline was the
unchanged qualified `8570b3c` binary. A fourth arm combined them. The controls
were read once outside the timed workflow. Each component ran three alternating
process pairs on padded 1080p, padded 4K and Planter 4K, with three warmups and
seven measured calls per process. All twelve arm/corpus encode comparisons
were byte-identical.

| Only changed component | Padded 1080p | Padded 4K | Planter 4K |
| --- | ---: | ---: | ---: |
| 7/13-tap transpose | -7.11% (3/3) | -5.27% (3/3) | -4.95% (3/3) |
| Fused ultra/mask | -8.45% (3/3) | -9.44% (3/3) | -8.44% (3/3) |
| Subsampling | -4.62% (3/3) | -7.05% (3/3) | -6.64% (3/3) |

These are changes in the affected **integrated GPU stages**, not isolated
per-dispatch timestamps. Every component improved in all three pairs on all
three workloads. Subsampling stages also contain unchanged Opsin work; the
transpose and ultra/mask family views overlap because ultra-Y contains both
kernels. They must not be added together. Stage durations are summed within
each raw sample before taking medians.

The isolated-arm stage topology verifies attribution: in the transpose arm,
only the eligible convolution dispatches change geometry; in the ultra/mask arm,
only the fused kernel changes; in the subsampling arm, only subsampling changes.
The hardware reports stage-boundary timestamps but no dispatch-boundary timestamps.
In particular, the transpose-only arm improves main ultra-X by 7.74% and main
mask blur by 9.49% on padded 4K, supporting both tap lengths without changing
the second ultra kernel, erosion or final composition.

The combined experimental screen improves affected-stage medians by 9.53% at
1080p, 7.68% at padded 4K and 6.81% at Planter 4K. Planter is only 2/3 favorable
pairs: one pair slows targeted stages 7.77%, all GPU stages 25.60%, and unchanged
low/medium and Malta stages roughly 40%. Tiny-image profiles also vary broadly.
All of these rows remain included. This is not evidence to assign the broad
slowdown to one changed kernel. The independent component results justified
testing the simpler selected implementation with fresh measurements.

## Final implementation: GPU stage confirmation

After removing experiment controls and simplifying the cached shape fields,
the final source was rebuilt and measured separately: three alternating process
pairs, three warmups and seven samples. Fully resident Metal, distance 1.2,
effort 7, `fused-tuned`, final score off, automatic CPU threads.

| Workload | Targeted stages baseline → candidate ms | Paired change | Faster pairs | All GPU change |
| --- | ---: | ---: | ---: | ---: |
| Doughnut 4672×5584 | 55.718 → 51.702 | -7.62% | 3/3 | -1.02% |
| Planter 4K | 17.767 → 16.240 | -8.60% | 3/3 | -1.51% |
| Padded 1080p | 4.347 → 4.067 | -6.54% | 3/3 | -0.09% |
| Padded 4K | 19.912 → 18.184 | -9.17% | 3/3 | -2.09% |
| Synthetic 128×96 | 0.190 → 0.182 | -4.13% | 3/3 | -1.59% |

Targeted stages are the union of the three component families. Some include
unchanged work. Fine profiling adds encoder/timestamp boundaries, with meaningful
overhead on tiny workloads; these intervals are separate from unprofiled latency.

## Warm complete encoding

The final whole-workflow protocol was chosen before these timings: **nine
alternating process pairs, five warmups and fifteen measured encodes per
process**. Longer batches address the prior seven-sample cohort's variation
around one-percent changes. This is a fresh cohort; its absolute baseline times
must not be compared to the earlier broad experiment as an optimization result.

The timed public-workflow boundary includes input preparation, backend selection,
quantization, codestream encoding and summary assembly. It excludes process and
backend creation, file I/O and JPEG conversion. Doughnut uses the same retained
4672×5584 PFM as the prior studies. Profiling is disabled in these measurements.

| Workload | Baseline ms | Candidate ms | Paired change | Faster pairs | Pair range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic 128×96 | 9.434 | 9.617 | +1.94% | 4/9 | -15.46% to +21.32% |
| Kodak 17 | 21.177 | 20.860 | -0.53% | 5/9 | -5.29% to +9.52% |
| Padded 1080p | 60.319 | 59.698 | -1.18% | 7/9 | -1.63% to +0.24% |
| Padded 4K | 199.737 | 195.812 | -1.26% | 7/9 | -2.75% to +2.85% |
| Planter 1080p | 74.465 | 73.028 | -1.93% | 7/9 | -3.74% to +4.90% |
| Planter 4K | 224.047 | 220.551 | -0.76% | 6/9 | -4.11% to +1.34% |
| Doughnut 4672×5584 | 678.552 | 677.773 | -0.01% | 5/9 | -1.29% to +1.72% |

Times are medians of process medians. Paired changes are medians of each pair's
relative change; they are not ratios of the two displayed medians and can have
a different sign. Every planned pair is retained. No favorable-result rerun or
outlier exclusion was used. The harness checked for competing GJXL, build and
study processes. Normal desktop work continued; OS/IDE background CPU activity
was observed, so the machine was not fully isolated.

The fresh final profiles confirm the narrower GPU hypothesis: targeted stages improve in all three pairs on every measured workload, by 6.54–9.17% on HD/4K and 7.62% on doughnut. Whole encoding has a smaller and less consistent benefit: HD/4K medians improve 0.76–1.93%, while Kodak is only 5/9 favorable pairs and doughnut is effectively flat (−0.01%, 5/9). The tiny workload is +1.94% with 4/9 favorable pairs; its pair range is wide, so this is an unresolved small-image risk rather than an established geometry-caused regression. These results support per-kernel selection as a promising direction but do not establish a repeatable complete-workflow gain across small, HD and 4K inputs. No production source or test is changed. A subsequent experiment could test a conservative image-size gate, with a prospectively fixed threshold and fresh whole-workflow pairs around that boundary; this round neither implements nor qualifies such a gate.

## Correctness and fallback

- 56 canonical/policy comparisons plus doughnut match the baseline byte-for-byte;
  four selected pairs decode identically with the pinned `djxl`. Doughnut's
  unchanged baseline artifact was reused after checking its input, encoder,
  encoded-output and decoded-output identities.
- The three focused Butteraugli/AQ suites pass with Metal API/shader validation.
  Full Release suite: **123/124**. The sole failure is the inherited CPU
  `quantization_pipeline` mismatch: actual `0.24919039011001587`, expected
  `0.24914586544036865`. The same baseline failure was recorded in the preceding
  geometry round; no new test failure is introduced by this implementation.
- The candidate’s profiling CLI test checks that 15-tap and other excluded plane kernels
  retain 8×8, and that eligible kernels use their selected shape or the supported
  8×8 fallback. The M4 Pro profile audit additionally requires the selected shapes
  at every eligible dispatch and identical grids/order elsewhere.
- The previous 6,444 guarded plane comparisons cover these same shapes and shader
  code. Their evidence and source/metallib identities are reused, not reported
  as a newly executed probe campaign.
- A separate compatibility library constrains the three selected pipelines to
  64 threads with Metal's documented
  [function attribute](https://developer.apple.com/documentation/metal/mtlcomputepipelinedescriptor/maxtotalthreadsperthreadgroup).
  A complete profiled 1080p encode under Metal validation confirms all three
  automatically use 8×8. The first fixture used an unsupported GNU-style
  attribute; the geometry audit rejected it. The corrected MSL-attribute fixture
  passes. Both attempts remain recorded; neither library is used for timings.
- The performance candidate and baseline share metallib SHA-256
  `b9cb6c302dbc7e377f597b1918e749ddc3a7beac0b9b29bf4bc8d7c17869e35c`.

## Evidence and reproduction

Local evidence is in `build/butteraugli-plane-selective/`: `stage-transpose/`,
`stage-ultra_mask/`, `stage-subsample/`, `stage-combined/`, `stage-confirm/`,
`wall-confirm/`, `selected-parity/`, and the frozen `src/` / `selected-src/`
and Release builds. The experimental and final patches, exact commands,
environments, inputs, runtime identities and raw outputs are retained.

The [result ledger](metal-butteraugli-plane-selective-results.json) records
component and final summaries with paired samples, the final source patch,
qualification and evidence hashes. The earlier broad report/probe and unrelated
compact-AC investigation files are preserved.

To repeat the component study, apply `screen.patch` to an isolated `8570b3c`
source export, build in Release, and set `GJXL_EXPERIMENT_BA_PLANE_GEOMETRY` to
`transpose`, `ultra_mask`, `subsample`, or `combined` when invoking the comparison
driver against the unchanged baseline. This environment variable exists only in
the experimental source. `screen.py` records the exact comparison invocations.

For final confirmation, use the selected source patch and frozen original
baseline with `tools/metal_dataflow/compare.py`: three pairs / three warmups /
seven samples with `--profile`, and a separate nine-pair / five-warmup /
fifteen-sample cohort without profiling. Choose new output directories. The
`confirm.py` script contains all seven whole-workflow workloads and input paths.
