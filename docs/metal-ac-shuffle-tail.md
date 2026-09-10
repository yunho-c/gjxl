# Explicit shuffle tails in fused Metal AC reductions

The combined magnitude/count and loss shuffle tails reduce the three affected
4K GPU stages by 16.90% on padded 4K and
16.75% on Planter 4K. Whole-workflow gains are
smaller and more variable: the 11-pair follow-up changes padded 4K by
-0.37% and Planter 4K by
-1.34%.
Outputs remain byte-identical across all 56 corpus and policy cases.

Measured September 8, 2026 on Apple M4 Pro / Mac16,7, 20 GPU cores, 48 GiB,
macOS 15.6, Apple Clang 17, Release C++20/libc++. The baseline is
`da014974a86880f485525a5aaf580c16f1754bcc`, including the previously qualified
[shared DCT basis](metal-shared-dct-basis.md). These are incremental gains;
they do not include the earlier basis-sharing improvement.

## Implementation and exactness

Only `src/gpu/metal/kernels/ac_strategy.metal` changes at runtime. The
`SimdTail` template parameters default to false. Only the existing DCT16,
16×8 and 8×16 `candidate_loss_parallel` wrappers enable them. Split kernels
and larger transforms retain their original shared trees.

Both reductions keep shared-memory levels through stride 32, including its
barrier. The first SIMD group of each channel loads the resulting 32 partial
sums into registers, then adds shuffle-down values at offsets 16, 8, 4, 2, 1.
These preserve the original binary tree feeding lane zero. Magnitude and
integer nonzero count use parallel tails; loss uses the same float tree.
There is no generic `simd_sum` or per-worker reassociation of coefficients.
All 32 lanes execute every shuffle. Unused higher-lane results do not feed
lane zero's result. Only channel-local lane zero writes the completed output.

The wrappers already require 32-lane SIMD execution. DCT16 and 16×8 have two
SIMD groups per channel; 8×16 has one. Using channel-local `tid < 32` selects
one whole SIMD group separately for X, Y and B. Other groups continue to the
existing inverse-helper barrier; they do not return from the kernel early.

Five threadgroup barriers are removed from each reduction: ten per candidate
threadgroup, without multiplying by the three channels. DCT16's two trees
drop from 16 to 6 reduction-loop barriers; each rectangle drops from 14 to 4.
These counts exclude the unchanged surrounding publication barriers.
The Y-coefficient/device publication barrier, residual staging barriers and
the inverse-helper barrier protecting magnitude-arena reuse remain intact.

Queried static threadgroup storage remains 13,312 bytes for DCT16 and 7,424
bytes for either rectangle. Launches remain 192, 192 and 96 threads. The full
scratch arrays still serve the upper tree and inverse/loss stages. No device
allocation, submission count, CPU merge/tie rule or final cost arithmetic changes.

## Separate-tail screen

Each variant passed 900 guarded bitwise cases normally and another 900 with
Metal API/shader validation before timing. Three alternating process pairs
per variant were measured at padded 1080p and padded 4K. Every process used
three warmups and seven measured calls. Padded-4K paired median changes:

| Variant | DCT16 | 16×8 | 8×16 | Affected combined | All AC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Magnitude/count | -9.34% | -14.06% | -12.02% | -11.48% | -4.76% |
| Loss | -5.56% | -6.73% | -5.59% | -5.91% | -2.31% |
| Both | -14.63% | -20.56% | -16.31% | -16.75% | -6.87% |

Both tails consistently improve all three families and their combination is
best at both resolutions. The combined variant was selected for qualification
and a separate confirmation campaign.

## GPU stage confirmation

Five alternating process pairs per input, three warmups and seven measured
calls per process. Supported stage timestamps measure each candidate kernel
plus its existing cost finalizer; per-dispatch timestamps are unavailable.
The padded-4K results are:

| GPU stage | Baseline ms | Shuffle ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| DCT16 candidates | 6.087 | 5.192 | -14.70% | 5/5 |
| 16×8 candidates | 3.806 | 3.021 | -20.67% | 5/5 |
| 8×16 candidates | 3.507 | 2.923 | -16.36% | 5/5 |
| Three affected stages combined | 13.420 | 11.142 | -16.90% | 5/5 |
| All AC stages | 32.945 | 30.670 | -7.01% | 5/5 |

Planter 4K's affected total changes from 13.497 to
11.229 ms (-16.75% paired).
Each affected family also improves on padded 1080p and synthetic 128×96.
Recorded dispatch inventories and geometry match across all variants and
confirmation pairs. The affected stages use the intended fused entry points.

## Complete workflow

Seven alternating process pairs per input, three warmups and seven measured
calls per process. GPU profiling and Metal API/shader validation are disabled.
Both sides run the identical benchmark executable and differ only in the
explicit `--metallib` argument. The shader library exactly matches the fresh
Release build used for correctness qualification.

The timed boundary is `metal-public-workflow`, fully resident AQ, SIMD,
fused-tuned AC, distance 1.2, effort 7, final score disabled. It includes input
preparation, backend selection, quantization, codestream encoding and summary
assembly. These are warm workflow timings, not cold-start or file-I/O timings.

| Input | Baseline ms | Shuffle ms | Paired change | Faster pairs | Pair range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic 128×96 | 10.548 | 10.436 | -2.06% | 5/7 | -7.06% to +61.76% |
| Padded 1919×1079 | 67.525 | 66.994 | +1.42% | 3/7 | -3.24% to +10.38% |
| Padded 3839×2159 | 217.940 | 217.123 | -1.64% | 6/7 | -7.01% to +0.07% |
| Kodak 17 | 20.380 | 20.270 | -1.35% | 5/7 | -4.03% to +1.21% |
| Planter 1080p | 69.967 | 68.558 | -0.99% | 5/7 | -5.28% to +1.13% |
| Planter 4K | 211.667 | 212.047 | -0.05% | 4/7 | -5.20% to +9.48% |

The first campaign has a noisy padded-1080p regression, an effectively flat
Planter-4K result, and a large positive outlier on the tiny input. Variation
also appears in CPU phases whose source is unchanged. That does not establish
the cause of the variability or erase these observed results.

A fixed follow-up therefore measured both padded and natural images at both
1080p and 4K, with 11 alternating pairs each and the same warmup/sample protocol.
It was chosen after reviewing the first wall campaign. Every original pair
was retained; no outliers were removed and no subsequent timing retries were
used to select a favorable result.

| Follow-up input | Baseline ms | Shuffle ms | Paired change | Faster pairs | Pair range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Padded 1919×1079 | 63.027 | 62.413 | -0.49% | 8/11 | -3.38% to +2.22% |
| Padded 3839×2159 | 207.918 | 206.365 | -0.37% | 6/11 | -2.82% to +4.95% |
| Planter 1080p | 68.583 | 68.239 | -0.44% | 8/11 | -1.73% to +0.39% |
| Planter 4K | 214.960 | 212.405 | -1.34% | 9/11 | -5.69% to +1.51% |

The combined tails are retained for the repeatable GPU-stage reduction with
exact output parity. The whole-encode effect is modest and varies between
campaigns. In the follow-up, the quantization-pipeline phase improves in
8–10 of 11 pairs for each workload, while the padded inputs' codestream phase
has paired median increases of 2.14% at 1080p and 1.20% at 4K. No serializer
code changed; these observations do not identify the cause of the CPU-phase
differences. The tiny input's first-campaign outlier remains in the report,
and a repeatable tiny-image whole-encode improvement is not established.

Times are medians of process medians. Changes are medians of paired relative
changes, so they need not equal the ratio of the displayed times. Grouped GPU
times are summed within each sample before taking medians. Pair ranges show
the spread of measured pairs, not confidence intervals.

Stage gains do not imply a similarly sized whole-encode gain. This experiment
measures timing and static storage, with no new occupancy or hardware-limiter
trace. Results are specific to this device, compiler, settings and warm workloads.

## Correctness and evidence

- Three variants × 900 normal and 900 API/shader-validation cases: 5,400
  guarded cases, with exact quant norms, per-channel magnitude/nonzero counts,
  loss and final costs. Guard regions and unused device scratch stay intact.
- The selected variant additionally passes 900 cases against the split oracle:
  6,300 guarded cases overall. Coverage includes all three shapes, five batch
  sizes, ten input patterns, three quant-norm sources and two row paddings.
- Fresh Release suite: 123/124. The sole failure is the inherited
  `quantization_pipeline` score at index 1: actual `0.24919039011001587`,
  expected `0.24914586544036865`. A fresh run on the baseline produces the
  identical diagnostic, quant values and raw values.
- `metal_ac_strategy`, `metal_ac_strategy_search` and `metal_aq_evaluation`
  pass under Metal API/shader validation.
- All 56 corpus/policy encodes are byte-identical: 38 canonical images,
  efforts 1–10, high density, maximum compression, final score, alternate AQ,
  rate control and CPU maximum-error fallback. Three decoded image pairs
  match using pinned `djxl` revision `e8ff0976`.
- The six timing campaigns contain 248 processes / 1,736 measured calls.
  Raw sample/log hashes, runtime identities, launch geometry and nonoverlapping
  process intervals are audited. Qualification runs are separate from timing.

Evidence remains under `build/ac-shuffle-tail/`: `identity.json`,
`prepare.py`, `screen.py`, `qualify.py`, `confirm.py`, `followup.py`, `analysis.json`,
`verification.json`, the three variant directories, `stage-confirm/`, `wall/`,
`wall-followup/` and `parity/`. The frozen experiments retain separate patches and libraries;
the scripts refuse to overwrite existing preparation artifacts.

Selected metallib SHA-256:
`05068b4bf72709c6ca5d4488ae49515aaa63f603d286b926dffb3ea45f1937e0`.

After production promotion and comment/format cleanup, the recompiled
metallib is byte-identical to this qualified and timed artifact. The final
source, library and report hashes are recorded in `promotion.json` and
`verification.json`.

To repeat the whole-encode comparison using the frozen artifacts, choose a
new output directory:

```sh
python3 tools/metal_dataflow/compare.py \
  --baseline-build build/ac-shuffle-tail/baseline \
  --candidate-build build/ac-shuffle-tail/both \
  --workload padded_4k --pairs 7 --warmups 3 --samples 7 \
  --output build/ac-shuffle-tail-repeat/wall
```

Add `--profile` and use a separate output directory for stage measurements.
Rebuilding all variants uses `prepare.py` against the guarded baseline revision
and the recorded shared-basis Release artifacts, followed by `screen.py`,
`qualify.py both`, `confirm.py` and `analyze.py`. Use a fresh artifact directory;
the checked-in source may already contain the selected change.
