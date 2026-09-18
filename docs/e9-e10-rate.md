# Effort-9/10 candidate ladder

Ordinary efforts 9 and 10 previously bypassed both effort-8 rate improvements:
they selected the older high-density writer and retained fast linear final
chroma-from-luma (CfL). The expanded writer and eight-step nonlinear final CfL
were selected only at effort 8.

## Evidence from the saved sweep

The completed September 15 fixed sweep uses main revision
`8e6faf05be62233b56e168da429891c17de62099`, 65 images, fully resident Metal,
and the pinned fast-ssim2 scorer. Its published mean PCHIP BD-rates against
libjxl e7 over quality [75,85] are:

| GJXL effort | BD-rate versus libjxl e7 |
| --- | ---: |
| 8 | -4.1954% |
| 9 | -2.7605% |
| 10 | -2.7699% |

Lower BD-rate is better. Recomputing each image's curve directly against GJXL
e8 gives **+1.4694% for e9** and **+1.4567% for e10**. Akima gives +1.4783%
and +1.4523%, respectively. All 65 comparisons support [75,85] without
extrapolation, and only one image improves at each higher effort. These are
direct curve comparisons, not differences between the aggregate table entries.
[Baseline analysis and source hashes](e9-e10-rate-baseline.json) retain the
input identity. The figures describe the unchanged baseline; no candidate
corpus result is claimed.

## Current candidate policy

- Automatic compression at ordinary efforts 8-10 selects `kRateOptimized`:
  full HybridUint/alphabet-width search with balanced complete-file fallback.
- Ordinary fully resident Metal efforts 8-10 use eight-step nonlinear final
  CfL. CPU and explicit exact/throughput modes retain their current frontend.
- Explicit high-density mode retains its existing writer, four AQ updates,
  and frontend policy. Maximum compression retains its independent exhaustive
  writer override. Maximum-error mode keeps its existing final-CfL selection.
- Effort 8 retains three AQ updates; efforts 9-10 use four. No effort-1-8
  defaults change.
- Ordinary effort 10 additionally searches the existing 16x32, 32x16 and 32x32
  transform families at every 8x8 base-block anchor, instead of every two
  blocks. This applies to CPU and Metal paths that perform AC search.
  Maximum-throughput mode still bypasses AC search. High-density and
  maximum-error overrides retain their existing search spacing; the independent
  maximum-compression writer override does not change frontend selection.

| Ordinary effort | AQ updates | DCT32-family anchor spacing | Writer / resident final CfL |
| --- | ---: | ---: | --- |
| 8 | 3 | 2 blocks | Rate optimized / 8 nonlinear steps |
| 9 | 4 | 2 blocks | Same as e8 |
| 10 | 4 | 1 block | Same as e8 |

This adopts the four-update e9/e10 refinement and denser e10 placement from
libjxl revision `e8ff09762481785938d8e4e01333ed3917571161`, without claiming
algorithm/output parity. GJXL retains its qualified three-update e8 policy
(libjxl uses two). Learned Modular trees, additional prediction presets,
AC run-length coding, optimal-matching LZ77 and downsampling are outside this
candidate. Extra AQ/placement search can change pixels and does not guarantee
smaller files at matched quality for every image.

The initial inheritance commit `d3ae3dd` gave e8/e9 the same recipe. This
candidate deliberately distinguishes them with the fourth update.

The existing CPU and resident storage planners call the same entropy resolver
as the encoder, so e9/e10 reserve the rate-optimized writer's model, fallback,
and concurrent-search storage. Nonlinear final CfL uses the existing bounded
shader scratch and adds no image allocation. Dense e10 placement uses one
shared stage policy for CPU traversal, GPU candidate generation and both
host/device storage plans. A full 64x64 tile has 320 staged candidates, versus
258 previously; larger-family counts grow from 12/12/9 to 35/35/25. No
transform crosses the tile boundary. Candidate buffers and transform scratch
grow accordingly. Prepared searches rebuild candidates at each call, and
reuse bounds must cover both the largest dimensions and densest policy used
since reset. This extension inherits the
rate-optimized writer's runtime and memory tradeoffs; the balanced fallback
protects bytes for the same completed frame, not quality across frontend
recipes.

## Validation

The previous inheritance-only validation remains in
[e9-e10-rate-validation.json](e9-e10-rate-validation.json), tied to its recorded
source hashes. Candidate-ladder checks and exact commands are recorded in
[e9-e10-ladder-validation.json](e9-e10-ladder-validation.json).

The Release build passed all **17 focused suites**. All **16 CPU/Metal e1-e8
outputs** from the retained 80x72 synthetic fixture are byte-identical to
`d3ae3dd`. System `djxl` 0.12.0 decoded all four e9/e10 CPU/Metal outputs to
finite 80x72 RGB float pixels. These bounded fixtures verify compatibility and
decodability, not corpus rate-quality. The reproducible fixture emitter is
[e9-e10-ladder-fixtures.cpp](e9-e10-ladder-fixtures.cpp).

The focused regression scope covers:

- AQ counts and writer/CfL/placement override policies at efforts 1-10.
- A supplied-cost fixture whose winning 32x32 transform is at an odd anchor,
  reachable only by dense search; complete coverage and color-tile boundaries.
- CPU/Metal grid parity, partial tiles, all three Metal DCT implementations,
  and sparse/dense/sparse transitions in prepared resident searches.
- Candidate counts across 16,400 geometry/mode combinations, real host
  allocation peaks, allocation failures, and prepared-state recovery/reuse.
- Forwarding the search policy through quantization orchestration and bounded
  CPU, resident Metal, exact-coefficient and throughput workflows.
- Baseline/candidate byte comparisons for efforts 1-8 and external decoding
  of bounded synthetic e9/e10 fixtures.

No speed benchmark or new corpus sweep is part of this implementation.
Matched-quality BD-rate, secondary-quality review and controlled timing remain
necessary before claiming a rate/speed improvement or promoting this candidate
as a qualified replacement. The focused checks do not claim a full-suite pass.
