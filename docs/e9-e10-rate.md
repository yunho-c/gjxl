# Extend effort-8 rate policies to efforts 9 and 10

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

## Policy change

- Automatic compression at ordinary efforts 8-10 selects `kRateOptimized`:
  full HybridUint/alphabet-width search with balanced complete-file fallback.
- Ordinary fully resident Metal efforts 8-10 use eight-step nonlinear final
  CfL. CPU and explicit exact/throughput modes retain their current frontend.
- Explicit high-density mode retains its existing writer, four AQ updates,
  and frontend policy. Maximum compression retains its independent exhaustive
  writer override. Maximum-error mode keeps its existing final-CfL selection.
- Efforts 8-9 retain three AQ updates; effort 10 retains four. No effort-1-8
  defaults change.

Efforts 8 and 9 now have the same ordinary encoding recipe. Their equivalence
is intentional: a higher effort no longer selects the older rate path. A
future distinct e9 recipe needs its own qualification. Effort 10's fourth AQ
update can change pixels and does not guarantee smaller files at every
measured quality or for every image.

The existing CPU and resident storage planners call the same entropy resolver
as the encoder, so e9/e10 reserve the rate-optimized writer's model, fallback,
and concurrent-search storage. Nonlinear final CfL uses the existing bounded
shader scratch and adds no image allocation. This extension inherits the
rate-optimized writer's runtime and memory tradeoffs; the balanced fallback
protects bytes for the same completed frame, not quality across frontend
recipes.

## Validation scope

The workflow regression covers CPU efforts 1-10 and resident Metal e8/e9/e10,
checks e8/e9 byte equality on its fixture, verifies e10 retains its fourth AQ
update, and exercises explicit policy overrides across every effort. Existing
storage-envelope tests cover CPU/resident e9/e10 admission and execution.

A fresh Release build passed **all nine focused suites**: codestream workflow,
codestream encoder, C API, Metal AQ reconstruction, entropy and serializer
storage plans, and CPU/resident/whole-workflow storage plans. Benchmarks were
disabled in the build. Source/binary identities, commands, and log hashes are
retained in [validation](e9-e10-rate-validation.json).

No speed benchmark or new corpus sweep was run. A subsequent e9/e10 rate
sweep, secondary-quality review, and matched-quality timing are needed before
claiming measured improvements for the extended policy. This focused run does
not claim a full-suite or decoder-conformance qualification.
