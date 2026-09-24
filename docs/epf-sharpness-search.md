# EPF sharpness search

The CPU reference, CPU/Metal AQ integration, device candidate search, and
storage/profiling accounting are implemented. Validation on September 21, 2026
covered the Release build, CPU/Metal parity, decoder conformance, and a bounded
same-distance comparison. Existing baseline test failures are recorded below.

## Reference behavior

The reference is `lib/jxl/enc_heuristics.cc` in local libjxl revision
`9e7fba5d5b829054ec1aca96a4c7684b2fd1fd2f`, specifically
`ComputeARHeuristics` and `ComputeBlockL2Distance`. The implementation worktree
starts from GJXL `b1a7373`.

The CPU and Metal default policy enables search at effort 6 and above, distance at least
0.5, and nonzero EPF iterations. Maximum-error control and explicit
maximum-throughput mode retain their existing filtering policy. AQ uses the
neutral sharpness field (4); search runs only after the final coefficient
decisions. Within one distance attempt it changes the EPF control map, not
coefficients, quantization, transforms, or the number of EPF passes. Changed
codestream size can affect subsequent attempts in target-size control.

`VarDctEncodingOptions::adaptive_epf_sharpness` defaults to true and can be
disabled for ablation. CUDA retains fixed sharpness; its low-level AQ API
rejects explicit search requests until a CUDA implementation is available. The CLI, public-workflow encoding benchmark, and quality
benchmark expose `--epf-sharpness-search on|off`. Both benchmarks record the
setting in their raw JSON. The quality benchmark also writes the codestream for
independent decoding and quality measurement.

Distances through 4.5 use globally uniform candidate maps `{0, 2, 7}`;
larger distances use `{0, 4}`. Each candidate is filtered over the actual image
extent with decoder-equivalent boundaries. For each clipped 8x8 block, the
search sums squared XYB differences multiplied by the squared initial blurred
pixel mask, with channel weights `{12.339445295782363, 1, 0.2}`. Transform
padding is excluded from this error. Filtering is global even though scoring
and selection operate on 8x8 blocks: candidates cannot be reconstructed as
independent blocks because EPF crosses block boundaries.

## Effective selector and parallelism

The pinned libjxl source has two raster passes and a context histogram. However,
both `ctx_histo[val]` and `totals[context]` are `size_t`. Totals begin at one
and increment whenever a histogram count increments. Therefore each count is
strictly smaller than its total, the integer quotient is zero, and the
`log1p` term in the second pass is always zero. Its final multipliers are
independent of the first pass and of neighboring blocks.

GJXL preserves these effective semantics. It selects the minimum candidate
error after multiplying candidate zero's error by
`max(0.85970338919928291f, pow(0.98017198824148288f, min(5, distance)))`.
Ties retain the first candidate. The selector test includes an independent
literal implementation of both original raster passes and compares the
resulting maps at both candidate-set boundaries and across varied errors.
Changing the histogram quotient to floating point would be a policy change,
not an equivalent optimization, and is outside this implementation.

## Execution design

All candidates share the same coefficients, inverse transforms, and Gaborish
output. For candidate comparison, reconstruct once and apply forward Gaborish
once, then run EPF and block-error reduction for each candidate. This removes
repeated work while
preserving global filtering. The final mixed sharpness map must be applied
afresh when materializing the decoded image or its final quality score; a
mosaic of uniform-candidate outputs is not equivalent.

The Metal implementation keeps candidate images, errors, and selection on
device in the final AQ submission. One 32-lane SIMD group reduces each clipped
8x8 block's distortion; one thread per block selects the final value. Candidate
zero skips EPF only when its sigma lookup-table entry is zero. The search
reuses the linear-RGB scratch image for the common post-Gaborish XYB base and
the existing two filter scratch images for candidate passes. A resident
frontend reuses its original XYB and blurred mask. Other preparations upload
explicit immutable reference planes.

The final block map reaches both the ordinary owned frame and the completed
resident frame lease. Reuse resets the device map to 4 before a new AQ
attempt, and preparation/reconfiguration require a neutral input map.
Diagnostic final reconstructions and scores reapply the selected mixed map
through the usual postprocessing path, including a fresh Gaborish pass when
enabled.
An encoding-only final iteration still reconstructs coefficients for search,
while omitting the final Butteraugli evaluation. Device and host storage
admission and profiling bounds include the search. The resident profiler uses
`ar.sharpness_search`, plus `ar.reconstruction` when search requires the
otherwise skipped final reconstruction.

## Correctness checks

`epf_search` compares the parallel selector with the literal libjxl raster
algorithm, checks masked block errors and clipped edges, compares shared
reconstruction against repeated complete filtering, and checks that CPU AQ
updates and coefficients are unchanged by final search.

`metal_epf_search` covers Gaborish on/off, EPF iteration counts 0--3, exact and
device-coded coefficients, both candidate sets, the distance threshold,
prepared-operation reuse, candidate errors against CPU filtering, chosen-map
readback, reconstructed RGB, resident scored/unscored final frames,
owned/completed-frame parity, profiling parity, and readback-failure atomicity.
It also checks that the production search adds no submission. Candidate errors
pass a bound of `1e-8 + 3e-4 * cpu_error`; decisions with a clear error margin
agree, and near ties may differ within the tested error bound. Final RGB passes
an absolute tolerance of `1e-4`. These are fixture-level bounds, not a guarantee
for arbitrary input images. `metal_storage_plan` includes the uploaded-reference
and resident-reference layouts and rejects incompatible search policies.

The full codestream conformance suite includes searched maps from both
candidate sets, partial edge blocks, and mixed large transforms. Its independent
decoder comparison checks reconstructed linear pixels against the selected-map
CPU reconstruction, in addition to the in-process CPU/Metal comparisons.

## Validation results

Build configuration: Ninja, `CMAKE_BUILD_TYPE=Release`, tests and benchmarks
enabled, `GJXL_ENABLE_LIBJXL_REFERENCE=ON`, `GJXL_ENABLE_METAL_PROFILING=OFF`.
The full 151-test sweep plus targeted reruns passed 148 tests. Three failures
reproduce on an independent Release build of unchanged base revision `b1a7373`:

- `quantization_pipeline`: the same hard-edge score differs from its floating
  point golden by `4.4524669647216797e-05`; all reported scores, quantization
  values, and raw quant values match the base build.
- `low_effort_strategy_policy`: the old effort-9 update expectation is stale.
- `encoding_benchmark_cli`: its effort-9 entropy expectation and positive-time
  assertion fail identically on the base build.

Feature-specific regressions were corrected: disabled search contributes no
CPU profiling stage time; legacy golden checks explicitly retain fixed
sharpness; default CLI goldens reflect the new policy; installation includes
the new EPF header and the existing transitive AC-selection interface. The
installed C++20 and C++23 consumers pass. No Python test files were changed.

The independent decoder was libjxl `e8ff09762481785938d8e4e01333ed3917571161`.
Both new searched-map fixtures pass, with maximum absolute linear-RGB errors
of `6.73532e-06` and `7.39098e-06`. The full conformance run passes 24 of 25
fixtures and its workflow/corruption checks. The remaining single-block impulse
golden hash fails identically on the base build (`8627470560574437943`). It was
not regenerated as part of this feature.

Local logs are retained as `build/epf-full-tests.log`,
`build/epf-corrected-tests.log`, `build/epf-install-test.log`,
`build/epf-baseline-tests.log`, and `build/epf-{baseline-,}conformance.log`.
Decoded conformance artifacts are under `build/epf-conformance/` and
`build/epf-baseline-conformance/`.

## Bounded quality and runtime comparison

All pre-existing measurement jobs and their artifact audit completed before
these comparisons. One frozen Release binary encoded Kodak images 01, 07, 13,
and 19 with search on/off at efforts 6 and 8 and distances 1, 4, and 8. An
additional forest-stream 12 MP pair used effort 8, distance 1. The device was
an Apple M4 Pro with 48 GiB memory, macOS 15.6. Each job used one validation
encode, two warmups, seven complete-call timing samples, eight CPU threads,
the fully resident Metal workflow, and no final perceptual score or stage
profiling. The benchmark checks repeated-output byte identity. Pair order was
randomized with a recorded seed; independent decoding and SSIMULACRA2 scoring
followed all timing collection.

The table aggregates the four Kodak images. Byte changes compare the sums of
file sizes, quality changes are mean SSIMULACRA2 differences, and time changes
are geometric means of per-image median-time ratios.

| Effort | Distance | Bytes | SSIMULACRA2 delta | Encode time |
| --- | --- | --- | --- | --- |
| 6 | 1 | +0.68% | +0.110 | -0.04% |
| 6 | 4 | +2.74% | +0.532 | +7.35% |
| 6 | 8 | +4.43% | +2.227 | +3.62% |
| 8 | 1 | +0.69% | +0.122 | +2.66% |
| 8 | 4 | +2.84% | +0.450 | +2.89% |
| 8 | 8 | +4.59% | +2.128 | +2.81% |

All 25 pairs improved SSIMULACRA2 at the same requested distance. On the 12 MP
case, median complete-call time was 475.26 ms without search and 493.92 ms with
search (+3.92%); bytes increased 0.55% and SSIMULACRA2 increased 0.105. This
larger image also exercises tiled EPF dispatch.

These short paired runs are insufficient to resolve small latency differences:
individual ratios ranged from -10.50% to +11.98%. Negative time changes should
not be treated as a search optimization benefit. The results demonstrate
same-distance quality changes and bounded execution cost on this device; they
do not establish matched-quality bitrate savings, BD-rate, or corpus-wide
performance. Those remain separate experiments.

[`epf-sharpness-search-results.json`](epf-sharpness-search-results.json) retains
the per-case raw timing samples, sizes, scores, input/artifact hashes, binary
hashes, and protocol. `build/epf-qualification/` additionally retains the frozen
encoder, source patch and complete source-hash manifest, commands/script,
codestreams, decoded images, and logs. Its benchmark revision field identifies
the base commit; the feature was uncommitted at measurement time, and the
source/binary hashes identify the actual measured implementation.

## Example Methods wording

After the final quantization step, we select an EPF sharpness index for each
8-by-8 block by comparing two or three globally filtered reconstructions.
Each candidate uses a uniform sharpness index over the image; its block cost
is the masked, channel-weighted squared error in XYB space relative to the
original image. We retain libjxl's candidate sets and bias toward disabling
smoothing. In the resident GPU path, candidate evaluation shares the inverse
transforms and Gaborish output, while filtered images, block errors, and
selection remain on device. A SIMD group reduces each block's candidate error,
followed by an independent per-block minimum selection under the effective
selector semantics of the pinned libjxl revision. The resulting sharpness map
is serialized with the block metadata. The search follows adaptive quantization,
so it does not change coefficient decisions within a distance attempt.
