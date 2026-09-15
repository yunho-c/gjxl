# Effort-8 nonlinear final-CfL integration

Item #2 is integrated above writer commit `4c31e04` on `feat/e8-rate`.
Ordinary effort-8 resident Metal encoding now uses the eight-step nonlinear
final chroma-from-luma search from the qualified prototype. This changes
reconstructed pixels, so its rate result must be measured at decoded quality.

## Policy and implementation

`FinalColorCorrelationIterations` selects eight steps when the selected
backend is Metal, effort is 8, the AQ mode is fully resident, density is
default, and rate control is not maximum error. CPU encoding, other efforts,
exact-coefficient and maximum-throughput modes, high-density overrides, and
maximum-error encoding retain their existing automatic CfL policy. A writer
compression override does not change the frontend CfL selection.

The nonlinear kernel uses the CPU reference's clipped residual objective,
four independent accumulation lanes, pairwise reduction, finite-difference
curvature estimate, regularization, step clamp, convergence threshold, and
integer factor rounding. It excludes transform low-frequency coefficients and
uses the channel-specific quantization weights. Non-finite inputs and steps
flag the resident error word before Metal's clamp can mask a NaN.

The final map is computed once, from the initial adjusted quantization field,
then retained across AQ iterations. It runs in the existing command buffer
using the existing coefficients, metadata, quantizer, and output buffers.
There are no new device allocations, host arrays, submissions, or readbacks
in this path. The nonlinear branch declares 112 bytes of threadgroup scratch;
the existing fast branch declares 64 bytes. The shared parameter structure
grows from 16 to 20 bytes, guarded by host and shader layout assertions.

The lower-level AQ option `color_correlation_iterations` accepts 1–20 steps
when `fast_color_correlation` is false. Its default remains 20 to preserve
the CPU nonlinear reference's behavior. Resident preparation uses zero to
select the established fast regression. Invalid limits are rejected before
changing caller output or scheduling work.

## Correctness and resource validation

The fresh Release build is `build/e8-cfl-integrated`. The original build,
the integrated writer build, and both prototype controls remain frozen.
`build.json` records production source, archive, and binary hashes; the
validation logs and replay evidence are under `build/e8-cfl-validation`.

- 144 exact CPU/Metal factor comparisons pass using identical prepared DCT
  coefficients. They cover all seven transform families, a partial color
  tile, raw quantization values spanning 1–256, global scales 1/3541/32768,
  fast regression and 1/8/20 nonlinear steps, flat and sparse inputs,
  clipping thresholds, rounding boundaries, and saturated factors.
- Invalid iteration limits, raw quantization, and NaN coefficients are
  rejected. The numeric-failure tests preserve the caller's map.
- CPU direct and prepared calculations agree at 1/8/20 steps. Workflow
  policy tests cover all efforts, backends, and relevant overrides.
- Profiling/cache tests confirm one final-CfL dispatch followed by reuse
  across AQ updates. Existing allocation, admission, storage-envelope, and
  many-iteration tests pass with the nonlinear branch enabled.
- Full Release CTest passes 137/138. The sole failure is the already
  reproduced `quantization_pipeline` pin: actual `0.24919039011001587`,
  expected `0.24914586544036865`.
- Full pinned-decoder conformance passes 22 fixtures and four workflow
  cases. One existing `single-block-impulse` hash pin fails identically in
  the frozen writer build: actual FNV `10525847038797353681`, expected
  `11459255244783164287`. Direct decoding of that retained stream succeeds.
  All 23 fixture streams and decoded images exactly match the frozen writer
  build. This also corrects the earlier writer report's claim of a completely
  passing full conformance run; its saved log contained the same pin failure.
- Kodak20 at distance 1.9 is byte-identical to the frozen writer binary at
  efforts 1–7, 9, and 10. This is a bounded policy regression check.

## Rate-study protocol

[`integrated_cfl_rate.py`](../../tools/e8_rate/integrated_cfl_rate.py) encodes
each point with both the production binary and an independently linked
combined prototype. The combined control was built from the frozen writer
archives and the eight-step shader overlay before production source edits.
Each output must match that control byte for byte and decode to exactly the
pixels of the earlier standalone eight-step CfL corpus. These checks isolate
integration drift from the intended frontend change.

The study uses the same 65 images and six distances as the writer integration:
0.55, 1, 1.9, 2.8, 4.6, and 6.4. The pinned fast-ssim2 revision is
`c3867954c7bec8a761951df9256354b305fd0cff`, with linear-sRGB PFM input.
PCHIP and Akima integrate log bytes over measured quality [75,85], without
extrapolation. Corpus summaries are arithmetic means of per-image BD-rates.
The anchors are the integrated writer alone, the original GJXL baseline, and
stock libjxl effort 8. Incremental and combined gains are computed directly
from these curves, not by adding the standalone prototype averages.

The secondary analysis in
[`integrated_cfl_secondary.py`](../../tools/e8_rate/integrated_cfl_secondary.py)
reuses saved Butteraugli scores only after verifying exact decoded hashes.
It substitutes the newly measured integrated byte sizes. Its six diagnostic
images have different shared quality intervals, so no corpus average is
reported for that metric.

## Measured rate result

All **390 points across 65 images** passed independent combined-prototype
byte replay and exact decoded equality with the standalone eight-step CfL
control. All 65 curves support the fixed fast-ssim2 [75,85] interval.

| Anchor | Mean PCHIP BD-rate | Akima sensitivity | Improved image curves |
| --- | ---: | ---: | ---: |
| Integrated writer alone (#1) | **-0.520057%** | -0.529036% | 48/65 |
| Original GJXL, before #1 and #2 | **-1.535135%** | -1.542573% | 64/65 |
| Stock libjxl effort 8 | **-1.389241%** | -1.395297% | 55/65 |

These are measured-quality rate results for this corpus and interval. In
particular, #2 improves the mean over #1 while worsening 17 image curves on
this metric. The complete results are in
[rate analysis](cfl-integrated-rate.json) and
[integration validation](cfl-integration-validation.json). The retained
manifest, all encode commands, streams, decoded hashes, scores, and analysis
are under `build/e8-cfl-integrated-rate`.

The existing Butteraugli cross-check remains mixed after substituting the
integrated byte sizes. Incremental PCHIP BD-rate versus the writer alone is:

| Diagnostic image | Butteraugli BD-rate |
| --- | ---: |
| CLIC truck (`0c49a5…`) | +0.976718% |
| CLIC pepper (`604308…`) | +0.655906% |
| Kodak02 | +0.201848% |
| Kodak20 | -4.549557% |
| CLIC `b939ac…` | -0.487624% |
| Campus interior, 12 MP | Unsupported: nonmonotonic curve |

Each supported comparison uses that image's common measured Butteraugli
interval across the controls. Exact decoded hashes justify reusing the
previously measured metric values; no fresh Butteraugli timing or aggregate
quality claim is made. See [secondary analysis](cfl-integrated-secondary.json)
for intervals, both interpolation methods, original-GJXL and libjxl anchors,
and metric provenance.

## Runtime boundary

Rate collection uses eight CPU participants, no warmup, and one sample per
point on the Apple M4 Pro. These timings are diagnostic only. The integrated
writer and nonlinear CfL costs have not received a combined latency study;
the earlier standalone prototype timing cannot establish that cost. Speed
optimization and latency qualification remain deferred as requested.
