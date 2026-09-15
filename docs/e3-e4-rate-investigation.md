# Effort 3/4 rate investigation

Investigation completed for the supplied comparison, September 14, 2026.
Controlled interventions identify different dominant causes at e3 and e4
and establish measured routes past the aggregate same-effort gap. Production
implementation and default qualification are separate from this investigation.

**E3 is primarily a DC/AC allocation problem for the plotted metric.**
Ordinary DC rounding with one extra precision bit and smoothing off changes
the full 65-image result from +2.085% to **−2.512%** against libjxl e3 over
SSIMULACRA2 75–85. This reallocates bits from AC to finer DC at matched score,
but independent Butteraugli exposes a material perceptual tradeoff.

**E4 has a substantial quality-preserving writer opportunity.** Its
prediction-aware DC uses extra precision and structured residuals, but the
balanced GJXL writer omits the integer-mapping search libjxl enables for that
case. A controlled ablation isolates that search as a major contributor.
Replacing only the writer on all 455 native e4 outputs saves **1.172%**
BD-rate versus native GJXL, with identical decoded floats, and changes the
same-effort comparison from +0.808% to **−0.374%**. All 65 curves improve
relative to native GJXL; 45/65 beat stock libjxl e4.

The recommended first implementation is DC-scoped entropy mapping search at
e4, retaining its existing reconstruction. Decouple DC grid precision from
rounding and qualify the e3 finer-grid recipe as a separate, metric-dependent
experiment. Neither the rate results nor these diagnostic timings qualify a
new production default or speed claim. The cohorts are the original images,
not held-out data; the large-image groups contain only three scenes.

![Completed rate routes](../build/e3-e4-rate-20260914/final-routes.png)

## Exact comparison recovered

The attachment is the `speed-bd-rate-before-after.png` generated under
`/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/gjxl-e1-4-frankenstein-20260914/`.
It compares every point against **libjxl effort 7**. The relevant GJXL e1–4
binary is revision `b1fbfc15f32a3c3ac3d1d2b5b7983d0ced22a00f`, whereas this
worktree is at `17fcd39`. The investigation uses the immutable plotted binary
and its original references, decoder and scorer.

Directly recomputing per-image BD-rate against the **same libjxl effort**
gives the following. These are arithmetic means of per-image percentages,
integrating log(bytes) over measured fast-SSIMULACRA2 75–85; positive means
GJXL requires more bytes. Every pair has complete coverage without
extrapolation. Both PCHIP and Akima are retained.

| Cohort | Images | GJXL e3 / libjxl e3 | GJXL e4 / libjxl e4 |
|---|---:|---:|---:|
| All | 65 | +2.085% | +0.808% |
| Kodak | 24 | +1.323% | +0.900% |
| CLIC | 32 | +2.524% | +0.750% |
| 12 MP | 3 | +1.986% | +0.525% |
| 24 MP | 3 | +2.888% | +1.249% |
| 48 MP | 3 | +2.793% | +0.535% |

Akima gives +2.070% and +0.861% overall. GJXL wins on 6/65 e3 inputs and
13/65 e4 inputs. These are direct curve comparisons, not differences or
ratios of the plot's aggregate percentages. The nine large inputs represent
three scenes at three sizes, not nine independent scenes. In particular,
the plotted +8.81% for 48 MP e4 is mostly a gap both low-effort encoders have
against libjxl e7; the same-effort gap is +0.535%.

The worst e3 input is CLIC `d1a9be98d1936065967adac50a6fb750` (+6.977%);
the worst e4 input is CLIC `b51d5fb537246482ef7a7ab63f093cab` (+5.275%).
The common portrait outlier is +6.737% at e3 and +2.559% at e4. This
content dependence motivates per-image controls rather than tuning only an
aggregate.

## Current algorithm boundaries

The pinned source shows GJXL e3/e4 share fixed DCT8, uniform initial
quantization `0.79 / distance`, Gaborish off, EPF2, weighted DC coding,
balanced entropy behavior, full coefficient-order selection, and matrix
scale policy. Three selected behaviors change at e4:

1. Zero AQ updates becomes one.
2. Ordinary DC rounding becomes prediction-aware DC quantization **and one
   extra DC precision bit**. The public mode bundles those two effects.
3. Adaptive DC smoothing becomes enabled.

Source anchors on the measured revision: `gjxl/src/codestream/workflow.h`
(`ResolveDcQuantization`, `ResolveAdaptiveDcSmoothing`),
`workflow_internal.h` (`AdaptiveQuantizationIterations`,
`UseUniformInitialQuantization`), and `workflow.cpp` (DC precision setup,
`ResolveEntropyBehavior`, `ResolveCoefficientOrderBehavior`).

Pinned libjxl `e8ff0976` uses fixed DCT8, uniform initialization and no
Gaborish at both e3/e4. It has no Butteraugli AQ refinement there, normally
enables DC smoothing, skips non-default CfL maps and AC per-block precision
adjustment, and uses distance-dependent EPF passes. E4 enables additional
writer behavior relative to e3. Thus equal effort labels do not imply
equivalent algorithms, and comparing GJXL e3/e4 alone cannot isolate AQ.

## Existing causal evidence and its limits

- The old `8e8c49a` frontend's Gaborish/initial-field interaction was isolated
  in `e4-rca-20260913/REPORT.md`. Its combined correction is already in the
  plotted e1–4 binary and cannot be credited as a future improvement.
- That study's 455 frozen-frame writer comparisons gave about 0.357%
  native writer overhead against a libjxl e4 writer, with exact decoded
  float equality. The frontend has since changed; this is evidence to
  prioritize frontend controls, not a numerical attribution of today's
  +0.808% residual. The current-frame follow-up below supplies the new budget.
- `e4-final-field-rate-20260914/REPORT.md` confirms meaningful final-field
  metadata cost and content-dependent AC allocation. Its hard global median
  policy amplified a one-block change into a frame-wide quality jump and
  failed qualification. Do not reuse that selector as the proposed fix.
- `e4-frontend-aq-20260914/REPORT.md` measures large time savings from zero
  AQ, but also cross-metric tradeoffs. It does not isolate e3's different DC
  defaults or prove a complete-corpus BD-rate win.

All study directories above are under
`/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/`.

## First new intervention: DC/AQ factorial

`tools/low_effort_rate/study.py` uses the plot's exact binary; no C++ source
change is required. Effort {3,4} × DC mode {round,prediction-aware} × smoothing
{off,on} separates the AQ and DC-policy interactions. The DC mode remains
a bundled rounding/precision intervention and must be split in a later
diagnostic if it is a material cause.

Ten diagnostic inputs cover Kodak controls, the e3 and e4 outliers, the
portrait, campus at 12/24/48 MP, alpine lake at 24 MP, and forest stream at
48 MP. This is an intentionally informative selection, not a representative
or held-out population. Five original distances per arm bracket the target
interval. Baseline cells reuse verified scores except for a fresh native
byte/decode/score replay at Q80 on every image and effort.

The frozen manifest checks 1,820 original e3/e4 codestream hashes, binary and
dylib hashes, selected input hashes, and the analysis code. Fresh encodes
perform a validation encode and repeated-byte check; changed outputs use
the pinned float decoder and independent scorer. Decoded PFMs are hashed
before removing those temporary files. Collection is serial, bounded to 400
factorial cells, resumable, and protected by a process lock and free-space
floor. Concurrent studies prevent interpreting diagnostic timing samples
as controlled speed measurements.

Artifacts: `build/e3-e4-rate-20260914/`, including `manifest.json`,
`direct-e3.json`, `direct-e4.json`, `observations.jsonl`, `commands.jsonl`,
`progress.json`, native controls, and all changed codestreams. The ledger
and live process, not this document, determine collection status.

## Completed factorial findings

All 400 cells completed: 300 changed-output observations, 20 fresh native
controls, and 80 verified saved native cells. The 640 recorded encoder and
decoder commands succeeded. The post-run audit rehashed all 400 referenced
codestreams and every frozen file; all 280 declared per-image comparisons
had valid curves. Native controls reproduced codestream bytes, decoded PFM
hashes and scores. See `audit.json` and `dc-factorial-analysis.json`.

The following **ten-input diagnostic** results must not be substituted for
the full 65-image means above. Native and candidate curves use the same five
sampled distances; the stock reference uses its saved unresampled curve.

| GJXL arm | BD-rate vs corresponding libjxl effort | Akima | Wins / 10 |
|---|---:|---:|---:|
| e3 native: round, smoothing off | +3.634% | +3.667% | 0 |
| e3 prediction-aware + extra bit, smoothing off | **−0.426%** | −0.299% | 6 |
| e3 prediction-aware + extra bit, smoothing on | +1.207% | +1.327% | 5 |
| e4 native: prediction-aware + extra bit, smoothing on | +1.753% | +1.808% | 0 |
| e4 prediction-aware + extra bit, smoothing off | **+0.368%** | +0.433% | 5 |

The e3 DC candidate improves BD-rate by **3.880% relative to native e3**,
improving 9/10 images; forest stream 48 MP regresses by 0.161%. The selected
cohort mean crosses below libjxl e3, but this is not a full-corpus win.
Disabling smoothing in native e4 saves 1.353% relative to native e4.

Keeping prediction-aware DC and its extra precision bit fixed, enabling
smoothing costs +1.648% at e3 and +1.380% at e4. Keeping that DC policy and
smoothing off fixed, adding the AQ update costs +0.688%; with smoothing on,
it costs +0.420%. These are complete-workflow causal interventions, including
their downstream effects, not frozen-coefficient filter comparisons. Their
percentage means are not additive. Detailed per-image contrasts are retained
in `causal-effects.json`.

This identifies DC allocation and the DC/filter/AQ interaction as material
causes on the diagnostic cohort. It also demonstrates why an e3/e4 difference
cannot be assigned entirely to AQ. The results concern SSIMULACRA2 75–85;
the completed cross-metric and visual follow-up is reported below.

![Baseline and DC factorial](../build/e3-e4-rate-20260914/dc-factorial.png)

### Splitting rounding from precision

A diagnostic workflow-object overlay uses the exact plotted build's existing
libraries and shader, with only an independent `extra_dc_precision` setting.
Ten neutral e3/e4 output checks and ten explicit-default precision checks
match the original binary byte-for-byte. Sources, linked objects and libraries,
build commands and qualification are retained in `precision-probe/`.

The first proposed full matrix encountered an explicit Metal validation
constraint: prediction-aware DC with zero extra precision is unsupported.
That arm was rejected before encoding. Five successful ordinary-rounding
extra-bit outputs and the failure are retained in `precision-factorial/`;
it is terminal, not a live or completed experiment.

The supported follow-up uses a three-setting sequence: ordinary rounding at
zero extra bits, ordinary rounding at one extra bit, then prediction-aware
rounding at one extra bit, all with smoothing off. This isolates the grid
change and then the rounding change along that path without relaxing the
Metal constraint. It does not establish the untested interaction at zero
extra bits. Seven diagnostic inputs and e3/e4 require 70 new scored outputs;
the other settings reuse the completed factorial. Current artifacts and
process status are in `precision-split/` and `precision-split.log`.

### Completed supported precision split

All 70 new scored outputs completed across seven selected images and both
efforts. Fourteen additional neutral overlay encodes, including campus at
48 MP, matched the plotted native bytes. Frozen-file and output hashes were
verified after collection. The unsupported zero-precision/prediction-aware
arm is not included in these results.

The table compares each policy with **ordinary rounding, zero extra bits,
smoothing off at the same effort**, on the seven-image subset. This e4
reference is a diagnostic arm, not native e4.

| DC setting, smoothing off | e3 BD-rate change | e4 BD-rate change |
|---|---:|---:|
| Ordinary rounding, one extra bit | −6.328% | −4.649% |
| Prediction-aware rounding, one extra bit | −4.762% | −3.439% |

Thus the finer DC grid supplies the gain along this supported intervention
path; prediction-aware rounding gives back part of it in the aggregate.
This is a controlled policy result, not an assertion that rounding has the
same effect on every input or at every precision. The untested zero-bit
interaction remains unspecified.

Ordinary rounding with the extra bit and smoothing off measures **−2.661%
against libjxl e3** and **−0.817% against libjxl e4** on these seven inputs.
It wins on 6/7 and 4/7 respectively. All curves are supported; individual
residuals still reach +0.707% at e3 and +2.569% at e4. These subset means
are not a claim of superiority on all 65 original images or other metrics.

DC coefficients represent low-frequency block content. A source-backed
interpretation is that the original DC/AC precision balance is inefficient
under this score range: spending more precision on DC permits a better
overall rate-quality curve. The controlled grid change establishes that
rate effect; the completed section accounting below quantifies how DC cost,
AC savings and AQ field changes combine.

### Completed full-corpus confirmation

The selected candidate is **ordinary DC rounding, one extra precision bit,
adaptive smoothing off**, with native AQ counts preserved separately at e3
and e4. The frozen run is `build/e3-e4-dc-grid-confirm-20260914/`, using the
qualified diagnostic workflow overlay and the plotted build's immutable
libraries and embedded shader.

All 1,820 cells completed across 65 original inputs and all seven original
settings: 910 new candidate outputs, 130 fresh native Q80 byte/decode/score
controls, and 780 verified saved native observations. The audit rechecked
324 frozen files, every output and applicable raw record, expected cell
identities, and 2,080 successful encode/decode commands. Every primary
75–85 comparison has all 65 supported curves, with no reversals.

| Comparison over SSIMULACRA2 75–85 | PCHIP | Akima | Improved images |
|---|---:|---:|---:|
| Candidate e3 / libjxl e3 | **−2.512%** | −2.461% | 62/65 |
| Candidate e4 / libjxl e4 | **−2.042%** | −1.982% | 55/65 |
| Candidate e3 / native GJXL e3 | −4.477% | | 65/65 |
| Candidate e4 / native GJXL e4 | −2.827% | | 63/65 |
| Zero-AQ candidate e3 recipe / libjxl e4 | −2.487% | | 62/65 |

The last row is explicitly a cross-effort recipe comparison. Holding ordinary
DC, its extra bit and smoothing off fixed, the one-AQ e4 candidate costs
**+0.463%** relative to the zero-AQ candidate on average; it improves 28/65
inputs, and its worst individual regression is +6.358%. This does not justify
a per-image switch without an actual validated selector.

| Cohort | Images | Candidate e3 / libjxl e3 | Candidate e4 / libjxl e4 |
|---|---:|---:|---:|
| Kodak | 24 | −2.643% | −1.912% |
| CLIC | 32 | −2.269% | −1.858% |
| 12 MP | 3 | −2.607% | −2.714% |
| 24 MP | 3 | −2.532% | −2.461% |
| 48 MP | 3 | −3.939% | −3.963% |

The candidate's worst residual against stock is +1.126% at e3 and +2.578%
at e4. Against native GJXL, e4's worst regression is +0.798%. Thus the
aggregate same-effort gap is eliminated and reversed at the claimed metric
and quality interval, while individual residuals remain. These are the
original 65 inputs, not a new held-out corpus.

Using the plot's **libjxl e7 reference**, rather than matching efforts, native
e3/e4 average +3.873%/+2.522%; the candidate averages **−0.869%/−0.375%**.
The 48 MP means improve from +10.859%/+8.808% to +3.672%/+3.942% and still
trail e7. CLIC also retains +0.686%/+1.761% against e7, while Kodak and the
12/24 MP cohorts cross below zero. These are direct per-image curve
comparisons, retained in `comparison-to-plotted-e7-reference.json`.

The saved stock sweep does not support a general low-quality conclusion:
there are zero full-resolution stock comparisons spanning 25–35, and only
12/65 or 16/65 spanning 45–55 for the candidate e3/e4 comparisons. Missing
brackets are recorded explicitly; there is no extrapolation or hidden
subset mean. Fresh full-resolution low-quality sampling is needed to extend
the claim.

Artifacts: `PROTOCOL.md`, `manifest.json`, `observations.jsonl`,
`audit.json`, `analysis.json`, and the figure below in
`build/e3-e4-dc-grid-confirm-20260914/`. The full audit/report command is
`python3 tools/low_effort_rate/confirm_report.py`.

![Full-corpus DC grid confirmation](../build/e3-e4-dc-grid-confirm-20260914/dc-grid-confirmation.png)

The earlier `build/e3-e4-rate-confirm-20260914/` protocol for prediction-aware
candidates was prepared but never collected; the completed precision split
motivated the separate ordinary-rounding candidate. Both are retained with
their original manifests. No production defaults have changed.

### Current-frame mechanism and independent quality checks

A separately linked encoder-object capture re-emits the selected native DC
groups and requires exact byte and bit-count identity before reporting
section costs. Its 48 fixed-distance cases reproduce the already measured
outputs exactly; four repeated captures also reproduce every diagnostic
field. DC global, DC tokens/headers, metadata tokens/headers, AC sections,
and frame/TOC/padding sum exactly to file size. Model diagnostics overlap
these totals and must not be added again. Quantizer-token costs are
selected-model estimates, not an exclusive metadata-section measurement.

Across all 12 diagnostic inputs at original Q80, the e3 extra-bit intervention
preserves the raw quantization field, sharpness, CfL maps, quantizer scale,
AC coefficient fingerprints and exact AC section bit count. Only the DC
quantized values and their coding change materially. Mean score rises by
1.681 points while file size rises by 5.096% at this fixed distance. This
is a precision-for-quality purchase, not direct lossless compression.

Source confirms why: `codec/dc_quantization.cpp:126` and
`gpu/metal/kernels/dc_processing.h:124` multiply inverse DC steps by
`2^extra_precision` and divide reconstruction steps by the same factor.
One extra bit halves the DC grid spacing. Prediction-aware quantization
then coarsens sufficiently large predicted residuals back to even integers
(`dc_processing.h:169`), explaining why selecting that mode is not the same
as uniformly retaining the finer ordinary-rounding grid. The measured
supported precision split isolates the aggregate effect of this behavior.

At e4, changing the DC policy also changes the AQ reconstruction used by
the refinement. All 12 fixed-distance cases change their final quantization
field, global scale and AC coefficients. That is an observed downstream AQ
interaction; the e4 effect cannot be described as DC-bit savings alone.
These observations are retained in
`build/e3-e4-frame-probe-20260914/fixed/` with a passed audit.

At matched score, the accounting directly shows the exchange. In the nine
accepted e3 pairs, DC tokens add **4.329 percentage points of native file
size**, AC sections save **7.601 points**, and all remaining components sum
to +0.029 points: net **−3.242%**. DC tokens increase and AC sections decrease
in every accepted pair. For the eleven accepted e4 pairs, the corresponding
contributions are **+1.502 points DC**, **−4.300 points AC**, and +0.072 points
elsewhere: net **−2.726%**. These additive numbers are arithmetic means of
per-image contributions normalized to each native file. They are not BD-rate
or geometric mean rate ratios. All 48 matched-output captures passed native
byte, section-sum and repeated-diagnostic checks; unmatched pairs are retained
but excluded from this matched-quality allocation summary.

The completed independent check in `build/e3-e4-cross-metric-20260914/`
calibrated 12 preselected inputs and six arms to actual SSIMULACRA2
80 ± 0.025, with at most 14 new probes per target. Conventional Butteraugli
uses decoded linear-sRGB floats and an explicit 80-nit intensity target.
The audit passed: 72 terminal targets, 61 matched, 11 unresolved, 387 fresh
scored probes, 24 exact stock byte/decode/score controls, 867 verified seed
and probe outputs, and 967 successful external commands. The initial stock
schema assertion failure was corrected before any calibration targets;
the original script, manifest, failure and exact amendment are retained.

| Actual score-80 comparison | Accepted pairs / 12 | Bytes | Butteraugli error |
|---|---:|---:|---:|
| Candidate e3 / native GJXL e3 | 9 | −3.248% | **+10.445%** |
| Candidate e3 / libjxl e3 | 9 | −2.214% | **+5.891%** |
| Candidate e4 / native GJXL e4 | 11 | −2.732% | **+4.482%** |
| Candidate e4 / libjxl e4 | 8 | −0.411% | −0.949% |

These are geometric mean ratios over each explicitly accepted subset.
Different rows can contain different images; their means are not additive
and cannot be compared as if coverage were identical. Lower Butteraugli
means less error. Actual paired score differences are at most 0.046 points.
The candidate e3 matched all 12 targets and candidate e4 matched 11; native
e3/e4 matched 9/12 and 12/12, and stock e3/e4 matched 9/12 and 8/12. No failed
target received a reset budget or a relaxed tolerance.

The colored portrait `b51...` illustrates the material tradeoff: e3 saves
2.667% relative to native GJXL at scores 79.9792/79.9853, while Butteraugli
rises from 2.0759 to 2.6169 (+26.062%). A raw Butteraugli difference map
localizes its largest 32×32 mean error increase around facial detail. The
e3 policy is therefore a demonstrated improvement for the plotted metric,
not a general perceptual-quality improvement. E4's accepted stock comparison
is encouraging on both measurements, but its limited, selected coverage
does not establish a general Butteraugli rate win.

Six inspected native-pixel comparison panels cover the colored portrait,
river texture, the toy's smooth background, a dark portrait background,
lake-water gradients and campus concrete. Each decode hash matches the
measured output. The display conversion is linear-sRGB to clipped 8-bit
sRGB; it is not fed to either metric. The panels show the expected loss of
fine detail and low-frequency contour changes across recipes. They are
illustrative inspections, not a blinded subjective validation or evidence
that one recipe wins visually on every image. See `visual/index.json`,
`visual/*.png`, and `butteraugli-maps/e3-error-localization.png`.

![Native-pixel portrait comparison](../build/e3-e4-cross-metric-20260914/visual/0bab54a4882655fdb7d05334.png)

The retained unresolved brackets also expose distance-to-quality jumps.
For example, native e3 on campus 48 MP crosses 80.587 to 79.732 over a
relative distance span of 0.00286%; stock e3 on the portrait crosses
81.721 to 79.581 over 0.00454%. Candidate e4's unresolved Kodak08 target
crosses 80.026 to 79.946 over 0.000286%. These bounded observations are
consistent with discrete quantizer transitions; they do not prove that
every unresolved target is globally unattainable. The fixed extra-bit
candidate introduces no content-selected global median or new distance
switch, but it still needs boundary qualification before promotion.

## Concrete implementation proposal

### Lossless writer cause isolated on current frames

The current-frame bridge now carries the actual extra DC precision into
libjxl's emitted group headers and state digest. It uses the retained
`e8ff0976`-derived precomputed-frame bridge (`466ee1b`), with frozen sources
and libraries, matching native filters, and independently verified decodes.
The unmodified old bridge omitted this precision and cannot be used for
the current fine-grid frames.

All **240** comparisons passed: 12 selected inputs × four native/candidate
recipes × five original distances. Every native codestream reproduces its
prior SHA256, every tail output reproduces the prior linear-sRGB decoded PFM
SHA256, and the bridge verifies copied state, repeated tail bytes and frame
immutability. All 12 curves support SSIMULACRA2 75–85.

| GJXL frame recipe | Same-frame libjxl writer / native writer BD-rate | Wins / 12 |
|---|---:|---:|
| Native e3 | −0.060% | 7 |
| Extra-bit ordinary e3 candidate | −0.086% | 9 |
| Native e4 | **−1.217%** | 12 |
| Extra-bit ordinary e4 candidate | −0.414% | 12 |

On this same twelve-input diagnostic cohort, native e3's +3.418% gap against
stock becomes +3.355% after swapping only the writer: it is predominantly a
frontend rate-allocation issue. Native e4's +1.571% becomes **+0.332%**: a
substantial part of that cohort's gap is lossless coding overhead. These
are direct curve comparisons; their arithmetic mean percentages are not
additive. They do not replace the 65-image means. Full artifacts are in
`build/e3-e4-tail-confirm-20260914/`.

The source exposes a specific difference. Libjxl's
`enc_ans.cc:1345` (`HistogramParams::ForModular`) enables a four-candidate
HybridUint search when extra DC precision is present. The candidates include
`(4,2,0)`, `(4,1,2)`, `(0,0,0)` and `(2,0,1)`, including an encoding that
retains sign/parity information in the entropy symbol. GJXL's balanced
`ans.cpp:2959` supplies only its fixed default mapping and uses prepared
symbol populations for that mapping. Prediction-aware quantization creates
structured residuals by rounding larger residuals to even integers; selecting
precision without the matching entropy search leaves useful structure
unexploited.

A second controlled intervention disables **only** libjxl's modular mapping
search, leaving the completed frame and remaining writer policy fixed.
All 96 outputs passed exact float checks; 48 neutral controls reproduced
both the previous native and previous tail bytes. At original Q80, the
search contributes **0.833 percentage points of native e4 file size** on
average: the writer difference moves from −1.214% with search to −0.380%
without it. This is an additive fixed-frame point attribution, not BD-rate.
The same switch contributes 0.000 points for native e3, 0.023 for the ordinary
extra-bit e3 candidate, and 0.013 for the ordinary extra-bit e4 candidate.
It is therefore particularly valuable for the native prediction-aware DC
residuals. These controls establish the mapping-search effect; assigning all
of it to one specific candidate mapping would require a further ablation.
See `build/e3-e4-tail-uint-confirm-20260914/`.

The 65-image/all-seven-distance confirmation of **native e4 with only the
writer replaced** completed in `build/e4-native-tail-full-20260914/`.
All **455** native byte and decoded-float identity checks passed, along with
910 external commands and the post-run verification of 1,390 frozen files.
All 65 primary curves are supported. The same-frame writer saves **1.172%**
BD-rate versus native (Akima: 1.171%) and improves every image; individual
savings span 0.550–2.001%.

Against stock libjxl e4, the full result changes from **+0.808% to −0.374%**
(Akima: +0.861% to −0.321%), with 45/65 wins. The cohort means are −0.172%
Kodak, −0.528% CLIC, −0.496% at 12 MP, +0.113% at 24 MP, and −0.703% at
48 MP. The worst individual residual remains +3.555%. This is an aggregate
same-effort win, not a claim that every image or cohort beats stock. Since
every sampled reconstruction is exactly the native one, this route preserves
its existing Butteraugli values and visual properties. It does not establish
matched-Butteraugli superiority over stock or production implementation cost.

### Recommended order

1. **Prioritize lossless e4 DC entropy work.** Add an explicitly scoped
   DC/metadata HybridUint search for the native extra-precision recipe,
   starting with the four measured libjxl candidates. Retain the current
   clustering and approximate histogram policy while isolating this change;
   measure exact serialized size including model overhead. GJXL's prepared
   fixed-symbol cache cannot simply be reused for other integer mappings:
   retain/rebuild the raw weighted DC token values for candidate evaluation.
   Keep the existing fixed-population AC path intact. Select no larger output
   than the original writer and require exact decoded floats. Then assess
   remaining AC context/model and DC-tree differences with the same frozen
   frames. The measured libjxl-tail budget is evidence for native work,
   not a claimed native implementation or speed result.

2. **Decouple DC grid precision from the rounding mode.** Introduce one
   internal policy resolver for the extra precision bit and pass it to both
   AQ reconstruction and the final codestream profile. The current public
   `prediction-aware` mode bundles two different decisions, which obscured
   the cause. Preserve explicit overrides and mode exceptions. Validate
   supported precision bounds and the prediction-aware/nonzero-precision
   constraint on CPU and Metal. The diagnostic workflow overlay demonstrates
   this wiring without replacing the original libraries or shader.
3. **Retain the measured candidate as an explicit experiment.** For normal
   photographic e3/e4 encoding, use ordinary DC rounding, one extra bit and
   smoothing off; preserve the effort's existing AQ count initially. This
   exact policy has already reversed the full-corpus SSIMULACRA2 gap. An
   unconditional default is not justified by the cross-metric tradeoff.
   First test ordinary extra-bit rounding with smoothing retained as a
   single-variable alternative, and independently evaluate DC versus AC
   allocation at matched Butteraugli. Do not silently choose a different
   recipe per image or add a hard global-median selector.
4. **Treat zero-AQ e4 as a separate rate/time decision.** Its mean
   SSIMULACRA2 rate is better than the one-AQ candidate, but AQ and filter
   behavior affect the independent perceptual result. Compare the same DC
   recipe with zero/one AQ under both metrics and complete-call timing.
   Do not assign the entire native e3/e4 difference to AQ or claim that
   removing a GPU pass guarantees an end-to-end speedup.
5. **Qualify a chosen default on a clean current-main build.** Freeze the
   recipe before a disjoint photographic corpus, add full-resolution
   samples bracketing the missing lower-quality bands, and include smooth
   ramps, near-flat colors, odd dimensions, DC/group boundaries and dense
   distance probes around quantizer transitions. Check decode validity,
   deterministic output, CPU/Metal profile propagation and effort/mode
   exceptions. Measure complete-call latency and memory with concurrent
   studies stopped naturally; report distributions, worst regressions and
   unresolved targets. Require the declared metric/visual tradeoff to be
   accepted before promoting a default.

The implementation target is concrete and already measured for the plot's
metric. Production promotion and a general claim of perceptual superiority
remain separate work. No production policy, commit, merge or publication has
been made.
