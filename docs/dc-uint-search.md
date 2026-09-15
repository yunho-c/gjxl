# E4 DC integer-mapping search

The balanced serializer now has a DC-only search over HybridUint mappings
`(4,2,0)`, `(4,1,2)`, `(0,0,0)` and `(2,0,1)`. Prediction-aware DC quantization
at e4 produces structured residuals whose low bits can be represented more
efficiently by the alternative mappings. This changes entropy representation;
DC precision, prediction, smoothing, AQ, AC coefficients and reconstruction
are unchanged.

The workflow enables the search at effort 4 with automatic compression,
default density, prediction-aware DC quantization, and a rate-control mode
other than maximum error. Other efforts, ordinary DC rounding, high density
and maximum compression retain their existing policies. The low-level
`VarDctCodestreamOptions::dc_uint_search` switch is false by default and
applies only to balanced DC ANS coding. Small streams selected for Prefix
coding retain that path.

The existing fixed-mapping populations still select Prefix versus ANS and
determine the balanced DC context clusters. The new path then aggregates raw
values by those clusters, retaining all integer information needed to search
alternative mappings. It keeps approximate histogram precision and uses
serialized model cost plus estimated token cost to select a mapping per
cluster and one alphabet width for the complete model. Equal estimated costs
prefer the incumbent mapping and the smaller width. Width selection includes
serialized histogram/configuration bits and estimated token bits. It builds
only the winning model; final emission supplies exact token counts. This is
a bounded heuristic search, not a guarantee of the globally smallest file.

AC retains the existing fixed-population fast path. Storage admission includes
the DC raw values, weighted counts, four mapping candidates, and one
alphabet-width model. CPU and resident Metal use the same policy resolver
for execution and admission.

## Validation

The new entropy tests check structured residual savings, full-range integers,
empty sections, unchanged clustering, equivalence of borrowed populations
and scanned streams, deterministic output with and without returned cost,
exact cost versus the independent ANS emitter, and atomic failure on invalid
contexts. Allocation-failure sweeps and bounded storage tests include the
new search. Whole-serializer and CPU/Metal workflow tests cover admission
and policy selection.

The original causal evidence is in the
[effort 3/4 investigation](e3-e4-rate-investigation.md). Native implementation
measurements are collected separately with
[`tools/dc_uint_search/qualify.py`](../tools/dc_uint_search/qualify.py).

## Native rate qualification

On the original 65 images and seven distances (455 cases), every freshly
built baseline reproduced the pinned native codestream bytes, and every
candidate reproduced its decoded linear-RGB float hash. All 455 candidate
files are smaller. Exact reconstruction identity preserves both the existing
SSIMULACRA2 scores and the output seen by other quality metrics.

| Comparison, SSIMULACRA2 75–85 | PCHIP mean BD-rate | Akima mean BD-rate |
|---|---:|---:|
| New native writer / native baseline | −0.805% | −0.805% |
| New native writer / stock libjxl e4 | −0.003% | +0.049% |

All 65 per-image curves improve against the native baseline; 38/65 beat
stock libjxl e4 with PCHIP. The aggregate stock comparison is effectively a
tie and is sensitive to interpolation. These are arithmetic means of
per-image BD percentages with complete coverage and no extrapolation. The
corpus is the original investigative cohort, not held-out data; the large
images represent three scenes at three sizes.

The implementation reduces native BD-rate by about 0.805%. The earlier
1.172% saving from replacing the entire writer remains a separate result;
the remaining writer differences are not implemented here.

Release validation passed 137/138 CTest checks. The sole failure,
`quantization_pipeline`, reproduces exactly when linked against the untouched
baseline libraries: score `0.24919039011001587` versus pinned
`0.24914586544036865`. No golden value was changed. The new focused entropy,
storage and DC policy checks all pass. Additional CLI comparisons at distance
3 on Kodak 01 verify identical decoded pixels for CPU and Metal e4 and
byte-identical output for 24 controls covering efforts 1–3 and 5–10, ordinary
DC rounding, high density and maximum compression.

Retained local evidence: `build/dc-uint-fast-qualification-20260914/`, including
the immutable manifest, source patch, baseline/candidate codestreams,
command ledger, raw samples, `analysis.json`, `audit.json`, `scope.json` and
`environment.json`. The baseline is `b1fbfc1` plus research-only commit
`6a8f8fd`; executable and source hashes distinguish the candidate built from
that checkout's implementation diff. The large corpus and build artifacts
are local and are not included in Git. The first implementation, which
constructed and measured multiple alphabet-width models, is archived in
`build/dc-uint-qualification-20260914/`. Its paired timing showed substantial
overhead, motivating the single-model estimate. The final selection produces
identical codestreams in 431/455 comparisons with that implementation; the
largest additional size is four bytes, and the mean BD-rate difference is
less than 0.00002 percentage points with PCHIP.

## Complete-call timing

The final run uses the 12 declared diagnostic images at their original Q80
distances, eight participating CPU threads, four alternating process pairs,
two warmups and five measured encodes per process. All 96 processes reproduce
their rate-run codestream hashes. The 480 samples measure complete public
encode calls, including final output publication.

The median of the 12 per-image paired latency ratios is **1.023×** (+2.3%
time). Per-image medians range from 0.990× to 1.076×. The initial multi-model
implementation observed a 1.103× median under its separate paired run; the
single-model selection retains essentially the same compression gain.

These are observations on an AC-powered Apple M4 Pro, not isolated latency
qualification. Other codec work was active in 22/34 sampled process snapshots,
including Butteraugli and quality scoring. Small deltas and the apparent
speedup on one input should not be interpreted as precise speed claims. An
uncontended run remains necessary for a firm latency qualification before
main-branch promotion.

[The committed results](dc-uint-search-results.json) include all 65 per-image
BD comparisons, the 12 timing summaries and paired ratios, validation counts,
tested source hashes and artifact provenance. All 1,517 final qualification
commands succeeded. The scoped e4 policy is enabled on `feat/dc-uint-search`;
this work does not merge or publish it to main.
