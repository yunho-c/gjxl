# Default DC integer mapping and context-map compression

The normal encoding workflow now adopts the qualified DC integer-mapping
search and native context-map compression. No extra CLI switch is needed.

| Feature | Default scope |
|---|---|
| DC integer-mapping search | Effort 4, automatic compression, default density, prediction-aware DC quantization, outside maximum-error control |
| Native context-map compression | Every native writer effort, including explicit compression modes |

The DC search keeps its qualified scope. Other efforts, ordinary DC rounding,
high density, maximum compression and maximum-error control keep their existing
DC entropy policies. Direct low-level frame serializers can still set
`VarDctCodestreamOptions::dc_uint_search` explicitly; the workflow resolves it
from the effort and mode. Context-map coding automatically chooses among the
legacy representation, fixed-width, raw/MTF, Prefix/ANS and bounded RLE choices.

The integration preserves current main's zero-AQ ordinary effort-4 frontend
and its AC token-writing improvement. CPU, resident Metal and compatibility
workflow admission use the same policy as execution. A regression assertion
checks that default effort 4 combines DC mapping search with zero AQ updates.

## Adoption check against current main

The baseline is `30bf1f3`. The candidate integrates the previously qualified
features with that revision, using separately built Release binaries. Both
arms have the same zero-AQ frontend; no old one-AQ quality scores are reused.

Twelve retained diagnostic images at their original Q80 distances all produce
smaller files, averaging **1.1863% fewer bytes** (range 0.7764–1.7275%). Each
pair decodes to exactly the same linear-sRGB float PFM. This is a mean of
fixed-reconstruction byte percentages, not BD-rate or held-out qualification.

Another 28 pairs cover CPU and Metal efforts 1–10, ordinary DC rounding, high
density, maximum compression and maximum-error control on Kodak 01. All have
identical decoded pixels. In total, 40 paired reconstructions and 160
qualification commands pass. The historical 65-image rate qualifications and
context-map decoder oracle remain in their separate reports.

The Release suite passes **138/139 tests**. The sole failure is the inherited
`quantization_pipeline` score mismatch, reproduced with the untouched main
baseline (`0.24919039011001587` versus pinned `0.24914586544036865`). No golden
score was changed. The install-consumer and execution/storage-policy checks
pass.

## Combined latency against current main

The separate timing check uses the same 12 retained operating points and the
frozen binaries above. Both arms run fully resident Metal with eight
participating CPU threads. Two blocks of four alternating process pairs per
image yield 192 accepted processes and 960 complete public encode calls,
after two warmups per process. Every output matches its decoded rate-run
codestream hash.

The median of the 12 per-image paired latency ratios is **1.06497×**, or
**+6.50% time**. Per-image medians range from +0.32% to +13.06%; the two block
medians are +6.54% and +7.73%. The four Kodak results range from +3.52% to
+13.06%, CLIC from +0.32% to +7.24%, and the larger scenes from +2.96% to
+10.14%.

This combined cost is larger than the earlier +0.59% context-map-only result.
That earlier comparison already had DC mapping search in both arms and used
the one-AQ frontend. The adoption comparison adds both features to current
main's faster zero-AQ frontend. The measured tradeoff for this diagnostic
selection is 1.19% fewer bytes at the same reconstruction for 6.50% more median
paired encode time. The two statistics use their stated, different aggregation
methods; this is neither a BD-rate measurement nor a whole-corpus speed claim.

The same predeclared CPU-pressure, competing-job, power and thermal checks as
the [context-map latency protocol](context-map-latency.md) govern admission.
One contaminated pair is excluded and retried, with both attempts retained.
All 96 accepted pairs pass the environment audit. The previously approved
temporary `suggestd` pause was used throughout this run; its watchdog resumed
the service after 223 seconds and the running state was verified. No pair
straddles the intervention. These are local warm-call measurements with
sampled environment checks, not a claim that all background work was absent.

## Evidence

The policy implementations and earlier qualifications are described in
[DC integer-mapping search](dc-uint-search.md),
[context-map compression](context-map-compression.md) and
[context-map latency](context-map-latency.md). Those earlier measurements used
the one-AQ frontend and retain their original scope.

The new adoption artifacts are retained separately in
`build/entropy-default-qualification-20260915/`: source archive, immutable
manifest, `qualify_defaults.py`, raw records, paired codestreams, command ledger,
scope controls, analysis and audit. The candidate and untouched main baseline
use `build/entropy-default-candidate/` and `build/entropy-default-baseline/`.
Timing and service records are in `build/entropy-default-latency-20260915/`.
[Committed results](entropy-defaults-results.json) retain per-image data,
validation counts, the source-manifest hash and artifact provenance.
