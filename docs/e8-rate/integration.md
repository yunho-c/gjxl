# Effort-8 writer integration

Implemented on `feat/e8-rate` above `b1fbfc15f32a3c3ac3d1d2b5b7983d0ced22a00f`,
following the decision to prioritize rate before speed. This is item #1 from
the rate investigation. CfL and transform-family work remain separate.

## Encoding policy

Ordinary effort 8 now resolves to `VarDctEntropyBehavior::kRateOptimized`.
Its expanded candidate uses the demonstrated high-density partition and
28 HybridUint configurations per cluster, evaluates every valid ANS alphabet
width from 32 through 256 symbols, and selects by model signaling plus exact
ordered token cost. Widths unable to represent the input are skipped. The
same model is selected whether or not the caller requests a returned cost.

The serializer first produces the established balanced encoding, then the
expanded encoding from the same borrowed completed frame. It publishes the
smaller complete codestream; ties retain balanced bytes. This comparison
includes headers, padding, and the TOC. Failure in either attempt preserves
the caller's output and profile. The fallback protects rate; it does not
suppress allocation or validation failures.

Efforts 1–7 keep balanced entropy, and efforts 9–10 keep the existing
high-density policy. Explicit high-density and maximum-compression overrides
retain their existing behavior. Coefficients, AQ iterations, coefficient-order
policy, and transform support are unchanged by this writer integration.

The storage planners include up to four concurrent width models, measured-cost
arrays even without a returned cost, and the balanced published bytes retained
while the expanded search runs. The two serializer attempts execute serially;
the plan combines their peak envelopes rather than retaining both full working
sets. Profiling counts actual work from both attempts while model/representation
statistics describe the selected stream. It also records both complete sizes
and whether balanced fallback was selected.

## Validation

The fresh Release build is `build/e8-integrated`; the original `build/e8`
archives and private-prototype binaries remain frozen. Build provenance is
retained in `build/e8-integrated/build.json`.

- Direct-ANS tests cover all four usable widths, invalid small-width cases,
  full-width uint32 values, measured-versus-written costs, deterministic
  model selection with and without returned cost, and a strict rate gain.
- Serializer tests cover both expanded selection and balanced ties, complete
  byte fallback, profile metadata/work counts, deterministic bytes, borrowed
  frame parity, managed memory reservations, and allocation-failure atomicity.
- Expanded entropy storage tests exercise both returned-cost and no-cost
  calls on populated streams, including allocation-failure sweeps.
- The complete Release CTest suite passed 137/138. Its only failure,
  `quantization_pipeline`, was reproduced from unchanged HEAD source linked
  against the frozen baseline libraries: actual `0.24919039011001587` versus
  pinned `0.24914586544036865`. Baseline reproduction is retained under
  `build/e8-integrated-baseline-check`.
- Full codestream conformance passed with the study's pinned decoder. The
  new behavior is included alongside balanced and high-density encoding for
  the conformance fixtures, with artifacts in
  `build/e8-integrated/pinned-conformance`.
- A retained Kodak20 check at distance 1.9 confirmed byte-identical baseline
  and integrated outputs at every other effort: 1–7, 9, and 10. This is a
  bounded policy-regression check, not corpus-wide qualification of those efforts.

The separate full-corpus production validation is collected by
[`integrated_rate.py`](../../tools/e8_rate/integrated_rate.py). It independently
encodes and decodes every point, requires exact decoded equality with the
baseline, and requires byte-for-byte replay of the smaller frozen baseline or
prototype output. Its manifest, ledger, commands, and analysis live under
`build/e8-integrated-rate`.

## Full-corpus result

The production binary passed all **390 points across 65 images**, at distances
0.55, 1, 1.9, 2.8, 4.6, and 6.4. Every output is byte-identical to the smaller
expanded-search prototype output and decodes to the baseline's exact pixels.
All 65 image curves improve; all 390 individual streams are smaller than the
baseline. The balanced fallback is covered by the flat-frame tie fixture.

On measured fast-ssim2 [75,85], with no extrapolation, the arithmetic mean of
per-image PCHIP BD-rates is **-1.020231% versus baseline GJXL** and
**-0.861853% versus stock libjxl e8**. Akima sensitivity gives
-1.018807% and
-0.858542%, respectively. These reproduce the
prototype's rate gain exactly. See [rate analysis](integrated-rate.json) and
[integration validation](integration-validation.json) for retained provenance.

The three test targets rebuilt after the final test-coverage changes
(`entropy_storage_plan`, `codestream_encoder`, and `vardct_frame_view`) also
passed. Exact decoded PFM equality was verified for all 22 conformance fixtures.

## Runtime boundary

This integration deliberately retains the full search. It also performs the
balanced fallback serialization, so the prototype's previously measured
26–97% time overhead must not be used as the integrated policy's runtime cost.
No new latency qualification or speed optimization is claimed. Search screening
and reuse of token/model work can be assessed in a later speed-focused change
against this rate-qualified behavior.
