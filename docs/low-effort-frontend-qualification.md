# Low-effort frontend qualification

Qualified on Apple M4 Pro in a Release build, based on `8e8c49a`, on
September 13, 2026. The default effort policy is:

| Effort | Gaborish | Initial quantization | AQ updates |
| --- | --- | --- | --- |
| 1–3 | Off | Uniform `0.79 / distance` | 0 |
| 4 | Off | Uniform `0.79 / distance` | 1 |
| 5–6 | On | Existing spatial field | 1 |
| 7 | On | Existing spatial field | 2 |
| 8–9 | On | Existing spatial field | 3 |
| 10 | On | Existing spatial field | 4 |

This adopts libjxl's e4/e5 Gaborish/initial-field boundary; it does not claim
that every effort heuristic is identical to libjxl. High-density,
maximum-error, and dedicated maximum-throughput overrides preserve their
existing initialization policy. Target-size retries use the selected effort
policy. EPF remains at two passes.

CPU and Metal initialize the uniform field directly. Metal skips gradient,
erosion, spatial modulation, and mask convolution, while retaining the
existing quantizer selection and CfL logic. Diagnostics receive constant
masking planes. Existing scratch allocation bounds remain conservative.
The serializer accepts Gaborish-off frames and reserves the longer frame
header. Default Gaborish-on header bytes remain unchanged.

## Production replay

240 production encodes reproduced the previously approved diagnostic
codestreams byte-for-byte. All 240 decoded PFMs also matched exactly using
the pinned libjxl decoder. The replay includes:

- 150 cases: 15 inputs × five original sweep distances × e3/e4.
- 30 cases: e1/e2 against the qualified zero-update output at distance 1.9.
- 60 cases: 15 inputs × e3/e4 × retained points nearest measured scores 30/50.

The diagnostic reference for e4 is Gaborish off, uniform initial field, and
one AQ update. Efforts 1–3 use its zero-update counterpart. The implementation
therefore preserves the measured reconstruction/rate behavior while removing
the diagnostic's unnecessary spatial initialization work. No new latency
claim is made by this replay.

Another 20 old/new pairs matched byte-for-byte: Kodak 01 at efforts 5–10,
and the CLIC portrait and 12 MP alpine-lake input at efforts 5 and 7, each
at distances 1.9 and 6.4.

## Tests and known baseline failures

- Release build completed.
- CTest: **136/137 passed**. The new uniform CPU pipeline test, CPU/Metal
  field/mask/quantizer comparisons, effort boundary/update-count tests,
  workflow/storage/admission tests, and conformance smoke passed.
- The failing `quantization_pipeline` hard-edge golden score is inherited:
  actual `0.24919039011001587`, expected `0.24914586544036865`. Compiling the
  old test against the old source headers and libraries reproduced the
  same score, quantization field, raw quantization values, and failure.
- Full conformance: the new Gaborish-off odd-size fixture passed pinned
  decoding with maximum pixel error `5.78165e-06`. All other fixtures passed
  except the inherited `single-block-impulse` stored-hash check. Both old and
  new builds produce hash `10525847038797353681`, versus the stored
  `11459255244783164287`. The fixture's early hash check prevents its later
  pixel comparison; this is not counted as a fully passing conformance run.
- `git diff --check` passed. Existing unrelated main-checkout changes were
  preserved. No Python test files are part of this change.

The decoder revision is `e8ff09762481785938d8e4e01333ed3917571161`.
Full conformance used that decoder and the installed `jxlinfo` metadata
utility. The rate studies used fast-ssim2
`c3867954c7bec8a761951df9256354b305fd0cff` and linear-sRGB float PFMs.

## Retained evidence

Production artifacts are under
`/Users/yunhocho/GitHub/gjxl-low-effort-frontend/build/release/qualification/`:

- `replay-manifest.json`, `replay.jsonl`, `replay-summary.json`, and
  `replay-commands.jsonl`: provenance, encoded/decoded hashes, and commands.
- `replay/`: retained production codestreams and benchmark records.
- `higher-efforts/results.json`: the 20 unchanged higher-effort pairs.
- `baseline-quantization-pipeline.log`, `baseline-conformance.log`, and
  `conformance.log`: direct baseline and conformance evidence.
- The parent `ctest-final.log` contains the final 137-test run.

The accepted rate tradeoffs and original study locations are documented in
[the low-effort policy](low-effort-dct8.md#uniform-field-qualification).
