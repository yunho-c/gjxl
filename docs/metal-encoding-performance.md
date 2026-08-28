# Metal encoding performance roadmap

This document owns the work required to make the complete in-memory VarDCT
encoder extremely fast on Apple GPUs. The target is a `50x` warm-backend
speedup over the current CPU workflow, measured from linear RGB input through
the final raw JPEG XL codestream bytes. File I/O is outside this boundary;
input preparation, backend handoff, quantization, synchronization, final
materialization, tokenization, entropy coding, and byte assembly are inside it.

[`metal-aq.md`](metal-aq.md) remains authoritative for adaptive-quantization
correctness and residency contracts. [`codestream.md`](codestream.md) remains
authoritative for the supported bitstream profile. This roadmap is the
cross-stage performance contract and must not turn a leaf-kernel speedup into
an encoder claim.

## Success criteria

The primary throughput gate is:

- Apple M4 Pro, Release build, SIMD-group Metal implementation.
- Padded 1080p and 4K workloads at Butteraugli target `1.2`.
- The same public `EncodeLinearRgbVarDctCodestream` boundary for CPU and Metal.
- One already-created Metal backend for the primary throughput number; cold
  backend creation is measured and reported separately.
- At least three independent processes. Each process warms both paths, then
  alternates CPU/Metal order for at least five paired samples.
- The reported speedup is the range of paired process medians, not a ratio of
  unrelated best cases.

Two accuracy tracks are explicit:

- `exact-coefficients` is the production track. Raw quantization, encoder frame,
  and codestream bytes remain exact; existing numerical gates are not widened.
- `fully-resident` is the throughput track. Byte identity is not required, but
  output must be deterministic for a fixed backend, structurally valid,
  independently accepted by the pinned `djxl`, finite after decoding, and
  measured for Butteraugli/decoded-pixel drift. Any policy or quality change is
  reported instead of being hidden behind a tolerance.

The `50x` objective may be satisfied first by the throughput track. Production
rollout remains separately gated on the exact track.

## Current baseline

The Milestone 9 Apple M4 Pro measurements in [`metal-aq.md`](metal-aq.md)
established the following 1080p public-workflow range:

| Path | Median range | Speedup |
| --- | ---: | ---: |
| CPU public workflow | 6345.9-6374.5 ms | 1.00x |
| Exact-coefficient Metal public workflow | 1084.1-1117.8 ms | 5.70-5.87x |

The experimental fully resident AQ operation reached `12.2-12.8x` for AQ
itself, not for the complete public encoder. A `50x` 1080p result requires a
complete time no greater than approximately `127 ms` against this CPU baseline.

## Measurement interface

`VarDctEncodingProfile` measures the public workflow without adding clock reads
to ordinary calls. It reports:

- input padding and linear-RGB-to-opsin preparation;
- backend selection;
- the complete quantization pipeline;
- codestream encoding;
- summary assembly; and
- total wall time.

The nested `VarDctCodestreamProfile` separates validation, DC tokenization, AC
tokenization, entropy optimization, section writing, and final assembly.
Profiles commit only after a successful encode and never weaken output
atomicity.

Use the focused benchmark while iterating:

```sh
just encode-benchmark padded_1080p simd 5 1 exact-coefficients
just encode-benchmark padded_1080p simd 5 1 fully-resident
just coefficient-benchmark padded_1080p 9 2
```

The benchmark performs one correctness validation, alternates CPU/Metal order,
prints every profile boundary, and reports paired speedups. The broader
`just aq-benchmark` matrix remains the correctness and exploratory phase gate.

## Ordered implementation plan

### P0. Establish the encoder profile and fast iteration loop - complete

- Profile the complete public workflow and every codestream subphase.
- Add a focused public-workflow benchmark that avoids running unrelated AQ
  diagnostic phases.
- Preserve ordinary API performance and atomic failure behavior.

### P1. Remove repeated CPU coefficient-staging overhead - in progress

- Replace per-transform heap allocation with reusable, maximum-strategy scratch.
- Prepare invariant forward transforms once per selected strategy grid and
  share them across final CfL and exact coefficient production.
- Fuse exact coefficient production with the packed reconstruction handoff where
  doing so avoids a second frame traversal.
- Retain exact raw quantization, frame, and codestream output.

Exit criterion: balanced profiles show a stable reduction in exact-coefficient
Metal AQ and public-workflow time with exact output.

The first retained P1 change replaces per-anchor coefficient, quantized-value,
pixel, and DC allocations with one maximum-strategy scratch arena per coding
call. Three interleaved A/B pairs on the padded 1080p workload used two warmups
and nine samples per process. Baseline/reused-scratch medians were
`96.076/93.288`, `96.920/92.912`, and `94.055/93.413` ms, or directional wins
of approximately `0.7-4.3%`. Sample ranges overlapped, so this supports the
narrow allocation change rather than a complete-pipeline speedup claim.

The second retained P1 change caches the three forward-transform planes after
strategy selection. Final CfL and exact coefficient coding reuse the cache for
all three AQ evaluations. The cache adds approximately `24.9 MB` of 1080p host
coefficient storage. In one balanced iteration process with one warmup and
three alternating samples, exact Metal public-workflow time fell from a prior
`1148.9-1162.5 ms` range to `712.9-752.9 ms`; the median paired speedup rose
from `5.64x` to `9.00x`. Exact codestream bytes remained unchanged. The same
fully resident check fell from `785.9-801.5 ms` to `617.3-628.9 ms`, reaching
a `10.28-10.49x` paired range. These are iteration results, not the independent-
process P6 claim.

### P2. Provide a fast GPU coefficient decision path

- Keep the current float fully resident implementation as the throughput
  baseline.
- Investigate decision-equivalent accumulation, selective high-precision repair,
  or a bounded throughput policy instead of assuming float ties are exact.
- Retain both explicit accuracy tracks; automatic selection remains exact until
  the production gate passes.

Exit criterion: the throughput track removes CPU forward transforms and
coefficient coding from every AQ evaluation and has independent decoder/quality
evidence. The exact track advances only if its decision contract passes.

The first throughput-policy slice replaced the iterative initial-CfL search
with a deterministic one-pass DCT-domain regression. On the padded 1080p workload, one
alternating process with one warmup and three samples measured Metal public
workflow at `512.0-542.8 ms` and paired speedup at `11.71-12.47x` (median
`11.92x`). Against exact coefficients, the full-pipeline score-history maximum
was `0.006122`; final quant field, block map, and reconstructed-RGB maxima were
`0.128`, `0.404`, and `0.913`. The codestream changed from `630517` to `630802`
bytes. This is an explicit quality-policy trade, not an exact-path optimization;
independent decode and Butteraugli checks remain part of P6.

That seed was subsequently replaced by a tilewise pixel-domain regression,
removing its CPU DCT pass while retaining per-tile luma/chroma covariance and
the final perceptual AQ loop. One alternating padded-1080p process with one
warmup and three samples measured Metal at `414.8-439.8 ms` (median `417.4 ms`)
and paired speedup at `14.47-15.26x` (median `15.19x`). The pipeline median was
`343.6 ms`; the codestream changed to `617220` bytes. Against exact
coefficients, full-pipeline maxima were `0.202` for the final field, `0.406` for
the block map, `0.0540` for score history, and `0.978` for reconstructed RGB.
This approximately `47 ms` public-boundary win increases the score delta from
the preceding DCT-domain seed, so it remains throughput-only and requires P6's
independent decoded-quality evidence before any broader use.

### P3. Make the whole frontend resident

- Introduce one prepared frame context shared by input conversion, initial quant,
  Gaborish, first-pass CfL, AC search, and iterative AQ.
- Stop uploading the same opsin image and allocating unrelated scratch arenas at
  the search-to-AQ boundary.
- Keep strategies, quant fields, and coefficients device-resident until the
  final frame or entropy handoff requires them.

The first retained P3 slice moves inverse Gaborish from the CPU to a mirrored-
boundary Metal 5x5 primitive for fully resident mode. Direct CPU/Metal coverage
observed `2.98e-7` maximum error, including aliased output and atomic invalid-
input behavior. A warm padded-1080p public benchmark with one warmup and three
alternating samples measured Metal at `445.6-467.5 ms` (median `464.1 ms`) and
paired speedup at `13.71-15.12x` (median `13.88x`). The quantization-pipeline
median was `390.2 ms`, and the codestream remained `630802` bytes relative to
the preceding fast-CfL slice. This is still a host/device round trip rather than
the prepared frame context required by the P3 exit criterion.

### P4. Remove per-evaluation CPU synchronization

- Port final CfL, EPF inverse sigma, and deterministic quant-field updates.
- Execute all three default AQ evaluations under one resident orchestration
  boundary when policy dependencies allow it.
- Read back only final policy results and the requested final frame/image.

### P5. Parallelize the codestream tail

- Parallelize independent DC/AC group tokenization and section writing.
- Reuse coefficient orders and token storage instead of rebuilding them per
  group.
- Profile histogram optimization separately before considering a GPU entropy
  implementation.

### P6. Close the 50x gate

- Run the required independent-process 1080p and 4K benchmark matrix.
- Validate deterministic output, pinned-`djxl` acceptance, decoded pixels, and
  Butteraugli drift for the winning accuracy track.
- Record warm/cold latency, device memory, peak scratch, output size, and every
  profile phase. A missing phase or excluded transfer invalidates the claim.

## Stop rules

- Do not optimize a standalone DCT, Butteraugli, or AC-search kernel unless the
  public profile identifies it as a material part of the remaining budget.
- Do not call a faster but decision-changing path exact.
- Do not use one-shot or sequential before/after timings for a retained
  optimization.
- Do not widen an existing production tolerance to make a performance patch
  pass; use the explicit throughput track when accuracy is intentionally traded.
