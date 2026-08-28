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

Four accuracy tracks are explicit:

- `exact-coefficients` is the production track. Raw quantization, encoder frame,
  and codestream bytes remain exact; existing numerical gates are not widened.
- `fully-resident` is the resident-quality track. Byte identity is not required,
  but output must be deterministic for a fixed backend, structurally valid,
  independently accepted by the pinned `djxl`, finite after decoding, and
  measured for Butteraugli/decoded-pixel drift. Any policy or quality change is
  reported instead of being hidden behind a tolerance.
- `throughput` is a more aggressive opt-in policy layered on the resident
  evaluator. It performs one AQ update instead of the default two and reports
  its additional size and quality drift separately.
- `maximum-throughput` is the explicit speed-first track. It uses only DCT8,
  quantizes the adjusted initial field directly on Metal, and omits inverse
  reconstruction and perceptual AQ scoring. Its score history is therefore
  empty, and decoded quality must be measured independently.

The `50x` objective may be satisfied first by the maximum-throughput track.
Production rollout remains separately gated on the exact track.

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
just encode-benchmark padded_1080p simd 5 1 throughput
just encode-benchmark padded_1080p simd 5 1 maximum-throughput
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

The third retained P1 change enumerates and validates the strategy layout
serially, then prepares independent forward transforms on up to eight host
workers. Workloads below 256x256 coefficients stay serial, and inability to
start the worker set falls back to the serial implementation. A 256x256
contract test compares the prepared result with direct coefficient coding.
One padded-1080p process with one warmup and three alternating samples measured
the fully resident Metal workflow at `295.8-328.2 ms` (median `309.6 ms`) and
paired speedup at `18.92-20.86x` (median `20.14x`), versus the preceding
`372.1 ms` Metal median. The exact-coefficient workflow measured
`601.8-623.8 ms` (median `607.6 ms`) and retained its `630517` byte codestream,
reaching a `10.23x` paired median. These are single-process iteration results;
the P6 independent-process gate remains open.

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

The second retained P3 slice defers exact-coefficient staging and the full
reconstruction/postprocess diagnostic readbacks until an operation actually
requests them. A padded 1920x1080 throughput preparation therefore avoids
allocating and value-initializing three unused `3 * pixel_count` float stores,
or approximately `71.2 MiB` of host memory. The normal final coefficient, DC,
and linear-image readbacks remain eager because the public workflow consumes
them. Lazy allocation occurs only after the prepared evaluator atomically owns
its operation slot, and allocation failure restores the ready state. One warm
padded-1080p process with one warmup and three alternating samples measured
Metal at `371.8-374.4 ms` (median `372.1 ms`) and paired speedup at
`16.58-16.74x` (median `16.59x`). The prior retained result was not rerun as a
same-process binary A/B, so this timing is directional; the deterministic
memory reduction and unchanged output contracts are the retained claims.

The third retained P3 slice parallelizes the independent RGB-to-opsin rows on
up to twelve host workers while preserving each pixel's operation order and
the existing scratch-image atomic commit. Images below 256x256 stay serial and
worker-start failure falls back to a complete serial conversion. On the padded
1080p public workflow, input preparation fell from the preceding `27.7 ms`
median to `9.2-11.8 ms` (median `10.6 ms`). One process with one warmup and five
alternating samples measured Metal at `278.1-297.9 ms` (median `292.1 ms`) and
paired speedup at `20.54-22.03x` (median `21.02x`). Exact pixel conversion and
the existing frame/codestream contracts remain unchanged; this is still an
iteration result rather than the P6 process matrix.

The fourth retained P3 slice computes the initial quantizer's independent
four-row masking bands on up to twelve host workers. Each band preserves the
original vertical and horizontal accumulation order, small images stay serial,
and a 256x256 contract test covers finite output plus atomic invalid-input
behavior. A single full-profile sample reduced the standalone padded-1080p
initial-quant field from the preceding `42.7 ms` observation to `32.7 ms`.
One process with one warmup and three alternating public samples measured
Metal at `259.7-299.4 ms` (median `271.9 ms`) and paired speedup at
`20.30-23.37x` (median `22.22x`). The range remains noisy, so the exact
four-row decomposition and unchanged output are the primary retained claims.

### P4. Remove per-evaluation CPU synchronization

- Port final CfL, EPF inverse sigma, and deterministic quant-field updates.
- Execute all three default AQ evaluations under one resident orchestration
  boundary when policy dependencies allow it.
- Read back only final policy results and the requested final frame/image.

The first retained P4 slice prepares one fast, strategy-aware final-CfL map
from the adjusted initial quant field and reuses it for all fully resident AQ
evaluations. Exact mode still recomputes the quant-dependent map. One warm
padded-1080p process with one warmup and three alternating samples measured
Metal at `394.4-424.5 ms` (median `410.1 ms`) and paired speedup at
`14.91-16.09x` (median `15.50x`). Relative to exact coefficients, the score
maximum remained `0.0540`; field, block-map, and RGB maxima were `0.202`,
`0.424`, and `0.979`. This only removes repeated host regression: quantizer and
EPF preparation plus three submission/readback boundaries remain.

The second retained P4 slice adds an explicit `throughput` mode that uses the
resident evaluator but caps the public pipeline at one AQ update. Existing
`fully-resident` calls continue to honor the configured iteration count. On
one padded-1080p process with one warmup and five alternating samples, the new
mode measured `237.4-270.0 ms` (median `266.5 ms`) and paired speedup at
`22.69-25.89x` (median `23.23x`). Relative to the exact pipeline, one full
profile observed `0.161` final-field, `0.494` block-map, `0.0768` final-score,
and `0.972` reconstructed-RGB maximum error; the codestream was `618217` bytes
versus `630517` exact and `617093` for two-update fully resident. A rejected
zero-update experiment reached a `243.3 ms` public median but increased the
final-score error to `0.356` and the codestream to `637091` bytes. Throughput
mode is never automatic and requires explicitly forced Metal.
An independently installed `djxl` 0.12 decoder also accepted the CLI's
throughput sample and produced a 17x13 linear-RGB PFM.

The third retained P4 slice adds a separate `maximum-throughput` workflow. It
uses a complete DCT8 grid and a frame-only prepared Metal operation that stops
after forward transforms and quantized coefficient readback. Inverse Gaborish
runs in that same command buffer immediately before coefficient coding, so the
filtered image never returns to the host. Frame-only preparation also omits the
unused original-image upload and prepared Butteraugli state.

Three independent Apple M4 Pro Release processes each used one warmup and five
alternating 1080p pairs. CPU process medians were `6260.8-6266.9 ms`; Metal
process medians were `104.0-115.6 ms`; paired process medians were
`54.73-60.42x`. Every individual pair was `51.95-62.85x`. The quantization
pipeline itself measured `73.1-78.7 ms` by process median. The `710572`-byte
maximum-throughput codestream was `12.7%` larger than the `630517`-byte exact
output.

The same three-process protocol at padded 4K measured CPU medians of
`24972.8-25062.9 ms`, Metal medians of `367.6-387.5 ms`, and paired process
medians of `64.35-68.20x`; every pair was `61.04-69.12x`. The
maximum-throughput codestream was `2846429` bytes versus `2512415` exact, a
`13.3%` increase.

Independent decodes with both installed `djxl` 0.12 and pinned libjxl revision
`e8ff09762481785938d8e4e01333ed3917571161` succeeded for the 17x13 CLI sample
and a 510x532 natural Flower image. Native Butteraugli distances at target
`1.0` were `1.1921773` and `1.34573984`, respectively. The mode is never
automatic and exposes no internal perceptual score; these independent
decoded-quality measurements are the only quality claim for the speed-first
policy.

#### Post-adjustment reconciliation checkpoint (2026-08-28)

After the performance stack was reconciled with rate control and the resident
`AdjustQuantBlockAC` decision was composed into production coefficient coding,
the public boundary was remeasured on the same Apple M4 Pro Release build. The
new block-grid raw-quant readback, codestream serialization, and all ordinary
workflow preparation are included.

Maximum-throughput used three independent processes, each with one warmup and
five alternating CPU/Metal pairs. CPU process medians were
`6313.1-6369.3 ms`, Metal process medians were `113.6-118.7 ms`, and paired
process-median speedups were `53.43-55.58x`; every individual pair was
`51.03-59.21x`. Metal quantization-pipeline process medians were
`80.1-80.8 ms`. The current maximum-throughput codestream is `765599` bytes
versus `637706` exact, a `20.1%` increase. These figures supersede the earlier
pre-adjustment 1080p size and speed checkpoint for the current implementation.

The other modes received one process with one warmup and three alternating
pairs, so these are directional checkpoints rather than retained
multi-process ranges:

| Mode | Metal median (range) | Paired speedup median (range) | Bytes |
| --- | ---: | ---: | ---: |
| exact-coefficients | 626.3 ms (600.0-673.0) | 10.06x (9.34-10.51) | 637706 |
| fully-resident | 279.0 ms (272.8-279.2) | 22.56x (22.55-23.15) | 622784 |
| throughput | 256.7 ms (248.3-257.3) | 24.68x (24.56-25.43) | 623449 |

The exact track retained identical CPU/Metal codestream bytes. All four modes
also produced independently decodable 17x13 CLI codestreams after the
adjustment integration, including target-size maximum-throughput and
maximum-error throughput requests. Decoded-quality and corpus-size claims from
the earlier maximum-throughput output do not transfer to the changed adjusted
codestream; they require a fresh named-corpus quality run.

#### Resident-frontend completion checkpoint (2026-08-28)

The five-step initial-CfL/initial-quant residency sequence supersedes the
maximum-throughput timing above. Three fresh Release processes each used one
warmup and five alternating CPU/Metal pairs. CPU process medians were
`6219.973-6313.653 ms`, Metal process medians were `78.077-81.855 ms`, and
paired process-median speedups were `77.496-80.666x`; every individual pair
was `69.529-92.592x`. Metal quantization-pipeline process medians were
`42.781-45.619 ms`. The codestream remains `765599` bytes.

The experimental AC-search handoff also produced directional one-process
improvements. Fully-resident measured `270.234 ms` (`266.775-270.819 ms`)
and `23.108x` paired speedup (`23.015-23.325x`) at `622784` bytes;
throughput measured `238.075 ms` (`236.711-243.464 ms`) and `26.597x`
(`26.195-26.849x`) at `623449` bytes. These two rows use one warmup and three
alternating pairs and remain regression signals rather than retained
multi-process claims.

One additional 4K process with one warmup and three alternating pairs was a
directional regression check. CPU median was `25388.6 ms`, Metal median was
`417.7 ms`, and paired speedup median was `61.21x` with a
`57.36-65.25x` per-pair range. The current output was `3061311` bytes versus
`2540027` exact (`20.5%` larger). A refreshed retained 4K claim would still
require the other two independent processes specified by the primary gate.

### P5. Parallelize the codestream tail

- Parallelize independent DC/AC group tokenization and section writing.
- Reuse coefficient orders and token storage instead of rebuilding them per
  group.
- Profile histogram optimization separately before considering a GPU entropy
  implementation.

The first retained P5 slice writes independent DC and AC group sections on up
to eight host workers after entropy models are fixed. On one padded-1080p
alternating process with one warmup and three samples, section writing fell
from the prior `18.7-22.2 ms` range to `2.7-3.5 ms`; complete codestream
encoding measured `26.8-30.8 ms`. Metal public workflow measured
`376.2-409.0 ms` (median `376.5 ms`) with paired speedup `15.28-16.59x`
(median `16.53x`). Existing frame, conformance, installed-consumer, and
deterministic-byte tests cover the parallel path.

### P6. Close the 50x gate

- The required independent-process 1080p and 4K benchmark matrix is complete
  for maximum-throughput mode, with process-median speedups above `50x`.
- Deterministic output, independent-`djxl` acceptance, decoded pixels, and
  Butteraugli drift are validated for the two decoded quality fixtures above.
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
