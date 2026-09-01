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
  evaluator. Encoding applies the default two AQ updates but omits the third,
  diagnostic-only reconstruction and perceptual score; diagnostic APIs retain
  their earlier one-update tradeoff.
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
just metal-encode-benchmark padded_4k simd 7 2 fully-resident
just coefficient-benchmark padded_1080p 9 2
```

The benchmark performs one correctness validation, alternates CPU/Metal order,
prints every profile boundary, and reports paired speedups. The broader
`just aq-benchmark` matrix remains the correctness and exploratory phase gate.
For large-image optimization loops, `metal-encode-benchmark` performs the same
initial CPU/Metal codestream validation and then times only repeated complete
Metal public encodes. This avoids placing a roughly 26-second 4K CPU encode
between every approximately one-second Metal sample. It is the preferred 4K
regression signal for Metal changes; the alternating multi-process CPU/Metal
protocol above remains the final speedup gate.

### Reproducible Metal profiling

Use the profiling workflow when aggregate phase timings identify a GPU-heavy
region but do not explain its command-buffer and encoder behavior:

```sh
just metal-profile
just metal-profile synthetic_128x96 simd 1 1 fully-resident
```

The default invocation creates a new, never-overwritten directory under
`logs/metal-profile/`, named with the UTC capture time, workload, AQ mode, and
Git revision. It configures an isolated Release build at
`build/metal-profile`, compiles the optimized Metal shaders with line tables
and recorded sources, generates `metal/gjxl.metallibsym`, and makes the
benchmark load that exact external metallib.

Each artifact contains:

- `raw-samples.json`, with every untraced workflow phase recorded as integer
  nanoseconds for all seven default samples;
- `gpu-stage-samples.json`, with raw timestamp intervals, stable stage and
  stage-local dispatch IDs, stable submission IDs, invocation numbers,
  dispatch geometry, typed host wall spans, and device counter-sampling
  capabilities for fully-resident or throughput AQ;
- `gpu-stage-summary.json`, with median cumulative time, call and dispatch
  counts, command-buffer percentage, per-submission and per-iteration
  breakdowns, and wall-span medians;
- `capture.trace`, `trace-toc.xml`, `trace.stdout`, and `trace-sample.json` for
  one instrumented Metal sample;
- build, benchmark, and `xctrace` logs;
- `manifest.json`, including the complete commands and exit status, Git state,
  selected environment, macOS/Xcode/Metal/xctrace versions, display/GPU data,
  and SHA-256 hashes of the benchmark, metallib, and symbol companion; and
- starting worktree/index patches plus an untracked-file hash inventory.

Raw workflow schema 7 keeps the elapsed `codestream_dc_tokenization`,
`codestream_ac_tokenization`, `codestream_entropy_optimization`,
`codestream_section_writing`, and `codestream_assembly` phases, and adds finer
codestream counters. Names ending in `_work` are aggregate worker time: work
performed by overlapping DC, coefficient-order, AC-candidate, and section
tasks is summed, so those counters can exceed their enclosing wall-clock phase
and must not be added to complete-encode latency. The new breakdown separates
block-context and coefficient preparation, coefficient tokenization, prefix
histogram construction/clustering/code building, prefix value collection,
HybridUint configuration search, final prefix-model serialization and checked
cost assembly, ANS value aggregation,
HybridUint and histogram/model searches, exact ANS token costing, entropy
selection, model/header and token-stream writing, candidate measurement, and
the selected candidate's final header/TOC/section/output assembly. The
`substage_work_timing` field records the `aggregate-worker-time` semantics.
Entropy optimization retains each candidate model's exact per-section token
bit counts. Candidate measurement reuses those counts to evaluate physical
section and TOC sizes without traversing or materializing token payloads.
Model/header and token-stream writing therefore cover only the selected
candidate; `codestream_section_writing` includes both the measurement span and
that final serialization span.

Open `capture.trace` in Instruments. Keep `gjxl.metallib` and
`gjxl.metallibsym` together in the profiling build so Instruments can resolve
shader symbols and source locations. The driver allows a dirty checkout but
records it. If tracked or untracked source state changes during the workflow,
the artifact is preserved as failed and the command exits nonzero. Missing
tools, failed builds, failed benchmark validation, and failed trace export are
handled the same way.

Treat `raw-samples.json` as the performance comparison evidence. Profile schema
3 covers every current compute submission in one profiled public encode and
adds a stable `group_id` for comparing coarse regions across substage-schema
changes. The resident AQ command buffer and the measured frontend hotspots are
split into logical stage encoders. Its stable fully-resident submission IDs are
`frontend.prepare_aq.reference`, `frontend.initial_quantization`,
`frontend.ac_strategy`, `frontend.quant_adjustment`, and `resident.aq`.
Reference preparation now occurs when the single complete resident evaluator
is constructed before AC search; after search, that compatible evaluator
reports `frontend.reconfigure_aq` instead of a second `frontend.prepare_aq`
span. The resident evaluator folds preprocessing and initial CfL into
`frontend.initial_quantization`; there is no separate preprocessing submission
or host CfL upload on that path.

The `frontend.ac_strategy` group contains one substage for each candidate
transform strategy. The `aq.reconstruction` group contains `reset`,
`quantizer`, optional per-strategy `forward.*` and `final_cfl` preparation,
and `coefficients.*`, inverse-transform `dct*`, and `scatter.*` substages for
each active reconstruction strategy. Forward preparation is reported only when
cached coefficients are unavailable; all of its gather/forward substages remain
ahead of every strategy reconstruction. The three reconstruction substages
retain production ordering while exposing which part of the former combined
batch is material. These extra stage-mode boundaries remain attribution
instrumentation rather than emulated per-dispatch timestamps.
`gpu-stage-summary.json` reports both substage totals and
`stage_groups`/`group_iterations`; schema-1 and schema-2 input remains readable
by treating each legacy stage as its own group.

Typed steady-clock wall spans separate preparation, upload, wait, readback,
and host work around those submissions. Operation spans can contain narrower
spans, so wall spans are hierarchical evidence and must not be summed. GPU
percentages use the sum of all captured command-buffer durations in the
sample. Counter sampling changes encoder boundaries for the resident buffer,
and Metal System Trace adds its own instrumentation overhead, so both remain
attribution tools rather than latency evidence. Dispatch timing can be
requested from the benchmark with `--gpu-profile dispatch`, but capability
preflight fails before submission on devices such as the Apple M4 Pro that do
not expose dispatch-boundary counter sampling. The workflow never emulates
dispatch timestamps by changing every dispatch into a separate encoder.

The diagnostic session is created only by the explicitly profiled entrypoint.
Ordinary encoding does not allocate profile metadata or read clocks. A profile
is published only after the whole caller-visible operation succeeds, and
stable per-ID invocation numbers disambiguate repeated submissions or wall
spans without exposing Metal types above the backend capability boundary.

#### Current fully-resident stage baseline (2026-08-28)

Commit `4e28177` was profiled on the Apple M4 Pro with the SIMD implementation,
target distance `1.2`, two warmups, and seven samples for both required
large-image workloads. The artifacts are
`20260828T235258Z-padded_1080p-fully-resident-4e28177196f9` and
`20260828T235344Z-padded_4k-fully-resident-4e28177196f9` under
`logs/metal-profile/`. Both completed with an unchanged source fingerprint and
the same benchmark and metallib hashes. Median sampled-stage coverage was
`99.970%` at 1080p and `99.969%` at 4K; every profile sample contained one
resident submission with `553` dispatches.

The uninstrumented public-workflow medians and the separately instrumented
resident command-buffer medians were:

| Workload | Public total | Input preparation | Quantization pipeline | Codestream encoding | Profiled resident GPU buffer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Padded 1080p | `259.939 ms` | `15.639 ms` | `208.461 ms` | `34.310 ms` | `74.206 ms` |
| Padded 4K | `980.884 ms` | `65.374 ms` | `822.910 ms` | `86.894 ms` | `298.128 ms` |

The resident stage ranking was:

| Stage | 1080p median | 1080p GPU share | 4K median | 4K GPU share | Dispatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| `aq.reconstruction` | `24.231 ms` | `32.410%` | `83.732 ms` | `28.087%` | `192` |
| `butteraugli.psycho.main` | `15.283 ms` | `20.548%` | `66.705 ms` | `22.393%` | `90` |
| `butteraugli.malta.main` | `13.736 ms` | `18.517%` | `57.242 ms` | `19.174%` | `36` |
| `butteraugli.psycho.sub` | `4.070 ms` | `5.442%` | `17.717 ms` | `5.946%` | `99` |
| `butteraugli.malta.sub` | `3.493 ms` | `4.684%` | `14.038 ms` | `4.708%` | `36` |
| `butteraugli.l2.main` | `3.414 ms` | `4.572%` | `14.270 ms` | `4.785%` | `3` |
| `butteraugli.mask_final.main` | `3.165 ms` | `4.252%` | `13.768 ms` | `4.617%` | `18` |

The top three stages account for `71.475%` of sampled command-buffer time at
1080p and `69.654%` at 4K. All Butteraugli stages together account for
`45.786 ms`/`61.552%` and `195.236 ms`/`65.473%`, respectively. Per-iteration
medians are stable: reconstruction is approximately `8.0 ms` per 1080p
evaluation and `28.0 ms` per 4K evaluation, while the two largest Butteraugli
stages are approximately `5.1/4.6 ms` and `22.2/19.0 ms` per evaluation.

This makes `aq.reconstruction` the first stage-level optimization target. Its
dispatch inventory points first to the repeated transform gather, coefficient
encode, and reconstructed-pixel scatter sequence, followed by the resident
quantizer histogram sequence. The next targets are the full-resolution
Butteraugli psychoacoustic and Malta stages. The psychoacoustic inventory is
dominated by repeated transpose convolution dispatches, but this device does
not expose dispatch-boundary timestamps, so a candidate must be isolated with
a matched-build stage A/B rather than attributed from dispatch count alone.

#### Butteraugli convolution and Malta experiment (2026-08-28)

The experiment started from `56a0790` on the Apple M4 Pro. Candidate screens
used the standalone padded-1080p Butteraugli benchmark with two warmups, 11
rotated samples per process, and three alternating matched process pairs where
the result was close. Tiling the 7/13/15/33-tap transpose convolutions did not
pass this screen: the 16x8 variant increased the resident-consumer median from
`16.441 ms` to `19.128 ms`, while the two 32x8 pairs regressed from
`17.595/16.801 ms` to `17.927/19.700 ms`. Both convolution variants were
discarded.

Suppressing the otherwise unused Malta response-plane store improved the
median resident-comparison result from `20.281 ms` to `19.184 ms`; its
resident-consumer result improved from `16.913 ms` to `16.588 ms`, with wins
in two of three pairs. The retained `a1e1725` change goes further: it combines
Malta scaling and response accumulation in one 32x8 dispatch, stages a
four-pixel halo in threadgroup memory, and writes the response plane only when
a diagnostic stage capture requests it. A device unable to launch the
256-thread tile uses the original two-dispatch path. This reduces each sampled
main/sub Malta stage from 36 dispatches to 18.

The 16x8 fused candidate already reduced the standalone medians from
`19.352/15.980 ms` to `15.639/13.063 ms` for resident comparison/consumer. A
three-pair geometry screen selected 32x8: its resident-comparison median was
`14.666 ms` versus `15.646 ms` for 16x8, while resident consumer was effectively
tied at `12.841 ms` versus `12.887 ms`.

The final public-encode gate used fully resident SIMD AQ at distance `1.2`, two
warmups, seven samples, and three alternating process pairs per workload. Each
timing below is the median of the three process medians; the candidate won both
required boundaries in all six pairs.

| Workload | Baseline total | Retained total | Delta | Baseline quantization | Retained quantization | Delta | GPU bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Padded 1080p | `279.257 ms` | `269.496 ms` | `-3.5%` | `146.643 ms` | `136.144 ms` | `-7.2%` | `453341` |
| Padded 4K | `967.390 ms` | `922.174 ms` | `-4.7%` | `591.302 ms` | `543.879 ms` | `-8.0%` | `1745707` |

The output byte counts match the baseline in every run. The existing fully
resident CPU/Metal codestream difference remains (`480842/453341` bytes at
1080p and `1853069/1745707` bytes at 4K); this experiment neither introduces
nor expands that policy difference. No numerical tolerance was widened. The
Metal Butteraugli test passes with maximum map/score error `0.000549316` and
maximum diagnostic-stage error `0.000396729`.

Post-change schema-2 profiles at `a1e1725` are
`20260829T014323Z-padded_1080p-fully-resident-a1e17256d763` and
`20260829T014405Z-padded_4k-fully-resident-a1e17256d763` under
`logs/metal-profile/`. Both completed successfully with identical benchmark
and metallib hashes. Their sampled-stage coverage is `99.979%` and `99.994%`;
the resident submission contains 490 dispatches.

| Workload | Public total | Quantization | Resident GPU buffer | Malta main | Malta sub |
| --- | ---: | ---: | ---: | ---: | ---: |
| Padded 1080p | `270.375 ms` | `138.453 ms` | `62.018 ms` | `4.872 ms` | `1.298 ms` |
| Padded 4K | `921.975 ms` | `539.170 ms` | `240.059 ms` | `19.156 ms` | `4.860 ms` |

Relative to the earlier `4e28177` stage profile, Malta main/sub are lower by
`64.5%/62.8%` at 1080p and `66.5%/65.4%` at 4K. This cross-commit comparison
also contains intervening resident-architecture changes, so the matched
component and public-workflow A/B results above are the isolated speedup
evidence. In the complete post-change GPU profile, Malta is no longer a top-
three stage: frontend AC-strategy search, AQ reconstruction, and Butteraugli
psychoacoustic processing are now the leading GPU targets.

#### Frontend AC-strategy reduction experiment (2026-08-28)

Commit `02f8c4d` removes a worst-case threadgroup-storage assumption from the
AC-strategy residual and cost kernels. The previous kernels reserved reduction
arrays for 1024 coefficients for every strategy. The residual kernel now uses
dynamic storage sized to the active coefficient count. The cost kernel uses
three dynamically sized arrays and reduces all color channels concurrently,
while preserving the existing per-channel reduction tree and final channel
accumulation order. For an 8x8 strategy, residual reduction storage falls from
`8192` to `512` bytes and cost storage changes from `4096` to `768` bytes; the
cost kernel also performs one barrier/reduction sequence instead of three.

Three alternating five-sample standalone-search pairs at 1920x1080 improved
the complete GPU-search median from `44.179 ms` to `41.414 ms` (`-6.3%`). The
CPU and GPU strategy grids remained identical. The large-image public gate
used fully resident SIMD AQ at distance `1.2`, two warmups, seven samples, and
three alternating process pairs per workload:

| Workload | Baseline total | Retained total | Delta | Baseline quantization | Retained quantization | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Padded 1080p | `291.679 ms` | `282.745 ms` | `-3.1%` | `147.914 ms` | `144.159 ms` | `-2.5%` |
| Padded 4K | `956.768 ms` | `948.079 ms` | `-0.9%` | `557.771 ms` | `551.464 ms` | `-1.1%` |

Quantization improved in all six pairs. Public total improved in all three
1080p pairs and two of three 4K pairs; the remaining 4K pair regressed by
`0.39%`. GPU output sizes remained `453341` and `1745707` bytes.

Post-change profiles are
`20260829T034606Z-padded_1080p-fully-resident-02f8c4d5c050` and
`20260829T034636Z-padded_4k-fully-resident-02f8c4d5c050` under
`logs/metal-profile/`, with `99.979%` and `99.989%` sampled-stage coverage.
Against the immediately preceding `a1e1725` profiles, the attributed
`frontend.ac_strategy` GPU stage changed from `25.849` to `24.755 ms` at
1080p (`-4.2%`) and from `103.717` to `98.078 ms` at 4K (`-5.4%`). Its wait
span changed from `29.224` to `28.565 ms` and from `116.733` to `112.239 ms`,
respectively. All AC candidate/search parity tests passed without tolerance
changes. The complete suite remained 57/58, with only the inherited pinned CPU
quantization-pipeline score mismatch.

These schema-1 artifacts exposed the instrumentation boundary that motivated
schema 2; schema 3 adds stable groups and finer substages. The extended profile
now attributes resident initial-field and preprocessing work, strategy search,
reference preparation, quant-field
adjustment, host CfL scheduling, transfers, waits, and the resident policy
under the same diagnostic session. Uninstrumented public-workflow runs remain
the final judge for any retained performance claim.

#### Frontend AC-strategy candidate-pipeline fusion (2026-08-29)

The SIMD AC-candidate path now performs three dispatches per transform family
instead of five. Candidate pixels are gathered directly into the forward-DCT
threadgroup tile, eliminating the packed-pixel scratch write/read. Residual
coefficients are likewise retained in threadgroup memory for the inverse DCT,
eliminating its scratch write/read and separate dispatch. The masked cost pass
remains separate because it combines all three reconstructed color transforms.
Scalar, factored, and mixed DCT selections retain the original gather,
transform, residual, inverse, and cost fallback.

Direct Metal cost tests cover every candidate transform with scalar, SIMD, and
factored DCT implementations. The SIMD path retained the preceding maximum
absolute/relative cost errors, and the complete CPU/GPU search grids remained
identical. The serial Release suite passes 58/59 tests; the sole failure is the
unchanged pinned CPU `quantization_pipeline` score difference of
`4.4524669647216797e-05` at index 1.

A clean same-revision schema-stage pair used two warmups and five samples.
Across the seven AC-strategy families, the sampled stage total changed from
`27.075` to `18.670 ms` at padded 1080p (`-31.0%`) and from `103.845` to
`66.072 ms` at padded 4K (`-36.4%`). The dispatch inventory fell from 35 to
21. These sampled timestamps explain the candidate; they are not the retained
latency claim.

The final unprofiled public gate used fully resident SIMD AQ at distance `1.2`,
two warmups, seven samples, and three process pairs per workload. Pair order was
baseline/candidate, candidate/baseline, then baseline/candidate. Each table
value is the median of the three process medians.

| Workload | Baseline total | Fused total | Delta | Baseline quantization | Fused quantization | Delta | GPU bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Padded 1080p | `524.688 ms` | `514.141 ms` | `-2.0%` | `137.237 ms` | `128.111 ms` | `-6.7%` | `420268` |
| Padded 4K | `1475.565 ms` | `1440.820 ms` | `-2.4%` | `491.362 ms` | `452.611 ms` | `-7.9%` | `1640942` |

Quantization improved in all six pairs: `4.80-10.13%` at 1080p and
`6.86-9.08%` at 4K. Total time improved by `1.62-2.43%` in two 1080p pairs and
was effectively tied in the other (`+0.06%`, or `0.315 ms`); all three 4K
pairs improved by `1.70-4.54%`. GPU output size was unchanged in every sample.

#### Butteraugli convolution-consumer fusion (2026-08-29)

Commit `cada229` retains the pass-fusion alternative after the generic tiled
convolution experiments above regressed. The five-tap vertical blur for all
three input channels now feeds Opsin directly. The final 33-tap convolution
pass for all three XYB channels is combined with low/medium decomposition, and
the final 15- and 7-tap passes are combined with their high- and ultra-frequency
consumers. The B-channel 15-tap blur writes its final result in place instead
of copying a temporary plane. The fused kernels preserve the original clipped
support, weight accumulation order, channel transforms, and stage-capture
outputs; the replaced pointwise pipelines were removed.

The final standalone padded-1080p gate used two warmups, 11 rotated samples,
and three alternating process pairs. The median of the three process medians
improved from `13.063 ms` to `11.576 ms` for resident consumer end to end
(`-11.4%`) and from `13.060 ms` to `11.884 ms` for resident comparison
(`-9.0%`). The candidate won all six paired boundaries.

The public-encode gate used fully resident SIMD AQ at distance `1.2`, two
warmups, seven samples, and three alternating process pairs per workload. Each
value below is the median of the three process medians; the candidate won all
six total and quantization comparisons.

| Workload | Baseline total | Retained total | Delta | Baseline quantization | Retained quantization | Delta | GPU bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Padded 1080p | `288.015 ms` | `276.935 ms` | `-3.8%` | `148.335 ms` | `141.101 ms` | `-4.9%` | `453341` |
| Padded 4K | `939.301 ms` | `911.371 ms` | `-3.0%` | `557.946 ms` | `527.846 ms` | `-5.4%` | `1745707` |

Fresh schema-2 profiles are
`20260829T041348Z-padded_1080p-fully-resident-cada2296d59f` and
`20260829T041429Z-padded_4k-fully-resident-cada2296d59f` under
`logs/metal-profile/`. They completed with `99.976%` and `99.990%` sampled-stage
coverage and unchanged source fingerprints. Against the immediately preceding
`02f8c4d` profiles:

| Workload | Reference preparation | Resident GPU buffer | Psycho main | Psycho sub |
| --- | ---: | ---: | ---: | ---: |
| Padded 1080p | `7.068 -> 5.475 ms` (`-22.5%`) | `63.036 -> 57.799 ms` (`-8.3%`) | `15.695 -> 11.375 ms` (`-27.5%`) | `4.164 -> 3.331 ms` (`-20.0%`) |
| Padded 4K | `29.336 -> 21.961 ms` (`-25.1%`) | `242.628 -> 220.389 ms` (`-9.2%`) | `65.332 -> 47.541 ms` (`-27.2%`) | `17.491 -> 12.981 ms` (`-25.8%`) |

Each resident psycho main/sub stage falls by 33 dispatches, reducing the
resident submission from 490 to 424 dispatches. Reference preparation falls
from 66 to 44 dispatches. The profiles' separate uninstrumented public runs
also improve total/quantization medians from `278.660/142.311` to
`268.863/134.087 ms` at 1080p and from `939.299/552.598` to
`901.521/510.337 ms` at 4K. The matched multi-process table above remains the
primary latency evidence.

No tolerance changed. The Metal Butteraugli gate retains maximum map/score
error `0.000549316` and maximum stage error `0.000396729`; output byte counts
are unchanged. The full suite remains 57/58, with every Metal test passing and
only the inherited pinned CPU quantization-pipeline score mismatch
(`4.4524669647216797e-05`).

#### AQ reconstruction pass streamlining (2026-08-29)

Commit `3fdc682` retains three changes to the resident reconstruction path.
Production submissions now clear only the reconstruction error word instead of
poisoning every scratch output; the diagnostic `RunReconstruction` path still
enables the full poison-and-coverage check. Reconstruction coefficient kernels
dispatch `min(256, coefficient_count)` threads per transform, avoiding idle
lanes for 8x8 and rectangular strategies. Finally, quantizer histogram clears
are folded into the selection initialization and bucket-selection passes,
removing the separate clear pipeline and 24 dispatches per resident submission.

The public-encode gate used fully resident SIMD AQ at distance `1.2`, two
warmups, seven samples, and three alternating process pairs per workload. Each
value is the median of the three process medians:

| Workload | Baseline total | Retained total | Delta | Baseline quantization | Retained quantization | Delta | Pair wins (total/quantization) | GPU bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Padded 1080p | `262.348 ms` | `261.766 ms` | `-0.2%` | `129.751 ms` | `128.308 ms` | `-1.1%` | `2/3`, `3/3` | `453341` |
| Padded 4K | `880.241 ms` | `874.980 ms` | `-0.6%` | `501.877 ms` | `493.544 ms` | `-1.7%` | `3/3`, `3/3` | `1745707` |

Fresh schema-2 profiles are
`20260829T043329Z-padded_1080p-fully-resident-3fdc68228eb1` and
`20260829T043357Z-padded_4k-fully-resident-3fdc68228eb1` under
`logs/metal-profile/`. They completed with `99.976%` and `99.981%`
sampled-stage coverage. Against the immediately preceding `cada229` profiles:

| Workload | AQ reconstruction | Resident GPU buffer | Reconstruction dispatches | Resident dispatches |
| --- | ---: | ---: | ---: | ---: |
| Padded 1080p | `23.228 -> 20.349 ms` (`-12.4%`) | `57.799 -> 54.777 ms` (`-5.2%`) | `165 -> 141` | `424 -> 400` |
| Padded 4K | `77.971 -> 67.101 ms` (`-13.9%`) | `220.389 -> 211.791 ms` (`-3.9%`) | `165 -> 141` | `424 -> 400` |

The profiles' separate uninstrumented public runs improve total/quantization
medians from `268.863/134.087` to `261.140/127.475 ms` at 1080p and from
`901.521/510.337` to `878.653/499.096 ms` at 4K. The alternating-process table
above remains the primary latency evidence. A matching dynamic threadgroup
experiment on final frame coefficient encoding was rejected: across three
alternating 1080p pairs, quantization changed from `126.794` to `126.997 ms`
and total time from `260.940` to `262.878 ms`.

No output bytes or tolerances changed. The exact Metal reconstruction test
passes, as do all other Metal tests. The complete suite remains 57/58 with only
the inherited pinned CPU quantization-pipeline score mismatch.

#### Butteraugli prepared-scratch aliasing (2026-08-29)

Commit `9f35e4c` makes the prepared Butteraugli arena reflect actual plane
lifetimes. Psycho-image input and horizontal-blur planes are dead before
difference encoding starts, so their six full-resolution allocations now back
the three AC and three DC accumulators. Psycho convolution intermediates are
similarly reused as difference scratch. Cached reference data, the reference
mask, and final diagnostic/multiscale staging remain distinct.

This reduces the arena from 39 to 33 full-resolution planes. The 1080p AQ
memory gate reports staging reduction from `503446324` to `453679924` bytes
(`-49.8 MB`, `-9.9%`) and peak scratch from `391471392` to `341704992` bytes
(`-49.8 MB`, `-12.7%`). Padded 4K avoids `198921624` bytes of prepared storage.

This is a footprint result, not a latency claim. Three alternating padded-4K
public pairs produced nearly unchanged aggregate quantization medians
(`509.335 -> 509.135 ms`, two of three wins); total medians changed from
`891.992` to `886.217 ms`, also two of three wins. A diagnostic trace likewise
left reference GPU work effectively unchanged (`22.486` versus `22.196 ms`).
Output remained `1745707` bytes. The exact Metal Butteraugli and AQ tests pass
with unchanged error bounds, and the complete suite remains 57/58 with only
the inherited pinned CPU quantization-pipeline score mismatch.

A follow-up dispatch-fusion experiment combined the three five-tap horizontal
blur channels. It removed four reference-preparation dispatches and two
dispatches per comparison scale, but the direct padded-1080p resident-
comparison median regressed from `11.002` to `11.181 ms` (`+1.6%`) and lost all
three alternating pairs. Resident consumer end to end changed only from
`10.523` to `10.479 ms` (`-0.4%`). The fusion was rejected; lower dispatch
count did not offset the larger per-thread kernel.

#### Butteraugli masked-AC tail fusion (2026-08-29)

Commit `a0d75c4` folds masked-AC accumulation into final map composition. The
production path now reads the two blurred activity masks directly while
forming the final distance value, avoiding an in-place full-plane AC update
and its following read. The separate masked-AC pipeline remains available only
when that diagnostic stage is explicitly captured, preserving the stage
oracle.

The padded-1080p screening trace reduced mask/final main from `3.211` to
`2.921 ms` (`-9.0%`) and mask/final sub from `1.351` to `1.261 ms` (`-6.7%`).
Each stage loses three dispatches, and the resident submission changes from
`400` to `394` dispatches and `54.777` to `53.447 ms` (`-2.4%`). Sampled-stage
coverage remained `99.973%`. All Metal tests pass with unchanged Butteraugli
map, score, and stage error bounds; the complete suite remains 57/58 with only
the inherited pinned CPU quantization-pipeline score mismatch. A clean public-
workflow latency gate is deferred to the combined tail result because unrelated
system indexing invalidated the contemporaneous CPU-facing pair.

A second tail experiment folded multiscale final-map composition into the sub-
scale final kernel. It removed the intermediate sub-map write and three more
resident dispatches, but improved the 1080p sub tail by only `0.041 ms`
(`1.261 -> 1.220 ms`). That did not justify specialized output routing, so the
experiment was rejected.

#### Final kernel-series integration audit (2026-08-29)

Merge commit `2973d43` incorporates target `refactor/metal-cpp` commit
`ff3cac3`, including its new codestream density and entropy work. The target
changes do not overlap the Metal source tree; `src/gpu/metal` is byte-identical
before and after the merge, so the per-kernel profiles and matched optimization
gates above remain applicable to the merged implementation.

The merged tree passes 58/59 tests, including every Metal test and the new
codestream tests. The sole failure remains the inherited pinned CPU
quantization-pipeline score mismatch (`4.4524669647216797e-05`). Latest-target
validation encodes produce identical baseline/candidate GPU sizes of `420268`
bytes at padded 1080p and `1640942` bytes at padded 4K.

No aggregate latest-target latency is claimed here. During the final gate the
host load average reached `28`, and normally sub-second 1080p public encodes
expanded to `2.7-3.6 s`; single-sample 4K runs expanded to `7.3-7.9 s` with
contradictory quantization ordering. Those runs establish successful merged-
target encoding and byte parity only. The clean alternating-process tables in
the individual sections remain the performance evidence.

#### Unified resident evaluator preparation (2026-08-29)

Commit `f3181c1` replaces the two prepared evaluators in fully-resident mode
with one complete evaluator. It is initially configured with a provisional
DCT8 grid, prepares the Butteraugli reference once, and supplies resident
preprocessing, initial quantization, CfL, and AC-search views. After strategy
search, the same allocation is reconfigured with the selected grid instead of
constructing a second evaluator and uploading the search-domain image and
initial field again. Forward coefficient coding selects the evaluator's
Gaborish-preprocessed resident image when that search-domain view is active.

The public-encode gate compared `5972c25` with `f3181c1` on the Apple M4 Pro.
It used fully resident SIMD AQ at distance `1.2`, Metal-only validation, two
warmups, seven samples, and three alternating process pairs per workload. Each
timing below is the median of the three process medians:

| Workload | Baseline total | Unified total | Delta | Baseline quantization | Unified quantization | Delta | Pair wins (total/quantization) | GPU bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Padded 1080p | `518.015 ms` | `513.871 ms` | `-0.8%` | `132.968 ms` | `131.778 ms` | `-0.9%` | `2/3`, `2/3` | `420268` |
| Padded 4K | `1471.748 ms` | `1461.458 ms` | `-0.7%` | `490.984 ms` | `475.263 ms` | `-3.2%` | `3/3`, `3/3` | `1640942` |

All 21 samples per build and workload produced the same byte count and final
score as the baseline. The 1080p latency result is modest and has overlapping
sample ranges; the stronger retained signal is the 4K quantization boundary,
where every process pair improved.

Matched seven-sample schema-2 profiles verify the structural change. The
baseline's second `frontend.prepare_aq` wall span measured `17.149 ms` at
1080p and `69.378 ms` at 4K. It is absent after unification; final strategy
installation instead reports `frontend.reconfigure_aq` at `0.282 ms` and
`1.010 ms`. Complete preparation moves earlier, so those values are not direct
latency savings. The combined initial-quantization and quant-adjustment spans
fall from `16.325` to `8.241 ms` at 1080p and from `45.377` to `25.214 ms` at
4K, consistent with retaining the initial field and coding image on the same
evaluator. The resident AQ buffer itself is effectively unchanged
(`59.880 -> 59.810 ms` and `230.050 -> 229.272 ms`).

The complete Release suite passes 58/59 tests. Every Metal test and the updated
ordered profiling contract pass; the sole failure remains the inherited pinned
CPU quantization-pipeline score mismatch of
`4.4524669647216797e-05`. Focused coverage additionally verifies that repeated
fully-resident targets retain the exact same evaluator allocation and preserve
frame and codestream output.

#### Final-field frame-only experiment (2026-08-29)

The resident dependency audit found no identical AQ iteration to cache or
remove. Between the first and second updates, every float field entry changed
at both padded 1080p and padded 4K. The final raw-quant grid changed in
`3936/32400` and `15071/129600` blocks, respectively, and the device quantizer
changed from `4644/16` to `4563/17` at 1080p and from `4647/16` to `4561/17`
at 4K. Because the global quantizer changes, unchanged raw-quant blocks do not
provide a safe reconstruction-reuse boundary.

The initial retained experiment changed only explicit throughput encoding. It
performs both configured AQ evaluations and dependent field updates, then runs
the resident quantizer and coefficient encoder once for the resulting field.
It omits inverse transforms, reconstructed-pixel scatter, loop filters,
opsin-to-linear conversion, Butteraugli, block reduction, and the final
non-updating policy dispatch. The same frame-only materialization is now the
default for public fully resident encoding as well. Both modes return the
scores of fields they actually evaluated and mark that no score corresponds to
the final frame. `collect_final_butteraugli_score` and the benchmark/CLI
`--collect-final-score` switch restore the terminal diagnostic without
changing the configured update count or codestream.

Post-merge schema-3 profiles on the Apple M4 Pro used SIMD AQ at distance
`1.2`, two warmups, and seven samples. Stage coverage was `99.940%` at 1080p
and `99.988%` at 4K. The final frame-only quantizer/coefficient work measured
approximately `5.54 ms` and `16.97 ms`, respectively.

| Workload | Full resident GPU buffer | Final-frame GPU buffer | Delta | Full resident AQ wall span | Final-frame AQ wall span | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Padded 1080p | `55.024 ms` | `42.225 ms` | `-23.3%` | `62.378 ms` | `51.077 ms` | `-18.1%` |
| Padded 4K | `213.045 ms` | `158.730 ms` | `-25.5%` | `239.519 ms` | `186.642 ms` | `-22.1%` |

One same-process public-workflow screen measured quantization medians of
`130.144` versus `148.094 ms` at 1080p (`-12.1%`) and `506.602` versus
`537.848 ms` at 4K (`-5.8%`). Complete public medians were noisy because the
host entropy tail dominated and had wide ranges, so they are not retained as
an end-to-end speedup claim. Both large-workload modes produced the same byte
counts (`420268` and `1640942`), and the focused public integration test
requires byte-for-byte throughput/fully-resident codestream equality plus
exact equality of the two shared score-history entries.

A later three-pair gate alternated process order with the same two warmups and
seven samples while host load average was approximately `19`. Its process-
median summary was contradictory:

| Workload | Full total | Final-frame total | Pair wins | Full quantization | Final-frame quantization | Pair wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Padded 1080p | `730.770 ms` | `652.230 ms` | `1/3` | `164.658 ms` | `134.396 ms` | `2/3` |
| Padded 4K | `1822.152 ms` | `2085.274 ms` | `1/3` | `506.547 ms` | `533.199 ms` | `1/3` |

Individual 4K total ranges extended from `1485.961` to `2902.937 ms`, and
quantization ranges from `427.075` to `879.504 ms`. This gate does not
establish a public-workflow latency improvement. The retained evidence is the
timestamped removal of resident GPU work and exact output parity; a lower-load
alternating rerun remains necessary before assigning a stable large-image
end-to-end speedup. The default is justified by removal of known GPU work plus
exact codestream equality, not by that noisy public timing gate.

A promotion gate on the integrated branch ran at a load average of about
`4.4`, using the Metal-only public workflow, SIMD AQ at distance `1.2`, two
warmups, seven samples, and three alternating process pairs per workload. Pair
order was unscored/scored, scored/unscored, then unscored/scored. Each value is
the median of the three process medians; `unscored` is the new default and
`scored` uses `--collect-final-score`.

| Workload | Unscored total | Scored total | Delta | Unscored quantization | Scored quantization | Delta | Pair wins (total/quantization) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Padded 1080p | `825.499 ms` | `866.280 ms` | `-4.7%` | `144.104 ms` | `143.791 ms` | `+0.2%` | `3/3`, `2/3` |
| Padded 4K | `2346.020 ms` | `2619.188 ms` | `-10.4%` | `454.836 ms` | `546.225 ms` | `-16.7%` | `3/3`, `3/3` |

The complete serialized tail was still variable and dominated total time, so
the host totals are supporting evidence rather than a precise isolated-pass
speedup. The earlier stage timestamps remain the direct attribution. Every
sample retained `420268` bytes at 1080p and `1640942` bytes at 4K. Focused
integration tests additionally require byte-for-byte equality for both fully
resident and throughput score opt-ins, exact equality of their shared score
entries, and an explicit summary-validity bit distinguishing update scores
from the terminal encoded-frame score.

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

#### Resident quantization-preparation checkpoint (2026-08-28)

The next P4 slice moves strategy-aware initial field adjustment and each
resident evaluation's median/MAD quantizer plus raw-quant construction to
Metal. The existing adjusted-coefficient kernel now consumes the device
quantizer and prepares EPF inverse sigma from the final device raw quant, so
resident evaluations no longer allocate or upload host raw-quant and sigma
fields. The exact track remains on the original CPU preparation path. CPU
distance-driven policy updates, block-distance readback, fixed-CfL upload, and
the serial evaluation boundaries remain.

The iterative selector uses exact four-pass byte-radix histograms for both the
upper median and median absolute deviation. Direct mixed-strategy tests retain
CPU-identical quantizer parameters, bounded field tolerance, output padding,
allocation reuse, and atomic numeric failure. The complete Release suite was
`55/56`; the only failure was the unchanged pinned `4.45247e-05` CPU golden
mismatch at score index 1.

One same-machine directional comparison used clean detached `ad2c529` as the
pre-change baseline and the working tree as the new path. Each command used
one warmup and five public-workflow samples. Fully resident moved from a
`303.2 ms` total median (`263.3-377.2`) to `298.4 ms`
(`261.0-358.1`), while quantization-pipeline medians moved from `259.6 ms` to
`250.9 ms`. Throughput was effectively neutral: `251.1 ms`
(`229.9-282.2`) before and `252.7 ms` (`235.5-317.2`) after, with pipeline
medians of `206.4 ms` and `205.9 ms`. The ranges overlap substantially, so
these are regression signals rather than a retained speedup claim.

The 4672x5584 doughnut sample encoded successfully through `just encode` in
the fully-resident mode: reported preparation was `311.8 ms`, the selected AQ
attempt was `4531.3 ms`, total encoder time was `4843.1 ms`, and the output was
`3127908` bytes. Independent `djxl` 0.12 decoding produced a 4672x5584 PFM.

#### Resident invariant-metadata checkpoint (2026-08-28)

The following P4 slice keeps the fixed CfL map beside the already persistent
strategy, anchor, and EPF metadata. Fully-resident and throughput evaluations
now omit both CfL planes, so their only per-evaluation input transfer is the
updated float quant field. The map is uploaded once per configured
rate-control attempt; successful strategy reconfiguration invalidates it and
requires an explicit rebind before another evaluation.

The profiled direct contract confirms exactly four uploaded bytes per block on
the resident path, with no CfL bytes. Binding allocates no device memory and
submits no command buffer. Host-map poisoning, repeated-evaluation parity,
stale-binding rejection, explicit rebound, public resident AQ, and the Metal
quantization pipeline pass their focused tests. A serial complete Release run
passed `55/56`; the only failure was the unchanged pinned `4.45247e-05` CPU
golden mismatch at score index 1. A parallel run also exposed the existing
Metal profile's sensitivity to concurrent device timing, while ten consecutive
isolated profile runs passed.

A same-machine one-warmup/five-sample directional check measured fully
resident at `303.2 ms` total (`266.0-329.5`) and `253.1 ms` in the
quantization pipeline. Throughput measured `257.0 ms` (`236.5-383.7`) and
`209.6 ms` in the pipeline. The immediately preceding checkpoint was
`298.4/250.9 ms` and `252.7/205.9 ms`, respectively. The ranges overlap, so
the small transfer reduction is performance-neutral at this public boundary;
its main value is simplifying the dependency chain before evaluation
submission and synchronization are fused.

The 4672x5584 doughnut sample also completed at target `1.2` in `4698.8 ms`
(`191.9 ms` preparation and `4506.9 ms` selected attempt), producing a
`2769119`-byte codestream. Independent `djxl` 0.12 decoded it back to a
4672x5584 PFM.

#### Chained resident-evaluation checkpoint (2026-08-28)

The next P4 slice moves the deterministic distance-driven Butteraugli policy
update onto Metal and encodes all dependent evaluations into one compute
command buffer. The adjusted initial field is retained separately for the
second-update pull, up to five scores remain device-resident, and numeric
failure state is sticky across every pass. One final completion wait reads the
final field, final block map, all scores, quantizer/frame state, and requested
reconstructed RGB. Initial field adjustment and CPU fixed-CfL preparation
remain separate. Exact-coefficient and maximum-error modes are unchanged.

Direct Release tests compare the fused zero-to-four-update path with the serial
policy oracle, verify one dependent-evaluation submission independent of the
iteration count, preserve padded outputs, and inject upload, submission,
completion, numeric, and readback failures. The complete serial suite passed
`55/56`; the sole failure remains the inherited pinned CPU score mismatch of
`4.4524669647216797e-05` at index 1.

One Apple M4 Pro directional public-workflow run used one warmup and five
alternating samples at padded 1080p and target `1.2`. Fully resident measured
`288.2 ms` total (`260.6-383.4`) and `238.2 ms` in the quantization pipeline,
versus the preceding `303.2/253.1 ms` checkpoint. Throughput measured
`259.2 ms` total (`232.5-410.1`) and `215.8 ms` in the pipeline, versus
`257.0/209.6 ms`. The ranges are noisy and overlap; the fully-resident shift is
a directional 5-6% improvement, while throughput is performance-neutral.

The 4672x5584 doughnut sample retained its `2769119`-byte output and `1.69531`
reported final score. A single run measured `5157.6 ms` total (`188.1 ms`
preparation and `4969.4 ms` selected attempt), and independent `djxl` 0.12
decoding produced a 4672x5584 PFM. This natural-image timing is noisier and
slower than the preceding single run, so it is a decode/regression check rather
than a speedup claim.

#### Selective resident-readback checkpoint (2026-08-28)

The production fully-resident and throughput Butteraugli workflows now use an
encoding-only materialization request. After the fused command buffer, Metal
reads one device error word, the contiguous score history, and only the
quantizer/raw-quant/quantized DC and AC payload needed for the final frame.
Final float quant, block-map, and reconstructed-RGB transfers are independently
optional and remain enabled for public diagnostic calls. Exact-coefficient,
maximum-error, maximum-throughput, public AQ, and public pipeline contracts are
unchanged. Host RGB and resident quant-field staging are allocated lazily when
those diagnostics are actually requested.

Direct tests account for each readback byte class, exercise zero through four
updates, compare full and frame-only coefficient output, preserve padded and
poisoned diagnostic storage, and inject staging, upload, submission,
completion, numeric, and readback failures. The encoding-only pipeline produces
the same frame, score history, and codestream as the full resident diagnostic
path.

One Apple M4 Pro balanced public-workflow run used one warmup and five samples
at padded 1080p and target `1.2`. Fully resident measured `274.965 ms` total
(`257.452-299.968`) and `230.141 ms` in the quantization pipeline, versus the
preceding `288.205/238.171 ms` checkpoint. Throughput measured `241.725 ms`
total (`223.755-264.516`) and `198.266 ms` in the pipeline, versus
`259.226/215.820 ms`. The corresponding median reductions are `4.6%`/`3.4%`
and `6.8%`/`8.1%`; the ranges are noisy, so these remain directional.

The 4672x5584 doughnut sample again produced exactly `2769119` bytes and score
`1.69531`. One run measured `4547.1 ms` total (`183.4 ms` preparation and
`4363.7 ms` selected attempt), and `djxl` 0.12 independently decoded it to a
4672x5584 PFM. As before, this single natural-image timing is a regression and
decode check rather than a retained latency distribution.

#### Codestream-only RGB-readback checkpoint (2026-08-28)

The internal encoding-only materialization now also covers exact-coefficient
and maximum-error Metal workflows. Their serial CPU policy still consumes the
per-evaluation block map and metric scalars, but the last evaluation requests
the `VarDctEncoderFrame` without reconstructed linear RGB. Exact-coefficient
mode already owns the authoritative frame on the host, so its last evaluation
does not download coefficients either. Maximum-throughput was already
frame-only. Public AQ and quantization-pipeline calls retain their complete
diagnostic output.

Direct Release tests compare full and frame-only generic evaluations, account
for Butteraugli and maximum-error readback classes, and verify zero RGB bytes
for frame-only output. Encoding-only exact and six-evaluation maximum-error
pipelines preserve score history, maximum-error outcome, frame, and codestream
while poisoned quant-field, block-map, and RGB diagnostics remain untouched.
The focused Metal, policy, pipeline, and codestream-workflow tests pass.

One Apple M4 Pro balanced public-workflow run used one warmup and five
padded-1080p exact-coefficient samples at target `1.2`. Metal measured
`624.544 ms` total (`614.039-641.636`), `362.392 ms` in the quantization
pipeline, and `10.384x` paired speedup (`10.107-10.601x`); CPU and Metal both
produced `636092` bytes. The removed final RGB transfer is `24847212` bytes for
the `1919x1079` source. Because no isolated same-revision pre-change
distribution was retained, this is a post-change checkpoint rather than a
speedup attribution.

#### Butteraugli Malta-accumulation checkpoint (2026-08-28)

The next resident-kernel slice removes six full-plane dispatches from every
Butteraugli difference scale. The first UHF Malta response for each populated
AC channel now initializes that accumulator directly; AC channel 2 and all
three DC planes are left for the following L2 kernel, which overwrites them.
Later Malta responses retain the established UHF, HF, then MF addition order.

Three independent Apple M4 Pro Release process pairs alternated retained
pre-change and post-change standalone benchmark binaries. Each process used
two warmups and eleven rotated padded-1080p samples. Resident-consumer E2E
medians moved from `17.682333-17.727541 ms` to
`16.933625-17.032958 ms`, a per-pair reduction of `3.74-4.23%`.
Resident-comparison median ranges overlapped, so the leaf-kernel result is not
yet attributed to the complete public encoder. Focused standalone
Butteraugli, resident AQ, policy, and quantization-pipeline coverage passes
without changing numerical tolerances or materialization contracts.

#### Butteraugli reference-mask checkpoint (2026-08-28)

Prepared Butteraugli now retains the blurred full-resolution reference
activity mask used by fuzzy erosion and masked AC comparison. Repeated
comparisons omit the invariant precompute and two-pass 13-tap blur; the
half-resolution scale preserves its prior recomputation sequence. At padded
1080p the cache adds `8,282,432` allocation bytes while peak logical comparison
scratch remains unchanged.

Three Apple M4 Pro Release process pairs alternated the retained post-Malta
binary and the cache. Each process used two warmups and eleven rotated samples.
Resident-comparison medians moved from `16.998583-17.261250 ms` to
`16.527125-16.825083 ms`, a per-pair reduction of `2.18-3.50%`.
Resident-consumer E2E medians moved from `16.944083-17.127541 ms` to
`16.546041-16.824500 ms`, a `0.82-3.40%` reduction. Preparation and
unamortized first-comparison results were mixed, so this is a repeated-AQ
comparison optimization rather than a cold or one-shot claim. Focused
Butteraugli, resident policy, and quantization-pipeline gates pass unchanged.

#### Butteraugli response-accumulation checkpoint (2026-08-28)

Each Malta response dispatch now also initializes or adds to its AC
accumulator while retaining the response plane for diagnostic capture. The
fixed UHF, HF, then MF order is unchanged. This removes six full-plane
copy/add dispatches and response rereads per scale; the obsolete clear and add
pipelines are no longer created.

Three matched-build Apple M4 Pro Release process pairs alternated the cached-
mask baseline and fused path. Each process used two warmups and eleven rotated
padded-1080p samples. Resident-consumer E2E improved in all three pairs, from
`16.591958-17.633542 ms` to `16.212292-17.284333 ms`, a `0.87-2.29%`
reduction. Resident-comparison medians improved in two pairs and were neutral
in one, so the retained claim stays at the repeated consumer boundary.
Numerical tolerances, response capture, memory accounting, and public APIs are
unchanged.

#### Large-image resident-kernel checkpoint (2026-08-28)

The cumulative resident-kernel comparison uses `960ebcc` as the pre-change
baseline and `46d0b0d` as the optimized path. The measured changes are the
redundant Malta-clear removal, cached full-resolution reference mask, and
fused Malta response accumulation described above; the AC quant-norm benchmark
commit changes no production shader. Both detached builds include only the
same `metal-public-workflow` benchmark scope, which performs one CPU/Metal
codestream validation pair and then measures repeated complete Metal public
encodes without interleaving a long CPU encode between GPU samples.

Three independent Apple M4 Pro Release process pairs were run at both required
large-image sizes. Every process used two warmups; the first pair used eleven
samples per build and the other two used seven. Build order was reversed
between pairs. The fully-resident output remained deterministic at `620711`
bytes for padded 1080p and `2472782` bytes for padded 4K.

| Workload | Baseline total medians | Optimized total medians | Per-pair total reduction | Baseline Metal-pipeline medians | Optimized Metal-pipeline medians | Per-pair pipeline reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Padded 1080p | `255.491-261.794 ms` | `252.409-257.179 ms` | `1.21-2.71%` | `207.921-214.073 ms` | `204.360-208.125 ms` | `1.65-3.68%` |
| Padded 4K | `972.082-998.564 ms` | `956.953-972.133 ms` | `1.20-3.50%` | `821.673-846.639 ms` | `809.325-820.309 ms` | `1.50-3.11%` |

The focused codestream, Butteraugli operation, Butteraugli differential,
resident AQ, policy, reconstruction, postprocess, and Metal quantization-
pipeline gates pass `8/8`. The complete serial Release suite passes `55/56`;
the sole failure is the unchanged pinned CPU `quantization_pipeline` score
mismatch of `4.4524669647216797e-05` at index 1.

The first 4K pair's process-wide peak RSS was `4.58-4.68 GB`; this includes
the one-time CPU validation, libraries, allocator retention, all host output,
and Metal allocations. The two observations are not used to claim a memory
reduction. A separate alternating CPU/Metal 4K smoke pair measured about
`26.2 s` for CPU and `1.23-1.29 s` for Metal, but repeated paired samples had
wide thermal and unified-memory variance. The Metal-only public scope is
therefore the retained kernel-regression signal; the alternating protocol
remains the end-user speedup gate.

Several broader source-level opportunities were rejected rather than retained
without a measured large-boundary result. Production-only removal of three
coefficient-sized scratch poisons and one DC scratch poison would avoid
`75,038,400` reset bytes per padded-1080p reconstruction while preserving
final-output poisons and full diagnostic poisoning. Its matched command-buffer
median moved only from `20.672 ms` to `20.609 ms` with overlapping ranges, so
the added mode and weaker production scratch-coverage signal were removed. An
8x8 threadgroup-tiled EPF implementation passed the direct odd/partial/filter
corpus and AQ-policy tests but measured `20.606 ms` at the same command-buffer
boundary; a 16x8 variant was also neutral. Both were removed. Hoisting 5-tap
Gaussian weight normalization into dispatch constants likewise failed to
provide a stable matched-build win and was removed. These results keep the
remaining optimization order tied to measured 1080p and 4K public encode time
rather than memory-traffic estimates alone.

#### Resident quant-adjustment scheduling checkpoint (2026-08-29)

Fresh split-stage profiling at `63f46f3` showed that the combined reconstruction
labels had obscured the real cost: the inverse 32x32 and rectangular DCTs were
only a small part of the batch, while `AdjustQuantBlockAC` ran a serial
coefficient scan in thread zero of every 64-256-thread coefficient group. The
retained path moves that exact scan into a separate kernel with one thread per
transform anchor. Thousands of independent anchors are therefore packed into
ordinary threadgroups instead of reserving a complete threadgroup while all
but one thread waits. The selected raw quant and four Y thresholds are staged
in the already-live gathered-pixel buffer, then consumed by the unchanged
parallel coefficient kernel. No new persistent allocation or host handoff is
introduced.

Seven independent Apple M4 Pro process pairs alternated the detached
`63f46f3` baseline and the optimized build. Every process used one warmup and
one measured `metal-public-workflow` encode with fully-resident SIMD AQ at
target `1.2`. This boundary begins with the generated linear-RGB image and ends
with the in-memory codestream; input generation and process/backend creation
are excluded. The quantization-pipeline result improved in every pair:

| Workload | Baseline pipeline median (range) | Optimized pipeline median (range) | Median reduction | Per-pair reduction |
| --- | ---: | ---: | ---: | ---: |
| Padded 1080p | `99.915 ms` (`99.132-104.955`) | `89.395 ms` (`88.837-91.743`) | `10.520 ms` / `10.53%` | `8.18-13.95%` |
| Padded 4K | `401.058 ms` (`393.493-417.474`) | `357.289 ms` (`345.014-395.121`) | `43.769 ms` / `10.91%` | `5.35-13.28%` |

Complete public-encode medians moved from `463.488` to `457.110 ms` at 1080p
and from `1406.304` to `1357.618 ms` at 4K. Codestream serialization dominated
those totals and remained noisy: 1080p pairwise totals were mixed, while six of
seven 4K pairs improved. The retained performance claim is therefore the
complete quantization-pipeline boundary, with the public totals reported only
as end-to-end context. Both variants produced identical `420268`-byte 1080p
and `1640942`-byte 4K output sizes in every measured process. A separate
seeded-plasma `3839x2159` parity encode compared the actual files: baseline and
optimized fully-resident outputs were byte-identical at `1161067` bytes with
SHA-256 `3df202cd4bcc3eba3f759378cfe6aba176aae85715b831569e78ae9c22ae79a9`;
`djxl` 0.12 independently decoded the optimized file to `3839x2159` pixels.

With the deliberately finer stage boundaries, median sampled 4K
`aq.reconstruction` time fell from `72.658` to `18.468 ms`; the three large
coefficient substages fell from `22.307/15.837/11.927 ms` to
`2.324/1.497/1.176 ms`. These timestamp-separated values explain attribution
but are not substituted for the process timings above. A rejected prototype
instead parallelized each anchor with 12 KiB of threadgroup scratch. It kept
exact output but regressed all seven 4K pairs from a `396.125` to `412.294 ms`
pipeline median, so that implementation was removed.

The complete Release suite passes `58/59`; the sole failure is the inherited
pinned CPU `quantization_pipeline` score mismatch of
`4.4524669647216797e-05` at index 1. Metal reconstruction, AQ evaluation,
postprocess, and complete GPU-pipeline tests pass, including exact frame and
codestream checks and unchanged numeric tolerances.

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

## Image-level throughput driver

`VarDctBatchEncoder` is a persistent bounded worker pool for independent public
workflow calls. A batch retains input order and returns one status, codestream,
summary, and timing record per image. Each worker owns its prepared image state
while all forced or automatically selected Metal calls reuse the process-wide
production backend. This is coarse-grained pipelining: CPU preparation and
codestream serialization for some images can overlap other images' Metal work,
but images are not packed into a new fused kernel dispatch.

The benchmark compares a persistent one-worker driver with persistent
`1`, `2`, `4`, and `8` worker drivers. Each row processes exactly the reported
batch size. Samples alternate serial-first and batch-first order, and every
result must match the single-image codestream and deterministic summary before
its timing is accepted. The measured boundary starts with an already-generated
linear RGB image and ends with in-memory codestreams; image decoding, input
generation, file I/O, and driver construction are excluded. The same synthetic
read-only source is submitted for every item, while all preparation, GPU
scratch, summaries, and codestream outputs remain independent.

A directional Apple M4 Pro Release run on 2026-08-28, rebased onto `0e96f0b`,
used maximum-throughput Metal AQ, one warmup per path, and three alternating
paired samples. It is one process, not a retained cross-process performance
claim:

| Workload | Batch | Batch median | Median/image | Images/s | Paired speedup median (range) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64x64 | 1 | 6.602 ms | 6.602 ms | 151.5 | 1.011x (0.932-1.052x) |
| 64x64 | 2 | 6.756 ms | 3.378 ms | 296.0 | 2.063x (1.874-2.120x) |
| 64x64 | 4 | 10.458 ms | 2.615 ms | 382.5 | 2.594x (2.358-2.724x) |
| 64x64 | 8 | 8.087 ms | 1.011 ms | 989.2 | 3.743x (3.676-4.389x) |
| 256x192 | 1 | 8.236 ms | 8.236 ms | 121.4 | 1.006x (0.909-1.025x) |
| 256x192 | 2 | 9.229 ms | 4.614 ms | 216.7 | 1.853x (1.686-1.932x) |
| 256x192 | 4 | 9.287 ms | 2.322 ms | 430.7 | 3.117x (2.952-3.295x) |
| 256x192 | 8 | 18.436 ms | 2.305 ms | 433.9 | 4.001x (3.809-4.577x) |
| 512x384 | 1 | 14.233 ms | 14.233 ms | 70.3 | 1.029x (0.977-1.032x) |
| 512x384 | 2 | 21.719 ms | 10.859 ms | 92.1 | 1.777x (1.737-1.808x) |
| 512x384 | 4 | 20.242 ms | 5.060 ms | 197.6 | 2.900x (2.584-3.013x) |
| 512x384 | 8 | 33.538 ms | 4.192 ms | 238.5 | 4.253x (3.435-4.724x) |
| 1080p | 1 | 98.469 ms | 98.469 ms | 10.2 | 1.032x (0.984-1.036x) |
| 1080p | 2 | 114.191 ms | 57.095 ms | 17.5 | 1.800x (1.643-1.931x) |
| 1080p | 4 | 160.499 ms | 40.125 ms | 24.9 | 2.593x (2.303-2.636x) |
| 1080p | 8 | 363.938 ms | 45.492 ms | 22.0 | 2.287x (2.214-2.431x) |
| 4K | 1 | 341.618 ms | 341.618 ms | 2.9 | 1.003x (0.902-1.046x) |
| 4K | 2 | 458.673 ms | 229.337 ms | 4.4 | 1.450x (1.351-1.636x) |
| 4K | 4 | 978.583 ms | 244.646 ms | 4.1 | 1.317x (1.299-2.047x) |
| 4K | 8 | 1837.427 ms | 229.678 ms | 4.4 | 1.355x (1.288-1.820x) |

The batch-size-one rows remain near parity, so the driver does not improve
single-image latency. This directional run shows positive median overlap at
every image size, with the strongest scaling at the three smaller sizes and
flatter, more variable gains at 1080p and especially 4K. It does not establish
stable scaling for separately decoded source images, exact AQ, target-size
searches, or a multi-process service. Reproduce the complete matrix with:

```sh
just image-batch-benchmark all 1,2,4,8 3 1 metal maximum-throughput
```

## Stop rules

- Do not optimize a standalone DCT, Butteraugli, or AC-search kernel unless the
  public profile identifies it as a material part of the remaining budget.
- Do not call a faster but decision-changing path exact.
- Do not use one-shot or sequential before/after timings for a retained
  optimization.
- Do not widen an existing production tolerance to make a performance patch
  pass; use the explicit throughput track when accuracy is intentionally traded.
