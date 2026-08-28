# Rate-control completeness roadmap

This document tracks the bounded work required to complete rate-control policy
for GJXL's current VarDCT encoder profile. It covers the existing Butteraugli
target workflow, maximum-error adaptive quantization (AQ), best-effort
target-size control, and the encoder-side coefficient adjustment that affects
the authoritative quantization decisions.

The scope is the current seven-strategy, single-pass, 4:4:4 profile documented
in [`quantization.md`](quantization.md) and [`codestream.md`](codestream.md).
Unless a task says otherwise, estimates assume one engineer already familiar
with this codebase, the pinned libjxl revision, access to a real Metal device,
and the existing CPU, Metal, codestream, and independent-decoder test harnesses.
Estimates are planning ranges rather than commitments and are not additive.

Compression-model optimization is a separate concern. A target-size controller
may consume the output of the current entropy writer, but ANS, custom
coefficient orders, LZ77, and rate-distortion retuning do not belong to this
roadmap.

## Definitions

For this roadmap, rate-control completeness comprises four independently
testable capabilities:

1. **Butteraugli target:** retain the existing perceptual-distance target and
   deterministic AQ update policy.
2. **Maximum error:** constrain the maximum per-channel reconstruction error in
   each selected transform footprint using an alternate AQ update policy.
3. **Target size:** search the quality control variable for the closest valid
   codestream at or below a requested byte or bits-per-pixel budget.
4. **Complete coefficient decisions:** apply the encoder-side
   `AdjustQuantBlockAC` heuristic before final coefficient quantization.

Maximum-error control and `AdjustQuantBlockAC` improve encoder-decision parity;
they are not themselves target-bitrate algorithms. Target-size and target-BPP
control are outer search policies over complete encodes.

## Current foundation

The public workflow currently accepts one `butteraugli_target`, selects a CPU
or Metal backend, returns the encoded byte count, and preserves atomic output
behavior. The AQ implementation already provides:

- an initial quant field and AC-strategy grid;
- a deterministic bounded iteration policy;
- complete encode/reconstruct/measure evaluations;
- a per-block distance map and scalar score;
- final `VarDctEncoderFrame` materialization;
- a qualified Metal exact-coefficient path; and
- an experimental fully resident Metal coefficient path.

The remaining rate-control gap is the resampling-specific AQ bypass. The bypass
policy is small, but the current codestream profile does not support
resampling, so exposing that rule would not yet provide an end-to-end feature.

The public request/result types represent all four modes and validate only the
active mode. Maximum-error requests are implemented on the CPU and explicitly
forced Metal paths. Automatic maximum-error selection remains on the CPU; this
milestone did not widen the qualified automatic-backend gate. The result
reports the requested and achieved per-channel XYB errors, normalized maximum,
fixed evaluation count, and whether the request was met, exhausted the
iteration budget, or exhausted the representable quantization range.

Maximum-error AQ uses a named internal initialization target of `1.0` and a
fixed initial DC quantization of `16 * sqrt(0.1)`, independent of the inactive
public Butteraugli target. It performs five pinned transform-local updates and
one final verification evaluation. The policy retains the closest evaluated
field at or below the hard maximum, preventing a later below-half update from
discarding a valid result through loop-filter interaction.

Target-byte and target-BPP requests use a deterministic bounded subdivision
over complete encodes in the Butteraugli interval `[0.01, 10.0]`. The search
does not use endpoint sizes as proof of monotonicity. The default allows 12
attempts and accepts the largest result at or below the budget within a
relative tolerance of `0.005`; callers may instead request the closest absolute
byte count, whose tolerance is symmetric. The result reports the effective byte
and tolerance budgets, selection policy, selected Butteraugli target, total and
failed attempt counts, whether the requested tolerance was met, and whether the
bounded search was exhausted. Infeasible requests still return a valid best
candidate and report `target_size_met = false`.

The `gjxl_rate_control_probe` tool measures the implemented Butteraugli mode
across strictly increasing targets and one or more PFM corpus inputs. For
example:

```sh
just rate-control-probe testdata/codestream_sample.pfm 1.0,1.2 --backend cpu
```

The probe emits CSV containing the input and extent, requested target, encoded
bytes and BPP, final score, actual backend and Metal mode, per-encode and total
time, size-monotonicity flag, and strategy counts.

Forced Metal exposes four coefficient/AQ execution choices without changing
the outer rate-control request:

- `exact-coefficients` retains authoritative CPU coefficient decisions and is
  the only Metal mode eligible for automatic selection;
- `fully-resident` keeps iterative coefficient coding, reconstruction, and
  scoring on Metal for the requested AQ iteration count;
- `throughput` uses the same resident path with one AQ update; and
- `maximum-throughput` fixes DCT8, encodes the adjusted initial field, and
  skips reconstruction and perceptual scoring.

The three experimental modes require an explicitly forced Metal backend.
Target-byte and target-BPP searches can use all four because they select from
actual serialized sizes. Maximum-error control can use exact, fully resident,
or throughput modes, but rejects maximum-throughput because that path does not
perform the error evaluation. The corpus rate-control probe likewise excludes
maximum-throughput because its CSV contract requires a score history.

The qualified Metal boundary intentionally retains authoritative coefficient
decisions on the CPU. CPU coefficient coding now applies the pinned
`AdjustQuantBlockAC` policy in Y, X, B evaluation order, stores the selected
shared raw quant at the transform anchor only, retains Y's adjusted dead-zone
thresholds, and requantizes all three channels from that decision. The
exact-coefficient Metal path consumes the resulting frame and adjusted raw
field, including its EPF input, so frame and codestream decisions remain exact.

The fully resident and throughput paths now apply `AdjustQuantBlockAC` directly
to their Metal forward coefficients. The selected shared raw quant is stored at
the transform anchor, the adjusted Y thresholds feed Y quantization, and EPF
inverse sigma is recomputed on device from the selected anchor. Final frame
materialization reads back the block-resolution raw-quant field but no pixel
image for this decision. The paths remain experimental because FP32 forward-
transform ties can still differ from the CPU double-precision coefficient
oracle and compound across AQ iterations; this work does not weaken that
boundary or make either mode eligible for automatic selection.

## Dependency order

```text
rate-control request and result contract
                  |
                  v
       rate-curve corpus probe
                  |
          +-------+--------+
          |                |
          v                v
simple target-size    CPU decision parity
search                AdjustQuantBlockAC
          |                |
          v                v
robust search and     maximum-error CPU AQ
prepared reuse             |
                           v
                  qualified Metal parity
                           |
                           v
                  experimental resident work
```

## Easy tasks

Easy tasks should require at most several focused days and should not change
the numerical coefficient oracle.

### 1. Define the public request contract

**Estimate:** 1–2 days.

Extend `VarDctEncodingOptions` with one unambiguous rate-control mode. The
contract should support:

- the existing positive finite Butteraugli target;
- three positive finite maximum-error limits;
- a nonzero target byte count; and
- a positive finite target BPP, converted using the unpadded source pixel
  count.

Only one mode may be active. Invalid or internally inconsistent requests must
return `InvalidArgument` without modifying the codestream or summary. Preserve
source compatibility for callers that only set `butteraugli_target`.

Acceptance criteria:

- default construction retains current behavior;
- every mode validates its own numerical range;
- inactive mode fields cannot affect output; and
- invalid requests preserve caller-visible outputs exactly.

### 2. Report the achieved result

**Estimate:** 1–2 days.

Extend `VarDctEncodingSummary` without exposing temporary storage. Report:

- requested rate-control mode;
- requested and effective byte budget;
- achieved bytes and BPP;
- selected Butteraugli control value for a size search;
- encode-attempt count;
- whether the requested size tolerance was met; and
- the existing backend, Metal AQ mode, strategy counts, and score history for
  the selected candidate.

An infeasible but valid target-size request should return the selected valid
candidate and report that the target was not met. Invalid input and encoder
failure remain status errors.

### 3. Add a rate-curve corpus probe

**Estimate:** 2–4 days.

Add a benchmark or diagnostic that encodes a fixed corpus over a configured
Butteraugli-target range and records:

- target and achieved perceptual score;
- encoded bytes and BPP;
- selected strategy counts;
- CPU, exact-coefficient Metal, or fully resident Metal mode;
- per-attempt and total wall time; and
- whether successive samples are monotonic in size.

The probe establishes real search brackets, plateaus, and non-monotonic cases.
It must not be presented as a compression-ratio claim until run on a named
corpus at matched quality.

### 4. Implement a simple target-size controller

**Estimate:** 2–5 days.

Build a bounded outer search around the existing public encode. Start with a
bracketed bisection over `butteraugli_target`; retain every successful candidate
so the selected output never requires an uncounted final re-encode.

Recommended initial semantics are:

- prefer the largest valid codestream at or below the byte budget;
- if no candidate is below the budget, return the smallest valid candidate;
- break equal-size ties using the lower perceptual score when one is available,
  then the lower target value;
- stop on the configured byte tolerance, an unchanged-size plateau, exhausted
  bracket, or maximum attempt count; and
- produce identical bytes and summary for identical input and options.

This first controller may perform several complete encodes. It establishes
functional target-size control, not a latency-qualified production path.

### 5. Define termination and infeasibility policy

**Estimate:** 1–2 days.

Specify and test:

- absolute or relative byte tolerance;
- maximum encode attempts;
- initial bracket and bounded bracket expansion;
- repeated-size plateau detection;
- minimum-codestream-size targets;
- oversized targets for which the highest supported quality remains smaller;
  and
- deterministic selection when strategy changes make the observed rate curve
  locally non-monotonic.

The controller must always terminate. It must not assume strict monotonicity.

### 6. Add the resampling-specific AQ bypass rule

**Estimate:** less than 1 day after resampling exists.

Mirror the pinned policy that skips the Butteraugli AQ loop for downsampled
opsin input below its high-distance threshold. Unit-test the predicate
independently now if useful, but do not advertise resampling rate-control
support: image resampling, frame signaling, reconstruction, and conformance are
separate hard work and remain unsupported by the current profile.

## Medium tasks

Medium tasks change authoritative encoder decisions or require reusable
multi-attempt orchestration. They need direct pinned-oracle coverage in addition
to end-to-end tests.

### 7. Implement CPU `AdjustQuantBlockAC`

**Estimate:** 1–2 weeks.

**Status:** complete for the current seven-strategy profile.

Port the pinned encoder heuristic for the current seven strategies. It must:

- inspect unquantized coefficients for all three XYB channels;
- derive the shared raw-quant adjustment in Y, X, B evaluation order;
- retain the adjusted Y dead-zone thresholds;
- clamp raw quant exactly at the encoder-policy limit;
- use the adjusted anchor raw quant as the shared coefficient-coding decision
  for all three channels; and
- requantize all channels using the final shared decision.

The existing fixed `QuantizeAcBlock` primitive should remain independently
testable. Do not hide cross-channel mutation inside an API that appears to
quantize one channel in isolation.

Acceptance criteria:

- direct parity with the pinned helper for DCT8, square, and rectangular
  strategies;
- sparse, flat, active, high-frequency-border, threshold-tie, and quant-limit
  fixtures;
- exact raw-quant and quantized-coefficient parity;
- pinned anchor-only raw-quant storage with covered non-anchor cells preserved;
  and
- unchanged atomic failure behavior.

### 8. Integrate adjusted coefficients into qualified Metal AQ

**Estimate:** 2–4 days after the CPU implementation.

**Status:** complete for exact-coefficient Metal. The resident device
composition is tracked separately in task 13.

The production Metal path already accepts authoritative CPU coefficient
decisions. Feed the adjusted raw-quant field and coefficient frame through that
existing exact-coefficient boundary, then rerun the established frame,
codestream, reconstruction, distance-map, and score-history parity gates.

This task does not authorize selecting the fully resident coefficient path
automatically.

### 9. Implement maximum-error AQ on the CPU

**Estimate:** 1–2 weeks.

**Status:** complete for the current seven-strategy profile.

Add an alternate deterministic policy that, for each transform anchor:

- computes the maximum normalized reconstruction error over all covered source
  pixels and all three channels;
- ignores padded pixels outside the original source extent;
- propagates one multiplier over the complete transform footprint;
- targets the specified accepted error interval; and
- performs a fixed, documented maximum number of evaluations.

Define a deterministic initial DC quantization choice instead of implicitly
reusing an unrelated public Butteraugli target. The maximum-error limits are the
control contract; any internal initialization parameter must remain an
implementation detail or be named separately.

Acceptance criteria:

- direct pinned-policy fixtures for below-half, accepted, and over-limit error;
- all seven strategies, edge transforms, three independently limiting
  channels, and zero-padding coverage;
- deterministic quant fields, raw quant, coefficients, and evaluation count;
- independent verification that the selected output satisfies the requested
  bounds when the representable quantization range permits it; and
- explicit reporting when a bound is infeasible.

### 10. Add resident maximum-error reduction

**Estimate:** 1–2 weeks after the CPU oracle.

**Status:** complete for exact-coefficient and experimental fully resident
Metal modes; automatic selection remains CPU-only.

Reuse the prepared Metal reconstruction and add a strategy-aware per-transform
maximum-error reduction. Keep reconstructed images resident and read back only
the bounded transform-error map required by the unchanged CPU update policy.

Validation must compare the reduction directly against the CPU oracle before
testing the composed policy. Exact-coefficient Metal must preserve the CPU
raw-quant and final-frame decisions. Fully resident numerical deviations remain
a separately reported experimental result.

The prepared Metal evaluator now selects either Butteraugli or maximum-error as
its active metric. The maximum-error kernel compares resident coding and
filtered reconstructed opsin, ignores padded pixels, reduces each of the seven
strategy footprints, and returns only the block map plus three actual channel
maxima per transform. It performs one submission and no steady-state device
allocation per evaluation.

The direct mixed-strategy, odd-edge fixture agrees with the CPU reduction oracle
within `1e-6`. The composed exact-coefficient path preserves the CPU final frame
and codestream exactly; accumulated quant-field, error-map, score, achieved
error, and reconstruction diagnostics remain within the existing `2e-3`
cross-backend cap. The fully resident path returns deterministic finite results
but retains its experimental decision-parity status.

### 11. Harden target-size search

**Estimate:** 1–2 weeks after the simple controller and corpus probe.

**Status:** complete for the current public byte and BPP modes.

Replace strict-bisection assumptions with a deterministic bounded search that
handles:

- local non-monotonicity caused by strategy and integer-quant changes;
- several quality targets producing identical bytes;
- failed candidates without discarding earlier valid output;
- configurable under-budget versus closest-absolute selection;
- byte and BPP targets through one normalized internal budget; and
- attempt-budget exhaustion with a complete result summary.

Use actual serialized bytes as the authoritative rate measurement. Estimated
entropy cost may guide candidate selection but cannot satisfy the target-size
contract.

The hardened search subdivides the widest remaining target interval with a
stable lower-target tie-break, so local reversals do not corrupt a monotonic
bracket. Every successful candidate remains eligible for deterministic final
selection. Candidate-local encoder failures count against the bounded attempt
budget but no longer discard an earlier valid result; if every attempt fails,
the first failure is returned atomically. A successful evaluator result with an
inconsistent byte count or summary remains a terminal internal-contract error.

`TargetSizeSelectionPolicy::kLargestAtOrBelow` retains the source-compatible
default. `kClosestAbsolute` minimizes absolute byte error and prefers the
under-budget candidate on equal-distance ties. Equal-size candidates in both
modes use final score when present and then the lower Butteraugli target as a
stable tie-break. A scoreless maximum-throughput candidate remains valid
because serialized size, not an internal perceptual evaluation, is the rate
contract.

### 12. Reuse target-invariant preparation across attempts

**Estimate:** 2–4 weeks.

**Status:** complete for CPU and exact-coefficient/experimental resident Metal
attempts in the current profile.

Refactor the monolithic public workflow into prepared and per-attempt state so a
target-size search can retain:

- validated source geometry and converted source data;
- target-invariant tables and allocations;
- prepared Butteraugli reference state;
- reusable Metal pipelines, buffers, and scratch; and
- candidate codestream and summary storage.

Do not assume that the initial quant field, AC-strategy grid, color correlation,
or final AQ state is invariant across quality targets. Cache only inputs proven
independent of the searched control value. Report preparation, aggregate search,
and selected-attempt time separately.

The workflow now separates `PreparedWorkflow` from `EncodePreparedAttempt`.
Preparation retains validated geometry, the edge-extended linear source,
converted coding opsin, inverse-Gaborish opsin, initial color correlation,
default EPF sharpness, CPU workspaces, and a native
`PreparedButteraugliReference`. The cached reference stores both required
perceptual scales, so every CPU AQ evaluation and every subsequent search
attempt transforms only the distorted reconstruction. Initial quantization,
strategy selection, final color correlation, quantizer state, coefficient
decisions, reconstruction, and AQ output are recomputed for each target.

The first Metal attempt prepares one worst-case frame allocation and device
Butteraugli reference. Later attempts call `PreparedAqEvaluation::Reconfigure`
to upload only the new strategy and EPF metadata while retaining source,
pipelines, buffers, and scratch. Direct mixed-to-DCT8 reconfiguration matches a
fresh preparation exactly, performs no device allocation or evaluation
submission, and preserves the prior state after invalid metadata. Complete
two-target GPU attempts retain the same prepared object while matching one-shot
frames and codestreams exactly.

`EncodeLinearRgbVarDctCodestreamProfiled` keeps non-deterministic timing out of
`VarDctEncodingSummary`. It reports source/host preparation, every attempted
encode including serialization and failures, aggregate search including final
candidate selection, the retained attempt, and end-to-end time atomically. The
CLI prints the same preparation, selected-attempt, aggregate-search, and total
boundaries.

#### RC3 latency snapshot

These measurements are observational, not compression-ratio claims. They used
a Release build on an Apple M4 Pro running macOS 15.6, one warmup followed by
five sequential samples, the under-budget policy, a 12-attempt maximum, and
the exact-coefficient Metal mode. The `128x96` input is a three-channel center
crop of pinned libjxl's `grayscale_patches_on_splines.pfm`; the `17x13` input is
`testdata/codestream_sample.pfm`. Times are min-max milliseconds.

| Input and request | Backend | Attempts | Preparation | Selected attempt | Aggregate search | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 17x13, 280 B, tolerance 0.1 | CPU | 7 | 0.194-0.251 | 1.120-3.011 | 7.722-11.052 | 7.917-11.303 |
| 17x13, 280 B, tolerance 0.005 | CPU | 12 | 0.190-0.194 | 1.143-1.190 | 12.792-13.109 | 12.984-13.299 |
| 128x96, 3400 B, tolerance 0.1 | CPU | 7 | 6.612-6.792 | 34.679-39.020 | 235.894-252.850 | 242.687-259.633 |
| 128x96, 3400 B, tolerance 0.005 | CPU | 12 | 6.680-6.956 | 35.062-39.393 | 403.268-429.504 | 410.025-436.360 |
| 128x96, 3400 B, tolerance 0.1 | forced Metal | 7 | 1.757-1.890 | 10.658-11.117 | 119.682-131.878 | 121.572-133.672 |
| 128x96, 3400 B, tolerance 0.005 | forced Metal | 12 | 1.817-1.914 | 10.629-12.041 | 167.916-186.970 | 169.775-188.787 |

Both tolerances selected deterministic 272-byte and 3206-byte codestreams for
the small and cropped inputs respectively. Tolerance `0.1` met each budget;
the tighter tolerance exhausted the bounded search without finding a candidate
inside the requested window. CPU and forced Metal bytes were identical. Pinned
`djxl` independently decoded the four `128x96`
CPU/Metal, loose/tight outputs and both `17x13` backend outputs.

### 13. Port `AdjustQuantBlockAC` to the fully resident path

**Estimate:** 2–4 weeks plus numerical-parity contingency.

**Status:** complete for the current seven-strategy resident and frame-only
Metal coefficient paths. CPU-identical resident forward coefficients remain
outside the claim.

The tested adjustment primitive is composed into the existing per-transform
resident coefficient kernel, so it adds neither an allocation nor a command
submission. One thread selects the cross-channel shared quant and adjusted Y
thresholds; the threadgroup then requantizes all channels with that decision.
The same command rewrites EPF inverse sigma over the transform footprint from
the adjusted anchor raw quant and the prepared sharpness LUT. Covered
non-anchor raw-quant cells remain unchanged, matching the pinned encoder.

Direct threshold, tie, quant-limit, and non-finite fixtures cover all seven
strategies. A mixed-strategy batched integration test checks shared raw quant,
adjusted Y coefficients, EPF inverse sigma, one-submission execution, and zero
post-preparation allocation against CPU oracles. Final frame assembly adds only
the block-grid raw-quant readback already required for codestream state; it
does not round-trip coefficients or a pixel-sized image for the decision.

The remaining research risk is a different claim: the resident Metal forward
transform is FP32 while the CPU oracle accumulates in double precision. An
input near a threshold may therefore make a different, internally consistent
adjustment decision. Resident output remains experimental and is not described
as CPU-bit-exact.

## Suggested milestones

### RC0: Observable best-effort size control

Complete tasks 1–5.

**Status:** complete for the current byte and BPP request contract.

Exit criteria:

- the existing API remains source-compatible;
- target bytes and target BPP produce deterministic valid codestreams;
- attempt count, achieved size, selected target, and target-met state are
  reported; and
- the corpus probe documents rate-curve plateaus and exceptions.

Expected effort: approximately one focused week.

### RC1: Current-profile CPU decision completeness

Complete tasks 7 and 9, then validate the existing Butteraugli path again.

**Status:** complete for the current seven-strategy profile.

Exit criteria:

- `AdjustQuantBlockAC` matches the pinned oracle;
- maximum-error AQ is available on the CPU;
- Butteraugli, maximum-error, and target-size modes are independently testable;
  and
- complete frame and codestream output remains deterministic.

Expected cumulative effort: approximately 3–5 weeks.

### RC2: Qualified Metal completeness

Complete tasks 8 and 10.

**Status:** complete for the current profile. This does not widen automatic
Metal selection for maximum-error requests.

Exit criteria:

- exact-coefficient Metal preserves authoritative CPU coefficient decisions in
  Butteraugli and maximum-error modes;
- resident maximum-error reduction passes its direct CPU oracle;
- bounded readback and zero-reallocation properties are retained; and
- automatic backend selection is not widened without a separate measured
  rollout gate.

### RC3: Production target-size latency

Complete tasks 11 and 12.

**Status:** complete for the current profile and bounded search contract.

Exit criteria:

- target-size search handles non-monotonic and infeasible corpus cases;
- target-invariant preparation is reused across attempts;
- timing includes preparation, every attempted encode, serialization, and
  candidate selection; and
- performance is reported as repeated ranges for named image sizes and search
  tolerances.

Expected cumulative effort for RC0 through RC3: approximately 6–10 weeks.

Task 13 is complete as an experimental resident feature and remains outside
the automatic-selection RC3 exit criterion.

## Cross-cutting validation

Every completed milestone should cover:

- valid raw JPEG XL decoding with the pinned independent decoder;
- atomic output on invalid input, allocation failure, backend failure, and
  exhausted search;
- deterministic bytes and summaries for repeated identical calls;
- odd source dimensions, padded edges, and every supported strategy;
- exact CPU/exact-coefficient Metal frame and codestream comparison;
- explicit tolerance reporting for fully resident experiments;
- target-size behavior below the minimum size, inside the normal rate curve,
  on a plateau, near a non-monotonic transition, and above the maximum tested
  quality; and
- maximum-error behavior where X, Y, or B independently determines the update.

Performance claims must include all encode attempts and serialization. A
single selected-attempt timing is not a target-size latency measurement.

## Hard work outside this roadmap

The following must not be counted as easy or medium rate-control completion:

- end-to-end spatial resampling and upsampling signaling;
- pure fully resident coefficient-decision equivalence with the current CPU
  double-precision oracle;
- GPU tokenization or entropy coding;
- ANS, custom coefficient orders, LZ77, or broader compression optimization;
- additional AC strategies, chroma subsampling, progressive passes, HDR, or
  non-default opsin transforms; and
- Modular-only or lossless encoding.

## Relevant implementations

- [`workflow.h`](../src/codestream/workflow.h)
- [`workflow.cpp`](../src/codestream/workflow.cpp)
- [`adaptive_quantization.h`](../src/codec/adaptive_quantization.h)
- [`adaptive_quantization.cpp`](../src/codec/adaptive_quantization.cpp)
- [`adaptive_quantization_internal.h`](../src/codec/adaptive_quantization_internal.h)
- [`butteraugli.h`](../src/codec/butteraugli.h)
- [`maximum_error.h`](../src/codec/maximum_error.h)
- [`maximum_error.cpp`](../src/codec/maximum_error.cpp)
- [`gpu/metal/kernels/aq_reduction.metal`](../src/gpu/metal/kernels/aq_reduction.metal)
- [`quantization.h`](../src/codec/quantization.h)
- [`quantization.cpp`](../src/codec/quantization.cpp)
- [`gpu/ops/adaptive_quantization.h`](../src/gpu/ops/adaptive_quantization.h)
- [`gpu/ops/adaptive_quantization.cpp`](../src/gpu/ops/adaptive_quantization.cpp)
- [`metal-aq.md`](metal-aq.md)
- [`quantization.md`](quantization.md)
- [`codestream.md`](codestream.md)

Pinned reference implementations:

- `third_party/libjxl/lib/jxl/enc_group.cc`
- `third_party/libjxl/lib/jxl/enc_adaptive_quantization.cc`
- `third_party/libjxl/tools/scripts/cjxl_bisect_size`
- `third_party/libjxl/tools/scripts/cjxl_bisect_bpp`
