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

The remaining rate-control gaps are:

- no prepared state reused across target-size attempts;
- no hardened failed-candidate or configurable closest-absolute size search;
- no fully resident `AdjustQuantBlockAC` device pass; and
- no resampling-specific AQ bypass. The bypass policy is small, but the current
  codestream profile does not support resampling.

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

Target-byte and target-BPP requests use a bounded search over complete encodes
in the Butteraugli interval `[0.01, 10.0]`. The default allows 12 attempts and
accepts the largest result at or below the budget within a relative tolerance
of `0.005`; the result reports the effective byte and tolerance budgets,
selected Butteraugli target, attempt count, and whether the requested tolerance
was met. Infeasible requests still return a valid best candidate and report
`target_size_met = false`.

The `gjxl_rate_control_probe` tool measures the implemented Butteraugli mode
across strictly increasing targets and one or more PFM corpus inputs. For
example:

```sh
just rate-control-probe testdata/codestream_sample.pfm 1.0,1.2 --backend cpu
```

The probe emits CSV containing the input and extent, requested target, encoded
bytes and BPP, final score, actual backend and Metal mode, per-encode and total
time, size-monotonicity flag, and strategy counts.

The qualified Metal boundary intentionally retains authoritative coefficient
decisions on the CPU. CPU coefficient coding now applies the pinned
`AdjustQuantBlockAC` policy in Y, X, B evaluation order, stores the selected
shared raw quant at the transform anchor only, retains Y's adjusted dead-zone
thresholds, and requantizes all three channels from that decision. The
exact-coefficient Metal path consumes the resulting frame and adjusted raw
field, including its EPF input, so frame and codestream decisions remain exact.

The fully resident path remains experimental and explicitly uses the fixed-raw-
quant decision mode as its CPU oracle until task 13 lands. FP32 coefficient ties
can change integer coefficient decisions and compound across AQ iterations.
Rate-control work must not silently weaken that boundary.

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
- break equal-size ties using the lower perceptual score, then the lower target
  value;
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

**Status:** complete for exact-coefficient Metal; the fully resident device
port remains task 13.

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

### 12. Reuse target-invariant preparation across attempts

**Estimate:** 2–4 weeks.

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

### 13. Port `AdjustQuantBlockAC` to the fully resident path

**Estimate:** 2–4 weeks plus numerical-parity contingency.

Add a device pass that computes the cross-channel shared-quant adjustment,
retains the adjusted Y thresholds, and requantizes the transform without
pixel-sized host readback. Validate its intermediate decisions directly for all
strategies before composing it with iterative AQ.

This task sits at the medium/hard boundary. The heuristic contains
threshold-sensitive integer decisions, while the current fully resident FP32
transform path already fails the unchanged production decision gate. A bounded
CPU decision handoff is acceptable only if the resulting path is described as
a hybrid boundary rather than fully resident. Pure-resident exact decision
parity remains research-risky and has no guaranteed schedule.

## Suggested milestones

### RC0: Observable best-effort size control

Complete tasks 1–5.

Exit criteria:

- the existing API remains source-compatible;
- target bytes and target BPP produce deterministic valid codestreams;
- attempt count, achieved size, selected target, and target-met state are
  reported; and
- the corpus probe documents rate-curve plateaus and exceptions.

Expected effort: approximately one focused week.

### RC1: Current-profile CPU decision completeness

Complete tasks 7 and 9, then validate the existing Butteraugli path again.

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

Exit criteria:

- target-size search handles non-monotonic and infeasible corpus cases;
- target-invariant preparation is reused across attempts;
- timing includes preparation, every attempted encode, serialization, and
  candidate selection; and
- performance is reported as repeated ranges for named image sizes and search
  tolerances.

Expected cumulative effort for RC0 through RC3: approximately 6–10 weeks.

Task 13 is an experimental follow-on and is not an RC3 exit criterion.

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
