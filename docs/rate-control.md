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
- an explicit Metal exact-coefficient compatibility path; and
- the default fully resident Metal coefficient path.

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

Metal exposes four coefficient/AQ execution choices without changing the outer
rate-control request:

- `exact-coefficients` retains authoritative CPU coefficient decisions and is
  the explicit reference/compatibility mode;
- `fully-resident` is the encoding default. It keeps iterative coefficient
  coding, reconstruction, and scoring on Metal for the requested AQ update
  count, then omits the terminal encoded-field diagnostic by default;
- `throughput` uses the same resident encoding path and default omission while
  retaining a separate one-update contract for complete diagnostic API calls;
  and
- `maximum-throughput` fixes DCT8, encodes the adjusted initial field, and
  skips reconstruction and perceptual scoring.

Automatic Butteraugli-target encoding may select fully resident Metal inside
the qualified geometry, device, and target interval. Automatic target-byte and
target-BPP searches with the resident default stay on CPU so one search cannot
mix CPU and resident rate curves; explicitly selected exact-coefficient mode
retains automatic target-size eligibility. Forced Metal target-size searches
can use all four modes because they select from actual serialized sizes.
Maximum-error control can use exact, fully resident, or throughput modes when
Metal is forced, but rejects maximum-throughput because that path does not
perform the error evaluation. Automatic maximum-error remains CPU-only. The
corpus rate-control probe likewise excludes maximum-throughput because its CSV
contract requires a score history.

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
modes use a score only when `final_butteraugli_score_evaluated` says that it
measures the encoded field, then use the lower Butteraugli target as a stable
tie-break. Resident encoding's retained update scores precede its final encoded
field, and even an explicitly collected terminal resident score remains a
diagnostic, so resident scores are not used as target-selection tie-breaks. A
scoreless
maximum-throughput candidate remains valid because serialized size, not an
internal perceptual evaluation, is the rate contract.

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

## Resident frontend sequence

Rate-control policy is complete for the current profile, but the experimental
resident paths still have avoidable CPU preprocessing and host/device
handoffs. The implementation order is deliberately narrower than changing the
authoritative exact-coefficient policy:

1. compute the fast pixel-domain initial CfL map from resident opsin;
2. generate the initial quant field and masks on Metal, retaining the small
   block-grid CPU quantizer decision at first;
3. move median, median absolute deviation, and raw-quant construction to Metal;
4. pass resident initial fields and masks directly into the experimental AC
   search paths; and
5. audit exact-coefficient and automatic modes for non-regression after every
   slice.

### Resident initial CfL

**Status:** complete for maximum-throughput frame-only encoding.

The frame-only preparation may now request its deterministic fast initial CfL
map from the already uploaded, pre-Gaborish coding opsin. The CfL kernel runs in
the existing coefficient-coding submission before inverse Gaborish and writes
the two signed 64x64-tile maps directly into the prepared reconstruction
buffers. It preserves the CPU policy's four-lane accumulation order and exact
integer factors on flat, structured, and partial-edge tile fixtures.

No pixel image or coefficient buffer crosses the host boundary for this step.
Final frame assembly reads back only the two compact tile maps, which are part
of the codestream state. Direct tests require one submission and zero
post-preparation allocations. Exact-coefficient, fully-resident, and throughput
mode inputs retain their prior host-map contracts; automatic backend selection
is unchanged.

A directional Apple M4 Pro Release run on padded 1080p used one warmup and five
CPU/Metal pairs. The public maximum-throughput Metal boundary measured a
`122.553 ms` median (`107.175-124.989 ms`), with `79.980 ms`
(`64.153-83.358 ms`) in the quantization pipeline and a `52.672x` paired
speedup median. The range overlaps the pre-port checkpoint, so this single
process is a regression guard, not a retained speedup claim. A balanced
multi-process comparison belongs at the end of the complete frontend sequence.

### Resident initial quantization with a bounded CPU decision

**Status:** complete for maximum-throughput frame-only encoding.

The pre-Gaborish resident opsin now feeds one Metal submission containing the
initial luma gradient and pixel mask, quarter-resolution aggregation, fuzzy
erosion, strategy mask, per-block gamma/high-frequency/blue modulation, and
the codec's mirrored 5x5 pixel-mask blur. The prepared frame context retains
all intermediate planes and is reused by the subsequent resident-CfL and
coefficient submission. The full-resolution masking maps are read back only to
preserve the existing diagnostic output contract; the CPU quantizer consumes
the much smaller final block field.

Direct structured and flat fixtures exercise two perceptual targets, the
high-target dampening branch, image boundaries, repeated prepared use, injected
numeric failure, atomic output, one submission per evaluation, and zero
post-preparation device allocation. Against the CPU oracle, the observed
maximum errors are below `2e-6` for the quant and strategy fields and `2e-5`
for the blurred pixel mask. The padded-1080p maximum-throughput codestream
remains `765599` bytes, matching the pre-port resident-CfL checkpoint.

A directional Apple M4 Pro Release run with one warmup and three CPU/Metal
pairs measured a public Metal median of `82.757 ms` (`68.933-94.248 ms`) and a
quantization-pipeline median of `43.492 ms` (`34.167-49.406 ms`). The paired
speedup median was `75.287x`. Relative to the preceding resident-CfL slice's
single-process medians, this is a directional `1.48x` public-boundary and
`1.84x` pipeline improvement. Retained claims still wait for the final balanced
multi-process audit.

### Resident median, MAD, and raw quantization

**Status:** complete for maximum-throughput frame-only encoding.

The maximum-throughput decision path no longer consumes the host copy of the
initial block field. Metal pads the positive block values to a power of two,
performs deterministic bitonic selection for the upper median, repeats the
selection over absolute deviations, derives the serialized quantizer
parameters, and constructs the raw-quant field in the same initial-quant
submission. Only the scalar DC-quant policy input is computed on the CPU. The
two quantizer integers cross the boundary for frame metadata; raw quant remains
resident until coefficient coding and its final codestream-state readback.

The existing initial quant, strategy-mask, and pixel-mask host copies remain so
the diagnostic pipeline output contract is unchanged, but none feeds the
maximum-throughput decision. The frame submission also omits the initial
raw-quant and EPF uploads: fixed DCT8 coverage lets resident
`AdjustQuantBlockAC` rewrite every raw value and corresponding inverse sigma.
Direct tests compare quantizer parameters and every initial raw-quant value
exactly with the CPU median/MAD oracle across flat and structured inputs and two
targets. The edge-geometry integration fixture retains exact final frame and
codestream bytes.

One Apple M4 Pro Release process with one warmup and three CPU/Metal pairs
measured a public Metal median of `81.219 ms` (`76.894-82.208 ms`) and a
quantization-pipeline median of `40.666 ms` (`37.408-43.346 ms`). The paired
speedup median was `76.484x`, and the padded-1080p output remained `765599`
bytes. Relative to the bounded-CPU-decision slice, the directional medians
improved by `1.02x` publicly and `1.07x` inside the pipeline. These remain
single-process checkpoints rather than the final retained claim.

### Resident AC-search inputs

**Status:** complete for fully-resident and throughput encoding.

Those two experimental paths now generate their initial quant field and masks
through a reusable prepared frontend. The same initial-quant submission also
materializes inverse-Gaborish opsin when the profile requires it. Checked
non-owning device views keep that search-domain image, the blurred pixel mask,
and the initial block field alive through AC candidate evaluation. The search
therefore omits its former full opsin and mask allocations/uploads, and its
Metal residual/cost kernels derive each strategy-aware quant norm directly
from the resident field. Only compact candidate descriptors and scalar costs
cross for the established CPU merge order and tie policy.

The diagnostic host copies remain part of the public pipeline output, and the
CPU still performs the existing initial-CfL/final-CfL policy work. Exact-
coefficient and automatic workflows retain their prior CPU initial-field and
ordinary GPU-search contracts; maximum-throughput continues to bypass AC
search entirely. Repeated prepared rate-control attempts reuse both the AQ and
resident-frontend allocations.

Direct Metal tests deliberately replace every candidate's host quant norm with
`1.0` and recover the CPU-oracle costs for all seven strategies, demonstrating
that the device field is authoritative. Prepared-view state/geometry checks,
flat and structured initial-quant fixtures, complete fully-resident and
throughput integration, and exact final frame/codestream parity on the focused
fixture also pass.

### Invariant resident evaluation metadata

**Status:** complete for fully-resident and throughput encoding.

Strategy records, anchor batches, and EPF sharpness already remain resident
from preparation or the rate-control attempt's `Reconfigure` call. The fixed
throughput CfL map is now bound once alongside that metadata after the initial
field adjustment. Every AQ evaluation omits both CfL planes and uploads only
the updated float quant field; device quantizer, raw-quant, and EPF-sigma
construction consume that field without another host metadata boundary.

The binding is explicit rather than pointer-cached. Host CfL input is rejected
while a resident binding is active, successful strategy reconfiguration marks
the old binding stale, and evaluation cannot resume until the new map is
bound. Direct tests poison the original host maps after binding, compare
repeated resident output exactly with an ordinary upload, verify zero
allocation/submission during binding, and measure the production input upload
as exactly one float per block.

One Apple M4 Pro Release process with one warmup and three alternating pairs
measured padded-1080p fully-resident encoding at `270.234 ms`
(`266.775-270.819 ms`) and `23.108x` paired speedup
(`23.015-23.325x`), with `622784` bytes. Throughput measured `238.075 ms`
(`236.711-243.464 ms`) and `26.597x` (`26.195-26.849x`), with `623449`
bytes. Relative to the preceding same-protocol checkpoints, the directional
Metal medians improved by `1.03x` and `1.08x`; these remain single-process
signals rather than retained multi-process claims.

### Chained resident Butteraugli policy

**Status:** complete for fully-resident and throughput Butteraugli encoding.

The prepared evaluator now exposes an optional bounded resident-policy
operation. The CPU computes the shared libjxl-derived field bounds and DC-quant
scalar, then Metal preserves the adjusted initial field and executes every
configured evaluation and dependent field update in one command buffer. A
five-float device score history covers the supported zero-to-four update
range. Only the final quant field, final block map, complete score history, and
requested final frame/RGB are read after one completion wait.

The Metal update matches the CPU policy's iteration-one pull toward the initial
field, early `pow(difference, 0.2)` rule, positive `lround`-equivalent progress
check, quantizer-scale increment, and bounds clamp. Device numeric failures are
sticky across passes, and caller outputs commit only after every final readback
validates. The high-level policy falls back only when a prepared backend
reports the fused capability as unavailable. Maximum-error control retains its
six-evaluation best-feasible policy, and exact-coefficient mode retains its
CPU update loop.

Focused tests cover every Butteraugli update count from zero through four,
serial-oracle tolerance, one dependent-evaluation submission, output padding,
final frame validity, and atomic upload/submission/completion/numeric/readback
failure. The initial adjustment and invariant CfL binding are intentionally
outside the fused command buffer.

### Selective codestream output materialization

**Status:** complete for all Metal codestream workflows.

The codestream workflow now uses an internal encoding-only pipeline output for
exact-coefficient, fully-resident, and throughput modes; maximum-throughput
retains its existing frame-only path. It requests the policy result and final
`VarDctEncoderFrame`, but leaves the final float quant field, block-distance
map, reconstructed linear RGB, and public initial-field diagnostics untouched.
Maximum-error control forwards its required final outcome through the same
lean output.

The fused resident finalizer reads the device error word and contiguous score
history, followed only by quantizer metadata, raw quant, and quantized DC/AC
needed to assemble the frame. The serial exact-coefficient and maximum-error
policies retain their bounded block/scalar readbacks, but their final
evaluation no longer downloads reconstructed linear RGB. Exact-coefficient
mode reuses its authoritative host frame and therefore performs no final
coefficient-frame download either.

All public diagnostic APIs preserve their prior output contract, and an
unavailable fused backend may still use the serial fallback. Focused tests
account for each readback byte class, cover zero through four resident updates,
exercise exact-coefficient and six-evaluation maximum-error encoding, preserve
padded or poisoned diagnostic destinations, and prove frame/codestream parity
plus atomic staging and device-failure paths.

On Apple M4 Pro, one warmup and five balanced padded-1080p samples at target
`1.2` measured fully resident at `274.965 ms` total (`257.452-299.968`) and
`230.141 ms` in the quantization pipeline. Throughput measured `241.725 ms`
total (`223.755-264.516`) and `198.266 ms` in the pipeline. Relative to the
preceding chained-policy checkpoint, the median reductions are `4.6%`/`3.4%`
for fully resident total/pipeline and `6.8%`/`8.1%` for throughput. These are
directional same-machine results; the ranges remain the primary latency
evidence.

A fresh Apple M4 Pro Release run with one warmup and five balanced
padded-1080p exact-coefficient samples measured `624.544 ms` total
(`614.039-641.636`), `362.392 ms` in the quantization pipeline, and `10.384x`
paired CPU/Metal speedup (`10.107-10.601x`). CPU and Metal produced the same
`636092`-byte codestream. This optimization removes a `24847212`-byte final RGB
transfer for that geometry, but there is no isolated same-revision pre-change
distribution, so the timing is a checkpoint rather than a claimed speedup.

### Final resident-frontend audit

**Status:** complete for the five-step sequence.

At this recorded audit checkpoint, exact-coefficient and automatic workflows
still used CPU initial quantization and the ordinary GPU search upload contract,
while maximum-throughput remained explicitly forced, DCT8-only, and scoreless.
Focused exact/automatic integration retained exact frame and codestream bytes.
The later codestream policy promotion made fully resident the automatic/default
Metal encoding path without changing these historical measurements. The
full-scope diagnostic benchmark was repaired
to pair an exact coefficient frame with its post-`AdjustQuantBlockAC` raw-quant
and EPF state; it now exercises the resident, exact-coefficient, and
perceptual-tail phases instead of failing setup.

A complete Release build succeeded. CTest passed `55/56`; the sole exception
is the inherited `quantization_pipeline` pinned Butteraugli golden mismatch
(`4.45247e-05` at score index 1), whose tolerance and expected value were not
changed. All Metal, AC-search, AQ, codestream, CLI, and install-consumer tests
passed. Installed `djxl` 0.12 independently decoded 17x13 outputs from
exact-coefficient, fully-resident, throughput, and maximum-throughput modes.

The retained padded-1080p maximum-throughput measurement uses three fresh
Release processes, each with one warmup and five alternating CPU/Metal pairs.
CPU process medians were `6219.973-6313.653 ms`, Metal process medians were
`78.077-81.855 ms`, and paired process-median speedups were
`77.496-80.666x`; individual pairs ranged from `69.529x` to `92.592x`.
Metal quantization-pipeline process medians were `42.781-45.619 ms`, and every
run produced the unchanged `765599`-byte codestream. Relative to the prior
post-adjustment checkpoint, this is a `1.39-1.52x` public-boundary and
`1.76-1.89x` pipeline improvement, depending on paired process endpoints.

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
