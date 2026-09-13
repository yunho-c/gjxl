# DC quantization and smoothing implementation boundary

CPU and resident Metal integration are implemented and checked against the
pinned library, CPU/device oracles, and independent decoder; see
`dc-processing.md`. The completed bounded matched-quality study and its
limitations are in `dc-processing-qualification/REPORT.md`. This document
records the frame contract and device ordering.

## Preserve the frame contract

Keep `quantized_dc()` authoritative and `dc()` as its **unsmoothed**,
decoder-equivalent dequantization. Both `VarDctEncoderFrame::valid()` and
`AssembleVarDctEncoderFrame` currently enforce this relationship. Extra precision
changes dequantization by `2^-extra_dc_precision`; smoothing belongs at the
reconstruction consumer, before DC-to-low-frequency conversion. This avoids
making validation allocate/filter an image or retaining a stale filtered cache.

The CPU consumer is `ReconstructQuantizedCoefficients` in
`src/codec/reconstruction.cpp`. It reads `frame.dc()` and converts each
transform's DC block footprint into low-frequency coefficients. With
smoothing enabled, it now uses a temporary filtered DC image. Final frame
assembly and serialization do not need that temporary image.

## Quantization choices

Expose ordinary rounding and prediction-aware quantization separately from
lossless residual prediction. Carry the encoder-only quantization choice through
CPU adaptive-quantization and GPU evaluator options; retain only decoder-visible
precision and smoothing settings in the frame profile. The selected lossless
predictor can also select the prediction-aware quantization predictor, allowing
both libjxl's gradient and weighted cases to be tested explicitly.

Pinned libjxl `enc_modular.cc:1547` (`QuantizeWP`) and `QuantizeGradient` use an
extra precision bit, a 0.62 prediction-residual deadzone, and ordinary rounding
near zero; residual magnitudes above two use even-integer quantization. Chroma
uses reconstructed quantized Y. Predictor state resets per DC group and channel.
Preserve group boundaries rather than predicting across an entire frame.

The native weighted state should be shared by quantization and tokenization
through a codec-internal helper, avoiding a codec-to-serializer dependency.
Keep the existing public prediction header as a compatibility include if the
common enum moves to the codec layer. Public quantization failure must leave
both output planes unchanged, including range and managed-allocation failures.

## Smoothing semantics

Pinned `compressed_dc.cc:64` uses a 3x3 weighted neighborhood, one maximum
normalized gap across X/Y/B, and a shared attenuation factor. Use the pinned
floating-point association and fused operations explicitly. The normalization
uses the base quantizer's DC steps, even with extra DC precision. Border DC
samples stay unchanged, and dimensions at most two are identity cases. Test
against the pinned implementation, including channel coupling and border cases.

## Resident Metal ordering

Current quantization kernels extract DC, quantize it, and embed reconstructed
low frequencies in one transform dispatch. Prediction-aware quantization needs
all raw DC samples available before group traversal. Smoothing also needs
neighboring DC samples across transform and DC-group boundaries.

For the opt-in path: finish raw-DC extraction and AC quantization for all
transform batches; perform group-local DC quantization; optionally smooth into
a separate plane; replace the reconstructed low frequencies; then run inverse
transforms and ordinary perceptual evaluation. Keep the existing fused path for
default settings. Final-only encoding needs quantized DC and correct header
flags, without running a smoothing pass that has no reconstruction consumer.

Relevant paths are `metal_aq_reconstruction.cpp`,
`kernels/aq_reconstruction.metal`, `metal_aq_evaluation.cpp`, and the Metal
storage planners. Cover evaluation-free e3, iterative AQ, frame-only and exact
coefficient modes, owned/borrowed completed frames, and mixed transforms.
Account predictor scratch and filtered DC storage explicitly; never introduce
an unreported host round trip into fully resident execution.

## Serialization and validation

`WriteSimpleDcGroupModularHeader` writes the frame's two precision bits in
actual group writing and cost measurement. `WriteSimpleFrameHeader` writes
flags zero when smoothing is enabled, or the existing skip-smoothing flag
otherwise. Newly supported profile fields are normalized before the remaining
simple-profile gate. Default header bytes and maximum-storage bounds remain.

## Metal integration review checklist

- Preserve the default fused dispatch. For the opt-in path, all coefficient
  batches precede DC quantization/filtering and all inverse transforms follow
  LLF replacement. Apply the same ordering to ordinary and profiled execution.
- Final-only specialization must extract raw DC even without reconstruction
  storage. `EncodeFrameSubmission` has a separate shader/argument layout from
  `EncodeReconstructionCoefficientBatch(..., false)`; cover both.
- Exact-coefficient reconstruction uploads already quantized coefficients.
  Smoothing must run on their decoder dequantization before LLF embedding;
  it must not re-quantize the supplied integers.
- Replicate `Quantizer::Create`'s arithmetic association: DC steps are
  `(65536 / global_scale) / quant_dc / {4096,512,256}`, followed by the extra
  precision scale. The existing shader's `q / inverse_step` is not an exact
  substitute for `q * dc_step` on the new path.
- Weighted state resets per group/channel. CPU uses five two-row arrays.
  Metal stores five full error planes per group and runs diagonals in one
  threadgroup: weighted dependencies precede `x + 2*y`, gradient dependencies
  precede `x + y`. The original serial previous-row error mutation is recovered
  from the current row's west/two-west base errors. Each sample writes its own
  errors exactly once, with a device-memory barrier after every diagonal.
  All threads participate in barriers, including inactive rows of partial
  groups. This trades managed scratch storage for useful parallelism.
  The initial serial GPU prototype showed substantial pilot overhead; its
  partial evidence remains in `build/dc-processing-pilot-v1`.
- Add filtered DC and predictor scratch to the actual arena layout and to
  resident/compatibility workflow admission. Add dispatches/stage metadata to
  `metal_aq_profile_storage_plan` as well as runtime profiling.
- Update the completed-frame CPU metadata cache at
  `MetalPreparedAqEvaluation::FinishCompletedFrame` (currently the
  dequantization loop near `metal_aq_evaluation.cpp:3088`) for extra precision
  and separate-product CfL rounding, matching owned-frame assembly.
- Keep cache compatibility aware of mode/predictor. These fields already
  propagate into `AqEvaluationOptions`; execution now implements them and
  validates prediction-aware precision before allocation.

Validate each mode separately and together with pinned oracle arithmetic,
CPU/Metal reconstruction comparisons, external decoder output, deterministic
encodes, invalid-input/allocator atomicity, and bounded resource plans. Evaluate
rate at measured decoded quality, including encoder and decoder cost. Changes
to quantization/smoothing cannot reuse lossless-predictor byte savings as their
rate-quality evidence.
