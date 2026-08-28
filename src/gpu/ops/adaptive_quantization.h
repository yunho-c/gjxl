// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <memory>
#include <vector>

#include "codec/adaptive_quantization.h"
#include "gpu/backend.h"
#include "gpu/ops/aq_evaluation.h"

namespace gjxl {

/// Selects where GPU adaptive-quantization evaluation begins.
enum class GpuAdaptiveQuantizationMode {
  /// CPU coefficient decisions are authoritative; the GPU starts at inverse
  /// reconstruction. This is the production and automatic-workflow default.
  kExactCoefficients,
  /// Forward transforms, coefficient coding, and reconstruction remain on the
  /// GPU. This is an explicit experimental mode and may change encoder
  /// decisions relative to the CPU reference.
  kFullyResident,
  /// Uses the fully resident evaluator but caps the complete pipeline at one
  /// host-synchronized AQ update instead of two. This is an explicit
  /// speed/size/quality trade and is never selected automatically.
  kThroughput,
  /// Uses a separate frame-only pipeline with fixed DCT8 strategies and no
  /// perceptual AQ evaluations. This maximum-throughput policy is never
  /// selected automatically and does not produce a score history.
  kMaximumThroughput,
};

struct GpuAdaptiveQuantizationPolicyOutput {
  PlaneF32View quant_field;
  PlaneF32View block_distance_map;
  std::vector<double>* score_history = nullptr;
};

struct GpuFrameOnlyQuantizationOutput {
  PlaneF32View quant_field;
  VarDctEncoderFrame* frame = nullptr;
};

/// Quantizes one explicitly selected field into an encoder frame without
/// inverse reconstruction or perceptual scoring. This is an experimental
/// speed/quality trade for callers that do not require AQ diagnostics.
[[nodiscard]] Status RunGpuFrameOnlyQuantization(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  const ColorCorrelationMap& color_correlation,
  AdaptiveQuantizationOptions options,
  GpuFrameOnlyQuantizationOutput output);

/// Frame-only variant that computes the fast initial chroma-from-luma map
/// from the already resident coding image in the encoding submission.
[[nodiscard]] Status RunGpuFrameOnlyQuantizationResidentInitialCfl(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuFrameOnlyQuantizationOutput output);

/// Maximum-throughput frontend that computes initial quantization and initial
/// CfL from the prepared resident coding image. Quantizer selection and raw
/// quantization remain in the same prepared Metal allocation.
[[nodiscard]] Status RunGpuFrameOnlyQuantizationResidentFrontend(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneU8View epf_sharpness,
  InitialQuantizationOptions initial_options,
  AdaptiveQuantizationOptions options,
  InitialQuantFieldOutput initial_output,
  GpuFrameOnlyQuantizationOutput output);

/// Runs the bounded adaptive-quantization policy with a prepared GPU evaluator.
///
/// The optional AQ capability is required; this operation never silently falls
/// back to CPU evaluation. Caller-visible output is committed only after every
/// requested evaluation succeeds.
[[nodiscard]] Status RunGpuAdaptiveQuantizationPolicy(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuAdaptiveQuantizationPolicyOutput output);

/// Runs the bounded policy with the explicitly selected GPU evaluation mode.
/// Resident modes are intended for error measurement and numerical research;
/// neither promises CPU-identical encoder decisions. `kThroughput` changes the
/// complete pipeline's policy iteration bound, not this direct operation.
/// `kMaximumThroughput` is unsupported by this direct operation.
[[nodiscard]] Status RunGpuAdaptiveQuantizationPolicy(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuAdaptiveQuantizationMode mode,
  GpuAdaptiveQuantizationPolicyOutput output);

/// Runs GPU adaptive quantization and materializes the final resident
/// reconstruction and encoder frame from the last evaluation. Exact CPU
/// quantized and dequantized reconstruction coefficients are transformed,
/// filtered, and evaluated by the prepared GPU operation.
///
/// Prepared AQ support is required; this operation never silently falls back
/// to CPU evaluation. All caller-visible outputs are committed atomically.
[[nodiscard]] Status RunGpuAdaptiveQuantization(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  AdaptiveQuantizationOutput output);

/// Runs full GPU adaptive quantization with an explicit evaluation mode.
/// Resident output is valid and atomic, but may differ from the CPU
/// quant field, frame, and codestream because coefficient ties are resolved in
/// GPU float arithmetic.
[[nodiscard]] Status RunGpuAdaptiveQuantization(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuAdaptiveQuantizationMode mode,
  AdaptiveQuantizationOutput output);

namespace adaptive_quantization_gpu_internal {

/// Reusable frame-level GPU AQ state for repeated rate-control attempts.
///
/// The state remembers the backend, source views, and evaluation options that
/// define the prepared allocation. Compatible calls only rebind the strategy
/// and EPF metadata; incompatible calls transparently prepare a new state.
struct PreparedAdaptiveQuantization {
  GpuBackend* backend = nullptr;
  ConstImage3FView original_linear_rgb;
  ConstImage3FView coding_opsin;
  AqEvaluationOptions evaluation_options;
  std::unique_ptr<PreparedAqEvaluation> evaluation;
  GpuBackend* resident_frontend_backend = nullptr;
  ConstImage3FView resident_frontend_original_linear_rgb;
  ConstImage3FView resident_frontend_coding_opsin;
  SimpleVarDctCodestreamProfile resident_frontend_profile;
  std::unique_ptr<PreparedAqEvaluation> resident_frontend;
};

[[nodiscard]] Status RunPreparedGpuAdaptiveQuantization(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuAdaptiveQuantizationMode mode,
  PreparedAdaptiveQuantization* prepared,
  AdaptiveQuantizationOutput output);

}  // namespace adaptive_quantization_gpu_internal

}  // namespace gjxl
