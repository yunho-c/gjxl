// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "codec/adaptive_quantization.h"
#include "gpu/backend.h"
#include "gpu/ops/ac_strategy_search.h"
#include "gpu/ops/aq_evaluation.h"

namespace gjxl {

/// Selects where GPU adaptive-quantization evaluation begins.
enum class GpuAdaptiveQuantizationMode {
  /// CPU coefficient decisions are authoritative; the GPU starts at inverse
  /// reconstruction. Encoding workflows retain this as an explicit reference
  /// and compatibility mode; source-compatible diagnostic overloads use it by
  /// default.
  kExactCoefficients,
  /// Forward transforms, coefficient coding, and reconstruction remain on the
  /// GPU. This is the default Metal encoding mode and may change encoder
  /// decisions relative to the CPU reference.
  kFullyResident,
  /// Encoding-only workflows apply both default AQ updates, then quantize the
  /// final field directly into the frame without reconstructing and scoring
  /// it a third time. Diagnostic workflows retain the original one-update
  /// speed/size/quality trade. This mode is never selected automatically.
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
/// Resident modes do not promise CPU-identical encoder decisions.
/// `kThroughput` changes the complete diagnostic pipeline's policy iteration
/// bound, not this direct operation.
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

/// Selects optional diagnostic results for an internal full-output call.
/// The encoder always requests the frame and score history, while public APIs
/// retain the default of materializing every diagnostic. An explicit
/// throughput encode may omit the unevaluated final field's perceptual result.
struct AdaptiveQuantizationMaterialization {
  bool quant_field = true;
  bool block_distance_map = true;
  bool reconstructed_linear_rgb = true;
  bool final_perceptual_evaluation = true;
};

/// Reusable frame-level GPU AQ state for repeated rate-control attempts.
///
/// The state remembers the quantization-pipeline generation, backend, source
/// views, and evaluation options that define the prepared allocation.
/// Compatible calls only rebind the strategy and EPF metadata; a new borrowed
/// source generation transparently invalidates source-dependent GPU state.
struct PreparedAdaptiveQuantization {
  PreparedAcStrategySearch ac_strategy_search;
  uint64_t quantization_pipeline_generation = 0;
  ConstDeviceImage3View resident_coding_opsin;
  GpuBackend* backend = nullptr;
  ConstImage3FView original_linear_rgb;
  ConstImage3FView coding_opsin;
  AqEvaluationOptions evaluation_options;
  bool resident_quantization = false;
  bool frame_only_resident_frontend = false;
  std::unique_ptr<PreparedAqEvaluation> evaluation;
};

/// Reuses a compatible maximum-throughput evaluator across rate-control
/// attempts. A failed prepared operation is discarded before returning.
[[nodiscard]] Status RunPreparedGpuFrameOnlyQuantizationResidentFrontend(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneU8View epf_sharpness,
  InitialQuantizationOptions initial_options,
  AdaptiveQuantizationOptions options,
  PreparedAdaptiveQuantization* prepared,
  InitialQuantFieldOutput initial_output,
  GpuFrameOnlyQuantizationOutput output);

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
  AdaptiveQuantizationOutput output,
  AdaptiveQuantizationMaterialization materialization = {});

}  // namespace adaptive_quantization_gpu_internal

}  // namespace gjxl
