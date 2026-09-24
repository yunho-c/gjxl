// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "codec/quantization_pipeline.h"
#include "codestream/encoder_internal.h"
#include "codestream/encoding_result_internal.h"
#include "codestream/workflow.h"
#include "gpu/backend.h"
#include "gpu/ops/gpu_execution_profile_internal.h"
#include "gpu/ops/resident_input.h"

namespace gjxl::codestream_internal {

/// Distance and EPF-iteration gates are evaluated per encoding attempt.
/// Storage preparation uses this target-independent capability decision so
/// a target-size retry may cross distance 0.5 without exceeding admission.
[[nodiscard]] constexpr bool UseEpfSharpnessSearch(
    const VarDctEncodingOptions& options) noexcept {
  return options.adaptive_epf_sharpness && options.effort >= 6 &&
         options.backend != VarDctBackendPreference::kCuda &&
         options.rate_control_mode != VarDctRateControlMode::kMaximumError &&
         options.gpu_aq_mode != GpuAdaptiveQuantizationMode::kMaximumThroughput;
}

/// Scope for modular DC mapping search. Keep explicit frontend and
/// entropy modes, ordinary DC rounding, and other efforts on their own policy.
[[nodiscard]] constexpr bool UseDcUintSearch(
  const VarDctEncodingOptions& options) noexcept {
  return options.effort == 4 &&
    options.density_mode == VarDctDensityMode::kDefault &&
    options.compression_mode == VarDctCompressionMode::kAutomatic &&
    options.rate_control_mode != VarDctRateControlMode::kMaximumError &&
    ResolveDcQuantization(options) == DcQuantizationMode::kPredictionAware;
}

/// Ordinary efforts 8-10 share the eight-step nonlinear final CfL recipe.
/// Zero denotes the established fast regression. This changes no host/device
/// arena dimensions; it adds only bounded threadgroup-local shader scratch.
[[nodiscard]] constexpr uint32_t FinalColorCorrelationIterations(
  const VarDctEncodingOptions& options, bool selected_metal) noexcept {
  return selected_metal && options.effort >= 8 &&
      options.gpu_aq_mode == GpuAdaptiveQuantizationMode::kFullyResident &&
      options.density_mode == VarDctDensityMode::kDefault &&
      options.rate_control_mode != VarDctRateControlMode::kMaximumError
    ? 8u : 0u;
}

/// Shared policy resolution for execution and whole-workflow storage planning.
/// The caller validates effort and density mode before using this recipe.
[[nodiscard]] constexpr size_t AdaptiveQuantizationIterations(
  const VarDctEncodingOptions& options) noexcept {
  if (options.density_mode == VarDctDensityMode::kHighDensity) return 4;
  if (options.effort <= 3) return 0;
  // Ordinary e4 skips perceptual refinement; explicit error/density recipes
  // retain their existing update policy.
  if (options.effort == 4 &&
      options.density_mode == VarDctDensityMode::kDefault &&
      options.rate_control_mode != VarDctRateControlMode::kMaximumError) return 0;
  if (options.effort <= 6) return 1;
  if (options.effort == 7) return 2;
  if (options.effort == 8) return 3;
  return 4;
}

/// Ordinary e10 searches every DCT32-family anchor. Explicit density/error
/// recipes retain their existing placement policy, independently of writer mode.
[[nodiscard]] constexpr bool UseDenseDct32Search(
  const VarDctEncodingOptions& options) noexcept {
  return options.effort == 10 &&
    options.density_mode == VarDctDensityMode::kDefault &&
    options.rate_control_mode != VarDctRateControlMode::kMaximumError;
}

/// Shared transform policy for execution and CPU/Metal storage admission.
/// Explicit high-density and maximum-error modes preserve their search policy.
[[nodiscard]] constexpr bool UseFixedDct8Strategy(
  const VarDctEncodingOptions& options) noexcept {
  return options.effort <= 4 &&
    options.density_mode == VarDctDensityMode::kDefault &&
    options.rate_control_mode != VarDctRateControlMode::kMaximumError;
}

/// Default low efforts match libjxl's e1-4 Gaborish/initial-field policy.
/// Explicit transform/error modes retain their existing perceptual recipe.
[[nodiscard]] constexpr bool UseUniformInitialQuantization(
  const VarDctEncodingOptions& options) noexcept {
  return UseFixedDct8Strategy(options) &&
    options.gpu_aq_mode != GpuAdaptiveQuantizationMode::kMaximumThroughput;
}

inline void ConfigureInitialQuantizationPolicy(
  const VarDctEncodingOptions& options,
  CpuQuantizationPipelineOptions* pipeline) noexcept {
  pipeline->uniform_initial_quantization = UseUniformInitialQuantization(options);
  pipeline->adaptive_quantization.search_epf_sharpness = UseEpfSharpnessSearch(options);
  pipeline->adaptive_quantization.profile.loop_filter.gaborish =
    !pipeline->uniform_initial_quantization;
}

/// Internal complete-encode result. Candidate bytes stay charged until the
/// outer C/C++ or batch adapter explicitly publishes them.
[[nodiscard]] Status EncodeLinearRgbVarDctCodestreamOwned(
  ConstImage3FView linear_rgb, VarDctEncodingOptions options,
  CodestreamBuffer* codestream, OwnedEncodingSummary* summary = nullptr,
  OwnedEncodingTiming* timing = nullptr,
  // Optional unstarted scope, entered after admission and kept alive by the
  // adapter through its result retention/cache-trim epilogue.
  thread_budget_internal::CpuExecutionScope* outer_cpu_execution = nullptr);

/// Internal synchronous packed-input handoff. The view borrows from owner;
/// moving the complete object transfers both to the encoding workflow.
struct ResidentEncodingInput {
  ConstImage3FView linear_rgb;
  std::unique_ptr<PreparedResidentInput> owner;
};

/// Called after C-adapter memory/CPU admission. Requires forced resident GPU
/// encoding; failure leaves the output unchanged. The generator is invoked
/// synchronously and is not retained.
[[nodiscard]] Status
PrepareResidentEncodingInput(Extent2D source, const VarDctEncodingOptions& options,
                             Status (*fill)(const void*, Image3FView),
                             const void* context, ResidentEncodingInput* input);

/// Consumes the prepared input. Its owner is released at the workflow's normal
/// last-use boundary, after all borrowing GPU work has completed.
[[nodiscard]] Status
EncodeResidentLinearRgbVarDctCodestreamOwned(ResidentEncodingInput input,
                                             VarDctEncodingOptions options,
                                             CodestreamBuffer* codestream);

struct QuantizationMatrixScaleStats {
  float x_edge = 0.0f;
  float b_edge = 0.0f;
  float exposed_blue = 0.0f;

  bool operator==(const QuantizationMatrixScaleStats&) const = default;
};

struct QuantizationMatrixScales {
  uint8_t x = 2;
  uint8_t b = 2;

  bool operator==(const QuantizationMatrixScales&) const = default;
};

/// Returns whether this encoding policy uses source-dependent matrix-scale
/// statistics. High density follows libjxl's effort-9-like behavior even when
/// its explicit effort value is lower. Maximum-error control never uses the
/// statistics because its matrix scales are fixed at 2/2.
[[nodiscard]] bool ShouldComputeQuantizationMatrixScaleStats(
  const VarDctEncodingOptions& options) noexcept;

/// Resolves the public compression request to one serializer behavior.
[[nodiscard]] VarDctEntropyBehavior ResolveEntropyBehavior(
  const VarDctEncodingOptions& options) noexcept;

/// Resolves the public effort request independently from entropy intensity.
[[nodiscard]] VarDctCoefficientOrderBehavior
ResolveCoefficientOrderBehavior(
  const VarDctEncodingOptions& options) noexcept;

/// Computes libjxl's source-dependent X/B matrix-scale statistics over the
/// unpadded opsin image. Failure leaves `stats` unchanged.
[[nodiscard]] Status ComputeQuantizationMatrixScaleStats(
  ConstImage3FView opsin,
  QuantizationMatrixScaleStats* stats);

/// Computes matrix-scale statistics without revalidating an Opsin view whose
/// complete finite-value provenance is owned by the synchronous workflow.
/// Passing any other view violates this internal contract. Failure leaves
/// `stats` unchanged.
[[nodiscard]] Status ComputeQuantizationMatrixScaleStatsFromFiniteOpsin(
  ConstImage3FView opsin,
  QuantizationMatrixScaleStats* stats);

/// Selects libjxl's X/B matrix scales for one complete encode attempt.
/// Maximum-error control always selects 2/2 and ignores the Butteraugli
/// target. Failure leaves `scales` unchanged.
[[nodiscard]] Status SelectQuantizationMatrixScales(
  const QuantizationMatrixScaleStats& stats,
  VarDctRateControlMode mode,
  float butteraugli_target,
  QuantizationMatrixScales* scales);

struct VarDctEncodingProfile {
  /// Maximum CPU threads simultaneously participating in this encode.
  size_t peak_cpu_participants = 0;
  /// Largest authoritative AC owner at a serializer handoff across attempts,
  /// excluding other metadata/device scratch. Width belongs to that owner.
  size_t ac_coefficient_bytes = 0;
  size_t ac_storage_bytes = 0;
  uint64_t input_preparation_nanoseconds = 0;
  uint64_t input_geometry_and_storage_nanoseconds = 0;
  uint64_t input_color_transform_nanoseconds = 0;
  uint64_t input_matrix_scale_stats_nanoseconds = 0;
  uint64_t input_resident_preparation_nanoseconds = 0;
  uint64_t input_quantization_preparation_nanoseconds = 0;
  uint64_t backend_selection_nanoseconds = 0;
  uint64_t quantization_pipeline_nanoseconds = 0;
  uint64_t codestream_encoding_nanoseconds = 0;
  uint64_t summary_assembly_nanoseconds = 0;
  uint64_t total_nanoseconds = 0;
  VarDctExecutionBackend execution_backend = VarDctExecutionBackend::kCpu;
  VarDctCodestreamProfile codestream;

  bool operator==(const VarDctEncodingProfile&) const = default;
};

[[nodiscard]] bool IsAutomaticMetalGeometryEligible(
  Extent2D padded_extent) noexcept;

[[nodiscard]] bool IsAutomaticMetalTargetEligible(
  float butteraugli_target) noexcept;

[[nodiscard]] bool IsAutomaticMetalBackendQualified(
  const GpuBackend& backend) noexcept;

/// Initializes and validates the process-cached production Metal backend.
/// Used by frontends that promise eager failure for an explicitly forced
/// Metal execution policy.
[[nodiscard]] Status EnsureProductionMetalBackendAvailable();

/// Initializes and validates the process-cached production CUDA backend.
/// Used by frontends that promise eager failure for an explicitly forced
/// CUDA execution policy.
[[nodiscard]] Status EnsureProductionCudaBackendAvailable();

[[nodiscard]] Status EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  GpuBackend* backend,
  bool backend_is_qualified_for_automatic,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary = nullptr);

/// Diagnostic-only public-workflow entry point. On failure, `codestream`,
/// `summary`, and `profile` remain unchanged.
[[nodiscard]] Status
EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  GpuBackend* backend,
  bool backend_is_qualified_for_automatic,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary,
  VarDctEncodingProfile* profile);

/// Diagnostic-only public-workflow entry point with resident GPU
/// timestamps. On failure, every caller-visible output remains unchanged.
[[nodiscard]] Status
EncodeLinearRgbVarDctCodestreamGpuProfiledWithBackendForTesting(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  GpuBackend* backend,
  bool backend_is_qualified_for_automatic,
  gpu_profile_internal::GpuProfilingMode profiling_mode,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary,
  VarDctEncodingProfile* profile,
  gpu_profile_internal::GpuExecutionProfile* gpu_profile);

}  // namespace gjxl::codestream_internal
