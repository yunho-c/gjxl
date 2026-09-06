// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/managed_allocator.h"
#include "codec/adaptive_quantization.h"
#include "codec/maximum_error.h"

namespace gjxl::adaptive_quantization_internal {

/// Stages measured inside one encode/reconstruct/measure evaluation.
enum class EvaluationStage : size_t {
  kFieldConstruction,
  kCoefficientCoding,
  kReconstruction,
  kLoopFilters,
  kColorConversion,
  kButteraugli,
  kBlockReduction,
  kCount,
};

inline constexpr size_t kEvaluationStageCount =
    static_cast<size_t>(EvaluationStage::kCount);

/// Nanosecond timings for one complete AQ evaluation.
struct EvaluationProfile {
  uint64_t total_nanoseconds = 0;
  std::array<uint64_t, kEvaluationStageCount> stage_nanoseconds{};

  [[nodiscard]] bool operator==(const EvaluationProfile&) const = default;
};

/// Timings for one complete invocation of the iterative AQ policy.
struct AdaptiveQuantizationProfile {
  uint64_t loop_setup_nanoseconds = 0;
  uint64_t quant_field_update_nanoseconds = 0;
  uint64_t output_commit_nanoseconds = 0;
  std::vector<EvaluationProfile> evaluations;

  [[nodiscard]] bool
  operator==(const AdaptiveQuantizationProfile&) const = default;
};

/// Bounded result returned by one adaptive-quantization evaluator.
struct AdaptiveQuantizationEvaluation {
  resource_budget_internal::ManagedVector<float> block_distance;
  Quantizer quantizer;
  double score = 0.0;
  MaximumErrorReduction maximum_error;
};

/// Supplies the expensive encode/reconstruct/measure portion of AQ while the
/// deterministic quant-field policy remains shared on the CPU.
class AdaptiveQuantizationEvaluator {
public:
  virtual ~AdaptiveQuantizationEvaluator() = default;

  AdaptiveQuantizationEvaluator(const AdaptiveQuantizationEvaluator&) = delete;
  AdaptiveQuantizationEvaluator& operator=(
    const AdaptiveQuantizationEvaluator&) = delete;

  [[nodiscard]] virtual Status Evaluate(
    ConstPlaneF32View quant_field,
    float quant_dc,
    bool is_final_evaluation,
    AdaptiveQuantizationEvaluation* evaluation,
    EvaluationProfile* profile) = 0;

protected:
  AdaptiveQuantizationEvaluator() = default;
};

/// Atomic scratch result of the shared bounded AQ policy.
struct AdaptiveQuantizationPolicyResult {
  resource_budget_internal::ManagedVector<float> quant_field;
  resource_budget_internal::ManagedVector<float> block_distance;
  std::vector<double> score_history;
  MaximumErrorResult maximum_error;
};

struct ButteraugliPolicySetup {
  float quant_dc = 0.0f;
  float lower_bound = 0.0f;
  float upper_bound = 0.0f;
};

/// Computes the libjxl-derived bounds and DC quantization shared by the CPU
/// policy loop and resident GPU implementations.
[[nodiscard]] Status PrepareButteraugliPolicy(
  ConstPlaneF32View adjusted_initial_quant_field,
  float butteraugli_target,
  ButteraugliPolicySetup* setup);

/// Validates the input and option contract shared by CPU and GPU evaluators.
[[nodiscard]] Status ValidateAdaptiveQuantizationPolicyInputs(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options);

/// Validates the same policy contract when the coding image is resident and
/// only its padded geometry is available on the host.
[[nodiscard]] Status ValidateResidentAdaptiveQuantizationPolicyInputs(
  ConstImage3FView original_linear_rgb,
  Extent2D opsin_extent,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options);

/// Runs initial adjustment, bounds, clamp, power, rounding-progress, and
/// iteration order identically for every evaluator.
[[nodiscard]] Status RunAdaptiveQuantizationPolicy(
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  AdaptiveQuantizationOptions options,
  AdaptiveQuantizationEvaluator& evaluator,
  AdaptiveQuantizationPolicyResult* result,
  AdaptiveQuantizationProfile* profile);

/// Runs the shared policy from a field whose strategy-aware initial adjustment
/// has already been applied. The remaining bounds, updates, and iteration
/// ordering are identical to `RunAdaptiveQuantizationPolicy`.
[[nodiscard]] Status RunAdaptiveQuantizationPolicyAdjusted(
  const AcStrategyGrid& strategies,
  ConstPlaneF32View adjusted_initial_quant_field,
  AdaptiveQuantizationOptions options,
  AdaptiveQuantizationEvaluator& evaluator,
  AdaptiveQuantizationPolicyResult* result,
  AdaptiveQuantizationProfile* profile);

/// Runs the production AQ implementation and atomically returns diagnostics.
///
/// The profile has one evaluation entry per score. On failure, both the
/// caller-visible AQ output and the prior profile remain unchanged.
[[nodiscard]] Status FindBestQuantizationProfiled(
    ConstImage3FView original_linear_rgb, ConstImage3FView opsin,
    const AcStrategyGrid& strategies, ConstPlaneF32View initial_quant_field,
    ConstPlaneU8View epf_sharpness, AdaptiveQuantizationOptions options,
    AdaptiveQuantizationOutput output, AdaptiveQuantizationProfile* profile);

/// Runs CPU AQ using an already prepared perceptual reference. Maximum-error
/// mode ignores the pointer. The reference must match the source and active
/// Butteraugli options.
[[nodiscard]] Status FindBestQuantizationPrepared(
    ConstImage3FView original_linear_rgb, ConstImage3FView opsin,
    const AcStrategyGrid& strategies, ConstPlaneF32View initial_quant_field,
    ConstPlaneU8View epf_sharpness, AdaptiveQuantizationOptions options,
    PreparedButteraugliReference* prepared_reference,
    AdaptiveQuantizationOutput output);

}  // namespace gjxl::adaptive_quantization_internal
