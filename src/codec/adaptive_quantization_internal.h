// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "codec/adaptive_quantization.h"

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

/// Runs the production AQ implementation and atomically returns diagnostics.
///
/// The profile has one evaluation entry per score. On failure, both the
/// caller-visible AQ output and the prior profile remain unchanged.
[[nodiscard]] Status FindBestQuantizationProfiled(
    ConstImage3FView original_linear_rgb, ConstImage3FView opsin,
    const AcStrategyGrid& strategies, ConstPlaneF32View initial_quant_field,
    ConstPlaneU8View epf_sharpness, AdaptiveQuantizationOptions options,
    AdaptiveQuantizationOutput output, AdaptiveQuantizationProfile* profile);

}  // namespace gjxl::adaptive_quantization_internal
