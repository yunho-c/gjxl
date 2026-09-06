// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>

#include "core/ac_strategy.h"
#include "gpu/ops/profile_storage_plan.h"

namespace gjxl::metal_internal {

inline constexpr size_t kMaximumKernelIdBytes = 128;
inline constexpr char kAcStrategyProfileGroupId[] = "frontend.ac_strategy";
inline constexpr std::array<AcStrategyType, 7> kSupportedAqStrategies = {
    AcStrategyType::kDct8,     AcStrategyType::kDct16x16,
    AcStrategyType::kDct32x32, AcStrategyType::kDct16x8,
    AcStrategyType::kDct8x16,  AcStrategyType::kDct32x16,
    AcStrategyType::kDct16x32,
};

constexpr const char *AcStrategyProfileStageId(AcStrategyType strategy) {
  switch (strategy) {
  case AcStrategyType::kDct8:
    return "frontend.ac_strategy.dct8";
  case AcStrategyType::kDct16x8:
    return "frontend.ac_strategy.dct16x8";
  case AcStrategyType::kDct8x16:
    return "frontend.ac_strategy.dct8x16";
  case AcStrategyType::kDct16x16:
    return "frontend.ac_strategy.dct16";
  case AcStrategyType::kDct32x16:
    return "frontend.ac_strategy.dct32x16";
  case AcStrategyType::kDct16x32:
    return "frontend.ac_strategy.dct16x32";
  case AcStrategyType::kDct32x32:
    return "frontend.ac_strategy.dct32";
  default:
    return "frontend.ac_strategy.unsupported";
  }
}

struct AcSubmissionStorageOptions {
  size_t batches = 0; // Includes empty batches: validation reserves all.
  size_t nonempty_batches = 0; // Use batches as the preflight upper bound.
  bool profiling = false;
  size_t maximum_submission_id_length = sizeof(kAcStrategyProfileGroupId) - 1;
};

struct AcSubmissionStoragePlan {
  size_t batch_capacity = 0;
  size_t stage_capacity = 0;
  size_t maximum_dispatches = 0;
  resource_budget_internal::HostStorageBound input;
  gpu_profile_internal::SubmissionProfileStoragePlan profile;
  resource_budget_internal::HostStorageBound working;
  bool operator==(const AcSubmissionStoragePlan &) const = default;
};

/// Fresh validated-batch and synchronous callback-input arrays, plus optional
/// original/resolved profile graphs. Each nonempty batch dispatches at most
/// five kernels across all current DCT/fusion modes. Input arrays die before
/// profile resolution; working covers both phases. Device buffers, candidate
/// arrays, session aggregation and previous output graphs are separate. No
/// managed allocation on success; failures leave output unchanged.
[[nodiscard]] Status
ComputeAcSubmissionStoragePlan(const AcSubmissionStorageOptions &options,
                               AcSubmissionStoragePlan *out);

struct ResidentAqProfileInputOptions {
  size_t iterations = 0;
  bool evaluate_final_field = true;
  bool butteraugli_sinks = false;
  bool gaborish = false;
  size_t epf_iterations = 0;
};

struct ResidentAqProfileInputStoragePlan {
  size_t score_count = 0;
  size_t stage_capacity = 0;
  resource_budget_internal::HostStorageBound input;
  bool operator==(const ResidentAqProfileInputStoragePlan &) const = default;
};

/// The executing resident policy's exact reserve recipe, using the actual
/// private context type. Includes only its two callback-input arrays, not the
/// recorded/resolved graph, score output, evaluator or completed frame. Stage
/// capacity is NOT a bound on GPU dispatches or timestamp samples.
[[nodiscard]] Status ComputeResidentAqProfileInputStoragePlan(
    const ResidentAqProfileInputOptions &options,
    ResidentAqProfileInputStoragePlan *out);

} // namespace gjxl::metal_internal
