// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "gpu/metal/metal_storage_plan.h"
#include "gpu/metal/metal_submission_storage_plan.h"

namespace gjxl::metal_internal {

[[nodiscard]] constexpr const char *
AqReconstructionProfileStageId(AcStrategyType strategy) noexcept {
  switch (strategy) {
  case AcStrategyType::kDct8:
    return "aq.reconstruction.dct8";
  case AcStrategyType::kDct16x8:
    return "aq.reconstruction.dct16x8";
  case AcStrategyType::kDct8x16:
    return "aq.reconstruction.dct8x16";
  case AcStrategyType::kDct16x16:
    return "aq.reconstruction.dct16";
  case AcStrategyType::kDct32x16:
    return "aq.reconstruction.dct32x16";
  case AcStrategyType::kDct16x32:
    return "aq.reconstruction.dct16x32";
  case AcStrategyType::kDct32x32:
    return "aq.reconstruction.dct32";
  default:
    return "aq.reconstruction.unsupported";
  }
}

[[nodiscard]] constexpr const char *
AqReconstructionCoefficientProfileStageId(AcStrategyType strategy) noexcept {
  switch (strategy) {
  case AcStrategyType::kDct8:
    return "aq.reconstruction.coefficients.dct8";
  case AcStrategyType::kDct16x8:
    return "aq.reconstruction.coefficients.dct16x8";
  case AcStrategyType::kDct8x16:
    return "aq.reconstruction.coefficients.dct8x16";
  case AcStrategyType::kDct16x16:
    return "aq.reconstruction.coefficients.dct16";
  case AcStrategyType::kDct32x16:
    return "aq.reconstruction.coefficients.dct32x16";
  case AcStrategyType::kDct16x32:
    return "aq.reconstruction.coefficients.dct16x32";
  case AcStrategyType::kDct32x32:
    return "aq.reconstruction.coefficients.dct32";
  default:
    return "aq.reconstruction.coefficients.unsupported";
  }
}

[[nodiscard]] constexpr const char *
AqReconstructionScatterProfileStageId(AcStrategyType strategy) noexcept {
  switch (strategy) {
  case AcStrategyType::kDct8:
    return "aq.reconstruction.scatter.dct8";
  case AcStrategyType::kDct16x8:
    return "aq.reconstruction.scatter.dct16x8";
  case AcStrategyType::kDct8x16:
    return "aq.reconstruction.scatter.dct8x16";
  case AcStrategyType::kDct16x16:
    return "aq.reconstruction.scatter.dct16";
  case AcStrategyType::kDct32x16:
    return "aq.reconstruction.scatter.dct32x16";
  case AcStrategyType::kDct16x32:
    return "aq.reconstruction.scatter.dct16x32";
  case AcStrategyType::kDct32x32:
    return "aq.reconstruction.scatter.dct32";
  default:
    return "aq.reconstruction.scatter.unsupported";
  }
}

[[nodiscard]] constexpr const char *
AqForwardCoefficientProfileStageId(AcStrategyType strategy) noexcept {
  switch (strategy) {
  case AcStrategyType::kDct8:
    return "aq.reconstruction.forward.dct8";
  case AcStrategyType::kDct16x8:
    return "aq.reconstruction.forward.dct16x8";
  case AcStrategyType::kDct8x16:
    return "aq.reconstruction.forward.dct8x16";
  case AcStrategyType::kDct16x16:
    return "aq.reconstruction.forward.dct16";
  case AcStrategyType::kDct32x16:
    return "aq.reconstruction.forward.dct32x16";
  case AcStrategyType::kDct16x32:
    return "aq.reconstruction.forward.dct16x32";
  case AcStrategyType::kDct32x32:
    return "aq.reconstruction.forward.dct32";
  default:
    return "aq.reconstruction.forward.unsupported";
  }
}

[[nodiscard]] constexpr const char *
AqFinalFrameProfileStageId(AcStrategyType strategy) noexcept {
  switch (strategy) {
  case AcStrategyType::kDct8:
    return "aq.final_frame.dct8";
  case AcStrategyType::kDct16x8:
    return "aq.final_frame.dct16x8";
  case AcStrategyType::kDct8x16:
    return "aq.final_frame.dct8x16";
  case AcStrategyType::kDct16x16:
    return "aq.final_frame.dct16";
  case AcStrategyType::kDct32x16:
    return "aq.final_frame.dct32x16";
  case AcStrategyType::kDct16x32:
    return "aq.final_frame.dct16x32";
  case AcStrategyType::kDct32x32:
    return "aq.final_frame.dct32";
  default:
    return "aq.final_frame.unsupported";
  }
}

inline constexpr std::array<size_t, 6> kButteraugliMaltaAccumulationOrder{
    4, 5, 2, 3, 0, 1};

constexpr size_t NextButteraugliReductionCount(size_t count) {
  return count / kButteraugliReductionWidth +
         static_cast<size_t>(count % kButteraugliReductionWidth != 0);
}
constexpr size_t ButteraugliReductionDispatchCount(size_t count) {
  if (count == 0)
    return 0;
  size_t passes = 0;
  do {
    count = NextButteraugliReductionCount(count);
    ++passes;
  } while (count != 1);
  return passes;
}

struct InitialQuantSortPlan {
  size_t count = 0;
  size_t dispatches = 0; // Both sorts, not prepare/median/finalization.
  bool operator==(const InitialQuantSortPlan &) const = default;
};
[[nodiscard]] Status ComputeInitialQuantSortPlan(size_t blocks,
                                                 InitialQuantSortPlan *out);

struct ButteraugliDispatchPlan {
  bool expanded = false;
  bool multiscale = false;
  size_t reference = 0;
  size_t comparison = 0;
  size_t resident_comparison = 0; // Zero when resident sinks are unavailable.
  bool operator==(const ButteraugliDispatchPlan &) const = default;
};

/// Production prepared comparisons only: no ForTesting stage capture. Includes
/// the two-dispatch Malta hardware fallback and geometry-dependent reduction.
/// anchor_count is used only for resident sinks; zero requests no sink bound.
[[nodiscard]] Status
ComputeButteraugliDispatchPlan(Extent2D source, size_t anchor_count,
                               size_t nonempty_families,
                               ButteraugliDispatchPlan *out);

enum class AqProfileFrameOutput { kNone, kOwned, kCompleted };

struct ResidentAqProfileStoragePlan {
  size_t maximum_dispatches = 0;
  size_t maximum_id_length = 0; // Stage/group/wall/submission, not kernel IDs.
  ResidentAqProfileInputStoragePlan metadata;
  gpu_profile_internal::SubmissionProfileStoragePlan graph;
  resource_budget_internal::HostStorageBound wall;
  resource_budget_internal::HostStorageBound working;
  bool operator==(const ResidentAqProfileStoragePlan &) const = default;
};

/// Policy-derived input arrays, original/resolved graph and output wall stages.
/// Bounds every first/cached run using all seven families and block-count
/// anchors; actual final selection is not needed before admission. The sink
/// flag must match source geometry. Evaluator, frames, score histories, caller
/// inputs, old output graphs and parent-session aggregation remain separate.
[[nodiscard]] Status
ComputeResidentAqProfileStoragePlan(Extent2D source, Extent2D coding,
                                    const ResidentAqProfileInputOptions &policy,
                                    AqProfileFrameOutput frame,
                                    ResidentAqProfileStoragePlan *out);

struct AqAuxiliaryProfileOptions {
  Extent2D source;
  Extent2D coding;
  bool resident_ac_strategy_inputs = false;
  bool resident_initial_cfl = false;
  bool frame_only_resident_quantizer = false;
  bool gaborish = false;
};

struct AqAuxiliaryProfileStoragePlan {
  size_t reference_dispatches = 0;
  size_t initial_dispatches = 0;
  size_t adjustment_dispatches = 0;
  InitialQuantSortPlan sort;
  gpu_profile_internal::SubmissionProfileStoragePlan reference;
  gpu_profile_internal::SubmissionProfileStoragePlan initial;
  gpu_profile_internal::SubmissionProfileStoragePlan adjustment;
  bool operator==(const AqAuxiliaryProfileStoragePlan &) const = default;
};

/// Three independent, optional submissions. Caller selects/composes the phases
/// used by its workflow; this does not validate a full evaluator preparation.
/// Their callback inputs are stack objects (no backing). Returned graphs, old
/// graphs and session aggregation can coexist and must be composed separately.
/// All planning is allocation-free on success and preserves output on failure.
[[nodiscard]] Status
ComputeAqAuxiliaryProfileStoragePlan(const AqAuxiliaryProfileOptions &options,
                                     AqAuxiliaryProfileStoragePlan *out);

} // namespace gjxl::metal_internal
