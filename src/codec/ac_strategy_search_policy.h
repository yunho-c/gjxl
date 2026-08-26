// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>

#include "core/ac_strategy.h"

namespace gjxl::ac_strategy_internal {

struct CandidateStage {
  AcStrategyType strategy;
  float entropy_multiplier;
  size_t anchor_step;
};

/// Candidate families used by the hierarchical search. DCT32-family anchors
/// begin on two-block steps; smaller families may begin on every base block.
inline constexpr std::array kCandidateStages = {
  CandidateStage{AcStrategyType::kDct8, 1.0f, 1},
  CandidateStage{AcStrategyType::kDct16x8, 1.21f, 1},
  CandidateStage{AcStrategyType::kDct8x16, 1.21f, 1},
  CandidateStage{AcStrategyType::kDct16x16, 1.34f, 1},
  CandidateStage{AcStrategyType::kDct32x16, 1.49f, 2},
  CandidateStage{AcStrategyType::kDct16x32, 1.49f, 2},
  CandidateStage{AcStrategyType::kDct32x32, 1.48f, 2},
};

[[nodiscard]] constexpr float CandidateEntropyMultiplier(
  AcStrategyType strategy) noexcept {

  for (const CandidateStage& stage : kCandidateStages) {
    if (stage.strategy == strategy) {
      return stage.entropy_multiplier;
    }
  }
  return 0.0f;
}

[[nodiscard]] constexpr size_t CandidateAnchorStep(
  AcStrategyType strategy) noexcept {

  for (const CandidateStage& stage : kCandidateStages) {
    if (stage.strategy == strategy) {
      return stage.anchor_step;
    }
  }
  return 0;
}

static_assert(
  CandidateEntropyMultiplier(AcStrategyType::kDct16x8) ==
  CandidateEntropyMultiplier(AcStrategyType::kDct8x16));
static_assert(
  CandidateEntropyMultiplier(AcStrategyType::kDct32x16) ==
  CandidateEntropyMultiplier(AcStrategyType::kDct16x32));
static_assert(CandidateAnchorStep(AcStrategyType::kDct32x32) == 2);

enum class FirstLevelDivision {
  kSquare,
  kVertical,
  kHorizontal,
};

/// Preserves libjxl's strict comparisons and horizontal tie fallback.
[[nodiscard]] constexpr FirstLevelDivision ChooseFirstLevelDivision(
  float square_cost,
  float vertical_cost,
  float horizontal_cost) noexcept {

  if (square_cost < vertical_cost && square_cost < horizontal_cost) {
    return FirstLevelDivision::kSquare;
  }
  if (vertical_cost < horizontal_cost) {
    return FirstLevelDivision::kVertical;
  }
  return FirstLevelDivision::kHorizontal;
}

}  // namespace gjxl::ac_strategy_internal
