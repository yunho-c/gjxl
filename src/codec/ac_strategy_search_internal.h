// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <span>

#include "codec/ac_strategy.h"

namespace gjxl::ac_strategy_internal {

/// Dense per-anchor candidate costs, indexed by global base-block position.
/// Empty strategy spans are allowed only for strategies the search never uses.
struct CandidateCostTableView {
  Extent2D block_extent;
  std::array<std::span<const float>, kAcStrategyCount> strategy_costs{};
};

/// Runs the normal hierarchical search while sourcing leaf costs from a table.
/// Merge order, priority checks, and tie-breaking remain CPU-defined.
[[nodiscard]] Status FindAcStrategyGridFromCandidateCosts(
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  AcStrategySearchOptions options,
  const CandidateCostTableView& candidate_costs,
  AcStrategyGrid* out);

/// Resident variant whose CPU merge needs only coding geometry because all
/// leaf costs have already been evaluated from device images. pixel_mask may
/// be entirely empty; the CPU merge does not read it.
[[nodiscard]] Status FindAcStrategyGridFromResidentCandidateCosts(
  Extent2D opsin_extent,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  AcStrategySearchOptions options,
  const CandidateCostTableView& candidate_costs,
  AcStrategyGrid* out);

}  // namespace gjxl::ac_strategy_internal
