// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

namespace gjxl::ac_strategy_internal {

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
