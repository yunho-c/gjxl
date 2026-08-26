// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "core/ac_strategy.h"

namespace {

bool CheckCreation() {
  gjxl::AcStrategyGrid grid;
  if (!gjxl::AcStrategyGrid::Create({5, 4}, &grid).ok() ||
      !grid.valid() ||
      grid.complete() ||
      grid.extent() != gjxl::Extent2D{5, 4}) {
    std::cerr << "AC-strategy grid creation failed\n";
    return false;
  }

  gjxl::AcStrategyGrid unused;
  if (gjxl::AcStrategyGrid::Create({0, 4}, &unused).ok() ||
      gjxl::AcStrategyGrid::Create(
        {std::numeric_limits<size_t>::max(), 2},
        &unused).ok() ||
      gjxl::AcStrategyGrid::Create({1, 1}, nullptr).ok()) {
    std::cerr << "Invalid AC-strategy grid creation was accepted\n";
    return false;
  }

  return true;
}

bool CheckPlacementAndCells() {
  gjxl::AcStrategyGrid grid;
  if (!gjxl::AcStrategyGrid::Create({6, 4}, &grid).ok() ||
      !grid.Set(0, 0, gjxl::AcStrategyType::kDct16x16).ok() ||
      !grid.Set(2, 0, gjxl::AcStrategyType::kDct16x32).ok() ||
      !grid.Set(0, 2, gjxl::AcStrategyType::kDct8x16).ok() ||
      !grid.Set(0, 3, gjxl::AcStrategyType::kDct8x16).ok()) {
    std::cerr << "Valid AC-strategy placement failed\n";
    return false;
  }

  gjxl::AcStrategyCell anchor;
  gjxl::AcStrategyCell covered;
  if (!grid.Get(2, 0, &anchor).ok() ||
      anchor.strategy != gjxl::AcStrategyType::kDct16x32 ||
      !anchor.is_anchor ||
      !grid.Get(5, 1, &covered).ok() ||
      covered.strategy != gjxl::AcStrategyType::kDct16x32 ||
      covered.is_anchor ||
      !grid.occupied(5, 1) ||
      grid.occupied(5, 3)) {
    std::cerr << "AC-strategy cell encoding is incorrect\n";
    return false;
  }

  if (grid.Set(1, 1, gjxl::AcStrategyType::kDct8).ok() ||
      grid.Set(3, 3, gjxl::AcStrategyType::kDct16x32).ok() ||
      grid.Set(
        5,
        3,
        static_cast<gjxl::AcStrategyType>(255)).ok() ||
      grid.Get(5, 3, &covered).ok() ||
      grid.Get(0, 0, nullptr).ok()) {
    std::cerr << "Invalid AC-strategy operation was accepted\n";
    return false;
  }

  grid.fill_empty_dct8();
  size_t anchors = 0;
  if (!grid.complete() ||
      !grid.ForEachAnchor(
        [&](size_t, size_t, gjxl::AcStrategyType) {
          ++anchors;
          return gjxl::Status::Ok();
        }).ok() ||
      anchors != 12) {
    std::cerr << "Multiblock AC-strategy iteration is incorrect\n";
    return false;
  }

  return true;
}

bool CheckCompletionAndIteration() {
  gjxl::AcStrategyGrid grid;
  if (!gjxl::AcStrategyGrid::Create({4, 4}, &grid).ok()) {
    return false;
  }

  size_t unexpected_calls = 0;
  if (grid.ForEachAnchor(
        [&](size_t, size_t, gjxl::AcStrategyType) {
          ++unexpected_calls;
          return gjxl::Status::Ok();
        }).ok() ||
      unexpected_calls != 0) {
    std::cerr << "Incomplete strategy grid was iterable\n";
    return false;
  }

  grid.fill_dct8();
  if (!grid.complete()) {
    return false;
  }

  size_t anchors = 0;
  bool order_is_row_major = true;
  size_t previous_index = 0;
  const gjxl::Status status = grid.ForEachAnchor(
    [&](size_t x, size_t y, gjxl::AcStrategyType strategy) {
      const size_t index = y * grid.extent().width + x;
      if (strategy != gjxl::AcStrategyType::kDct8 ||
          (anchors != 0 && index <= previous_index)) {
        order_is_row_major = false;
      }
      previous_index = index;
      ++anchors;
      return gjxl::Status::Ok();
    });
  if (!status.ok() || !order_is_row_major || anchors != 16) {
    std::cerr << "AC-strategy anchor iteration is incorrect\n";
    return false;
  }

  grid.clear();
  if (grid.complete() || grid.occupied(0, 0)) {
    std::cerr << "AC-strategy grid clear failed\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!CheckCreation() ||
      !CheckPlacementAndCells() ||
      !CheckCompletionAndIteration()) {
    return EXIT_FAILURE;
  }

  std::cout << "All AC-strategy grid tests passed.\n";
  return EXIT_SUCCESS;
}
