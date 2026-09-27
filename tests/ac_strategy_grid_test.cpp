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

std::vector<uint8_t> Encode(const gjxl::AcStrategyGrid& grid) {
  const auto extent = grid.extent();
  std::vector<uint8_t> cells(extent.width * extent.height);
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < extent.width; ++x) {
      gjxl::AcStrategyCell cell;
      if (!grid.Get(x, y, &cell).ok()) return {};
      cells[y * extent.width + x] = (uint8_t(cell.strategy) << 1) | cell.is_anchor;
    }
  return cells;
}

// Independent pre-optimization import: checked placement, complete cover, then
// exact comparison of every ownership byte.
bool ImportOracle(gjxl::Extent2D extent, std::span<const uint8_t> cells) {
  gjxl::AcStrategyGrid grid;
  if (!gjxl::AcStrategyGrid::Create(extent, &grid).ok()) return false;
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < extent.width; ++x) {
      const auto cell = cells[y * extent.width + x];
      if ((cell & 1u) &&
          !grid.Set(x, y, gjxl::AcStrategyType(cell >> 1)).ok()) return false;
    }
  return grid.complete() && std::ranges::equal(Encode(grid), cells);
}

bool CheckEncodedImport() {
  using gjxl::AcStrategyGrid;
  AcStrategyGrid sentinel;
  if (!AcStrategyGrid::Create({1, 1}, &sentinel).ok()) return false;
  sentinel.fill_dct8();
  const auto check = [&](gjxl::Extent2D extent, const std::vector<uint8_t>& cells) {
    AcStrategyGrid out = sentinel;
    const bool expected = ImportOracle(extent, cells);
    const bool accepted = AcStrategyGrid::CreateFromEncodedCells(extent, cells, &out).ok();
    return accepted == expected && (accepted
      ? out.complete() && std::ranges::equal(Encode(out), cells)
      : out.extent() == sentinel.extent() && Encode(out) == Encode(sentinel));
  };
  for (const auto& info : gjxl::kAcStrategyInfos) {
    AcStrategyGrid grid;
    if (!AcStrategyGrid::Create(info.covered_blocks, &grid).ok() ||
        !grid.Set(0, 0, info.type).ok() || !check(grid.extent(), Encode(grid)))
      return false;
  }
  // Include mixed/edge partitions, unknown/sentinel bytes, truncated covers,
  // changed anchor bits and ownership, and newly overlapping rectangles.
  AcStrategyGrid grid;
  if (!AcStrategyGrid::Create({8, 8}, &grid).ok() ||
      !grid.Set(0, 0, gjxl::AcStrategyType::kDct16x32).ok() ||
      !grid.Set(4, 0, gjxl::AcStrategyType::kDct32x32).ok() ||
      !grid.Set(0, 2, gjxl::AcStrategyType::kDct32x16).ok()) return false;
  grid.fill_empty_dct8();
  auto cells = Encode(grid);
  for (size_t i = 0; i < cells.size(); ++i) {
    const uint8_t saved = cells[i];
    for (unsigned value = 0; value < 256; ++value) {
      cells[i] = uint8_t(value);
      if (!check(grid.extent(), cells)) return false;
    }
    cells[i] = saved;
  }
  // Anti-diagonal 2x2 rectangles can overlap without covering either anchor.
  cells.assign(16, 1);
  cells[1] = cells[4] = 9;
  cells[2] = cells[5] = cells[6] = cells[8] = cells[9] = 8;
  cells[15] = 0;  // Hole balances overlap: sum of covered areas still equals 16.
  if (!check({4, 4}, cells)) return false;
  AcStrategyGrid out = sentinel;
  if (AcStrategyGrid::CreateFromEncodedCells({0, 1}, {}, &out).ok() ||
      AcStrategyGrid::CreateFromEncodedCells({SIZE_MAX, 2}, {}, &out).ok() ||
      AcStrategyGrid::CreateFromEncodedCells({1, 1}, {}, &out).ok() ||
      AcStrategyGrid::CreateFromEncodedCells({1, 1}, {cells.data(), 1}, nullptr).ok() ||
      Encode(out) != Encode(sentinel)) return false;
  return true;
}

}  // namespace

int main() {
  if (!CheckEncodedImport() || !CheckCreation() ||
      !CheckPlacementAndCells() ||
      !CheckCompletionAndIteration()) {
    return EXIT_FAILURE;
  }

  std::cout << "All AC-strategy grid tests passed.\n";
  return EXIT_SUCCESS;
}
