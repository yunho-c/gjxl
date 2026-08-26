// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "core/block_grid.h"
#include "core/geometry.h"
#include "core/status.h"

namespace gjxl {

// JPEG XL VarDCT AC strategies. Index matches JPEG XL format and reference implementation (libjxl).
enum class AcStrategyType : uint8_t {
  kDct8 = 0,
  kIdentity = 1,
  kDct2x2 = 2,
  kDct4x4 = 3,
  kDct16x16 = 4,
  kDct32x32 = 5,
  kDct16x8 = 6,
  kDct8x16 = 7,
  kDct32x8 = 8,
  kDct8x32 = 9,
  kDct32x16 = 10,
  kDct16x32 = 11,
  kDct4x8 = 12,
  kDct8x4 = 13,
  kAfv0 = 14,
  kAfv1 = 15,
  kAfv2 = 16,
  kAfv3 = 17,
  kDct64x64 = 18,
  kDct64x32 = 19,
  kDct32x64 = 20,
  kDct128x128 = 21,
  kDct128x64 = 22,
  kDct64x128 = 23,
  kDct256x256 = 24,
  kDct256x128 = 25,
  kDct128x256 = 26,
  kCount = 27,
};

inline constexpr size_t kAcStrategyCount =
  static_cast<size_t>(AcStrategyType::kCount);

// Pin the starts of the format's strategy families as an independent shield
// against accidental renumbering during future table edits.
static_assert(static_cast<uint8_t>(AcStrategyType::kDct8) == 0);
static_assert(static_cast<uint8_t>(AcStrategyType::kDct16x8) == 6);
static_assert(static_cast<uint8_t>(AcStrategyType::kDct4x8) == 12);
static_assert(static_cast<uint8_t>(AcStrategyType::kAfv0) == 14);
static_assert(static_cast<uint8_t>(AcStrategyType::kDct64x64) == 18);
static_assert(static_cast<uint8_t>(AcStrategyType::kDct128x128) == 21);
static_assert(static_cast<uint8_t>(AcStrategyType::kDct256x256) == 24);
static_assert(static_cast<uint8_t>(AcStrategyType::kDct128x256) == 26);
static_assert(kAcStrategyCount == 27);

struct AcStrategyInfo {
  AcStrategyType type;
  std::string_view name;

  // Number of JPEG XL 8x8 base blocks covered by one complete transform.
  // Strategy names use libjxl's rows-by-columns convention, whereas Extent2D
  // is always width-by-height. For example, DCT16x8 covers {1, 2} blocks.
  Extent2D covered_blocks;

  [[nodiscard]] constexpr Extent2D pixel_extent() const noexcept {
    return {
      .width = covered_blocks.width * kJxlBlockDimension,
      .height = covered_blocks.height * kJxlBlockDimension,
    };
  }

  /// Physical coefficient storage, with the longer frequency axis contiguous.
  [[nodiscard]] constexpr Extent2D coefficient_extent() const noexcept {
    const Extent2D pixels = pixel_extent();
    return pixels.width > pixels.height
      ? pixels
      : Extent2D{pixels.height, pixels.width};
  }

  /// LLF storage occupied by one DC value per covered 8x8 base block.
  [[nodiscard]] constexpr Extent2D low_frequency_extent() const noexcept {
    return covered_blocks.width > covered_blocks.height
      ? covered_blocks
      : Extent2D{covered_blocks.height, covered_blocks.width};
  }

  /// Maps natural [vertical][horizontal] frequencies to libjxl storage.
  [[nodiscard]] constexpr size_t coefficient_index(
    size_t vertical_frequency,
    size_t horizontal_frequency) const noexcept {

    const Extent2D pixels = pixel_extent();
    if (pixels.height < pixels.width) {
      return vertical_frequency * pixels.width + horizontal_frequency;
    }

    return horizontal_frequency * pixels.height + vertical_frequency;
  }

  [[nodiscard]] constexpr size_t coefficient_count() const noexcept {
    const Extent2D extent = pixel_extent();
    return extent.width * extent.height;
  }
};

inline constexpr std::array<AcStrategyInfo, kAcStrategyCount>
kAcStrategyInfos{{
  {AcStrategyType::kDct8, "DCT8", {1, 1}},
  {AcStrategyType::kIdentity, "IDENTITY", {1, 1}},
  {AcStrategyType::kDct2x2, "DCT2x2", {1, 1}},
  {AcStrategyType::kDct4x4, "DCT4x4", {1, 1}},
  {AcStrategyType::kDct16x16, "DCT16x16", {2, 2}},
  {AcStrategyType::kDct32x32, "DCT32x32", {4, 4}},
  {AcStrategyType::kDct16x8, "DCT16x8", {1, 2}},
  {AcStrategyType::kDct8x16, "DCT8x16", {2, 1}},
  {AcStrategyType::kDct32x8, "DCT32x8", {1, 4}},
  {AcStrategyType::kDct8x32, "DCT8x32", {4, 1}},
  {AcStrategyType::kDct32x16, "DCT32x16", {2, 4}},
  {AcStrategyType::kDct16x32, "DCT16x32", {4, 2}},
  {AcStrategyType::kDct4x8, "DCT4x8", {1, 1}},
  {AcStrategyType::kDct8x4, "DCT8x4", {1, 1}},
  {AcStrategyType::kAfv0, "AFV0", {1, 1}},
  {AcStrategyType::kAfv1, "AFV1", {1, 1}},
  {AcStrategyType::kAfv2, "AFV2", {1, 1}},
  {AcStrategyType::kAfv3, "AFV3", {1, 1}},
  {AcStrategyType::kDct64x64, "DCT64x64", {8, 8}},
  {AcStrategyType::kDct64x32, "DCT64x32", {4, 8}},
  {AcStrategyType::kDct32x64, "DCT32x64", {8, 4}},
  {AcStrategyType::kDct128x128, "DCT128x128", {16, 16}},
  {AcStrategyType::kDct128x64, "DCT128x64", {8, 16}},
  {AcStrategyType::kDct64x128, "DCT64x128", {16, 8}},
  {AcStrategyType::kDct256x256, "DCT256x256", {32, 32}},
  {AcStrategyType::kDct256x128, "DCT256x128", {16, 32}},
  {AcStrategyType::kDct128x256, "DCT128x256", {32, 16}},
}};

static_assert([] {
  for (size_t i = 0; i < kAcStrategyInfos.size(); ++i) {
    if (static_cast<size_t>(kAcStrategyInfos[i].type) != i) {
      return false;
    }
  }

  return true;
}());

[[nodiscard]] constexpr const AcStrategyInfo* GetAcStrategyInfo(
  AcStrategyType type) noexcept {

  const size_t index = static_cast<size_t>(type);

  if (index >= kAcStrategyInfos.size()) {
    return nullptr;
  }

  return &kAcStrategyInfos[index];
}

struct AcStrategyCell {
  AcStrategyType strategy = AcStrategyType::kDct8;
  bool is_anchor = false;
};

/// Owns one encoded AC-strategy cell per JPEG XL 8x8 base block.
///
/// Every covered cell stores the selected strategy. Exactly the top-left cell
/// of each complete transform is marked as its anchor.
class AcStrategyGrid {
public:
  AcStrategyGrid() = default;

  [[nodiscard]] static Status Create(
    Extent2D extent,
    AcStrategyGrid* out) {

    if (out == nullptr) {
      return Status::InvalidArgument(
        "AC-strategy grid output is null");
    }

    if (extent.empty()) {
      return Status::InvalidArgument(
        "AC-strategy grid extent must be non-empty");
    }

    size_t cell_count = 0;
    if (!extent.try_area(&cell_count)) {
      return Status::InvalidArgument(
        "AC-strategy grid extent is too large");
    }

    try {
      AcStrategyGrid result;
      result.extent_ = extent;
      result.cells_.assign(cell_count, kInvalidCell);
      *out = std::move(result);
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate AC-strategy grid");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "AC-strategy grid extent is too large");
    }

    return Status::Ok();
  }

  [[nodiscard]] bool valid() const noexcept {
    size_t cell_count = 0;
    return !extent_.empty() &&
      extent_.try_area(&cell_count) &&
      cells_.size() == cell_count;
  }

  [[nodiscard]] Extent2D extent() const noexcept {
    return extent_;
  }

  [[nodiscard]] bool complete() const noexcept {
    return valid() && std::ranges::none_of(
      cells_,
      [](uint8_t cell) { return cell == kInvalidCell; });
  }

  [[nodiscard]] bool occupied(size_t x, size_t y) const noexcept {
    return valid() && x < extent_.width && y < extent_.height &&
      cells_[y * extent_.width + x] != kInvalidCell;
  }

  void clear() noexcept {
    std::ranges::fill(cells_, kInvalidCell);
  }

  void fill_dct8() noexcept {
    std::ranges::fill(
      cells_,
      EncodeCell(AcStrategyType::kDct8, true));
  }

  void fill_empty_dct8() noexcept {
    std::ranges::replace(
      cells_,
      kInvalidCell,
      EncodeCell(AcStrategyType::kDct8, true));
  }

  [[nodiscard]] Status Set(
    size_t x,
    size_t y,
    AcStrategyType strategy) {

    if (!valid()) {
      return Status::InvalidArgument(
        "AC-strategy grid is invalid");
    }

    const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
    if (info == nullptr) {
      return Status::InvalidArgument(
        "Unknown AC strategy");
    }

    const Extent2D covered = info->covered_blocks;
    if (x >= extent_.width ||
        y >= extent_.height ||
        covered.width > extent_.width - x ||
        covered.height > extent_.height - y) {
      return Status::InvalidArgument(
        "AC strategy does not fit in the grid");
    }

    for (size_t dy = 0; dy < covered.height; ++dy) {
      for (size_t dx = 0; dx < covered.width; ++dx) {
        if (cells_[(y + dy) * extent_.width + x + dx] != kInvalidCell) {
          return Status::InvalidArgument(
            "AC strategy overlaps an occupied block");
        }
      }
    }

    for (size_t dy = 0; dy < covered.height; ++dy) {
      for (size_t dx = 0; dx < covered.width; ++dx) {
        cells_[(y + dy) * extent_.width + x + dx] =
          EncodeCell(strategy, dx == 0 && dy == 0);
      }
    }

    return Status::Ok();
  }

  [[nodiscard]] Status Get(
    size_t x,
    size_t y,
    AcStrategyCell* out) const {

    if (out == nullptr) {
      return Status::InvalidArgument(
        "AC-strategy cell output is null");
    }

    if (!valid() || x >= extent_.width || y >= extent_.height) {
      return Status::InvalidArgument(
        "AC-strategy cell coordinates are invalid");
    }

    const uint8_t encoded = cells_[y * extent_.width + x];
    if (encoded == kInvalidCell) {
      return Status::InvalidArgument(
        "AC-strategy cell is unoccupied");
    }

    *out = {
      .strategy = static_cast<AcStrategyType>(encoded >> 1),
      .is_anchor = (encoded & 1u) != 0,
    };
    return Status::Ok();
  }

  template <typename Function>
  [[nodiscard]] Status ForEachAnchor(Function&& function) const {
    if (!complete()) {
      return Status::InvalidArgument(
        "AC-strategy grid is incomplete");
    }

    for (size_t y = 0; y < extent_.height; ++y) {
      for (size_t x = 0; x < extent_.width; ++x) {
        const uint8_t encoded = cells_[y * extent_.width + x];
        if ((encoded & 1u) == 0) {
          continue;
        }

        Status status = function(
          x,
          y,
          static_cast<AcStrategyType>(encoded >> 1));
        if (!status.ok()) {
          return status;
        }
      }
    }

    return Status::Ok();
  }

private:
  static constexpr uint8_t kInvalidCell = 0xff;

  [[nodiscard]] static constexpr uint8_t EncodeCell(
    AcStrategyType strategy,
    bool is_anchor) noexcept {

    return static_cast<uint8_t>(
      (static_cast<uint8_t>(strategy) << 1) |
      static_cast<uint8_t>(is_anchor));
  }

  Extent2D extent_;
  std::vector<uint8_t> cells_;
};

}  // namespace gjxl
