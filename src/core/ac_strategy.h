// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "core/block_grid.h"
#include "core/geometry.h"

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

}  // namespace gjxl
