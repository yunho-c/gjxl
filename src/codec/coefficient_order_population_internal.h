// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace gjxl::vardct_frame_internal {

// Physical coefficient positions. Rectangular orientations share a family.
inline constexpr std::array<size_t, 7> kOrderPopulationSizes =
  {64, 0, 256, 1024, 128, 0, 512};
inline constexpr std::array<size_t, 7> kOrderPopulationOffsets =
  {0, 0, 64, 320, 1344, 0, 1472};
inline constexpr size_t kOrderPopulationStride = 1984;
inline constexpr size_t kOrderPopulationFullCount = 3 * kOrderPopulationStride;
inline constexpr size_t kOrderPopulationCount = kOrderPopulationFullCount + 3 * 64;

inline constexpr size_t OrderPopulationFamily(size_t coefficient_count) {
  for (size_t family = 0; family < kOrderPopulationSizes.size(); ++family) {
    if (coefficient_count != 0 && kOrderPopulationSizes[family] == coefficient_count)
      return family;
  }
  return kOrderPopulationSizes.size();
}

// Optional, immutable producer cache with the completed frame's lifetime.
// The producer guarantees exact zero counts from that frame's final AC and
// strategies. Full counts are always present; pure DCT8 also has the pinned
// group-first sampled counts. All absent-family/sample bins are zero. Consumers
// check shape and bounds, not semantic equality (which requires a full scan).
// Empty storage requests the CPU recount, including its wider counter fallback.
struct CoefficientOrderPopulationView {
  std::span<const uint32_t> counts;
  uint16_t present_mask = 0;
};

}  // namespace gjxl::vardct_frame_internal
