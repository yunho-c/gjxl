// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace gjxl::vardct_frame_internal {

// Physical coefficient positions for the five currently supported order
// families. Rectangular orientations share a family and physical layout.
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

// An internal producer computes these exact populations from the same final
// quantized AC and strategies it submits to frame assembly. Full counts are
// always present. Only a pure DCT8 frame has sampled counts, using the pinned
// AC-group-first PRNG traversal. All absent-family/sample bins are zero.
// Assembly copies the data, checks its shape and bounds, and does not recount
// coefficients; semantic equality is the producer's responsibility.
struct CoefficientOrderPopulation {
  std::array<uint32_t, kOrderPopulationCount> counts{};
  uint16_t present_mask = 0;
};

}  // namespace gjxl::vardct_frame_internal
