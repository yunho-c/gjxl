// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl's AC block-context and coefficient-order maps.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "core/ac_strategy.h"

namespace gjxl::codestream_internal {

inline constexpr size_t kSimpleCoefficientOrderCount = 13;

// Mapping from raw AC strategy code to the coefficient-order family used by
// the JPEG XL block-context map.
inline constexpr std::array<uint8_t, kAcStrategyCount> kSimpleStrategyOrder = {
  0, 1, 1, 1, 2, 3, 4, 4, 5,  5,  6,  6,  1,  1,
  1, 1, 1, 1, 7, 8, 8, 9, 10, 10, 11, 12, 12,
};

// Compact initial-profile block-context map as serialized in DC global. The
// decoder swaps channels 0 and 1 before indexing these three order-family
// rows; SimpleBlockContext applies the same transform for tokenization.
inline constexpr std::array<uint8_t, 3 * kSimpleCoefficientOrderCount>
kSimpleBlockContextMap = {
  0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
};

static_assert([] {
  for (const uint8_t order : kSimpleStrategyOrder) {
    if (order >= kSimpleCoefficientOrderCount) {
      return false;
    }
  }
  for (const uint8_t context : kSimpleBlockContextMap) {
    if (context >= 4) {
      return false;
    }
  }
  return true;
}());

[[nodiscard]] constexpr uint8_t SimpleBlockContext(
  AcStrategyType strategy, size_t channel) noexcept {
  const size_t channel_row = channel < 2 ? channel ^ 1u : 2;
  const size_t order =
    kSimpleStrategyOrder[static_cast<size_t>(strategy)];
  return kSimpleBlockContextMap[
    channel_row * kSimpleCoefficientOrderCount + order];
}

}  // namespace gjxl::codestream_internal
