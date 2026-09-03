// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl's coefficient-order encoder.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "codestream/entropy.h"
#include "codestream/simple_ac_context.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;
enum class VarDctCoefficientOrderBehavior : uint8_t;

inline constexpr size_t kSimplePermutationContextCount = 8;

/// Custom physical coefficient scans indexed by order family and channel.
/// Families absent from `used_order_mask` have empty vectors and use the
/// format's natural order.
struct SimpleCoefficientOrders {
  uint16_t used_order_mask = 0;
  std::array<
    std::array<std::vector<uint32_t>, 3>,
    codestream_internal::kSimpleCoefficientOrderCount> orders;

  friend bool operator==(
    const SimpleCoefficientOrders&,
    const SimpleCoefficientOrders&) = default;
};

/// Counts coefficient zeros over the completed frame and derives the one
/// pinned-libjxl-compatible custom-order candidate, atomically.
[[nodiscard]] Status ComputeSimpleCoefficientOrders(
  const VarDctEncoderFrame& frame,
  SimpleCoefficientOrders* orders);

namespace codestream_internal {

/// Serializer-only entry point for an already validated frame.
[[nodiscard]] Status ComputeSimpleCoefficientOrdersForEncoder(
  const VarDctEncoderFrame& frame,
  VarDctCoefficientOrderBehavior behavior,
  SimpleCoefficientOrders* orders);

}  // namespace codestream_internal

/// Validates supported family bits, dimensions, permutations, and LLF
/// prefixes.
[[nodiscard]] Status ValidateSimpleCoefficientOrders(
  const SimpleCoefficientOrders& orders);

/// Converts all selected physical scans to natural-order Lehmer tokens.
[[nodiscard]] Status TokenizeSimpleCoefficientOrders(
  const SimpleCoefficientOrders& orders,
  std::vector<EntropyToken>* tokens);

}  // namespace gjxl
