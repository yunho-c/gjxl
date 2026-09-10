// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl's coefficient-order encoder.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "codestream/storage.h"

#include "codestream/entropy.h"
#include "codestream/simple_ac_context.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;
namespace vardct_frame_internal {
class VarDctFrameView;
}
enum class VarDctCoefficientOrderBehavior : uint8_t;

inline constexpr size_t kSimplePermutationContextCount = 8;

/// Custom physical coefficient scans indexed by order family and channel.
/// Families absent from `used_order_mask` have empty vectors and use the
/// format's natural order.
struct SimpleCoefficientOrders {
  uint16_t used_order_mask = 0;
  std::array<
    std::array<codestream_internal::Storage<uint32_t>, 3>,
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
  const vardct_frame_internal::VarDctFrameView& frame,
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
  codestream_internal::Storage<EntropyToken>* tokens);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status TokenizeSimpleCoefficientOrders(
  const SimpleCoefficientOrders& orders,
  std::vector<EntropyToken, Allocator>* tokens) {
  return codestream_internal::LegacyStorageOutput(
    tokens, [&](auto* storage) { return TokenizeSimpleCoefficientOrders(orders, storage); });
}

}  // namespace gjxl
