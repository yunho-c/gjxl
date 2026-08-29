// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny and libjxl AC tokenization.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "codestream/entropy.h"
#include "core/ac_strategy.h"
#include "core/geometry.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;
struct SimpleBlockContextMap;
struct SimpleCoefficientOrders;
struct VarDctAcGroupView;

inline constexpr size_t kSimpleAcContextCount = 1980;

/// Logical AC tokens belonging to one row-major 256x256-pixel AC group.
struct SimpleAcGroupTokenStream {
  size_t block_x = 0;
  size_t block_y = 0;
  Extent2D block_extent;
  std::vector<EntropyToken> tokens;

  friend bool operator==(const SimpleAcGroupTokenStream&,
                         const SimpleAcGroupTokenStream&) = default;
};

/// Computes the default JPEG XL scan order as physical coefficient offsets.
[[nodiscard]] Status ComputeSimpleNaturalCoefficientOrder(
  AcStrategyType strategy, std::vector<uint32_t>* order);

/// Tokenizes one isolated AC group without entropy coding.
[[nodiscard]] Status TokenizeSimpleAcGroup(const VarDctAcGroupView& group,
                                           const AcStrategyGrid& strategies,
                                           std::vector<EntropyToken>* tokens);

/// Tokenizes one isolated AC group with the supplied custom order families.
[[nodiscard]] Status TokenizeSimpleAcGroup(
  const VarDctAcGroupView& group,
  const AcStrategyGrid& strategies,
  const SimpleCoefficientOrders& orders,
  std::vector<EntropyToken>* tokens);

/// Tokenizes every retained AC group in row-major order, atomically.
[[nodiscard]] Status TokenizeSimpleAcGroups(
  const VarDctEncoderFrame& frame,
  std::vector<SimpleAcGroupTokenStream>* groups);

/// Tokenizes every retained AC group with the supplied custom orders.
[[nodiscard]] Status TokenizeSimpleAcGroups(
  const VarDctEncoderFrame& frame,
  const SimpleCoefficientOrders& orders,
  std::vector<SimpleAcGroupTokenStream>* groups);

/// Tokenizes every retained AC group with custom orders and block contexts.
[[nodiscard]] Status TokenizeSimpleAcGroups(
  const VarDctEncoderFrame& frame,
  const SimpleCoefficientOrders& orders,
  const SimpleBlockContextMap& block_context_map,
  std::vector<SimpleAcGroupTokenStream>* groups);

}  // namespace gjxl
