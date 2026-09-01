// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny and libjxl AC tokenization.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
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

/// Block-context input shared by every token emitted for one transform/channel
/// pair. The final context label depends on the selected block-context map, but
/// these inputs do not.
struct SimpleAcBlockContextKey {
  uint16_t raw_quant = 1;
  uint8_t channel_row = 0;
  uint8_t order_family = 0;

  friend bool operator==(const SimpleAcBlockContextKey&,
                         const SimpleAcBlockContextKey&) = default;
};

/// Map-independent AC context descriptor. The high bit of local_context marks
/// coefficient tokens; the remaining bits hold either a nonzero bucket or
/// zero-density context. Values live in a separate contiguous array so entropy
/// candidates can share them directly.
struct SimpleAcTokenTemplate {
  static constexpr uint16_t kCoefficientFlag = uint16_t{1} << 15;

  uint16_t block_context_key = 0;
  uint16_t local_context = 0;

  [[nodiscard]] bool is_coefficient() const noexcept {
    return (local_context & kCoefficientFlag) != 0;
  }

  [[nodiscard]] uint16_t context_without_block() const noexcept {
    return local_context & static_cast<uint16_t>(~kCoefficientFlag);
  }

  friend bool operator==(const SimpleAcTokenTemplate&,
                         const SimpleAcTokenTemplate&) = default;
};

static_assert(sizeof(SimpleAcTokenTemplate) == 4);

/// AC-group token data that is invariant across block-context-map candidates.
struct SimpleAcGroupTokenTemplate {
  size_t block_x = 0;
  size_t block_y = 0;
  Extent2D block_extent;
  std::vector<SimpleAcBlockContextKey> block_context_keys;
  std::vector<uint32_t> values;
  std::vector<SimpleAcTokenTemplate> tokens;

  friend bool operator==(const SimpleAcGroupTokenTemplate&,
                         const SimpleAcGroupTokenTemplate&) = default;
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

/// Builds every retained AC group once for one coefficient-order selection.
/// The resulting values and local contexts can be reused by all block maps.
[[nodiscard]] Status BuildSimpleAcGroupTokenTemplates(
  const VarDctEncoderFrame& frame,
  const SimpleCoefficientOrders& orders,
  std::vector<SimpleAcGroupTokenTemplate>* groups);

/// Resolves one order template into the exact entropy-token streams for a
/// selected block-context map. The output remains unchanged on failure.
[[nodiscard]] Status MaterializeSimpleAcGroupTokenStreams(
  std::span<const SimpleAcGroupTokenTemplate> templates,
  const SimpleBlockContextMap& block_context_map,
  std::vector<SimpleAcGroupTokenStream>* groups);

/// Resolves only the candidate-specific contexts. Each output vector has the
/// same length and indexing as its template's shared values array.
[[nodiscard]] Status MaterializeSimpleAcGroupContexts(
  std::span<const SimpleAcGroupTokenTemplate> templates,
  const SimpleBlockContextMap& block_context_map,
  std::vector<std::vector<uint16_t>>* contexts);

}  // namespace gjxl
