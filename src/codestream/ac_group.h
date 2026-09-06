// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny and libjxl AC tokenization.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "codestream/storage.h"

#include "codestream/entropy.h"
#include "core/ac_strategy.h"
#include "core/geometry.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;
namespace vardct_frame_internal {
class VarDctFrameView;
}
struct SimpleBlockContextMap;
struct SimpleCoefficientOrders;
struct VarDctAcGroupView;

inline constexpr size_t kSimpleAcContextCount = 1980;

/// Logical AC tokens belonging to one row-major 256x256-pixel AC group.
struct SimpleAcGroupTokenStream {
  size_t block_x = 0;
  size_t block_y = 0;
  Extent2D block_extent;
  codestream_internal::Storage<EntropyToken> tokens;

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
  codestream_internal::Storage<SimpleAcBlockContextKey> block_context_keys;
  codestream_internal::Storage<uint32_t> values;
  codestream_internal::Storage<SimpleAcTokenTemplate> tokens;

  friend bool operator==(const SimpleAcGroupTokenTemplate&,
                         const SimpleAcGroupTokenTemplate&) = default;
};

/// Computes the default JPEG XL scan order as physical coefficient offsets.
[[nodiscard]] Status ComputeSimpleNaturalCoefficientOrder(
  AcStrategyType strategy, codestream_internal::Storage<uint32_t>* order);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status ComputeSimpleNaturalCoefficientOrder(
  AcStrategyType strategy,
  std::vector<uint32_t, Allocator>* order) {
  return codestream_internal::LegacyStorageOutput(
    order, [&](auto* storage) { return ComputeSimpleNaturalCoefficientOrder(strategy, storage); });
}

/// Tokenizes one isolated AC group without entropy coding.
[[nodiscard]] Status TokenizeSimpleAcGroup(const VarDctAcGroupView& group,
                                           const AcStrategyGrid& strategies,
                                           codestream_internal::Storage<EntropyToken>* tokens);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status TokenizeSimpleAcGroup(
  const VarDctAcGroupView& group,
  const AcStrategyGrid& strategies,
  std::vector<EntropyToken, Allocator>* tokens) {
  return codestream_internal::LegacyStorageOutput(
    tokens, [&](auto* storage) { return TokenizeSimpleAcGroup(group, strategies, storage); });
}

/// Tokenizes one isolated AC group with the supplied custom order families.
[[nodiscard]] Status TokenizeSimpleAcGroup(
  const VarDctAcGroupView& group,
  const AcStrategyGrid& strategies,
  const SimpleCoefficientOrders& orders,
  codestream_internal::Storage<EntropyToken>* tokens);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status TokenizeSimpleAcGroup(
  const VarDctAcGroupView& group,
  const AcStrategyGrid& strategies,
  const SimpleCoefficientOrders& orders,
  std::vector<EntropyToken, Allocator>* tokens) {
  return codestream_internal::LegacyStorageOutput(
    tokens, [&](auto* storage) { return TokenizeSimpleAcGroup(group, strategies, orders, storage); });
}

/// Tokenizes every retained AC group in row-major order, atomically.
[[nodiscard]] Status TokenizeSimpleAcGroups(
  const VarDctEncoderFrame& frame,
  codestream_internal::Storage<SimpleAcGroupTokenStream>* groups);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status TokenizeSimpleAcGroups(
  const VarDctEncoderFrame& frame,
  std::vector<SimpleAcGroupTokenStream, Allocator>* groups) {
  return codestream_internal::LegacyStorageOutput(
    groups, [&](auto* storage) { return TokenizeSimpleAcGroups(frame, storage); });
}

/// Tokenizes every retained AC group with the supplied custom orders.
[[nodiscard]] Status TokenizeSimpleAcGroups(
  const VarDctEncoderFrame& frame,
  const SimpleCoefficientOrders& orders,
  codestream_internal::Storage<SimpleAcGroupTokenStream>* groups);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status TokenizeSimpleAcGroups(
  const VarDctEncoderFrame& frame,
  const SimpleCoefficientOrders& orders,
  std::vector<SimpleAcGroupTokenStream, Allocator>* groups) {
  return codestream_internal::LegacyStorageOutput(
    groups, [&](auto* storage) { return TokenizeSimpleAcGroups(frame, orders, storage); });
}

/// Tokenizes every retained AC group with custom orders and block contexts.
[[nodiscard]] Status TokenizeSimpleAcGroups(
  const VarDctEncoderFrame& frame,
  const SimpleCoefficientOrders& orders,
  const SimpleBlockContextMap& block_context_map,
  codestream_internal::Storage<SimpleAcGroupTokenStream>* groups);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status TokenizeSimpleAcGroups(
  const VarDctEncoderFrame& frame,
  const SimpleCoefficientOrders& orders,
  const SimpleBlockContextMap& block_context_map,
  std::vector<SimpleAcGroupTokenStream, Allocator>* groups) {
  return codestream_internal::LegacyStorageOutput(
    groups, [&](auto* storage) { return TokenizeSimpleAcGroups(frame, orders, block_context_map, storage); });
}

/// Builds every retained AC group once for one coefficient-order selection.
/// The resulting values and local contexts can be reused by all block maps.
[[nodiscard]] Status BuildSimpleAcGroupTokenTemplates(
  const VarDctEncoderFrame& frame,
  const SimpleCoefficientOrders& orders,
  codestream_internal::Storage<SimpleAcGroupTokenTemplate>* groups);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status BuildSimpleAcGroupTokenTemplates(
  const VarDctEncoderFrame& frame,
  const SimpleCoefficientOrders& orders,
  std::vector<SimpleAcGroupTokenTemplate, Allocator>* groups) {
  return codestream_internal::LegacyStorageOutput(
    groups, [&](auto* storage) { return BuildSimpleAcGroupTokenTemplates(frame, orders, storage); });
}

/// Resolves one order template into the exact entropy-token streams for a
/// selected block-context map. The output remains unchanged on failure.
[[nodiscard]] Status MaterializeSimpleAcGroupTokenStreams(
  std::span<const SimpleAcGroupTokenTemplate> templates,
  const SimpleBlockContextMap& block_context_map,
  codestream_internal::Storage<SimpleAcGroupTokenStream>* groups);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status MaterializeSimpleAcGroupTokenStreams(
  std::span<const SimpleAcGroupTokenTemplate> templates,
  const SimpleBlockContextMap& block_context_map,
  std::vector<SimpleAcGroupTokenStream, Allocator>* groups) {
  return codestream_internal::LegacyStorageOutput(
    groups, [&](auto* storage) { return MaterializeSimpleAcGroupTokenStreams(templates, block_context_map, storage); });
}

/// Resolves only the candidate-specific contexts. Each output vector has the
/// same length and indexing as its template's shared values array.
[[nodiscard]] Status MaterializeSimpleAcGroupContexts(
  std::span<const SimpleAcGroupTokenTemplate> templates,
  const SimpleBlockContextMap& block_context_map,
  codestream_internal::Storage<codestream_internal::Storage<uint16_t>>* contexts);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status MaterializeSimpleAcGroupContexts(
  std::span<const SimpleAcGroupTokenTemplate> templates,
  const SimpleBlockContextMap& block_context_map,
  std::vector<std::vector<uint16_t>, Allocator>* contexts) {
  return codestream_internal::LegacyNestedStorageOutput(
    contexts, [&](auto* storage) { return MaterializeSimpleAcGroupContexts(templates, block_context_map, storage); });
}

namespace codestream_internal {

/// Already validated frame and coefficient orders; borrows only for this call.
[[nodiscard]] Status BuildSimpleAcGroupTokenTemplatesForEncoder(
  const vardct_frame_internal::VarDctFrameView& frame,
  const SimpleCoefficientOrders& orders,
  codestream_internal::Storage<SimpleAcGroupTokenTemplate>* groups);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status BuildSimpleAcGroupTokenTemplatesForEncoder(
  const vardct_frame_internal::VarDctFrameView& frame,
  const SimpleCoefficientOrders& orders,
  std::vector<SimpleAcGroupTokenTemplate, Allocator>* groups) {
  return codestream_internal::LegacyStorageOutput(
    groups, [&](auto* storage) { return BuildSimpleAcGroupTokenTemplatesForEncoder(frame, orders, storage); });
}

struct SimpleAcStrategyAnchor {
  size_t x = 0;
  size_t y = 0;
  AcStrategyType strategy = AcStrategyType::kDct8;
  size_t coefficient_count = 0;
};

struct SimpleAcNaturalOrders {
  std::array<codestream_internal::Storage<uint32_t>, kAcStrategyCount> orders;
};

struct SimpleAcSymbolPopulation {
  uint8_t symbol = 0;
  uint32_t count = 0;
};

struct SimpleAcContextPopulation {
  uint16_t context = 0;
  uint32_t symbol_offset = 0;
  uint16_t symbol_count = 0;
  uint64_t token_count = 0;
  uint64_t extra_bits = 0;
  uint32_t maximum_symbol = 0;
};

/// Final ordinary-path token data plus optional group-local fixed-HybridUint
/// populations. Population entries are compact and reference contiguous
/// sparse symbol/count runs.
struct SimpleAcGroupTokenData {
  codestream_internal::Storage<uint32_t> values;
  codestream_internal::Storage<uint16_t> contexts;
  codestream_internal::Storage<SimpleAcContextPopulation> context_populations;
  codestream_internal::Storage<SimpleAcSymbolPopulation> symbol_populations;
};

struct SimpleAcPopulationAccumulator {
  uint16_t context = 0;
  std::array<uint32_t, kPrefixAlphabetSize> counts{};
  uint64_t token_count = 0;
  uint64_t extra_bits = 0;
  uint32_t maximum_symbol = 0;
};

/// Reused by one serializer worker across independent AC groups.
struct SimpleAcTokenizationScratch {
  codestream_internal::Storage<SimpleAcStrategyAnchor> anchors;
  std::array<codestream_internal::Storage<uint8_t>, 3> nonzero_maps;
  codestream_internal::Storage<uint16_t> population_slots;
  codestream_internal::Storage<SimpleAcPopulationAccumulator> populations;
};

[[nodiscard]] Status PrepareSimpleAcNaturalOrders(
  SimpleAcNaturalOrders* orders);

/// One-pass ordinary serializer primitive for an already validated frame,
/// coefficient-order set, and block-context map.
[[nodiscard]] Status TokenizeSimpleAcGroupForEncoder(
  const vardct_frame_internal::VarDctFrameView& frame,
  const SimpleCoefficientOrders& orders,
  const SimpleAcNaturalOrders& natural_orders,
  const SimpleBlockContextMap& block_context_map,
  size_t group_index,
  bool collect_fixed_populations,
  SimpleAcTokenizationScratch* scratch,
  SimpleAcGroupTokenData* group);

/// Encoder-only per-group primitives. The enclosing serializer validates the
/// frame, coefficient orders, and block-context map once before dispatching
/// independent group tasks into fixed output slots.
[[nodiscard]] Status BuildSimpleAcGroupTokenTemplateForEncoder(
  const vardct_frame_internal::VarDctFrameView& frame,
  const SimpleCoefficientOrders& orders,
  size_t group_index,
  SimpleAcGroupTokenTemplate* group);

[[nodiscard]] Status MaterializeSimpleAcGroupContextsForEncoder(
  const SimpleAcGroupTokenTemplate& token_template,
  const SimpleBlockContextMap& block_context_map,
  codestream_internal::Storage<uint16_t>* contexts);

/// Compatibility adapter; managed callers select the non-template overload.
template <typename Allocator>
[[nodiscard]] Status MaterializeSimpleAcGroupContextsForEncoder(
  const SimpleAcGroupTokenTemplate& token_template,
  const SimpleBlockContextMap& block_context_map,
  std::vector<uint16_t, Allocator>* contexts) {
  return codestream_internal::LegacyStorageOutput(
    contexts, [&](auto* storage) { return MaterializeSimpleAcGroupContextsForEncoder(token_template, block_context_map, storage); });
}

}  // namespace codestream_internal

}  // namespace gjxl
