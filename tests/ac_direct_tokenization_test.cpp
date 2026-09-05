// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "codestream/ac_group.h"
#include "codestream/block_context_map.h"
#include "codestream/coefficient_order.h"
#include "codestream/entropy.h"
#include "quantized_frame_fixture.h"

namespace {

using gjxl_test::Check;

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct ReferencePopulation {
  uint16_t context = 0;
  std::array<uint64_t, gjxl::kPrefixAlphabetSize> counts{};
  uint64_t token_count = 0;
  uint64_t extra_bits = 0;
  uint32_t maximum_symbol = 0;
};

void CheckCoefficientValues(
  const gjxl::VarDctEncoderFrame& frame, size_t group_index,
  const gjxl::SimpleCoefficientOrders& orders,
  const gjxl::codestream_internal::SimpleAcNaturalOrders& natural,
  std::span<const gjxl::EntropyToken> tokens) {
  gjxl::VarDctAcGroupView group;
  Check(frame.GetAcGroup(group_index, &group));
  size_t source = 0;
  size_t token_index = 0;
  for (size_t y = 0; y < group.block_extent.height; ++y) {
    for (size_t x = 0; x < group.block_extent.width; ++x) {
      gjxl::AcStrategyCell cell;
      Check(frame.strategies().Get(group.block_x + x, group.block_y + y, &cell));
      if (!cell.is_anchor) continue;
      const auto& info = *gjxl::GetAcStrategyInfo(cell.strategy);
      const auto extent = info.coefficient_extent();
      const auto llf = info.low_frequency_extent();
      const size_t count = info.coefficient_count();
      const size_t strategy = static_cast<size_t>(cell.strategy);
      const size_t family = gjxl::codestream_internal::kSimpleStrategyOrder[strategy];
      for (const size_t channel : {size_t{1}, size_t{0}, size_t{2}}) {
        const auto coefficients = group.coefficients[channel].subspan(source, count);
        uint32_t nonzeros = 0;
        // Retain the original coordinate-wise scalar oracle. Both public and
        // direct tokenizers share the optimized counter, so comparing those
        // two alone would not independently verify the nonzero-count change.
        for (size_t cy = 0; cy < extent.height; ++cy) {
          for (size_t cx = 0; cx < extent.width; ++cx) {
            if (cx < llf.width && cy < llf.height) continue;
            if (coefficients[cy * extent.width + cx] != 0) ++nonzeros;
          }
        }
        Require(token_index < tokens.size() &&
                  tokens[token_index++].value == nonzeros,
                "AC nonzero count differs from coordinate-wise oracle");
        const auto& order = (orders.used_order_mask & (uint16_t{1} << family))
          ? orders.orders[family][channel] : natural.orders[strategy];
        uint32_t remaining = nonzeros;
        for (size_t scan = llf.width * llf.height;
             scan < count && remaining != 0; ++scan) {
          const int32_t coefficient = coefficients[order[scan]];
          Require(token_index < tokens.size() &&
                    tokens[token_index++].value == gjxl::PackSigned(coefficient),
                  "AC coefficient value differs from scalar oracle");
          remaining -= coefficient != 0;
        }
        Require(remaining == 0, "Scalar AC scan missed a nonzero coefficient");
      }
      source += count;
    }
  }
  Require(source == group.used_coefficient_count && token_index == tokens.size(),
          "Scalar AC oracle consumption differs");
}

std::vector<ReferencePopulation> ReferencePopulations(
  std::span<const gjxl::EntropyToken> tokens, size_t context_count) {
  std::vector<int32_t> slots(context_count, -1);
  std::vector<ReferencePopulation> result;
  for (const auto token : tokens) {
    Require(token.context < context_count, "Reference context is out of range");
    if (slots[token.context] == -1) {
      slots[token.context] = static_cast<int32_t>(result.size());
      result.push_back({.context = static_cast<uint16_t>(token.context)});
    }
    auto& population = result[static_cast<size_t>(slots[token.context])];
    gjxl::HybridUintToken encoded;
    Check(gjxl::EncodeHybridUint(token.value, gjxl::kDefaultHybridUintConfig,
                                &encoded));
    Require(encoded.symbol < population.counts.size(),
            "Reference symbol is out of range");
    ++population.counts[encoded.symbol];
    ++population.token_count;
    population.extra_bits += encoded.extra_bit_count;
    population.maximum_symbol = std::max(population.maximum_symbol,
                                         encoded.symbol);
  }
  return result;
}

void CheckPopulations(
  const gjxl::codestream_internal::SimpleAcGroupTokenData& actual,
  std::span<const ReferencePopulation> expected) {
  Require(actual.context_populations.size() == expected.size(),
          "Direct context population count differs");
  size_t next_symbol = 0;
  for (size_t index = 0; index < expected.size(); ++index) {
    const auto& a = actual.context_populations[index];
    const auto& b = expected[index];
    Require(a.context == b.context && a.token_count == b.token_count &&
              a.extra_bits == b.extra_bits && a.maximum_symbol == b.maximum_symbol &&
              a.symbol_offset == next_symbol,
            "Direct context population metadata differs");
    size_t symbol_count = 0;
    for (size_t symbol = 0; symbol < b.counts.size(); ++symbol) {
      if (b.counts[symbol] == 0) continue;
      Require(next_symbol < actual.symbol_populations.size(),
              "Direct sparse symbols are truncated");
      const auto& value = actual.symbol_populations[next_symbol++];
      Require(value.symbol == symbol && value.count == b.counts[symbol],
              "Direct sparse symbol or count differs");
      ++symbol_count;
    }
    Require(a.symbol_count == symbol_count, "Direct sparse symbol count differs");
  }
  Require(next_symbol == actual.symbol_populations.size(),
          "Direct sparse symbols have trailing data");
}

std::array<gjxl::SimpleBlockContextMap, 4> ContextMaps() {
  auto qf = gjxl::DefaultSimpleBlockContextMap();
  qf.qf_thresholds = {16, 64};
  qf.num_contexts = 6;
  qf.context_map.resize(
    3 * gjxl::codestream_internal::kSimpleCoefficientOrderCount * 3);
  for (size_t index = 0; index < qf.context_map.size(); ++index) {
    qf.context_map[index] = static_cast<uint8_t>((index / 3 + index % 3) % 6);
  }
  return {gjxl::DefaultSimpleBlockContextMap(),
          gjxl::JxlDefaultSimpleBlockContextMap(),
          gjxl::TwoChannelSimpleBlockContextMap(), qf};
}

void CheckRejectedGroup(
  const gjxl::VarDctEncoderFrame& frame,
  const gjxl::codestream_internal::SimpleAcNaturalOrders& natural,
  const gjxl::SimpleBlockContextMap& map,
  gjxl::codestream_internal::SimpleAcTokenizationScratch* scratch) {
  gjxl::codestream_internal::SimpleAcGroupTokenData output;
  output.values = {123};
  output.contexts = {9};
  output.context_populations.push_back({.context = 9, .token_count = 1});
  output.symbol_populations.push_back({.symbol = 7, .count = 1});
  const auto status = gjxl::codestream_internal::TokenizeSimpleAcGroupForEncoder(
    frame, {}, natural, map, frame.ac_group_count(), true, scratch, &output);
  Require(!status.ok() && output.values == std::vector<uint32_t>{123} &&
            output.contexts == std::vector<uint16_t>{9} &&
            output.context_populations.size() == 1 &&
            output.context_populations.front().context == 9 &&
            output.context_populations.front().token_count == 1 &&
            output.symbol_populations.size() == 1 &&
            output.symbol_populations.front().symbol == 7 &&
            output.symbol_populations.front().count == 1,
          "Rejected direct group changed its destination");
}

}  // namespace

int main() {
  try {
    const auto maps = ContextMaps();
    for (const auto& map : maps) Check(gjxl::ValidateSimpleBlockContextMap(map));
    gjxl::codestream_internal::SimpleAcNaturalOrders natural;
    Check(gjxl::codestream_internal::PrepareSimpleAcNaturalOrders(&natural));
    gjxl::codestream_internal::SimpleAcTokenizationScratch scratch;
    size_t cases = 0;
    for (size_t strategy = 0; strategy <= gjxl_test::kStrategies.size(); ++strategy) {
      for (size_t pattern = 0; pattern < 8; ++pattern) {
        const auto frame = gjxl_test::MakeFrame(strategy, pattern);
        gjxl::SimpleCoefficientOrders custom;
        Check(gjxl::ComputeSimpleCoefficientOrders(frame, &custom));
        const std::array orders = {gjxl::SimpleCoefficientOrders{}, custom};
        for (const auto& order : orders) {
          for (const auto& map : maps) {
            // The checked public/template route is separate from the direct
            // production helper being tested.
            std::vector<gjxl::SimpleAcGroupTokenStream> reference;
            Check(gjxl::TokenizeSimpleAcGroups(frame, order, map, &reference));
            Require(reference.size() == frame.ac_group_count(),
                    "Reference group count differs");
            for (size_t group = 0; group < reference.size(); ++group) {
              CheckCoefficientValues(frame, group, order, natural, reference[group].tokens);
              const auto expected = ReferencePopulations(
                reference[group].tokens, map.ac_context_count());
              for (const bool collect : {true, false}) {
                gjxl::codestream_internal::SimpleAcGroupTokenData actual;
                Check(gjxl::codestream_internal::TokenizeSimpleAcGroupForEncoder(
                  frame, order, natural, map, group, collect, &scratch, &actual));
                Require(actual.values.size() == reference[group].tokens.size() &&
                          actual.contexts.size() == actual.values.size(),
                        "Direct token dimensions differ");
                for (size_t index = 0; index < actual.values.size(); ++index) {
                  Require(actual.values[index] == reference[group].tokens[index].value &&
                            actual.contexts[index] == reference[group].tokens[index].context,
                          "Direct token differs from independent template path");
                }
                if (collect) {
                  CheckPopulations(actual, expected);
                } else {
                  Require(actual.context_populations.empty() &&
                            actual.symbol_populations.empty(),
                          "Disabled population collection retained output");
                }
                ++cases;
              }
            }
          }
        }
        CheckRejectedGroup(frame, natural, maps.front(), &scratch);
      }
    }
    Require(cases == 4096, "Direct tokenization coverage is incomplete");
    std::cout << "Verified " << cases
              << " direct AC group cases with exact tokens and populations.\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
