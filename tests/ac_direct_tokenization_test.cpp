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
      for (size_t pattern = 0; pattern < 6; ++pattern) {
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
    Require(cases == 3072, "Direct tokenization coverage is incomplete");
    std::cout << "Verified " << cases
              << " direct AC group cases with exact tokens and populations.\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
