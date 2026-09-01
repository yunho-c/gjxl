// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

#include "codestream/ans_internal.h"
#include "codestream/entropy.h"
#include "codestream/entropy_internal.h"
#include "codestream/huffman.h"

namespace {

template <size_t Size>
bool HasBytes(
  const gjxl::BitWriter& writer,
  const std::array<uint8_t, Size>& expected) {

  return std::ranges::equal(writer.padded_bytes(), expected);
}

bool CheckSignedPacking() {
  return
    gjxl::PackSigned(std::numeric_limits<int32_t>::min()) ==
      std::numeric_limits<uint32_t>::max() &&
    gjxl::PackSigned(-1) == 1 &&
    gjxl::PackSigned(0) == 0 &&
    gjxl::PackSigned(1) == 2 &&
    gjxl::PackSigned(std::numeric_limits<int32_t>::max()) ==
      std::numeric_limits<uint32_t>::max() - 1;
}

bool CheckDeterministicHuffmanScratch() {
  constexpr std::array<uint64_t, 5> equal_counts = {1, 1, 1, 1, 1};
  constexpr std::array<uint8_t, 5> expected_depths = {2, 2, 2, 3, 3};
  constexpr std::array<uint16_t, 5> expected_bits = {0, 2, 1, 3, 7};
  std::array<uint8_t, equal_counts.size()> depths{};
  std::array<uint16_t, equal_counts.size()> bits{};
  if (!gjxl::codestream_internal::CreateHuffmanTree(
        equal_counts, 15, depths).ok() ||
      depths != expected_depths ||
      !gjxl::codestream_internal::ConvertBitDepthsToSymbols(
        depths, bits).ok() ||
      bits != expected_bits) {
    std::cerr << "Equal-count Huffman ordering changed\n";
    return false;
  }

  constexpr std::array<uint64_t, 8> limited_counts = {
    1000, 1, 1, 1, 1, 1, 1, 1};
  constexpr std::array<uint8_t, 8> limited_expected = {
    3, 3, 3, 3, 3, 3, 3, 3};
  std::array<uint8_t, limited_counts.size()> limited_depths{};
  if (!gjxl::codestream_internal::CreateHuffmanTree(
        limited_counts, 3, limited_depths).ok() ||
      limited_depths != limited_expected) {
    std::cerr << "Depth-limited Huffman retry changed\n";
    return false;
  }
  return true;
}

bool CheckHybridUintBoundaries() {
  gjxl::HybridUintToken token;
  if (!gjxl::EncodeHybridUint(
        0, gjxl::kDefaultHybridUintConfig, &token).ok() ||
      token != gjxl::HybridUintToken{0, 0, 0}) {
    std::cerr << "HybridUint zero encoding is incorrect\n";
    return false;
  }

  for (uint32_t exponent = 4; exponent <= 31; ++exponent) {
    const uint32_t power = uint32_t{1} << exponent;
    if (!gjxl::EncodeHybridUint(
          power, gjxl::kDefaultHybridUintConfig, &token).ok() ||
        token.symbol != 4 * exponent ||
        token.extra_bit_count != exponent - 2 ||
        token.extra_bits != 0) {
      std::cerr << "HybridUint exponent transition failed at "
                << exponent << '\n';
      return false;
    }

    const uint32_t below = power - 1;
    const uint32_t expected_symbol = exponent == 4 ? 15 : 4 * exponent - 1;
    const uint8_t expected_extra_count =
      exponent == 4 ? 0 : static_cast<uint8_t>(exponent - 3);
    const uint32_t expected_extra_bits = exponent == 4
      ? 0
      : (uint32_t{1} << (exponent - 3)) - 1;
    if (!gjxl::EncodeHybridUint(
          below, gjxl::kDefaultHybridUintConfig, &token).ok() ||
        token.symbol != expected_symbol ||
        token.extra_bit_count != expected_extra_count ||
        token.extra_bits != expected_extra_bits) {
      std::cerr << "HybridUint pre-transition failed at "
                << exponent << '\n';
      return false;
    }
  }

  if (!gjxl::EncodeHybridUint(
        std::numeric_limits<uint16_t>::max(),
        gjxl::kDefaultHybridUintConfig,
        &token).ok() ||
      token != gjxl::HybridUintToken{63, 13, 0x1FFF} ||
      !gjxl::EncodeHybridUint(
        std::numeric_limits<uint32_t>::max(),
        gjxl::kDefaultHybridUintConfig,
        &token).ok() ||
      token != gjxl::HybridUintToken{127, 29, 0x1FFFFFFF} ||
      gjxl::EncodeHybridUint(
        1, {2, 2, 1}, &token).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      gjxl::EncodeHybridUint(
        1, gjxl::kDefaultHybridUintConfig, nullptr).code() !=
        gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "HybridUint limit handling is incorrect\n";
    return false;
  }
  return true;
}

std::vector<gjxl::EntropyToken> ComplexFixtureTokens() {
  std::vector<gjxl::EntropyToken> tokens;
  for (uint32_t value = 0; value <= 20; ++value) {
    for (uint32_t repeat = value; repeat <= 20; ++repeat) {
      tokens.push_back({0, value});
    }
  }
  return tokens;
}

std::vector<gjxl::EntropyToken> ContextFixtureTokens() {
  std::vector<gjxl::EntropyToken> tokens;
  for (uint32_t context = 0; context < 4; ++context) {
    const uint32_t first = context * 4;
    for (size_t repeat = 0; repeat < 100; ++repeat) {
      tokens.push_back({context, first});
      tokens.push_back({context, first + 1});
    }
  }
  return tokens;
}

bool CheckDeterministicEntropyFixtures() {
  std::vector<gjxl::EntropyToken> complex = ComplexFixtureTokens();
  const std::array<std::vector<gjxl::EntropyToken>, 1> complex_sections = {
    complex};
  gjxl::EntropyCode complex_code;
  gjxl::BitWriter complex_writer;
  gjxl::BitWriter complex_repeat;
  if (!gjxl::OptimizeEntropyCode(
        complex_sections, {.context_count = 1}, &complex_code).ok() ||
      !gjxl::WriteEntropyCode(complex_code, &complex_writer).ok() ||
      !gjxl::WriteEntropyCode(complex_code, &complex_repeat).ok() ||
      complex_writer.bits_written() == 0 ||
      complex_writer.bits_written() != complex_repeat.bits_written() ||
      !std::ranges::equal(
        complex_writer.padded_bytes(), complex_repeat.padded_bytes())) {
    std::cerr << "Complex Huffman fixture is not deterministic\n";
    return false;
  }

  std::vector<gjxl::EntropyToken> contexts = ContextFixtureTokens();
  const std::array<std::vector<gjxl::EntropyToken>, 1> context_sections = {
    contexts};
  gjxl::EntropyCode context_code;
  gjxl::BitWriter map_writer;
  gjxl::BitWriter entropy_writer;
  gjxl::EntropyCodeCost context_cost;
  if (!gjxl::OptimizeEntropyCode(
        context_sections, {.context_count = 4}, &context_code,
        &context_cost).ok() ||
      context_code.context_map != std::vector<uint8_t>({0, 1, 2, 3}) ||
      !gjxl::WriteContextMap(context_code, &map_writer).ok() ||
      !gjxl::WriteEntropyCode(context_code, &entropy_writer).ok() ||
      context_cost.model_bits != entropy_writer.bits_written() ||
      map_writer.bits_written() == 0 || entropy_writer.bits_written() == 0) {
    std::cerr << "Context-map fixture is invalid\n";
    return false;
  }
  return true;
}

bool CheckUintConfigSerialization() {
  std::array<uint64_t, gjxl::kPrefixAlphabetSize> counts{};
  counts[0] = 1;
  counts[1] = 1;
  gjxl::PrefixCode prefix;
  if (!gjxl::BuildPrefixCode(counts, &prefix).ok()) {
    return false;
  }
  const std::array<gjxl::PrefixCode, 1> prefixes = {prefix};
  constexpr std::array configs = {
    gjxl::HybridUintConfig{4, 2, 0},
    gjxl::HybridUintConfig{4, 1, 2},
    gjxl::HybridUintConfig{0, 0, 0},
    gjxl::HybridUintConfig{2, 0, 1},
  };
  constexpr std::array<size_t, configs.size()> expected_bits = {
    21, 21, 16, 20};
  for (size_t index = 0; index < configs.size(); ++index) {
    gjxl::BitWriter writer;
    const std::array<gjxl::HybridUintConfig, 1> selected = {configs[index]};
    if (!gjxl::WritePrefixCodes(prefixes, selected, &writer).ok() ||
        writer.bits_written() != expected_bits[index]) {
      std::cerr << "HybridUint configuration serialization failed at "
                << index << ": " << writer.bits_written() << " bits\n";
      return false;
    }
  }

  gjxl::BitWriter atomic;
  if (!atomic.WriteBits(3, 5).ok()) {
    return false;
  }
  const std::array<gjxl::HybridUintConfig, 0> missing;
  if (gjxl::WritePrefixCodes(prefixes, missing, &atomic).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      atomic.bits_written() != 3 ||
      !HasBytes(atomic, std::array<uint8_t, 1>{5})) {
    std::cerr << "Rejected HybridUint vector changed its destination\n";
    return false;
  }
  return true;
}

bool CheckDegeneratePrefixPayload() {
  constexpr gjxl::HybridUintConfig config{2, 0, 1};
  constexpr uint32_t value = 9;
  constexpr size_t repetitions = 5;
  gjxl::HybridUintToken encoded;
  if (!gjxl::EncodeHybridUint(value, config, &encoded).ok() ||
      encoded.extra_bit_count == 0 ||
      encoded.symbol >= gjxl::kPrefixAlphabetSize) {
    std::cerr << "Degenerate-prefix fixture could not encode its value\n";
    return false;
  }

  std::array<uint64_t, gjxl::kPrefixAlphabetSize> counts{};
  counts[encoded.symbol] = repetitions;
  gjxl::PrefixCode prefix;
  if (!gjxl::BuildPrefixCode(counts, &prefix).ok() ||
      prefix.degenerate_symbol != encoded.symbol ||
      prefix.depths[encoded.symbol] != 1) {
    std::cerr << "Degenerate prefix was not identified\n";
    return false;
  }

  gjxl::EntropyCode code{
    .context_count = 1,
    .context_map = {0},
    .uint_configs = {config},
    .prefix_codes = {prefix},
  };
  std::vector<gjxl::EntropyToken> tokens(
    repetitions, gjxl::EntropyToken{0, value});
  gjxl::BitWriter model;
  gjxl::BitWriter payload;
  if (!gjxl::WriteEntropyCode(code, &model).ok() ||
      !gjxl::WriteTokenStream(tokens, code, &payload).ok() ||
      model.bits_written() == 0 ||
      payload.bits_written() != repetitions * encoded.extra_bit_count) {
    std::cerr << "Degenerate prefix emitted a token-prefix bit\n";
    return false;
  }
  return true;
}

bool CheckBalancedOptimization() {
  constexpr size_t kContexts = 16;
  std::array<uint32_t, 2 * kContexts> values{};
  std::array<bool, 2 * kContexts> found{};
  size_t found_count = 0;
  for (uint32_t value = 0; found_count < found.size(); ++value) {
    gjxl::HybridUintToken encoded;
    if (!gjxl::EncodeHybridUint(
          value, gjxl::kDefaultHybridUintConfig, &encoded).ok()) {
      return false;
    }
    if (encoded.symbol < found.size() && !found[encoded.symbol]) {
      values[encoded.symbol] = value;
      found[encoded.symbol] = true;
      ++found_count;
    }
  }
  std::vector<gjxl::EntropyToken> tokens;
  for (uint32_t context = 0; context < kContexts; ++context) {
    for (size_t repeat = 0; repeat < 256; ++repeat) {
      tokens.push_back({context, values[2 * context + (repeat & 1u)]});
    }
  }
  const std::array<std::vector<gjxl::EntropyToken>, 1> sections = {tokens};
  gjxl::EntropyCode code;
  gjxl::EntropyCodeCost cost;
  if (!gjxl::OptimizeEntropyCode(
        sections, {.context_count = kContexts}, &code, &cost).ok() ||
      code.prefix_codes.size() <= 8 ||
      code.prefix_codes.size() > gjxl::kMaximumPrefixClusters ||
      code.uint_configs.size() != code.prefix_codes.size() ||
      cost.cluster_count != code.prefix_codes.size()) {
    std::cerr << "Balanced cluster search did not use the extended range\n";
    std::cerr << "Selected " << code.prefix_codes.size()
              << " clusters with " << cost.model_bits << " model bits and "
              << cost.token_bits << " token bits\n";
    return false;
  }
  const bool has_nondefault = std::ranges::any_of(
    code.uint_configs,
    [](gjxl::HybridUintConfig config) {
      return config != gjxl::kDefaultHybridUintConfig;
    });
  gjxl::BitWriter model;
  gjxl::BitWriter payload;
  if (!has_nondefault || !gjxl::WriteEntropyCode(code, &model).ok() ||
      !gjxl::WriteTokenStream(tokens, code, &payload).ok() ||
      cost.model_bits != model.bits_written() ||
      cost.token_bits != payload.bits_written()) {
    std::cerr << "Balanced entropy cost attribution is incorrect\n";
    return false;
  }

  const gjxl::EntropyCode unchanged = code;
  const gjxl::EntropyCodeCost sentinel{11, 22, 33};
  gjxl::EntropyCodeCost rejected_cost = sentinel;
  const std::array<std::vector<gjxl::EntropyToken>, 1> invalid_sections = {
    std::vector<gjxl::EntropyToken>{{kContexts, 0}}};
  if (gjxl::OptimizeEntropyCode(
        invalid_sections, {.context_count = kContexts}, &code,
        &rejected_cost).code() != gjxl::StatusCode::kInvalidArgument ||
      code != unchanged || rejected_cost != sentinel) {
    std::cerr << "Rejected balanced optimization changed its outputs\n";
    return false;
  }
  return true;
}

bool CheckCountedPrefixOptimization() {
  constexpr uint32_t kContextCount = 4;
  // Every context contributes more than the 4096-occurrence threshold so the
  // selected clusters exercise counted aggregation even when none are merged.
  constexpr size_t kOccurrencesPerContext = 4352;
  std::array<std::vector<gjxl::EntropyToken>, 5> sections;
  constexpr std::array<size_t, 3> kPopulatedSections = {1, 2, 4};
  for (uint32_t context = 0; context < kContextCount; ++context) {
    for (size_t repeat = 0; repeat < kOccurrencesPerContext; ++repeat) {
      uint32_t value = 0;
      switch (context) {
        case 0:
          value = (repeat & 15u) == 0 ? 1 : 0;
          break;
        case 1:
          value = (repeat & 7u) == 0 ? 17 : 16;
          break;
        case 2:
          value = (repeat & 31u) == 0 ? 65537 : 257;
          break;
        case 3:
          value = (repeat & 63u) == 0
            ? std::numeric_limits<uint32_t>::max() - 1
            : std::numeric_limits<uint32_t>::max();
          break;
      }
      sections[kPopulatedSections[(repeat + context) %
                                  kPopulatedSections.size()]]
        .push_back({context, value});
    }
  }

  gjxl::EntropyCode code;
  gjxl::EntropyCode repeat_code;
  gjxl::EntropyCodeCost cost;
  gjxl::EntropyCodeCost repeat_cost;
  const gjxl::EntropyCodeOptions options{.context_count = kContextCount};
  if (!gjxl::OptimizeEntropyCode(sections, options, &code, &cost).ok() ||
      !gjxl::OptimizeEntropyCode(
        sections, options, &repeat_code, &repeat_cost).ok() ||
      code != repeat_code || cost != repeat_cost ||
      code.mode != gjxl::EntropyCodingMode::kPrefix ||
      code.context_map.size() != kContextCount ||
      code.prefix_codes.size() < 2 ||
      cost.cluster_count != code.prefix_codes.size()) {
    std::cerr << "Counted prefix selection is invalid or non-deterministic\n";
    return false;
  }

  gjxl::BitWriter model;
  gjxl::BitWriter repeat_model;
  gjxl::BitWriter payload;
  gjxl::BitWriter repeat_payload;
  if (!gjxl::WriteEntropyCode(code, &model).ok() ||
      !gjxl::WriteEntropyCode(repeat_code, &repeat_model).ok()) {
    std::cerr << "Counted prefix model serialization failed\n";
    return false;
  }
  for (const std::vector<gjxl::EntropyToken>& section : sections) {
    if (!gjxl::WriteTokenStream(section, code, &payload).ok() ||
        !gjxl::WriteTokenStream(
          section, repeat_code, &repeat_payload).ok()) {
      std::cerr << "Counted prefix token serialization failed\n";
      return false;
    }
  }
  if (cost.model_bits != model.bits_written() ||
      cost.token_bits != payload.bits_written() ||
      model.bits_written() != repeat_model.bits_written() ||
      payload.bits_written() != repeat_payload.bits_written() ||
      !std::ranges::equal(
        model.padded_bytes(), repeat_model.padded_bytes()) ||
      !std::ranges::equal(
        payload.padded_bytes(), repeat_payload.padded_bytes())) {
    std::cerr << "Counted prefix cost disagrees with serialized output\n";
    return false;
  }
  const auto hash = [](std::span<const uint8_t> bytes) {
    uint64_t result = 1469598103934665603ull;
    for (uint8_t byte : bytes) {
      result ^= byte;
      result *= 1099511628211ull;
    }
    return result;
  };
  const std::vector<gjxl::HybridUintConfig> expected_configs = {
    {0, 0, 0}, {4, 2, 0}, {4, 1, 2}, {4, 2, 0}};
  if (code.context_map != std::vector<uint8_t>({0, 1, 2, 3}) ||
      code.uint_configs != expected_configs || model.bits_written() != 144 ||
      payload.bits_written() != 166464 ||
      hash(model.padded_bytes()) != 5815996224897546142ull ||
      hash(payload.padded_bytes()) != 6576315826512740406ull) {
    std::cerr << "Counted prefix decision or serialized bytes changed\n";
    return false;
  }
  return true;
}

bool CheckFullWidthMultiSectionOptimization() {
  std::array<std::vector<gjxl::EntropyToken>, 3> sections;
  sections[0] = {{299, std::numeric_limits<uint32_t>::max()}, {0, 0}};
  sections[1] = {{299, std::numeric_limits<uint32_t>::max() - 1}};

  gjxl::EntropyCode code;
  if (!gjxl::OptimizeEntropyCode(
        sections, {.context_count = 300}, &code).ok() ||
      code.context_count != 300 || code.context_map.size() != 300 ||
      code.prefix_codes.empty()) {
    std::cerr << "Multi-section entropy optimization failed\n";
    return false;
  }
  gjxl::BitWriter tokens;
  if (!gjxl::WriteTokenStream(sections[0], code, &tokens).ok() ||
      tokens.bits_written() == 0) {
    std::cerr << "Full-width token stream was not encoded\n";
    return false;
  }

  const gjxl::EntropyCode unchanged = code;
  const std::array<std::vector<gjxl::EntropyToken>, 1> invalid_sections = {
    std::vector<gjxl::EntropyToken>{{300, 0}}};
  if (gjxl::OptimizeEntropyCode(
        invalid_sections, {.context_count = 300}, &code).code() !=
        gjxl::StatusCode::kInvalidArgument || code != unchanged) {
    std::cerr << "Rejected entropy optimization changed its output\n";
    return false;
  }

  gjxl::BitWriter unchanged_writer;
  if (!unchanged_writer.WriteBits(3, 5).ok()) {
    return false;
  }
  const std::array<gjxl::EntropyToken, 1> invalid_tokens = {{{300, 0}}};
  if (gjxl::WriteTokenStream(
        invalid_tokens, unchanged, &unchanged_writer).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      unchanged_writer.bits_written() != 3 ||
      !HasBytes(unchanged_writer, std::array<uint8_t, 1>{5})) {
    std::cerr << "Rejected token stream changed its destination\n";
    return false;
  }

  gjxl::EntropyCode malformed = unchanged;
  bool changed_prefix_bit = false;
  for (gjxl::PrefixCode& prefix : malformed.prefix_codes) {
    for (size_t symbol = 0; symbol < prefix.depths.size(); ++symbol) {
      if (prefix.depths[symbol] != 0) {
        prefix.bits[symbol] ^= 1;
        changed_prefix_bit = true;
        break;
      }
    }
    if (changed_prefix_bit) {
      break;
    }
  }
  if (!changed_prefix_bit ||
      gjxl::WriteEntropyCode(malformed, &unchanged_writer).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      unchanged_writer.bits_written() != 3 ||
      !HasBytes(unchanged_writer, std::array<uint8_t, 1>{5})) {
    std::cerr << "Malformed prefix code changed its destination\n";
    return false;
  }

  const std::array<std::vector<gjxl::EntropyToken>, 2> empty_sections;
  gjxl::EntropyCode empty_code;
  gjxl::BitWriter empty_writer;
  if (!gjxl::OptimizeEntropyCode(
        empty_sections, {.context_count = 3}, &empty_code).ok() ||
      empty_code.context_map != std::vector<uint8_t>({0, 0, 0}) ||
      empty_code.prefix_codes.size() != 1 ||
      !gjxl::WriteEntropyCode(empty_code, &empty_writer).ok()) {
    std::cerr << "Empty entropy streams are not deterministic\n";
    return false;
  }
  return true;
}

bool CheckInitialContextPreclustering() {
  std::vector<gjxl::EntropyToken> tokens = ContextFixtureTokens();
  const std::array<std::vector<gjxl::EntropyToken>, 1> sections = {tokens};
  const std::array<uint8_t, 4> initial_map = {0, 0, 1, 1};
  gjxl::EntropyCode code;
  if (!gjxl::OptimizeEntropyCode(
        sections,
        {
          .context_count = 4,
          .initial_context_map = initial_map,
          .initial_histogram_count = 2,
        },
        &code).ok() ||
      code.context_map != std::vector<uint8_t>({0, 0, 1, 1}) ||
      code.prefix_codes.size() != 2) {
    std::cerr << "Initial entropy context map was not composed correctly\n";
    return false;
  }

  const gjxl::EntropyCode unchanged = code;
  const std::array<uint8_t, 4> invalid_map = {0, 0, 1, 2};
  if (gjxl::OptimizeEntropyCode(
        sections,
        {
          .context_count = 4,
          .initial_context_map = invalid_map,
          .initial_histogram_count = 2,
        },
        &code).code() != gjxl::StatusCode::kInvalidArgument ||
      code != unchanged) {
    std::cerr << "Invalid initial context map changed its output\n";
    return false;
  }
  return true;
}

bool CheckAnsFrequencyReciprocalDivision() {
  if (gjxl::codestream_internal::AnsFrequencyReciprocal(0) != 0) {
    std::cerr << "Absent ANS symbol has a nonzero reciprocal\n";
    return false;
  }
  uint32_t random = 0x474A584Cu;
  for (uint32_t frequency = 1; frequency <= gjxl::kAnsTableSize;
       ++frequency) {
    const uint64_t reciprocal =
      gjxl::codestream_internal::AnsFrequencyReciprocal(
        static_cast<uint16_t>(frequency));
    const uint64_t expected_reciprocal =
      ((uint64_t{1} << gjxl::codestream_internal::kAnsReciprocalPrecision) -
       1) /
        frequency +
      1;
    if (reciprocal != expected_reciprocal) {
      std::cerr << "ANS reciprocal differs at frequency "
                << frequency << '\n';
      return false;
    }
    const uint32_t maximum_state = static_cast<uint32_t>(
      std::min<uint64_t>(
        std::numeric_limits<uint32_t>::max(),
        (static_cast<uint64_t>(frequency) << 20) - 1));
    const auto verify = [&](uint32_t state) {
      if (state != 0 && reciprocal >
          std::numeric_limits<uint64_t>::max() / state) {
        std::cerr << "ANS reciprocal product overflows at frequency "
                  << frequency << " state " << state << '\n';
        return false;
      }
      const uint32_t quotient =
        gjxl::codestream_internal::DivideAnsStateByReciprocal(
          state, reciprocal);
      if (quotient != state / frequency) {
        std::cerr << "ANS reciprocal quotient differs at frequency "
                  << frequency << " state " << state << '\n';
        return false;
      }
      return true;
    };
    const std::array<uint32_t, 6> boundary_states = {
      0,
      1,
      std::min(frequency - 1, maximum_state),
      std::min(frequency, maximum_state),
      std::min(frequency + 1, maximum_state),
      maximum_state,
    };
    for (uint32_t state : boundary_states) {
      if (!verify(state)) return false;
    }
    const uint32_t last_multiple =
      maximum_state - maximum_state % frequency;
    for (int32_t offset = -8; offset <= 8; ++offset) {
      const int64_t state = static_cast<int64_t>(last_multiple) + offset;
      if (state >= 0 && state <= maximum_state &&
          !verify(static_cast<uint32_t>(state))) {
        return false;
      }
    }
    for (size_t sample = 0; sample < 256; ++sample) {
      random = random * 1664525u + 1013904223u;
      const uint32_t state = static_cast<uint32_t>(
        (static_cast<uint64_t>(random) *
         (static_cast<uint64_t>(maximum_state) + 1)) >>
        32);
      if (!verify(state)) return false;
    }
  }
  return true;
}

bool CheckAnsRoundTripContract() {
  std::array<std::vector<gjxl::EntropyToken>, 4> sections;
  for (uint32_t context = 0; context < 4; ++context) {
    for (uint32_t repeat = 0; repeat < 512; ++repeat) {
      uint32_t value = 4 * context + (repeat & 1u);
      if ((repeat & 31u) == 0) {
        value = (uint32_t{1} << (16 + context)) + repeat;
      }
      sections[1 + ((repeat + context) % 3)].push_back({context, value});
    }
  }
  gjxl::EntropyCode prefix;
  gjxl::EntropyCode ans;
  gjxl::EntropyCodeCost ans_cost;
  gjxl::BitWriter model;
  gjxl::BitWriter repeat_model;
  if (!gjxl::OptimizeEntropyCode(
        sections, {.context_count = 4}, &prefix).ok() ||
      !gjxl::OptimizeAnsEntropyCode(
        sections, prefix, &ans, &ans_cost).ok() ||
      ans.mode != gjxl::EntropyCodingMode::kAns ||
      ans.ans_log_alpha_size < 5 || ans.ans_log_alpha_size > 8 ||
      ans.ans_histograms.size() < 2 || !ans.prefix_codes.empty() ||
      !gjxl::WriteEntropyCode(ans, &model).ok() ||
      !gjxl::WriteEntropyCode(ans, &repeat_model).ok() ||
      ans_cost.model_bits != model.bits_written() ||
      ans_cost.cluster_count != ans.ans_histograms.size() ||
      model.bits_written() == 0 ||
      model.bits_written() != repeat_model.bits_written() ||
      !std::ranges::equal(
        model.padded_bytes(), repeat_model.padded_bytes())) {
    std::cerr << "ANS model or payload contract failed\n";
    return false;
  }

  uint64_t serialized_bits = 0;
  uint64_t bits_without_renormalization = 32 * sections.size();
  bool has_extra_bits = false;
  for (const std::vector<gjxl::EntropyToken>& section : sections) {
    gjxl::BitWriter payload;
    gjxl::BitWriter repeat_payload;
    if (!gjxl::WriteTokenStream(section, ans, &payload).ok() ||
        !gjxl::WriteTokenStream(section, ans, &repeat_payload).ok() ||
        payload.bits_written() != repeat_payload.bits_written() ||
        !std::ranges::equal(
          payload.padded_bytes(), repeat_payload.padded_bytes()) ||
        (section.empty() && payload.bits_written() != 32)) {
      std::cerr << "ANS section payload contract failed\n";
      return false;
    }
    serialized_bits += payload.bits_written();
    for (const gjxl::EntropyToken& token : section) {
      const size_t cluster = ans.context_map[token.context];
      gjxl::HybridUintToken encoded;
      if (!gjxl::EncodeHybridUint(
            token.value, ans.uint_configs[cluster], &encoded).ok()) {
        return false;
      }
      bits_without_renormalization += encoded.extra_bit_count;
      has_extra_bits |= encoded.extra_bit_count != 0;
    }
  }
  if (!has_extra_bits || serialized_bits <= bits_without_renormalization ||
      ans_cost.token_bits != serialized_bits) {
    std::cerr << "ANS count-only cost disagrees with serialized sections\n";
    return false;
  }

  gjxl::EntropyCode malformed = ans;
  bool damaged = false;
  for (gjxl::AnsHistogram& histogram : malformed.ans_histograms) {
    for (std::vector<uint16_t>& reverse : histogram.reverse_maps) {
      if (!reverse.empty()) {
        reverse[0] = gjxl::kAnsTableSize;
        damaged = true;
        break;
      }
    }
    if (damaged) break;
  }
  gjxl::BitWriter atomic;
  if (!damaged || !atomic.WriteBits(3, 5).ok() ||
      gjxl::WriteTokenStream(sections[1], malformed, &atomic).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      atomic.bits_written() != 3 ||
      !HasBytes(atomic, std::array<uint8_t, 1>{5})) {
    std::cerr << "Malformed ANS lookup changed its destination\n";
    return false;
  }

  const auto rejects_reciprocals = [&](bool remove) {
    gjxl::EntropyCode invalid = ans;
    bool changed = false;
    for (gjxl::AnsHistogram& histogram : invalid.ans_histograms) {
      if (histogram.reciprocal_frequencies.empty()) continue;
      if (remove) {
        histogram.reciprocal_frequencies.clear();
      } else {
        ++histogram.reciprocal_frequencies.front();
      }
      changed = true;
      break;
    }
    gjxl::BitWriter destination;
    return changed && destination.WriteBits(3, 5).ok() &&
      gjxl::WriteTokenStream(
        sections[1], invalid, &destination).code() ==
        gjxl::StatusCode::kInvalidArgument &&
      destination.bits_written() == 3 &&
      HasBytes(destination, std::array<uint8_t, 1>{5});
  };
  if (!rejects_reciprocals(false) || !rejects_reciprocals(true)) {
    std::cerr << "Malformed ANS reciprocal changed its destination\n";
    return false;
  }
  return true;
}

bool CheckAnsAdaptiveModelSelection() {
  std::vector<gjxl::EntropyToken> skewed;
  for (uint32_t value = 0; value < 32; ++value) {
    const size_t repetitions = value < 10 ? size_t{1024} >> value : 1;
    for (size_t repeat = 0; repeat < repetitions; ++repeat) {
      skewed.push_back({0, value});
    }
  }
  const std::array<std::vector<gjxl::EntropyToken>, 1> skewed_sections = {
    skewed};
  gjxl::EntropyCode skewed_prefix;
  gjxl::EntropyCode skewed_ans;
  if (!gjxl::OptimizeEntropyCode(
        skewed_sections, {.context_count = 1}, &skewed_prefix).ok() ||
      !gjxl::OptimizeAnsEntropyCode(
        skewed_sections, skewed_prefix, &skewed_ans).ok() ||
      skewed_ans.uint_configs !=
        std::vector<gjxl::HybridUintConfig>{{3, 1, 0}} ||
      skewed_ans.ans_log_alpha_size != 5 ||
      skewed_ans.ans_histograms.size() != 1 ||
      skewed_ans.ans_histograms[0].method != 1 ||
      skewed_ans.ans_histograms[0].frequencies != std::vector<uint16_t>{
        2008, 1024, 512, 256, 128, 64, 32, 16, 16, 8, 16, 16,
      }) {
    std::cerr << "ANS precision/config selection fixture failed\n";
    return false;
  }

  std::vector<gjxl::EntropyToken> sparse(200, {0, 0});
  sparse.insert(sparse.end(), 100, {0, 1});
  sparse.push_back({0, std::numeric_limits<uint32_t>::max()});
  const std::array<std::vector<gjxl::EntropyToken>, 1> sparse_sections = {
    sparse};
  gjxl::EntropyCode sparse_prefix;
  gjxl::EntropyCode sparse_ans;
  gjxl::BitWriter sparse_model;
  gjxl::BitWriter sparse_payload;
  if (!gjxl::OptimizeEntropyCode(
        sparse_sections, {.context_count = 1}, &sparse_prefix).ok() ||
      !gjxl::OptimizeAnsEntropyCode(
        sparse_sections, sparse_prefix, &sparse_ans).ok() ||
      sparse_ans.uint_configs !=
        std::vector<gjxl::HybridUintConfig>{{0, 0, 0}} ||
      sparse_ans.ans_log_alpha_size != 6 ||
      sparse_ans.ans_histograms.size() != 1 ||
      sparse_ans.ans_histograms[0].method != 3 ||
      sparse_ans.ans_histograms[0].frequencies.size() != 33 ||
      sparse_ans.ans_histograms[0].frequencies[0] != 2544 ||
      sparse_ans.ans_histograms[0].frequencies[1] != 1536 ||
      sparse_ans.ans_histograms[0].frequencies.back() != 16 ||
      !std::ranges::all_of(
        sparse_ans.ans_histograms[0].frequencies.begin() + 2,
        sparse_ans.ans_histograms[0].frequencies.end() - 1,
        [](uint16_t frequency) { return frequency == 0; }) ||
      !gjxl::WriteEntropyCode(sparse_ans, &sparse_model).ok() ||
      !gjxl::WriteTokenStream(sparse, sparse_ans, &sparse_payload).ok() ||
      sparse_model.bits_written() != 59 ||
      sparse_payload.bits_written() < 32) {
    std::cerr << "ANS sparse/RLE selection fixture failed\n";
    return false;
  }

  std::vector<gjxl::EntropyToken> repeated_sparse;
  repeated_sparse.reserve(sparse.size() * 32);
  for (size_t repeat = 0; repeat < 32; ++repeat) {
    repeated_sparse.insert(
      repeated_sparse.end(), sparse.begin(), sparse.end());
  }
  const std::array<std::vector<gjxl::EntropyToken>, 1>
    repeated_sparse_sections = {repeated_sparse};
  gjxl::EntropyCode repeated_sparse_prefix;
  gjxl::EntropyCode repeated_sparse_ans;
  gjxl::EntropyCodeCost repeated_sparse_cost;
  gjxl::BitWriter repeated_sparse_model;
  gjxl::BitWriter repeated_sparse_payload;
  if (!gjxl::OptimizeEntropyCode(
        repeated_sparse_sections, {.context_count = 1},
        &repeated_sparse_prefix).ok() ||
      !gjxl::OptimizeAnsEntropyCode(
        repeated_sparse_sections, repeated_sparse_prefix,
        &repeated_sparse_ans, &repeated_sparse_cost).ok() ||
      !gjxl::WriteEntropyCode(
        repeated_sparse_ans, &repeated_sparse_model).ok() ||
      !gjxl::WriteTokenStream(
        repeated_sparse, repeated_sparse_ans,
        &repeated_sparse_payload).ok() ||
      repeated_sparse_cost.model_bits !=
        repeated_sparse_model.bits_written() ||
      repeated_sparse_cost.token_bits !=
        repeated_sparse_payload.bits_written()) {
    std::cerr << "ANS counted-value aggregation fixture failed\n";
    return false;
  }

  gjxl::EntropyCode malformed = skewed_ans;
  malformed.ans_histograms[0].method = 0;
  gjxl::BitWriter atomic;
  if (!atomic.WriteBits(3, 5).ok() ||
      gjxl::WriteEntropyCode(malformed, &atomic).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      atomic.bits_written() != 3 ||
      !HasBytes(atomic, std::array<uint8_t, 1>{5})) {
    std::cerr << "Malformed ANS representation changed its destination\n";
    return false;
  }
  return true;
}

bool CheckAnsSmallHistograms() {
  const std::array<std::vector<gjxl::EntropyToken>, 2> fixtures = {{
    std::vector<gjxl::EntropyToken>(32, {0, 7}),
    {{0, 3}, {0, 11}, {0, 3}, {0, 11}},
  }};
  for (size_t index = 0; index < fixtures.size(); ++index) {
    const std::array<std::vector<gjxl::EntropyToken>, 1> sections = {
      fixtures[index]};
    gjxl::EntropyCode prefix;
    gjxl::EntropyCode ans;
    gjxl::EntropyCodeCost ans_cost;
    gjxl::BitWriter model;
    gjxl::BitWriter payload;
    if (!gjxl::OptimizeEntropyCode(
          sections, {.context_count = 1}, &prefix).ok() ||
        !gjxl::OptimizeAnsEntropyCode(
          sections, prefix, &ans, &ans_cost).ok() ||
        ans.ans_histograms.size() != 1 ||
        std::ranges::count_if(
          ans.ans_histograms[0].frequencies,
          [](uint16_t frequency) { return frequency != 0; }) != index + 1 ||
        !gjxl::WriteEntropyCode(ans, &model).ok() ||
        !gjxl::WriteTokenStream(fixtures[index], ans, &payload).ok() ||
        ans_cost.model_bits != model.bits_written() ||
        ans_cost.token_bits != payload.bits_written()) {
      std::cerr << "ANS small-histogram fixture failed at " << index << '\n';
      return false;
    }
  }
  return true;
}

bool CheckExactTokenBitCounting() {
  std::vector<std::vector<gjxl::EntropyToken>> sections = {
    {},
    {{0, 0}, {1, 1}, {0, 17}, {1, UINT32_MAX}},
    {{1, 4}, {0, 4}, {1, 255}, {0, 65536}, {1, UINT32_MAX}},
  };
  gjxl::EntropyCode prefix;
  gjxl::EntropyCodeCost prefix_cost;
  if (!gjxl::OptimizeEntropyCode(
         sections, {.context_count = 2}, &prefix, &prefix_cost).ok()) {
    std::cerr << "Prefix bit-count fixture optimization failed\n";
    return false;
  }
  gjxl::EntropyCode ans;
  gjxl::EntropyCodeCost ans_cost;
  if (!gjxl::OptimizeAnsEntropyCode(
         sections, prefix, &ans, &ans_cost).ok()) {
    std::cerr << "ANS bit-count fixture optimization failed\n";
    return false;
  }

  const std::array<const gjxl::EntropyCode*, 2> codes = {&prefix, &ans};
  const std::array<const gjxl::EntropyCodeCost*, 2> costs = {
    &prefix_cost, &ans_cost};
  for (size_t code_index = 0; code_index < codes.size(); ++code_index) {
    const gjxl::EntropyCode& code = *codes[code_index];
    const gjxl::EntropyCodeCost& cost = *costs[code_index];
    if (cost.section_token_bits.size() != sections.size()) {
      std::cerr << "Retained section token-bit count is incomplete\n";
      return false;
    }
    uint64_t retained_total = 0;
    for (size_t section_index = 0; section_index < sections.size();
         ++section_index) {
      const auto& section = sections[section_index];
      gjxl::BitWriter writer;
      uint64_t measured = std::numeric_limits<uint64_t>::max();
      if (!gjxl::WriteTokenStream(section, code, &writer).ok() ||
          !gjxl::codestream_internal::CountTokenStreamBits(
             section, code, &measured).ok() ||
          measured != writer.bits_written() ||
          measured != cost.section_token_bits[section_index] ||
          (code.mode == gjxl::EntropyCodingMode::kAns && section.empty() &&
           measured != 32) ||
          retained_total > std::numeric_limits<uint64_t>::max() - measured) {
        std::cerr << "Exact token bit count differs from serialization\n";
        return false;
      }
      retained_total += measured;
    }
    if (retained_total != cost.token_bits) {
      std::cerr << "Retained section token bits differ from total cost\n";
      return false;
    }
  }

  const std::array<gjxl::EntropyToken, 1> invalid = {{{2, 0}}};
  uint64_t unchanged = 0xA5A5A5A5A5A5A5A5ull;
  if (gjxl::codestream_internal::CountTokenStreamBits(
        invalid, prefix, &unchanged).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      unchanged != 0xA5A5A5A5A5A5A5A5ull ||
      gjxl::codestream_internal::CountTokenStreamBits(
        sections[1], prefix, nullptr).code() !=
        gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "Rejected token bit count changed its output\n";
    return false;
  }
  gjxl::EntropyCode malformed = prefix;
  malformed.context_map.clear();
  if (gjxl::codestream_internal::CountTokenStreamBits(
        sections[1], malformed, &unchanged).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      unchanged != 0xA5A5A5A5A5A5A5A5ull) {
    std::cerr << "Malformed token bit count changed its output\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckSignedPacking()) {
    std::cerr << "Signed packing is incorrect\n";
    return EXIT_FAILURE;
  }
  if (!CheckHybridUintBoundaries() ||
      !CheckDeterministicHuffmanScratch() ||
      !CheckUintConfigSerialization() ||
      !CheckDegeneratePrefixPayload() ||
      !CheckDeterministicEntropyFixtures() ||
      !CheckBalancedOptimization() ||
      !CheckCountedPrefixOptimization() ||
      !CheckFullWidthMultiSectionOptimization() ||
      !CheckInitialContextPreclustering() ||
      !CheckAnsFrequencyReciprocalDivision() ||
      !CheckAnsRoundTripContract() ||
      !CheckAnsAdaptiveModelSelection() ||
      !CheckAnsSmallHistograms() ||
      !CheckExactTokenBitCounting()) {
    return EXIT_FAILURE;
  }
  std::cout << "All entropy primitive tests passed.\n";
  return EXIT_SUCCESS;
}
