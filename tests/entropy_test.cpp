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

#include "codestream/entropy.h"

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

bool CheckTinyReferenceFixtures() {
  // Generated with local libjxl-tiny f60855353d1345efa361713ca8511a8d45fb5957.
  const std::array<uint8_t, 9> kComplexEntropy = {
    0x49, 0x22, 0x85, 0x1E, 0xC3, 0x01, 0xA0, 0xBA, 0x03};
  const std::array<uint8_t, 5> kContextMap = {
    0x48, 0x62, 0x6C, 0x72, 0xD8};
  const std::array<uint8_t, 19> kContextEntropy = {
    0x48, 0x62, 0x6C, 0x72, 0xD8, 0x49, 0x90, 0x20, 0x41, 0x22,
    0x94, 0x4E, 0x4E, 0xCB, 0x62, 0x0B, 0xB3, 0xB8, 0x01};

  std::vector<gjxl::EntropyToken> complex = ComplexFixtureTokens();
  const std::array<std::vector<gjxl::EntropyToken>, 1> complex_sections = {
    complex};
  gjxl::EntropyCode complex_code;
  gjxl::BitWriter complex_writer;
  if (!gjxl::OptimizeEntropyCode(
        complex_sections, {.context_count = 1}, &complex_code).ok() ||
      !gjxl::WriteEntropyCode(complex_code, &complex_writer).ok() ||
      complex_writer.bits_written() != 66 ||
      !HasBytes(complex_writer, kComplexEntropy)) {
    std::cerr << "Complex Huffman fixture differs from libjxl-tiny\n";
    return false;
  }

  std::vector<gjxl::EntropyToken> contexts = ContextFixtureTokens();
  const std::array<std::vector<gjxl::EntropyToken>, 1> context_sections = {
    contexts};
  gjxl::EntropyCode context_code;
  gjxl::BitWriter map_writer;
  gjxl::BitWriter entropy_writer;
  if (!gjxl::OptimizeEntropyCode(
        context_sections, {.context_count = 4}, &context_code).ok() ||
      context_code.context_map != std::vector<uint8_t>({0, 1, 2, 3}) ||
      !gjxl::WriteContextMap(context_code, &map_writer).ok() ||
      map_writer.bits_written() != 40 ||
      !HasBytes(map_writer, kContextMap) ||
      !gjxl::WriteEntropyCode(context_code, &entropy_writer).ok() ||
      entropy_writer.bits_written() != 145 ||
      !HasBytes(entropy_writer, kContextEntropy)) {
    std::cerr << "Context-map fixture differs from libjxl-tiny\n";
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

}  // namespace

int main() {
  if (!CheckSignedPacking()) {
    std::cerr << "Signed packing is incorrect\n";
    return EXIT_FAILURE;
  }
  if (!CheckHybridUintBoundaries() ||
      !CheckTinyReferenceFixtures() ||
      !CheckFullWidthMultiSectionOptimization() ||
      !CheckInitialContextPreclustering()) {
    return EXIT_FAILURE;
  }
  std::cout << "All entropy primitive tests passed.\n";
  return EXIT_SUCCESS;
}
