// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <bit>
#include <cstdint>
#include <span>
#include <vector>

#include "codestream/entropy.h"

namespace gjxl::codestream_internal {

struct EntropyWorkProfile;

/// Converts a value after the caller has established config.valid(). The
/// value itself is unrestricted: every uint32_t has a HybridUint encoding.
[[nodiscard]] constexpr HybridUintToken EncodeHybridUintValidated(
  uint32_t value, HybridUintConfig config) noexcept {
  HybridUintToken result;
  const uint32_t split_token = uint32_t{1} << config.split_exponent;
  if (value < split_token) {
    result.symbol = value;
  } else {
    const uint32_t exponent =
      31u - static_cast<uint32_t>(std::countl_zero(value));
    const uint32_t mantissa = value - (uint32_t{1} << exponent);
    result.symbol = split_token +
      ((exponent - config.split_exponent) <<
       (config.msb_in_token + config.lsb_in_token)) +
      ((mantissa >> (exponent - config.msb_in_token)) <<
       config.lsb_in_token) +
      (mantissa & ((uint32_t{1} << config.lsb_in_token) - 1));
    result.extra_bit_count = static_cast<uint8_t>(
      exponent - config.msb_in_token - config.lsb_in_token);
    const uint64_t mask =
      (uint64_t{1} << result.extra_bit_count) - 1;
    result.extra_bits = static_cast<uint32_t>(
      (value >> config.lsb_in_token) & mask);
  }
  return result;
}

struct WeightedValue {
  uint32_t value = 0;
  uint64_t count = 0;

  friend bool operator==(const WeightedValue&, const WeightedValue&) = default;
};

/// Fixed-HybridUint ANS statistics retained by the direct balanced partition.
/// These are the final clustered populations, so balanced model construction
/// does not need to collect, sort, and re-encode the original token values.
struct PreparedFixedAnsCluster {
  std::array<uint64_t, kMaximumAnsAlphabetSize> counts{};
  uint64_t token_count = 0;
  uint64_t extra_bits = 0;
  uint32_t maximum_symbol = 0;

  friend bool operator==(
    const PreparedFixedAnsCluster&,
    const PreparedFixedAnsCluster&) = default;
};

/// Raw values aggregated in ascending order for the exact prefix partition.
/// The same cluster populations can feed ANS model construction without
/// collecting and sorting every token a second time.
struct PreparedEntropyClusters {
  uint32_t context_count = 0;
  std::vector<uint8_t> context_map;
  std::vector<std::vector<WeightedValue>> values;
  HybridUintConfig fixed_uint_config = kDefaultHybridUintConfig;
  std::vector<PreparedFixedAnsCluster> fixed_ans_clusters;

  friend bool operator==(
    const PreparedEntropyClusters&,
    const PreparedEntropyClusters&) = default;
};

/// Aggregates raw values in ascending order. Small inputs sort occurrences;
/// large inputs count the bounded dense prefix and sort only sparse values.
[[nodiscard]] Status AggregateEntropyValues(
  std::span<uint32_t> values,
  std::vector<WeightedValue>* aggregated);

/// Owning convenience overload that releases the raw values after aggregation.
[[nodiscard]] Status AggregateEntropyValues(
  std::vector<uint32_t> values,
  std::vector<WeightedValue>* aggregated);

/// Builds the prefix model and retains its exact aggregated cluster values.
/// All outputs remain unchanged on failure.
[[nodiscard]] Status OptimizeEntropyCodeAndPrepareClusters(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCodeOptions& options,
  EntropyCode* code,
  EntropyCodeCost* cost,
  PreparedEntropyClusters* prepared,
  EntropyWorkProfile* profile = nullptr);

/// Builds one fixed-config Prefix model with a single fast clustering pass.
/// Used only after ordinary serializer policy has selected Prefix coding.
[[nodiscard]] Status OptimizeFastPrefixEntropyCode(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCodeOptions& options,
  EntropyCode* code,
  EntropyCodeCost* cost = nullptr,
  EntropyWorkProfile* profile = nullptr);

/// Counts the exact bits emitted by WriteTokenStream without materializing the
/// encoded payload. The output remains unchanged on failure.
[[nodiscard]] Status CountTokenStreamBits(
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  uint64_t* bit_count);

[[nodiscard]] Status CountTokenStreamBits(
  EntropyTokenStreamView tokens,
  const EntropyCode& code,
  uint64_t* bit_count);

}  // namespace gjxl::codestream_internal
