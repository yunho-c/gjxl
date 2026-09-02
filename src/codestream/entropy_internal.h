// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "codestream/entropy.h"

namespace gjxl::codestream_internal {

struct EntropyWorkProfile;

struct WeightedValue {
  uint32_t value = 0;
  uint64_t count = 0;

  friend bool operator==(const WeightedValue&, const WeightedValue&) = default;
};

/// Raw values aggregated in ascending order for the exact prefix partition.
/// The same cluster populations can feed ANS model construction without
/// collecting and sorting every token a second time.
struct PreparedEntropyClusters {
  uint32_t context_count = 0;
  std::vector<uint8_t> context_map;
  std::vector<std::vector<WeightedValue>> values;

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
