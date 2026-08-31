// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "codestream/entropy.h"

namespace gjxl::codestream_internal {

struct WeightedValue {
  uint32_t value = 0;
  uint64_t count = 0;
};

/// Aggregates raw values in ascending order. Small inputs sort occurrences;
/// large inputs count the bounded dense prefix and sort only sparse values.
[[nodiscard]] Status AggregateEntropyValues(
  std::vector<uint32_t> values,
  std::vector<WeightedValue>* aggregated);

[[nodiscard]] Status ValidateAnsEntropyCode(const EntropyCode& code);

[[nodiscard]] Status WriteAnsEntropyCodeModel(
  const EntropyCode& code,
  BitWriter* writer);

[[nodiscard]] Status WriteAnsTokenStream(
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  BitWriter* writer);

/// Counts one ANS stream by traversing the exact encoder state without
/// allocating or materializing reverse bit chunks. The caller must supply a
/// validated ANS code.
[[nodiscard]] Status CountAnsTokenStreamBits(
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  uint64_t* bit_count);

}  // namespace gjxl::codestream_internal
