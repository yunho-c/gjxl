// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "codestream/entropy.h"

namespace gjxl::codestream_internal {

inline constexpr uint32_t kAnsReciprocalPrecision = 44;

/// Returns ceil(2^44 / frequency), or zero for an absent symbol.
[[nodiscard]] constexpr uint64_t AnsFrequencyReciprocal(
  uint16_t frequency) noexcept {
  return frequency == 0
    ? 0
    : ((uint64_t{1} << kAnsReciprocalPrecision) + frequency - 1) /
        frequency;
}

/// Divides a normalized ANS state by its frequency using the precomputed
/// reciprocal. The caller must ensure state < frequency * 2^20.
[[nodiscard]] constexpr uint32_t DivideAnsStateByReciprocal(
  uint32_t state,
  uint64_t reciprocal) noexcept {
  return static_cast<uint32_t>(
    (static_cast<uint64_t>(state) * reciprocal) >>
      kAnsReciprocalPrecision);
}

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

[[nodiscard]] Status WriteAnsTokenStream(
  EntropyTokenStreamView tokens,
  const EntropyCode& code,
  BitWriter* writer);

/// Counts one ANS stream by traversing the exact encoder state without
/// allocating or materializing reverse bit chunks. The caller must supply a
/// validated ANS code.
[[nodiscard]] Status CountAnsTokenStreamBits(
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  uint64_t* bit_count);

[[nodiscard]] Status CountAnsTokenStreamBits(
  EntropyTokenStreamView tokens,
  const EntropyCode& code,
  uint64_t* bit_count);

}  // namespace gjxl::codestream_internal
