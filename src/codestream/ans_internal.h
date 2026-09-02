// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "codestream/entropy_internal.h"

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

[[nodiscard]] Status ValidateAnsEntropyCode(const EntropyCode& code);

struct PreparedAnsEntropyCandidate {
  EntropyCode code;
  uint64_t model_bits = 0;
  uint64_t minimum_token_bits = 0;
  bool survives = true;
};

/// ANS alphabet-width candidates whose exact ordered recurrence is deferred.
struct PreparedAnsEntropyCode {
  std::vector<PreparedAnsEntropyCandidate> candidates;
  size_t section_count = 0;
};

enum class DirectAnsEntropyMode {
  kBalanced,
  kHighDensity,
};

/// Builds one ANS model directly from the requested contexts. Unlike the
/// maximum-compression path, this does not derive the partition from an
/// optimized Prefix model or compete across alphabet widths exactly.
[[nodiscard]] Status OptimizeDirectAnsEntropyCode(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCodeOptions& options,
  DirectAnsEntropyMode mode,
  EntropyCode* code,
  EntropyCodeCost* cost = nullptr,
  EntropyWorkProfile* profile = nullptr);

/// Builds ANS models without traversing the ordered streams for exact cost.
[[nodiscard]] Status PrepareAnsEntropyCodeWithPreparedClusters(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCode& prefix_partition,
  const PreparedEntropyClusters& prepared,
  PreparedAnsEntropyCode* deferred,
  EntropyWorkProfile* profile = nullptr);

/// Measures one split section across the prepared alphabet-width candidates.
[[nodiscard]] Status MeasurePreparedAnsEntropyCodeSection(
  EntropyTokenStreamView tokens,
  const PreparedAnsEntropyCode& prepared,
  std::span<uint64_t> candidate_bits);

/// Selects the exact winning width from section-major measurements.
[[nodiscard]] Status FinalizePreparedAnsEntropyCode(
  PreparedAnsEntropyCode* prepared,
  std::span<const uint64_t> section_candidate_bits,
  EntropyCode* code,
  EntropyCodeCost* cost = nullptr);

/// Builds ANS from the prefix optimizer's retained value populations instead
/// of collecting and aggregating the same ordered token streams again.
[[nodiscard]] Status OptimizeAnsEntropyCodeWithPreparedClusters(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCode& prefix_partition,
  const PreparedEntropyClusters& prepared,
  EntropyCode* code,
  EntropyCodeCost* cost = nullptr,
  EntropyWorkProfile* profile = nullptr);

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
