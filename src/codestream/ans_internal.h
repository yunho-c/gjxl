// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "codestream/storage.h"
#include "core/geometry.h"

#include "codestream/entropy_internal.h"

namespace gjxl::codestream_internal {

/// Use the qualified larger AC model budget for 4K-area frames and above.
/// Pixel area makes the policy independent of orientation and padded blocks.
[[nodiscard]] constexpr size_t AcAnsClusterLimit(Extent2D frame) noexcept {
  constexpr size_t minimum_pixels = 3840 * 2160;
  return !frame.empty() && frame.width >= 1 + (minimum_pixels - 1) / frame.height
    ? kMaximumAnsClusters : kDefaultDirectAnsClusters;
}

inline constexpr uint32_t kAnsReciprocalPrecision = 44;
inline constexpr size_t kAnsAlphabetWidthCount = 4;

/// Integer lower bound on sum(count * log2(total/count)), without libm rounding.
/// Unsupported large populations return false so normal validation still runs.
[[nodiscard]] inline bool AnsShannonLowerBound(
  const std::array<uint64_t, kMaximumAnsAlphabetSize>& counts,
  uint64_t* bits) noexcept {
  if (bits == nullptr) return false;
  uint64_t total = 0;
  for (uint64_t count : counts) {
    if (count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        count > std::numeric_limits<uint32_t>::max() - total) return false;
    total += count;
  }
  constexpr uint32_t kFractionBits = 24;
  uint64_t entropy = 0;
  for (uint64_t count : counts) {
    if (count == 0) continue;
    uint32_t integer = std::bit_width(total) - std::bit_width(count);
    uint64_t scaled = count << integer;
    if (scaled > total) {
      --integer;
      scaled >>= 1;
    }
    // total/scaled is in [1,2). With z=(x-1)/(x+1),
    // log2(x) = (2/ln(2)) * (z + z^3/3 + z^5/5 + ...).
    // Every omitted term is positive; every fixed-point operation rounds
    // down. 288539/100000 is strictly less than 2/ln(2).
    const uint64_t z = ((total - scaled) << kFractionBits) / (total + scaled);
    const uint64_t z2 = (z * z) >> kFractionBits;
    uint64_t power = z, sum = z;
    for (uint32_t divisor = 3; divisor <= 9; divisor += 2) {
      power = (power * z2) >> kFractionBits;
      sum += power / divisor;
    }
    const uint64_t fraction = sum * 288539 / 100000;
    entropy += count * ((uint64_t{integer} << kFractionBits) + fraction);
  }
  // total <= 2^32-1 and each log bound <= 32, so accumulation uses < 61 bits.
  *bits = entropy >> kFractionBits;
  return true;
}

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
  codestream_internal::Storage<PreparedAnsEntropyCandidate> candidates;
  size_t section_count = 0;
};

enum class DirectAnsEntropyMode {
  kBalanced,
  kHighDensity,
  /// High-density partition/config search, comparing every alphabet width.
  kRateOptimized,
};

inline constexpr size_t kAnsHistogramPrecisionShiftCount = 12;

/// Pinned from libjxl enc_ans.cc at the repository's reference revision.
[[nodiscard]] std::span<const HybridUintConfig>
HighDensityAnsUintConfigs() noexcept;

/// Population-precision shifts searched by each direct ANS policy. The flat
/// histogram candidate is evaluated separately.
[[nodiscard]] std::array<bool, kAnsHistogramPrecisionShiftCount>
DirectAnsHistogramPrecisionShifts(DirectAnsEntropyMode mode) noexcept;

/// Builds one ANS model directly from the requested contexts. Unlike the
/// maximum-compression path, this does not derive the partition from an
/// optimized Prefix model. kRateOptimized compares all alphabet widths using
/// exact model and ordered token costs, even when cost is null.
[[nodiscard]] Status OptimizeDirectAnsEntropyCode(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCodeOptions& options,
  DirectAnsEntropyMode mode,
  EntropyCode* code,
  EntropyCodeCost* cost = nullptr,
  EntropyWorkProfile* profile = nullptr);

/// Rate-optimized direct models with exact width selection deferred until the
/// caller measures each section. Uses the same partition/configuration search.
[[nodiscard]] Status PrepareRateOptimizedAnsEntropyCode(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCodeOptions& options,
  PreparedAnsEntropyCode* deferred,
  EntropyWorkProfile* profile = nullptr);

/// Balanced direct-ANS construction from already encoded per-context symbol
/// populations. Ordered streams remain authoritative for final token cost and
/// emission, but are not traversed to rebuild the same histograms.
[[nodiscard]] Status OptimizeDirectAnsEntropyCodeWithFixedPopulations(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCodeOptions& options,
  std::span<const PreparedFixedAnsCluster> context_populations,
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
