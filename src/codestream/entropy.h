// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's prefix entropy encoder.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "codestream/storage.h"

#include "codestream/bit_writer.h"
#include "core/status.h"

namespace gjxl {

namespace codestream_internal {
struct EntropyWorkProfile;
}  // namespace codestream_internal

inline constexpr size_t kPrefixAlphabetSize = 128;
inline constexpr size_t kMaximumPrefixClusters = 32;
inline constexpr size_t kAnsTableSize = 4096;
inline constexpr size_t kMaximumAnsAlphabetSize = 256;

struct EntropyToken {
  uint32_t context = 0;
  uint32_t value = 0;

  friend bool operator==(const EntropyToken&, const EntropyToken&) = default;
};

/// Read-only entropy-token stream backed either by conventional interleaved
/// tokens or by shared values plus candidate-specific 16-bit contexts.
struct EntropyTokenStreamView {
  std::span<const EntropyToken> tokens;
  std::span<const uint32_t> values;
  std::span<const uint16_t> contexts;
  bool split = false;

  [[nodiscard]] static constexpr EntropyTokenStreamView Interleaved(
    std::span<const EntropyToken> source) noexcept {
    return {.tokens = source};
  }

  [[nodiscard]] static constexpr EntropyTokenStreamView Split(
    std::span<const uint32_t> source_values,
    std::span<const uint16_t> source_contexts) noexcept {
    return {
      .values = source_values,
      .contexts = source_contexts,
      .split = true,
    };
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return split
      ? tokens.empty() && values.size() == contexts.size()
      : values.empty() && contexts.empty();
  }

  [[nodiscard]] constexpr size_t size() const noexcept {
    return split ? values.size() : tokens.size();
  }

  [[nodiscard]] constexpr EntropyToken operator[](size_t index) const noexcept {
    return split ? EntropyToken{contexts[index], values[index]} : tokens[index];
  }
};

/// Encodes signed integers as 0, 1, 2, 3... in 0, -1, 1, -2... order.
[[nodiscard]] constexpr uint32_t PackSigned(int32_t value) noexcept {
  return (static_cast<uint32_t>(value) << 1) ^
    ((static_cast<uint32_t>(~value) >> 31) - 1);
}

struct HybridUintConfig {
  uint8_t split_exponent = 4;
  uint8_t msb_in_token = 2;
  uint8_t lsb_in_token = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return split_exponent <= 15 &&
      msb_in_token <= split_exponent &&
      lsb_in_token <= split_exponent - msb_in_token;
  }

  friend bool operator==(
    const HybridUintConfig&,
    const HybridUintConfig&) = default;
};

inline constexpr HybridUintConfig kDefaultHybridUintConfig{4, 2, 0};

struct HybridUintToken {
  uint32_t symbol = 0;
  uint8_t extra_bit_count = 0;
  uint32_t extra_bits = 0;

  friend bool operator==(
    const HybridUintToken&,
    const HybridUintToken&) = default;
};

[[nodiscard]] Status EncodeHybridUint(
  uint32_t value,
  HybridUintConfig config,
  HybridUintToken* token);

struct PrefixCode {
  std::array<uint8_t, kPrefixAlphabetSize> depths{};
  std::array<uint16_t, kPrefixAlphabetSize> bits{};
  uint16_t degenerate_symbol = kPrefixAlphabetSize;

  friend bool operator==(const PrefixCode&, const PrefixCode&) = default;
};

enum class EntropyCodingMode : uint8_t {
  kPrefix,
  kAns,
};

/// One normalized 12-bit ANS population and its encoder lookup tables.
struct AnsHistogram {
  codestream_internal::Storage<uint16_t> frequencies;
  codestream_internal::Storage<codestream_internal::Storage<uint16_t>> reverse_maps;
  /// Exact ceil(2^44 / frequency) encoder divisors; zero for absent symbols.
  codestream_internal::Storage<uint64_t> reciprocal_frequencies;

  /// Zero selects the flat representation; 1 through 12 encode shift + 1.
  /// Small one- and two-symbol populations ignore both representation fields.
  uint8_t method = 12;
  uint16_t omit_position = 0;

  friend bool operator==(const AnsHistogram&, const AnsHistogram&) = default;
};

struct EntropyCode {
  EntropyCodingMode mode = EntropyCodingMode::kPrefix;
  uint32_t context_count = 0;
  codestream_internal::Storage<uint8_t> context_map;
  codestream_internal::Storage<HybridUintConfig> uint_configs;
  codestream_internal::Storage<PrefixCode> prefix_codes;
  uint8_t ans_log_alpha_size = 0;
  codestream_internal::Storage<AnsHistogram> ans_histograms;

  friend bool operator==(const EntropyCode&, const EntropyCode&) = default;
};

struct EntropyCodeOptions {
  uint32_t context_count = 0;

  /// Optional map from token contexts to pre-clustered histogram indexes.
  std::span<const uint8_t> initial_context_map;
  uint32_t initial_histogram_count = 0;
  HybridUintConfig uint_config = kDefaultHybridUintConfig;
};

struct EntropyCodeCost {
  uint64_t model_bits = 0;
  uint64_t token_bits = 0;
  size_t cluster_count = 0;
  /// Exact token bits for each input section, in input order. Their sum is
  /// token_bits whenever a cost was requested from an optimizer.
  codestream_internal::Storage<uint64_t> section_token_bits;

  friend bool operator==(
    const EntropyCodeCost&,
    const EntropyCodeCost&) = default;
};

/// Builds one entropy model over all supplied section token streams.
[[nodiscard]] Status OptimizeEntropyCode(
  std::span<const std::vector<EntropyToken>> section_tokens,
  const EntropyCodeOptions& options,
  EntropyCode* code,
  EntropyCodeCost* cost = nullptr,
  codestream_internal::EntropyWorkProfile* profile = nullptr);

/// Optimizes an entropy model without requiring interleaved token storage.
[[nodiscard]] Status OptimizeEntropyCode(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCodeOptions& options,
  EntropyCode* code,
  EntropyCodeCost* cost = nullptr,
  codestream_internal::EntropyWorkProfile* profile = nullptr);

/// Builds an ANS model using an optimized prefix code's context partition.
/// HybridUint configurations and normalized populations are screened using an
/// ANS-specific cost estimate, then alphabet widths compete on exact serialized
/// model-plus-token cost. Inputs and outputs remain unchanged on failure.
[[nodiscard]] Status OptimizeAnsEntropyCode(
  std::span<const std::vector<EntropyToken>> section_tokens,
  const EntropyCode& prefix_partition,
  EntropyCode* code,
  EntropyCodeCost* cost = nullptr,
  codestream_internal::EntropyWorkProfile* profile = nullptr);

/// Optimizes ANS from interleaved or shared-value token streams.
[[nodiscard]] Status OptimizeAnsEntropyCode(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCode& prefix_partition,
  EntropyCode* code,
  EntropyCodeCost* cost = nullptr,
  codestream_internal::EntropyWorkProfile* profile = nullptr);

/// Builds a canonical maximum-15-bit prefix code for a histogram.
[[nodiscard]] Status BuildPrefixCode(
  std::span<const uint64_t> counts,
  PrefixCode* code);

/// Serializes the prefix-code marker, HybridUint configurations, and trees.
[[nodiscard]] Status WritePrefixCodes(
  std::span<const PrefixCode> prefix_codes,
  HybridUintConfig config,
  BitWriter* writer);

/// Serializes one HybridUint configuration per prefix code.
[[nodiscard]] Status WritePrefixCodes(
  std::span<const PrefixCode> prefix_codes,
  std::span<const HybridUintConfig> configs,
  BitWriter* writer);

/// Serializes the optimized context map.
[[nodiscard]] Status WriteContextMap(
  const EntropyCode& code,
  BitWriter* writer);

/// Serializes the context map and all prefix codes.
[[nodiscard]] Status WriteEntropyCode(
  const EntropyCode& code,
  BitWriter* writer);

/// Encodes a section's tokens using an already optimized entropy model.
[[nodiscard]] Status WriteTokenStream(
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  BitWriter* writer);

/// Encodes a split or interleaved stream using an optimized entropy model.
[[nodiscard]] Status WriteTokenStream(
  EntropyTokenStreamView tokens,
  const EntropyCode& code,
  BitWriter* writer);

}  // namespace gjxl
