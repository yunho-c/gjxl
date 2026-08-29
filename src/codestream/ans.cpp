// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl's enc_ans.cc and ans_common.cc.

#include "codestream/ans_internal.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gjxl {
namespace {

constexpr uint32_t kAnsLogTableSize = 12;
constexpr uint32_t kAnsSignature = 0x13;

struct AliasEntry {
  uint16_t cutoff = 0;
  uint8_t right_value = 0;
  uint16_t offsets1 = 0;
  uint16_t frequency0 = 0;
  uint16_t frequency1 = 0;
};

struct ReverseBitChunk {
  uint32_t bits = 0;
  uint8_t bit_count = 0;
};

Status AllocationFailure() {
  return Status::OutOfMemory("ANS entropy allocation failed");
}

Status StoreVarLenUint8(size_t value, BitWriter* writer) {
  if (writer == nullptr || value > std::numeric_limits<uint8_t>::max()) {
    return Status::InvalidArgument("Variable-length uint8 is invalid");
  }
  if (value == 0) {
    return writer->WriteBits(1, 0);
  }
  if (Status status = writer->WriteBits(1, 1); !status.ok()) {
    return status;
  }
  const size_t highest_bit = std::bit_width(value) - 1;
  if (Status status = writer->WriteBits(3, highest_bit); !status.ok()) {
    return status;
  }
  return writer->WriteBits(
    highest_bit, value - (size_t{1} << highest_bit));
}

Status NormalizeHistogram(
  const std::array<uint64_t, kPrefixAlphabetSize>& raw,
  std::vector<uint16_t>* frequencies) {

  if (frequencies == nullptr) {
    return Status::InvalidArgument("ANS histogram output is null");
  }
  size_t alphabet_size = raw.size();
  while (alphabet_size != 0 && raw[alphabet_size - 1] == 0) {
    --alphabet_size;
  }
  if (alphabet_size == 0) {
    frequencies->clear();
    return Status::Ok();
  }
  uint64_t total = 0;
  size_t populated = 0;
  size_t only_symbol = 0;
  for (size_t symbol = 0; symbol < alphabet_size; ++symbol) {
    if (raw[symbol] != 0) {
      if (total > std::numeric_limits<uint64_t>::max() - raw[symbol]) {
        return Status::InvalidArgument("ANS histogram count overflow");
      }
      total += raw[symbol];
      only_symbol = symbol;
      ++populated;
    }
  }
  frequencies->assign(alphabet_size, 0);
  if (populated == 1) {
    (*frequencies)[only_symbol] = kAnsTableSize;
    return Status::Ok();
  }

  struct Remainder {
    size_t symbol = 0;
    uint64_t remainder = 0;
  };
  std::vector<Remainder> remainders;
  remainders.reserve(populated);
  size_t normalized_total = 0;
  for (size_t symbol = 0; symbol < alphabet_size; ++symbol) {
    if (raw[symbol] == 0) {
      continue;
    }
    const unsigned __int128 scaled =
      static_cast<unsigned __int128>(raw[symbol]) * kAnsTableSize;
    const uint64_t quotient = static_cast<uint64_t>(scaled / total);
    const uint64_t remainder = static_cast<uint64_t>(scaled % total);
    const uint16_t frequency = static_cast<uint16_t>(
      std::max<uint64_t>(quotient, 1));
    (*frequencies)[symbol] = frequency;
    normalized_total += frequency;
    remainders.push_back({symbol, remainder});
  }
  if (normalized_total < kAnsTableSize) {
    std::stable_sort(
      remainders.begin(), remainders.end(),
      [](const Remainder& left, const Remainder& right) {
        return left.remainder > right.remainder;
      });
    size_t missing = kAnsTableSize - normalized_total;
    for (size_t index = 0; missing != 0; ++index, --missing) {
      ++(*frequencies)[remainders[index % remainders.size()].symbol];
    }
  } else if (normalized_total > kAnsTableSize) {
    size_t excess = normalized_total - kAnsTableSize;
    while (excess != 0) {
      size_t selected = frequencies->size();
      for (size_t symbol = 0; symbol < frequencies->size(); ++symbol) {
        if ((*frequencies)[symbol] > 1 &&
            (selected == frequencies->size() ||
             (*frequencies)[symbol] > (*frequencies)[selected])) {
          selected = symbol;
        }
      }
      if (selected == frequencies->size()) {
        return Status::Internal("ANS histogram cannot be normalized");
      }
      const size_t removable = std::min<size_t>(
        excess, (*frequencies)[selected] - 1);
      (*frequencies)[selected] -= static_cast<uint16_t>(removable);
      excess -= removable;
    }
  }
  return Status::Ok();
}

Status InitializeAliasTable(
  std::span<const uint16_t> frequencies,
  size_t log_alpha_size,
  std::vector<AliasEntry>* table) {

  if (table == nullptr || log_alpha_size < 5 || log_alpha_size > 8 ||
      frequencies.size() > (size_t{1} << log_alpha_size)) {
    return Status::InvalidArgument("ANS alias-table dimensions are invalid");
  }
  const size_t table_size = size_t{1} << log_alpha_size;
  const uint32_t entry_size = kAnsTableSize >> log_alpha_size;
  table->assign(table_size, AliasEntry{});
  std::vector<uint32_t> distribution(frequencies.begin(), frequencies.end());
  while (!distribution.empty() && distribution.back() == 0) {
    distribution.pop_back();
  }
  if (distribution.empty()) {
    distribution.push_back(kAnsTableSize);
  }
  uint32_t sum = 0;
  int single_symbol = -1;
  for (size_t symbol = 0; symbol < distribution.size(); ++symbol) {
    sum += distribution[symbol];
    if (distribution[symbol] == kAnsTableSize) {
      single_symbol = static_cast<int>(symbol);
    }
  }
  if (sum != kAnsTableSize) {
    return Status::InvalidArgument("ANS frequencies do not sum to the table");
  }
  if (single_symbol >= 0) {
    for (size_t index = 0; index < table_size; ++index) {
      AliasEntry& entry = (*table)[index];
      entry.right_value = static_cast<uint8_t>(single_symbol);
      entry.offsets1 = static_cast<uint16_t>(entry_size * index);
      entry.frequency1 = kAnsTableSize;
    }
    return Status::Ok();
  }

  std::vector<uint32_t> underfull;
  std::vector<uint32_t> overfull;
  std::vector<uint32_t> cutoffs(table_size, 0);
  for (size_t index = 0; index < distribution.size(); ++index) {
    cutoffs[index] = distribution[index];
    if (cutoffs[index] < entry_size) {
      underfull.push_back(static_cast<uint32_t>(index));
    } else if (cutoffs[index] > entry_size) {
      overfull.push_back(static_cast<uint32_t>(index));
    }
  }
  for (size_t index = distribution.size(); index < table_size; ++index) {
    underfull.push_back(static_cast<uint32_t>(index));
  }
  while (!overfull.empty()) {
    const uint32_t overfull_index = overfull.back();
    overfull.pop_back();
    if (underfull.empty()) {
      return Status::Internal("ANS alias table is unbalanced");
    }
    const uint32_t underfull_index = underfull.back();
    underfull.pop_back();
    const uint32_t missing = entry_size - cutoffs[underfull_index];
    cutoffs[overfull_index] -= missing;
    AliasEntry& entry = (*table)[underfull_index];
    entry.right_value = static_cast<uint8_t>(overfull_index);
    entry.offsets1 = static_cast<uint16_t>(cutoffs[overfull_index]);
    if (cutoffs[overfull_index] < entry_size) {
      underfull.push_back(overfull_index);
    } else if (cutoffs[overfull_index] > entry_size) {
      overfull.push_back(overfull_index);
    }
  }
  for (size_t index = 0; index < table_size; ++index) {
    AliasEntry& entry = (*table)[index];
    if (cutoffs[index] == entry_size) {
      entry.right_value = static_cast<uint8_t>(index);
      entry.offsets1 = 0;
      entry.cutoff = 0;
    } else {
      entry.offsets1 = static_cast<uint16_t>(
        entry.offsets1 - cutoffs[index]);
      entry.cutoff = static_cast<uint16_t>(cutoffs[index]);
    }
    entry.frequency0 = index < distribution.size()
      ? static_cast<uint16_t>(distribution[index])
      : 0;
    entry.frequency1 = entry.right_value < distribution.size()
      ? static_cast<uint16_t>(distribution[entry.right_value])
      : 0;
  }
  return Status::Ok();
}

Status BuildReverseMaps(
  std::span<const uint16_t> frequencies,
  size_t log_alpha_size,
  std::vector<std::vector<uint16_t>>* reverse_maps) {

  if (reverse_maps == nullptr) {
    return Status::InvalidArgument("ANS reverse-map output is null");
  }
  reverse_maps->clear();
  reverse_maps->resize(frequencies.size());
  for (size_t symbol = 0; symbol < frequencies.size(); ++symbol) {
    (*reverse_maps)[symbol].resize(frequencies[symbol]);
  }
  if (frequencies.empty()) {
    return Status::Ok();
  }
  std::vector<AliasEntry> table;
  if (Status status = InitializeAliasTable(
        frequencies, log_alpha_size, &table);
      !status.ok()) {
    return status;
  }
  const size_t log_entry_size = kAnsLogTableSize - log_alpha_size;
  const size_t entry_mask = (size_t{1} << log_entry_size) - 1;
  for (size_t value = 0; value < kAnsTableSize; ++value) {
    const size_t table_index = value >> log_entry_size;
    const size_t position = value & entry_mask;
    const AliasEntry& entry = table[table_index];
    const bool right = position >= entry.cutoff;
    const size_t symbol = right ? entry.right_value : table_index;
    const size_t offset = (right ? entry.offsets1 : 0) + position;
    if (symbol >= reverse_maps->size() ||
        offset >= (*reverse_maps)[symbol].size()) {
      return Status::Internal("ANS reverse-map construction failed");
    }
    (*reverse_maps)[symbol][offset] = static_cast<uint16_t>(value);
  }
  return Status::Ok();
}

Status WriteAnsUintConfig(
  HybridUintConfig config,
  size_t log_alpha_size,
  BitWriter* writer) {

  if (writer == nullptr || !config.valid() ||
      config.split_exponent > log_alpha_size) {
    return Status::InvalidArgument("ANS HybridUint configuration is invalid");
  }
  if (Status status = writer->WriteBits(
        std::bit_width(log_alpha_size), config.split_exponent);
      !status.ok()) {
    return status;
  }
  if (config.split_exponent == log_alpha_size) {
    return Status::Ok();
  }
  if (Status status = writer->WriteBits(
        std::bit_width(static_cast<size_t>(config.split_exponent)),
        config.msb_in_token);
      !status.ok()) {
    return status;
  }
  const size_t lsb_choices =
    config.split_exponent - config.msb_in_token;
  return writer->WriteBits(
    std::bit_width(lsb_choices), config.lsb_in_token);
}

Status WriteAnsHistogram(
  const AnsHistogram& histogram,
  BitWriter* writer) {

  if (writer == nullptr) {
    return Status::InvalidArgument("ANS histogram output is null");
  }
  std::array<size_t, 2> symbols{};
  size_t symbol_count = 0;
  for (size_t symbol = 0; symbol < histogram.frequencies.size(); ++symbol) {
    if (histogram.frequencies[symbol] != 0) {
      if (symbol_count < symbols.size()) {
        symbols[symbol_count] = symbol;
      }
      ++symbol_count;
    }
  }
  if (symbol_count <= 2) {
    if (Status status = writer->WriteBits(1, 1); !status.ok()) {
      return status;
    }
    if (Status status = writer->WriteBits(1, symbol_count == 2 ? 1 : 0);
        !status.ok()) {
      return status;
    }
    if (symbol_count == 0) {
      return StoreVarLenUint8(0, writer);
    }
    for (size_t index = 0; index < symbol_count; ++index) {
      if (Status status = StoreVarLenUint8(symbols[index], writer);
          !status.ok()) {
        return status;
      }
    }
    return symbol_count == 2
      ? writer->WriteBits(
          kAnsLogTableSize, histogram.frequencies[symbols[0]])
      : Status::Ok();
  }

  if (Status status = writer->WriteBits(2, 0); !status.ok()) {
    return status;
  }
  // General histograms use shift 11, which represents every normalized count
  // exactly and leaves the largest population implicit.
  if (Status status = writer->WriteBits(3, 7); !status.ok()) {
    return status;
  }
  if (Status status = writer->WriteBits(3, 4); !status.ok()) {
    return status;
  }
  if (Status status = StoreVarLenUint8(
        histogram.frequencies.size() - 3, writer);
      !status.ok()) {
    return status;
  }

  size_t omit_position = 0;
  for (size_t symbol = 1; symbol < histogram.frequencies.size(); ++symbol) {
    if (histogram.frequencies[symbol] >
        histogram.frequencies[omit_position]) {
      omit_position = symbol;
    }
  }
  std::vector<uint8_t> bit_widths(histogram.frequencies.size(), 0);
  uint8_t omit_width = 10;
  for (size_t symbol = 0; symbol < histogram.frequencies.size(); ++symbol) {
    if (symbol == omit_position || histogram.frequencies[symbol] == 0) {
      continue;
    }
    bit_widths[symbol] = static_cast<uint8_t>(
      std::bit_width(histogram.frequencies[symbol]));
    omit_width = std::max<uint8_t>(
      omit_width,
      static_cast<uint8_t>(bit_widths[symbol] +
                           (symbol < omit_position ? 1 : 0)));
  }
  bit_widths[omit_position] = omit_width;
  constexpr std::array<uint8_t, kAnsLogTableSize + 2> kDepths = {
    5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 6, 7, 7,
  };
  constexpr std::array<uint8_t, kAnsLogTableSize + 2> kBits = {
    17, 11, 15, 3, 9, 7, 4, 2, 5, 6, 0, 33, 1, 65,
  };
  for (uint8_t width : bit_widths) {
    if (width >= kDepths.size()) {
      return Status::Internal("ANS population width is invalid");
    }
    if (Status status = writer->WriteBits(kDepths[width], kBits[width]);
        !status.ok()) {
      return status;
    }
  }
  for (size_t symbol = 0; symbol < histogram.frequencies.size(); ++symbol) {
    if (symbol == omit_position || bit_widths[symbol] <= 1) {
      continue;
    }
    const size_t bit_count = bit_widths[symbol] - 1;
    if (Status status = writer->WriteBits(
          bit_count,
          histogram.frequencies[symbol] - (uint16_t{1} << bit_count));
        !status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status MeasureAnsCode(
  std::span<const std::vector<EntropyToken>> section_tokens,
  const EntropyCode& code,
  EntropyCodeCost* cost) {

  if (cost == nullptr) {
    return Status::InvalidArgument("ANS cost output is null");
  }
  BitWriter model;
  if (Status status = WriteEntropyCode(code, &model); !status.ok()) {
    return status;
  }
  EntropyCodeCost candidate;
  candidate.model_bits = model.bits_written();
  candidate.cluster_count = code.ans_histograms.size();
  for (const std::vector<EntropyToken>& section : section_tokens) {
    BitWriter payload;
    if (Status status = WriteTokenStream(section, code, &payload);
        !status.ok()) {
      return status;
    }
    if (candidate.token_bits >
        std::numeric_limits<uint64_t>::max() - payload.bits_written()) {
      return Status::InvalidArgument("ANS token cost overflow");
    }
    candidate.token_bits += payload.bits_written();
  }
  *cost = candidate;
  return Status::Ok();
}

}  // namespace

Status codestream_internal::ValidateAnsEntropyCode(const EntropyCode& code) {
  if (code.mode != EntropyCodingMode::kAns || code.context_count == 0 ||
      code.context_map.size() != code.context_count ||
      !code.prefix_codes.empty() || code.ans_log_alpha_size < 5 ||
      code.ans_log_alpha_size > 8 || code.ans_histograms.empty() ||
      code.ans_histograms.size() > kMaximumPrefixClusters ||
      code.uint_configs.size() != code.ans_histograms.size()) {
    return Status::InvalidArgument("ANS entropy-code dimensions are invalid");
  }
  for (uint8_t cluster : code.context_map) {
    if (cluster >= code.ans_histograms.size()) {
      return Status::InvalidArgument("ANS context map is invalid");
    }
  }
  const size_t alphabet_limit = std::min<size_t>(
    kMaximumAnsAlphabetSize, size_t{1} << code.ans_log_alpha_size);
  for (size_t cluster = 0; cluster < code.ans_histograms.size(); ++cluster) {
    const HybridUintConfig config = code.uint_configs[cluster];
    if (!config.valid() || config.split_exponent > code.ans_log_alpha_size) {
      return Status::InvalidArgument("ANS HybridUint config is invalid");
    }
    const AnsHistogram& histogram = code.ans_histograms[cluster];
    if (histogram.frequencies.size() > alphabet_limit ||
        histogram.reverse_maps.size() != histogram.frequencies.size() ||
        (!histogram.frequencies.empty() &&
         histogram.frequencies.back() == 0)) {
      return Status::InvalidArgument("ANS histogram dimensions are invalid");
    }
    size_t total = 0;
    std::array<bool, kAnsTableSize> seen{};
    for (size_t symbol = 0; symbol < histogram.frequencies.size(); ++symbol) {
      const uint16_t frequency = histogram.frequencies[symbol];
      total += frequency;
      if (histogram.reverse_maps[symbol].size() != frequency) {
        return Status::InvalidArgument("ANS reverse map is invalid");
      }
      for (uint16_t value : histogram.reverse_maps[symbol]) {
        if (value >= kAnsTableSize || seen[value]) {
          return Status::InvalidArgument("ANS reverse-map entry is invalid");
        }
        seen[value] = true;
      }
    }
    if (total != 0 && total != kAnsTableSize) {
      return Status::InvalidArgument("ANS histogram total is invalid");
    }
  }
  return Status::Ok();
}

Status codestream_internal::WriteAnsEntropyCodeModel(
  const EntropyCode& code,
  BitWriter* writer) {

  if (writer == nullptr) {
    return Status::InvalidArgument("ANS entropy-code output is null");
  }
  BitWriter temporary;
  if (Status status = WriteContextMap(code, &temporary); !status.ok()) {
    return status;
  }
  if (Status status = temporary.WriteBits(1, 0); !status.ok()) {
    return status;
  }
  if (Status status = temporary.WriteBits(
        2, code.ans_log_alpha_size - 5);
      !status.ok()) {
    return status;
  }
  for (HybridUintConfig config : code.uint_configs) {
    if (Status status = WriteAnsUintConfig(
          config, code.ans_log_alpha_size, &temporary);
        !status.ok()) {
      return status;
    }
  }
  for (const AnsHistogram& histogram : code.ans_histograms) {
    if (Status status = WriteAnsHistogram(histogram, &temporary);
        !status.ok()) {
      return status;
    }
  }
  return writer->Append(temporary);
}

Status codestream_internal::WriteAnsTokenStream(
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  BitWriter* writer) {

  if (writer == nullptr) {
    return Status::InvalidArgument("ANS token-stream output is null");
  }
  try {
    uint32_t state = kAnsSignature << 16;
    std::vector<ReverseBitChunk> reverse_chunks;
    reverse_chunks.reserve(2 * tokens.size());
    for (size_t index = tokens.size(); index != 0; --index) {
      const EntropyToken& token = tokens[index - 1];
      if (token.context >= code.context_count) {
        return Status::InvalidArgument("ANS token context is out of range");
      }
      const size_t cluster = code.context_map[token.context];
      HybridUintToken encoded;
      if (Status status = EncodeHybridUint(
            token.value, code.uint_configs[cluster], &encoded);
          !status.ok()) {
        return status;
      }
      const AnsHistogram& histogram = code.ans_histograms[cluster];
      if (encoded.symbol >= histogram.frequencies.size() ||
          histogram.frequencies[encoded.symbol] == 0) {
        return Status::InvalidArgument("ANS token symbol is absent");
      }
      if (encoded.extra_bit_count != 0) {
        reverse_chunks.push_back({
          encoded.extra_bits, encoded.extra_bit_count});
      }
      const uint32_t frequency = histogram.frequencies[encoded.symbol];
      if ((state >> (32 - kAnsLogTableSize)) >= frequency) {
        reverse_chunks.push_back({state & 0xFFFFu, 16});
        state >>= 16;
      }
      const uint32_t quotient = state / frequency;
      const uint32_t remainder = state - quotient * frequency;
      const std::vector<uint16_t>& reverse =
        histogram.reverse_maps[encoded.symbol];
      if (remainder >= reverse.size() || reverse[remainder] >= kAnsTableSize) {
        return Status::InvalidArgument("ANS reverse-map entry is invalid");
      }
      state = (quotient << kAnsLogTableSize) + reverse[remainder];
    }
    BitWriter temporary;
    if (Status status = temporary.WriteBits(32, state); !status.ok()) {
      return status;
    }
    for (auto chunk = reverse_chunks.rbegin(); chunk != reverse_chunks.rend();
         ++chunk) {
      if (Status status = temporary.WriteBits(
            chunk->bit_count, chunk->bits);
          !status.ok()) {
        return status;
      }
    }
    return writer->Append(temporary);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
}

Status OptimizeAnsEntropyCode(
  std::span<const std::vector<EntropyToken>> section_tokens,
  const EntropyCode& prefix_partition,
  EntropyCode* code,
  EntropyCodeCost* cost) {

  if (code == nullptr || prefix_partition.mode != EntropyCodingMode::kPrefix ||
      prefix_partition.prefix_codes.empty()) {
    return Status::InvalidArgument("ANS partition input is invalid");
  }
  BitWriter validated_prefix;
  if (Status status = WriteEntropyCode(
        prefix_partition, &validated_prefix);
      !status.ok()) {
    return status;
  }
  try {
    EntropyCode candidate;
    candidate.mode = EntropyCodingMode::kAns;
    candidate.context_count = prefix_partition.context_count;
    candidate.context_map = prefix_partition.context_map;
    candidate.uint_configs = prefix_partition.uint_configs;
    candidate.ans_histograms.resize(prefix_partition.prefix_codes.size());
    std::vector<std::array<uint64_t, kPrefixAlphabetSize>> counts(
      candidate.ans_histograms.size());
    size_t maximum_symbol = 0;
    for (const std::vector<EntropyToken>& section : section_tokens) {
      for (const EntropyToken& token : section) {
        if (token.context >= candidate.context_count) {
          return Status::InvalidArgument("ANS token context is out of range");
        }
        const size_t cluster = candidate.context_map[token.context];
        HybridUintToken encoded;
        if (Status status = EncodeHybridUint(
              token.value, candidate.uint_configs[cluster], &encoded);
            !status.ok()) {
          return status;
        }
        if (encoded.symbol >= kPrefixAlphabetSize ||
            counts[cluster][encoded.symbol] ==
              std::numeric_limits<uint64_t>::max()) {
          return Status::InvalidArgument("ANS symbol count overflow");
        }
        ++counts[cluster][encoded.symbol];
        maximum_symbol = std::max<size_t>(maximum_symbol, encoded.symbol);
      }
    }
    candidate.ans_log_alpha_size = static_cast<uint8_t>(
      std::max<size_t>(5, std::bit_width(maximum_symbol)));
    if (candidate.ans_log_alpha_size > 8) {
      return Status::InvalidArgument("ANS alphabet is too large");
    }
    for (HybridUintConfig config : candidate.uint_configs) {
      if (config.split_exponent > candidate.ans_log_alpha_size) {
        return Status::InvalidArgument(
          "ANS HybridUint config exceeds its alphabet");
      }
    }
    for (size_t cluster = 0; cluster < candidate.ans_histograms.size();
         ++cluster) {
      AnsHistogram& histogram = candidate.ans_histograms[cluster];
      if (Status status = NormalizeHistogram(
            counts[cluster], &histogram.frequencies);
          !status.ok()) {
        return status;
      }
      if (Status status = BuildReverseMaps(
            histogram.frequencies, candidate.ans_log_alpha_size,
            &histogram.reverse_maps);
          !status.ok()) {
        return status;
      }
    }
    EntropyCodeCost candidate_cost;
    if (Status status = MeasureAnsCode(
          section_tokens, candidate, &candidate_cost);
        !status.ok()) {
      return status;
    }
    *code = std::move(candidate);
    if (cost != nullptr) {
      *cost = candidate_cost;
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
}

}  // namespace gjxl
