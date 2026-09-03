// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl's enc_ans.cc and ans_common.cc.

#include "codestream/ans_internal.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <queue>
#include <span>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "codestream/profile_internal.h"

namespace gjxl {
namespace {

using ProfileClock = std::chrono::steady_clock;
using EntropyWorkProfile = codestream_internal::EntropyWorkProfile;

ProfileClock::time_point ProfileBegin(const EntropyWorkProfile* profile) {
  return profile == nullptr ? ProfileClock::time_point{} : ProfileClock::now();
}

void ProfileEnd(
  EntropyWorkProfile* profile,
  ProfileClock::time_point begin,
  uint64_t EntropyWorkProfile::* destination) {

  if (profile == nullptr) return;
  profile->*destination += static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      ProfileClock::now() - begin).count());
}

constexpr uint32_t kAnsLogTableSize = 12;
static_assert(
  kAnsLogTableSize ==
  codestream_internal::kAnsHistogramPrecisionShiftCount);
constexpr uint32_t kAnsSignature = 0x13;
constexpr size_t kAnsAlphabetWidthCount = 4;
constexpr size_t kExactLog2TableSize = 1 << 16;
// Retain the four prefix candidates, then add the four ANS configurations that
// materially improved the established corpus. The wider 28-choice experiment
// only saved four additional bytes and added measurable complete-encode cost.
constexpr std::array<HybridUintConfig, 8> kAnsUintConfigs = {{
  {4, 2, 0},
  {4, 1, 2},
  {0, 0, 0},
  {2, 0, 1},
  {3, 1, 0},
  {3, 2, 0},
  {4, 1, 0},
  {5, 2, 0},
}};

constexpr std::array<HybridUintConfig, 28> kHighDensityAnsUintConfigs = {{
  {4, 2, 0}, {4, 1, 0}, {4, 2, 1}, {4, 2, 2}, {4, 1, 2},
  {5, 2, 0}, {5, 1, 0}, {5, 2, 1}, {5, 2, 2}, {5, 1, 2},
  {3, 2, 0}, {3, 1, 0}, {3, 2, 1}, {3, 1, 2},
  {4, 1, 3}, {5, 1, 4}, {5, 2, 3}, {6, 1, 5}, {6, 2, 4},
  {6, 0, 0}, {0, 0, 0}, {2, 0, 1}, {7, 0, 0}, {8, 0, 0},
  {9, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0},
}};

enum class AnsHistogramSearch {
  kFast,
  kApproximate,
  kPrecise,
};

struct AnsOptimizationPolicy {
  std::span<const HybridUintConfig> uint_configs;
  AnsHistogramSearch histogram_search = AnsHistogramSearch::kPrecise;
  bool smallest_alphabet_width = false;
};

constexpr AnsOptimizationPolicy kMaximumCompressionAnsPolicy{
  .uint_configs = kAnsUintConfigs,
};

uint32_t PopulationCountPrecision(uint32_t log_count, uint32_t shift) {
  const int32_t precision = std::min<int32_t>(
    log_count,
    static_cast<int32_t>(shift) -
      static_cast<int32_t>((kAnsLogTableSize - log_count) >> 1));
  return precision < 0 ? 0 : static_cast<uint32_t>(precision);
}

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

double ExactCountLog2(uint64_t value) {
  static const std::array<double, kExactLog2TableSize + 1> table = [] {
    std::array<double, kExactLog2TableSize + 1> values{};
    for (size_t index = 1; index < values.size(); ++index) {
      values[index] = std::log2(static_cast<double>(index));
    }
    return values;
  }();
  return value <= kExactLog2TableSize
    ? table[static_cast<size_t>(value)]
    : std::log2(static_cast<double>(value));
}

class BitCountWriter {
public:
  [[nodiscard]] size_t bits_written() const noexcept {
    return bits_written_;
  }

  [[nodiscard]] Status WriteBits(size_t bit_count, uint64_t bits) {
    if (bit_count > BitWriter::kMaxBitsPerWrite) {
      return Status::InvalidArgument(
        "A bit-counter call cannot exceed 56 bits");
    }
    if ((bits >> bit_count) != 0) {
      return Status::InvalidArgument(
        "Bit-counter input has set bits outside the requested width");
    }
    if (bits_written_ > std::numeric_limits<size_t>::max() - bit_count) {
      return Status::OutOfMemory("Bit-counter size overflow");
    }
    bits_written_ += bit_count;
    return Status::Ok();
  }

private:
  size_t bits_written_ = 0;
};

template <typename Writer>
Status StoreVarLenUint8(size_t value, Writer* writer) {
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
  const std::array<uint64_t, kMaximumAnsAlphabetSize>& raw,
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

struct AllowedPopulation {
  int32_t count = 0;
  uint16_t step_log = 0;
  int32_t delta_log2 = 0;
};

struct AllowedPopulations {
  std::array<std::array<AllowedPopulation, kAnsTableSize>,
             kAnsLogTableSize> values{};
  std::array<std::array<uint16_t, kAnsTableSize>,
             kAnsLogTableSize> indexes{};
};

const std::array<uint32_t, kAnsTableSize + 1>& PopulationLog2Table() {
  static const auto table = [] {
    std::array<uint32_t, kAnsTableSize + 1> values{};
    for (size_t count = 1; count < values.size(); ++count) {
      values[count] = static_cast<uint32_t>(std::llround(
        std::ldexp(std::log2(static_cast<double>(count)) /
                     kAnsLogTableSize,
                   31)));
    }
    return values;
  }();
  return table;
}

uint32_t SmallestPopulationIncrementLog(uint32_t count, uint32_t shift) {
  if (count == 0) {
    return 0;
  }
  const uint32_t log_count = std::bit_width(count) - 1;
  return log_count - PopulationCountPrecision(log_count, shift);
}

const AllowedPopulations& GetAllowedPopulations() {
  static const auto allowed = [] {
    AllowedPopulations result;
    for (uint32_t shift = 0; shift < kAnsLogTableSize; ++shift) {
      auto& values = result.values[shift];
      auto& indexes = result.indexes[shift];
      int32_t last = -1;
      size_t slot = 0;
      values[0].delta_log2 = 0;
      values[0].step_log = 0;
      for (int32_t count = kAnsTableSize - 1; count >= 0; --count) {
        const int32_t current = count &
          ~((int32_t{1} << SmallestPopulationIncrementLog(count, shift)) - 1);
        if (current == last) {
          continue;
        }
        last = current;
        values[slot].count = current;
        indexes[current] = static_cast<uint16_t>(slot);
        if (current == 0) {
          values[slot].delta_log2 = std::numeric_limits<int32_t>::max();
          values[slot].step_log = 0;
        } else if (slot > 0) {
          const int32_t previous = values[slot - 1].count;
          values[slot].delta_log2 = static_cast<int32_t>(std::llround(
            std::ldexp(
              std::log2(static_cast<double>(previous) / current) /
                kAnsLogTableSize,
              31)));
          values[slot].step_log = static_cast<uint16_t>(
            std::bit_width(static_cast<uint32_t>(previous - current)) - 1);
        }
        ++slot;
      }
    }
    return result;
  }();
  return allowed;
}

bool RebalanceHistogram(
  const std::array<uint64_t, kMaximumAnsAlphabetSize>& raw,
  size_t alphabet_size,
  uint32_t shift,
  std::vector<uint16_t>* frequencies,
  uint16_t* omit_position) {

  if (frequencies == nullptr || omit_position == nullptr ||
      alphabet_size == 0 || shift >= kAnsLogTableSize) {
    return false;
  }
  uint64_t total = 0;
  for (size_t symbol = 0; symbol < alphabet_size; ++symbol) {
    if (raw[symbol] > static_cast<uint64_t>(
          std::numeric_limits<int32_t>::max()) ||
        total > std::numeric_limits<uint64_t>::max() - raw[symbol]) {
      return false;
    }
    total += raw[symbol];
  }
  if (total == 0) {
    return false;
  }

  // Greedily choose representable populations while one omitted (largest)
  // population balances the total to kAnsTableSize. This follows libjxl's
  // integer entropy-delta algorithm; only the initial approximation uses
  // floating point.
  struct EntropyDelta {
    int32_t frequency = 0;
    size_t count_index = 0;
    size_t symbol = 0;
  };
  const auto& log2_table = PopulationLog2Table();
  const auto& allowed = GetAllowedPopulations().values[shift];
  const auto& allowed_index = GetAllowedPopulations().indexes[shift];
  std::vector<int32_t> counts(alphabet_size, 0);
  std::vector<EntropyDelta> bins;
  bins.reserve(alphabet_size);
  const double scale = static_cast<double>(kAnsTableSize) /
    static_cast<double>(total);
  size_t remainder_position = 0;
  int64_t maximum_frequency = 0;
  int32_t rest = kAnsTableSize;
  for (size_t symbol = 0; symbol < alphabet_size; ++symbol) {
    const int32_t frequency = static_cast<int32_t>(raw[symbol]);
    if (frequency > maximum_frequency) {
      remainder_position = symbol;
      maximum_frequency = frequency;
    }
    const double target = frequency * scale;
    int32_t count = std::max<int32_t>(
      static_cast<int32_t>(std::round(target)), frequency > 0 ? 1 : 0);
    count = std::min<int32_t>(count, kAnsTableSize - 1);
    const uint32_t step_log = SmallestPopulationIncrementLog(count, shift);
    count &= ~((int32_t{1} << step_log) - 1);
    counts[symbol] = count;
    rest -= count;
    if (target > 1.0) {
      bins.push_back({frequency, allowed_index[count], symbol});
    }
  }
  bins.erase(std::remove_if(
    bins.begin(), bins.end(),
    [remainder_position](const EntropyDelta& delta) {
      return delta.symbol == remainder_position;
    }), bins.end());
  rest += counts[remainder_position];

  if (!bins.empty()) {
    std::array<int64_t, kAnsLogTableSize - 1> balance_increase{};
    std::array<int64_t, kAnsLogTableSize - 1> balance_decrease{};
    const uint32_t maximum_log = allowed[1].step_log;
    const auto increase_delta = [&](const EntropyDelta& delta) {
      return delta.frequency *
          static_cast<int64_t>(allowed[delta.count_index].delta_log2) -
        balance_increase[allowed[delta.count_index].step_log];
    };
    const auto decrease_delta = [&](const EntropyDelta& delta) {
      return delta.frequency *
          static_cast<int64_t>(allowed[delta.count_index + 1].delta_log2) -
        balance_decrease[allowed[delta.count_index + 1].step_log];
    };
    while (true) {
      for (uint32_t log = 0; log <= maximum_log; ++log) {
        const int32_t delta = int32_t{1} << log;
        if (rest >= static_cast<int32_t>(kAnsTableSize)) {
          balance_increase[log] = 0;
          balance_decrease[log] = 0;
        } else if (rest > 1) {
          balance_increase[log] = rest > delta
            ? maximum_frequency * static_cast<int64_t>(
                log2_table[rest] - log2_table[rest - delta])
            : std::numeric_limits<int64_t>::max();
          balance_decrease[log] = rest + delta < kAnsTableSize
            ? maximum_frequency * static_cast<int64_t>(
                log2_table[rest + delta] - log2_table[rest])
            : 0;
        } else {
          balance_increase[log] = std::numeric_limits<int64_t>::max();
          balance_decrease[log] = std::numeric_limits<int64_t>::max();
        }
      }
      auto best_increase = std::max_element(
        bins.begin(), bins.end(), [&](const EntropyDelta& left,
                                      const EntropyDelta& right) {
          return (increase_delta(left) >>
                  allowed[left.count_index].step_log) <
            (increase_delta(right) >>
             allowed[right.count_index].step_log);
        });
      if (increase_delta(*best_increase) > 0) {
        rest -= int32_t{1} << allowed[best_increase->count_index--].step_log;
      } else {
        auto best_decrease = std::min_element(
          bins.begin(), bins.end(), [&](const EntropyDelta& left,
                                        const EntropyDelta& right) {
            return (decrease_delta(left) >>
                    allowed[left.count_index + 1].step_log) <
              (decrease_delta(right) >>
               allowed[right.count_index + 1].step_log);
          });
        if (decrease_delta(*best_decrease) >= 0) {
          break;
        }
        rest += int32_t{1} << allowed[++best_decrease->count_index].step_log;
      }
    }
    for (const EntropyDelta& delta : bins) {
      counts[delta.symbol] = allowed[delta.count_index].count;
    }
    for (size_t symbol = 0; symbol < remainder_position; ++symbol) {
      if (counts[symbol] >= 2048) {
        counts[remainder_position] = counts[symbol];
        remainder_position = symbol;
        break;
      }
    }
  }
  counts[remainder_position] = rest;
  if (rest <= 0) {
    return false;
  }
  frequencies->resize(alphabet_size);
  for (size_t symbol = 0; symbol < alphabet_size; ++symbol) {
    if (counts[symbol] < 0 || counts[symbol] > kAnsTableSize) {
      return false;
    }
    (*frequencies)[symbol] = static_cast<uint16_t>(counts[symbol]);
  }
  *omit_position = static_cast<uint16_t>(remainder_position);
  return true;
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

Status BuildAnsEncoderTables(
  std::span<const uint16_t> frequencies,
  size_t log_alpha_size,
  std::vector<std::vector<uint16_t>>* reverse_maps,
  std::vector<uint64_t>* reciprocal_frequencies) {

  if (reverse_maps == nullptr || reciprocal_frequencies == nullptr) {
    return Status::InvalidArgument("ANS encoder-table output is null");
  }
  std::vector<std::vector<uint16_t>> candidate_reverse_maps(
    frequencies.size());
  std::vector<uint64_t> candidate_reciprocals(frequencies.size());
  for (size_t symbol = 0; symbol < frequencies.size(); ++symbol) {
    candidate_reverse_maps[symbol].resize(frequencies[symbol]);
    candidate_reciprocals[symbol] =
      codestream_internal::AnsFrequencyReciprocal(frequencies[symbol]);
  }
  if (frequencies.empty()) {
    *reverse_maps = std::move(candidate_reverse_maps);
    *reciprocal_frequencies = std::move(candidate_reciprocals);
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
    if (symbol >= candidate_reverse_maps.size() ||
        offset >= candidate_reverse_maps[symbol].size()) {
      return Status::Internal("ANS reverse-map construction failed");
    }
    candidate_reverse_maps[symbol][offset] = static_cast<uint16_t>(value);
  }
  *reverse_maps = std::move(candidate_reverse_maps);
  *reciprocal_frequencies = std::move(candidate_reciprocals);
  return Status::Ok();
}

Status WriteAnsUintConfig(
  HybridUintConfig config,
  size_t log_alpha_size,
  BitWriter* writer) {

  if (writer == nullptr || !config.valid() || log_alpha_size < 5 ||
      log_alpha_size > 8 ||
      config.split_exponent >=
        (size_t{1} << std::bit_width(log_alpha_size))) {
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

template <typename Writer>
Status WriteAnsHistogramTo(
  const AnsHistogram& histogram,
  Writer* writer) {

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

  if (histogram.method == 0) {
    if (Status status = writer->WriteBits(2, 2); !status.ok()) {
      return status;
    }
    return StoreVarLenUint8(histogram.frequencies.size() - 1, writer);
  }

  if (Status status = writer->WriteBits(2, 0); !status.ok()) {
    return status;
  }
  const size_t method = histogram.method;
  if (method == 0 || method > kAnsLogTableSize) {
    return Status::InvalidArgument("ANS histogram method is invalid");
  }
  const size_t upper_bound_log = std::bit_width(kAnsLogTableSize) - 1;
  const size_t method_log = std::bit_width(method) - 1;
  if (Status status = writer->WriteBits(
        method_log, (size_t{1} << method_log) - 1);
      !status.ok()) {
    return status;
  }
  if (method_log != upper_bound_log) {
    if (Status status = writer->WriteBits(1, 0); !status.ok()) {
      return status;
    }
  }
  if (Status status = writer->WriteBits(
        method_log, ((size_t{1} << method_log) - 1) & method);
      !status.ok()) {
    return status;
  }
  if (Status status = StoreVarLenUint8(
        histogram.frequencies.size() - 3, writer);
      !status.ok()) {
    return status;
  }

  const size_t omit_position = histogram.omit_position;
  if (omit_position >= histogram.frequencies.size()) {
    return Status::InvalidArgument("ANS omit position is invalid");
  }
  if (histogram.frequencies.size() > kMaximumAnsAlphabetSize) {
    return Status::InvalidArgument("ANS histogram alphabet is too large");
  }
  std::array<uint8_t, kMaximumAnsAlphabetSize> bit_widths{};
  std::array<uint8_t, kMaximumAnsAlphabetSize> same{};
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
  constexpr uint8_t kMinimumRepetitions = 5;
  size_t run_start = 0;
  for (size_t symbol = 1; symbol <= histogram.frequencies.size(); ++symbol) {
    if (symbol == histogram.frequencies.size() ||
        symbol == omit_position ||
        symbol == omit_position + 1 ||
        histogram.frequencies[symbol] !=
          histogram.frequencies[run_start]) {
      same[run_start] = static_cast<uint8_t>(symbol - run_start);
      run_start = symbol;
    }
  }
  constexpr size_t kRepeatWidth = kAnsLogTableSize + 1;
  for (size_t symbol = 0; symbol < histogram.frequencies.size(); ++symbol) {
    const uint8_t width = bit_widths[symbol];
    if (width >= kDepths.size()) {
      return Status::Internal("ANS population width is invalid");
    }
    if (Status status = writer->WriteBits(kDepths[width], kBits[width]);
        !status.ok()) {
      return status;
    }
    if (same[symbol] >= kMinimumRepetitions) {
      if (Status status = writer->WriteBits(
            kDepths[kRepeatWidth], kBits[kRepeatWidth]);
          !status.ok()) {
        return status;
      }
      if (Status status = StoreVarLenUint8(
            same[symbol] - kMinimumRepetitions, writer);
          !status.ok()) {
        return status;
      }
      symbol += same[symbol] - 1;
    }
  }
  for (size_t symbol = 0; symbol < histogram.frequencies.size(); ++symbol) {
    if (symbol == omit_position || bit_widths[symbol] <= 1) {
      continue;
    }
    const size_t log_count = bit_widths[symbol] - 1;
    const size_t bit_count = PopulationCountPrecision(
      static_cast<uint32_t>(log_count), histogram.method - 1);
    const size_t drop_bits = log_count - bit_count;
    if (Status status = writer->WriteBits(
          bit_count,
          (histogram.frequencies[symbol] >> drop_bits) -
            (uint16_t{1} << bit_count));
        !status.ok()) {
      return status;
    }
    if (same[symbol] >= kMinimumRepetitions) {
      symbol += same[symbol] - 1;
    }
  }
  return Status::Ok();
}

Status WriteAnsHistogram(
  const AnsHistogram& histogram,
  BitWriter* writer) {

  return WriteAnsHistogramTo(histogram, writer);
}

Status EstimateAnsHistogramCost(
  const std::array<uint64_t, kMaximumAnsAlphabetSize>& raw,
  const AnsHistogram& histogram,
  double* cost) {

  if (cost == nullptr) {
    return Status::InvalidArgument("ANS histogram cost is null");
  }
  BitCountWriter model;
  if (Status status = WriteAnsHistogramTo(histogram, &model); !status.ok()) {
    return status;
  }
  double candidate_cost = static_cast<double>(model.bits_written());
  for (size_t symbol = 0; symbol < histogram.frequencies.size(); ++symbol) {
    if (raw[symbol] == 0) {
      continue;
    }
    if (histogram.frequencies[symbol] == 0) {
      return Status::Internal("ANS candidate omitted a populated symbol");
    }
    candidate_cost += static_cast<double>(raw[symbol]) *
      (kAnsLogTableSize -
       ExactCountLog2(histogram.frequencies[symbol]));
  }
  *cost = candidate_cost;
  return Status::Ok();
}

Status BuildBestAnsHistogram(
  const std::array<uint64_t, kMaximumAnsAlphabetSize>& raw,
  AnsHistogramSearch search,
  AnsHistogram* histogram,
  size_t* candidate_count = nullptr) {

  if (histogram == nullptr) {
    return Status::InvalidArgument("ANS histogram output is null");
  }
  size_t alphabet_size = raw.size();
  while (alphabet_size != 0 && raw[alphabet_size - 1] == 0) {
    --alphabet_size;
  }
  size_t populated = 0;
  for (size_t symbol = 0; symbol < alphabet_size; ++symbol) {
    populated += raw[symbol] != 0 ? 1 : 0;
  }
  if (populated <= 2) {
    if (Status status = NormalizeHistogram(raw, &histogram->frequencies);
        !status.ok()) {
      return status;
    }
    histogram->method = kAnsLogTableSize;
    histogram->omit_position = 0;
    if (candidate_count != nullptr) {
      *candidate_count = 1;
    }
    return Status::Ok();
  }

  double best_cost = std::numeric_limits<double>::infinity();
  AnsHistogram best;
  auto consider = [&](AnsHistogram candidate) -> Status {
    if (candidate_count != nullptr) {
      ++*candidate_count;
    }
    double candidate_cost = 0.0;
    if (Status status = EstimateAnsHistogramCost(
          raw, candidate, &candidate_cost);
        !status.ok()) {
      return status;
    }
    if (candidate_cost < best_cost) {
      best_cost = candidate_cost;
      best = std::move(candidate);
    }
    return Status::Ok();
  };

  AnsHistogram flat;
  flat.method = 0;
  flat.omit_position = 0;
  flat.frequencies.assign(
    alphabet_size,
    static_cast<uint16_t>(kAnsTableSize / alphabet_size));
  for (size_t symbol = 0; symbol < kAnsTableSize % alphabet_size; ++symbol) {
    ++flat.frequencies[symbol];
  }
  if (Status status = consider(std::move(flat)); !status.ok()) {
    return status;
  }
  std::array<bool, kAnsLogTableSize> shifts{};
  if (search == AnsHistogramSearch::kFast) {
    shifts[0] = true;
    shifts[kAnsLogTableSize / 2] = true;
    shifts[kAnsLogTableSize - 1] = true;
  } else {
    shifts = codestream_internal::DirectAnsHistogramPrecisionShifts(
      search == AnsHistogramSearch::kApproximate
        ? codestream_internal::DirectAnsEntropyMode::kBalanced
        : codestream_internal::DirectAnsEntropyMode::kHighDensity);
  }
  for (uint32_t shift = 0; shift < kAnsLogTableSize; ++shift) {
    if (!shifts[shift]) {
      continue;
    }
    AnsHistogram candidate;
    candidate.method = static_cast<uint8_t>(shift + 1);
    if (!RebalanceHistogram(
          raw, alphabet_size, shift, &candidate.frequencies,
          &candidate.omit_position)) {
      return Status::Internal("ANS histogram rebalancing failed");
    }
    if (Status status = consider(std::move(candidate)); !status.ok()) {
      return status;
    }
  }
  *histogram = std::move(best);
  return Status::Ok();
}

struct DirectAnsHistogram {
  std::array<uint64_t, kMaximumAnsAlphabetSize> counts{};
  uint64_t total_count = 0;
  uint64_t extra_bits = 0;
  uint32_t maximum_symbol = 0;
  double shannon_bits = 0.0;

  bool Add(const HybridUintToken& token) {
    const size_t symbol = token.symbol;
    if (symbol >= counts.size() ||
        counts[symbol] == std::numeric_limits<uint64_t>::max() ||
        total_count == std::numeric_limits<uint64_t>::max() ||
        extra_bits > std::numeric_limits<uint64_t>::max() -
          token.extra_bit_count) {
      return false;
    }
    ++counts[symbol];
    ++total_count;
    extra_bits += token.extra_bit_count;
    maximum_symbol = std::max<uint32_t>(maximum_symbol, token.symbol);
    return true;
  }

  bool AddHistogram(const DirectAnsHistogram& other) {
    if (total_count >
          std::numeric_limits<uint64_t>::max() - other.total_count ||
        extra_bits >
          std::numeric_limits<uint64_t>::max() - other.extra_bits) {
      return false;
    }
    const size_t alphabet_size = std::max<size_t>(
      total_count == 0 ? 0 : maximum_symbol + 1,
      other.total_count == 0 ? 0 : other.maximum_symbol + 1);
    for (size_t symbol = 0; symbol < alphabet_size; ++symbol) {
      if (counts[symbol] >
          std::numeric_limits<uint64_t>::max() - other.counts[symbol]) {
        return false;
      }
      counts[symbol] += other.counts[symbol];
    }
    total_count += other.total_count;
    extra_bits += other.extra_bits;
    maximum_symbol = std::max(maximum_symbol, other.maximum_symbol);
    return true;
  }
};

double DirectHistogramShannonBits(const DirectAnsHistogram& histogram) {
  if (histogram.total_count == 0) {
    return 0.0;
  }
  const double total = static_cast<double>(histogram.total_count);
  double bits = total * ExactCountLog2(histogram.total_count);
  const size_t alphabet_size = histogram.maximum_symbol + 1;
  for (uint64_t count :
       std::span(histogram.counts).first(alphabet_size)) {
    if (count != 0) {
      bits -= static_cast<double>(count) *
        ExactCountLog2(count);
    }
  }
  return bits;
}

Status DirectHistogramDistance(
  const DirectAnsHistogram& left,
  const DirectAnsHistogram& right,
  double* distance) {

  if (distance == nullptr) {
    return Status::InvalidArgument("Direct ANS distance output is null");
  }
  if (left.total_count >
      std::numeric_limits<uint64_t>::max() - right.total_count) {
    return Status::InvalidArgument("Direct ANS histogram count overflow");
  }
  const uint64_t total_count = left.total_count + right.total_count;
  if (total_count == 0) {
    *distance = 0.0;
    return Status::Ok();
  }
  double combined_bits = static_cast<double>(total_count) *
    ExactCountLog2(total_count);
  const size_t alphabet_size = std::max(
    left.total_count == 0 ? 0 : left.maximum_symbol + 1,
    right.total_count == 0 ? 0 : right.maximum_symbol + 1);
  for (size_t symbol = 0; symbol < alphabet_size; ++symbol) {
    if (left.counts[symbol] >
        std::numeric_limits<uint64_t>::max() - right.counts[symbol]) {
      return Status::InvalidArgument("Direct ANS histogram count overflow");
    }
    const uint64_t count = left.counts[symbol] + right.counts[symbol];
    if (count != 0) {
      combined_bits -= static_cast<double>(count) * ExactCountLog2(count);
    }
  }
  *distance = combined_bits - left.shannon_bits - right.shannon_bits;
  return Status::Ok();
}

Status CanonicalizeDirectClusters(
  const std::vector<DirectAnsHistogram>& input,
  std::vector<DirectAnsHistogram>* clustered,
  std::vector<uint32_t>* symbols) {

  if (clustered == nullptr || symbols == nullptr ||
      symbols->size() != input.size() || clustered->empty()) {
    return Status::InvalidArgument("Direct ANS cluster state is invalid");
  }
  std::vector<size_t> new_indexes(
    clustered->size(), std::numeric_limits<size_t>::max());
  size_t next_index = 0;
  for (size_t histogram = 0; histogram < input.size(); ++histogram) {
    if (input[histogram].total_count == 0) {
      continue;
    }
    const size_t cluster = (*symbols)[histogram];
    if (cluster >= clustered->size()) {
      return Status::Internal("Direct ANS cluster index is invalid");
    }
    if (new_indexes[cluster] == std::numeric_limits<size_t>::max()) {
      new_indexes[cluster] = next_index++;
    }
  }
  if (next_index == 0) {
    clustered->assign(1, DirectAnsHistogram{});
    std::fill(symbols->begin(), symbols->end(), 0);
    return Status::Ok();
  }
  std::vector<DirectAnsHistogram> reordered(next_index);
  for (size_t histogram = 0; histogram < input.size(); ++histogram) {
    if (input[histogram].total_count == 0) {
      (*symbols)[histogram] = 0;
      continue;
    }
    const size_t cluster = (*symbols)[histogram];
    const size_t new_index = new_indexes[cluster];
    if (new_index >= reordered.size() ||
        !reordered[new_index].AddHistogram(input[histogram])) {
      return Status::InvalidArgument("Direct ANS histogram count overflow");
    }
    (*symbols)[histogram] = static_cast<uint32_t>(new_index);
  }
  for (DirectAnsHistogram& histogram : reordered) {
    histogram.shannon_bits = DirectHistogramShannonBits(histogram);
  }
  *clustered = std::move(reordered);
  return Status::Ok();
}

Status FastClusterDirectAnsHistograms(
  const std::vector<DirectAnsHistogram>& input,
  std::vector<DirectAnsHistogram>* clustered,
  std::vector<uint32_t>* symbols) {

  if (input.empty() || clustered == nullptr || symbols == nullptr) {
    return Status::InvalidArgument("Direct ANS clustering input is invalid");
  }
  constexpr size_t kMaximumClusters = kMaximumPrefixClusters;
  constexpr double kMinimumDistinctDistance = 48.0;
  std::vector<DirectAnsHistogram> source = input;
  for (DirectAnsHistogram& histogram : source) {
    histogram.shannon_bits = DirectHistogramShannonBits(histogram);
  }
  clustered->clear();
  clustered->reserve(std::min(kMaximumClusters, source.size()));
  symbols->assign(source.size(), kMaximumClusters);
  std::vector<double> distances(
    source.size(), std::numeric_limits<double>::infinity());
  size_t largest_index = 0;
  for (size_t index = 0; index < source.size(); ++index) {
    if (source[index].total_count == 0) {
      (*symbols)[index] = 0;
      distances[index] = 0.0;
    } else if (source[index].total_count > source[largest_index].total_count) {
      largest_index = index;
    }
  }
  while (clustered->size() < std::min(kMaximumClusters, source.size())) {
    (*symbols)[largest_index] = static_cast<uint32_t>(clustered->size());
    clustered->push_back(source[largest_index]);
    distances[largest_index] = 0.0;
    largest_index = 0;
    for (size_t index = 0; index < source.size(); ++index) {
      if (distances[index] == 0.0) {
        continue;
      }
      double distance = 0.0;
      if (Status status = DirectHistogramDistance(
            source[index], clustered->back(), &distance);
          !status.ok()) {
        return status;
      }
      distances[index] = std::min(distances[index], distance);
      if (distances[index] > distances[largest_index]) {
        largest_index = index;
      }
    }
    if (distances[largest_index] < kMinimumDistinctDistance) {
      break;
    }
  }
  for (size_t index = 0; index < source.size(); ++index) {
    if ((*symbols)[index] != kMaximumClusters) {
      continue;
    }
    size_t best = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    for (size_t cluster = 0; cluster < clustered->size(); ++cluster) {
      double distance = 0.0;
      if (Status status = DirectHistogramDistance(
            source[index], (*clustered)[cluster], &distance);
          !status.ok()) {
        return status;
      }
      if (distance < best_distance) {
        best = cluster;
        best_distance = distance;
      }
    }
    if (!(*clustered)[best].AddHistogram(source[index])) {
      return Status::InvalidArgument("Direct ANS histogram count overflow");
    }
    (*clustered)[best].shannon_bits =
      DirectHistogramShannonBits((*clustered)[best]);
    (*symbols)[index] = static_cast<uint32_t>(best);
  }
  return CanonicalizeDirectClusters(source, clustered, symbols);
}

Status DirectAnsPopulationCost(
  const DirectAnsHistogram& histogram,
  double* cost) {

  if (cost == nullptr) {
    return Status::InvalidArgument("Direct ANS population cost is null");
  }
  AnsHistogram normalized;
  if (Status status = BuildBestAnsHistogram(
        histogram.counts, AnsHistogramSearch::kFast, &normalized);
      !status.ok()) {
    return status;
  }
  return EstimateAnsHistogramCost(histogram.counts, normalized, cost);
}

Status RefineBestDirectAnsClusters(
  const std::vector<DirectAnsHistogram>& input,
  std::vector<DirectAnsHistogram>* clustered,
  std::vector<uint32_t>* symbols) {

  if (clustered == nullptr || symbols == nullptr || clustered->empty()) {
    return Status::InvalidArgument("Direct ANS refinement input is invalid");
  }
  struct Pair {
    double cost = 0.0;
    uint32_t first = 0;
    uint32_t second = 0;
    uint32_t version = 0;

    bool operator<(const Pair& other) const {
      return std::tie(cost, first, second, version) >
        std::tie(other.cost, other.first, other.second, other.version);
    }
  };
  std::vector<double> costs(clustered->size());
  for (size_t index = 0; index < clustered->size(); ++index) {
    if (Status status = DirectAnsPopulationCost(
          (*clustered)[index], &costs[index]); !status.ok()) {
      return status;
    }
  }
  std::vector<uint32_t> versions(clustered->size(), 1);
  std::vector<uint32_t> renumbering(clustered->size());
  std::iota(renumbering.begin(), renumbering.end(), uint32_t{0});
  std::priority_queue<Pair> pairs;
  const auto enqueue = [&](uint32_t left, uint32_t right,
                           auto* queue) -> Status {
    DirectAnsHistogram merged = (*clustered)[left];
    if (!merged.AddHistogram((*clustered)[right])) {
      return Status::InvalidArgument("Direct ANS histogram count overflow");
    }
    double merged_cost = 0.0;
    if (Status status = DirectAnsPopulationCost(merged, &merged_cost);
        !status.ok()) {
      return status;
    }
    const double cost = merged_cost - costs[left] - costs[right];
    if (cost < 0.0) {
      queue->push({
        cost, std::min(left, right), std::max(left, right),
        std::max(versions[left], versions[right])});
    }
    return Status::Ok();
  };
  for (uint32_t left = 0; left < clustered->size(); ++left) {
    for (uint32_t right = left + 1; right < clustered->size(); ++right) {
      if (Status status = enqueue(left, right, &pairs); !status.ok()) {
        return status;
      }
    }
  }
  uint32_t next_version = 2;
  while (!pairs.empty()) {
    const Pair pair = pairs.top();
    pairs.pop();
    if (pair.version !=
          std::max(versions[pair.first], versions[pair.second]) ||
        versions[pair.first] == 0 || versions[pair.second] == 0) {
      continue;
    }
    if (!(*clustered)[pair.first].AddHistogram(
          (*clustered)[pair.second])) {
      return Status::InvalidArgument("Direct ANS histogram count overflow");
    }
    if (Status status = DirectAnsPopulationCost(
          (*clustered)[pair.first], &costs[pair.first]); !status.ok()) {
      return status;
    }
    for (uint32_t& cluster : renumbering) {
      if (cluster == pair.second) {
        cluster = pair.first;
      }
    }
    versions[pair.second] = 0;
    versions[pair.first] = next_version++;
    for (uint32_t other = 0; other < clustered->size(); ++other) {
      if (other == pair.first || versions[other] == 0) {
        continue;
      }
      if (Status status = enqueue(pair.first, other, &pairs); !status.ok()) {
        return status;
      }
    }
  }
  std::vector<uint32_t> reverse(clustered->size(), 0);
  size_t alive = 0;
  for (size_t index = 0; index < clustered->size(); ++index) {
    if (versions[index] == 0) {
      continue;
    }
    (*clustered)[alive] = (*clustered)[index];
    reverse[index] = static_cast<uint32_t>(alive++);
  }
  clustered->resize(alive);
  for (uint32_t& cluster : *symbols) {
    cluster = reverse[renumbering[cluster]];
  }
  return CanonicalizeDirectClusters(input, clustered, symbols);
}

Status PrepareDirectAnsPartition(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCodeOptions& options,
  codestream_internal::DirectAnsEntropyMode mode,
  std::span<const codestream_internal::PreparedFixedAnsCluster>
    fixed_context_populations,
  EntropyCode* partition,
  codestream_internal::PreparedEntropyClusters* prepared,
  EntropyWorkProfile* profile) {

  if (partition == nullptr || prepared == nullptr ||
      options.context_count == 0 || !options.uint_config.valid()) {
    return Status::InvalidArgument("Direct ANS partition output is invalid");
  }
  size_t histogram_count = options.context_count;
  if (!options.initial_context_map.empty()) {
    if (options.initial_context_map.size() != options.context_count ||
        options.initial_histogram_count == 0 ||
        options.initial_histogram_count > 256) {
      return Status::InvalidArgument("Initial entropy context map is invalid");
    }
    histogram_count = options.initial_histogram_count;
    for (uint8_t histogram : options.initial_context_map) {
      if (histogram >= histogram_count) {
        return Status::InvalidArgument(
          "Initial context-map entry is out of range");
      }
    }
  } else if (options.initial_histogram_count != 0) {
    return Status::InvalidArgument(
      "Initial histogram count requires an initial context map");
  }

  try {
    const ProfileClock::time_point histogram_begin = ProfileBegin(profile);
    std::vector<DirectAnsHistogram> histograms(histogram_count);
    if (!fixed_context_populations.empty()) {
      if (mode != codestream_internal::DirectAnsEntropyMode::kBalanced ||
          fixed_context_populations.size() != options.context_count) {
        return Status::InvalidArgument(
          "Prepared direct ANS populations are invalid");
      }
      for (size_t context = 0; context < options.context_count; ++context) {
        const auto& population = fixed_context_populations[context];
        if (population.maximum_symbol >= kMaximumAnsAlphabetSize) {
          return Status::InvalidArgument(
            "Prepared direct ANS symbol is out of range");
        }
        uint64_t count_sum = 0;
        uint64_t extra_bit_sum = 0;
        uint32_t actual_maximum_symbol = 0;
        for (size_t symbol = 0; symbol < population.counts.size(); ++symbol) {
          const uint64_t count = population.counts[symbol];
          if (count_sum > std::numeric_limits<uint64_t>::max() - count) {
            return Status::InvalidArgument(
              "Prepared direct ANS population overflows");
          }
          count_sum += count;
          if (count == 0) continue;
          actual_maximum_symbol = static_cast<uint32_t>(symbol);
          const uint32_t split_token =
            uint32_t{1} << options.uint_config.split_exponent;
          uint32_t extra_bit_count = 0;
          if (symbol >= split_token) {
            const uint32_t in_token =
              options.uint_config.msb_in_token +
              options.uint_config.lsb_in_token;
            const uint32_t exponent = options.uint_config.split_exponent +
              ((static_cast<uint32_t>(symbol) - split_token) >> in_token);
            extra_bit_count = exponent -
              options.uint_config.msb_in_token -
              options.uint_config.lsb_in_token;
          }
          if (count != 0 && extra_bit_count >
              (std::numeric_limits<uint64_t>::max() - extra_bit_sum) / count) {
            return Status::InvalidArgument(
              "Prepared direct ANS extra bits overflow");
          }
          extra_bit_sum += count * extra_bit_count;
        }
        if (count_sum != population.token_count ||
            extra_bit_sum != population.extra_bits ||
            (count_sum != 0 &&
             actual_maximum_symbol != population.maximum_symbol)) {
          return Status::InvalidArgument(
            "Prepared direct ANS population count differs");
        }
        DirectAnsHistogram source{
          .counts = population.counts,
          .total_count = population.token_count,
          .extra_bits = population.extra_bits,
          .maximum_symbol = population.maximum_symbol,
        };
        if (options.initial_context_map.empty()) {
          histograms[context] = std::move(source);
        } else {
          const size_t histogram = options.initial_context_map[context];
          if (!histograms[histogram].AddHistogram(source)) {
            return Status::InvalidArgument("ANS histogram count overflow");
          }
        }
      }
    } else {
      for (const EntropyTokenStreamView section : section_tokens) {
        if (!section.valid()) {
          return Status::InvalidArgument("ANS token-stream view is invalid");
        }
        for (size_t index = 0; index < section.size(); ++index) {
          const EntropyToken token = section[index];
          if (token.context >= options.context_count) {
            return Status::InvalidArgument(
              "ANS token context is out of range");
          }
          HybridUintToken encoded;
          if (Status status = EncodeHybridUint(
                token.value, options.uint_config, &encoded);
              !status.ok()) {
            return status;
          }
          const size_t histogram = options.initial_context_map.empty()
            ? token.context
            : options.initial_context_map[token.context];
          if (!histograms[histogram].Add(encoded)) {
            return Status::InvalidArgument("ANS histogram count overflow");
          }
        }
      }
    }
    ProfileEnd(
      profile, histogram_begin,
      &EntropyWorkProfile::ans_histogram_build_nanoseconds);

    const ProfileClock::time_point clustering_begin = ProfileBegin(profile);
    std::vector<DirectAnsHistogram> clustered;
    std::vector<uint32_t> histogram_symbols;
    Status status = FastClusterDirectAnsHistograms(
      histograms, &clustered, &histogram_symbols);
    if (status.ok() &&
        mode == codestream_internal::DirectAnsEntropyMode::kHighDensity) {
      status = RefineBestDirectAnsClusters(
        histograms, &clustered, &histogram_symbols);
    }
    ProfileEnd(
      profile, clustering_begin,
      &EntropyWorkProfile::ans_histogram_build_nanoseconds);
    if (!status.ok()) {
      return status;
    }

    EntropyCode candidate_partition;
    candidate_partition.context_count = options.context_count;
    candidate_partition.context_map.resize(options.context_count);
    for (size_t context = 0; context < options.context_count; ++context) {
      const size_t initial = options.initial_context_map.empty()
        ? context
        : options.initial_context_map[context];
      if (initial >= histogram_symbols.size() ||
          histogram_symbols[initial] >= clustered.size()) {
        return Status::Internal("Direct ANS context map is incomplete");
      }
      candidate_partition.context_map[context] =
        static_cast<uint8_t>(histogram_symbols[initial]);
    }

    codestream_internal::PreparedEntropyClusters candidate_prepared;
    candidate_prepared.context_count = candidate_partition.context_count;
    candidate_prepared.context_map = candidate_partition.context_map;
    if (mode == codestream_internal::DirectAnsEntropyMode::kBalanced) {
      candidate_prepared.fixed_uint_config = options.uint_config;
      candidate_prepared.fixed_ans_clusters.resize(clustered.size());
      for (size_t cluster = 0; cluster < clustered.size(); ++cluster) {
        candidate_prepared.fixed_ans_clusters[cluster] = {
          .counts = clustered[cluster].counts,
          .token_count = clustered[cluster].total_count,
          .extra_bits = clustered[cluster].extra_bits,
          .maximum_symbol = clustered[cluster].maximum_symbol,
        };
      }
    } else {
      const ProfileClock::time_point value_begin = ProfileBegin(profile);
      std::vector<std::vector<uint32_t>> cluster_values(clustered.size());
      for (const EntropyTokenStreamView section : section_tokens) {
        for (size_t index = 0; index < section.size(); ++index) {
          const EntropyToken token = section[index];
          const size_t cluster =
            candidate_partition.context_map[token.context];
          cluster_values[cluster].push_back(token.value);
        }
      }
      ProfileEnd(
        profile, value_begin,
        &EntropyWorkProfile::ans_value_collection_nanoseconds);
      const ProfileClock::time_point aggregation_begin = ProfileBegin(profile);
      candidate_prepared.values.resize(clustered.size());
      for (size_t cluster = 0; cluster < clustered.size(); ++cluster) {
        if (Status aggregate = codestream_internal::AggregateEntropyValues(
              std::move(cluster_values[cluster]),
              &candidate_prepared.values[cluster]);
            !aggregate.ok()) {
          return aggregate;
        }
      }
      ProfileEnd(
        profile, aggregation_begin,
        &EntropyWorkProfile::ans_value_aggregation_nanoseconds);
    }
    *prepared = std::move(candidate_prepared);
    *partition = std::move(candidate_partition);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

template <typename EmitChunk>
Status AdvanceAnsState(
  const HybridUintToken& encoded,
  const AnsHistogram& histogram,
  EmitChunk&& emit_chunk,
  uint32_t* state) {

  if (state == nullptr || encoded.symbol >= histogram.frequencies.size() ||
      encoded.symbol >= histogram.reciprocal_frequencies.size() ||
      histogram.frequencies[encoded.symbol] == 0 ||
      histogram.reciprocal_frequencies[encoded.symbol] == 0) {
    return Status::InvalidArgument("ANS token symbol is absent");
  }
  if (encoded.extra_bit_count != 0) {
    emit_chunk(encoded.extra_bits, encoded.extra_bit_count);
  }
  const uint32_t frequency = histogram.frequencies[encoded.symbol];
  if ((*state >> (32 - kAnsLogTableSize)) >= frequency) {
    emit_chunk(*state & 0xFFFFu, 16);
    *state >>= 16;
  }
  // Renormalization guarantees state < frequency * 2^20. At 44-bit
  // reciprocal precision, the product fits uint64_t and its high bits are the
  // exact integer quotient.
  const uint32_t quotient =
    codestream_internal::DivideAnsStateByReciprocal(
      *state, histogram.reciprocal_frequencies[encoded.symbol]);
  const uint32_t remainder = *state - quotient * frequency;
  const std::vector<uint16_t>& reverse =
    histogram.reverse_maps[encoded.symbol];
  if (remainder >= reverse.size() || reverse[remainder] >= kAnsTableSize) {
    return Status::InvalidArgument("ANS reverse-map entry is invalid");
  }
  *state = (quotient << kAnsLogTableSize) + reverse[remainder];
  return Status::Ok();
}

template <typename EmitChunk>
Status ProcessAnsTokenStream(
  EntropyTokenStreamView tokens,
  const EntropyCode& code,
  EmitChunk&& emit_chunk,
  uint32_t* final_state) {

  uint32_t state = kAnsSignature << 16;
  for (size_t index = tokens.size(); index != 0; --index) {
    const EntropyToken token = tokens[index - 1];
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
    if (Status status = AdvanceAnsState(
          encoded, code.ans_histograms[cluster], emit_chunk, &state);
        !status.ok()) {
      return status;
    }
  }
  *final_state = state;
  return Status::Ok();
}

Status CountAnsTokenStreamBitsInternal(
  EntropyTokenStreamView tokens,
  const EntropyCode& code,
  uint64_t* bit_count) {

  if (bit_count == nullptr || !tokens.valid()) {
    return Status::InvalidArgument("ANS token cost output is null");
  }
  // Each token emits at most its 31 HybridUint extra bits and one 16-bit
  // renormalization chunk. Every section also emits its 32-bit final state.
  constexpr uint64_t kMaximumBitsPerToken = 31 + 16;
  if (tokens.size() >
      (std::numeric_limits<uint64_t>::max() - 32) /
        kMaximumBitsPerToken) {
    return Status::InvalidArgument("ANS token cost overflow");
  }
  uint64_t candidate = 32;
  uint32_t final_state = 0;
  const auto count_chunk = [&candidate](uint32_t, uint8_t chunk_bits) {
    candidate += chunk_bits;
  };
  if (Status status = ProcessAnsTokenStream(
        tokens, code, count_chunk, &final_state);
      !status.ok()) {
    return status;
  }
  *bit_count = candidate;
  return Status::Ok();
}

Status MeasureAnsCode(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCode& code,
  uint64_t model_bits,
  EntropyCodeCost* cost) {

  if (cost == nullptr) {
    return Status::InvalidArgument("ANS cost output is null");
  }
  EntropyCodeCost candidate;
  candidate.model_bits = model_bits;
  candidate.cluster_count = code.ans_histograms.size();
  candidate.section_token_bits.reserve(section_tokens.size());
  for (const EntropyTokenStreamView section : section_tokens) {
    uint64_t section_bits = 0;
    if (Status status = CountAnsTokenStreamBitsInternal(
          section, code, &section_bits);
        !status.ok()) {
      return status;
    }
    if (candidate.token_bits >
        std::numeric_limits<uint64_t>::max() - section_bits) {
      return Status::InvalidArgument("ANS token cost overflow");
    }
    candidate.token_bits += section_bits;
    candidate.section_token_bits.push_back(section_bits);
  }
  *cost = std::move(candidate);
  return Status::Ok();
}

bool HasEquivalentAnsSymbolCoding(
  const EntropyCode& left,
  const EntropyCode& right) {

  if (left.mode != EntropyCodingMode::kAns ||
      right.mode != EntropyCodingMode::kAns ||
      left.context_count != right.context_count ||
      left.context_map != right.context_map ||
      left.uint_configs != right.uint_configs ||
      left.ans_histograms.size() != right.ans_histograms.size()) {
    return false;
  }
  for (size_t cluster = 0; cluster < left.ans_histograms.size(); ++cluster) {
    const AnsHistogram& left_histogram = left.ans_histograms[cluster];
    const AnsHistogram& right_histogram = right.ans_histograms[cluster];
    if (left_histogram.frequencies != right_histogram.frequencies) {
      return false;
    }
  }
  return true;
}

bool HasEquivalentAnsTokenCoding(
  const EntropyCode& left,
  const EntropyCode& right) {

  if (!HasEquivalentAnsSymbolCoding(left, right)) {
    return false;
  }
  for (size_t cluster = 0; cluster < left.ans_histograms.size(); ++cluster) {
    if (left.ans_histograms[cluster].reverse_maps !=
        right.ans_histograms[cluster].reverse_maps) {
      return false;
    }
  }
  return true;
}

Status MeasureAnsCodes(
  std::span<const EntropyTokenStreamView> section_tokens,
  std::span<const EntropyCode* const> codes,
  std::span<const uint64_t> model_bits,
  std::span<EntropyCodeCost> costs) {

  // Compatible widths encode each token to the same HybridUint symbol and
  // normalized frequency. Traverse the ordered tokens once, but retain one
  // exact rANS state and reverse map per width because those tables can change
  // renormalization decisions.
  if (codes.empty() || codes.size() > kAnsAlphabetWidthCount ||
      model_bits.size() != codes.size() || costs.size() != codes.size()) {
    return Status::InvalidArgument("ANS survivor dimensions are invalid");
  }
  for (size_t candidate = 0; candidate < codes.size(); ++candidate) {
    if (codes[candidate] == nullptr ||
        !HasEquivalentAnsSymbolCoding(*codes[0], *codes[candidate])) {
      return Status::InvalidArgument("ANS survivor symbols differ");
    }
    costs[candidate] = {
      .model_bits = model_bits[candidate],
      .token_bits = 0,
      .cluster_count = codes[candidate]->ans_histograms.size(),
      .section_token_bits = std::vector<uint64_t>(section_tokens.size()),
    };
  }
  if (codes.size() == 1) {
    return MeasureAnsCode(
      section_tokens, *codes[0], model_bits[0], &costs[0]);
  }

  constexpr uint64_t kMaximumBitsPerToken = 31 + 16;
  std::array<uint32_t, kAnsAlphabetWidthCount> states{};
  std::array<uint64_t, kAnsAlphabetWidthCount> section_bits{};
  const EntropyCode& reference = *codes[0];
  for (size_t section_index = 0; section_index < section_tokens.size();
       ++section_index) {
    const EntropyTokenStreamView section = section_tokens[section_index];
    if (!section.valid()) {
      return Status::InvalidArgument("ANS token-stream view is invalid");
    }
    if (section.size() >
        (std::numeric_limits<uint64_t>::max() - 32) /
          kMaximumBitsPerToken) {
      return Status::InvalidArgument("ANS token cost overflow");
    }
    std::fill_n(states.begin(), codes.size(), kAnsSignature << 16);
    std::fill_n(section_bits.begin(), codes.size(), uint64_t{32});
    for (size_t token_index = section.size(); token_index != 0;
         --token_index) {
      const EntropyToken token = section[token_index - 1];
      if (token.context >= reference.context_count) {
        return Status::InvalidArgument("ANS token context is out of range");
      }
      const size_t cluster = reference.context_map[token.context];
      HybridUintToken encoded;
      if (Status status = EncodeHybridUint(
            token.value, reference.uint_configs[cluster], &encoded);
          !status.ok()) {
        return status;
      }
      for (size_t candidate = 0; candidate < codes.size(); ++candidate) {
        const auto count_chunk = [&section_bits, candidate](
                                   uint32_t, uint8_t chunk_bits) {
          section_bits[candidate] += chunk_bits;
        };
        if (Status status = AdvanceAnsState(
              encoded, codes[candidate]->ans_histograms[cluster],
              count_chunk, &states[candidate]);
            !status.ok()) {
          return status;
        }
      }
    }
    for (size_t candidate = 0; candidate < codes.size(); ++candidate) {
      if (costs[candidate].token_bits >
          std::numeric_limits<uint64_t>::max() - section_bits[candidate]) {
        return Status::InvalidArgument("ANS token cost overflow");
      }
      costs[candidate].token_bits += section_bits[candidate];
      costs[candidate].section_token_bits[section_index] =
        section_bits[candidate];
    }
  }
  return Status::Ok();
}

}  // namespace

Status codestream_internal::CountAnsTokenStreamBits(
  EntropyTokenStreamView tokens,
  const EntropyCode& code,
  uint64_t* bit_count) {

  return CountAnsTokenStreamBitsInternal(tokens, code, bit_count);
}

Status codestream_internal::CountAnsTokenStreamBits(
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  uint64_t* bit_count) {
  return CountAnsTokenStreamBits(
    EntropyTokenStreamView::Interleaved(tokens), code, bit_count);
}

Status codestream_internal::AggregateEntropyValues(
  std::span<uint32_t> values,
  std::vector<WeightedValue>* aggregated) {

  if (aggregated == nullptr) {
    return Status::InvalidArgument("Aggregated entropy values are null");
  }
  // Large coefficient-token arrays contain many duplicate small values. Avoid
  // sorting every occurrence, while retaining the cheaper sort for inputs too
  // small to amortize zeroing and scanning the dense table.
  constexpr size_t kDenseValueCount = 1 << 16;
  constexpr size_t kMinimumCountingInput = 1 << 12;
  std::vector<WeightedValue> candidate;
  if (values.size() < kMinimumCountingInput) {
    std::ranges::sort(values);
    candidate.reserve(std::min(values.size(), kMaximumAnsAlphabetSize));
    for (uint32_t value : values) {
      if (candidate.empty() || candidate.back().value != value) {
        candidate.push_back({value, 1});
      } else if (candidate.back().count ==
                 std::numeric_limits<uint64_t>::max()) {
        return Status::InvalidArgument("Entropy value count overflow");
      } else {
        ++candidate.back().count;
      }
    }
    *aggregated = std::move(candidate);
    return Status::Ok();
  }

  std::vector<uint64_t> dense_counts(kDenseValueCount);
  // Keep uncommon larger raw values sparse so the dense allocation remains
  // bounded for arbitrary uint32_t inputs.
  std::unordered_map<uint32_t, uint64_t> sparse_counts;
  for (uint32_t value : values) {
    uint64_t& count = value < dense_counts.size()
      ? dense_counts[value]
      : sparse_counts[value];
    if (count == std::numeric_limits<uint64_t>::max()) {
      return Status::InvalidArgument("Entropy value count overflow");
    }
    ++count;
  }

  candidate.reserve(
    std::min(values.size(), dense_counts.size() + sparse_counts.size()));
  for (size_t value = 0; value < dense_counts.size(); ++value) {
    if (dense_counts[value] != 0) {
      candidate.push_back(
        {static_cast<uint32_t>(value), dense_counts[value]});
    }
  }
  const size_t dense_value_count = candidate.size();
  for (const auto& [value, count] : sparse_counts) {
    candidate.push_back({value, count});
  }
  std::ranges::sort(
    candidate.begin() + dense_value_count, candidate.end(), {},
    &WeightedValue::value);
  *aggregated = std::move(candidate);
  return Status::Ok();
}

Status codestream_internal::AggregateEntropyValues(
  std::vector<uint32_t> values,
  std::vector<WeightedValue>* aggregated) {

  return AggregateEntropyValues(std::span<uint32_t>(values), aggregated);
}

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
    if (!config.valid() || config.split_exponent >=
          (size_t{1} << std::bit_width(code.ans_log_alpha_size))) {
      return Status::InvalidArgument("ANS HybridUint config is invalid");
    }
    const AnsHistogram& histogram = code.ans_histograms[cluster];
    if (histogram.frequencies.size() > alphabet_limit ||
        histogram.reverse_maps.size() != histogram.frequencies.size() ||
        histogram.reciprocal_frequencies.size() !=
          histogram.frequencies.size() ||
        (!histogram.frequencies.empty() &&
         histogram.frequencies.back() == 0)) {
      return Status::InvalidArgument("ANS histogram dimensions are invalid");
    }
    size_t total = 0;
    size_t populated = 0;
    std::array<bool, kAnsTableSize> seen{};
    for (size_t symbol = 0; symbol < histogram.frequencies.size(); ++symbol) {
      const uint16_t frequency = histogram.frequencies[symbol];
      total += frequency;
      populated += frequency != 0 ? 1 : 0;
      if (histogram.reverse_maps[symbol].size() != frequency) {
        return Status::InvalidArgument("ANS reverse map is invalid");
      }
      if (histogram.reciprocal_frequencies[symbol] !=
          codestream_internal::AnsFrequencyReciprocal(frequency)) {
        return Status::InvalidArgument("ANS frequency reciprocal is invalid");
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
    if (histogram.method > kAnsLogTableSize) {
      return Status::InvalidArgument("ANS histogram method is invalid");
    }
    if (populated > 2) {
      if (histogram.method == 0) {
        const size_t alphabet_size = histogram.frequencies.size();
        const size_t base = kAnsTableSize / alphabet_size;
        const size_t remainder = kAnsTableSize % alphabet_size;
        for (size_t symbol = 0; symbol < alphabet_size; ++symbol) {
          const size_t expected = base + (symbol < remainder ? 1 : 0);
          if (histogram.frequencies[symbol] != expected) {
            return Status::InvalidArgument("ANS flat histogram is invalid");
          }
        }
      } else {
        if (histogram.omit_position >= histogram.frequencies.size() ||
            histogram.frequencies[histogram.omit_position] == 0) {
          return Status::InvalidArgument("ANS omit position is invalid");
        }
        const uint32_t shift = histogram.method - 1;
        for (size_t symbol = 0; symbol < histogram.frequencies.size();
             ++symbol) {
          const uint16_t frequency = histogram.frequencies[symbol];
          if (symbol == histogram.omit_position || frequency == 0) {
            continue;
          }
          const uint32_t log_count = std::bit_width(frequency) - 1;
          const uint32_t precision =
            PopulationCountPrecision(log_count, shift);
          const uint32_t dropped_bits = log_count - precision;
          if ((frequency & ((uint16_t{1} << dropped_bits) - 1)) != 0) {
            return Status::InvalidArgument(
              "ANS histogram population is not representable");
          }
        }
      }
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
  EntropyTokenStreamView tokens,
  const EntropyCode& code,
  BitWriter* writer) {

  if (writer == nullptr || !tokens.valid()) {
    return Status::InvalidArgument("ANS token-stream output is null");
  }
  try {
    std::vector<ReverseBitChunk> reverse_chunks;
    reverse_chunks.reserve(2 * tokens.size());
    const auto append_chunk = [&reverse_chunks](
                                uint32_t bits, uint8_t bit_count) {
      reverse_chunks.push_back({bits, bit_count});
    };
    uint32_t state = 0;
    if (Status status = ProcessAnsTokenStream(
          tokens, code, append_chunk, &state);
        !status.ok()) {
      return status;
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

Status codestream_internal::WriteAnsTokenStream(
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  BitWriter* writer) {
  return WriteAnsTokenStream(
    EntropyTokenStreamView::Interleaved(tokens), code, writer);
}

Status ValidatePreparedEntropyClusters(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCode& prefix_partition,
  const codestream_internal::PreparedEntropyClusters& prepared) {

  const bool has_values = !prepared.values.empty();
  const bool has_fixed = !prepared.fixed_ans_clusters.empty();
  const size_t cluster_count = has_values
    ? prepared.values.size()
    : prepared.fixed_ans_clusters.size();
  if (prepared.context_count != prefix_partition.context_count ||
      prepared.context_map != prefix_partition.context_map ||
      has_values == has_fixed || cluster_count == 0 ||
      (!prefix_partition.prefix_codes.empty() &&
       cluster_count != prefix_partition.prefix_codes.size()) ||
      std::ranges::any_of(
        prepared.context_map,
        [cluster_count](uint8_t cluster) {
          return cluster >= cluster_count;
        })) {
    return Status::InvalidArgument(
      "Prepared entropy clusters do not match the prefix partition");
  }
  uint64_t prepared_count = 0;
  if (has_values) {
    for (const auto& cluster : prepared.values) {
      uint32_t previous = 0;
      bool first = true;
      for (const codestream_internal::WeightedValue weighted_value : cluster) {
        if (weighted_value.count == 0 ||
            (!first && weighted_value.value <= previous) ||
            prepared_count > std::numeric_limits<uint64_t>::max() -
              weighted_value.count) {
          return Status::InvalidArgument(
            "Prepared entropy cluster values are invalid");
        }
        prepared_count += weighted_value.count;
        previous = weighted_value.value;
        first = false;
      }
    }
  } else {
    if (!prepared.fixed_uint_config.valid()) {
      return Status::InvalidArgument(
        "Prepared fixed ANS configuration is invalid");
    }
    for (const auto& cluster : prepared.fixed_ans_clusters) {
      uint64_t count_sum = 0;
      size_t maximum_symbol = 0;
      for (size_t symbol = 0; symbol < cluster.counts.size(); ++symbol) {
        if (cluster.counts[symbol] >
            std::numeric_limits<uint64_t>::max() - count_sum) {
          return Status::InvalidArgument(
            "Prepared fixed ANS population overflowed");
        }
        count_sum += cluster.counts[symbol];
        if (cluster.counts[symbol] != 0) maximum_symbol = symbol;
      }
      if (count_sum != cluster.token_count ||
          maximum_symbol != cluster.maximum_symbol ||
          prepared_count > std::numeric_limits<uint64_t>::max() -
            cluster.token_count) {
        return Status::InvalidArgument(
          "Prepared fixed ANS population is invalid");
      }
      prepared_count += cluster.token_count;
    }
  }
  uint64_t token_count = 0;
  for (const EntropyTokenStreamView section : section_tokens) {
    if (!section.valid() ||
        section.size() > std::numeric_limits<uint64_t>::max() - token_count) {
      return Status::InvalidArgument("ANS token-stream view is invalid");
    }
    token_count += section.size();
  }
  if (prepared_count != token_count) {
    return Status::InvalidArgument(
      "Prepared entropy cluster count differs from token streams");
  }
  return Status::Ok();
}

Status OptimizeAnsEntropyCodeImpl(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCode& prefix_partition,
  const codestream_internal::PreparedEntropyClusters* prepared,
  AnsOptimizationPolicy policy,
  EntropyCode* code,
  EntropyCodeCost* cost,
  codestream_internal::PreparedAnsEntropyCode* deferred,
  codestream_internal::EntropyWorkProfile* profile) {

  const bool direct_partition =
    prepared != nullptr && prefix_partition.prefix_codes.empty();
  if ((code == nullptr) == (deferred == nullptr) ||
      prefix_partition.mode != EntropyCodingMode::kPrefix ||
      (!direct_partition && prefix_partition.prefix_codes.empty())) {
    return Status::InvalidArgument("ANS partition input is invalid");
  }
  if (!direct_partition) {
    const ProfileClock::time_point prefix_validation_begin =
      ProfileBegin(profile);
    BitWriter validated_prefix;
    if (Status status = WriteEntropyCode(
          prefix_partition, &validated_prefix);
        !status.ok()) {
      return status;
    }
    ProfileEnd(
      profile, prefix_validation_begin,
      &EntropyWorkProfile::ans_prefix_validation_nanoseconds);
  }
  try {
    const bool prepared_fixed = direct_partition &&
      !prepared->fixed_ans_clusters.empty();
    const size_t cluster_count = direct_partition
      ? (prepared_fixed
          ? prepared->fixed_ans_clusters.size()
          : prepared->values.size())
      : prefix_partition.prefix_codes.size();
    std::vector<std::vector<uint32_t>> values;
    if (prepared != nullptr) {
      const ProfileClock::time_point validation_begin =
        ProfileBegin(profile);
      Status status = ValidatePreparedEntropyClusters(
        section_tokens, prefix_partition, *prepared);
      ProfileEnd(
        profile, validation_begin,
        &EntropyWorkProfile::ans_prepared_value_validation_nanoseconds);
      if (!status.ok()) {
        return status;
      }
    } else {
      const ProfileClock::time_point value_collection_begin =
        ProfileBegin(profile);
      values.resize(cluster_count);
      for (const EntropyTokenStreamView section : section_tokens) {
        if (!section.valid()) {
          return Status::InvalidArgument("ANS token-stream view is invalid");
        }
        for (size_t index = 0; index < section.size(); ++index) {
          const EntropyToken token = section[index];
          if (token.context >= prefix_partition.context_count) {
            return Status::InvalidArgument("ANS token context is out of range");
          }
          const size_t cluster = prefix_partition.context_map[token.context];
          if (cluster >= values.size()) {
            return Status::Internal("ANS cluster index is invalid");
          }
          values[cluster].push_back(token.value);
        }
      }
      ProfileEnd(
        profile, value_collection_begin,
        &EntropyWorkProfile::ans_value_collection_nanoseconds);
    }

    constexpr size_t kMinimumLogAlphaSize = 5;
    constexpr size_t kMaximumLogAlphaSize = 8;
    constexpr size_t kLogAlphaSizeCount =
      kMaximumLogAlphaSize - kMinimumLogAlphaSize + 1;
    struct ConfigWidthStats {
      bool valid = false;
      uint64_t config_bits = 0;
      double estimated_bits = 0.0;
    };
    struct ConfigCandidate {
      HybridUintConfig config;
      AnsHistogram histogram;
      size_t maximum_symbol = 0;
      uint64_t extra_bits = 0;
      double estimated_bits = 0.0;
      std::array<ConfigWidthStats, kLogAlphaSizeCount> width_stats;
    };
    std::vector<std::vector<ConfigCandidate>> options(cluster_count);
    for (size_t cluster = 0; cluster < cluster_count; ++cluster) {
      std::vector<codestream_internal::WeightedValue> weighted_values;
      std::span<const codestream_internal::WeightedValue> cluster_values;
      if (prepared_fixed) {
        // The balanced partition already retained the fixed-config symbol
        // population and does not require raw values.
      } else if (prepared != nullptr) {
        cluster_values = prepared->values[cluster];
      } else {
        const ProfileClock::time_point aggregation_begin =
          ProfileBegin(profile);
        if (Status status = codestream_internal::AggregateEntropyValues(
              std::move(values[cluster]), &weighted_values);
            !status.ok()) {
          return status;
        }
        ProfileEnd(
          profile, aggregation_begin,
          &EntropyWorkProfile::ans_value_aggregation_nanoseconds);
        cluster_values = weighted_values;
      }
      for (HybridUintConfig config : policy.uint_configs) {
        if (profile != nullptr) {
          ++profile->ans_uint_config_candidate_count;
        }
        const ProfileClock::time_point uint_config_begin =
          ProfileBegin(profile);
        std::array<uint64_t, kMaximumAnsAlphabetSize> counts{};
        uint64_t extra_bits = 0;
        size_t maximum_symbol = 0;
        bool valid = true;
        if (prepared_fixed) {
          if (policy.uint_configs.size() != 1 ||
              config != prepared->fixed_uint_config) {
            return Status::Internal(
              "Prepared fixed ANS configuration differs from policy");
          }
          const auto& fixed = prepared->fixed_ans_clusters[cluster];
          counts = fixed.counts;
          extra_bits = fixed.extra_bits;
          maximum_symbol = fixed.maximum_symbol;
        } else {
          for (const codestream_internal::WeightedValue& weighted_value :
               cluster_values) {
            HybridUintToken encoded;
            if (Status status = EncodeHybridUint(
                  weighted_value.value, config, &encoded);
                !status.ok()) {
              return status;
            }
            if (encoded.symbol >= kMaximumAnsAlphabetSize ||
                counts[encoded.symbol] >
                  std::numeric_limits<uint64_t>::max() -
                    weighted_value.count ||
                (encoded.extra_bit_count != 0 &&
                 weighted_value.count >
                   (std::numeric_limits<uint64_t>::max() - extra_bits) /
                     encoded.extra_bit_count)) {
              valid = false;
              break;
            }
            counts[encoded.symbol] += weighted_value.count;
            extra_bits += weighted_value.count * encoded.extra_bit_count;
            maximum_symbol = std::max<size_t>(
              maximum_symbol, encoded.symbol);
          }
        }
        ProfileEnd(
          profile, uint_config_begin,
          &EntropyWorkProfile::ans_uint_config_nanoseconds);
        if (!valid) {
          continue;
        }
        ConfigCandidate option;
        option.config = config;
        option.maximum_symbol = maximum_symbol;
        option.extra_bits = extra_bits;
        const ProfileClock::time_point histogram_begin =
          ProfileBegin(profile);
        size_t histogram_candidate_count = 0;
        if (Status status = BuildBestAnsHistogram(
              counts, policy.histogram_search, &option.histogram,
              &histogram_candidate_count);
            !status.ok()) {
          return status;
        }
        if (profile != nullptr) {
          profile->ans_histogram_candidate_count +=
            histogram_candidate_count;
        }
        ProfileEnd(
          profile, histogram_begin,
          &EntropyWorkProfile::ans_histogram_build_nanoseconds);
        const ProfileClock::time_point config_cost_begin =
          ProfileBegin(profile);
        if (Status status = EstimateAnsHistogramCost(
              counts, option.histogram, &option.estimated_bits);
            !status.ok()) {
          return status;
        }
        option.estimated_bits += static_cast<double>(extra_bits);
        for (size_t log_alpha_size = kMinimumLogAlphaSize;
             log_alpha_size <= kMaximumLogAlphaSize; ++log_alpha_size) {
          if (option.maximum_symbol >= (size_t{1} << log_alpha_size) ||
              option.config.split_exponent >=
                (size_t{1} << std::bit_width(log_alpha_size))) {
            continue;
          }
          BitWriter config_writer;
          if (Status status = WriteAnsUintConfig(
                option.config, log_alpha_size, &config_writer);
              !status.ok()) {
            return status;
          }
          ConfigWidthStats& stats =
            option.width_stats[log_alpha_size - kMinimumLogAlphaSize];
          stats.valid = true;
          stats.config_bits = config_writer.bits_written();
          stats.estimated_bits = option.estimated_bits +
            static_cast<double>(stats.config_bits);
        }
        ProfileEnd(
          profile, config_cost_begin,
          &EntropyWorkProfile::ans_uint_config_nanoseconds);
        options[cluster].push_back(std::move(option));
      }
      if (options[cluster].empty()) {
        return Status::InvalidArgument("No valid ANS HybridUint config");
      }
    }

    if (section_tokens.size() >
        std::numeric_limits<uint64_t>::max() / 32) {
      return Status::InvalidArgument("ANS token cost overflow");
    }
    const uint64_t minimum_section_bits =
      uint64_t{32} * section_tokens.size();
    std::vector<codestream_internal::PreparedAnsEntropyCandidate>
      width_candidates;
    width_candidates.reserve(kLogAlphaSizeCount);
    const ProfileClock::time_point model_build_begin =
      ProfileBegin(profile);
    for (size_t log_alpha_size = kMinimumLogAlphaSize;
         log_alpha_size <= kMaximumLogAlphaSize; ++log_alpha_size) {
      EntropyCode candidate;
      candidate.mode = EntropyCodingMode::kAns;
      candidate.context_count = prefix_partition.context_count;
      candidate.context_map = prefix_partition.context_map;
      candidate.ans_log_alpha_size = static_cast<uint8_t>(log_alpha_size);
      candidate.uint_configs.resize(cluster_count);
      candidate.ans_histograms.resize(cluster_count);
      uint64_t minimum_token_bits = minimum_section_bits;
      bool valid = true;
      for (size_t cluster = 0; cluster < cluster_count; ++cluster) {
        const ConfigCandidate* selected = nullptr;
        double selected_cost = std::numeric_limits<double>::infinity();
        for (const ConfigCandidate& option : options[cluster]) {
          const ConfigWidthStats& stats =
            option.width_stats[log_alpha_size - kMinimumLogAlphaSize];
          if (!stats.valid) {
            continue;
          }
          if (stats.estimated_bits < selected_cost) {
            selected = &option;
            selected_cost = stats.estimated_bits;
          }
        }
        if (selected == nullptr) {
          valid = false;
          break;
        }
        if (minimum_token_bits >
            std::numeric_limits<uint64_t>::max() - selected->extra_bits) {
          return Status::InvalidArgument("ANS token cost overflow");
        }
        minimum_token_bits += selected->extra_bits;
        candidate.uint_configs[cluster] = selected->config;
        candidate.ans_histograms[cluster] = selected->histogram;
        if (Status status = BuildAnsEncoderTables(
              candidate.ans_histograms[cluster].frequencies,
              log_alpha_size,
              &candidate.ans_histograms[cluster].reverse_maps,
              &candidate.ans_histograms[cluster].reciprocal_frequencies);
            !status.ok()) {
          return status;
        }
      }
      if (!valid) {
        continue;
      }
      BitWriter model;
      if (Status status = WriteEntropyCode(candidate, &model); !status.ok()) {
        return status;
      }
      width_candidates.push_back({
        std::move(candidate), model.bits_written(), minimum_token_bits, true});
      if (profile != nullptr) {
        ++profile->ans_alphabet_width_candidate_count;
      }
      if (policy.smallest_alphabet_width) {
        break;
      }
    }

    for (size_t candidate_index = 0;
         candidate_index < width_candidates.size(); ++candidate_index) {
      codestream_internal::PreparedAnsEntropyCandidate& candidate =
        width_candidates[candidate_index];
      if (!candidate.survives) {
        continue;
      }
      for (size_t other_index = candidate_index + 1;
           other_index < width_candidates.size(); ++other_index) {
        codestream_internal::PreparedAnsEntropyCandidate& other =
          width_candidates[other_index];
        if (!other.survives || !HasEquivalentAnsTokenCoding(
              candidate.code, other.code)) {
          continue;
        }
        if (other.model_bits < candidate.model_bits) {
          candidate.survives = false;
          break;
        }
        other.survives = false;
      }
    }
    ProfileEnd(
      profile, model_build_begin,
      &EntropyWorkProfile::ans_model_build_nanoseconds);

    if (deferred == nullptr && cost == nullptr &&
        policy.smallest_alphabet_width) {
      const auto selected = std::ranges::find_if(
        width_candidates,
        [](const auto& candidate) { return candidate.survives; });
      if (selected == width_candidates.end()) {
        return Status::InvalidArgument("No valid ANS alphabet size");
      }
      *code = std::move(selected->code);
      return Status::Ok();
    }

    if (deferred != nullptr) {
      if (std::ranges::none_of(
            width_candidates,
            [](const auto& candidate) { return candidate.survives; })) {
        return Status::InvalidArgument("No valid ANS alphabet size");
      }
      *deferred = {
        .candidates = std::move(width_candidates),
        .section_count = section_tokens.size(),
      };
      return Status::Ok();
    }

    EntropyCodeCost best_cost;
    uint64_t best_total = std::numeric_limits<uint64_t>::max();
    size_t best_candidate = std::numeric_limits<size_t>::max();
    std::array<bool, kAnsAlphabetWidthCount> grouped{};
    for (size_t group_seed = 0; group_seed < width_candidates.size();
         ++group_seed) {
      if (grouped[group_seed] || !width_candidates[group_seed].survives) {
        continue;
      }
      std::array<size_t, kAnsAlphabetWidthCount> candidate_indexes{};
      std::array<const EntropyCode*, kAnsAlphabetWidthCount> candidate_codes{};
      std::array<uint64_t, kAnsAlphabetWidthCount> candidate_model_bits{};
      size_t group_size = 0;
      for (size_t candidate_index = group_seed;
           candidate_index < width_candidates.size(); ++candidate_index) {
        codestream_internal::PreparedAnsEntropyCandidate& candidate =
          width_candidates[candidate_index];
        if (grouped[candidate_index] || !candidate.survives ||
            !HasEquivalentAnsSymbolCoding(
              width_candidates[group_seed].code, candidate.code)) {
          continue;
        }
        grouped[candidate_index] = true;
        const uint64_t candidate_lower_bound = candidate.model_bits >
            std::numeric_limits<uint64_t>::max() -
              candidate.minimum_token_bits
          ? std::numeric_limits<uint64_t>::max()
          : candidate.model_bits + candidate.minimum_token_bits;
        if (candidate_lower_bound > best_total ||
            (candidate_lower_bound == best_total &&
             candidate_index > best_candidate)) {
          continue;
        }
        candidate_indexes[group_size] = candidate_index;
        candidate_codes[group_size] = &candidate.code;
        candidate_model_bits[group_size] = candidate.model_bits;
        ++group_size;
      }
      if (group_size == 0) {
        continue;
      }
      std::array<EntropyCodeCost, kAnsAlphabetWidthCount> candidate_costs{};
      const ProfileClock::time_point token_cost_begin =
        ProfileBegin(profile);
      if (Status status = MeasureAnsCodes(
            section_tokens,
            std::span(candidate_codes).first(group_size),
            std::span(candidate_model_bits).first(group_size),
            std::span(candidate_costs).first(group_size));
          !status.ok()) {
        return status;
      }
      ProfileEnd(
        profile, token_cost_begin,
        &EntropyWorkProfile::ans_token_cost_nanoseconds);
      for (size_t group_index = 0; group_index < group_size; ++group_index) {
        const EntropyCodeCost& candidate_cost = candidate_costs[group_index];
        const uint64_t candidate_total = candidate_cost.model_bits >
            std::numeric_limits<uint64_t>::max() - candidate_cost.token_bits
          ? std::numeric_limits<uint64_t>::max()
          : candidate_cost.model_bits + candidate_cost.token_bits;
        const size_t candidate_index = candidate_indexes[group_index];
        if (candidate_total < best_total ||
            (candidate_total == best_total &&
             candidate_index < best_candidate)) {
          best_total = candidate_total;
          best_candidate = candidate_index;
          best_cost = candidate_cost;
        }
      }
    }
    if (best_total == std::numeric_limits<uint64_t>::max() ||
        best_candidate >= width_candidates.size()) {
      return Status::InvalidArgument("No valid ANS alphabet size");
    }
    *code = std::move(width_candidates[best_candidate].code);
    if (cost != nullptr) {
      *cost = std::move(best_cost);
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
}

Status codestream_internal::PrepareAnsEntropyCodeWithPreparedClusters(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCode& prefix_partition,
  const PreparedEntropyClusters& prepared,
  PreparedAnsEntropyCode* deferred,
  EntropyWorkProfile* profile) {

  if (deferred == nullptr) {
    return Status::InvalidArgument("Deferred ANS output is null");
  }
  return OptimizeAnsEntropyCodeImpl(
    section_tokens, prefix_partition, &prepared,
    kMaximumCompressionAnsPolicy, nullptr, nullptr, deferred, profile);
}

Status codestream_internal::MeasurePreparedAnsEntropyCodeSection(
  EntropyTokenStreamView tokens,
  const PreparedAnsEntropyCode& prepared,
  std::span<uint64_t> candidate_bits) {

  if (!tokens.valid() || !tokens.split || prepared.candidates.empty() ||
      prepared.candidates.size() > kAnsAlphabetWidthCount ||
      candidate_bits.size() != prepared.candidates.size()) {
    return Status::InvalidArgument(
      "Prepared ANS section measurement is invalid");
  }
  if (tokens.size() >
      (std::numeric_limits<uint64_t>::max() - 32) / (31 + 16)) {
    return Status::InvalidArgument("ANS token cost overflow");
  }

  struct MeasurementGroup {
    const EntropyCode* reference = nullptr;
    std::array<size_t, kAnsAlphabetWidthCount> candidate_indexes{};
    size_t candidate_count = 0;
    std::array<uint32_t, kAnsAlphabetWidthCount> states{};
    std::array<uint64_t, kAnsAlphabetWidthCount> bits{};
  };
  std::array<MeasurementGroup, kAnsAlphabetWidthCount> groups{};
  size_t group_count = 0;
  std::array<bool, kAnsAlphabetWidthCount> grouped{};
  bool has_survivor = false;
  for (size_t group_seed = 0; group_seed < prepared.candidates.size();
       ++group_seed) {
    if (grouped[group_seed] || !prepared.candidates[group_seed].survives) {
      continue;
    }
    const EntropyCode& reference = prepared.candidates[group_seed].code;
    if (reference.mode != EntropyCodingMode::kAns ||
        reference.context_count == 0 ||
        reference.context_map.size() != reference.context_count ||
        reference.uint_configs.size() != reference.ans_histograms.size() ||
        reference.ans_histograms.empty()) {
      return Status::InvalidArgument("Prepared ANS candidate is invalid");
    }
    MeasurementGroup& group = groups[group_count++];
    group.reference = &reference;
    for (size_t candidate_index = group_seed;
         candidate_index < prepared.candidates.size(); ++candidate_index) {
      const PreparedAnsEntropyCandidate& candidate =
        prepared.candidates[candidate_index];
      if (grouped[candidate_index] || !candidate.survives ||
          !HasEquivalentAnsSymbolCoding(reference, candidate.code)) {
        continue;
      }
      grouped[candidate_index] = true;
      group.candidate_indexes[group.candidate_count++] = candidate_index;
    }
    if (group.candidate_count == 0) {
      return Status::Internal("Prepared ANS group is empty");
    }
    std::fill_n(
      group.states.begin(), group.candidate_count, kAnsSignature << 16);
    std::fill_n(group.bits.begin(), group.candidate_count, uint64_t{32});
    has_survivor = true;
  }
  if (!has_survivor) {
    return Status::InvalidArgument("Prepared ANS candidates have no survivor");
  }

  std::array<uint64_t, kAnsAlphabetWidthCount> measured{};
  for (size_t token_index = tokens.size(); token_index != 0; --token_index) {
    const size_t index = token_index - 1;
    const uint32_t value = tokens.values[index];
    for (size_t group_index = 0; group_index < group_count; ++group_index) {
      MeasurementGroup& group = groups[group_index];
      const EntropyCode& reference = *group.reference;
      const size_t context = tokens.contexts[index];
      if (context >= reference.context_count) {
        return Status::InvalidArgument("ANS token context is out of range");
      }
      const size_t cluster = reference.context_map[context];
      if (cluster >= reference.uint_configs.size()) {
        return Status::InvalidArgument("ANS cluster index is invalid");
      }
      HybridUintToken encoded;
      if (Status status = EncodeHybridUint(
            value, reference.uint_configs[cluster], &encoded);
          !status.ok()) {
        return status;
      }
      for (size_t lane = 0; lane < group.candidate_count; ++lane) {
        const size_t candidate_index = group.candidate_indexes[lane];
        const EntropyCode& candidate =
          prepared.candidates[candidate_index].code;
        if (cluster >= candidate.ans_histograms.size()) {
          return Status::InvalidArgument("ANS cluster index is invalid");
        }
        const auto count_chunk = [&group, lane](
                                   uint32_t, uint8_t chunk_bits) {
          group.bits[lane] += chunk_bits;
        };
        if (Status status = AdvanceAnsState(
              encoded, candidate.ans_histograms[cluster], count_chunk,
              &group.states[lane]);
            !status.ok()) {
          return status;
        }
      }
    }
  }
  for (size_t group_index = 0; group_index < group_count; ++group_index) {
    const MeasurementGroup& group = groups[group_index];
    for (size_t lane = 0; lane < group.candidate_count; ++lane) {
      measured[group.candidate_indexes[lane]] = group.bits[lane];
    }
  }
  std::ranges::copy(
    std::span<const uint64_t>(measured).first(prepared.candidates.size()),
    candidate_bits.begin());
  return Status::Ok();
}

Status codestream_internal::FinalizePreparedAnsEntropyCode(
  PreparedAnsEntropyCode* prepared,
  std::span<const uint64_t> section_candidate_bits,
  EntropyCode* code,
  EntropyCodeCost* cost) {

  if (prepared == nullptr || code == nullptr || prepared->candidates.empty() ||
      prepared->candidates.size() > kAnsAlphabetWidthCount) {
    return Status::InvalidArgument("Prepared ANS finalization is invalid");
  }
  const size_t candidate_count = prepared->candidates.size();
  if (prepared->section_count >
        std::numeric_limits<size_t>::max() / candidate_count ||
      section_candidate_bits.size() !=
        prepared->section_count * candidate_count) {
    return Status::InvalidArgument(
      "Prepared ANS section measurement is incomplete");
  }
  const size_t section_count = prepared->section_count;

  EntropyCodeCost best_cost;
  uint64_t best_total = std::numeric_limits<uint64_t>::max();
  size_t best_candidate = std::numeric_limits<size_t>::max();
  for (size_t candidate_index = 0; candidate_index < candidate_count;
       ++candidate_index) {
    const PreparedAnsEntropyCandidate& candidate =
      prepared->candidates[candidate_index];
    if (!candidate.survives) {
      continue;
    }
    EntropyCodeCost candidate_cost;
    candidate_cost.model_bits = candidate.model_bits;
    candidate_cost.cluster_count = candidate.code.ans_histograms.size();
    candidate_cost.section_token_bits.reserve(section_count);
    for (size_t section_index = 0; section_index < section_count;
         ++section_index) {
      const uint64_t section_bits =
        section_candidate_bits[section_index * candidate_count +
          candidate_index];
      if (candidate_cost.token_bits >
          std::numeric_limits<uint64_t>::max() - section_bits) {
        return Status::InvalidArgument("ANS token cost overflow");
      }
      candidate_cost.token_bits += section_bits;
      candidate_cost.section_token_bits.push_back(section_bits);
    }
    const uint64_t candidate_total = candidate.model_bits >
        std::numeric_limits<uint64_t>::max() - candidate_cost.token_bits
      ? std::numeric_limits<uint64_t>::max()
      : candidate.model_bits + candidate_cost.token_bits;
    if (candidate_total < best_total ||
        (candidate_total == best_total && candidate_index < best_candidate)) {
      best_total = candidate_total;
      best_candidate = candidate_index;
      best_cost = std::move(candidate_cost);
    }
  }
  if (best_total == std::numeric_limits<uint64_t>::max() ||
      best_candidate >= candidate_count) {
    return Status::InvalidArgument("No valid ANS alphabet size");
  }
  EntropyCode selected =
    std::move(prepared->candidates[best_candidate].code);
  *prepared = {};
  *code = std::move(selected);
  if (cost != nullptr) {
    *cost = std::move(best_cost);
  }
  return Status::Ok();
}

Status OptimizeAnsEntropyCode(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCode& prefix_partition,
  EntropyCode* code,
  EntropyCodeCost* cost,
  codestream_internal::EntropyWorkProfile* profile) {
  return OptimizeAnsEntropyCodeImpl(
    section_tokens, prefix_partition, nullptr, kMaximumCompressionAnsPolicy,
    code, cost, nullptr, profile);
}

Status codestream_internal::OptimizeDirectAnsEntropyCode(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCodeOptions& options,
  DirectAnsEntropyMode mode,
  EntropyCode* code,
  EntropyCodeCost* cost,
  EntropyWorkProfile* profile) {

  if (mode == DirectAnsEntropyMode::kBalanced) {
    return OptimizeDirectAnsEntropyCodeWithFixedPopulations(
      section_tokens, options, {}, code, cost, profile);
  }

  if (code == nullptr) {
    return Status::InvalidArgument("Direct ANS output is null");
  }
  switch (mode) {
    case DirectAnsEntropyMode::kBalanced:
    case DirectAnsEntropyMode::kHighDensity:
      break;
    default:
      return Status::InvalidArgument("Direct ANS mode is invalid");
  }
  EntropyCode partition;
  PreparedEntropyClusters prepared;
  Status status = PrepareDirectAnsPartition(
    section_tokens, options, mode, {}, &partition, &prepared, profile);
  if (!status.ok()) {
    return status;
  }
  const std::array<HybridUintConfig, 1> balanced_configs = {
    options.uint_config};
  const AnsOptimizationPolicy policy = mode == DirectAnsEntropyMode::kBalanced
    ? AnsOptimizationPolicy{
        .uint_configs = balanced_configs,
        .histogram_search = AnsHistogramSearch::kApproximate,
        .smallest_alphabet_width = true,
      }
    : AnsOptimizationPolicy{
        .uint_configs = HighDensityAnsUintConfigs(),
        .histogram_search = AnsHistogramSearch::kPrecise,
        .smallest_alphabet_width = true,
      };
  return OptimizeAnsEntropyCodeImpl(
    section_tokens, partition, &prepared, policy,
    code, cost, nullptr, profile);
}

Status codestream_internal::OptimizeDirectAnsEntropyCodeWithFixedPopulations(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCodeOptions& options,
  std::span<const PreparedFixedAnsCluster> context_populations,
  EntropyCode* code,
  EntropyCodeCost* cost,
  EntropyWorkProfile* profile) {

  if (code == nullptr) {
    return Status::InvalidArgument("Direct ANS output is null");
  }
  EntropyCode partition;
  PreparedEntropyClusters prepared;
  Status status = PrepareDirectAnsPartition(
    section_tokens, options, DirectAnsEntropyMode::kBalanced,
    context_populations, &partition, &prepared, profile);
  if (!status.ok()) return status;
  const std::array<HybridUintConfig, 1> balanced_configs = {
    options.uint_config};
  const AnsOptimizationPolicy policy{
    .uint_configs = balanced_configs,
    .histogram_search = AnsHistogramSearch::kApproximate,
    .smallest_alphabet_width = true,
  };
  return OptimizeAnsEntropyCodeImpl(
    section_tokens, partition, &prepared, policy,
    code, cost, nullptr, profile);
}

std::span<const HybridUintConfig>
codestream_internal::HighDensityAnsUintConfigs() noexcept {
  return kHighDensityAnsUintConfigs;
}

std::array<bool, codestream_internal::kAnsHistogramPrecisionShiftCount>
codestream_internal::DirectAnsHistogramPrecisionShifts(
  DirectAnsEntropyMode mode) noexcept {
  std::array<bool, kAnsLogTableSize> shifts{};
  if (mode == DirectAnsEntropyMode::kBalanced) {
    for (size_t shift = 0; shift < kAnsLogTableSize; shift += 2) {
      shifts[shift] = true;
    }
    shifts[kAnsLogTableSize - 1] = true;
  } else if (mode == DirectAnsEntropyMode::kHighDensity) {
    shifts.fill(true);
  }
  return shifts;
}

Status codestream_internal::OptimizeAnsEntropyCodeWithPreparedClusters(
  std::span<const EntropyTokenStreamView> section_tokens,
  const EntropyCode& prefix_partition,
  const PreparedEntropyClusters& prepared,
  EntropyCode* code,
  EntropyCodeCost* cost,
  EntropyWorkProfile* profile) {
  return OptimizeAnsEntropyCodeImpl(
    section_tokens, prefix_partition, &prepared,
    kMaximumCompressionAnsPolicy, code, cost, nullptr, profile);
}

Status OptimizeAnsEntropyCode(
  std::span<const std::vector<EntropyToken>> section_tokens,
  const EntropyCode& prefix_partition,
  EntropyCode* code,
  EntropyCodeCost* cost,
  codestream_internal::EntropyWorkProfile* profile) {
  try {
    std::vector<EntropyTokenStreamView> views;
    views.reserve(section_tokens.size());
    for (const std::vector<EntropyToken>& section : section_tokens) {
      views.push_back(EntropyTokenStreamView::Interleaved(section));
    }
    return OptimizeAnsEntropyCode(
      views, prefix_partition, code, cost, profile);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
}

}  // namespace gjxl
