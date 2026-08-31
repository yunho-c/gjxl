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
#include <span>
#include <stdexcept>
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
constexpr uint32_t kAnsSignature = 0x13;
constexpr size_t kAnsAlphabetWidthCount = 4;
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
  std::vector<uint8_t> bit_widths(histogram.frequencies.size(), 0);
  std::vector<uint8_t> same(histogram.frequencies.size(), 0);
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
  for (size_t symbol = 1; symbol <= bit_widths.size(); ++symbol) {
    if (symbol == bit_widths.size() || symbol == omit_position ||
        symbol == omit_position + 1 ||
        histogram.frequencies[symbol] !=
          histogram.frequencies[run_start]) {
      same[run_start] = static_cast<uint8_t>(symbol - run_start);
      run_start = symbol;
    }
  }
  constexpr size_t kRepeatWidth = kAnsLogTableSize + 1;
  for (size_t symbol = 0; symbol < bit_widths.size(); ++symbol) {
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

Status EstimateAnsHistogramCost(
  const std::array<uint64_t, kMaximumAnsAlphabetSize>& raw,
  const AnsHistogram& histogram,
  double* cost) {

  if (cost == nullptr) {
    return Status::InvalidArgument("ANS histogram cost is null");
  }
  BitWriter model;
  if (Status status = WriteAnsHistogram(histogram, &model); !status.ok()) {
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
       std::log2(static_cast<double>(histogram.frequencies[symbol])));
  }
  *cost = candidate_cost;
  return Status::Ok();
}

Status BuildBestAnsHistogram(
  const std::array<uint64_t, kMaximumAnsAlphabetSize>& raw,
  AnsHistogram* histogram) {

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
    return Status::Ok();
  }

  double best_cost = std::numeric_limits<double>::infinity();
  AnsHistogram best;
  auto consider = [&](AnsHistogram candidate) -> Status {
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
  for (uint32_t shift = 0; shift < kAnsLogTableSize; ++shift) {
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

template <typename EmitChunk>
Status AdvanceAnsState(
  const HybridUintToken& encoded,
  const AnsHistogram& histogram,
  EmitChunk&& emit_chunk,
  uint32_t* state) {

  if (state == nullptr || encoded.symbol >= histogram.frequencies.size() ||
      histogram.frequencies[encoded.symbol] == 0) {
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
  const uint32_t quotient = *state / frequency;
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
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  EmitChunk&& emit_chunk,
  uint32_t* final_state) {

  uint32_t state = kAnsSignature << 16;
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
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  uint64_t* bit_count) {

  if (bit_count == nullptr) {
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
  std::span<const std::vector<EntropyToken>> section_tokens,
  const EntropyCode& code,
  uint64_t model_bits,
  EntropyCodeCost* cost) {

  if (cost == nullptr) {
    return Status::InvalidArgument("ANS cost output is null");
  }
  EntropyCodeCost candidate;
  candidate.model_bits = model_bits;
  candidate.cluster_count = code.ans_histograms.size();
  for (const std::vector<EntropyToken>& section : section_tokens) {
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
  }
  *cost = candidate;
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
  std::span<const std::vector<EntropyToken>> section_tokens,
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
      model_bits[candidate], 0, codes[candidate]->ans_histograms.size()};
  }
  if (codes.size() == 1) {
    return MeasureAnsCode(
      section_tokens, *codes[0], model_bits[0], &costs[0]);
  }

  constexpr uint64_t kMaximumBitsPerToken = 31 + 16;
  std::array<uint32_t, kAnsAlphabetWidthCount> states{};
  std::array<uint64_t, kAnsAlphabetWidthCount> section_bits{};
  const EntropyCode& reference = *codes[0];
  for (const std::vector<EntropyToken>& section : section_tokens) {
    if (section.size() >
        (std::numeric_limits<uint64_t>::max() - 32) /
          kMaximumBitsPerToken) {
      return Status::InvalidArgument("ANS token cost overflow");
    }
    std::fill_n(states.begin(), codes.size(), kAnsSignature << 16);
    std::fill_n(section_bits.begin(), codes.size(), uint64_t{32});
    for (size_t token_index = section.size(); token_index != 0;
         --token_index) {
      const EntropyToken& token = section[token_index - 1];
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
    }
  }
  return Status::Ok();
}

}  // namespace

Status codestream_internal::CountAnsTokenStreamBits(
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  uint64_t* bit_count) {

  return CountAnsTokenStreamBitsInternal(tokens, code, bit_count);
}

Status codestream_internal::AggregateEntropyValues(
  std::vector<uint32_t> values,
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
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  BitWriter* writer) {

  if (writer == nullptr) {
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

Status OptimizeAnsEntropyCode(
  std::span<const std::vector<EntropyToken>> section_tokens,
  const EntropyCode& prefix_partition,
  EntropyCode* code,
  EntropyCodeCost* cost,
  codestream_internal::EntropyWorkProfile* profile) {

  if (code == nullptr || prefix_partition.mode != EntropyCodingMode::kPrefix ||
      prefix_partition.prefix_codes.empty()) {
    return Status::InvalidArgument("ANS partition input is invalid");
  }
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
  try {
    const size_t cluster_count = prefix_partition.prefix_codes.size();
    const ProfileClock::time_point value_collection_begin =
      ProfileBegin(profile);
    std::vector<std::vector<uint32_t>> values(cluster_count);
    for (const std::vector<EntropyToken>& section : section_tokens) {
      for (const EntropyToken& token : section) {
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
      const ProfileClock::time_point aggregation_begin =
        ProfileBegin(profile);
      std::vector<codestream_internal::WeightedValue> weighted_values;
      if (Status status = codestream_internal::AggregateEntropyValues(
            std::move(values[cluster]), &weighted_values);
          !status.ok()) {
        return status;
      }
      ProfileEnd(
        profile, aggregation_begin,
        &EntropyWorkProfile::ans_value_aggregation_nanoseconds);
      for (HybridUintConfig config : kAnsUintConfigs) {
        const ProfileClock::time_point uint_config_begin =
          ProfileBegin(profile);
        std::array<uint64_t, kMaximumAnsAlphabetSize> counts{};
        uint64_t extra_bits = 0;
        size_t maximum_symbol = 0;
        bool valid = true;
        for (const codestream_internal::WeightedValue& weighted_value :
             weighted_values) {
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
          maximum_symbol = std::max<size_t>(maximum_symbol, encoded.symbol);
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
        if (Status status = BuildBestAnsHistogram(
              counts, &option.histogram);
            !status.ok()) {
          return status;
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
    struct WidthCandidate {
      EntropyCode code;
      uint64_t model_bits = 0;
      uint64_t minimum_token_bits = 0;
      bool survives = true;
    };
    std::vector<WidthCandidate> width_candidates;
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
        if (Status status = BuildReverseMaps(
              candidate.ans_histograms[cluster].frequencies,
              log_alpha_size,
              &candidate.ans_histograms[cluster].reverse_maps);
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
    }

    for (size_t candidate_index = 0;
         candidate_index < width_candidates.size(); ++candidate_index) {
      WidthCandidate& candidate = width_candidates[candidate_index];
      if (!candidate.survives) {
        continue;
      }
      for (size_t other_index = candidate_index + 1;
           other_index < width_candidates.size(); ++other_index) {
        WidthCandidate& other = width_candidates[other_index];
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
        WidthCandidate& candidate = width_candidates[candidate_index];
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
      *cost = best_cost;
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
}

}  // namespace gjxl
