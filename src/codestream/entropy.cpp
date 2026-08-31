// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's prefix entropy encoder and histogram
// clustering code.

#include "codestream/entropy.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "codestream/ans_internal.h"
#include "codestream/huffman.h"
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

constexpr size_t kCodeLengthCodeCount = 18;
constexpr std::array<HybridUintConfig, 4> kBalancedUintConfigs = {{
  {4, 2, 0},
  {4, 1, 2},
  {0, 0, 0},
  {2, 0, 1},
}};

struct Histogram {
  std::array<uint64_t, kPrefixAlphabetSize> counts{};
  uint64_t total_count = 0;
  size_t symbol_limit = 0;
  double bit_cost = 0.0;
  bool bit_cost_valid = false;

  bool Add(size_t symbol) {
    if (symbol >= counts.size() ||
        counts[symbol] == std::numeric_limits<uint64_t>::max() ||
        total_count == std::numeric_limits<uint64_t>::max()) {
      return false;
    }
    ++counts[symbol];
    ++total_count;
    symbol_limit = std::max(symbol_limit, symbol + 1);
    bit_cost_valid = false;
    return true;
  }

  bool AddHistogram(const Histogram& other) {
    if (total_count >
        std::numeric_limits<uint64_t>::max() - other.total_count) {
      return false;
    }
    for (size_t index = 0; index < other.symbol_limit; ++index) {
      if (counts[index] >
          std::numeric_limits<uint64_t>::max() - other.counts[index]) {
        return false;
      }
    }
    for (size_t index = 0; index < other.symbol_limit; ++index) {
      counts[index] += other.counts[index];
    }
    total_count += other.total_count;
    symbol_limit = std::max(symbol_limit, other.symbol_limit);
    bit_cost_valid = false;
    return true;
  }

};

uint8_t EncodedPrefixDepth(const PrefixCode& prefix, size_t symbol) {
  return prefix.degenerate_symbol == symbol ? 0 : prefix.depths[symbol];
}

Status HistogramCountsBitCost(
  std::span<const uint64_t> counts,
  uint64_t total_count,
  double* bit_cost) {

  if (bit_cost == nullptr || counts.size() > kPrefixAlphabetSize) {
    return Status::InvalidArgument("Invalid histogram cost input");
  }
  *bit_cost = 0.0;
  if (total_count == 0) {
    return Status::Ok();
  }
  std::array<uint8_t, kPrefixAlphabetSize> depths;
  if (Status status = codestream_internal::CreateHuffmanTree(
        counts, 15, std::span<uint8_t>(depths).first(counts.size()));
      !status.ok()) {
    return status;
  }
  size_t populated_symbols = 0;
  for (uint64_t count : counts) {
    populated_symbols += count != 0;
  }
  if (populated_symbols == 1) {
    return Status::Ok();
  }
  for (size_t index = 0; index < counts.size(); ++index) {
    *bit_cost += static_cast<double>(counts[index]) * depths[index];
  }
  return Status::Ok();
}

Status HistogramBitCost(Histogram* histogram) {
  if (histogram == nullptr) {
    return Status::InvalidArgument("Histogram output is null");
  }
  if (histogram->bit_cost_valid) {
    return Status::Ok();
  }
  if (Status status = HistogramCountsBitCost(
        std::span<const uint64_t>(histogram->counts).first(
          histogram->symbol_limit),
        histogram->total_count, &histogram->bit_cost);
      !status.ok()) {
    return status;
  }
  histogram->bit_cost_valid = true;
  return Status::Ok();
}

Status HistogramDistance(
  const Histogram& left,
  const Histogram& right,
  double* distance) {

  if (distance == nullptr) {
    return Status::InvalidArgument("Histogram distance output is null");
  }
  if (left.total_count == 0 || right.total_count == 0) {
    *distance = 0.0;
    return Status::Ok();
  }
  if (left.total_count >
      std::numeric_limits<uint64_t>::max() - right.total_count) {
    return Status::InvalidArgument("Combined histogram count overflow");
  }
  const size_t symbol_limit = std::max(
    left.symbol_limit, right.symbol_limit);
  std::array<uint64_t, kPrefixAlphabetSize> combined_counts;
  for (size_t symbol = 0; symbol < symbol_limit; ++symbol) {
    if (left.counts[symbol] >
        std::numeric_limits<uint64_t>::max() - right.counts[symbol]) {
      return Status::InvalidArgument("Combined histogram count overflow");
    }
    combined_counts[symbol] = left.counts[symbol] + right.counts[symbol];
  }
  double combined_bit_cost = 0.0;
  if (Status status = HistogramCountsBitCost(
        std::span<const uint64_t>(combined_counts).first(symbol_limit),
        left.total_count + right.total_count, &combined_bit_cost);
      !status.ok()) {
    return status;
  }
  *distance = combined_bit_cost - left.bit_cost - right.bit_cost;
  return Status::Ok();
}

double HistogramShapeDistance(
  const Histogram& left,
  const Histogram& right) {

  if (left.total_count == 0 || right.total_count == 0) {
    return 0.0;
  }
  // Normalized L1 distance is cheap enough to screen every candidate cap
  // before the exact prefix-cost refinement below.
  const double left_scale = 1.0 / static_cast<double>(left.total_count);
  const double right_scale = 1.0 / static_cast<double>(right.total_count);
  double distance = 0.0;
  const size_t symbol_limit = std::max(
    left.symbol_limit, right.symbol_limit);
  for (size_t symbol = 0; symbol < symbol_limit; ++symbol) {
    distance += std::abs(
      static_cast<double>(left.counts[symbol]) * left_scale -
      static_cast<double>(right.counts[symbol]) * right_scale);
  }
  return distance * static_cast<double>(
    std::min(left.total_count, right.total_count));
}

Status FastClusterHistograms(
  const std::vector<Histogram>& input,
  size_t maximum_histograms,
  bool fill_to_limit,
  std::vector<Histogram>* output,
  std::vector<uint32_t>* histogram_symbols,
  std::vector<size_t>* seed_indexes = nullptr) {

  if (input.empty() || maximum_histograms == 0 ||
      output == nullptr || histogram_symbols == nullptr) {
    return Status::InvalidArgument("Invalid histogram-clustering input");
  }
  output->clear();
  output->reserve(maximum_histograms);
  if (seed_indexes != nullptr) {
    seed_indexes->clear();
    seed_indexes->reserve(maximum_histograms);
  }
  histogram_symbols->assign(input.size(),
                            static_cast<uint32_t>(maximum_histograms));

  std::vector<double> distances(
    input.size(), std::numeric_limits<double>::max());
  // Shape-only seed discovery returns before exact assignment and never reads
  // bit_cost. Every other path needs prepared costs for HistogramDistance.
  const bool needs_exact_bit_costs =
    !fill_to_limit || seed_indexes == nullptr;
  size_t largest_index = 0;
  for (size_t index = 0; index < input.size(); ++index) {
    if (input[index].total_count == 0) {
      (*histogram_symbols)[index] = 0;
      distances[index] = 0.0;
      continue;
    }
    if (needs_exact_bit_costs && !input[index].bit_cost_valid) {
      return Status::Internal("Source histogram cost is not prepared");
    }
    if (input[index].total_count > input[largest_index].total_count) {
      largest_index = index;
    }
  }

  constexpr double kMinimumDistinctDistance = 64.0;
  while (output->size() < maximum_histograms) {
    (*histogram_symbols)[largest_index] =
      static_cast<uint32_t>(output->size());
    if (seed_indexes != nullptr) {
      seed_indexes->push_back(largest_index);
    }
    output->push_back(input[largest_index]);
    distances[largest_index] = 0.0;
    largest_index = 0;
    for (size_t index = 0; index < input.size(); ++index) {
      if (distances[index] == 0.0) {
        continue;
      }
      double distance = 0.0;
      if (fill_to_limit) {
        distance = HistogramShapeDistance(input[index], output->back());
      } else {
        if (Status status = HistogramDistance(
              input[index], output->back(), &distance);
            !status.ok()) {
          return status;
        }
      }
      distances[index] = std::min(distance, distances[index]);
      if (distances[index] > distances[largest_index]) {
        largest_index = index;
      }
    }
    if ((fill_to_limit && distances[largest_index] <= 0.0) ||
        (!fill_to_limit &&
         distances[largest_index] < kMinimumDistinctDistance)) {
      break;
    }
  }

  if (seed_indexes != nullptr) {
    return Status::Ok();
  }

  for (size_t index = 0; index < input.size(); ++index) {
    if ((*histogram_symbols)[index] != maximum_histograms) {
      continue;
    }
    size_t best = 0;
    double best_distance = 0.0;
    if (Status status = HistogramDistance(
          input[index], (*output)[best], &best_distance);
        !status.ok()) {
      return status;
    }
    for (size_t candidate = 1; candidate < output->size(); ++candidate) {
      double distance = 0.0;
      if (Status status = HistogramDistance(
            input[index], (*output)[candidate], &distance);
          !status.ok()) {
        return status;
      }
      if (distance < best_distance) {
        best = candidate;
        best_distance = distance;
      }
    }
    if (!(*output)[best].AddHistogram(input[index])) {
      return Status::InvalidArgument("Clustered histogram count overflow");
    }
    if (Status status = HistogramBitCost(&(*output)[best]); !status.ok()) {
      return status;
    }
    (*histogram_symbols)[index] = static_cast<uint32_t>(best);
  }
  return Status::Ok();
}

Status AssignHistogramsToSeeds(
  const std::vector<Histogram>& input,
  std::span<const size_t> seed_indexes,
  bool use_shape_distance,
  std::vector<Histogram>* output,
  std::vector<uint32_t>* histogram_symbols) {

  if (input.empty() || seed_indexes.empty() || output == nullptr ||
      histogram_symbols == nullptr) {
    return Status::InvalidArgument("Invalid histogram seed assignment");
  }
  // Shape distance uses only counts and total_count. Compaction or refinement
  // reconstructs any subsequently needed exact codes from those counts.
  if (!use_shape_distance) {
    for (const Histogram& histogram : input) {
      if (!histogram.bit_cost_valid) {
        return Status::Internal("Source histogram cost is not prepared");
      }
    }
  }
  output->clear();
  output->reserve(seed_indexes.size());
  histogram_symbols->assign(
    input.size(), static_cast<uint32_t>(seed_indexes.size()));
  for (size_t cluster = 0; cluster < seed_indexes.size(); ++cluster) {
    const size_t seed = seed_indexes[cluster];
    if (seed >= input.size() ||
        (input[seed].total_count == 0 && seed_indexes.size() != 1) ||
        (*histogram_symbols)[seed] != seed_indexes.size()) {
      return Status::Internal("Histogram seed is invalid");
    }
    output->push_back(input[seed]);
    (*histogram_symbols)[seed] = static_cast<uint32_t>(cluster);
  }
  for (size_t index = 0; index < input.size(); ++index) {
    if (input[index].total_count == 0) {
      (*histogram_symbols)[index] = 0;
      continue;
    }
    if ((*histogram_symbols)[index] != seed_indexes.size()) {
      continue;
    }
    size_t best = 0;
    double best_distance = 0.0;
    if (use_shape_distance) {
      best_distance = HistogramShapeDistance(input[index], (*output)[best]);
    } else {
      if (Status status = HistogramDistance(
            input[index], (*output)[best], &best_distance);
          !status.ok()) {
        return status;
      }
    }
    for (size_t candidate = 1; candidate < output->size(); ++candidate) {
      double distance = 0.0;
      if (use_shape_distance) {
        distance =
          HistogramShapeDistance(input[index], (*output)[candidate]);
      } else {
        if (Status status = HistogramDistance(
              input[index], (*output)[candidate], &distance);
            !status.ok()) {
          return status;
        }
      }
      if (distance < best_distance) {
        best = candidate;
        best_distance = distance;
      }
    }
    if (!(*output)[best].AddHistogram(input[index])) {
      return Status::InvalidArgument("Clustered histogram count overflow");
    }
    if (!use_shape_distance) {
      if (Status status = HistogramBitCost(&(*output)[best]); !status.ok()) {
        return status;
      }
    }
    (*histogram_symbols)[index] = static_cast<uint32_t>(best);
  }
  return Status::Ok();
}

Status CompactClusters(
  const std::vector<Histogram>& input,
  std::vector<Histogram>* clustered,
  std::vector<uint32_t>* symbols) {

  if (clustered == nullptr || symbols == nullptr ||
      symbols->size() != input.size()) {
    return Status::InvalidArgument("Invalid histogram compaction input");
  }
  std::vector<size_t> new_indexes(
    clustered->size(), std::numeric_limits<size_t>::max());
  size_t next_index = 0;
  for (size_t index = 0; index < input.size(); ++index) {
    if (input[index].total_count == 0) {
      continue;
    }
    const uint32_t symbol = (*symbols)[index];
    if (symbol >= clustered->size()) {
      return Status::Internal("Histogram cluster index is invalid");
    }
    if (new_indexes[symbol] == std::numeric_limits<size_t>::max()) {
      new_indexes[symbol] = next_index++;
    }
  }
  if (next_index == 0) {
    clustered->assign(1, Histogram{});
    std::fill(symbols->begin(), symbols->end(), 0);
    return Status::Ok();
  }

  std::vector<Histogram> reordered(next_index);
  for (size_t index = 0; index < input.size(); ++index) {
    if (input[index].total_count == 0) {
      (*symbols)[index] = 0;
      continue;
    }
    const size_t new_index = new_indexes[(*symbols)[index]];
    if (new_index >= reordered.size() ||
        !reordered[new_index].AddHistogram(input[index])) {
      return Status::InvalidArgument("Clustered histogram count overflow");
    }
    (*symbols)[index] = static_cast<uint32_t>(new_index);
  }
  for (Histogram& histogram : reordered) {
    if (Status status = HistogramBitCost(&histogram); !status.ok()) {
      return status;
    }
  }
  *clustered = std::move(reordered);
  return Status::Ok();
}

Status RefineHistogramClusters(
  const std::vector<Histogram>& input,
  size_t maximum_sweeps,
  std::vector<Histogram>* clustered,
  std::vector<uint32_t>* symbols) {

  if (clustered == nullptr || clustered->empty() || symbols == nullptr ||
      symbols->size() != input.size()) {
    return Status::InvalidArgument("Invalid histogram refinement input");
  }
  for (size_t sweep = 0; sweep < maximum_sweeps; ++sweep) {
    std::vector<PrefixCode> cluster_codes(clustered->size());
    for (size_t cluster = 0; cluster < clustered->size(); ++cluster) {
      if (Status status = BuildPrefixCode(
            std::span<const uint64_t>((*clustered)[cluster].counts).first(
              (*clustered)[cluster].symbol_limit),
            &cluster_codes[cluster]);
          !status.ok()) {
        return status;
      }
    }
    bool changed = false;
    for (size_t index = 0; index < input.size(); ++index) {
      if (input[index].total_count == 0) {
        continue;
      }
      const size_t source = (*symbols)[index];
      if (source >= clustered->size()) {
        return Status::Internal("Histogram refinement state is invalid");
      }

      size_t best = 0;
      uint64_t best_cost = std::numeric_limits<uint64_t>::max();
      for (size_t candidate = 0; candidate < clustered->size(); ++candidate) {
        uint64_t candidate_cost = 0;
        bool representable = true;
        for (size_t symbol = 0; symbol < input[index].symbol_limit; ++symbol) {
          const uint64_t count = input[index].counts[symbol];
          if (count == 0) {
            continue;
          }
          if (cluster_codes[candidate].depths[symbol] == 0) {
            representable = false;
            break;
          }
          const uint8_t depth =
            EncodedPrefixDepth(cluster_codes[candidate], symbol);
          if ((depth != 0 &&
               count > std::numeric_limits<uint64_t>::max() / depth) ||
              candidate_cost > std::numeric_limits<uint64_t>::max() -
                                 count * depth) {
            representable = false;
            break;
          }
          candidate_cost += count * depth;
        }
        if (representable && candidate_cost < best_cost) {
          best = candidate;
          best_cost = candidate_cost;
        }
      }
      if (best_cost == std::numeric_limits<uint64_t>::max()) {
        best = source;
      }
      changed = changed || best != source;
      (*symbols)[index] = static_cast<uint32_t>(best);
    }
    if (Status status = CompactClusters(input, clustered, symbols);
        !status.ok()) {
      return status;
    }
    if (!changed) {
      break;
    }
  }
  return Status::Ok();
}

Status ClusterHistograms(
  const std::vector<Histogram>& input,
  size_t maximum_histograms,
  bool fill_to_limit,
  size_t refinement_sweeps,
  std::vector<Histogram>* histograms,
  std::vector<uint8_t>* context_map) {

  if (histograms == nullptr || context_map == nullptr || input.empty() ||
      maximum_histograms == 0) {
    return Status::InvalidArgument("Invalid histogram-clustering output");
  }
  if (input.size() == 1) {
    *histograms = input;
    context_map->assign(1, 0);
    return Status::Ok();
  }

  maximum_histograms = std::min(maximum_histograms, input.size());
  std::vector<Histogram> clustered;
  std::vector<uint32_t> symbols;
  if (Status status = FastClusterHistograms(
        input, maximum_histograms, fill_to_limit, &clustered, &symbols);
      !status.ok()) {
    return status;
  }
  if (refinement_sweeps != 0) {
    if (Status status = RefineHistogramClusters(
          input, refinement_sweeps, &clustered, &symbols);
        !status.ok()) {
      return status;
    }
  }

  std::vector<size_t> new_indexes(
    clustered.size(), std::numeric_limits<size_t>::max());
  std::vector<Histogram> reordered(clustered.size());
  size_t next_index = 0;
  for (uint32_t symbol : symbols) {
    if (symbol >= clustered.size()) {
      return Status::Internal("Histogram cluster index is invalid");
    }
    if (new_indexes[symbol] == std::numeric_limits<size_t>::max()) {
      new_indexes[symbol] = next_index;
      reordered[next_index] = clustered[symbol];
      ++next_index;
    }
  }
  reordered.resize(next_index);
  context_map->resize(symbols.size());
  for (size_t index = 0; index < symbols.size(); ++index) {
    (*context_map)[index] =
      static_cast<uint8_t>(new_indexes[symbols[index]]);
  }
  *histograms = std::move(reordered);
  return Status::Ok();
}

Status ClusterHistogramsFromSeeds(
  const std::vector<Histogram>& input,
  std::span<const size_t> seed_indexes,
  bool use_shape_distance,
  size_t refinement_sweeps,
  std::vector<Histogram>* histograms,
  std::vector<uint8_t>* context_map) {

  if (histograms == nullptr || context_map == nullptr) {
    return Status::InvalidArgument("Invalid seeded histogram output");
  }
  std::vector<Histogram> clustered;
  std::vector<uint32_t> symbols;
  if (Status status = AssignHistogramsToSeeds(
        input, seed_indexes, use_shape_distance, &clustered, &symbols);
      !status.ok()) {
    return status;
  }
  if (refinement_sweeps != 0) {
    if (Status status = RefineHistogramClusters(
          input, refinement_sweeps, &clustered, &symbols);
        !status.ok()) {
      return status;
    }
  } else if (Status status = CompactClusters(input, &clustered, &symbols);
             !status.ok()) {
    return status;
  }
  context_map->resize(symbols.size());
  for (size_t index = 0; index < symbols.size(); ++index) {
    if (symbols[index] >= clustered.size()) {
      return Status::Internal("Histogram cluster index is invalid");
    }
    (*context_map)[index] = static_cast<uint8_t>(symbols[index]);
  }
  *histograms = std::move(clustered);
  return Status::Ok();
}

Status ValidateConfig(HybridUintConfig config) {
  if (!config.valid()) {
    return Status::InvalidArgument("Invalid HybridUint configuration");
  }
  return Status::Ok();
}

Status ValidateFullWidthConfig(HybridUintConfig config) {
  if (Status status = ValidateConfig(config); !status.ok()) {
    return status;
  }
  HybridUintToken largest;
  if (Status status = EncodeHybridUint(
        std::numeric_limits<uint32_t>::max(), config, &largest);
      !status.ok()) {
    return status;
  }
  if (largest.symbol >= kPrefixAlphabetSize) {
    return Status::InvalidArgument(
      "HybridUint configuration exceeds the prefix alphabet");
  }
  return Status::Ok();
}

Status ValidatePrefixCode(const PrefixCode& prefix) {
  std::array<size_t, 16> depth_counts{};
  size_t symbol_count = 0;
  for (size_t symbol = 0; symbol < prefix.depths.size(); ++symbol) {
    const uint8_t depth = prefix.depths[symbol];
    if (depth > 15 ||
        (depth == 0 && prefix.bits[symbol] != 0) ||
        (depth != 0 && prefix.bits[symbol] >= (uint32_t{1} << depth))) {
      return Status::InvalidArgument("Prefix code is malformed");
    }
    ++depth_counts[depth];
    symbol_count += depth != 0;
  }
  if (prefix.degenerate_symbol < kPrefixAlphabetSize) {
    if (symbol_count != 1 ||
        prefix.depths[prefix.degenerate_symbol] != 1 ||
        prefix.bits[prefix.degenerate_symbol] != 0) {
      return Status::InvalidArgument(
        "Degenerate prefix code is inconsistent");
    }
  } else if (prefix.degenerate_symbol != kPrefixAlphabetSize ||
             symbol_count == 1) {
    return Status::InvalidArgument(
      "Prefix-code degenerate symbol is invalid");
  }

  size_t available_slots = 1;
  for (size_t depth = 1; depth < depth_counts.size(); ++depth) {
    available_slots *= 2;
    if (depth_counts[depth] > available_slots) {
      return Status::InvalidArgument("Prefix code is oversubscribed");
    }
    available_slots -= depth_counts[depth];
  }

  std::array<uint16_t, kPrefixAlphabetSize> canonical_bits{};
  if (Status status = codestream_internal::ConvertBitDepthsToSymbols(
        prefix.depths, canonical_bits);
      !status.ok()) {
    return status;
  }
  if (canonical_bits != prefix.bits) {
    return Status::InvalidArgument("Prefix bits are not canonical");
  }
  return Status::Ok();
}

Status ValidateEntropyCode(const EntropyCode& code) {
  if (code.mode == EntropyCodingMode::kAns) {
    return codestream_internal::ValidateAnsEntropyCode(code);
  }
  if (code.mode != EntropyCodingMode::kPrefix ||
      code.ans_log_alpha_size != 0 || !code.ans_histograms.empty() ||
      code.context_count == 0 ||
      code.context_map.size() != code.context_count ||
      code.prefix_codes.empty() ||
      code.prefix_codes.size() > kMaximumPrefixClusters ||
      code.uint_configs.size() != code.prefix_codes.size()) {
    return Status::InvalidArgument("Entropy-code dimensions are invalid");
  }
  for (uint8_t cluster : code.context_map) {
    if (cluster >= code.prefix_codes.size()) {
      return Status::InvalidArgument("Entropy context map is invalid");
    }
  }
  for (size_t index = 0; index < code.prefix_codes.size(); ++index) {
    if (Status status = ValidateConfig(code.uint_configs[index]);
        !status.ok()) {
      return status;
    }
    if (Status status = ValidatePrefixCode(code.prefix_codes[index]);
        !status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status WriteBits(BitWriter* writer, size_t count, uint64_t bits) {
  return writer->WriteBits(count, bits);
}

void ReverseRange(std::vector<uint8_t>* values, size_t begin, size_t end) {
  std::reverse(values->begin() + static_cast<ptrdiff_t>(begin),
               values->begin() + static_cast<ptrdiff_t>(end));
}

void WriteHuffmanTreeRepetitions(
  uint8_t previous_value,
  uint8_t value,
  size_t repetitions,
  std::vector<uint8_t>* tree,
  std::vector<uint8_t>* extra_bits) {

  if (previous_value != value) {
    tree->push_back(value);
    extra_bits->push_back(0);
    --repetitions;
  }
  if (repetitions == 7) {
    tree->push_back(value);
    extra_bits->push_back(0);
    --repetitions;
  }
  if (repetitions < 3) {
    for (size_t index = 0; index < repetitions; ++index) {
      tree->push_back(value);
      extra_bits->push_back(0);
    }
    return;
  }

  repetitions -= 3;
  const size_t start = tree->size();
  while (true) {
    tree->push_back(16);
    extra_bits->push_back(static_cast<uint8_t>(repetitions & 0x3));
    repetitions >>= 2;
    if (repetitions == 0) {
      break;
    }
    --repetitions;
  }
  ReverseRange(tree, start, tree->size());
  ReverseRange(extra_bits, start, extra_bits->size());
}

void WriteHuffmanTreeZeroRepetitions(
  size_t repetitions,
  std::vector<uint8_t>* tree,
  std::vector<uint8_t>* extra_bits) {

  if (repetitions == 11) {
    tree->push_back(0);
    extra_bits->push_back(0);
    --repetitions;
  }
  if (repetitions < 3) {
    for (size_t index = 0; index < repetitions; ++index) {
      tree->push_back(0);
      extra_bits->push_back(0);
    }
    return;
  }

  repetitions -= 3;
  const size_t start = tree->size();
  while (true) {
    tree->push_back(17);
    extra_bits->push_back(static_cast<uint8_t>(repetitions & 0x7));
    repetitions >>= 3;
    if (repetitions == 0) {
      break;
    }
    --repetitions;
  }
  ReverseRange(tree, start, tree->size());
  ReverseRange(extra_bits, start, extra_bits->size());
}

void DecideHuffmanRle(
  std::span<const uint8_t> depths,
  bool* use_nonzero_rle,
  bool* use_zero_rle) {

  size_t total_zero_repetitions = 0;
  size_t total_nonzero_repetitions = 0;
  size_t zero_runs = 1;
  size_t nonzero_runs = 1;
  for (size_t index = 0; index < depths.size();) {
    const uint8_t value = depths[index];
    size_t repetitions = 1;
    while (index + repetitions < depths.size() &&
           depths[index + repetitions] == value) {
      ++repetitions;
    }
    if (repetitions >= 3 && value == 0) {
      total_zero_repetitions += repetitions;
      ++zero_runs;
    }
    if (repetitions >= 4 && value != 0) {
      total_nonzero_repetitions += repetitions;
      ++nonzero_runs;
    }
    index += repetitions;
  }
  *use_nonzero_rle = total_nonzero_repetitions > nonzero_runs * 2;
  *use_zero_rle = total_zero_repetitions > zero_runs * 2;
}

void EncodeHuffmanTree(
  std::span<const uint8_t> depths,
  std::vector<uint8_t>* tree,
  std::vector<uint8_t>* extra_bits) {

  size_t encoded_length = depths.size();
  while (encoded_length != 0 && depths[encoded_length - 1] == 0) {
    --encoded_length;
  }

  bool use_nonzero_rle = false;
  bool use_zero_rle = false;
  if (depths.size() > 50) {
    DecideHuffmanRle(
      depths.first(encoded_length), &use_nonzero_rle, &use_zero_rle);
  }

  uint8_t previous_value = 8;
  for (size_t index = 0; index < encoded_length;) {
    const uint8_t value = depths[index];
    size_t repetitions = 1;
    if ((value != 0 && use_nonzero_rle) ||
        (value == 0 && use_zero_rle)) {
      while (index + repetitions < encoded_length &&
             depths[index + repetitions] == value) {
        ++repetitions;
      }
    }
    if (value == 0) {
      WriteHuffmanTreeZeroRepetitions(repetitions, tree, extra_bits);
    } else {
      WriteHuffmanTreeRepetitions(
        previous_value, value, repetitions, tree, extra_bits);
      previous_value = value;
    }
    index += repetitions;
  }
}

Status StoreHuffmanTreeOfHuffmanTree(
  int code_count,
  std::span<const uint8_t> code_length_depths,
  BitWriter* writer) {

  constexpr std::array<uint8_t, kCodeLengthCodeCount> kStorageOrder = {
    1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  };
  constexpr std::array<uint8_t, 6> kSymbols = {0, 7, 3, 2, 1, 15};
  constexpr std::array<uint8_t, 6> kBitLengths = {2, 4, 3, 2, 2, 4};

  size_t codes_to_store = kCodeLengthCodeCount;
  if (code_count > 1) {
    while (codes_to_store != 0 &&
           code_length_depths[kStorageOrder[codes_to_store - 1]] == 0) {
      --codes_to_store;
    }
  }
  size_t skip = 0;
  if (code_length_depths[kStorageOrder[0]] == 0 &&
      code_length_depths[kStorageOrder[1]] == 0) {
    skip = 2;
    if (code_length_depths[kStorageOrder[2]] == 0) {
      skip = 3;
    }
  }
  if (Status status = WriteBits(writer, 2, skip); !status.ok()) {
    return status;
  }
  for (size_t index = skip; index < codes_to_store; ++index) {
    const uint8_t depth = code_length_depths[kStorageOrder[index]];
    if (depth >= kSymbols.size()) {
      return Status::Internal("Code-length Huffman depth is invalid");
    }
    if (Status status = WriteBits(
          writer, kBitLengths[depth], kSymbols[depth]);
        !status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status StoreHuffmanTree(
  std::span<const uint8_t> depths,
  BitWriter* writer) {

  std::vector<uint8_t> tree;
  std::vector<uint8_t> extra_bits;
  tree.reserve(2 * depths.size());
  extra_bits.reserve(2 * depths.size());
  EncodeHuffmanTree(depths, &tree, &extra_bits);

  std::array<uint64_t, kCodeLengthCodeCount> histogram{};
  for (uint8_t value : tree) {
    if (value >= histogram.size()) {
      return Status::Internal("Huffman-tree RLE symbol is invalid");
    }
    ++histogram[value];
  }

  int code_count = 0;
  int only_code = 0;
  for (size_t index = 0; index < histogram.size(); ++index) {
    if (histogram[index] != 0) {
      if (code_count == 0) {
        only_code = static_cast<int>(index);
        code_count = 1;
      } else {
        code_count = 2;
        break;
      }
    }
  }

  std::array<uint8_t, kCodeLengthCodeCount> code_length_depths{};
  std::array<uint16_t, kCodeLengthCodeCount> code_length_symbols{};
  if (Status status = codestream_internal::CreateHuffmanTree(
        histogram, 5, code_length_depths);
      !status.ok()) {
    return status;
  }
  if (Status status = codestream_internal::ConvertBitDepthsToSymbols(
        code_length_depths, code_length_symbols);
      !status.ok()) {
    return status;
  }
  if (Status status = StoreHuffmanTreeOfHuffmanTree(
        code_count, code_length_depths, writer);
      !status.ok()) {
    return status;
  }
  if (code_count == 1) {
    code_length_depths[static_cast<size_t>(only_code)] = 0;
  }

  for (size_t index = 0; index < tree.size(); ++index) {
    const uint8_t symbol = tree[index];
    if (Status status = WriteBits(
          writer,
          code_length_depths[symbol],
          code_length_symbols[symbol]);
        !status.ok()) {
      return status;
    }
    if (symbol == 16) {
      if (Status status = WriteBits(writer, 2, extra_bits[index]);
          !status.ok()) {
        return status;
      }
    } else if (symbol == 17) {
      if (Status status = WriteBits(writer, 3, extra_bits[index]);
          !status.ok()) {
        return status;
      }
    }
  }
  return Status::Ok();
}

Status StoreVarLenUint16(size_t value, BitWriter* writer) {
  if (value > std::numeric_limits<uint16_t>::max()) {
    return Status::InvalidArgument("Variable-length uint16 is too large");
  }
  if (value == 0) {
    return WriteBits(writer, 1, 0);
  }
  if (Status status = WriteBits(writer, 1, 1); !status.ok()) {
    return status;
  }
  const size_t highest_bit =
    std::numeric_limits<size_t>::digits - 1 -
    static_cast<size_t>(std::countl_zero(value));
  if (Status status = WriteBits(writer, 4, highest_bit); !status.ok()) {
    return status;
  }
  return WriteBits(
    writer, highest_bit, value - (size_t{1} << highest_bit));
}

Status StoreSimpleHuffmanTree(
  const PrefixCode& code,
  std::array<size_t, 4> symbols,
  size_t symbol_count,
  size_t symbol_bits,
  BitWriter* writer) {

  if (Status status = WriteBits(writer, 2, 1); !status.ok()) {
    return status;
  }
  if (Status status = WriteBits(writer, 2, symbol_count - 1); !status.ok()) {
    return status;
  }
  std::stable_sort(
    symbols.begin(),
    symbols.begin() + static_cast<ptrdiff_t>(symbol_count),
    [&code](size_t left, size_t right) {
      return code.depths[left] < code.depths[right];
    });
  for (size_t index = 0; index < symbol_count; ++index) {
    if (Status status = WriteBits(writer, symbol_bits, symbols[index]);
        !status.ok()) {
      return status;
    }
  }
  if (symbol_count == 4) {
    return WriteBits(writer, 1, code.depths[symbols[0]] == 1 ? 1 : 0);
  }
  return Status::Ok();
}

Status WritePrefixCodeInternal(const PrefixCode& code, BitWriter* writer) {
  size_t symbol_count = 0;
  std::array<size_t, 4> first_symbols{};
  size_t alphabet_size = 1;
  for (size_t symbol = 0; symbol < code.depths.size(); ++symbol) {
    if (code.depths[symbol] != 0) {
      if (symbol_count < first_symbols.size()) {
        first_symbols[symbol_count] = symbol;
      }
      ++symbol_count;
      alphabet_size = symbol + 1;
    }
  }

  size_t symbol_bits = 0;
  for (size_t value = alphabet_size - 1; value != 0; value >>= 1) {
    ++symbol_bits;
  }
  if (symbol_count <= 1) {
    if (Status status = WriteBits(writer, 4, 1); !status.ok()) {
      return status;
    }
    return WriteBits(writer, symbol_bits, first_symbols[0]);
  }
  if (symbol_count <= 4) {
    return StoreSimpleHuffmanTree(
      code, first_symbols, symbol_count, symbol_bits, writer);
  }
  return StoreHuffmanTree(
    std::span<const uint8_t>(code.depths).first(alphabet_size), writer);
}

Status WriteUintConfig(HybridUintConfig config, BitWriter* writer) {
  constexpr size_t kPrefixLogAlphabetSize = 15;
  if (Status status = ValidateConfig(config); !status.ok()) {
    return status;
  }
  if (Status status = WriteBits(writer, 4, config.split_exponent);
      !status.ok()) {
    return status;
  }
  if (config.split_exponent == kPrefixLogAlphabetSize) {
    return Status::Ok();
  }
  const size_t msb_bits = std::bit_width(
    static_cast<size_t>(config.split_exponent));
  if (Status status = WriteBits(writer, msb_bits, config.msb_in_token);
      !status.ok()) {
    return status;
  }
  const size_t lsb_choices = static_cast<size_t>(
    config.split_exponent - config.msb_in_token);
  const size_t lsb_bits = std::bit_width(lsb_choices);
  return WriteBits(writer, lsb_bits, config.lsb_in_token);
}

Status WritePrefixCodesInternal(
  std::span<const PrefixCode> prefix_codes,
  std::span<const HybridUintConfig> configs,
  BitWriter* writer) {

  if (prefix_codes.empty() || configs.size() != prefix_codes.size()) {
    return Status::InvalidArgument("Invalid prefix-code configurations");
  }
  if (Status status = WriteBits(writer, 1, 1); !status.ok()) {
    return status;
  }
  for (const HybridUintConfig config : configs) {
    if (Status status = WriteUintConfig(config, writer); !status.ok()) {
      return status;
    }
  }
  for (const PrefixCode& code : prefix_codes) {
    size_t alphabet_size = 1;
    for (size_t symbol = 0; symbol < code.depths.size(); ++symbol) {
      if (code.depths[symbol] != 0) {
        alphabet_size = symbol + 1;
      }
    }
    if (Status status = StoreVarLenUint16(alphabet_size - 1, writer);
        !status.ok()) {
      return status;
    }
  }
  for (const PrefixCode& code : prefix_codes) {
    size_t alphabet_size = 1;
    for (size_t symbol = 0; symbol < code.depths.size(); ++symbol) {
      if (code.depths[symbol] != 0) {
        alphabet_size = symbol + 1;
      }
    }
    if (alphabet_size > 1) {
      if (Status status = WritePrefixCodeInternal(code, writer);
          !status.ok()) {
        return status;
      }
    }
  }
  return Status::Ok();
}

Status WriteValue(
  uint32_t value,
  const PrefixCode& prefix,
  HybridUintConfig config,
  BitWriter* writer) {

  HybridUintToken token;
  if (Status status = EncodeHybridUint(value, config, &token); !status.ok()) {
    return status;
  }
  if (token.symbol >= prefix.depths.size() ||
      prefix.depths[token.symbol] == 0) {
    return Status::InvalidArgument(
      "Token symbol is absent from its prefix code");
  }
  const uint8_t prefix_depth = EncodedPrefixDepth(prefix, token.symbol);
  const uint64_t data = prefix.bits[token.symbol] |
    (static_cast<uint64_t>(token.extra_bits) << prefix_depth);
  return WriteBits(
    writer, prefix_depth + token.extra_bit_count, data);
}

Status WriteContextMapInternal(const EntropyCode& code, BitWriter* writer) {
  if (*std::max_element(code.context_map.begin(), code.context_map.end()) == 0) {
    return WriteBits(writer, 3, 1);
  }

  BitWriter best;
  size_t best_bits = std::numeric_limits<size_t>::max();
  for (const HybridUintConfig config : kBalancedUintConfigs) {
    std::array<uint64_t, kPrefixAlphabetSize> counts{};
    bool valid = true;
    for (uint8_t cluster : code.context_map) {
      HybridUintToken token;
      if (Status status = EncodeHybridUint(cluster, config, &token);
          !status.ok()) {
        return status;
      }
      if (token.symbol >= counts.size() ||
          counts[token.symbol] == std::numeric_limits<uint64_t>::max()) {
        valid = false;
        break;
      }
      ++counts[token.symbol];
    }
    if (!valid) {
      continue;
    }

    PrefixCode prefix;
    if (Status status = BuildPrefixCode(counts, &prefix); !status.ok()) {
      return status;
    }
    const std::array<PrefixCode, 1> prefixes = {prefix};
    const std::array<HybridUintConfig, 1> configs = {config};
    BitWriter candidate;
    if (Status status = WritePrefixCodesInternal(
          prefixes, configs, &candidate);
        !status.ok()) {
      return status;
    }
    for (uint8_t cluster : code.context_map) {
      if (Status status = WriteValue(cluster, prefix, config, &candidate);
          !status.ok()) {
        return status;
      }
    }
    if (candidate.bits_written() < best_bits) {
      best_bits = candidate.bits_written();
      best = std::move(candidate);
    }
  }
  if (best_bits == std::numeric_limits<size_t>::max()) {
    return Status::InvalidArgument("Context map cannot be entropy encoded");
  }
  if (Status status = WriteBits(writer, 3, 0); !status.ok()) {
    return status;
  }
  return writer->Append(best);
}

Status AppendTemporary(BitWriter* destination, BitWriter* temporary) {
  if (destination == nullptr || temporary == nullptr) {
    return Status::InvalidArgument("Bit-writer output is null");
  }
  return destination->Append(*temporary);
}

bool AddBits(uint64_t value, uint64_t* total) {
  if (total == nullptr ||
      *total > std::numeric_limits<uint64_t>::max() - value) {
    return false;
  }
  *total += value;
  return true;
}

Status BuildClusterCode(
  std::span<const codestream_internal::WeightedValue> values,
  HybridUintConfig base_config,
  bool optimize_config,
  HybridUintConfig* selected_config,
  PrefixCode* selected_prefix,
  uint64_t* selected_token_bits) {

  if (selected_config == nullptr || selected_prefix == nullptr ||
      selected_token_bits == nullptr) {
    return Status::InvalidArgument("Cluster-code output is null");
  }
  std::vector<HybridUintConfig> configs = {base_config};
  if (optimize_config) {
    for (const HybridUintConfig config : kBalancedUintConfigs) {
      if (std::find(configs.begin(), configs.end(), config) == configs.end()) {
        configs.push_back(config);
      }
    }
  }

  uint64_t best_cost = std::numeric_limits<uint64_t>::max();
  uint64_t best_token_bits = 0;
  HybridUintConfig best_config;
  PrefixCode best_prefix;
  for (const HybridUintConfig config : configs) {
    if (Status status = ValidateConfig(config); !status.ok()) {
      return status;
    }
    std::array<uint64_t, kPrefixAlphabetSize> counts{};
    uint64_t extra_bits = 0;
    bool valid = true;
    for (const codestream_internal::WeightedValue& weighted_value : values) {
      HybridUintToken token;
      if (Status status = EncodeHybridUint(
            weighted_value.value, config, &token);
          !status.ok()) {
        return status;
      }
      if (token.symbol >= counts.size() ||
          counts[token.symbol] >
            std::numeric_limits<uint64_t>::max() - weighted_value.count ||
          (token.extra_bit_count != 0 &&
           weighted_value.count >
             (std::numeric_limits<uint64_t>::max() - extra_bits) /
               token.extra_bit_count)) {
        valid = false;
        break;
      }
      counts[token.symbol] += weighted_value.count;
      extra_bits += weighted_value.count * token.extra_bit_count;
    }
    if (!valid) {
      continue;
    }

    PrefixCode prefix;
    if (Status status = BuildPrefixCode(counts, &prefix); !status.ok()) {
      return status;
    }
    uint64_t token_bits = extra_bits;
    for (size_t symbol = 0; symbol < counts.size(); ++symbol) {
      if (counts[symbol] == 0) {
        continue;
      }
      if (prefix.depths[symbol] == 0) {
        valid = false;
        break;
      }
      const uint8_t depth = EncodedPrefixDepth(prefix, symbol);
      if ((depth != 0 &&
           counts[symbol] > std::numeric_limits<uint64_t>::max() / depth) ||
          !AddBits(counts[symbol] * depth, &token_bits)) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      continue;
    }

    const std::array<PrefixCode, 1> prefixes = {prefix};
    const std::array<HybridUintConfig, 1> candidate_configs = {config};
    BitWriter model;
    if (Status status = WritePrefixCodesInternal(
          prefixes, candidate_configs, &model);
        !status.ok()) {
      return status;
    }
    if (model.bits_written() == 0) {
      return Status::Internal("Prefix model omitted its marker");
    }
    uint64_t total = token_bits;
    if (!AddBits(model.bits_written() - 1, &total)) {
      return Status::InvalidArgument("Entropy cost overflow");
    }
    if (total < best_cost) {
      best_cost = total;
      best_token_bits = token_bits;
      best_config = config;
      best_prefix = prefix;
    }
  }
  if (best_cost == std::numeric_limits<uint64_t>::max()) {
    return Status::InvalidArgument("Cluster values exceed the prefix alphabet");
  }
  *selected_config = best_config;
  *selected_prefix = best_prefix;
  *selected_token_bits = best_token_bits;
  return Status::Ok();
}

Status BuildEntropyCodeForPartition(
  std::span<const std::vector<EntropyToken>> section_tokens,
  const EntropyCodeOptions& options,
  std::span<const uint8_t> clustered_map,
  size_t cluster_count,
  bool optimize_configs,
  EntropyCode* code,
  EntropyCodeCost* cost,
  EntropyWorkProfile* profile) {

  if (code == nullptr || cost == nullptr || cluster_count == 0 ||
      cluster_count > kMaximumPrefixClusters) {
    return Status::InvalidArgument("Invalid entropy partition output");
  }
  EntropyCode candidate;
  candidate.context_count = options.context_count;
  candidate.context_map.resize(options.context_count);
  for (size_t context = 0; context < options.context_count; ++context) {
    const size_t initial = options.initial_context_map.empty()
      ? context
      : options.initial_context_map[context];
    if (initial >= clustered_map.size()) {
      return Status::Internal("Clustered context map is incomplete");
    }
    candidate.context_map[context] = clustered_map[initial];
  }

  const ProfileClock::time_point value_collection_begin =
    ProfileBegin(profile);
  std::vector<std::vector<uint32_t>> values(cluster_count);
  for (const std::vector<EntropyToken>& section : section_tokens) {
    for (const EntropyToken& token : section) {
      if (token.context >= candidate.context_count) {
        return Status::InvalidArgument("Entropy token context is out of range");
      }
      const size_t cluster = candidate.context_map[token.context];
      if (cluster >= values.size()) {
        return Status::Internal("Entropy cluster index is invalid");
      }
      values[cluster].push_back(token.value);
    }
  }
  ProfileEnd(
    profile, value_collection_begin,
    &EntropyWorkProfile::prefix_value_collection_nanoseconds);

  const ProfileClock::time_point config_search_begin =
    ProfileBegin(profile);
  candidate.uint_configs.resize(cluster_count);
  candidate.prefix_codes.resize(cluster_count);
  std::vector<uint64_t> cluster_token_bits(cluster_count);
  for (size_t cluster = 0; cluster < cluster_count; ++cluster) {
    std::vector<codestream_internal::WeightedValue> weighted_values;
    if (Status status = codestream_internal::AggregateEntropyValues(
          std::move(values[cluster]), &weighted_values);
        !status.ok()) {
      return status;
    }
    if (Status status = BuildClusterCode(
          weighted_values, options.uint_config, optimize_configs,
          &candidate.uint_configs[cluster], &candidate.prefix_codes[cluster],
          &cluster_token_bits[cluster]);
        !status.ok()) {
      return status;
    }
  }
  ProfileEnd(
    profile, config_search_begin,
    &EntropyWorkProfile::prefix_config_search_nanoseconds);
  const ProfileClock::time_point exact_measurement_begin =
    ProfileBegin(profile);
  BitWriter model;
  EntropyCodeCost candidate_cost;
  Status status = WriteEntropyCode(candidate, &model);
  if (status.ok()) {
    candidate_cost.model_bits = model.bits_written();
    candidate_cost.cluster_count = candidate.prefix_codes.size();
    for (uint64_t token_bits : cluster_token_bits) {
      if (!AddBits(token_bits, &candidate_cost.token_bits)) {
        status = Status::InvalidArgument("Entropy cost overflow");
        break;
      }
    }
  }
  ProfileEnd(
    profile, exact_measurement_begin,
    &EntropyWorkProfile::prefix_exact_measurement_nanoseconds);
  if (!status.ok()) {
    return status;
  }
  *code = std::move(candidate);
  *cost = candidate_cost;
  return Status::Ok();
}

Status BuildFixedEntropyCodeForPartition(
  const EntropyCodeOptions& options,
  std::span<const uint8_t> clustered_map,
  std::span<const Histogram> histograms,
  uint64_t extra_bits,
  EntropyCode* code,
  EntropyCodeCost* cost) {

  if (code == nullptr || cost == nullptr || histograms.empty() ||
      histograms.size() > kMaximumPrefixClusters) {
    return Status::InvalidArgument("Invalid fixed entropy partition output");
  }
  EntropyCode candidate;
  candidate.context_count = options.context_count;
  candidate.context_map.resize(options.context_count);
  for (size_t context = 0; context < options.context_count; ++context) {
    const size_t initial = options.initial_context_map.empty()
      ? context
      : options.initial_context_map[context];
    if (initial >= clustered_map.size()) {
      return Status::Internal("Clustered context map is incomplete");
    }
    candidate.context_map[context] = clustered_map[initial];
  }
  candidate.uint_configs.assign(histograms.size(), options.uint_config);
  candidate.prefix_codes.resize(histograms.size());
  uint64_t token_bits = extra_bits;
  for (size_t cluster = 0; cluster < histograms.size(); ++cluster) {
    if (Status status = BuildPrefixCode(
          std::span<const uint64_t>(histograms[cluster].counts).first(
            histograms[cluster].symbol_limit),
          &candidate.prefix_codes[cluster]);
        !status.ok()) {
      return status;
    }
    for (size_t symbol = 0;
         symbol < histograms[cluster].symbol_limit;
         ++symbol) {
      const uint64_t count = histograms[cluster].counts[symbol];
      if (count == 0) {
        continue;
      }
      if (candidate.prefix_codes[cluster].depths[symbol] == 0) {
        return Status::InvalidArgument("Entropy token bit count overflow");
      }
      const uint8_t depth =
        EncodedPrefixDepth(candidate.prefix_codes[cluster], symbol);
      if ((depth != 0 &&
           count > std::numeric_limits<uint64_t>::max() / depth) ||
          !AddBits(count * depth, &token_bits)) {
        return Status::InvalidArgument("Entropy token bit count overflow");
      }
    }
  }
  BitWriter model;
  if (Status status = WriteEntropyCode(candidate, &model); !status.ok()) {
    return status;
  }
  *code = std::move(candidate);
  *cost = {
    .model_bits = model.bits_written(),
    .token_bits = token_bits,
    .cluster_count = histograms.size(),
  };
  return Status::Ok();
}

}  // namespace

Status EncodeHybridUint(
  uint32_t value,
  HybridUintConfig config,
  HybridUintToken* token) {

  if (token == nullptr) {
    return Status::InvalidArgument("HybridUint output is null");
  }
  if (!config.valid()) {
    return Status::InvalidArgument("Invalid HybridUint configuration");
  }

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
  *token = result;
  return Status::Ok();
}

Status BuildPrefixCode(
  std::span<const uint64_t> counts,
  PrefixCode* code) {

  if (code == nullptr || counts.size() > kPrefixAlphabetSize) {
    return Status::InvalidArgument("Invalid prefix-code histogram");
  }
  PrefixCode candidate;
  if (Status status = codestream_internal::CreateHuffmanTree(
        counts,
        15,
        std::span<uint8_t>(candidate.depths).first(counts.size()));
      !status.ok()) {
    return status;
  }
  if (Status status = codestream_internal::ConvertBitDepthsToSymbols(
        std::span<const uint8_t>(candidate.depths).first(counts.size()),
        std::span<uint16_t>(candidate.bits).first(counts.size()));
      !status.ok()) {
    return status;
  }
  size_t populated_symbols = 0;
  for (size_t symbol = 0; symbol < counts.size(); ++symbol) {
    if (counts[symbol] != 0) {
      ++populated_symbols;
      candidate.degenerate_symbol = static_cast<uint16_t>(symbol);
    }
  }
  if (populated_symbols != 1) {
    candidate.degenerate_symbol = kPrefixAlphabetSize;
  }
  *code = candidate;
  return Status::Ok();
}

Status OptimizeEntropyCode(
  std::span<const std::vector<EntropyToken>> section_tokens,
  const EntropyCodeOptions& options,
  EntropyCode* code,
  EntropyCodeCost* cost,
  codestream_internal::EntropyWorkProfile* profile) {

  if (code == nullptr || options.context_count == 0) {
    return Status::InvalidArgument("Invalid entropy-code output or context count");
  }
  if (Status status = ValidateFullWidthConfig(options.uint_config);
      !status.ok()) {
    return status;
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
    const auto profile_call = [&](uint64_t EntropyWorkProfile::* destination,
                                  auto&& operation) {
      const ProfileClock::time_point begin = ProfileBegin(profile);
      Status status = operation();
      ProfileEnd(profile, begin, destination);
      return status;
    };
    const ProfileClock::time_point histogram_build_begin =
      ProfileBegin(profile);
    std::vector<Histogram> histograms(histogram_count);
    uint64_t fixed_extra_bits = 0;
    for (const std::vector<EntropyToken>& section : section_tokens) {
      for (const EntropyToken& token : section) {
        if (token.context >= options.context_count) {
          return Status::InvalidArgument("Entropy token context is out of range");
        }
        HybridUintToken encoded;
        if (Status status = EncodeHybridUint(
              token.value, options.uint_config, &encoded);
            !status.ok()) {
          return status;
        }
        if (encoded.symbol >= kPrefixAlphabetSize) {
          return Status::InvalidArgument(
            "HybridUint symbol is outside the prefix alphabet");
        }
        if (!AddBits(encoded.extra_bit_count, &fixed_extra_bits)) {
          return Status::InvalidArgument("Entropy token bit count overflow");
        }
        const size_t histogram = options.initial_context_map.empty()
          ? token.context
          : options.initial_context_map[token.context];
        if (!histograms[histogram].Add(encoded.symbol)) {
          return Status::InvalidArgument("Entropy histogram count overflow");
        }
      }
    }
    ProfileEnd(
      profile, histogram_build_begin,
      &EntropyWorkProfile::prefix_histogram_build_nanoseconds);

    // Exact cluster candidates all start from these same source histograms.
    // Cache their costs once so copies can reuse them while mutations continue
    // to invalidate only the affected cluster state.
    const ProfileClock::time_point histogram_cost_begin =
      ProfileBegin(profile);
    for (Histogram& histogram : histograms) {
      if (Status status = HistogramBitCost(&histogram); !status.ok()) {
        return status;
      }
    }
    ProfileEnd(
      profile, histogram_cost_begin,
      &EntropyWorkProfile::prefix_histogram_cost_nanoseconds);

    std::vector<Histogram> legacy_histograms;
    std::vector<uint8_t> legacy_map;
    if (Status status = profile_call(
          &EntropyWorkProfile::prefix_clustering_nanoseconds,
          [&] {
            return ClusterHistograms(
              histograms, std::min<size_t>(8, histograms.size()), false, 0,
              &legacy_histograms, &legacy_map);
          });
        !status.ok()) {
      return status;
    }

    EntropyCode best_code;
    EntropyCodeCost best_cost;
    if (Status status = profile_call(
          &EntropyWorkProfile::prefix_code_build_nanoseconds,
          [&] {
            return BuildFixedEntropyCodeForPartition(
              options, legacy_map, legacy_histograms, fixed_extra_bits,
              &best_code, &best_cost);
          });
        !status.ok()) {
      return status;
    }

    uint64_t best_total = best_cost.model_bits;
    if (!AddBits(best_cost.token_bits, &best_total)) {
      return Status::InvalidArgument("Entropy cost overflow");
    }
    std::vector<uint8_t> best_partition_map = legacy_map;
    size_t best_partition_count = legacy_histograms.size();
    // Build one deterministic farthest-first seed order, then reuse its
    // prefixes to screen all supported cluster caps by serialized cost.
    std::vector<Histogram> maximum_seeded_histograms;
    std::vector<uint32_t> maximum_seeded_symbols;
    std::vector<size_t> seed_indexes;
    if (Status status = profile_call(
          &EntropyWorkProfile::prefix_clustering_nanoseconds,
          [&] {
            return FastClusterHistograms(
              histograms,
              std::min(kMaximumPrefixClusters, histograms.size()), true,
              &maximum_seeded_histograms, &maximum_seeded_symbols,
              &seed_indexes);
          });
        !status.ok()) {
      return status;
    }
    size_t best_partition_cap = std::min<size_t>(8, seed_indexes.size());
    constexpr std::array<size_t, 6> kCandidateClusterCaps = {
      1, 2, 4, 8, 16, 32};
    size_t previous_cap = 0;
    for (size_t cap : kCandidateClusterCaps) {
      cap = std::min(cap, seed_indexes.size());
      if (cap == 0 || cap == previous_cap) {
        continue;
      }
      previous_cap = cap;
      std::vector<Histogram> candidate_histograms;
      std::vector<uint8_t> candidate_map;
      if (Status status = profile_call(
            &EntropyWorkProfile::prefix_clustering_nanoseconds,
            [&] {
              return ClusterHistogramsFromSeeds(
                histograms,
                std::span<const size_t>(seed_indexes).first(cap), true, 0,
                &candidate_histograms, &candidate_map);
            });
          !status.ok()) {
        return status;
      }
      EntropyCode candidate_code;
      EntropyCodeCost candidate_cost;
      if (Status status = profile_call(
            &EntropyWorkProfile::prefix_code_build_nanoseconds,
            [&] {
              return BuildFixedEntropyCodeForPartition(
                options, candidate_map, candidate_histograms,
                fixed_extra_bits, &candidate_code, &candidate_cost);
            });
          !status.ok()) {
        return status;
      }
      uint64_t candidate_total = candidate_cost.model_bits;
      if (!AddBits(candidate_cost.token_bits, &candidate_total)) {
        return Status::InvalidArgument("Entropy cost overflow");
      }
      if (candidate_total < best_total) {
        best_total = candidate_total;
        best_code = std::move(candidate_code);
        best_cost = candidate_cost;
        best_partition_map = std::move(candidate_map);
        best_partition_count = candidate_histograms.size();
        best_partition_cap = cap;
      }
    }

    // Refine the screened winner and its next larger cap with actual prefix
    // depths. This captures most of the exact-search benefit without scoring
    // every cap through repeated Huffman rebuilds.
    std::vector<Histogram> refined_histograms;
    std::vector<uint8_t> refined_map;
    if (Status status = profile_call(
          &EntropyWorkProfile::prefix_clustering_nanoseconds,
          [&] {
            return ClusterHistogramsFromSeeds(
              histograms,
              std::span<const size_t>(seed_indexes).first(best_partition_cap),
              false, 2, &refined_histograms, &refined_map);
          });
        !status.ok()) {
      return status;
    }
    EntropyCode refined_code;
    EntropyCodeCost refined_cost;
    if (Status status = profile_call(
          &EntropyWorkProfile::prefix_code_build_nanoseconds,
          [&] {
            return BuildFixedEntropyCodeForPartition(
              options, refined_map, refined_histograms, fixed_extra_bits,
              &refined_code, &refined_cost);
          });
        !status.ok()) {
      return status;
    }
    uint64_t refined_total = refined_cost.model_bits;
    if (!AddBits(refined_cost.token_bits, &refined_total)) {
      return Status::InvalidArgument("Entropy cost overflow");
    }
    if (refined_total < best_total) {
      best_total = refined_total;
      best_code = std::move(refined_code);
      best_cost = refined_cost;
      best_partition_map = std::move(refined_map);
      best_partition_count = refined_histograms.size();
    }

    const size_t higher_cap =
      std::min(seed_indexes.size(), best_partition_cap * 2);
    if (higher_cap > best_partition_cap) {
      std::vector<Histogram> higher_histograms;
      std::vector<uint8_t> higher_map;
      if (Status status = profile_call(
            &EntropyWorkProfile::prefix_clustering_nanoseconds,
            [&] {
              return ClusterHistogramsFromSeeds(
                histograms,
                std::span<const size_t>(seed_indexes).first(higher_cap),
                false, 2, &higher_histograms, &higher_map);
            });
          !status.ok()) {
        return status;
      }
      EntropyCode higher_code;
      EntropyCodeCost higher_cost;
      if (Status status = profile_call(
            &EntropyWorkProfile::prefix_code_build_nanoseconds,
            [&] {
              return BuildFixedEntropyCodeForPartition(
                options, higher_map, higher_histograms, fixed_extra_bits,
                &higher_code, &higher_cost);
            });
          !status.ok()) {
        return status;
      }
      uint64_t higher_total = higher_cost.model_bits;
      if (!AddBits(higher_cost.token_bits, &higher_total)) {
        return Status::InvalidArgument("Entropy cost overflow");
      }
      if (higher_total < best_total) {
        best_total = higher_total;
        best_code = std::move(higher_code);
        best_cost = higher_cost;
        best_partition_map = std::move(higher_map);
        best_partition_count = higher_histograms.size();
      }
    }

    EntropyCode configured_code;
    EntropyCodeCost configured_cost;
    if (Status status = BuildEntropyCodeForPartition(
          section_tokens, options, best_partition_map,
          best_partition_count, true, &configured_code,
          &configured_cost, profile);
        !status.ok()) {
      return status;
    }
    uint64_t configured_total = configured_cost.model_bits;
    if (!AddBits(configured_cost.token_bits, &configured_total)) {
      return Status::InvalidArgument("Entropy cost overflow");
    }
    if (configured_total < best_total) {
      best_code = std::move(configured_code);
      best_cost = configured_cost;
    }
    *code = std::move(best_code);
    if (cost != nullptr) {
      *cost = best_cost;
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Entropy-code allocation failed");
  } catch (const std::length_error&) {
    return Status::OutOfMemory("Entropy-code allocation is too large");
  }
}

Status WritePrefixCodes(
  std::span<const PrefixCode> prefix_codes,
  HybridUintConfig config,
  BitWriter* writer) {

  try {
    const std::vector<HybridUintConfig> configs(prefix_codes.size(), config);
    return WritePrefixCodes(prefix_codes, configs, writer);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Prefix-code serialization allocation failed");
  } catch (const std::length_error&) {
    return Status::OutOfMemory("Prefix-code serialization is too large");
  }
}

Status WritePrefixCodes(
  std::span<const PrefixCode> prefix_codes,
  std::span<const HybridUintConfig> configs,
  BitWriter* writer) {

  if (writer == nullptr || prefix_codes.empty() ||
      prefix_codes.size() > kMaximumPrefixClusters ||
      configs.size() != prefix_codes.size()) {
    return Status::InvalidArgument("Invalid prefix-code serialization input");
  }
  EntropyCode validation;
  validation.context_count = 1;
  validation.context_map = {0};
  validation.uint_configs.assign(configs.begin(), configs.end());
  validation.prefix_codes.assign(prefix_codes.begin(), prefix_codes.end());
  if (Status status = ValidateEntropyCode(validation); !status.ok()) {
    return status;
  }

  BitWriter temporary;
  try {
    if (Status status = WritePrefixCodesInternal(
          prefix_codes, configs, &temporary);
        !status.ok()) {
      return status;
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Prefix-code serialization allocation failed");
  } catch (const std::length_error&) {
    return Status::OutOfMemory("Prefix-code serialization is too large");
  }
  return AppendTemporary(writer, &temporary);
}

Status WriteContextMap(const EntropyCode& code, BitWriter* writer) {
  if (writer == nullptr) {
    return Status::InvalidArgument("Context-map output is null");
  }
  if (Status status = ValidateEntropyCode(code); !status.ok()) {
    return status;
  }
  BitWriter temporary;
  try {
    if (Status status = WriteContextMapInternal(code, &temporary);
        !status.ok()) {
      return status;
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Context-map serialization allocation failed");
  } catch (const std::length_error&) {
    return Status::OutOfMemory("Context-map serialization is too large");
  }
  return AppendTemporary(writer, &temporary);
}

Status WriteEntropyCode(const EntropyCode& code, BitWriter* writer) {
  if (writer == nullptr) {
    return Status::InvalidArgument("Entropy-code output is null");
  }
  if (Status status = ValidateEntropyCode(code); !status.ok()) {
    return status;
  }
  if (code.mode == EntropyCodingMode::kAns) {
    return codestream_internal::WriteAnsEntropyCodeModel(code, writer);
  }
  BitWriter temporary;
  try {
    if (Status status = WriteContextMapInternal(code, &temporary);
        !status.ok()) {
      return status;
    }
    if (Status status = WritePrefixCodesInternal(
          code.prefix_codes, code.uint_configs, &temporary);
        !status.ok()) {
      return status;
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Entropy-code serialization allocation failed");
  } catch (const std::length_error&) {
    return Status::OutOfMemory("Entropy-code serialization is too large");
  }
  return AppendTemporary(writer, &temporary);
}

Status WriteTokenStream(
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  BitWriter* writer) {

  if (writer == nullptr) {
    return Status::InvalidArgument("Token-stream output is null");
  }
  if (Status status = ValidateEntropyCode(code); !status.ok()) {
    return status;
  }
  if (code.mode == EntropyCodingMode::kAns) {
    return codestream_internal::WriteAnsTokenStream(tokens, code, writer);
  }
  BitWriter temporary;
  for (const EntropyToken& token : tokens) {
    if (token.context >= code.context_count) {
      return Status::InvalidArgument("Entropy token context is out of range");
    }
    const uint8_t cluster = code.context_map[token.context];
    if (Status status = WriteValue(
          token.value,
          code.prefix_codes[cluster],
          code.uint_configs[cluster],
          &temporary);
        !status.ok()) {
      return status;
    }
  }
  return AppendTemporary(writer, &temporary);
}

}  // namespace gjxl
