// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's prefix entropy encoder and histogram
// clustering code.

#include "codestream/entropy.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "codestream/huffman.h"

namespace gjxl {
namespace {

constexpr size_t kCodeLengthCodeCount = 18;

struct Histogram {
  std::array<uint64_t, kPrefixAlphabetSize> counts{};
  uint64_t total_count = 0;
  double bit_cost = 0.0;

  bool Add(size_t symbol) {
    if (symbol >= counts.size() ||
        counts[symbol] == std::numeric_limits<uint64_t>::max() ||
        total_count == std::numeric_limits<uint64_t>::max()) {
      return false;
    }
    ++counts[symbol];
    ++total_count;
    return true;
  }

  bool AddHistogram(const Histogram& other) {
    if (total_count >
        std::numeric_limits<uint64_t>::max() - other.total_count) {
      return false;
    }
    for (size_t index = 0; index < counts.size(); ++index) {
      if (counts[index] >
          std::numeric_limits<uint64_t>::max() - other.counts[index]) {
        return false;
      }
    }
    for (size_t index = 0; index < counts.size(); ++index) {
      counts[index] += other.counts[index];
    }
    total_count += other.total_count;
    return true;
  }
};

Status HistogramBitCost(Histogram* histogram) {
  if (histogram == nullptr) {
    return Status::InvalidArgument("Histogram output is null");
  }
  histogram->bit_cost = 0.0;
  if (histogram->total_count == 0) {
    return Status::Ok();
  }
  std::array<uint8_t, kPrefixAlphabetSize> depths{};
  if (Status status = codestream_internal::CreateHuffmanTree(
        histogram->counts, 15, depths);
      !status.ok()) {
    return status;
  }
  for (size_t index = 0; index < depths.size(); ++index) {
    histogram->bit_cost +=
      static_cast<double>(histogram->counts[index]) * depths[index];
  }
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
  Histogram combined = left;
  if (!combined.AddHistogram(right)) {
    return Status::InvalidArgument("Combined histogram count overflow");
  }
  if (Status status = HistogramBitCost(&combined); !status.ok()) {
    return status;
  }
  *distance = combined.bit_cost - left.bit_cost - right.bit_cost;
  return Status::Ok();
}

Status FastClusterHistograms(
  const std::vector<Histogram>& input,
  size_t maximum_histograms,
  std::vector<Histogram>* output,
  std::vector<uint32_t>* histogram_symbols) {

  if (input.empty() || maximum_histograms == 0 ||
      output == nullptr || histogram_symbols == nullptr) {
    return Status::InvalidArgument("Invalid histogram-clustering input");
  }
  output->clear();
  output->reserve(maximum_histograms);
  histogram_symbols->assign(input.size(),
                            static_cast<uint32_t>(maximum_histograms));

  std::vector<double> distances(
    input.size(), std::numeric_limits<double>::max());
  std::vector<Histogram> prepared = input;
  size_t largest_index = 0;
  for (size_t index = 0; index < prepared.size(); ++index) {
    if (prepared[index].total_count == 0) {
      (*histogram_symbols)[index] = 0;
      distances[index] = 0.0;
      continue;
    }
    if (Status status = HistogramBitCost(&prepared[index]); !status.ok()) {
      return status;
    }
    if (prepared[index].total_count >
        prepared[largest_index].total_count) {
      largest_index = index;
    }
  }

  constexpr double kMinimumDistinctDistance = 64.0;
  while (output->size() < maximum_histograms) {
    (*histogram_symbols)[largest_index] =
      static_cast<uint32_t>(output->size());
    output->push_back(prepared[largest_index]);
    distances[largest_index] = 0.0;
    largest_index = 0;
    for (size_t index = 0; index < prepared.size(); ++index) {
      if (distances[index] == 0.0) {
        continue;
      }
      double distance = 0.0;
      if (Status status = HistogramDistance(
            prepared[index], output->back(), &distance);
          !status.ok()) {
        return status;
      }
      distances[index] = std::min(distance, distances[index]);
      if (distances[index] > distances[largest_index]) {
        largest_index = index;
      }
    }
    if (distances[largest_index] < kMinimumDistinctDistance) {
      break;
    }
  }

  for (size_t index = 0; index < prepared.size(); ++index) {
    if ((*histogram_symbols)[index] != maximum_histograms) {
      continue;
    }
    size_t best = 0;
    double best_distance = 0.0;
    if (Status status = HistogramDistance(
          prepared[index], (*output)[best], &best_distance);
        !status.ok()) {
      return status;
    }
    for (size_t candidate = 1; candidate < output->size(); ++candidate) {
      double distance = 0.0;
      if (Status status = HistogramDistance(
            prepared[index], (*output)[candidate], &distance);
          !status.ok()) {
        return status;
      }
      if (distance < best_distance) {
        best = candidate;
        best_distance = distance;
      }
    }
    if (!(*output)[best].AddHistogram(prepared[index])) {
      return Status::InvalidArgument("Clustered histogram count overflow");
    }
    if (Status status = HistogramBitCost(&(*output)[best]); !status.ok()) {
      return status;
    }
    (*histogram_symbols)[index] = static_cast<uint32_t>(best);
  }
  return Status::Ok();
}

Status ClusterHistograms(
  std::vector<Histogram>* histograms,
  std::vector<uint8_t>* context_map) {

  if (histograms == nullptr || context_map == nullptr || histograms->empty()) {
    return Status::InvalidArgument("Invalid histogram-clustering output");
  }
  if (histograms->size() == 1) {
    context_map->assign(1, 0);
    return Status::Ok();
  }

  const size_t maximum_histograms =
    std::min(kMaximumPrefixClusters, histograms->size());
  std::vector<Histogram> clustered;
  std::vector<uint32_t> symbols;
  if (Status status = FastClusterHistograms(
        *histograms, maximum_histograms, &clustered, &symbols);
      !status.ok()) {
    return status;
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

Status ValidateConfig(HybridUintConfig config) {
  if (!config.valid()) {
    return Status::InvalidArgument("Invalid HybridUint configuration");
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
  for (size_t symbol = 0; symbol < prefix.depths.size(); ++symbol) {
    const uint8_t depth = prefix.depths[symbol];
    if (depth > 15 ||
        (depth == 0 && prefix.bits[symbol] != 0) ||
        (depth != 0 && prefix.bits[symbol] >= (uint32_t{1} << depth))) {
      return Status::InvalidArgument("Prefix code is malformed");
    }
    ++depth_counts[depth];
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
  if (Status status = ValidateConfig(code.uint_config); !status.ok()) {
    return status;
  }
  if (code.context_count == 0 ||
      code.context_map.size() != code.context_count ||
      code.prefix_codes.empty() ||
      code.prefix_codes.size() > kMaximumPrefixClusters) {
    return Status::InvalidArgument("Entropy-code dimensions are invalid");
  }
  for (uint8_t cluster : code.context_map) {
    if (cluster >= code.prefix_codes.size()) {
      return Status::InvalidArgument("Entropy context map is invalid");
    }
  }
  for (const PrefixCode& prefix : code.prefix_codes) {
    if (Status status = ValidatePrefixCode(prefix); !status.ok()) {
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

Status WritePrefixCodesInternal(
  std::span<const PrefixCode> prefix_codes,
  HybridUintConfig config,
  BitWriter* writer) {

  if (Status status = WriteBits(writer, 1, 1); !status.ok()) {
    return status;
  }
  for (size_t index = 0; index < prefix_codes.size(); ++index) {
    if (Status status = WriteBits(writer, 4, config.split_exponent);
        !status.ok()) {
      return status;
    }
    if (Status status = WriteBits(writer, 3, config.msb_in_token);
        !status.ok()) {
      return status;
    }
    if (Status status = WriteBits(writer, 2, config.lsb_in_token);
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
  const uint8_t prefix_depth = prefix.depths[token.symbol];
  const uint64_t data = prefix.bits[token.symbol] |
    (static_cast<uint64_t>(token.extra_bits) << prefix_depth);
  return WriteBits(
    writer, prefix_depth + token.extra_bit_count, data);
}

Status WriteContextMapInternal(const EntropyCode& code, BitWriter* writer) {
  if (*std::max_element(code.context_map.begin(), code.context_map.end()) == 0) {
    return WriteBits(writer, 3, 1);
  }
  if (Status status = WriteBits(writer, 3, 0); !status.ok()) {
    return status;
  }

  std::array<uint64_t, kPrefixAlphabetSize> counts{};
  for (uint8_t cluster : code.context_map) {
    ++counts[cluster];
  }
  PrefixCode context_prefix;
  if (Status status = BuildPrefixCode(counts, &context_prefix); !status.ok()) {
    return status;
  }
  const std::array<PrefixCode, 1> prefix_codes = {context_prefix};
  if (Status status = WritePrefixCodesInternal(
        prefix_codes, kDefaultHybridUintConfig, writer);
      !status.ok()) {
    return status;
  }
  for (uint8_t cluster : code.context_map) {
    if (Status status = WriteValue(
          cluster, context_prefix, kDefaultHybridUintConfig, writer);
        !status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status AppendTemporary(BitWriter* destination, BitWriter* temporary) {
  if (destination == nullptr || temporary == nullptr) {
    return Status::InvalidArgument("Bit-writer output is null");
  }
  return destination->Append(*temporary);
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
  *code = candidate;
  return Status::Ok();
}

Status OptimizeEntropyCode(
  std::span<const std::vector<EntropyToken>> section_tokens,
  const EntropyCodeOptions& options,
  EntropyCode* code) {

  if (code == nullptr || options.context_count == 0) {
    return Status::InvalidArgument("Invalid entropy-code output or context count");
  }
  if (Status status = ValidateConfig(options.uint_config); !status.ok()) {
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
    std::vector<Histogram> histograms(histogram_count);
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
        const size_t histogram = options.initial_context_map.empty()
          ? token.context
          : options.initial_context_map[token.context];
        if (!histograms[histogram].Add(encoded.symbol)) {
          return Status::InvalidArgument("Entropy histogram count overflow");
        }
      }
    }

    std::vector<uint8_t> clustered_map;
    if (Status status = ClusterHistograms(&histograms, &clustered_map);
        !status.ok()) {
      return status;
    }

    EntropyCode candidate;
    candidate.uint_config = options.uint_config;
    candidate.context_count = options.context_count;
    candidate.context_map.resize(options.context_count);
    for (size_t context = 0; context < options.context_count; ++context) {
      const size_t initial = options.initial_context_map.empty()
        ? context
        : options.initial_context_map[context];
      candidate.context_map[context] = clustered_map[initial];
    }
    candidate.prefix_codes.resize(histograms.size());
    for (size_t index = 0; index < histograms.size(); ++index) {
      if (Status status = BuildPrefixCode(
            histograms[index].counts, &candidate.prefix_codes[index]);
          !status.ok()) {
        return status;
      }
    }
    *code = std::move(candidate);
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

  if (writer == nullptr || prefix_codes.empty() ||
      prefix_codes.size() > kMaximumPrefixClusters) {
    return Status::InvalidArgument("Invalid prefix-code serialization input");
  }
  if (Status status = ValidateConfig(config); !status.ok()) {
    return status;
  }
  EntropyCode validation;
  validation.uint_config = config;
  validation.context_count = 1;
  validation.context_map = {0};
  validation.prefix_codes.assign(prefix_codes.begin(), prefix_codes.end());
  if (Status status = ValidateEntropyCode(validation); !status.ok()) {
    return status;
  }

  BitWriter temporary;
  try {
    if (Status status = WritePrefixCodesInternal(
          prefix_codes, config, &temporary);
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
  BitWriter temporary;
  try {
    if (Status status = WriteContextMapInternal(code, &temporary);
        !status.ok()) {
      return status;
    }
    if (Status status = WritePrefixCodesInternal(
          code.prefix_codes, code.uint_config, &temporary);
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
  BitWriter temporary;
  for (const EntropyToken& token : tokens) {
    if (token.context >= code.context_count) {
      return Status::InvalidArgument("Entropy token context is out of range");
    }
    const uint8_t cluster = code.context_map[token.context];
    if (Status status = WriteValue(
          token.value,
          code.prefix_codes[cluster],
          code.uint_config,
          &temporary);
        !status.ok()) {
      return status;
    }
  }
  return AppendTemporary(writer, &temporary);
}

}  // namespace gjxl
