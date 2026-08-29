// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's Huffman-tree encoder.

#include "codestream/huffman.h"

#include <algorithm>
#include <array>
#include <limits>

namespace gjxl::codestream_internal {
namespace {

struct HuffmanNode {
  uint64_t total_count;
  int16_t left;
  int16_t right_or_symbol;
};

constexpr size_t kMaximumHuffmanAlphabetSize = 128;
constexpr size_t kMaximumHuffmanNodeCount =
  2 * kMaximumHuffmanAlphabetSize + 1;

bool SetDepth(
  const HuffmanNode& node,
  std::span<const HuffmanNode> pool,
  std::span<uint8_t> depths,
  uint8_t level) {

  if (node.left >= 0) {
    if (level == std::numeric_limits<uint8_t>::max()) {
      return false;
    }
    ++level;
    return SetDepth(pool[static_cast<size_t>(node.left)], pool, depths, level) &&
      SetDepth(
        pool[static_cast<size_t>(node.right_or_symbol)],
        pool,
        depths,
        level);
  }
  if (node.right_or_symbol < 0 ||
      static_cast<size_t>(node.right_or_symbol) >= depths.size()) {
    return false;
  }
  depths[static_cast<size_t>(node.right_or_symbol)] = level;
  return true;
}

uint16_t ReverseBits(uint8_t bit_count, uint16_t value) {
  constexpr std::array<uint8_t, 16> kReversedNibble = {
    0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
    0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF,
  };
  uint32_t reversed = kReversedNibble[value & 0xF];
  for (uint8_t index = 4; index < bit_count; index += 4) {
    reversed <<= 4;
    value = static_cast<uint16_t>(value >> 4);
    reversed |= kReversedNibble[value & 0xF];
  }
  reversed >>= static_cast<uint8_t>(-bit_count) & 0x3;
  return static_cast<uint16_t>(reversed);
}

}  // namespace

Status CreateHuffmanTree(
  std::span<const uint64_t> counts,
  uint8_t maximum_depth,
  std::span<uint8_t> depths) {

  if (counts.size() != depths.size() ||
      counts.size() > kMaximumHuffmanAlphabetSize ||
      maximum_depth == 0 || maximum_depth > 15) {
    return Status::InvalidArgument("Invalid Huffman-tree dimensions");
  }
  std::fill(depths.begin(), depths.end(), 0);
  if (counts.empty()) {
    return Status::Ok();
  }

  std::array<HuffmanNode, kMaximumHuffmanNodeCount> tree;
  for (uint64_t count_limit = 1;;) {
    size_t tree_size = 0;

    for (size_t index = counts.size(); index != 0;) {
      --index;
      if (counts[index] != 0) {
        tree[tree_size++] = {
          std::max(counts[index], count_limit - 1),
          -1,
          static_cast<int16_t>(index),
        };
      }
    }

    const size_t leaf_count = tree_size;
    if (leaf_count == 0) {
      return Status::Ok();
    }
    if (leaf_count == 1) {
      depths[static_cast<size_t>(tree[0].right_or_symbol)] = 1;
      return Status::Ok();
    }

    // The former stable sort received leaves in descending-symbol order. Make
    // that tie-break explicit so an in-place sort preserves the exact tree.
    std::sort(
      tree.begin(),
      tree.begin() + static_cast<ptrdiff_t>(leaf_count),
      [](const HuffmanNode& left, const HuffmanNode& right) {
        if (left.total_count != right.total_count) {
          return left.total_count < right.total_count;
        }
        return left.right_or_symbol > right.right_or_symbol;
      });

    const HuffmanNode sentinel{
      std::numeric_limits<uint64_t>::max(), -1, -1};
    tree[tree_size++] = sentinel;
    tree[tree_size++] = sentinel;

    size_t leaf = 0;
    size_t internal = leaf_count + 1;
    for (size_t remaining = leaf_count - 1; remaining != 0; --remaining) {
      size_t left = 0;
      size_t right = 0;
      if (tree[leaf].total_count <= tree[internal].total_count) {
        left = leaf++;
      } else {
        left = internal++;
      }
      if (tree[leaf].total_count <= tree[internal].total_count) {
        right = leaf++;
      } else {
        right = internal++;
      }
      if (tree[left].total_count >
          std::numeric_limits<uint64_t>::max() - tree[right].total_count) {
        return Status::InvalidArgument("Huffman histogram count overflow");
      }
      HuffmanNode& parent = tree[tree_size - 1];
      parent.total_count = tree[left].total_count + tree[right].total_count;
      parent.left = static_cast<int16_t>(left);
      parent.right_or_symbol = static_cast<int16_t>(right);
      tree[tree_size++] = sentinel;
    }

    std::fill(depths.begin(), depths.end(), 0);
    const std::span<const HuffmanNode> pool(tree.data(), tree_size);
    if (!SetDepth(tree[2 * leaf_count - 1], pool, depths, 0)) {
      return Status::Internal("Failed to assign Huffman-tree depths");
    }
    if (*std::max_element(depths.begin(), depths.end()) <= maximum_depth) {
      return Status::Ok();
    }
    if (count_limit > std::numeric_limits<uint64_t>::max() / 2) {
      return Status::Internal("Unable to limit Huffman-tree depth");
    }
    count_limit *= 2;
  }
}

Status ConvertBitDepthsToSymbols(
  std::span<const uint8_t> depths,
  std::span<uint16_t> bits) {

  if (depths.size() != bits.size()) {
    return Status::InvalidArgument("Huffman depth and symbol sizes differ");
  }
  std::fill(bits.begin(), bits.end(), 0);
  std::array<uint16_t, 16> depth_counts{};
  for (uint8_t depth : depths) {
    if (depth > 15) {
      return Status::InvalidArgument("Huffman depth exceeds 15 bits");
    }
    ++depth_counts[depth];
  }
  depth_counts[0] = 0;

  std::array<uint16_t, 16> next_code{};
  uint32_t code = 0;
  for (size_t depth = 1; depth < next_code.size(); ++depth) {
    code = (code + depth_counts[depth - 1]) << 1;
    if (code > std::numeric_limits<uint16_t>::max()) {
      return Status::InvalidArgument("Huffman code space overflow");
    }
    next_code[depth] = static_cast<uint16_t>(code);
  }
  for (size_t index = 0; index < depths.size(); ++index) {
    if (depths[index] != 0) {
      bits[index] = ReverseBits(
        depths[index], next_code[depths[index]]++);
    }
  }
  return Status::Ok();
}

}  // namespace gjxl::codestream_internal
