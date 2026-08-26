// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's Huffman-tree encoder.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "core/status.h"

namespace gjxl::codestream_internal {

[[nodiscard]] Status CreateHuffmanTree(
  std::span<const uint64_t> counts,
  uint8_t maximum_depth,
  std::span<uint8_t> depths);

[[nodiscard]] Status ConvertBitDepthsToSymbols(
  std::span<const uint8_t> depths,
  std::span<uint16_t> bits);

}  // namespace gjxl::codestream_internal
