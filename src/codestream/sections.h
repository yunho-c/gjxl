// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's frame-section assembly.

#pragma once

#include <cstddef>
#include <span>

#include "codestream/bit_writer.h"
#include "core/status.h"

namespace gjxl {

inline constexpr size_t kMaximumTocSectionSize =
  (size_t{1} << 10) + (size_t{1} << 14) +
  (size_t{1} << 22) + (size_t{1} << 30) - 1;

/// Writes the no-permutation flag, alignment, section sizes, and final padding.
[[nodiscard]] Status WriteTocSizes(
  std::span<const size_t> section_sizes,
  BitWriter* output);

/// Pads each source section and concatenates it into an aligned destination.
[[nodiscard]] Status ConcatenateSections(
  std::span<const BitWriter> sections,
  BitWriter* output);

/// Atomically writes a TOC followed by the byte-aligned section payloads.
[[nodiscard]] Status WriteTocAndSections(
  std::span<const BitWriter> sections,
  BitWriter* output);

}  // namespace gjxl
