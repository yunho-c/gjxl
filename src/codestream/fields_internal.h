// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Primitive field writing extracted from the libjxl-tiny-derived header writer.
#pragma once

#include "codestream/bit_writer.h"

namespace gjxl::codestream_internal {

inline constexpr size_t kMaximumJxlDimension = 0x3FFFFFFFu;
struct BitField {
  size_t width;
  uint64_t value;
};

// Composition primitives. The caller validates fields and owns the temporary
// writer or allotment that makes a complete header failure-atomic.
[[nodiscard]] Status WriteFields(BitWriter* writer, std::span<const BitField> fields);
[[nodiscard]] Status WriteSize(uint32_t size, BitWriter* writer);
[[nodiscard]] Status AppendTemporary(BitWriter* writer, const BitWriter& temporary);

}  // namespace gjxl::codestream_internal
