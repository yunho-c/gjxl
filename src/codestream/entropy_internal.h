// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <span>

#include "codestream/entropy.h"

namespace gjxl::codestream_internal {

/// Counts the exact bits emitted by WriteTokenStream without materializing the
/// encoded payload. The output remains unchanged on failure.
[[nodiscard]] Status CountTokenStreamBits(
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  uint64_t* bit_count);

}  // namespace gjxl::codestream_internal
