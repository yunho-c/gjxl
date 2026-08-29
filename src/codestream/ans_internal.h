// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <span>
#include <vector>

#include "codestream/entropy.h"

namespace gjxl::codestream_internal {

[[nodiscard]] Status ValidateAnsEntropyCode(const EntropyCode& code);

[[nodiscard]] Status WriteAnsEntropyCodeModel(
  const EntropyCode& code,
  BitWriter* writer);

[[nodiscard]] Status WriteAnsTokenStream(
  std::span<const EntropyToken> tokens,
  const EntropyCode& code,
  BitWriter* writer);

}  // namespace gjxl::codestream_internal
