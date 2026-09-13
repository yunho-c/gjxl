// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "codestream/entropy.h"
#include "codestream/storage.h"
#include "core/image.h"
#include "core/status.h"
#include <span>

namespace gjxl::codestream_internal {
// Default JPEG XL weighted preset 0. State resets at each channel/group.
// Failure preserves tokens, including numeric-range and allocation failures.
[[nodiscard]] Status TokenizeWeightedDcGroup(ConstImage3I32View dc,
                                             Storage<EntropyToken> *tokens);

// Converts only the DC branch of the pinned 45-context tree. Metadata and
// context numbering stay fixed. Validates the whole shape before mutation.
[[nodiscard]] Status UseWeightedDcTree(std::span<EntropyToken> tokens);
} // namespace gjxl::codestream_internal
