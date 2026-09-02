// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;

namespace codestream_internal {

struct LibjxlTailOptions {
  int effort = 7;
  size_t thread_count = 1;
};

[[nodiscard]] bool LibjxlTailExperimentAvailable() noexcept;

/// Serializes a completed frame through the pinned internal libjxl bridge.
/// Failure leaves `output` unchanged.
[[nodiscard]] Status
EncodeVarDctCodestreamWithLibjxl(const VarDctEncoderFrame &frame,
                                 LibjxlTailOptions options,
                                 std::vector<uint8_t> *output);

} // namespace codestream_internal
} // namespace gjxl
