// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <vector>

#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;

/// Serializes one validated initial-profile frame as a raw JPEG XL codestream.
/// Failure leaves `output` unchanged.
[[nodiscard]] Status EncodeVarDctCodestream(
  const VarDctEncoderFrame& frame, std::vector<uint8_t>* output);

}  // namespace gjxl
