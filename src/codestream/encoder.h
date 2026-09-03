// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <vector>

#include "codestream/entropy_behavior.h"
#include "core/status.h"

namespace gjxl {

class VarDctEncoderFrame;

enum class VarDctCoefficientOrderBehavior : uint8_t {
  /// Derives coefficient orders from every transform anchor.
  kFull,
  /// Matches libjxl's effort-7-like DCT8 policy by deterministically sampling
  /// approximately half of transform anchors. Other order families remain
  /// fully sampled.
  kEffort7Dct8Sampled,
};

struct VarDctCodestreamOptions {
  VarDctEntropyBehavior entropy_behavior =
    VarDctEntropyBehavior::kBalanced;
  VarDctCoefficientOrderBehavior coefficient_order_behavior =
    VarDctCoefficientOrderBehavior::kFull;
};

/// Serializes one validated initial-profile frame as a raw JPEG XL codestream.
/// Failure leaves `output` unchanged.
[[nodiscard]] Status EncodeVarDctCodestream(
  const VarDctEncoderFrame& frame, std::vector<uint8_t>* output);

/// Serializes with an explicitly resolved entropy behavior. Failure leaves
/// `output` unchanged.
[[nodiscard]] Status EncodeVarDctCodestream(
  const VarDctEncoderFrame& frame,
  VarDctCodestreamOptions options,
  std::vector<uint8_t>* output);

}  // namespace gjxl
