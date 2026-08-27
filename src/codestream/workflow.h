// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "core/ac_strategy.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

/// Options for the initial CPU-only public VarDCT encoding workflow.
struct VarDctEncodingOptions {
  float butteraugli_target = 1.0f;
};

/// Encoder analysis reported without exposing temporary pipeline storage.
struct VarDctEncodingSummary {
  Extent2D extent;
  size_t encoded_bytes = 0;
  std::array<size_t, kAcStrategyCount> strategy_counts{};
  std::vector<double> score_history;

  friend bool operator==(
    const VarDctEncodingSummary&,
    const VarDctEncodingSummary&) = default;
};

/// Converts linear sRGB, runs the native CPU quantization/AQ pipeline, and
/// serializes one initial-profile raw JPEG XL codestream.
///
/// Input may be strided. Failure leaves both caller-visible outputs unchanged.
/// `summary` may be null when analysis reporting is not required.
[[nodiscard]] Status EncodeLinearRgbVarDctCodestream(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary = nullptr);

}  // namespace gjxl
