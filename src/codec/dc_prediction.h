// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>

namespace gjxl {

/// Lossless prediction of already quantized DC samples. Both choices preserve
/// decoded pixels; the weighted predictor also selects its matching context
/// property and fixed tree. This does not change DC quantization or smoothing.
enum class VarDctDcPrediction : uint8_t {
  kGradient,
  kWeighted,
};

/// Default for complete-frame encoding. Explicit gradient coding remains
/// available independently of the size-adaptive predefined DC tree.
inline constexpr VarDctDcPrediction kDefaultDcPrediction =
    VarDctDcPrediction::kWeighted;

[[nodiscard]] constexpr bool IsValidDcPrediction(VarDctDcPrediction mode) {
  return mode == VarDctDcPrediction::kGradient ||
         mode == VarDctDcPrediction::kWeighted;
}

} // namespace gjxl
