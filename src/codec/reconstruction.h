// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/vardct_frame.h"
#include "codec/dc_quantization.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

/// Transforms and quantizes a padded XYB image using the selected strategies.
/// The completed frame owns encoder control fields, quantized DC, and grouped
/// quantized AC coefficients.
/// Prediction-aware DC requires a nonzero profile.extra_dc_precision; the
/// encoder's opt-in policy uses one bit. DC reconstruction remains unsmoothed
/// in the completed frame and is filtered at the reconstruction consumer.
[[nodiscard]] Status ComputeQuantizedCoefficients(
  ConstImage3FView opsin,
  VarDctFrameInput input,
  SimpleVarDctCodestreamProfile profile,
  VarDctEncoderFrame* out,
  AcCoefficientDecisionMode decision_mode =
    AcCoefficientDecisionMode::kAdjustedSharedQuant,
  DcQuantizationMode dc_quantization = DcQuantizationMode::kRound,
  VarDctDcPrediction dc_prediction = VarDctDcPrediction::kGradient);

/// Dequantizes one coefficient frame, restores CfL and decoder-equivalent DC,
/// and applies the inverse transforms. Output is committed atomically.
[[nodiscard]] Status ReconstructQuantizedCoefficients(
  const VarDctEncoderFrame& frame,
  Image3FView output);

}  // namespace gjxl
