// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/prepared_coefficients_internal.h"
#include "codec/reconstruction.h"

namespace gjxl::prepared_coefficients_internal {

[[nodiscard]] Status ComputeQuantizedCoefficientsImpl(
  ConstImage3FView opsin,
  const PreparedForwardDctCoefficients* prepared,
  VarDctFrameInput input,
  SimpleVarDctCodestreamProfile profile,
  VarDctEncoderFrame* out,
  AcCoefficientDecisionMode decision_mode,
  DcQuantizationMode dc_quantization,
  VarDctDcPrediction dc_prediction);

/// Quantizes a previously prepared forward-transform set. The prepared set
/// must describe the strategy grid in `input`.
[[nodiscard]] Status ComputeQuantizedCoefficientsPrepared(
  const PreparedForwardDctCoefficients& prepared,
  VarDctFrameInput input,
  SimpleVarDctCodestreamProfile profile,
  VarDctEncoderFrame* out,
  AcCoefficientDecisionMode decision_mode =
    AcCoefficientDecisionMode::kAdjustedSharedQuant,
  DcQuantizationMode dc_quantization = DcQuantizationMode::kRound,
  VarDctDcPrediction dc_prediction = VarDctDcPrediction::kGradient);

}  // namespace gjxl::prepared_coefficients_internal
