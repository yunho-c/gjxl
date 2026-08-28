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
  AcCoefficientDecisionMode decision_mode);

/// Quantizes a previously prepared forward-transform set. The prepared set
/// must describe the strategy grid in `input`.
[[nodiscard]] Status ComputeQuantizedCoefficientsPrepared(
  const PreparedForwardDctCoefficients& prepared,
  VarDctFrameInput input,
  SimpleVarDctCodestreamProfile profile,
  VarDctEncoderFrame* out,
  AcCoefficientDecisionMode decision_mode =
    AcCoefficientDecisionMode::kAdjustedSharedQuant);

}  // namespace gjxl::prepared_coefficients_internal
