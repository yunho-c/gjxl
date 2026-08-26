// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/vardct_frame.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

/// Transforms and quantizes a padded XYB image using the selected strategies.
/// The completed frame owns all encoder control fields and grouped coefficients.
[[nodiscard]] Status ComputeQuantizedCoefficients(
  ConstImage3FView opsin,
  VarDctFrameInput input,
  CoefficientCodingOptions options,
  VarDctEncoderFrame* out);

/// Dequantizes one coefficient frame, restores CfL and LLF, and applies the
/// inverse transforms. Output is committed only after every block succeeds.
[[nodiscard]] Status ReconstructQuantizedCoefficients(
  const VarDctEncoderFrame& frame,
  Image3FView output);

}  // namespace gjxl
