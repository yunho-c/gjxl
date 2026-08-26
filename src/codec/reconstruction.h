// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/vardct_frame.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

/// Transforms and quantizes a padded XYB image using the selected strategies.
/// The completed frame owns encoder control fields, quantized DC, and grouped
/// quantized AC coefficients.
[[nodiscard]] Status ComputeQuantizedCoefficients(
  ConstImage3FView opsin,
  VarDctFrameInput input,
  SimpleVarDctCodestreamProfile profile,
  VarDctEncoderFrame* out);

/// Dequantizes one coefficient frame, restores CfL and decoder-equivalent DC,
/// and applies the inverse transforms. Output is committed atomically.
[[nodiscard]] Status ReconstructQuantizedCoefficients(
  const VarDctEncoderFrame& frame,
  Image3FView output);

}  // namespace gjxl
