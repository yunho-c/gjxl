// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/image.h"
#include "core/status.h"

namespace gjxl {

/// Converts linear sRGB to JPEG XL's default internal XYB representation.
[[nodiscard]] Status LinearRgbToOpsin(
  ConstImage3FView linear_rgb,
  float intensity_target,
  Image3FView opsin);

/// Converts JPEG XL's default internal XYB representation to linear sRGB.
[[nodiscard]] Status OpsinToLinearRgb(
  ConstImage3FView opsin,
  float intensity_target,
  Image3FView linear_rgb);

}  // namespace gjxl
