// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/image.h"
#include "core/status.h"

namespace gjxl {

// Weights are named by squared distance from the center pixel.
struct Symmetric5Weights {
  float distance0 = 0.0f;
  float distance1 = 0.0f;
  float distance2 = 0.0f;
  float distance4 = 0.0f;
  float distance8 = 0.0f;
  float distance5 = 0.0f;
};

/// Applies a mirrored-boundary symmetric 5x5 convolution.
/// Input and output must be distinct, equally sized planes.
[[nodiscard]] Status ConvolveSymmetric5(
  ConstPlaneF32View input,
  Symmetric5Weights weights,
  PlaneF32View output);

}  // namespace gjxl
