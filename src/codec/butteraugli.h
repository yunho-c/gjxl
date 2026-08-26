// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/ac_strategy.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

struct ButteraugliOptions {
  float hf_asymmetry = 1.0f;
  float x_multiplier = 1.0f;
  float intensity_target = 80.0f;
};

/// Computes the pinned libjxl Butteraugli map and its maximum aggregate score.
/// Inputs are linear sRGB values. Outputs are committed atomically.
[[nodiscard]] Status ComputeButteraugliDistance(
  ConstImage3FView reference_linear_rgb,
  ConstImage3FView distorted_linear_rgb,
  ButteraugliOptions options,
  PlaneF32View distance_map,
  double* score);

/// Reduces a pixel-resolution Butteraugli map to one 16-norm distance per
/// base block. Every cell covered by a multiblock strategy receives the same
/// transform-wide value, matching libjxl's AQ feedback map.
[[nodiscard]] Status ReduceButteraugliDistanceMap(
  ConstPlaneF32View distance_map,
  const AcStrategyGrid& strategies,
  PlaneF32View block_distance_map);

}  // namespace gjxl
