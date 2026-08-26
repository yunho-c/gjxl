// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>

#include "codec/chroma_from_luma.h"
#include "core/ac_strategy.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

struct AcStrategyCostOptions {
  float butteraugli_target = 1.0f;
  float entropy_multiplier = 1.0f;

  // X, Y, and B multiples of the transformed Y channel. Y must remain zero.
  std::array<float, 3> cfl_factors{};
};

/// Aggregates an initial quant field over one strategy footprint.
/// Coordinates are expressed in JPEG XL 8x8 base blocks.
[[nodiscard]] Status ComputeAcStrategyQuantNorm(
  AcStrategyType strategy,
  size_t block_x,
  size_t block_y,
  ConstPlaneF32View quant_field,
  float* quant_norm);

/// Estimates libjxl's quantization-aware cost for one complete transform.
/// Coordinates are expressed in JPEG XL 8x8 base blocks.
[[nodiscard]] Status EstimateAcStrategyCost(
  AcStrategyType strategy,
  size_t block_x,
  size_t block_y,
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  AcStrategyCostOptions options,
  float* cost);

struct AcStrategySearchOptions {
  float butteraugli_target = 1.0f;
};

/// Selects a complete, non-overlapping strategy grid using libjxl's
/// hierarchical 8x8/16x16/32x32 merge policy and gjxl's supported DCT set.
[[nodiscard]] Status FindAcStrategyGrid(
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  AcStrategySearchOptions options,
  AcStrategyGrid* out);

}  // namespace gjxl
