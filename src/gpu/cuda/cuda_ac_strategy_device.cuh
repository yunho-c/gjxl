// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cmath>

#include "gpu/cuda/cuda_kernels.h"

namespace gjxl::cuda_internal {

struct AcStrategyCandidateDevice {
  uint32_t block_x;
  uint32_t block_y;
  float quant_norm;
  float entropy_multiplier;
  float cfl_x;
  float cfl_b;
};

static_assert(sizeof(AcStrategyCandidateDevice) == 6 * sizeof(uint32_t));

__device__ inline bool CandidateValid(
  AcStrategyCandidateDevice candidate,
  CudaAcStrategyBatchParams params) {
  return params.transform_width <= params.pixel_width &&
    params.transform_height <= params.pixel_height &&
    candidate.block_x <=
      (params.pixel_width - params.transform_width) / 8 &&
    candidate.block_y <=
      (params.pixel_height - params.transform_height) / 8 &&
    isfinite(candidate.quant_norm) && candidate.quant_norm > 0.0f &&
    isfinite(candidate.entropy_multiplier) &&
    candidate.entropy_multiplier > 0.0f &&
    (params.use_device_cfl != 0u ||
     (isfinite(candidate.cfl_x) && isfinite(candidate.cfl_b)));
}

struct AcStrategyChannelRateDevice {
  float magnitude;
  uint32_t nonzero_count;
};

static_assert(sizeof(AcStrategyChannelRateDevice) == 2 * sizeof(float));

__device__ inline float ComputeCflFactor(
  const signed char* y_to_x,
  const signed char* y_to_b,
  AcStrategyCandidateDevice candidate,
  unsigned int channel,
  CudaAcStrategyBatchParams params) {
  if (channel == 1u) return 0.0f;
  if (params.use_device_cfl == 0u) {
    return channel == 0u ? candidate.cfl_x : candidate.cfl_b;
  }
  constexpr float kCflScale = 1.0f / 84.0f;
  const size_t tile_index =
    static_cast<size_t>(candidate.block_y / 8u) *
      params.color_tile_row_stride + candidate.block_x / 8u;
  return channel == 0u
    ? static_cast<float>(y_to_x[tile_index]) * kCflScale
    : 1.0f + static_cast<float>(y_to_b[tile_index]) * kCflScale;
}

}  // namespace gjxl::cuda_internal
