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

}  // namespace gjxl::cuda_internal
