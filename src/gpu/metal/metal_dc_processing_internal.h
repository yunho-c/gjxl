// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <cstdint>

namespace gjxl::metal_internal {

// Matches kernels/dc_processing.h. Quantizer values may instead be consumed
// from the resident two-word quantizer buffer when use_resident_quantizer=1.
struct AqDcProcessingParams {
  uint32_t width;
  uint32_t height;
  uint32_t global_scale;
  uint32_t quant_dc;
  uint32_t quantization_mode;
  uint32_t predictor;
  uint32_t extra_precision;
  uint32_t use_resident_quantizer;
};
static_assert(sizeof(AqDcProcessingParams) == 32);

} // namespace gjxl::metal_internal
