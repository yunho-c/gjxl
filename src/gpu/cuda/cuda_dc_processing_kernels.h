// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
namespace gjxl::cuda_internal {
struct CudaDcProcessingParams {
  uint32_t width = 0, height = 0;
  uint32_t global_scale = 0, quant_dc = 0;
  uint32_t quantization_mode = 0, predictor = 0, extra_precision = 0;
  uint32_t use_resident_quantizer = 0;
};
[[nodiscard]] cudaError_t
LaunchCudaDcQuantization(float *dc, int *quantized, unsigned int *error,
                         CudaDcProcessingParams params,
                         const unsigned int *quantizer, cudaStream_t stream);
[[nodiscard]] cudaError_t
LaunchCudaDcSmoothing(const float *dc, float *smoothed, unsigned int *error,
                      CudaDcProcessingParams params,
                      const unsigned int *quantizer, cudaStream_t stream);
} // namespace gjxl::cuda_internal
