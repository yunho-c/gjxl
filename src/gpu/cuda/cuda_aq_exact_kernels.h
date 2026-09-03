// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>

namespace gjxl::cuda_internal {

struct CudaAqAnchor {
  uint32_t x = 0;
  uint32_t y = 0;
};

struct CudaAqExactBatch {
  uint32_t anchor_offset = 0;
  uint32_t anchor_count = 0;
  uint32_t coefficient_offset = 0;
  uint32_t coefficient_count = 0;
  uint32_t pixel_width = 0;
  uint32_t pixel_height = 0;
  uint32_t covered_width = 0;
  uint32_t covered_height = 0;
};

struct CudaAqGaborishParams {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t input_stride = 0;
  uint32_t output_stride = 0;
  float center_weight[3]{};
  float axis_weight[3]{};
  float diagonal_weight[3]{};
};

struct CudaAqEpfParams {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t input_stride = 0;
  uint32_t output_stride = 0;
  uint32_t inverse_sigma_stride = 0;
  uint32_t pass = 0;
  float sigma_scale = 0.0f;
  float border_sad_multiplier = 0.0f;
  float channel_scale[3]{};
};

struct CudaAqColorParams {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t input_stride = 0;
  uint32_t output_stride = 0;
  float scale = 1.0f;
};

[[nodiscard]] cudaError_t LaunchCudaAqScatterReconstruction(
    const CudaAqAnchor* anchors, const float* inverse,
    std::array<float*, 3> reconstructed, uint32_t coding_stride,
    CudaAqExactBatch batch, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqGaborish(
    std::array<const float*, 3> input, std::array<float*, 3> output,
    unsigned int* error, CudaAqGaborishParams params, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqEpf(std::array<const float*, 3> input,
                                          const float* inverse_sigma,
                                          std::array<float*, 3> output,
                                          unsigned int* error,
                                          CudaAqEpfParams params,
                                          cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqOpsinToLinear(
    std::array<const float*, 3> input, std::array<float*, 3> output,
    unsigned int* error, CudaAqColorParams params, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqReduceButteraugli(
    const float* distance_map, uint32_t distance_stride,
    const CudaAqAnchor* anchors, float* block_distance, uint32_t block_stride,
    unsigned int* error, uint32_t source_width, uint32_t source_height,
    CudaAqExactBatch batch, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaAqReduceMaximumError(
    std::array<const float*, 3> reference,
    std::array<const float*, 3> reconstructed, uint32_t reference_stride,
    uint32_t reconstruction_stride, const CudaAqAnchor* anchors,
    float* block_error, uint32_t block_stride, float* transform_channel_maximum,
    unsigned int* error, uint32_t source_width, uint32_t source_height,
    std::array<float, 3> limits, CudaAqExactBatch batch, cudaStream_t stream);

}  // namespace gjxl::cuda_internal
