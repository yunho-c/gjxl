// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace gjxl::cuda_internal {

inline constexpr size_t kCudaButteraugliWorkingPlaneCount = 33;
inline constexpr size_t kCudaButteraugliPsychoPlaneCount = 10;
inline constexpr size_t kCudaButteraugliKernelCount = 5;

struct CudaButteraugliPlan {
  std::array<const float*, 3> reference{};
  std::array<uint32_t, 3> reference_stride{};
  std::array<float*, kCudaButteraugliWorkingPlaneCount> planes{};
  std::array<float*, kCudaButteraugliPsychoPlaneCount> reference_sub{};
  std::array<const float*, kCudaButteraugliKernelCount> kernels{};
  std::array<float*, 2> reduction{};
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t working_width = 0;
  uint32_t working_height = 0;
  uint32_t sub_width = 0;
  uint32_t sub_height = 0;
  uint32_t xborder = 0;
  uint32_t yborder = 0;
  uint32_t expanded = 0;
  uint32_t multiscale = 0;
  float hf_asymmetry = 1.0f;
  float x_multiplier = 1.0f;
  float intensity_target = 255.0f;
};

[[nodiscard]] cudaError_t LaunchCudaButteraugliPrepare(
    const CudaButteraugliPlan& plan, cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaButteraugliCompare(
    const CudaButteraugliPlan& plan, std::array<const float*, 3> distorted,
    std::array<uint32_t, 3> distorted_stride, float* distance_map,
    uint32_t distance_stride, float* score, cudaStream_t stream);

}  // namespace gjxl::cuda_internal
