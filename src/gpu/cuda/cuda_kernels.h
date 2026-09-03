// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace gjxl::cuda_internal {

[[nodiscard]] cudaError_t InitializeCudaDctBasis();

[[nodiscard]] cudaError_t LaunchCudaDct(
  bool forward,
  const float* input,
  float* output,
  size_t transform_count,
  unsigned int width,
  unsigned int height,
  cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaPointwiseAffine(
  const float* input,
  float* output,
  unsigned int width,
  unsigned int height,
  unsigned int input_stride,
  unsigned int output_stride,
  float scale,
  float bias,
  cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaSeparableConvolutionPass(
  bool horizontal,
  const float* input,
  const float* kernel,
  float* output,
  unsigned int width,
  unsigned int height,
  unsigned int input_stride,
  unsigned int output_stride,
  unsigned int kernel_size,
  cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaSymmetric5Convolution(
  const float* input,
  float* output,
  unsigned int width,
  unsigned int height,
  unsigned int input_stride,
  unsigned int output_stride,
  float distance0,
  float distance1,
  float distance2,
  float distance4,
  float distance8,
  float distance5,
  cudaStream_t stream);

[[nodiscard]] cudaError_t LaunchCudaMaximumReduction(
  const float* input,
  float* output,
  unsigned int width,
  unsigned int input_stride,
  unsigned int input_count,
  cudaStream_t stream);

}  // namespace gjxl::cuda_internal
