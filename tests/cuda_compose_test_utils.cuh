// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/cuda_butteraugli_kernels.h"

namespace {
constexpr uint32_t kWidth = 256;
constexpr size_t kPrefix = 13;
constexpr size_t kSuffix = 19;
constexpr float kGuard = -12345.0f;

void Check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}

struct Buffer {
  explicit Buffer(size_t size) : host(size + kPrefix + kSuffix, kGuard) {
    Check(cudaMalloc(&allocation, host.size() * sizeof(float)));
    data = allocation + kPrefix;
  }
  ~Buffer() {
    if (allocation)
      cudaFree(allocation);
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  void Upload() {
    Check(cudaMemcpy(allocation, host.data(), host.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    // Pageable H2D may return after staging, before DMA completes. Test kernels
    // use a non-blocking stream, so explicitly finish this default-stream copy.
    Check(cudaStreamSynchronize(nullptr));
  }
  std::vector<float> Download() const {
    std::vector<float> result(host.size());
    Check(cudaMemcpy(result.data(), allocation, result.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < kPrefix; ++i)
      if (result[i] != kGuard)
        throw std::runtime_error("prefix guard");
    for (size_t i = result.size() - kSuffix; i < result.size(); ++i)
      if (result[i] != kGuard)
        throw std::runtime_error("suffix guard");
    return result;
  }
  float *allocation = nullptr;
  float *data = nullptr;
  std::vector<float> host;
};

// Independent copies of the separate passes, deliberately not sharing the
// production reduction helper or composition function.
__global__ void ReferenceCompose(const float *main_map, const float *sub_map,
                                 float *output, uint32_t width, uint32_t height,
                                 uint32_t main_stride, uint32_t sub_stride,
                                 uint32_t output_stride) {
  const size_t flat = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (flat >= size_t(width) * height)
    return;
  const uint32_t y = uint32_t(flat / width);
  const uint32_t x = uint32_t(flat - size_t(y) * width);
  output[size_t(y) * output_stride + x] =
      main_map[size_t(y) * main_stride + x] * 0.85f +
      0.5f * sub_map[size_t(y / 2) * sub_stride + x / 2];
}

__global__ void ReferenceMaximum(const float *input, float *output,
                                 uint32_t width, uint32_t stride,
                                 uint32_t count) {
  __shared__ float values[kWidth];
  const uint32_t i = blockIdx.x * kWidth + threadIdx.x;
  float value = -INFINITY;
  if (i < count) {
    const uint32_t y = i / width;
    value = input[size_t(y) * stride + i - y * width];
    if (!isfinite(value) || value < 0.0f)
      value = NAN;
  }
  values[threadIdx.x] = value;
  __syncthreads();
  for (uint32_t step = kWidth / 2; step; step /= 2) {
    if (threadIdx.x < step) {
      const float other = values[threadIdx.x + step];
      values[threadIdx.x] = isnan(values[threadIdx.x]) || isnan(other)
                                ? NAN
                                : fmaxf(values[threadIdx.x], other);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    output[blockIdx.x] = values[0];
}

void Reference(const gjxl::cuda_internal::CudaButteraugliComposePlan &p,
               cudaStream_t stream) {
  uint32_t count = p.width * p.height;
  uint32_t width = p.width, stride = p.output_stride;
  ReferenceCompose<<<(count + 255) / 256, 256, 0, stream>>>(
      p.main_map, p.sub_map, p.output, p.width, p.height, p.main_stride,
      p.sub_stride, p.output_stride);
  Check(cudaGetLastError());
  const float *input = p.output;
  bool use_a = true;
  for (;;) {
    const uint32_t blocks = (count + 255) / 256;
    float *output = blocks == 1 ? p.score : p.reduction[use_a ? 0 : 1];
    ReferenceMaximum<<<blocks, 256, 0, stream>>>(input, output, width, stride,
                                                 count);
    Check(cudaGetLastError());
    if (blocks == 1)
      return;
    input = output;
    count = width = stride = blocks;
    use_a = !use_a;
  }
}

void Same(const std::vector<float> &a, const std::vector<float> &b) {
  if (a.size() != b.size() ||
      std::memcmp(a.data(), b.data(), a.size() * sizeof(float)))
    throw std::runtime_error("bitwise map/input mismatch");
}

} // namespace
