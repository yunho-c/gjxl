// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/cuda/cuda_aq_butteraugli_kernels.h"
#include <cuda_runtime.h>

namespace gjxl::cuda_internal {
namespace {
__device__ float PropagatingMaximum(float a, float b) {
  return isnan(a) || isnan(b) ? NAN : fmaxf(a, b);
}
__device__ float ComposedPixel(CudaAqComposeReductionPlan p, CudaAqAnchor a,
                               uint32_t valid_width, uint32_t index) {
  const uint32_t x = a.x * 8 + index % valid_width,
                 y = a.y * 8 + index / valid_width;
  return p.main_map[size_t(y) * p.main_stride + x] * 0.85f +
         0.5f * p.sub_map[size_t(y / 2) * p.sub_stride + x / 2];
}
__device__ void StoreReductions(CudaAqComposeReductionPlan p, CudaAqAnchor a,
                                uint32_t anchor_index, uint32_t pixels,
                                float sum, float maximum) {
  p.maxima[anchor_index] = maximum;
  const float reduced =
      1.2f * powf(sum / static_cast<float>(pixels), 1.0f / 16.0f);
  if (!isfinite(reduced) || reduced < 0.0f) {
    atomicOr(p.error, 32u);
    return;
  }
  for (uint32_t y = 0; y < p.batch.covered_height; ++y)
    for (uint32_t x = 0; x < p.batch.covered_width; ++x)
      p.block_map[size_t(a.y + y) * p.block_stride + a.x + x] = reduced;
}
__global__ void ComposeReductionKernel(CudaAqComposeReductionPlan p) {
  __shared__ float sums[256], maxima[256];
  const uint32_t ai = p.batch.anchor_offset + blockIdx.x;
  const auto a = p.anchors[ai];
  if (a.x * 8 >= p.width || a.y * 8 >= p.height) {
    if (!threadIdx.x) {
      atomicOr(p.error, 8u);
      p.maxima[ai] = NAN;
    }
    return;
  }
  const uint32_t w = min(p.batch.pixel_width, p.width - a.x * 8);
  const uint32_t h = min(p.batch.pixel_height, p.height - a.y * 8),
                 pixels = w * h;
  float sum = 0.0f, maximum = -INFINITY;
  for (uint32_t i = threadIdx.x; i < pixels; i += 256) {
    float value = ComposedPixel(p, a, w, i);
    if (!isfinite(value) || value < 0.0f) {
      atomicOr(p.error, 16u);
      maximum = NAN;
      value = 0.0f;
    } else
      maximum = PropagatingMaximum(maximum, value);
    value *= value;
    value *= value;
    value *= value;
    value *= value;
    sum += value;
  }
  sums[threadIdx.x] = sum;
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (uint32_t step = 128; step; step /= 2) {
    if (threadIdx.x < step) {
      sums[threadIdx.x] += sums[threadIdx.x + step];
      maxima[threadIdx.x] =
          PropagatingMaximum(maxima[threadIdx.x], maxima[threadIdx.x + step]);
    }
    __syncthreads();
  }
  if (!threadIdx.x)
    StoreReductions(p, a, ai, pixels, sums[0], maxima[0]);
}
template <unsigned Warps>
__global__ void ComposeDct8ReductionKernel(CudaAqComposeReductionPlan p) {
  const uint32_t local = blockIdx.x * Warps + threadIdx.x / 32;
  if (local >= p.batch.anchor_count)
    return;
  const uint32_t ai = p.batch.anchor_offset + local, lane = threadIdx.x % 32;
  const auto a = p.anchors[ai];
  if (a.x * 8 >= p.width || a.y * 8 >= p.height) {
    if (!lane) {
      atomicOr(p.error, 8u);
      p.maxima[ai] = NAN;
    }
    return;
  }
  const uint32_t w = min(8u, p.width - a.x * 8),
                 h = min(8u, p.height - a.y * 8), pixels = w * h;
  float sum = 0.0f, maximum = -INFINITY;
  for (uint32_t i = lane; i < pixels; i += 32) {
    float value = ComposedPixel(p, a, w, i);
    if (!isfinite(value) || value < 0.0f) {
      atomicOr(p.error, 16u);
      maximum = NAN;
      value = 0.0f;
    } else
      maximum = PropagatingMaximum(maximum, value);
    value *= value;
    value *= value;
    value *= value;
    // Original 256-thread DCT8 has one pixel per lane. Its last square is
    // rounded before the tree adds lane i+32, so do not contract this fold.
    value = __fmul_rn(value, value);
    sum = __fadd_rn(sum, value);
  }
  for (uint32_t step = 16; step; step /= 2) {
    const float other_sum = __shfl_down_sync(0xffffffffu, sum, step);
    const float other_max = __shfl_down_sync(0xffffffffu, maximum, step);
    if (lane < step) {
      sum += other_sum;
      maximum = PropagatingMaximum(maximum, other_max);
    }
  }
  if (!lane)
    StoreReductions(p, a, ai, pixels, sum, maximum);
}

} // namespace

cudaError_t LaunchCudaAqComposeReduction(CudaAqComposeReductionPlan plan,
                                         cudaStream_t stream) {
  if (plan.batch.anchor_count == 0)
    return cudaSuccess;
  if (plan.batch.pixel_width == 8 && plan.batch.pixel_height == 8) {
    ComposeDct8ReductionKernel<4>
        <<<(plan.batch.anchor_count + 3) / 4, 128, 0, stream>>>(plan);
  } else {
    ComposeReductionKernel<<<plan.batch.anchor_count, 256, 0, stream>>>(plan);
  }
  return cudaGetLastError();
}
} // namespace gjxl::cuda_internal
