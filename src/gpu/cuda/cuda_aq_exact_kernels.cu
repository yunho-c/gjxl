// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "gpu/cuda/cuda_aq_exact_kernels.h"

namespace gjxl::cuda_internal {
namespace {

constexpr uint32_t kThreads = 256;

__device__ float FastCubeRootAndAdd(
  float value, float add) {
  constexpr float kOneThird = 1.0f / 3.0f;
  constexpr float kFourThirds = 4.0f / 3.0f;
  constexpr uint32_t kExponentBias = 0x54800000u;
  constexpr uint32_t kExponentMultiplier = 0x002AAAAAu;
  const uint32_t bits = __float_as_uint(value);
  const uint32_t estimate_bits =
    bits == 0 ? 0 : kExponentBias - (bits >> 23) * kExponentMultiplier;
  float reciprocal = __uint_as_float(estimate_bits);
  const float divided = __fmul_rn(kOneThird, value);
#pragma unroll
  for (uint32_t iteration = 0; iteration < 3; ++iteration) {
    const float squared = __fmul_rn(reciprocal, reciprocal);
    reciprocal = fmaf(-divided,
      __fmul_rn(squared, squared),
      __fmul_rn(kFourThirds, reciprocal));
  }
  float squared = __fmul_rn(reciprocal, reciprocal);
  reciprocal = fmaf(kOneThird,
    fmaf(-value, __fmul_rn(squared, squared), reciprocal),
    reciprocal);
  squared = __fmul_rn(reciprocal, reciprocal);
  return fmaf(squared, value, add);
}

__global__ void LinearRgbToOpsinKernel(
  const float* input_r,
  const float* input_g,
  const float* input_b,
  float* output_x,
  float* output_y,
  float* output_b,
  unsigned int* error,
  CudaLinearRgbToOpsinParams params) {
  const size_t index =
    static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count =
    static_cast<size_t>(params.padded_width) * params.padded_height;
  if (index >= count) return;
  const uint32_t output_y_index =
    static_cast<uint32_t>(index / params.padded_width);
  const uint32_t output_x_index = static_cast<uint32_t>(
    index - static_cast<size_t>(output_y_index) * params.padded_width);
  const uint32_t source_x = min(output_x_index, params.source_width - 1);
  const uint32_t source_y = min(output_y_index, params.source_height - 1);
  const size_t source_index =
    static_cast<size_t>(source_y) * params.source_stride + source_x;
  const float red = input_r[source_index];
  const float green = input_g[source_index];
  const float blue = input_b[source_index];
  if (!isfinite(red) || !isfinite(green) || !isfinite(blue)) {
    atomicOr(error, 1u);
  }
  constexpr float kBias = 0.0037930732552754493f;
  constexpr float kMatrix[3][3] = {{0.30f, 0.622f, 0.078f},
    {0.23f, 0.692f, 0.078f},
    {0.24342268924547819f, 0.20476744424496821f, 0.55180986650955360f}};
  const float scale = params.intensity_target / 255.0f;
  const float bias_cuberoot = FastCubeRootAndAdd(kBias, 0.0f);
  float gamma[3];
#pragma unroll
  for (uint32_t row = 0; row < 3; ++row) {
    float value = fmaf(__fmul_rn(scale, kMatrix[row][2]), blue, kBias);
    value = fmaf(__fmul_rn(scale, kMatrix[row][1]), green, value);
    value = fmaf(__fmul_rn(scale, kMatrix[row][0]), red, value);
    gamma[row] = FastCubeRootAndAdd(fmaxf(0.0f, value), -bias_cuberoot);
  }
  const float transformed_x = __fmul_rn(0.5f, __fsub_rn(gamma[0], gamma[1]));
  const float transformed_y = __fmul_rn(0.5f, __fadd_rn(gamma[0], gamma[1]));
  if (!isfinite(transformed_x) || !isfinite(transformed_y) ||
      !isfinite(gamma[2])) {
    atomicOr(error, 2u);
  }
  const size_t output_index =
    static_cast<size_t>(output_y_index) * params.output_stride + output_x_index;
  output_x[output_index] = transformed_x;
  output_y[output_index] = transformed_y;
  output_b[output_index] = gamma[2];
}

__global__ void OpsinMatrixScaleStatsKernel(
  const float* opsin_x,
  const float* opsin_y,
  const float* opsin_b,
  unsigned int* stats,
  unsigned int* error,
  CudaLinearRgbToOpsinParams params) {
  const size_t index =
    static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count =
    static_cast<size_t>(params.source_width) * params.source_height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.source_width);
  const uint32_t x =
    static_cast<uint32_t>(index - static_cast<size_t>(y) * params.source_width);
  if (x == 0 || y == 0) return;
  const size_t current = static_cast<size_t>(y) * params.output_stride + x;
  const size_t left = current - 1;
  const size_t up = current - params.output_stride;
  const float current_x = opsin_x[current];
  const float current_y = opsin_y[current];
  const float current_b = opsin_b[current];
  const float x_edge = fmaxf(fabsf(__fsub_rn(current_x, opsin_x[left])),
    fabsf(__fsub_rn(current_x, opsin_x[up])));
  const float current_difference = __fsub_rn(current_b, current_y);
  const float left_difference = __fsub_rn(opsin_b[left], opsin_y[left]);
  const float up_difference = __fsub_rn(opsin_b[up], opsin_y[up]);
  const float b_edge =
    fmaxf(fabsf(__fsub_rn(current_difference, left_difference)),
      fabsf(__fsub_rn(current_difference, up_difference)));
  float exposed_blue = __fsub_rn(current_b, __fmul_rn(1.2f, current_y));
  if (exposed_blue >= 0.0f) {
    const float blue_edge =
      __fadd_rn(fabsf(__fsub_rn(current_b, opsin_b[left])),
        fabsf(__fsub_rn(current_b, opsin_b[up])));
    exposed_blue = __fmul_rn(exposed_blue, blue_edge);
  } else {
    exposed_blue = 0.0f;
  }
  if (!isfinite(x_edge) || !isfinite(b_edge) || !isfinite(exposed_blue)) {
    atomicOr(error, 4u);
    return;
  }
  atomicMax(stats, __float_as_uint(x_edge));
  atomicMax(stats + 1, __float_as_uint(b_edge));
  atomicMax(stats + 2, __float_as_uint(exposed_blue));
}

__device__ uint32_t MirrorOffset(uint32_t coordinate, int delta,
                                 uint32_t size) {
  const long long period = 2 * static_cast<long long>(size);
  long long phase = (static_cast<long long>(coordinate) + delta) % period;
  if (phase < 0) phase += period;
  return static_cast<uint32_t>(phase < size ? phase : period - 1 - phase);
}

__device__ float Sample(const float* plane, uint32_t stride, uint32_t width,
                        uint32_t height, uint32_t x, uint32_t y, int dx,
                        int dy) {
  return plane[static_cast<size_t>(MirrorOffset(y, dy, height)) * stride +
               MirrorOffset(x, dx, width)];
}

__global__ void ScatterReconstructionKernel(
    const CudaAqAnchor* anchors, const float* inverse, float* reconstructed_x,
    float* reconstructed_y, float* reconstructed_b, uint32_t coding_stride,
    CudaAqExactBatch params) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values_per_channel =
      static_cast<size_t>(params.anchor_count) * params.coefficient_count;
  if (index >= 3 * values_per_channel) return;
  const uint32_t channel = static_cast<uint32_t>(index / values_per_channel);
  const size_t channel_index = index - channel * values_per_channel;
  const uint32_t anchor_index =
      static_cast<uint32_t>(channel_index / params.coefficient_count);
  const uint32_t pixel_index =
      static_cast<uint32_t>(channel_index - static_cast<size_t>(anchor_index) *
                                                params.coefficient_count);
  const uint32_t x = pixel_index % params.pixel_width;
  const uint32_t y = pixel_index / params.pixel_width;
  const CudaAqAnchor anchor = anchors[params.anchor_offset + anchor_index];
  float* destination = channel == 0
                           ? reconstructed_x
                           : (channel == 1 ? reconstructed_y : reconstructed_b);
  destination[static_cast<size_t>(anchor.y * 8 + y) * coding_stride +
              anchor.x * 8 + x] = inverse[params.coefficient_offset + index];
}

__global__ void GaborishKernel(const float* input_x, const float* input_y,
                               const float* input_b, float* output_x,
                               float* output_y, float* output_b,
                               unsigned int* error,
                               CudaAqGaborishParams params) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.width);
  const uint32_t x =
      static_cast<uint32_t>(index - static_cast<size_t>(y) * params.width);
  const float* inputs[3] = {input_x, input_y, input_b};
  float* outputs[3] = {output_x, output_y, output_b};
  const size_t output_index = static_cast<size_t>(y) * params.output_stride + x;
  for (uint32_t channel = 0; channel < 3; ++channel) {
    const float* input = inputs[channel];
    const float axes = Sample(input, params.input_stride, params.width,
                              params.height, x, y, -1, 0) +
                       Sample(input, params.input_stride, params.width,
                              params.height, x, y, 1, 0) +
                       Sample(input, params.input_stride, params.width,
                              params.height, x, y, 0, -1) +
                       Sample(input, params.input_stride, params.width,
                              params.height, x, y, 0, 1);
    const float diagonals = Sample(input, params.input_stride, params.width,
                                   params.height, x, y, -1, -1) +
                            Sample(input, params.input_stride, params.width,
                                   params.height, x, y, 1, -1) +
                            Sample(input, params.input_stride, params.width,
                                   params.height, x, y, -1, 1) +
                            Sample(input, params.input_stride, params.width,
                                   params.height, x, y, 1, 1);
    float value = params.center_weight[channel] *
                  Sample(input, params.input_stride, params.width,
                         params.height, x, y, 0, 0);
    value += params.axis_weight[channel] * axes;
    value += params.diagonal_weight[channel] * diagonals;
    if (!isfinite(value)) {
      atomicOr(error, 1u);
      value = 0.0f;
    }
    outputs[channel][output_index] = value;
  }
}

constexpr uint32_t kEpfTileWidth = 32;
constexpr uint32_t kEpfTileHeight = 32;

template <uint32_t Pass>
__global__ void EpfTiledKernel(const float* input_x, const float* input_y,
  const float* input_b, const float* inverse_sigma,
  float* output_x, float* output_y, float* output_b,
  unsigned int* error, CudaAqEpfParams params, uint32_t tiles_per_row) {
  static_assert(Pass <= 2);
  static_assert((kEpfTileWidth * kEpfTileHeight) % kThreads == 0);
  constexpr uint32_t kRadius = Pass == 0 ? 3 : (Pass == 1 ? 2 : 1);
  constexpr uint32_t kTileStride = kEpfTileWidth + 2 * kRadius;
  constexpr uint32_t kTileSize = kTileStride * (kEpfTileHeight + 2 * kRadius);
  constexpr int kPass0Offsets[12][2] = {{0, -2}, {-1, -1}, {0, -1}, {1, -1},
    {-2, 0}, {-1, 0}, {1, 0}, {2, 0}, {-1, 1}, {0, 1}, {1, 1}, {0, 2}};
  constexpr int kCardinalOffsets[4][2] = {{0, -1}, {-1, 0}, {1, 0}, {0, 1}};
  __shared__ float tile[3][kTileSize];
  const uint32_t origin_x = (blockIdx.x % tiles_per_row) * kEpfTileWidth;
  const uint32_t origin_y = (blockIdx.x / tiles_per_row) * kEpfTileHeight;
  for (uint32_t index = threadIdx.x; index < kTileSize; index += kThreads) {
    const uint32_t source_x = MirrorOffset(origin_x,
      static_cast<int>(index % kTileStride) - static_cast<int>(kRadius), params.width);
    const uint32_t source_y = MirrorOffset(origin_y,
      static_cast<int>(index / kTileStride) - static_cast<int>(kRadius), params.height);
    const size_t source_index =
      static_cast<size_t>(source_y) * params.input_stride + source_x;
    tile[0][index] = input_x[source_index];
    tile[1][index] = input_y[source_index];
    tile[2][index] = input_b[source_index];
  }
  // All threads load the mirrored halo before any out-of-image or bypass
  // branch. Mirror the original patch coordinates, not an already mirrored
  // candidate center: those operations differ at small image boundaries.
  __syncthreads();

  // Each warp processes one row at a time. Keep consecutive rows in a loop
  // to reuse the tile without keeping several pixels' accumulators live.
#pragma unroll 1
  for (uint32_t pixel = threadIdx.x; pixel < kEpfTileWidth * kEpfTileHeight;
       pixel += kThreads) {
    const uint32_t local_x = pixel % kEpfTileWidth;
    const uint32_t local_y = pixel / kEpfTileWidth;
    const uint32_t x = origin_x + local_x;
    const uint32_t y = origin_y + local_y;
    // Later iterations keep x fixed and only increase y.
    if (x >= params.width || y >= params.height) return;
    const int center = (local_y + kRadius) * kTileStride + local_x + kRadius;
    const size_t output_index = static_cast<size_t>(y) * params.output_stride + x;
    float* outputs[3] = {output_x, output_y, output_b};
    const float block_inverse_sigma =
      inverse_sigma[static_cast<size_t>(y / 8) * params.inverse_sigma_stride + x / 8];
    if (block_inverse_sigma < -3.905242919921875f) {
#pragma unroll
      for (uint32_t channel = 0; channel < 3; ++channel) {
        outputs[channel][output_index] = tile[channel][center];
      }
      continue;
    }
    const bool block_border = x % 8 == 0 || x % 8 == 7 || y % 8 == 0 || y % 8 == 7;
    const float scaled_inverse_sigma = block_inverse_sigma * params.sigma_scale *
      (block_border ? params.border_sad_multiplier : 1.0f);
    float sum[3] = {tile[0][center], tile[1][center], tile[2][center]};
    float weight_sum = 1.0f;
    constexpr uint32_t kCandidateCount = Pass == 0 ? 12 : 4;
#pragma unroll
    for (uint32_t i = 0; i < kCandidateCount; ++i) {
      const int dx = Pass == 0 ? kPass0Offsets[i][0] : kCardinalOffsets[i][0];
      const int dy = Pass == 0 ? kPass0Offsets[i][1] : kCardinalOffsets[i][1];
      const int candidate = center + dy * static_cast<int>(kTileStride) + dx;
      float sad = 0.0f;
#pragma unroll
      for (uint32_t channel = 0; channel < 3; ++channel) {
        float channel_sad = 0.0f;
        if constexpr (Pass == 2) {
          channel_sad = fabsf(tile[channel][center] - tile[channel][candidate]);
        } else {
          constexpr int kPlusOffsets[5][2] = {
            {0, 0}, {0, -1}, {-1, 0}, {0, 1}, {1, 0}};
#pragma unroll
          for (uint32_t p = 0; p < 5; ++p) {
            const int offset = kPlusOffsets[p][1] * static_cast<int>(kTileStride) +
              kPlusOffsets[p][0];
            channel_sad += fabsf(tile[channel][center + offset] -
              tile[channel][candidate + offset]);
          }
        }
        sad = fmaf(channel_sad, params.channel_scale[channel], sad);
      }
      const float weight = fmaxf(0.0f, fmaf(sad, scaled_inverse_sigma, 1.0f));
      weight_sum += weight;
#pragma unroll
      for (uint32_t channel = 0; channel < 3; ++channel) {
        sum[channel] = fmaf(weight, tile[channel][candidate], sum[channel]);
      }
    }
#pragma unroll
    for (uint32_t channel = 0; channel < 3; ++channel) {
      float value = sum[channel] / weight_sum;
      if (!isfinite(value)) {
        atomicOr(error, 2u);
        value = 0.0f;
      }
      outputs[channel][output_index] = value;
    }
  }
}

__global__ void OpsinToLinearKernel(const float* input_x, const float* input_y,
                                    const float* input_b, float* output_r,
                                    float* output_g, float* output_b,
                                    unsigned int* error,
                                    CudaAqColorParams params) {
  constexpr float kOpsinBias = 0.0037930732552754493f;
  constexpr float kBiasCuberoot = 0.15595419704914093f;
  constexpr float kInverseOpsinMatrix[9] = {
      11.031566901960783f,  -9.866943921568629f, -0.16462299647058826f,
      -3.254147380392157f,  4.418770392156863f,  -0.16462299647058826f,
      -3.6588512862745097f, 2.7129230470588235f, 1.9459282392156863f};
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.width);
  const uint32_t x =
      static_cast<uint32_t>(index - static_cast<size_t>(y) * params.width);
  const size_t input_index = static_cast<size_t>(y) * params.input_stride + x;
  const size_t output_index = static_cast<size_t>(y) * params.output_stride + x;
  const float gamma[3] = {
      input_y[input_index] + input_x[input_index] + kBiasCuberoot,
      input_y[input_index] - input_x[input_index] + kBiasCuberoot,
      input_b[input_index] + kBiasCuberoot};
  float mixed[3];
  for (uint32_t channel = 0; channel < 3; ++channel) {
    mixed[channel] =
        gamma[channel] * gamma[channel] * gamma[channel] - kOpsinBias;
  }
  float* outputs[3] = {output_r, output_g, output_b};
  for (uint32_t row = 0; row < 3; ++row) {
    float value = params.scale * kInverseOpsinMatrix[3 * row] * mixed[0];
    value =
        fmaf(params.scale * kInverseOpsinMatrix[3 * row + 1], mixed[1], value);
    value =
        fmaf(params.scale * kInverseOpsinMatrix[3 * row + 2], mixed[2], value);
    if (!isfinite(value)) {
      atomicOr(error, 4u);
      value = 0.0f;
    }
    outputs[row][output_index] = value;
  }
}

__global__ void ReduceButteraugliKernel(
    const float* distance_map, uint32_t distance_stride,
    const CudaAqAnchor* anchors, float* block_distance, uint32_t block_stride,
    unsigned int* error, uint32_t source_width, uint32_t source_height,
    CudaAqExactBatch params) {
  __shared__ float partial[kThreads];
  const uint32_t local_anchor = blockIdx.x;
  if (local_anchor >= params.anchor_count) return;
  const CudaAqAnchor anchor = anchors[params.anchor_offset + local_anchor];
  const uint32_t x_begin = anchor.x * 8;
  const uint32_t y_begin = anchor.y * 8;
  if (x_begin >= source_width || y_begin >= source_height) {
    if (threadIdx.x == 0) atomicOr(error, 8u);
    return;
  }
  const uint32_t valid_width = min(params.pixel_width, source_width - x_begin);
  const uint32_t valid_height =
      min(params.pixel_height, source_height - y_begin);
  const uint32_t pixel_count = valid_width * valid_height;
  float sum = 0.0f;
  for (uint32_t index = threadIdx.x; index < pixel_count; index += blockDim.x) {
    const uint32_t x = index % valid_width;
    const uint32_t y = index / valid_width;
    float value =
        distance_map[static_cast<size_t>(y_begin + y) * distance_stride +
                     x_begin + x];
    if (!isfinite(value) || value < 0.0f) {
      atomicOr(error, 16u);
      value = 0.0f;
    }
    value *= value;
    value *= value;
    value *= value;
    value *= value;
    sum += value;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t width = kThreads / 2; width != 0; width >>= 1) {
    if (threadIdx.x < width)
      partial[threadIdx.x] += partial[threadIdx.x + width];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float reduced =
        1.2f * powf(partial[0] / static_cast<float>(pixel_count), 1.0f / 16.0f);
    if (!isfinite(reduced) || reduced < 0.0f) {
      atomicOr(error, 32u);
      return;
    }
    for (uint32_t dy = 0; dy < params.covered_height; ++dy) {
      for (uint32_t dx = 0; dx < params.covered_width; ++dx) {
        block_distance[static_cast<size_t>(anchor.y + dy) * block_stride +
                       anchor.x + dx] = reduced;
      }
    }
  }
}

__global__ void ReduceMaximumErrorKernel(
    const float* reference_x, const float* reference_y,
    const float* reference_b, const float* reconstructed_x,
    const float* reconstructed_y, const float* reconstructed_b,
    uint32_t reference_stride, uint32_t reconstruction_stride,
    const CudaAqAnchor* anchors, float* block_error, uint32_t block_stride,
    float* transform_channel_maximum, unsigned int* error,
    uint32_t source_width, uint32_t source_height, float limit_x, float limit_y,
    float limit_b, CudaAqExactBatch params) {
  __shared__ float partial_x[kThreads];
  __shared__ float partial_y[kThreads];
  __shared__ float partial_b[kThreads];
  const uint32_t local_anchor = blockIdx.x;
  if (local_anchor >= params.anchor_count) return;
  const CudaAqAnchor anchor = anchors[params.anchor_offset + local_anchor];
  const uint32_t x_begin = anchor.x * 8;
  const uint32_t y_begin = anchor.y * 8;
  if (x_begin >= source_width || y_begin >= source_height) {
    if (threadIdx.x == 0) atomicOr(error, 64u);
    return;
  }
  const uint32_t valid_width = min(params.pixel_width, source_width - x_begin);
  const uint32_t valid_height =
      min(params.pixel_height, source_height - y_begin);
  const uint32_t pixel_count = valid_width * valid_height;
  float maximum_x = 0.0f;
  float maximum_y = 0.0f;
  float maximum_b = 0.0f;
  for (uint32_t index = threadIdx.x; index < pixel_count; index += blockDim.x) {
    const uint32_t x = x_begin + index % valid_width;
    const uint32_t y = y_begin + index / valid_width;
    const size_t reference_index =
        static_cast<size_t>(y) * reference_stride + x;
    const size_t reconstruction_index =
        static_cast<size_t>(y) * reconstruction_stride + x;
    const float candidate_x = fabsf(reference_x[reference_index] -
                                    reconstructed_x[reconstruction_index]);
    const float candidate_y = fabsf(reference_y[reference_index] -
                                    reconstructed_y[reconstruction_index]);
    const float candidate_b = fabsf(reference_b[reference_index] -
                                    reconstructed_b[reconstruction_index]);
    if (!isfinite(candidate_x) || !isfinite(candidate_y) ||
        !isfinite(candidate_b)) {
      atomicOr(error, 128u);
    } else {
      maximum_x = fmaxf(maximum_x, candidate_x);
      maximum_y = fmaxf(maximum_y, candidate_y);
      maximum_b = fmaxf(maximum_b, candidate_b);
    }
  }
  partial_x[threadIdx.x] = maximum_x;
  partial_y[threadIdx.x] = maximum_y;
  partial_b[threadIdx.x] = maximum_b;
  __syncthreads();
  for (uint32_t width = kThreads / 2; width != 0; width >>= 1) {
    if (threadIdx.x < width) {
      partial_x[threadIdx.x] =
          fmaxf(partial_x[threadIdx.x], partial_x[threadIdx.x + width]);
      partial_y[threadIdx.x] =
          fmaxf(partial_y[threadIdx.x], partial_y[threadIdx.x + width]);
      partial_b[threadIdx.x] =
          fmaxf(partial_b[threadIdx.x], partial_b[threadIdx.x + width]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const size_t output_index =
        3 * static_cast<size_t>(params.anchor_offset + local_anchor);
    transform_channel_maximum[output_index] = partial_x[0];
    transform_channel_maximum[output_index + 1] = partial_y[0];
    transform_channel_maximum[output_index + 2] = partial_b[0];
    const float normalized =
        fmaxf(partial_x[0] / limit_x,
              fmaxf(partial_y[0] / limit_y, partial_b[0] / limit_b));
    if (!isfinite(normalized) || normalized < 0.0f) {
      atomicOr(error, 256u);
      return;
    }
    for (uint32_t dy = 0; dy < params.covered_height; ++dy) {
      for (uint32_t dx = 0; dx < params.covered_width; ++dx) {
        block_error[static_cast<size_t>(anchor.y + dy) * block_stride +
                    anchor.x + dx] = normalized;
      }
    }
  }
}

}  // namespace

cudaError_t LaunchCudaAqScatterReconstruction(
    const CudaAqAnchor* anchors, const float* inverse,
    std::array<float*, 3> reconstructed, uint32_t coding_stride,
    CudaAqExactBatch batch, cudaStream_t stream) {
  const size_t count =
      3 * static_cast<size_t>(batch.anchor_count) * batch.coefficient_count;
  if (count == 0) return cudaSuccess;
  const unsigned int blocks =
      static_cast<unsigned int>((count + kThreads - 1) / kThreads);
  ScatterReconstructionKernel<<<blocks, kThreads, 0, stream>>>(
      anchors, inverse, reconstructed[0], reconstructed[1], reconstructed[2],
      coding_stride, batch);
  return cudaGetLastError();
}

cudaError_t LaunchCudaAqGaborish(std::array<const float*, 3> input,
                                 std::array<float*, 3> output,
                                 unsigned int* error,
                                 CudaAqGaborishParams params,
                                 cudaStream_t stream) {
  const size_t count = static_cast<size_t>(params.width) * params.height;
  const unsigned int blocks =
      static_cast<unsigned int>((count + kThreads - 1) / kThreads);
  GaborishKernel<<<blocks, kThreads, 0, stream>>>(input[0], input[1], input[2],
                                                  output[0], output[1],
                                                  output[2], error, params);
  return cudaGetLastError();
}

cudaError_t LaunchCudaAqEpf(std::array<const float*, 3> input,
                            const float* inverse_sigma,
                            std::array<float*, 3> output, unsigned int* error,
                            CudaAqEpfParams params, cudaStream_t stream) {
  const uint32_t tiles_per_row = static_cast<uint32_t>(
    (static_cast<size_t>(params.width) + kEpfTileWidth - 1) / kEpfTileWidth);
  const unsigned int blocks = static_cast<unsigned int>(
    static_cast<size_t>(tiles_per_row) *
    ((static_cast<size_t>(params.height) + kEpfTileHeight - 1) / kEpfTileHeight));
  switch (params.pass) {
    case 0:
      EpfTiledKernel<0><<<blocks, kThreads, 0, stream>>>(
        input[0], input[1], input[2], inverse_sigma, output[0], output[1],
        output[2], error, params, tiles_per_row);
      break;
    case 1:
      EpfTiledKernel<1><<<blocks, kThreads, 0, stream>>>(
        input[0], input[1], input[2], inverse_sigma, output[0], output[1],
        output[2], error, params, tiles_per_row);
      break;
    case 2:
      EpfTiledKernel<2><<<blocks, kThreads, 0, stream>>>(
        input[0], input[1], input[2], inverse_sigma, output[0], output[1],
        output[2], error, params, tiles_per_row);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t LaunchCudaAqOpsinToLinear(std::array<const float*, 3> input,
                                      std::array<float*, 3> output,
                                      unsigned int* error,
                                      CudaAqColorParams params,
                                      cudaStream_t stream) {
  const size_t count = static_cast<size_t>(params.width) * params.height;
  const unsigned int blocks =
      static_cast<unsigned int>((count + kThreads - 1) / kThreads);
  OpsinToLinearKernel<<<blocks, kThreads, 0, stream>>>(
      input[0], input[1], input[2], output[0], output[1], output[2], error,
      params);
  return cudaGetLastError();
}

cudaError_t LaunchCudaLinearRgbToOpsin(
  std::array<const float*, 3> input,
  std::array<float*, 3> output,
  unsigned int* matrix_scale_stats,
  unsigned int* error,
  CudaLinearRgbToOpsinParams params,
  cudaStream_t stream) {
  (void)cudaGetLastError();
  const size_t padded_count =
    static_cast<size_t>(params.padded_width) * params.padded_height;
  const unsigned int padded_blocks =
    static_cast<unsigned int>((padded_count + kThreads - 1) / kThreads);
  LinearRgbToOpsinKernel<<<padded_blocks, kThreads, 0, stream>>>(input[0],
    input[1],
    input[2],
    output[0],
    output[1],
    output[2],
    error,
    params);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess || !params.compute_matrix_scale_stats) {
    return status;
  }
  const size_t source_count =
    static_cast<size_t>(params.source_width) * params.source_height;
  const unsigned int source_blocks =
    static_cast<unsigned int>((source_count + kThreads - 1) / kThreads);
  OpsinMatrixScaleStatsKernel<<<source_blocks, kThreads, 0, stream>>>(
    output[0], output[1], output[2], matrix_scale_stats, error, params);
  return cudaGetLastError();
}

cudaError_t LaunchCudaAqReduceButteraugli(
    const float* distance_map, uint32_t distance_stride,
    const CudaAqAnchor* anchors, float* block_distance, uint32_t block_stride,
    unsigned int* error, uint32_t source_width, uint32_t source_height,
    CudaAqExactBatch batch, cudaStream_t stream) {
  if (batch.anchor_count == 0) return cudaSuccess;
  ReduceButteraugliKernel<<<batch.anchor_count, kThreads, 0, stream>>>(
      distance_map, distance_stride, anchors, block_distance, block_stride,
      error, source_width, source_height, batch);
  return cudaGetLastError();
}

cudaError_t LaunchCudaAqReduceMaximumError(
    std::array<const float*, 3> reference,
    std::array<const float*, 3> reconstructed, uint32_t reference_stride,
    uint32_t reconstruction_stride, const CudaAqAnchor* anchors,
    float* block_error, uint32_t block_stride, float* transform_channel_maximum,
    unsigned int* error, uint32_t source_width, uint32_t source_height,
    std::array<float, 3> limits, CudaAqExactBatch batch, cudaStream_t stream) {
  if (batch.anchor_count == 0) return cudaSuccess;
  ReduceMaximumErrorKernel<<<batch.anchor_count, kThreads, 0, stream>>>(
      reference[0], reference[1], reference[2], reconstructed[0],
      reconstructed[1], reconstructed[2], reference_stride,
      reconstruction_stride, anchors, block_error, block_stride,
      transform_channel_maximum, error, source_width, source_height, limits[0],
      limits[1], limits[2], batch);
  return cudaGetLastError();
}

}  // namespace gjxl::cuda_internal
