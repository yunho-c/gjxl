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

__device__ float PatchSad(const float* input_x, const float* input_y,
                          const float* input_b, CudaAqEpfParams params,
                          uint32_t x, uint32_t y, int dx, int dy) {
  constexpr int kPlusOffsets[5][2] = {{0, 0}, {0, -1}, {-1, 0}, {0, 1}, {1, 0}};
  const float* inputs[3] = {input_x, input_y, input_b};
  float sad = 0.0f;
  for (uint32_t channel = 0; channel < 3; ++channel) {
    float channel_sad = 0.0f;
    for (uint32_t i = 0; i < 5; ++i) {
      const int ox = kPlusOffsets[i][0];
      const int oy = kPlusOffsets[i][1];
      channel_sad +=
          fabsf(Sample(inputs[channel], params.input_stride, params.width,
                       params.height, x, y, ox, oy) -
                Sample(inputs[channel], params.input_stride, params.width,
                       params.height, x, y, dx + ox, dy + oy));
    }
    sad = fmaf(channel_sad, params.channel_scale[channel], sad);
  }
  return sad;
}

__device__ float PixelSad(const float* input_x, const float* input_y,
                          const float* input_b, CudaAqEpfParams params,
                          uint32_t x, uint32_t y, int dx, int dy) {
  const float* inputs[3] = {input_x, input_y, input_b};
  float sad = 0.0f;
  for (uint32_t channel = 0; channel < 3; ++channel) {
    sad = fmaf(fabsf(Sample(inputs[channel], params.input_stride, params.width,
                            params.height, x, y, 0, 0) -
                     Sample(inputs[channel], params.input_stride, params.width,
                            params.height, x, y, dx, dy)),
               params.channel_scale[channel], sad);
  }
  return sad;
}

__global__ void EpfKernel(const float* input_x, const float* input_y,
                          const float* input_b, const float* inverse_sigma,
                          float* output_x, float* output_y, float* output_b,
                          unsigned int* error, CudaAqEpfParams params) {
  constexpr int kPass0Offsets[12][2] = {{0, -2}, {-1, -1}, {0, -1}, {1, -1},
                                        {-2, 0}, {-1, 0},  {1, 0},  {2, 0},
                                        {-1, 1}, {0, 1},   {1, 1},  {0, 2}};
  constexpr int kCardinalOffsets[4][2] = {{0, -1}, {-1, 0}, {1, 0}, {0, 1}};
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.width);
  const uint32_t x =
      static_cast<uint32_t>(index - static_cast<size_t>(y) * params.width);
  const size_t input_index = static_cast<size_t>(y) * params.input_stride + x;
  const size_t output_index = static_cast<size_t>(y) * params.output_stride + x;
  const float block_inverse_sigma =
      inverse_sigma[static_cast<size_t>(y / 8) * params.inverse_sigma_stride +
                    x / 8];
  const float* inputs[3] = {input_x, input_y, input_b};
  float* outputs[3] = {output_x, output_y, output_b};
  if (block_inverse_sigma < -3.905242919921875f) {
    for (uint32_t channel = 0; channel < 3; ++channel) {
      outputs[channel][output_index] = inputs[channel][input_index];
    }
    return;
  }
  const bool block_border =
      x % 8 == 0 || x % 8 == 7 || y % 8 == 0 || y % 8 == 7;
  const float scaled_inverse_sigma =
      block_inverse_sigma * params.sigma_scale *
      (block_border ? params.border_sad_multiplier : 1.0f);
  float sum[3] = {input_x[input_index], input_y[input_index],
                  input_b[input_index]};
  float weight_sum = 1.0f;
  const uint32_t candidate_count = params.pass == 0 ? 12 : 4;
  for (uint32_t i = 0; i < candidate_count; ++i) {
    const int dx =
        params.pass == 0 ? kPass0Offsets[i][0] : kCardinalOffsets[i][0];
    const int dy =
        params.pass == 0 ? kPass0Offsets[i][1] : kCardinalOffsets[i][1];
    const float sad =
        params.pass == 2
            ? PixelSad(input_x, input_y, input_b, params, x, y, dx, dy)
            : PatchSad(input_x, input_y, input_b, params, x, y, dx, dy);
    const float weight = fmaxf(0.0f, fmaf(sad, scaled_inverse_sigma, 1.0f));
    weight_sum += weight;
    for (uint32_t channel = 0; channel < 3; ++channel) {
      sum[channel] = fmaf(weight,
                          Sample(inputs[channel], params.input_stride,
                                 params.width, params.height, x, y, dx, dy),
                          sum[channel]);
    }
  }
  for (uint32_t channel = 0; channel < 3; ++channel) {
    float value = sum[channel] / weight_sum;
    if (!isfinite(value)) {
      atomicOr(error, 2u);
      value = 0.0f;
    }
    outputs[channel][output_index] = value;
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
  const size_t count = static_cast<size_t>(params.width) * params.height;
  const unsigned int blocks =
      static_cast<unsigned int>((count + kThreads - 1) / kThreads);
  EpfKernel<<<blocks, kThreads, 0, stream>>>(
      input[0], input[1], input[2], inverse_sigma, output[0], output[1],
      output[2], error, params);
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
