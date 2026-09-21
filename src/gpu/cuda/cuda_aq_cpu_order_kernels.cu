// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/cuda/cuda_aq_cpu_order_kernels.h"
#include "gpu/cuda/cuda_kernel_profile.h"

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

__global__ void CpuOrderGaborishKernel(const float* input_x,
                                       const float* input_y,
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
    const float axes = (Sample(input, params.input_stride, params.width,
                               params.height, x, y, -1, 0) +
                        Sample(input, params.input_stride, params.width,
                               params.height, x, y, 1, 0)) +
                       (Sample(input, params.input_stride, params.width,
                               params.height, x, y, 0, -1) +
                        Sample(input, params.input_stride, params.width,
                               params.height, x, y, 0, 1));
    const float diagonals = (Sample(input, params.input_stride, params.width,
                                    params.height, x, y, -1, -1) +
                             Sample(input, params.input_stride, params.width,
                                    params.height, x, y, 1, -1)) +
                            (Sample(input, params.input_stride, params.width,
                                    params.height, x, y, -1, 1) +
                             Sample(input, params.input_stride, params.width,
                                    params.height, x, y, 1, 1));
    float value = params.center_weight[channel] *
                  Sample(input, params.input_stride, params.width,
                         params.height, x, y, 0, 0);
    value = __fadd_rn(value, __fmul_rn(params.axis_weight[channel], axes));
    value =
        __fadd_rn(value, __fmul_rn(params.diagonal_weight[channel], diagonals));
    if (!isfinite(value)) {
      atomicOr(error, 1u);
      value = 0.0f;
    }
    outputs[channel][output_index] = value;
  }
}

__device__ void StoreLinearRgb(float x, float y, float b, float** outputs,
                               size_t output_index, unsigned int* error,
                               float scale, float bias_cuberoot) {
  constexpr float kOpsinBias = 0.0037930732552754493f;
  constexpr float kInverseOpsinMatrix[9] = {
      11.031566901960783f,  -9.866943921568629f, -0.16462299647058826f,
      -3.254147380392157f,  4.418770392156863f,  -0.16462299647058826f,
      -3.6588512862745097f, 2.7129230470588235f, 1.9459282392156863f};
  const float gamma[3] = {y + x + bias_cuberoot, y - x + bias_cuberoot,
                          b + bias_cuberoot};
  float mixed[3];
  for (uint32_t channel = 0; channel < 3; ++channel) {
    mixed[channel] = __fsub_rn(
        __fmul_rn(__fmul_rn(gamma[channel], gamma[channel]), gamma[channel]),
        kOpsinBias);
  }
  for (uint32_t row = 0; row < 3; ++row) {
    float value = scale * kInverseOpsinMatrix[3 * row] * mixed[0];
    value = fmaf(scale * kInverseOpsinMatrix[3 * row + 1], mixed[1], value);
    value = fmaf(scale * kInverseOpsinMatrix[3 * row + 2], mixed[2], value);
    if (!isfinite(value)) {
      atomicOr(error, 4u);
      value = 0.0f;
    }
    outputs[row][output_index] = value;
  }
}

__global__ void CpuOrderOpsinToLinearKernel(
    const float* input_x, const float* input_y, const float* input_b,
    float* output_r, float* output_g, float* output_b, unsigned int* error,
    CudaAqColorParams params, float bias_cuberoot) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(params.width) * params.height;
  if (index >= count) return;
  const uint32_t y = static_cast<uint32_t>(index / params.width);
  const uint32_t x =
      static_cast<uint32_t>(index - static_cast<size_t>(y) * params.width);
  const size_t input_index = static_cast<size_t>(y) * params.input_stride + x;
  const size_t output_index = static_cast<size_t>(y) * params.output_stride + x;
  float* outputs[3] = {output_r, output_g, output_b};
  StoreLinearRgb(input_x[input_index], input_y[input_index],
                 input_b[input_index], outputs, output_index, error,
                 params.scale, bias_cuberoot);
}

__global__ void CpuOrderReduceButteraugliKernel(
    const float* distance_map, uint32_t distance_stride,
    const CudaAqAnchor* anchors, float* block_distance, uint32_t block_stride,
    unsigned int* error, uint32_t source_width, uint32_t source_height,
    CudaAqExactBatch params) {
  const uint32_t local_anchor = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_anchor >= params.anchor_count) return;
  const CudaAqAnchor anchor = anchors[params.anchor_offset + local_anchor];
  const uint32_t x_begin = anchor.x * 8;
  const uint32_t y_begin = anchor.y * 8;
  if (x_begin >= source_width || y_begin >= source_height) {
    atomicOr(error, 8u);
    return;
  }
  const uint32_t valid_width = min(params.pixel_width, source_width - x_begin);
  const uint32_t valid_height =
      min(params.pixel_height, source_height - y_begin);
  const uint32_t pixel_count = valid_width * valid_height;
  float sum = 0.0f;
  for (uint32_t index = 0; index < pixel_count; ++index) {
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
    sum = __fadd_rn(sum, value);
  }
  // One lane owns a transform, preserving the CPU row-major accumulation.
  {
    const float reduced =
        1.2f * powf(sum / static_cast<float>(pixel_count), 1.0f / 16.0f);
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

__device__ size_t DctBasisOffset(unsigned length) {
  return length == 8 ? 0 : length == 16 ? 64 : 320;
}
__device__ size_t CoefficientIndex(unsigned width, unsigned height, unsigned v,
                                   unsigned u) {
  return height < width ? size_t(v) * width + u : size_t(u) * height + v;
}

__global__ void CpuOrderInverseDct(const float* input, float* output,
                                   unsigned width, unsigned height,
                                   const double* basis) {
  __shared__ double horizontal[1024];
  const unsigned count = width * height;
  const size_t base = size_t(blockIdx.x) * count;
  const double* bx = basis + DctBasisOffset(width);
  const double* by = basis + DctBasisOffset(height);
  for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
    const unsigned v = i / width, x = i % width;
    double sum = 0;
    for (unsigned u = 0; u < width; ++u)
      sum = __dadd_rn(
          sum,
          __dmul_rn(double(input[base + CoefficientIndex(width, height, v, u)]),
                    bx[u * width + x]));
    horizontal[i] = sum;
  }
  __syncthreads();
  for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
    const unsigned y = i / width, x = i % width;
    double sum = 0;
    for (unsigned v = 0; v < height; ++v)
      sum = __dadd_rn(sum,
                      __dmul_rn(by[v * height + y], horizontal[v * width + x]));
    output[base + i] = float(sum);
  }
}

}  // namespace

cudaError_t LaunchCudaCpuOrderInverseDct(const float* input, float* output,
                                         size_t count, unsigned width,
                                         unsigned height, const double* basis,
                                         cudaStream_t stream) {
  if ((width != 8 && width != 16 && width != 32) ||
      (height != 8 && height != 16 && height != 32) || count > 0x7fffffffu)
    return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  if (CudaKernelProfileScope profile{"CpuOrderInverseDct", unsigned(count),
                                     kThreads, stream};
      profile) {
    CpuOrderInverseDct<<<unsigned(count), kThreads, 0, stream>>>(
        input, output, width, height, basis);
  }
  return cudaGetLastError();
}

cudaError_t LaunchCudaCpuOrderGaborish(std::array<const float*, 3> input,
                                       std::array<float*, 3> output,
                                       unsigned* error,
                                       CudaAqGaborishParams params,
                                       cudaStream_t stream) {
  const unsigned blocks = unsigned(
      (size_t(params.width) * params.height + kThreads - 1) / kThreads);
  if (!blocks) return cudaSuccess;
  if (CudaKernelProfileScope profile{"CpuOrderGaborishKernel", blocks, kThreads,
                                     stream};
      profile) {
    CpuOrderGaborishKernel<<<blocks, kThreads, 0, stream>>>(
        input[0], input[1], input[2], output[0], output[1], output[2], error,
        params);
  }
  return cudaGetLastError();
}

cudaError_t LaunchCudaCpuOrderOpsinToLinear(std::array<const float*, 3> input,
                                            std::array<float*, 3> output,
                                            unsigned* error,
                                            CudaAqColorParams params,
                                            float bias_cuberoot,
                                            cudaStream_t stream) {
  const unsigned blocks = unsigned(
      (size_t(params.width) * params.height + kThreads - 1) / kThreads);
  if (!blocks) return cudaSuccess;
  if (CudaKernelProfileScope profile{"CpuOrderOpsinToLinearKernel", blocks,
                                     kThreads, stream};
      profile) {
    CpuOrderOpsinToLinearKernel<<<blocks, kThreads, 0, stream>>>(
        input[0], input[1], input[2], output[0], output[1], output[2], error,
        params, bias_cuberoot);
  }
  return cudaGetLastError();
}

cudaError_t LaunchCudaCpuOrderReduceButteraugli(
    const float* distance_map, uint32_t distance_stride,
    const CudaAqAnchor* anchors, float* block_distance, uint32_t block_stride,
    unsigned* error, uint32_t source_width, uint32_t source_height,
    CudaAqExactBatch batch, cudaStream_t stream) {
  if (!batch.anchor_count) return cudaSuccess;
  const unsigned blocks =
      unsigned((size_t(batch.anchor_count) + kThreads - 1) / kThreads);
  if (CudaKernelProfileScope profile{"CpuOrderReduceButteraugliKernel", blocks,
                                     kThreads, stream};
      profile) {
    CpuOrderReduceButteraugliKernel<<<blocks, kThreads, 0, stream>>>(
        distance_map, distance_stride, anchors, block_distance, block_stride,
        error, source_width, source_height, batch);
  }
  return cudaGetLastError();
}

}  // namespace gjxl::cuda_internal
