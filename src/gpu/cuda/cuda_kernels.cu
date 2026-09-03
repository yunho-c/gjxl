// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/cuda/cuda_kernels.h"

#include <array>
#include <cmath>
#include <cstddef>

namespace gjxl::cuda_internal {
namespace {

constexpr unsigned int kThreadsPerTransform = 256;
constexpr unsigned int kPrimitiveThreads = 256;
constexpr size_t kDctBasisElementCount =
  8 * 8 + 16 * 16 + 32 * 32 + 64 * 64;

__constant__ float kOrthonormalDctBasis[kDctBasisElementCount];

__host__ __device__ constexpr size_t DctBasisOffset(
  unsigned int length) {
  switch (length) {
    case 8:
      return 0;
    case 16:
      return 8 * 8;
    case 32:
      return 8 * 8 + 16 * 16;
    case 64:
      return 8 * 8 + 16 * 16 + 32 * 32;
    default:
      return kDctBasisElementCount;
  }
}

__device__ size_t CoefficientIndex(
  unsigned int width,
  unsigned int height,
  unsigned int vertical_frequency,
  unsigned int horizontal_frequency) {
  return height < width
    ? static_cast<size_t>(vertical_frequency) * width +
        horizontal_frequency
    : static_cast<size_t>(horizontal_frequency) * height +
        vertical_frequency;
}

__global__ void ForwardDctKernel(
  const float* input,
  float* output,
  unsigned int width,
  unsigned int height) {
  extern __shared__ float intermediate[];
  const size_t element_count = static_cast<size_t>(width) * height;
  const size_t base = static_cast<size_t>(blockIdx.x) * element_count;
  const float* horizontal_basis =
    kOrthonormalDctBasis + DctBasisOffset(width);
  const float* vertical_basis =
    kOrthonormalDctBasis + DctBasisOffset(height);

  for (size_t index = threadIdx.x; index < element_count;
       index += blockDim.x) {
    const unsigned int y = static_cast<unsigned int>(index / width);
    const unsigned int u = static_cast<unsigned int>(index % width);
    float value = 0.0f;
    for (unsigned int x = 0; x < width; ++x) {
      value += input[base + static_cast<size_t>(y) * width + x] *
        horizontal_basis[static_cast<size_t>(u) * width + x];
    }
    intermediate[index] = value;
  }
  __syncthreads();

  const float scale = rsqrtf(static_cast<float>(element_count));
  for (size_t index = threadIdx.x; index < element_count;
       index += blockDim.x) {
    const unsigned int v = static_cast<unsigned int>(index / width);
    const unsigned int u = static_cast<unsigned int>(index % width);
    float value = 0.0f;
    for (unsigned int y = 0; y < height; ++y) {
      value += vertical_basis[static_cast<size_t>(v) * height + y] *
        intermediate[static_cast<size_t>(y) * width + u];
    }
    output[base + CoefficientIndex(width, height, v, u)] = value * scale;
  }
}

__global__ void InverseDctKernel(
  const float* input,
  float* output,
  unsigned int width,
  unsigned int height) {
  extern __shared__ float intermediate[];
  const size_t element_count = static_cast<size_t>(width) * height;
  const size_t base = static_cast<size_t>(blockIdx.x) * element_count;
  const float* horizontal_basis =
    kOrthonormalDctBasis + DctBasisOffset(width);
  const float* vertical_basis =
    kOrthonormalDctBasis + DctBasisOffset(height);

  for (size_t index = threadIdx.x; index < element_count;
       index += blockDim.x) {
    const unsigned int v = static_cast<unsigned int>(index / width);
    const unsigned int x = static_cast<unsigned int>(index % width);
    float value = 0.0f;
    for (unsigned int u = 0; u < width; ++u) {
      value += input[
        base + CoefficientIndex(width, height, v, u)] *
        horizontal_basis[static_cast<size_t>(u) * width + x];
    }
    intermediate[index] = value;
  }
  __syncthreads();

  const float scale = sqrtf(static_cast<float>(element_count));
  for (size_t index = threadIdx.x; index < element_count;
       index += blockDim.x) {
    const unsigned int y = static_cast<unsigned int>(index / width);
    const unsigned int x = static_cast<unsigned int>(index % width);
    float value = 0.0f;
    for (unsigned int v = 0; v < height; ++v) {
      value += vertical_basis[static_cast<size_t>(v) * height + y] *
        intermediate[static_cast<size_t>(v) * width + x];
    }
    output[base + static_cast<size_t>(y) * width + x] = value * scale;
  }
}

__global__ void PointwiseAffineKernel(
  const float* input,
  float* output,
  unsigned int width,
  unsigned int height,
  unsigned int input_stride,
  unsigned int output_stride,
  float scale,
  float bias) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
    threadIdx.x;
  const size_t count = static_cast<size_t>(width) * height;
  if (index >= count) return;
  const unsigned int y = static_cast<unsigned int>(index / width);
  const unsigned int x = static_cast<unsigned int>(index -
    static_cast<size_t>(y) * width);
  output[static_cast<size_t>(y) * output_stride + x] =
    input[static_cast<size_t>(y) * input_stride + x] * scale + bias;
}

template <bool Horizontal>
__global__ void SeparableConvolutionKernel(
  const float* input,
  const float* weights,
  float* output,
  unsigned int width,
  unsigned int height,
  unsigned int input_stride,
  unsigned int output_stride,
  unsigned int kernel_size) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
    threadIdx.x;
  const size_t count = static_cast<size_t>(width) * height;
  if (index >= count) return;
  const unsigned int y = static_cast<unsigned int>(index / width);
  const unsigned int x = static_cast<unsigned int>(index -
    static_cast<size_t>(y) * width);
  const int radius = static_cast<int>(kernel_size / 2);
  const int center = static_cast<int>(Horizontal ? x : y);
  const int limit = static_cast<int>(Horizontal ? width : height);
  float sum = 0.0f;
  float weight_sum = 0.0f;
  for (int delta = -radius; delta <= radius; ++delta) {
    const int coordinate = center + delta;
    if (coordinate < 0 || coordinate >= limit) continue;
    const float weight = weights[delta + radius];
    const unsigned int source_x = Horizontal
      ? static_cast<unsigned int>(coordinate) : x;
    const unsigned int source_y = Horizontal
      ? y : static_cast<unsigned int>(coordinate);
    sum += input[static_cast<size_t>(source_y) * input_stride + source_x] *
      weight;
    weight_sum += weight;
  }
  output[static_cast<size_t>(y) * output_stride + x] = sum / weight_sum;
}

__device__ unsigned int MirrorRadius2(int coordinate, unsigned int size) {
  if (size == 1) return 0;
  if (coordinate < 0) return static_cast<unsigned int>(-coordinate - 1);
  if (coordinate >= static_cast<int>(size)) {
    return 2 * size - 1 - static_cast<unsigned int>(coordinate);
  }
  return static_cast<unsigned int>(coordinate);
}

__device__ float Symmetric5WeightedRow(
  const float* input,
  int x,
  int y,
  unsigned int width,
  unsigned int height,
  unsigned int input_stride,
  float center_weight,
  float near_weight,
  float far_weight) {
  const unsigned int source_y = MirrorRadius2(y, height);
  const unsigned int far_left = MirrorRadius2(x - 2, width);
  const unsigned int near_left = MirrorRadius2(x - 1, width);
  const unsigned int center = MirrorRadius2(x, width);
  const unsigned int near_right = MirrorRadius2(x + 1, width);
  const unsigned int far_right = MirrorRadius2(x + 2, width);
  const float* row = input + static_cast<size_t>(source_y) * input_stride;
  const float far = far_weight * (row[far_left] + row[far_right]);
  const float near = near_weight * (row[near_left] + row[near_right]);
  return far + (near + center_weight * row[center]);
}

__global__ void Symmetric5ConvolutionKernel(
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
  float distance5) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
    threadIdx.x;
  const size_t count = static_cast<size_t>(width) * height;
  if (index >= count) return;
  const unsigned int y = static_cast<unsigned int>(index / width);
  const unsigned int x = static_cast<unsigned int>(index -
    static_cast<size_t>(y) * width);
  float sum0 = Symmetric5WeightedRow(
    input, static_cast<int>(x), static_cast<int>(y), width, height,
    input_stride, distance0, distance1, distance2);
  sum0 += Symmetric5WeightedRow(
    input, static_cast<int>(x), static_cast<int>(y) - 2, width, height,
    input_stride, distance2, distance5, distance8);
  float sum1 = Symmetric5WeightedRow(
    input, static_cast<int>(x), static_cast<int>(y) + 2, width, height,
    input_stride, distance2, distance5, distance8);
  sum0 += Symmetric5WeightedRow(
    input, static_cast<int>(x), static_cast<int>(y) - 1, width, height,
    input_stride, distance1, distance4, distance5);
  sum1 += Symmetric5WeightedRow(
    input, static_cast<int>(x), static_cast<int>(y) + 1, width, height,
    input_stride, distance1, distance4, distance5);
  output[static_cast<size_t>(y) * output_stride + x] = sum0 + sum1;
}

__global__ void MaximumReductionKernel(
  const float* input,
  float* output,
  unsigned int width,
  unsigned int input_stride,
  unsigned int input_count) {
  __shared__ float values[kPrimitiveThreads];
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  float value = -INFINITY;
  if (index < input_count) {
    const unsigned int y = index / width;
    const unsigned int x = index - y * width;
    value = input[static_cast<size_t>(y) * input_stride + x];
  }
  values[threadIdx.x] = value;
  __syncthreads();
  for (unsigned int step = kPrimitiveThreads / 2; step != 0; step /= 2) {
    if (threadIdx.x < step) {
      values[threadIdx.x] = fmaxf(
        values[threadIdx.x], values[threadIdx.x + step]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[blockIdx.x] = values[0];
}

}  // namespace

cudaError_t InitializeCudaDctBasis() {
  std::array<float, kDctBasisElementCount> basis{};
  constexpr double kPi = 3.141592653589793238462643383279502884;
  for (const unsigned int length : {8u, 16u, 32u, 64u}) {
    const size_t offset = DctBasisOffset(length);
    const double scale = std::sqrt(2.0 / static_cast<double>(length));
    for (unsigned int frequency = 0; frequency < length; ++frequency) {
      const double alpha = frequency == 0 ? 1.0 / std::sqrt(2.0) : 1.0;
      for (unsigned int sample = 0; sample < length; ++sample) {
        const double angle =
          (static_cast<double>(sample) + 0.5) * frequency * kPi / length;
        basis[offset + static_cast<size_t>(frequency) * length + sample] =
          static_cast<float>(scale * alpha * std::cos(angle));
      }
    }
  }
  return cudaMemcpyToSymbol(
    kOrthonormalDctBasis, basis.data(), basis.size() * sizeof(float));
}

cudaError_t LaunchCudaDct(
  bool forward,
  const float* input,
  float* output,
  size_t transform_count,
  unsigned int width,
  unsigned int height,
  cudaStream_t stream) {
  const size_t shared_bytes =
    static_cast<size_t>(width) * height * sizeof(float);
  const dim3 grid(static_cast<unsigned int>(transform_count));
  const dim3 block(kThreadsPerTransform);
  if (forward) {
    ForwardDctKernel<<<grid, block, shared_bytes, stream>>>(
      input, output, width, height);
  } else {
    InverseDctKernel<<<grid, block, shared_bytes, stream>>>(
      input, output, width, height);
  }
  return cudaPeekAtLastError();
}

cudaError_t LaunchCudaPointwiseAffine(
  const float* input,
  float* output,
  unsigned int width,
  unsigned int height,
  unsigned int input_stride,
  unsigned int output_stride,
  float scale,
  float bias,
  cudaStream_t stream) {
  const size_t count = static_cast<size_t>(width) * height;
  const unsigned int blocks = static_cast<unsigned int>(
    (count + kPrimitiveThreads - 1) / kPrimitiveThreads);
  PointwiseAffineKernel<<<blocks, kPrimitiveThreads, 0, stream>>>(
    input, output, width, height, input_stride, output_stride, scale, bias);
  return cudaPeekAtLastError();
}

cudaError_t LaunchCudaSeparableConvolutionPass(
  bool horizontal,
  const float* input,
  const float* kernel,
  float* output,
  unsigned int width,
  unsigned int height,
  unsigned int input_stride,
  unsigned int output_stride,
  unsigned int kernel_size,
  cudaStream_t stream) {
  const size_t count = static_cast<size_t>(width) * height;
  const unsigned int blocks = static_cast<unsigned int>(
    (count + kPrimitiveThreads - 1) / kPrimitiveThreads);
  if (horizontal) {
    SeparableConvolutionKernel<true>
      <<<blocks, kPrimitiveThreads, 0, stream>>>(
        input, kernel, output, width, height, input_stride, output_stride,
        kernel_size);
  } else {
    SeparableConvolutionKernel<false>
      <<<blocks, kPrimitiveThreads, 0, stream>>>(
        input, kernel, output, width, height, input_stride, output_stride,
        kernel_size);
  }
  return cudaPeekAtLastError();
}

cudaError_t LaunchCudaSymmetric5Convolution(
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
  cudaStream_t stream) {
  const size_t count = static_cast<size_t>(width) * height;
  const unsigned int blocks = static_cast<unsigned int>(
    (count + kPrimitiveThreads - 1) / kPrimitiveThreads);
  Symmetric5ConvolutionKernel<<<blocks, kPrimitiveThreads, 0, stream>>>(
    input, output, width, height, input_stride, output_stride,
    distance0, distance1, distance2, distance4, distance8, distance5);
  return cudaPeekAtLastError();
}

cudaError_t LaunchCudaMaximumReduction(
  const float* input,
  float* output,
  unsigned int width,
  unsigned int input_stride,
  unsigned int input_count,
  cudaStream_t stream) {
  const unsigned int blocks =
    (input_count + kPrimitiveThreads - 1) / kPrimitiveThreads;
  MaximumReductionKernel<<<blocks, kPrimitiveThreads, 0, stream>>>(
    input, output, width, input_stride, input_count);
  return cudaPeekAtLastError();
}

}  // namespace gjxl::cuda_internal
