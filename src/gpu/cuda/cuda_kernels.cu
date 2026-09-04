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
// The generic kernels benefit from constant-memory broadcasts. Dominant
// shapes instead preload from this global copy with coalesced reads because
// their horizontal pass addresses a different basis row in every warp lane.
__device__ float kGlobalOrthonormalDctBasis[kDctBasisElementCount];

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

template <unsigned int Width, unsigned int Height,
          unsigned int TransformsPerBlock>
__global__ void ForwardDctPackedKernel(
  const float* input,
  float* output,
  size_t transform_count) {
  static_assert(Width == 8 || Width == 16);
  static_assert(Height == 8 || Height == 16);
  constexpr size_t kElementCount = static_cast<size_t>(Width) * Height;
  static_assert(kElementCount * TransformsPerBlock == kThreadsPerTransform);
  constexpr size_t kHorizontalBasisCount =
    static_cast<size_t>(Width) * (Width + 1);
  extern __shared__ float shared[];
  float* horizontal_basis = shared;
  float* intermediate = shared + kHorizontalBasisCount;

  for (size_t index = threadIdx.x;
       index < static_cast<size_t>(Width) * Width;
       index += blockDim.x) {
    const unsigned int u = static_cast<unsigned int>(index / Width);
    const unsigned int x = static_cast<unsigned int>(index % Width);
    horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x] =
      kGlobalOrthonormalDctBasis[DctBasisOffset(Width) + index];
  }
  __syncthreads();

  const unsigned int local_transform =
    static_cast<unsigned int>(threadIdx.x / kElementCount);
  const unsigned int local_index =
    static_cast<unsigned int>(threadIdx.x % kElementCount);
  const size_t transform_index =
    static_cast<size_t>(blockIdx.x) * TransformsPerBlock + local_transform;
  const bool active = transform_index < transform_count;
  const size_t base = transform_index * kElementCount;
  const unsigned int y = local_index / Width;
  const unsigned int u = local_index % Width;
  float value = 0.0f;
  if (active) {
    for (unsigned int x = 0; x < Width; ++x) {
      value += input[base + static_cast<size_t>(y) * Width + x] *
        horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x];
    }
  }
  intermediate[threadIdx.x] = value;
  __syncthreads();

  if (!active) return;
  const unsigned int v = local_index / Width;
  value = 0.0f;
  const float* vertical_basis =
    kOrthonormalDctBasis + DctBasisOffset(Height);
  const size_t intermediate_base =
    static_cast<size_t>(local_transform) * kElementCount;
  for (unsigned int source_y = 0; source_y < Height; ++source_y) {
    value += vertical_basis[static_cast<size_t>(v) * Height + source_y] *
      intermediate[intermediate_base +
        static_cast<size_t>(source_y) * Width + u];
  }
  constexpr bool kRowMajorCoefficients = Height < Width;
  const size_t coefficient_index = kRowMajorCoefficients
    ? static_cast<size_t>(v) * Width + u
    : static_cast<size_t>(u) * Height + v;
  output[base + coefficient_index] =
    value * rsqrtf(static_cast<float>(kElementCount));
}

template <unsigned int Width, unsigned int Height,
          unsigned int TransformsPerBlock>
__global__ void InverseDctPackedKernel(
  const float* input,
  float* output,
  size_t transform_count) {
  static_assert(Width == 8 || Width == 16);
  static_assert(Height == 8 || Height == 16);
  constexpr size_t kElementCount = static_cast<size_t>(Width) * Height;
  static_assert(kElementCount * TransformsPerBlock == kThreadsPerTransform);
  constexpr size_t kHorizontalBasisCount =
    static_cast<size_t>(Width) * (Width + 1);
  extern __shared__ float shared[];
  float* horizontal_basis = shared;
  float* intermediate = shared + kHorizontalBasisCount;

  for (size_t index = threadIdx.x;
       index < static_cast<size_t>(Width) * Width;
       index += blockDim.x) {
    const unsigned int u = static_cast<unsigned int>(index / Width);
    const unsigned int x = static_cast<unsigned int>(index % Width);
    horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x] =
      kGlobalOrthonormalDctBasis[DctBasisOffset(Width) + index];
  }
  __syncthreads();

  const unsigned int local_transform =
    static_cast<unsigned int>(threadIdx.x / kElementCount);
  const unsigned int local_index =
    static_cast<unsigned int>(threadIdx.x % kElementCount);
  const size_t transform_index =
    static_cast<size_t>(blockIdx.x) * TransformsPerBlock + local_transform;
  const bool active = transform_index < transform_count;
  const size_t base = transform_index * kElementCount;
  const unsigned int v = local_index / Width;
  const unsigned int x = local_index % Width;
  float value = 0.0f;
  if (active) {
    for (unsigned int u = 0; u < Width; ++u) {
      constexpr bool kRowMajorCoefficients = Height < Width;
      const size_t coefficient_index = kRowMajorCoefficients
        ? static_cast<size_t>(v) * Width + u
        : static_cast<size_t>(u) * Height + v;
      value += input[base + coefficient_index] *
        horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x];
    }
  }
  intermediate[threadIdx.x] = value;
  __syncthreads();

  if (!active) return;
  const unsigned int y = local_index / Width;
  value = 0.0f;
  const float* vertical_basis =
    kOrthonormalDctBasis + DctBasisOffset(Height);
  const size_t intermediate_base =
    static_cast<size_t>(local_transform) * kElementCount;
  for (unsigned int frequency = 0; frequency < Height; ++frequency) {
    value += vertical_basis[static_cast<size_t>(frequency) * Height + y] *
      intermediate[intermediate_base +
        static_cast<size_t>(frequency) * Width + x];
  }
  output[base + static_cast<size_t>(y) * Width + x] =
    value * sqrtf(static_cast<float>(kElementCount));
}

template <unsigned int Width, unsigned int Height>
__global__ void ForwardDctSpecializedKernel(
  const float* input,
  float* output) {
  static_assert(Width == 16 || Width == 32);
  static_assert(Height == 16 || Height == 32);
  constexpr size_t kElementCount = static_cast<size_t>(Width) * Height;
  constexpr size_t kHorizontalBasisCount =
    static_cast<size_t>(Width) * (Width + 1);
  extern __shared__ float shared[];
  // The extra column keeps a warp's fixed-sample, varying-frequency reads in
  // distinct shared-memory banks.
  float* horizontal_basis = shared;
  float* intermediate = shared + kHorizontalBasisCount;
  const size_t base = static_cast<size_t>(blockIdx.x) * kElementCount;

  for (size_t index = threadIdx.x;
       index < static_cast<size_t>(Width) * Width;
       index += blockDim.x) {
    const unsigned int u = static_cast<unsigned int>(index / Width);
    const unsigned int x = static_cast<unsigned int>(index % Width);
    horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x] =
      kGlobalOrthonormalDctBasis[DctBasisOffset(Width) + index];
  }
  __syncthreads();

  for (size_t index = threadIdx.x; index < kElementCount;
       index += blockDim.x) {
    const unsigned int y = static_cast<unsigned int>(index / Width);
    const unsigned int u = static_cast<unsigned int>(index % Width);
    float value = 0.0f;
    for (unsigned int x = 0; x < Width; ++x) {
      value += input[base + static_cast<size_t>(y) * Width + x] *
        horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x];
    }
    intermediate[index] = value;
  }
  __syncthreads();

  const float scale = rsqrtf(static_cast<float>(kElementCount));
  const float* vertical_basis =
    kOrthonormalDctBasis + DctBasisOffset(Height);
  for (size_t index = threadIdx.x; index < kElementCount;
       index += blockDim.x) {
    const unsigned int v = static_cast<unsigned int>(index / Width);
    const unsigned int u = static_cast<unsigned int>(index % Width);
    float value = 0.0f;
    for (unsigned int y = 0; y < Height; ++y) {
      value += vertical_basis[static_cast<size_t>(v) * Height + y] *
        intermediate[static_cast<size_t>(y) * Width + u];
    }
    constexpr bool kRowMajorCoefficients = Height < Width;
    const size_t coefficient_index = kRowMajorCoefficients
      ? static_cast<size_t>(v) * Width + u
      : static_cast<size_t>(u) * Height + v;
    output[base + coefficient_index] = value * scale;
  }
}

template <unsigned int Width, unsigned int Height>
__global__ void InverseDctSpecializedKernel(
  const float* input,
  float* output) {
  static_assert(Width == 16 || Width == 32);
  static_assert(Height == 16 || Height == 32);
  constexpr size_t kElementCount = static_cast<size_t>(Width) * Height;
  constexpr size_t kHorizontalBasisCount =
    static_cast<size_t>(Width) * (Width + 1);
  extern __shared__ float shared[];
  // Match the forward layout so varying sample indices remain bank-conflict
  // free in the inverse horizontal pass.
  float* horizontal_basis = shared;
  float* intermediate = shared + kHorizontalBasisCount;
  const size_t base = static_cast<size_t>(blockIdx.x) * kElementCount;

  for (size_t index = threadIdx.x;
       index < static_cast<size_t>(Width) * Width;
       index += blockDim.x) {
    const unsigned int u = static_cast<unsigned int>(index / Width);
    const unsigned int x = static_cast<unsigned int>(index % Width);
    horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x] =
      kGlobalOrthonormalDctBasis[DctBasisOffset(Width) + index];
  }
  __syncthreads();

  for (size_t index = threadIdx.x; index < kElementCount;
       index += blockDim.x) {
    const unsigned int v = static_cast<unsigned int>(index / Width);
    const unsigned int x = static_cast<unsigned int>(index % Width);
    float value = 0.0f;
    for (unsigned int u = 0; u < Width; ++u) {
      constexpr bool kRowMajorCoefficients = Height < Width;
      const size_t coefficient_index = kRowMajorCoefficients
        ? static_cast<size_t>(v) * Width + u
        : static_cast<size_t>(u) * Height + v;
      value += input[base + coefficient_index] *
        horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x];
    }
    intermediate[index] = value;
  }
  __syncthreads();

  const float scale = sqrtf(static_cast<float>(kElementCount));
  const float* vertical_basis =
    kOrthonormalDctBasis + DctBasisOffset(Height);
  for (size_t index = threadIdx.x; index < kElementCount;
       index += blockDim.x) {
    const unsigned int y = static_cast<unsigned int>(index / Width);
    const unsigned int x = static_cast<unsigned int>(index % Width);
    float value = 0.0f;
    for (unsigned int v = 0; v < Height; ++v) {
      value += vertical_basis[static_cast<size_t>(v) * Height + y] *
        intermediate[static_cast<size_t>(v) * Width + x];
    }
    output[base + static_cast<size_t>(y) * Width + x] = value * scale;
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
  cudaError_t error = cudaMemcpyToSymbol(
    kOrthonormalDctBasis, basis.data(), basis.size() * sizeof(float));
  if (error != cudaSuccess) return error;
  return cudaMemcpyToSymbol(
    kGlobalOrthonormalDctBasis, basis.data(), basis.size() * sizeof(float));
}

cudaError_t LaunchCudaDct(
  bool forward,
  const float* input,
  float* output,
  size_t transform_count,
  unsigned int width,
  unsigned int height,
  cudaStream_t stream) {
  const bool packed =
    (width == 8 && height == 8) ||
    (width == 16 && height == 8) ||
    (width == 8 && height == 16) ||
    (width == 16 && height == 16);
  const bool large_specialized =
    (width == 32 && height == 32) ||
    (width == 32 && height == 16) ||
    (width == 16 && height == 32);
  const size_t shared_floats = packed
    ? kThreadsPerTransform + static_cast<size_t>(width) * (width + 1)
    : static_cast<size_t>(width) * height +
        (large_specialized ? static_cast<size_t>(width) * (width + 1) : 0);
  const size_t shared_bytes = shared_floats * sizeof(float);
  const dim3 grid(static_cast<unsigned int>(transform_count));
  const dim3 block(kThreadsPerTransform);
  if (forward) {
    if (width == 8 && height == 8) {
      const dim3 packed_grid(static_cast<unsigned int>(
        (transform_count + 3) / 4));
      ForwardDctPackedKernel<8, 8, 4>
        <<<packed_grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 16 && height == 8) {
      const dim3 packed_grid(static_cast<unsigned int>(
        (transform_count + 1) / 2));
      ForwardDctPackedKernel<16, 8, 2>
        <<<packed_grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 8 && height == 16) {
      const dim3 packed_grid(static_cast<unsigned int>(
        (transform_count + 1) / 2));
      ForwardDctPackedKernel<8, 16, 2>
        <<<packed_grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 16 && height == 16) {
      ForwardDctPackedKernel<16, 16, 1>
        <<<grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 32 && height == 32) {
      ForwardDctSpecializedKernel<32, 32>
        <<<grid, block, shared_bytes, stream>>>(input, output);
    } else if (width == 32 && height == 16) {
      ForwardDctSpecializedKernel<32, 16>
        <<<grid, block, shared_bytes, stream>>>(input, output);
    } else if (width == 16 && height == 32) {
      ForwardDctSpecializedKernel<16, 32>
        <<<grid, block, shared_bytes, stream>>>(input, output);
    } else {
      ForwardDctKernel<<<grid, block, shared_bytes, stream>>>(
        input, output, width, height);
    }
  } else {
    if (width == 8 && height == 8) {
      const dim3 packed_grid(static_cast<unsigned int>(
        (transform_count + 3) / 4));
      InverseDctPackedKernel<8, 8, 4>
        <<<packed_grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 16 && height == 8) {
      const dim3 packed_grid(static_cast<unsigned int>(
        (transform_count + 1) / 2));
      InverseDctPackedKernel<16, 8, 2>
        <<<packed_grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 8 && height == 16) {
      const dim3 packed_grid(static_cast<unsigned int>(
        (transform_count + 1) / 2));
      InverseDctPackedKernel<8, 16, 2>
        <<<packed_grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 16 && height == 16) {
      InverseDctPackedKernel<16, 16, 1>
        <<<grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 32 && height == 32) {
      InverseDctSpecializedKernel<32, 32>
        <<<grid, block, shared_bytes, stream>>>(input, output);
    } else if (width == 32 && height == 16) {
      InverseDctSpecializedKernel<32, 16>
        <<<grid, block, shared_bytes, stream>>>(input, output);
    } else if (width == 16 && height == 32) {
      InverseDctSpecializedKernel<16, 32>
        <<<grid, block, shared_bytes, stream>>>(input, output);
    } else {
      InverseDctKernel<<<grid, block, shared_bytes, stream>>>(
        input, output, width, height);
    }
  }
  return cudaGetLastError();
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
  return cudaGetLastError();
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
  return cudaGetLastError();
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
  return cudaGetLastError();
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
  return cudaGetLastError();
}

}  // namespace gjxl::cuda_internal
