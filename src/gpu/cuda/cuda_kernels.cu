// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/cuda/cuda_kernels.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "gpu/cuda/cuda_ac_strategy_device.cuh"
#include "gpu/cuda/cuda_aq_exact_kernels.h"
#include "gpu/cuda/cuda_dct_factored.cuh"

namespace gjxl::cuda_internal {
namespace {

constexpr unsigned int kThreadsPerTransform = 256;
// Each large-transform lane accumulates two or four independent outputs.
constexpr unsigned int kSpecializedDctThreads = 256;
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

// AC search reads image rectangles directly into the DCT input tile. The
// contiguous transform API uses the same arithmetic with a plain pointer, so
// fusing the old gather pass changes neither the coefficient order nor sums.
template <unsigned int Width, unsigned int Height>
struct AcStrategyDctSource {
  const float* opsin_x;
  const float* opsin_y;
  const float* opsin_b;
  const AcStrategyCandidateDevice* candidates;
  CudaAcStrategyBatchParams params;

  struct Tile {
    const float* pixels;
    unsigned int row_stride;

    __device__ float operator[](size_t index) const {
      return pixels == nullptr ? NAN :
        pixels[(index / Width) * row_stride + index % Width];
    }
  };

  __device__ Tile ForTransform(size_t transform_index, bool active) const {
    // Inactive packed transforms must not fetch a descriptor. Invalid active
    // candidates still run through the barriers and propagate NaN to the cost.
    if (!active) return {nullptr, 0};
    const AcStrategyCandidateDevice candidate = candidates[transform_index / 3];
    if (!CandidateValid(candidate, params)) return {nullptr, 0};
    const unsigned int channel = static_cast<unsigned int>(transform_index % 3);
    const float* plane = channel == 0 ? opsin_x :
      (channel == 1 ? opsin_y : opsin_b);
    return {plane + static_cast<size_t>(candidate.block_y * 8) *
      params.opsin_row_stride + candidate.block_x * 8, params.opsin_row_stride};
  }
};

// Resident batches are channel-major, unlike the interleaved AC-search
// candidates. Read the validated anchor rectangle without a packed copy.
template <unsigned int Width>
struct AqDctImageSource {
  const float* planes[3];
  const CudaAqAnchor* anchors;
  CudaAqExactBatch batch;
  uint32_t stride;

  struct Tile {
    const float* pixels;
    uint32_t stride;
    __device__ float operator[](size_t index) const {
      return pixels[(index / Width) * stride + index % Width];
    }
  };

  __device__ Tile ForTransform(size_t transform, bool active) const {
    if (!active) return {nullptr, 0};
    const size_t channel = transform / batch.anchor_count;
    const size_t index = transform - channel * batch.anchor_count;
    const CudaAqAnchor anchor = anchors[batch.anchor_offset + index];
    // A dynamic planes[channel] subscript makes nvcc materialize this
    // by-value kernel argument in a local stack frame. Select fixed fields.
    const float* plane = channel == 0 ? planes[0] :
      (channel == 1 ? planes[1] : planes[2]);
    return {plane + static_cast<size_t>(anchor.y) * 8 * stride +
              anchor.x * 8, stride};
  }
};

// The factorized inverse already holds full pixel columns in registers.
// Store them directly in the reconstruction image, without a packed output
// or the tall-transform redistribution needed by AC-search loss reduction.
struct AqDctImageOutput {
  float* planes[3];
  const CudaAqAnchor* anchors;
  CudaAqExactBatch batch;
  uint32_t stride;

  template <unsigned int Height>
  __device__ void StoreColumns(const float* values, size_t transform,
                               unsigned int x) const {
    const size_t channel = transform / batch.anchor_count;
    const size_t index = transform - channel * batch.anchor_count;
    const CudaAqAnchor anchor = anchors[batch.anchor_offset + index];
    float* plane = channel == 0 ? planes[0] :
      (channel == 1 ? planes[1] : planes[2]);
    float* destination = plane +
      static_cast<size_t>(anchor.y) * 8 * stride + anchor.x * 8 + x;
#pragma unroll
    for (unsigned int y = 0; y < Height; ++y)
      destination[static_cast<size_t>(y) * stride] = values[y];
  }
};

template <unsigned int Width, unsigned int Height, typename Input>
__device__ auto DctInputTile(Input input, size_t transform_index, bool active) {
  if constexpr (std::is_pointer_v<Input>) {
    return input;
  } else {
    return input.ForTransform(transform_index, active);
  }
}

template <typename Tile>
__device__ float DctInputElement(Tile input, size_t base, size_t index) {
  if constexpr (std::is_pointer_v<Tile>) {
    return input[base + index];
  } else {
    return input[index];
  }
}

// Quantize AC residual coefficients directly into the inverse input tile.
// Channel rates are reduced before the horizontal pass reuses intermediate
// storage; no global residual-coefficient buffer is materialized.
struct AcStrategyResidualDctSource {
  const float* coefficients;
  const float* matrices;
  const float* quant_norms;
  const AcStrategyCandidateDevice* candidates;
  const signed char* y_to_x;
  const signed char* y_to_b;
  AcStrategyChannelRateDevice* channel_rates;
  CudaAcStrategyBatchParams params;

  template <unsigned int Width, unsigned int Height,
            unsigned int LocalThreads, unsigned int Values>
  __device__ void Load(float* tile, float* reduction, size_t transform_index,
                       unsigned int tid, bool active) const {
    constexpr unsigned int kCount = Width * Height;
    static_assert(LocalThreads * Values == kCount);
    const size_t candidate_index = transform_index / 3;
    const unsigned int channel = static_cast<unsigned int>(transform_index % 3);
    const size_t base = transform_index * kCount;
    const size_t y_base = (candidate_index * 3 + 1) * kCount;
    const unsigned int matrix_base = channel * kCount;
    const unsigned int inverse_matrix_base = (3 + channel) * kCount;
    bool valid = false;
    float norm = NAN;
    float factor = NAN;
    if (active) {
      const auto candidate = candidates[candidate_index];
      valid = CandidateValid(candidate, params);
      if (valid) {
        norm = quant_norms[candidate_index];
        factor = ComputeCflFactor(y_to_x, y_to_b, candidate, channel, params);
      }
    }
    static_assert(Values >= 2);
    float magnitude[Values / 2];
    unsigned int nonzero = 0;
#pragma unroll
    for (unsigned int pair = 0; pair < Values / 2; ++pair) {
#pragma unroll
      for (unsigned int half = 0; half < 2; ++half) {
        const unsigned int value = pair + half * (Values / 2);
        const unsigned int index = tid + value * LocalThreads;
        float rounded = NAN;
        float residual = NAN;
        if (valid) {
          const float decorrelated = coefficients[base + index] -
            coefficients[y_base + index] * factor;
          const float scaled = decorrelated *
            matrices[inverse_matrix_base + index] * norm;
          rounded = copysignf(floorf(fabsf(scaled) + 0.5f), scaled);
          residual = matrices[matrix_base + index] * (scaled - rounded);
        }
        constexpr bool kRowMajorCoefficients = Height < Width;
        const unsigned int v = kRowMajorCoefficients ? index / Width : index % Height;
        const unsigned int u = kRowMajorCoefficients ? index % Width : index / Height;
        tile[v * (Width + 1) + u] = active ? residual : 0.0f;
        // Compute the first halving step as each pair arrives. Counts are exact
        // integers and can accumulate immediately; magnitudes retain the original
        // FP32 halving tree without keeping every coefficient's rate live.
        const float rate = sqrtf(fabsf(rounded));
        if (half == 0) magnitude[pair] = rate;
        else magnitude[pair] += rate;
        nonzero += rounded != 0.0f ? 1u : 0u;
      }
    }
    // Match the separate residual kernel's halving order, even though the
    // inverse transform uses a different number of lanes per transform.
#pragma unroll
    for (unsigned int stride = Values / 4; stride != 0; stride /= 2) {
#pragma unroll
      for (unsigned int value = 0; value < stride; ++value) {
        magnitude[value] += magnitude[value + stride];
      }
    }
    if constexpr (LocalThreads > 32) {
      static_assert(LocalThreads == kSpecializedDctThreads);
      static_assert(kCount >= 2 * LocalThreads);
      reduction[tid] = magnitude[0];
      reinterpret_cast<unsigned int*>(reduction + LocalThreads)[tid] = nonzero;
      __syncthreads();
    }
    constexpr unsigned int kWarpWidth = LocalThreads < 32 ? LocalThreads : 32;
    if (tid < kWarpWidth) {
      float total_magnitude = magnitude[0];
      unsigned int total_nonzero = nonzero;
      if constexpr (LocalThreads > 32) {
        const auto* counts =
          reinterpret_cast<const unsigned int*>(reduction + LocalThreads);
        total_magnitude = reduction[tid] + reduction[tid + 128];
        total_magnitude += reduction[tid + 64] + reduction[tid + 192];
        float upper = reduction[tid + 32] + reduction[tid + 160];
        upper += reduction[tid + 96] + reduction[tid + 224];
        total_magnitude += upper;
        total_nonzero = counts[tid] + counts[tid + 128];
        total_nonzero += counts[tid + 64] + counts[tid + 192];
        unsigned int upper_nonzero = counts[tid + 32] + counts[tid + 160];
        upper_nonzero += counts[tid + 96] + counts[tid + 224];
        total_nonzero += upper_nonzero;
      }
      const unsigned int group_start = (threadIdx.x % 32) & ~(kWarpWidth - 1);
      const unsigned int mask = (0xffffffffu >> (32 - kWarpWidth)) << group_start;
#pragma unroll
      for (unsigned int stride = kWarpWidth / 2; stride != 0; stride /= 2) {
        total_magnitude += __shfl_down_sync(mask, total_magnitude, stride, kWarpWidth);
        total_nonzero += __shfl_down_sync(mask, total_nonzero, stride, kWarpWidth);
      }
      if (tid == 0 && active) {
        channel_rates[transform_index] = {total_magnitude, total_nonzero};
      }
    }
    // The caller's existing input-tile barrier protects these rate reads
    // before any thread overwrites intermediate storage with DCT outputs.
  }
};

// AC search consumes only a weighted eighth-power loss per channel, not
// reconstructed residual pixels. Reduce the inverse outputs in registers
// and reuse the now-dead horizontal basis storage for cross-warp reduction.
template <unsigned int Width, unsigned int Height>
struct AcStrategyDctLossOutput {
  const float* pixel_mask;
  const AcStrategyCandidateDevice* candidates;
  float* losses;
  CudaAcStrategyBatchParams params;

  template <unsigned int LocalThreads, unsigned int Values,
            bool OrthonormalPixels = true>
  __device__ void Store(float (&values)[Values], size_t transform_index,
                        unsigned int tid, float* reduction) const {
    const auto candidate = candidates[transform_index / 3];
    const unsigned int channel = static_cast<unsigned int>(transform_index % 3);
    const bool fits = Width <= params.pixel_width && Height <= params.pixel_height &&
      candidate.block_x <= (params.pixel_width - Width) / 8 &&
      candidate.block_y <= (params.pixel_height - Height) / 8;
    const size_t mask_base = static_cast<size_t>(candidate.block_y * 8) *
      params.pixel_mask_row_stride + candidate.block_x * 8;
    const float offset = channel == 0 ? 12.0f : channel == 1 ? 0.0f : 4.0f;
    const float scale = sqrtf(static_cast<float>(Width * Height));
#pragma unroll
    for (unsigned int value = 0; value < Values; ++value) {
      const unsigned int index = tid + value * LocalThreads;
      const float mask = fits ? pixel_mask[mask_base +
        (index / Width) * params.pixel_mask_row_stride + index % Width] : NAN;
      // Match the ordinary inverse's separately rounded FP32 pixel store.
      const float pixel = OrthonormalPixels
        ? __fmul_rn(values[value], scale) : values[value];
      float weighted = (mask + offset) * pixel;
      weighted *= weighted;
      weighted *= weighted;
      weighted *= weighted;
      values[value] = isfinite(mask) && mask > 0.0f ? weighted : NAN;
    }
    // Match the halving tree of the separate cost kernel.
#pragma unroll
    for (unsigned int stride = Values / 2; stride != 0; stride /= 2) {
#pragma unroll
      for (unsigned int value = 0; value < stride; ++value) {
        values[value] += values[value + stride];
      }
    }
    if constexpr (LocalThreads > 32) {
      // Only the large, single-transform blocks enter this path. Other lanes
      // can still read the intermediate/vertical basis, but not this storage.
      static_assert(LocalThreads == kSpecializedDctThreads);
      static_assert(Width * (Width + 1) >= LocalThreads);
      reduction[tid] = values[0];
      __syncthreads();
    }
    constexpr unsigned int kWarpWidth = LocalThreads < 32 ? LocalThreads : 32;
    if (tid < kWarpWidth) {
      float sum = values[0];
      if constexpr (LocalThreads > 32) {
        // One warp reads all eight warp-partials after a single barrier.
        // Keep the same 128/64/32 halving tree in registers.
        sum = reduction[tid] + reduction[tid + 128];
        sum += reduction[tid + 64] + reduction[tid + 192];
        float upper = reduction[tid + 32] + reduction[tid + 160];
        upper += reduction[tid + 96] + reduction[tid + 224];
        sum += upper;
      }
      // Each transform is a whole 8/16/32-lane group. Name that group exactly,
      // even when other tail groups have returned or progress independently.
      const unsigned int group_start = (threadIdx.x % 32) & ~(kWarpWidth - 1);
      const unsigned int mask = (0xffffffffu >> (32 - kWarpWidth)) << group_start;
#pragma unroll
      for (unsigned int stride = kWarpWidth / 2; stride != 0; stride /= 2) {
        sum += __shfl_down_sync(mask, sum, stride, kWarpWidth);
      }
      if (tid == 0) losses[transform_index] = sum;
    }
  }
};

constexpr unsigned int kFactoredDctThreads = 64;

template <unsigned int Width, unsigned int Height, typename Input>
__global__ void ForwardDctFactoredKernel(
  Input input, float* output, size_t transform_count) {
  constexpr unsigned int kLocal = Width > Height ? Width : Height;
  constexpr unsigned int kTransforms = kFactoredDctThreads / kLocal;
  constexpr unsigned int kTileSize = Height * (Width + 1);
  __shared__ float tiles[kTransforms * kTileSize];
  const unsigned int group = threadIdx.x / kLocal;
  const unsigned int tid = threadIdx.x % kLocal;
  const size_t transform = static_cast<size_t>(blockIdx.x) * kTransforms + group;
  const bool active = transform < transform_count;
  const size_t base = transform * Width * Height;
  float* tile = tiles + group * kTileSize;
  float values[kLocal], scratch[2 * kLocal];
  const auto source = DctInputTile<Width, Height>(input, transform, active);

  if constexpr (Width > Height) {
    // Wide transforms have row-major coefficients. Stage coalesced pixel
    // reads, then transform rows first so coefficient stores also coalesce.
    if (active) {
#pragma unroll
      for (unsigned int i = tid; i < Width * Height; i += kLocal) {
        tile[(i / Width) * (Width + 1) + i % Width] =
          DctInputElement(source, base, i);
      }
    }
    __syncthreads();
    if (active && tid < Height) {
#pragma unroll
      for (unsigned int x = 0; x < Width; ++x) values[x] = tile[tid * (Width + 1) + x];
      FactoredDct1D<Width, true>(values, scratch);
#pragma unroll
      for (unsigned int u = 0; u < Width; ++u) tile[tid * (Width + 1) + u] = values[u];
    }
    __syncthreads();
    if (active && tid < Width) {
#pragma unroll
      for (unsigned int y = 0; y < Height; ++y) values[y] = tile[y * (Width + 1) + tid];
      FactoredDct1D<Height, true>(values, scratch);
#pragma unroll
      for (unsigned int v = 0; v < Height; ++v)
        output[base + v * Width + tid] = values[v] * (1.0f / (Width * Height));
    }
  } else {
    // Tall/square transforms use column-major coefficients. Columns first
    // coalesces both the global input and final coefficient output.
    if (active && tid < Width) {
#pragma unroll
      for (unsigned int y = 0; y < Height; ++y)
        values[y] = DctInputElement(source, base, y * Width + tid);
      FactoredDct1D<Height, true>(values, scratch);
#pragma unroll
      for (unsigned int v = 0; v < Height; ++v) tile[v * (Width + 1) + tid] = values[v];
    }
    __syncthreads();
    if (active && tid < Height) {
#pragma unroll
      for (unsigned int u = 0; u < Width; ++u) values[u] = tile[tid * (Width + 1) + u];
      FactoredDct1D<Width, true>(values, scratch);
#pragma unroll
      for (unsigned int u = 0; u < Width; ++u)
        output[base + u * Height + tid] = values[u] * (1.0f / (Width * Height));
    }
  }
}

template <unsigned int Width, unsigned int Height, typename Input, typename Output>
__global__ void InverseDctFactoredKernel(
  Input input, Output output, size_t transform_count) {
  constexpr unsigned int kLocal = Width > Height ? Width : Height;
  [[maybe_unused]] constexpr unsigned int kValues = Width * Height / kLocal;
  constexpr unsigned int kTransforms = kFactoredDctThreads / kLocal;
  constexpr unsigned int kTileSize = Height * (Width + 1);
  __shared__ float tiles[kTransforms * kTileSize];
  const unsigned int group = threadIdx.x / kLocal;
  const unsigned int tid = threadIdx.x % kLocal;
  const size_t transform = static_cast<size_t>(blockIdx.x) * kTransforms + group;
  const bool active = transform < transform_count;
  [[maybe_unused]] const size_t base = transform * Width * Height;
  float* tile = tiles + group * kTileSize;
  float values[kLocal], scratch[2 * kLocal];
  if constexpr (!std::is_pointer_v<Input>) {
    // The residual source writes natural [v][u] coefficients and channel
    // rates. Subwarp reduction needs no separate shared scratch here.
    input.template Load<Width, Height, kLocal, kValues>(
      tile, tile, transform, tid, active);
    __syncthreads();
  } else if constexpr (Width > Height) {
    if (active) {
#pragma unroll
      for (unsigned int i = tid; i < Width * Height; i += kLocal)
        tile[(i / Width) * (Width + 1) + i % Width] = input[base + i];
    }
    __syncthreads();
  }
  if (active && tid < Height) {
#pragma unroll
    for (unsigned int u = 0; u < Width; ++u) {
      if constexpr (!std::is_pointer_v<Input> || Width > Height)
        values[u] = tile[tid * (Width + 1) + u];
      else
        values[u] = input[base + u * Height + tid];
    }
    FactoredDct1D<Width, false>(values, scratch);
    // Each lane has finished reading its own row before replacing it.
#pragma unroll
    for (unsigned int x = 0; x < Width; ++x) tile[tid * (Width + 1) + x] = values[x];
  }
  __syncthreads();
  if (active && tid < Width) {
#pragma unroll
    for (unsigned int v = 0; v < Height; ++v) values[v] = tile[v * (Width + 1) + tid];
    FactoredDct1D<Height, false>(values, scratch);
    if constexpr (std::is_pointer_v<Output>) {
#pragma unroll
      for (unsigned int y = 0; y < Height; ++y) output[base + y * Width + tid] = values[y];
    } else if constexpr (std::is_same_v<Output, AqDctImageOutput>) {
      output.template StoreColumns<Height>(values, transform, tid);
    } else if constexpr (Width >= Height) {
      float pixels[kValues];
#pragma unroll
      for (unsigned int y = 0; y < Height; ++y) pixels[y] = values[y];
      output.template Store<kLocal, kValues, false>(pixels, transform, tid, tile);
    }
  }
  if constexpr (!std::is_pointer_v<Output> &&
                !std::is_same_v<Output, AqDctImageOutput> && Height > Width) {
    // Redistribute tall output columns to row-major strided lane vectors.
    // This preserves the existing loss reduction tree and mask addressing.
    __syncthreads();
    if (active && tid < Width) {
#pragma unroll
      for (unsigned int y = 0; y < Height; ++y) tile[y * (Width + 1) + tid] = values[y];
    }
    __syncthreads();
    if (active) {
      float pixels[kValues];
#pragma unroll
      for (unsigned int value = 0; value < kValues; ++value) {
        const unsigned int index = tid + value * kLocal;
        pixels[value] = tile[(index / Width) * (Width + 1) + index % Width];
      }
      output.template Store<kLocal, kValues, false>(pixels, transform, tid, tile);
    }
  }
}

// Small transforms use eight independent accumulators per lane. Pack enough
// transforms into each block to retain 256 threads and share basis loading.
constexpr unsigned int kPackedDctOutputsPerThread = 8;

constexpr size_t PackedDctSharedFloats(unsigned int width, unsigned int height) {
  const size_t element_count = static_cast<size_t>(width) * height;
  const size_t local_threads = element_count / kPackedDctOutputsPerThread;
  const size_t transforms = kThreadsPerTransform / local_threads;
  const size_t intermediate_stride = element_count +
    (local_threads < 32 ? local_threads : 0);
  const size_t io_elements = static_cast<size_t>(height) * (width + 1);
  return static_cast<size_t>(width) * (width + 1) +
    transforms * (intermediate_stride + io_elements) +
    (height > kPackedDctOutputsPerThread
      ? static_cast<size_t>(height) * (height + 1) : 0);
}

template <unsigned int Width, unsigned int Height,
          unsigned int TransformsPerBlock, typename Input>
__global__ void ForwardDctPackedKernel(
  Input input,
  float* output,
  size_t transform_count) {
  static_assert(Width == 8 || Width == 16);
  static_assert(Height == 8 || Height == 16);
  constexpr size_t kElementCount = static_cast<size_t>(Width) * Height;
  constexpr unsigned int kLocalThreads =
    kThreadsPerTransform / TransformsPerBlock;
  constexpr unsigned int kOutputsPerThread = kElementCount / kLocalThreads;
  constexpr unsigned int kRowStep = kLocalThreads / Width;
  // Separate sub-warp transforms' shared banks while retaining dense rows.
  constexpr size_t kIntermediateStride = kElementCount +
    (kLocalThreads < 32 ? kLocalThreads : 0);
  static_assert(kLocalThreads % Width == 0);
  static_assert(kElementCount * TransformsPerBlock ==
    kThreadsPerTransform * kPackedDctOutputsPerThread);
  constexpr size_t kHorizontalBasisCount =
    static_cast<size_t>(Width) * (Width + 1);
  extern __shared__ float shared[];
  float* horizontal_basis = shared;
  float* intermediate = shared + kHorizontalBasisCount;
  constexpr size_t kPaddedElements = Height * (Width + 1);
  float* io_tile = intermediate + kIntermediateStride * TransformsPerBlock;
  float* vertical_basis_tile = io_tile + kPaddedElements * TransformsPerBlock;
  const float* vertical_basis = kOrthonormalDctBasis + DctBasisOffset(Height);
  const unsigned int local_transform = threadIdx.x / kLocalThreads;
  const unsigned int local_index = threadIdx.x % kLocalThreads;
  const size_t transform_index =
    static_cast<size_t>(blockIdx.x) * TransformsPerBlock + local_transform;
  const bool active = transform_index < transform_count;
  const auto input_tile =
    DctInputTile<Width, Height>(input, transform_index, active);
  const size_t base = transform_index * kElementCount;
  const size_t intermediate_base = local_transform * kIntermediateStride;
  const size_t tile_base = local_transform * kPaddedElements;

  // Multiple vertical basis addresses within a warp serialize constant
  // memory. Stage only those shapes; single-address warps keep broadcasts.
  if constexpr (kRowStep > 1) {
    for (size_t index = threadIdx.x; index < Height * Height;
         index += blockDim.x) {
      vertical_basis_tile[(index / Height) * (Height + 1) + index % Height] =
        kGlobalOrthonormalDctBasis[DctBasisOffset(Height) + index];
    }
  }

  for (size_t index = threadIdx.x;
       index < static_cast<size_t>(Width) * Width;
       index += blockDim.x) {
    const unsigned int u = static_cast<unsigned int>(index / Width);
    const unsigned int x = static_cast<unsigned int>(index % Width);
    horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x] =
      kGlobalOrthonormalDctBasis[DctBasisOffset(Width) + index];
  }
  // Stage complete, coalesced input rows. Inactive transforms initialize
  // their shared cells and participate in every barrier without global I/O.
  for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
    const unsigned int index = local_index + value * kLocalThreads;
    io_tile[tile_base + (index / Width) * (Width + 1) + index % Width] =
      active ? DctInputElement(input_tile, base, index) : 0.0f;
  }
  __syncthreads();

  const unsigned int first_row = local_index / Width;
  const unsigned int u = local_index % Width;
  float values[kOutputsPerThread] = {};
  if (active) {
    for (unsigned int x = 0; x < Width; ++x) {
      const float basis =
        horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x];
#pragma unroll
      for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
        const unsigned int y = first_row + value * kRowStep;
        values[value] +=
          io_tile[tile_base + static_cast<size_t>(y) * (Width + 1) + x] * basis;
      }
    }
  }
#pragma unroll
  for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
    intermediate[intermediate_base + local_index + value * kLocalThreads] =
      values[value];
    values[value] = 0.0f;
  }
  __syncthreads();

  for (unsigned int source_y = 0; source_y < Height; ++source_y) {
    const float sample = intermediate[intermediate_base +
      static_cast<size_t>(source_y) * Width + u];
#pragma unroll
    for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
      const unsigned int v = first_row + value * kRowStep;
      const float basis = kRowStep > 1
        ? vertical_basis_tile[static_cast<size_t>(v) * (Height + 1) + source_y]
        : vertical_basis[static_cast<size_t>(v) * Height + source_y];
      values[value] += basis * sample;
    }
  }
#pragma unroll
  for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
    const unsigned int v = first_row + value * kRowStep;
    io_tile[tile_base + static_cast<size_t>(v) * (Width + 1) + u] =
      values[value] * rsqrtf(static_cast<float>(kElementCount));
  }
  // Reuse the input tile to coalesce the native coefficient-layout stores.
  __syncthreads();
  if (!active) return;
#pragma unroll
  for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
    const unsigned int index = local_index + value * kLocalThreads;
    constexpr bool kRowMajorCoefficients = Height < Width;
    const unsigned int v = kRowMajorCoefficients ? index / Width : index % Height;
    const unsigned int u = kRowMajorCoefficients ? index % Width : index / Height;
    output[base + index] = io_tile[tile_base + v * (Width + 1) + u];
  }
}

template <unsigned int Width, unsigned int Height,
          unsigned int TransformsPerBlock, typename Input, typename Output>
__global__ void InverseDctPackedKernel(
  Input input,
  Output output,
  size_t transform_count) {
  static_assert(Width == 8 || Width == 16);
  static_assert(Height == 8 || Height == 16);
  constexpr size_t kElementCount = static_cast<size_t>(Width) * Height;
  constexpr unsigned int kLocalThreads =
    kThreadsPerTransform / TransformsPerBlock;
  constexpr unsigned int kOutputsPerThread = kElementCount / kLocalThreads;
  constexpr unsigned int kRowStep = kLocalThreads / Width;
  constexpr size_t kIntermediateStride = kElementCount +
    (kLocalThreads < 32 ? kLocalThreads : 0);
  static_assert(kLocalThreads % Width == 0);
  static_assert(kElementCount * TransformsPerBlock ==
    kThreadsPerTransform * kPackedDctOutputsPerThread);
  constexpr size_t kHorizontalBasisCount =
    static_cast<size_t>(Width) * (Width + 1);
  extern __shared__ float shared[];
  float* horizontal_basis = shared;
  float* intermediate = shared + kHorizontalBasisCount;
  constexpr size_t kPaddedElements = Height * (Width + 1);
  float* io_tile = intermediate + kIntermediateStride * TransformsPerBlock;
  float* vertical_basis_tile = io_tile + kPaddedElements * TransformsPerBlock;
  const float* vertical_basis = kOrthonormalDctBasis + DctBasisOffset(Height);
  const unsigned int local_transform = threadIdx.x / kLocalThreads;
  const unsigned int local_index = threadIdx.x % kLocalThreads;
  const size_t transform_index =
    static_cast<size_t>(blockIdx.x) * TransformsPerBlock + local_transform;
  const bool active = transform_index < transform_count;
  [[maybe_unused]] const size_t base = transform_index * kElementCount;
  const size_t intermediate_base = local_transform * kIntermediateStride;
  const size_t tile_base = local_transform * kPaddedElements;

  if constexpr (kRowStep > 1) {
    for (size_t index = threadIdx.x; index < Height * Height;
         index += blockDim.x) {
      vertical_basis_tile[(index / Height) * (Height + 1) + index % Height] =
        kGlobalOrthonormalDctBasis[DctBasisOffset(Height) + index];
    }
  }

  for (size_t index = threadIdx.x;
       index < static_cast<size_t>(Width) * Width;
       index += blockDim.x) {
    const unsigned int u = static_cast<unsigned int>(index / Width);
    const unsigned int x = static_cast<unsigned int>(index % Width);
    horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x] =
      kGlobalOrthonormalDctBasis[DctBasisOffset(Width) + index];
  }
  if constexpr (std::is_pointer_v<Input>) {
    for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
      const unsigned int index = local_index + value * kLocalThreads;
      constexpr bool kRowMajorCoefficients = Height < Width;
      const unsigned int v = kRowMajorCoefficients ? index / Width : index % Height;
      const unsigned int u = kRowMajorCoefficients ? index % Width : index / Height;
      io_tile[tile_base + v * (Width + 1) + u] =
        active ? input[base + index] : 0.0f;
    }
  } else {
    input.template Load<Width, Height, kLocalThreads, kOutputsPerThread>(
      io_tile + tile_base, intermediate + intermediate_base,
      transform_index, local_index, active);
  }
  __syncthreads();

  const unsigned int first_row = local_index / Width;
  const unsigned int x = local_index % Width;
  float values[kOutputsPerThread] = {};
  if (active) {
    for (unsigned int u = 0; u < Width; ++u) {
      const float basis =
        horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x];
#pragma unroll
      for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
        const unsigned int v = first_row + value * kRowStep;
        values[value] +=
          io_tile[tile_base + static_cast<size_t>(v) * (Width + 1) + u] * basis;
      }
    }
  }
#pragma unroll
  for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
    intermediate[intermediate_base + local_index + value * kLocalThreads] =
      values[value];
    values[value] = 0.0f;
  }
  __syncthreads();

  if (!active) return;
  for (unsigned int frequency = 0; frequency < Height; ++frequency) {
    const float sample = intermediate[intermediate_base +
      static_cast<size_t>(frequency) * Width + x];
#pragma unroll
    for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
      const unsigned int y = first_row + value * kRowStep;
      const float basis = kRowStep > 1
        ? vertical_basis_tile[static_cast<size_t>(frequency) * (Height + 1) + y]
        : vertical_basis[static_cast<size_t>(frequency) * Height + y];
      values[value] += basis * sample;
    }
  }
  if constexpr (std::is_pointer_v<Output>) {
#pragma unroll
    for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
      const unsigned int y = first_row + value * kRowStep;
      output[base + static_cast<size_t>(y) * Width + x] =
        values[value] * sqrtf(static_cast<float>(kElementCount));
    }
  } else {
    output.template Store<kLocalThreads, kOutputsPerThread>(
      values, transform_index, local_index, horizontal_basis);
  }
}

template <unsigned int Width, unsigned int Height, typename Input>
__global__ void ForwardDctSpecializedKernel(
  Input input,
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
  // Reuse this padded tile for coalesced input and transposed output. All
  // horizontal input reads finish before the second block barrier.
  float* io_tile = intermediate + kElementCount;
  float* vertical_basis_tile = io_tile + Height * (Width + 1);
  const size_t base = static_cast<size_t>(blockIdx.x) * kElementCount;
  const auto input_tile = DctInputTile<Width, Height>(input, blockIdx.x, true);

  for (size_t index = threadIdx.x;
       index < static_cast<size_t>(Width) * Width;
       index += blockDim.x) {
    const unsigned int u = static_cast<unsigned int>(index / Width);
    const unsigned int x = static_cast<unsigned int>(index % Width);
    horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x] =
      kGlobalOrthonormalDctBasis[DctBasisOffset(Width) + index];
  }
  for (size_t index = threadIdx.x; index < kElementCount;
       index += blockDim.x) {
    io_tile[(index / Width) * (Width + 1) + index % Width] =
      DctInputElement(input_tile, base, index);
  }
  if constexpr (Width < 32) {
    // A 16-wide transform has two vertical basis addresses per warp. Stage
    // that basis too; the 32-wide shapes retain constant-memory broadcasts.
    for (size_t index = threadIdx.x; index < Height * Height;
         index += blockDim.x) {
      vertical_basis_tile[(index / Height) * (Height + 1) + index % Height] =
        kGlobalOrthonormalDctBasis[DctBasisOffset(Height) + index];
    }
  }
  __syncthreads();

  constexpr unsigned int kOutputsPerThread =
    kElementCount / kSpecializedDctThreads;
  constexpr unsigned int kRowStep = kSpecializedDctThreads / Width;
  const unsigned int first_row = threadIdx.x / Width;
  const unsigned int u = threadIdx.x % Width;
  // Share basis/sample loads across independent accumulators without
  // reassociating any output's sequence of multiply-adds.
  float values[kOutputsPerThread] = {};
  for (unsigned int x = 0; x < Width; ++x) {
    const float basis =
      horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x];
#pragma unroll
    for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
      const unsigned int y = first_row + value * kRowStep;
      values[value] +=
        io_tile[static_cast<size_t>(y) * (Width + 1) + x] * basis;
    }
  }
#pragma unroll
  for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
    intermediate[threadIdx.x + value * kSpecializedDctThreads] = values[value];
    values[value] = 0.0f;
  }
  __syncthreads();

  const float scale = rsqrtf(static_cast<float>(kElementCount));
  const float* vertical_basis =
    kOrthonormalDctBasis + DctBasisOffset(Height);
  for (unsigned int y = 0; y < Height; ++y) {
    const float sample = intermediate[static_cast<size_t>(y) * Width + u];
#pragma unroll
    for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
      const unsigned int v = first_row + value * kRowStep;
      const float basis = Width < 32
        ? vertical_basis_tile[static_cast<size_t>(v) * (Height + 1) + y]
        : vertical_basis[static_cast<size_t>(v) * Height + y];
      values[value] += basis * sample;
    }
  }
#pragma unroll
  for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
    const unsigned int v = first_row + value * kRowStep;
    io_tile[static_cast<size_t>(v) * (Width + 1) + u] = values[value] * scale;
  }
  __syncthreads();
  for (size_t index = threadIdx.x; index < kElementCount;
       index += blockDim.x) {
    constexpr bool kRowMajorCoefficients = Height < Width;
    const size_t v = kRowMajorCoefficients ? index / Width : index % Height;
    const size_t u = kRowMajorCoefficients ? index % Width : index / Height;
    output[base + index] = io_tile[v * (Width + 1) + u];
  }
}

template <unsigned int Width, unsigned int Height, typename Input, typename Output>
__global__ void InverseDctSpecializedKernel(
  Input input,
  Output output) {
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
  // Load the native coefficient layout contiguously, then broadcast the
  // horizontal pass's samples from the padded shared tile.
  float* input_tile = intermediate + kElementCount;
  float* vertical_basis_tile = input_tile + Height * (Width + 1);
  [[maybe_unused]] const size_t base = static_cast<size_t>(blockIdx.x) * kElementCount;

  for (size_t index = threadIdx.x;
       index < static_cast<size_t>(Width) * Width;
       index += blockDim.x) {
    const unsigned int u = static_cast<unsigned int>(index / Width);
    const unsigned int x = static_cast<unsigned int>(index % Width);
    horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x] =
      kGlobalOrthonormalDctBasis[DctBasisOffset(Width) + index];
  }
  if constexpr (std::is_pointer_v<Input>) {
    for (size_t index = threadIdx.x; index < kElementCount;
         index += blockDim.x) {
      constexpr bool kRowMajorCoefficients = Height < Width;
      const size_t v = kRowMajorCoefficients ? index / Width : index % Height;
      const size_t u = kRowMajorCoefficients ? index % Width : index / Height;
      input_tile[v * (Width + 1) + u] = input[base + index];
    }
  } else {
    input.template Load<Width, Height, kSpecializedDctThreads,
                        kElementCount / kSpecializedDctThreads>(
      input_tile, intermediate, blockIdx.x, threadIdx.x, true);
  }
  if constexpr (Width < 32) {
    for (size_t index = threadIdx.x; index < Height * Height;
         index += blockDim.x) {
      vertical_basis_tile[(index / Height) * (Height + 1) + index % Height] =
        kGlobalOrthonormalDctBasis[DctBasisOffset(Height) + index];
    }
  }
  __syncthreads();

  constexpr unsigned int kOutputsPerThread =
    kElementCount / kSpecializedDctThreads;
  constexpr unsigned int kRowStep = kSpecializedDctThreads / Width;
  const unsigned int first_row = threadIdx.x / Width;
  const unsigned int x = threadIdx.x % Width;
  float values[kOutputsPerThread] = {};
  for (unsigned int u = 0; u < Width; ++u) {
    const float basis =
      horizontal_basis[static_cast<size_t>(u) * (Width + 1) + x];
#pragma unroll
    for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
      const unsigned int v = first_row + value * kRowStep;
      values[value] +=
        input_tile[static_cast<size_t>(v) * (Width + 1) + u] * basis;
    }
  }
#pragma unroll
  for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
    intermediate[threadIdx.x + value * kSpecializedDctThreads] = values[value];
    values[value] = 0.0f;
  }
  __syncthreads();

  const float scale = sqrtf(static_cast<float>(kElementCount));
  const float* vertical_basis =
    kOrthonormalDctBasis + DctBasisOffset(Height);
  for (unsigned int v = 0; v < Height; ++v) {
    const float sample = intermediate[static_cast<size_t>(v) * Width + x];
#pragma unroll
    for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
      const unsigned int y = first_row + value * kRowStep;
      const float basis = Width < 32
        ? vertical_basis_tile[static_cast<size_t>(v) * (Height + 1) + y]
        : vertical_basis[static_cast<size_t>(v) * Height + y];
      values[value] += basis * sample;
    }
  }
  if constexpr (std::is_pointer_v<Output>) {
#pragma unroll
    for (unsigned int value = 0; value < kOutputsPerThread; ++value) {
      const unsigned int y = first_row + value * kRowStep;
      output[base + static_cast<size_t>(y) * Width + x] = values[value] * scale;
    }
  } else {
    output.template Store<kSpecializedDctThreads, kOutputsPerThread>(
      values, blockIdx.x, threadIdx.x, horizontal_basis);
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
  bool forward, const float* input, float* output, size_t transform_count,
  unsigned int width, unsigned int height, cudaStream_t stream) {
  const auto launch = [&](auto w, auto h) {
    constexpr unsigned int kWidth = decltype(w)::value;
    constexpr unsigned int kHeight = decltype(h)::value;
    constexpr unsigned int kLocal = kWidth > kHeight ? kWidth : kHeight;
    constexpr unsigned int kTransforms = kFactoredDctThreads / kLocal;
    const unsigned int blocks = static_cast<unsigned int>(
      (transform_count + kTransforms - 1) / kTransforms);
    if (forward) {
      ForwardDctFactoredKernel<kWidth, kHeight>
        <<<blocks, kFactoredDctThreads, 0, stream>>>(input, output, transform_count);
    } else {
      InverseDctFactoredKernel<kWidth, kHeight>
        <<<blocks, kFactoredDctThreads, 0, stream>>>(input, output, transform_count);
    }
  };
  using N8 = std::integral_constant<unsigned int, 8>;
  using N16 = std::integral_constant<unsigned int, 16>;
  using N32 = std::integral_constant<unsigned int, 32>;
  if (width == 8 && height == 8) launch(N8{}, N8{});
  else if (width == 16 && height == 8) launch(N16{}, N8{});
  else if (width == 8 && height == 16) launch(N8{}, N16{});
  else if (width == 16 && height == 16) launch(N16{}, N16{});
  else if (width == 32 && height == 16) launch(N32{}, N16{});
  else if (width == 16 && height == 32) launch(N16{}, N32{});
  else if (width == 32 && height == 32) launch(N32{}, N32{});
  else return LaunchCudaDctMatrix(forward, input, output, transform_count,
    width, height, stream);
  return cudaGetLastError();
}

template <bool Forward>
cudaError_t LaunchAqImageDct(
  std::array<const float*, 3> coding, const float* input,
  std::array<float*, 3> reconstructed, float* output,
  const CudaAqAnchor* anchors, uint32_t stride, CudaAqExactBatch batch,
  cudaStream_t stream) {
  const size_t count = 3 * static_cast<size_t>(batch.anchor_count);
  const auto launch = [&](auto w, auto h) {
    if (count == 0) return;
    constexpr unsigned int kWidth = decltype(w)::value;
    constexpr unsigned int kHeight = decltype(h)::value;
    constexpr unsigned int kLocal = kWidth > kHeight ? kWidth : kHeight;
    constexpr unsigned int kTransforms = kFactoredDctThreads / kLocal;
    const unsigned int blocks = static_cast<unsigned int>(
      (count + kTransforms - 1) / kTransforms);
    if constexpr (Forward) {
      const AqDctImageSource<kWidth> source{
        {coding[0], coding[1], coding[2]}, anchors, batch, stride};
      ForwardDctFactoredKernel<kWidth, kHeight>
        <<<blocks, kFactoredDctThreads, 0, stream>>>(
          source, output + batch.coefficient_offset, count);
    } else {
      const AqDctImageOutput destination{
        {reconstructed[0], reconstructed[1], reconstructed[2]}, anchors,
        batch, stride};
      InverseDctFactoredKernel<kWidth, kHeight>
        <<<blocks, kFactoredDctThreads, 0, stream>>>(
          input + batch.coefficient_offset, destination, count);
    }
  };
  using N8 = std::integral_constant<unsigned int, 8>;
  using N16 = std::integral_constant<unsigned int, 16>;
  using N32 = std::integral_constant<unsigned int, 32>;
  if (batch.pixel_width == 8 && batch.pixel_height == 8) launch(N8{}, N8{});
  else if (batch.pixel_width == 16 && batch.pixel_height == 8) launch(N16{}, N8{});
  else if (batch.pixel_width == 8 && batch.pixel_height == 16) launch(N8{}, N16{});
  else if (batch.pixel_width == 16 && batch.pixel_height == 16) launch(N16{}, N16{});
  else if (batch.pixel_width == 32 && batch.pixel_height == 16) launch(N32{}, N16{});
  else if (batch.pixel_width == 16 && batch.pixel_height == 32) launch(N16{}, N32{});
  else if (batch.pixel_width == 32 && batch.pixel_height == 32) launch(N32{}, N32{});
  else return cudaErrorInvalidValue;
  return count == 0 ? cudaSuccess : cudaGetLastError();
}

cudaError_t LaunchCudaAqForwardDct(
  std::array<const float*, 3> coding, const CudaAqAnchor* anchors,
  float* coefficients, uint32_t coding_stride, CudaAqExactBatch batch,
  cudaStream_t stream) {
  return LaunchAqImageDct<true>(coding, nullptr, {}, coefficients, anchors,
    coding_stride, batch, stream);
}

cudaError_t LaunchCudaAqInverseDct(
  const float* coefficients, const CudaAqAnchor* anchors,
  std::array<float*, 3> reconstructed, uint32_t coding_stride,
  CudaAqExactBatch batch, cudaStream_t stream) {
  return LaunchAqImageDct<false>({}, coefficients, reconstructed, nullptr,
    anchors, coding_stride, batch, stream);
}

cudaError_t LaunchCudaDctMatrix(
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
    ? PackedDctSharedFloats(width, height)
    : static_cast<size_t>(width) * height +
        (large_specialized
          ? static_cast<size_t>(width + height) * (width + 1) +
            (width < 32 ? static_cast<size_t>(height) * (height + 1) : 0) : 0);
  const size_t shared_bytes = shared_floats * sizeof(float);
  const dim3 grid(static_cast<unsigned int>(transform_count));
  const dim3 block(
    large_specialized ? kSpecializedDctThreads : kThreadsPerTransform);
  const unsigned int packed_transforms = packed
    ? kThreadsPerTransform * kPackedDctOutputsPerThread / (width * height) : 1;
  const dim3 packed_grid(static_cast<unsigned int>(
    (transform_count + packed_transforms - 1) / packed_transforms));
  if (forward) {
    if (width == 8 && height == 8) {
      ForwardDctPackedKernel<8, 8, 4 * kPackedDctOutputsPerThread>
        <<<packed_grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 16 && height == 8) {
      ForwardDctPackedKernel<16, 8, 2 * kPackedDctOutputsPerThread>
        <<<packed_grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 8 && height == 16) {
      ForwardDctPackedKernel<8, 16, 2 * kPackedDctOutputsPerThread>
        <<<packed_grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 16 && height == 16) {
      ForwardDctPackedKernel<16, 16, kPackedDctOutputsPerThread>
        <<<packed_grid, block, shared_bytes, stream>>>(
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
      InverseDctPackedKernel<8, 8, 4 * kPackedDctOutputsPerThread>
        <<<packed_grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 16 && height == 8) {
      InverseDctPackedKernel<16, 8, 2 * kPackedDctOutputsPerThread>
        <<<packed_grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 8 && height == 16) {
      InverseDctPackedKernel<8, 16, 2 * kPackedDctOutputsPerThread>
        <<<packed_grid, block, shared_bytes, stream>>>(
          input, output, transform_count);
    } else if (width == 16 && height == 16) {
      InverseDctPackedKernel<16, 16, kPackedDctOutputsPerThread>
        <<<packed_grid, block, shared_bytes, stream>>>(
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

cudaError_t LaunchCudaAcStrategyForward(
  const float* opsin_x, const float* opsin_y, const float* opsin_b,
  const void* candidates, float* output, CudaAcStrategyBatchParams params,
  cudaStream_t stream) {
  const size_t transform_count = static_cast<size_t>(params.candidate_count) * 3;
  const auto launch = [&](auto width, auto height) {
    constexpr unsigned int kWidth = decltype(width)::value;
    constexpr unsigned int kHeight = decltype(height)::value;
    const AcStrategyDctSource<kWidth, kHeight> input{
      opsin_x, opsin_y, opsin_b,
      static_cast<const AcStrategyCandidateDevice*>(candidates), params};
    constexpr unsigned int kLocal = kWidth > kHeight ? kWidth : kHeight;
    constexpr unsigned int kTransforms = kFactoredDctThreads / kLocal;
    const unsigned int blocks = static_cast<unsigned int>(
      (transform_count + kTransforms - 1) / kTransforms);
    ForwardDctFactoredKernel<kWidth, kHeight>
      <<<blocks, kFactoredDctThreads, 0, stream>>>(input, output, transform_count);
  };
  using N8 = std::integral_constant<unsigned int, 8>;
  using N16 = std::integral_constant<unsigned int, 16>;
  using N32 = std::integral_constant<unsigned int, 32>;
  if (params.transform_width == 8 && params.transform_height == 8) {
    launch(N8{}, N8{});
  } else if (params.transform_width == 16 && params.transform_height == 8) {
    launch(N16{}, N8{});
  } else if (params.transform_width == 8 && params.transform_height == 16) {
    launch(N8{}, N16{});
  } else if (params.transform_width == 16 && params.transform_height == 16) {
    launch(N16{}, N16{});
  } else if (params.transform_width == 32 && params.transform_height == 16) {
    launch(N32{}, N16{});
  } else if (params.transform_width == 16 && params.transform_height == 32) {
    launch(N16{}, N32{});
  } else if (params.transform_width == 32 && params.transform_height == 32) {
    launch(N32{}, N32{});
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

template <typename Input>
cudaError_t LaunchAcStrategyInverseLossImpl(
  Input input, const float* pixel_mask,
  const void* candidates, float* losses, CudaAcStrategyBatchParams params,
  cudaStream_t stream) {
  const size_t transform_count = static_cast<size_t>(params.candidate_count) * 3;
  const auto launch = [&](auto width, auto height) {
    constexpr unsigned int kWidth = decltype(width)::value;
    constexpr unsigned int kHeight = decltype(height)::value;
    const AcStrategyDctLossOutput<kWidth, kHeight> output{
      pixel_mask, static_cast<const AcStrategyCandidateDevice*>(candidates),
      losses, params};
    constexpr unsigned int kLocal = kWidth > kHeight ? kWidth : kHeight;
    constexpr unsigned int kTransforms = kFactoredDctThreads / kLocal;
    const unsigned int blocks = static_cast<unsigned int>(
      (transform_count + kTransforms - 1) / kTransforms);
    InverseDctFactoredKernel<kWidth, kHeight>
      <<<blocks, kFactoredDctThreads, 0, stream>>>(input, output, transform_count);
  };
  using N8 = std::integral_constant<unsigned int, 8>;
  using N16 = std::integral_constant<unsigned int, 16>;
  using N32 = std::integral_constant<unsigned int, 32>;
  if (params.transform_width == 8 && params.transform_height == 8) {
    launch(N8{}, N8{});
  } else if (params.transform_width == 16 && params.transform_height == 8) {
    launch(N16{}, N8{});
  } else if (params.transform_width == 8 && params.transform_height == 16) {
    launch(N8{}, N16{});
  } else if (params.transform_width == 16 && params.transform_height == 16) {
    launch(N16{}, N16{});
  } else if (params.transform_width == 32 && params.transform_height == 16) {
    launch(N32{}, N16{});
  } else if (params.transform_width == 16 && params.transform_height == 32) {
    launch(N16{}, N32{});
  } else if (params.transform_width == 32 && params.transform_height == 32) {
    launch(N32{}, N32{});
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t LaunchCudaAcStrategyInverseLoss(
  const float* input, const float* pixel_mask,
  const void* candidates, float* losses, CudaAcStrategyBatchParams params,
  cudaStream_t stream) {
  return LaunchAcStrategyInverseLossImpl(
    input, pixel_mask, candidates, losses, params, stream);
}

cudaError_t LaunchCudaAcStrategyResidualInverseLoss(
  const float* coefficients, const float* matrices, const float* quant_norms,
  const signed char* y_to_x, const signed char* y_to_b, const float* pixel_mask,
  const void* candidates, void* channel_rates, float* losses,
  CudaAcStrategyBatchParams params, cudaStream_t stream) {
  const AcStrategyResidualDctSource input{
    coefficients, matrices, quant_norms,
    static_cast<const AcStrategyCandidateDevice*>(candidates), y_to_x, y_to_b,
    static_cast<AcStrategyChannelRateDevice*>(channel_rates), params};
  return LaunchAcStrategyInverseLossImpl(
    input, pixel_mask, candidates, losses, params, stream);
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
