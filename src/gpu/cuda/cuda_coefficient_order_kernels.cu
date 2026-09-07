// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/cuda/cuda_coefficient_order_kernels.h"

#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
#include "codec/coefficient_order_population_internal.h"

namespace gjxl::cuda_internal {
namespace {

struct PopulationBatch {
  uint32_t source_offset;
  uint32_t count;
  uint32_t size;
  uint32_t population_offset;
  uint32_t tiles;
};
struct PopulationBatches {
  PopulationBatch batches[7];
  uint32_t count;
};
static_assert(sizeof(PopulationBatches) == 144);

template <bool SampleDct8>
__global__ void CoefficientOrderPopulationKernel(
  const int* source, PopulationBatches list, const uint8_t* sampled_dct8,
  uint32_t* populations) {
  using namespace vardct_frame_internal;
  uint32_t tile = blockIdx.x, batch_index = 0;
  while (batch_index + 1 < list.count && tile >= list.batches[batch_index].tiles)
    tile -= list.batches[batch_index++].tiles;
  const PopulationBatch batch = list.batches[batch_index];
  const uint32_t coefficient_tiles = batch.size / 32;
  const uint32_t coefficient = (tile & (coefficient_tiles - 1)) * 32 + threadIdx.x;
  const uint32_t first = (tile >> (__ffs(coefficient_tiles) - 1)) * 64;
  constexpr uint32_t channels = SampleDct8 ? 6 : 3;
  __shared__ uint32_t partial[channels][8][32];
  uint32_t counts[channels]{};
  const size_t stride = static_cast<size_t>(batch.count) * batch.size;
  for (uint32_t i = threadIdx.y; i < 64; i += 8) {
    const uint32_t index = first + i;
    if (index < batch.count) {
      const size_t offset = batch.source_offset + static_cast<size_t>(index) * batch.size + coefficient;
      for (uint32_t channel = 0; channel < 3; ++channel) {
        const uint32_t zero = source[offset + channel * stride] == 0;
        counts[channel] += zero;
        if constexpr (SampleDct8) counts[channel + 3] += zero && sampled_dct8[index];
      }
    }
  }
  for (uint32_t channel = 0; channel < channels; ++channel)
    partial[channel][threadIdx.y][threadIdx.x] = counts[channel];
  __syncthreads();
  if (threadIdx.y == 0) {
    for (uint32_t channel = 0; channel < channels; ++channel) {
      uint32_t sum = 0;
      #pragma unroll
      for (uint32_t y = 0; y < 8; ++y) sum += partial[channel][y][threadIdx.x];
      const size_t offset = channel < 3
        ? channel * kOrderPopulationStride + batch.population_offset + coefficient
        : kOrderPopulationFullCount + (channel - 3) * 64 + coefficient;
      if (sum) atomicAdd(populations + offset, sum);
    }
  }
}

}  // namespace

cudaError_t LaunchCudaCoefficientOrderPopulation(
  const int* source, const std::array<CudaAqExactBatch, 7>& batches,
  const uint8_t* sampled_dct8, uint32_t* populations, cudaStream_t stream) {
  using namespace vardct_frame_internal;
  PopulationBatches list{};
  uint64_t tiles = 0, anchors = 0;
  bool pure_dct8 = true;
  for (const auto& batch : batches) {
    if (!batch.anchor_count) continue;
    const size_t family = OrderPopulationFamily(batch.coefficient_count);
    if (family == kOrderPopulationSizes.size()) return cudaErrorInvalidValue;
    pure_dct8 &= family == 0;
    anchors += batch.anchor_count;
    const uint64_t count = uint64_t{batch.anchor_count / 64 + (batch.anchor_count % 64 != 0)} *
      (batch.coefficient_count / 32);
    tiles += count;
    if (tiles > INT32_MAX || anchors > UINT32_MAX) return cudaErrorInvalidValue;
    list.batches[list.count++] = {batch.coefficient_offset, batch.anchor_count,
      batch.coefficient_count, static_cast<uint32_t>(kOrderPopulationOffsets[family]),
      static_cast<uint32_t>(count)};
  }
  if (!list.count) return cudaSuccess;
  if (!source || !populations || (pure_dct8 && (!sampled_dct8 || list.count != 1)))
    return cudaErrorInvalidValue;
  if (pure_dct8) {
    CoefficientOrderPopulationKernel<true><<<static_cast<uint32_t>(tiles), dim3(32, 8), 0, stream>>>(
      source, list, sampled_dct8, populations);
  } else {
    CoefficientOrderPopulationKernel<false><<<static_cast<uint32_t>(tiles), dim3(32, 8), 0, stream>>>(
      source, list, nullptr, populations);
  }
  return cudaGetLastError();
}

}  // namespace gjxl::cuda_internal
