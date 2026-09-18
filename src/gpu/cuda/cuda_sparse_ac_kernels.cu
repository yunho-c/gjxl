// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/cuda/cuda_sparse_ac_kernels.h"
#include <cuda_runtime.h>

namespace gjxl::cuda_internal {
namespace {
constexpr unsigned kThreads = 256;

template <typename T>
__global__ void SparseAc(const T* source, uint32_t count, uint64_t* masks,
    uint32_t* offsets, T* values, uint32_t* total) {
  constexpr unsigned warps = kThreads / 32;
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const uint64_t index = uint64_t{blockIdx.x} * kThreads + threadIdx.x;
  const T value = index < count ? source[index] : T{0};
  const uint32_t mask = __ballot_sync(0xffffffffu, value != 0);
  __shared__ uint32_t warp_masks[warps], prefix[warps], base;
  if (lane == 0) {
    warp_masks[warp] = mask;
    prefix[warp] = __popc(mask);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t sum = 0;
    #pragma unroll
    for (unsigned w = 0; w < warps; ++w) {
      const uint32_t n = prefix[w];
      prefix[w] = sum;
      sum += n;
    }
    base = sum ? atomicAdd(total, sum) : 0;
    #pragma unroll
    for (unsigned w = 0; w < warps; w += 2) {
      const uint64_t word = uint64_t{blockIdx.x} * (kThreads / 64) + w / 2;
      if (word < (uint64_t{count} + 63) / 64) {
        masks[word] = uint64_t{warp_masks[w]} | (uint64_t{warp_masks[w + 1]} << 32);
        offsets[word] = base + prefix[w];
      }
    }
  }
  __syncthreads();
  if (value != 0) {
    values[base + prefix[warp] + __popc(mask & ((uint32_t{1} << lane) - 1))] = value;
  }
}

template <typename T>
cudaError_t Dispatch(const void* source, uint32_t count, uint64_t* masks,
    uint32_t* offsets, void* values, uint32_t* total, cudaStream_t stream) {
  const uint32_t grid = count / kThreads + (count % kThreads != 0);
  SparseAc<T><<<grid, kThreads, 0, stream>>>(static_cast<const T*>(source),
      count, masks, offsets, static_cast<T*>(values), total);
  return cudaGetLastError();
}
}  // namespace

cudaError_t LaunchCudaSparseAc(const void* source, uint32_t count,
    unsigned width, uint64_t* masks, uint32_t* offsets, void* values,
    uint32_t* total, cudaStream_t stream) {
  if (width != 1 && width != 2 && width != 4) return cudaErrorInvalidValue;
  if (!count) return cudaSuccess;
  if (!source || !masks || !offsets || !values || !total) return cudaErrorInvalidValue;
  if (width == 1) return Dispatch<int8_t>(source, count, masks, offsets, values, total, stream);
  if (width == 2) return Dispatch<int16_t>(source, count, masks, offsets, values, total, stream);
  return Dispatch<int32_t>(source, count, masks, offsets, values, total, stream);
}
}  // namespace gjxl::cuda_internal
