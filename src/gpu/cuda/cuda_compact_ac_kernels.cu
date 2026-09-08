// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/cuda/cuda_compact_ac_kernels.h"
#include <cuda_runtime.h>

namespace gjxl {
namespace {
__global__ void CompactAcKernel(const int32_t *source, uint32_t count,
                                uint32_t *bytes, uint32_t *words,
                                uint32_t *flags) {
  const uint64_t index = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const uint64_t first = index * 4;
  uint32_t byte = 0, low = 0, high = 0, overflow = 0;
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const int32_t value = first + i < count ? source[first + i] : 0;
    const uint32_t bits = static_cast<uint32_t>(value);
    byte |= (bits & 255u) << (i * 8);
    if (i < 2)
      low |= (bits & 65535u) << (i * 16);
    else
      high |= (bits & 65535u) << ((i - 2) * 16);
    overflow |= (value < -128 || value > 127) ? 1u : 0u;
    overflow |= (value < -32768 || value > 32767) ? 2u : 0u;
  }
  if (first < count) {
    bytes[index] = byte;
    reinterpret_cast<uint2 *>(words)[index] = make_uint2(low, high);
  }
  const int byte_overflow = __syncthreads_or(overflow & 1u);
  const int word_overflow = __syncthreads_or(overflow & 2u);
  if (threadIdx.x == 0 && (byte_overflow || word_overflow)) {
    atomicOr(flags, (byte_overflow ? 1u : 0u) | (word_overflow ? 2u : 0u));
  }
}
} // namespace

cudaError_t LaunchCudaCompactAc(const int32_t *source, uint32_t count,
                                uint32_t *bytes, uint32_t *words,
                                uint32_t *flags, cudaStream_t stream) {
  if (count == 0)
    return cudaSuccess;
  if (!source || !bytes || !words || !flags)
    return cudaErrorInvalidValue;
  const uint32_t work = count / 4 + (count % 4 != 0);
  const uint32_t grid = work / 256 + (work % 256 != 0);
  CompactAcKernel<<<grid, 256, 0, stream>>>(source, count, bytes, words, flags);
  return cudaGetLastError();
}
} // namespace gjxl
