// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/cuda/cuda_direct_ac_pack_kernels.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace gjxl::cuda_internal {
namespace {
__global__ void PackCompactAcGroupsKernel(
    const CudaAqAnchor* anchors, const uint64_t* destination_offsets,
    const int32_t* source, uint32_t* bytes, uint32_t* words, uint32_t* flags,
    CudaAqExactBatch batch, uint32_t block_width, uint32_t block_height) {
  const uint32_t index = blockIdx.x;
  const CudaAqAnchor anchor = anchors[batch.anchor_offset + index];
  const uint32_t group_width = min(32u, block_width - (anchor.x / 32) * 32);
  const uint32_t group_height = min(32u, block_height - (anchor.y / 32) * 32);
  const uint32_t destination_stride = group_width * group_height * 64;
  const size_t source_stride = size_t{batch.anchor_count} * batch.coefficient_count;
  const size_t source_offset = batch.coefficient_offset + size_t{index} * batch.coefficient_count;
  const uint64_t destination_offset = destination_offsets[batch.anchor_offset + index];
  uint32_t overflow = 0;
  for (uint32_t quad = threadIdx.x; quad < batch.coefficient_count / 4;
       quad += blockDim.x) {
    for (uint32_t channel = 0; channel < 3; ++channel) {
      uint32_t byte = 0, low = 0, high = 0;
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const int32_t value = source[source_offset + channel * source_stride + quad * 4 + i];
        const uint32_t bits = static_cast<uint32_t>(value);
        byte |= (bits & 255u) << (i * 8);
        if (i < 2) low |= (bits & 65535u) << (i * 16);
        else high |= (bits & 65535u) << ((i - 2) * 16);
        overflow |= (value < -128 || value > 127) ? 1u : 0u;
        overflow |= (value < -32768 || value > 32767) ? 2u : 0u;
      }
      const uint64_t output_quad = (destination_offset + channel * destination_stride) / 4 + quad;
      bytes[output_quad] = byte;
      reinterpret_cast<uint2*>(words)[output_quad] = make_uint2(low, high);
    }
  }
  const int byte_overflow = __syncthreads_or(overflow & 1u);
  const int word_overflow = __syncthreads_or(overflow & 2u);
  if (threadIdx.x == 0 && (byte_overflow || word_overflow))
    atomicOr(flags, (byte_overflow ? 1u : 0u) | (word_overflow ? 2u : 0u));
}
}  // namespace

cudaError_t LaunchCudaPackCompactAcGroups(
    const CudaAqAnchor* anchors, const uint64_t* destination_offsets,
    const int32_t* source, uint32_t* bytes, uint32_t* words, uint32_t* flags,
    CudaAqExactBatch batch, uint32_t block_width, uint32_t block_height,
    cudaStream_t stream) {
  if (batch.anchor_count == 0) return cudaSuccess;
  if (!anchors || !destination_offsets || !source || !bytes || !words || !flags ||
      block_width == 0 || block_height == 0 || batch.coefficient_count == 0 ||
      batch.coefficient_count % 4 != 0 ||
      reinterpret_cast<uintptr_t>(bytes) % 4 != 0 ||
      reinterpret_cast<uintptr_t>(words) % 8 != 0)
    return cudaErrorInvalidValue;
  const uint32_t threads = min(256u, max(32u, batch.coefficient_count / 4));
  PackCompactAcGroupsKernel<<<batch.anchor_count, threads, 0, stream>>>(
      anchors, destination_offsets, source, bytes, words, flags, batch,
      block_width, block_height);
  return cudaGetLastError();
}
}  // namespace gjxl::cuda_internal
