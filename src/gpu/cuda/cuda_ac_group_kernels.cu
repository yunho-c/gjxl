// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/cuda/cuda_ac_group_kernels.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace gjxl::cuda_internal {
namespace {

__global__ void PackAcGroupsKernel(
    const CudaAqAnchor* anchors, const uint64_t* destination_offsets,
    const int* source, int* destination, CudaAqExactBatch batch,
    uint32_t block_width, uint32_t block_height) {
  const uint32_t index = blockIdx.x;
  const CudaAqAnchor anchor = anchors[batch.anchor_offset + index];
  const uint32_t group_width = min(32u, block_width - (anchor.x / 32) * 32);
  const uint32_t group_height = min(32u, block_height - (anchor.y / 32) * 32);
  const uint32_t destination_stride = group_width * group_height * 64;
  const size_t source_stride =
      static_cast<size_t>(batch.anchor_count) * batch.coefficient_count;
  const size_t source_offset = batch.coefficient_offset +
      static_cast<size_t>(index) * batch.coefficient_count;
  const uint64_t destination_offset =
      destination_offsets[batch.anchor_offset + index];
  for (uint32_t coefficient = threadIdx.x; coefficient < batch.coefficient_count;
       coefficient += blockDim.x) {
    for (uint32_t channel = 0; channel < 3; ++channel) {
      destination[destination_offset + channel * destination_stride + coefficient] =
        source[source_offset + channel * source_stride + coefficient];
    }
  }
}

}  // namespace

cudaError_t LaunchCudaPackAcGroups(
    const CudaAqAnchor* anchors, const uint64_t* destination_offsets,
    const int* source, int* destination, CudaAqExactBatch batch,
    uint32_t block_width, uint32_t block_height, cudaStream_t stream) {
  if (batch.anchor_count == 0) return cudaSuccess;
  if (anchors == nullptr || destination_offsets == nullptr || source == nullptr ||
      destination == nullptr || block_width == 0 || block_height == 0 ||
      batch.coefficient_count == 0) return cudaErrorInvalidValue;
  const uint32_t threads = batch.coefficient_count < 256 ? batch.coefficient_count : 256;
  PackAcGroupsKernel<<<batch.anchor_count, threads, 0, stream>>>(
      anchors, destination_offsets, source, destination, batch, block_width, block_height);
  return cudaGetLastError();
}

}  // namespace gjxl::cuda_internal
