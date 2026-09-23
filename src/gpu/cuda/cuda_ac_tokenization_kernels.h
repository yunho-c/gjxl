// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace gjxl::cuda_internal {
// Private experiment descriptors. Source/stride are int32 element offsets;
// unlike the Metal layout, the channel stride is explicit for CUDA families.
struct CudaTokenAnchor {
  uint32_t source, group, packed, channel_stride;
};
struct CudaTokenGroup {
  uint32_t first, anchors, output, map, width, height, tokens, reserved;
};
struct CudaTokenStrategy {
  uint32_t count, covered, width, height, order0, order1, order2, log2covered;
};
struct CudaTokenMetadata {
  uint32_t nonzeros, last, count, output;
};
struct CudaTokenPointers {
  const int32_t* coefficients;
  const CudaTokenAnchor* anchors;
  CudaTokenGroup* groups;
  const CudaTokenStrategy* strategies;
  const uint32_t* orders;
  CudaTokenMetadata* metadata;
  uint8_t* maps;
  uint32_t* values;
  uint16_t* contexts;
  uint32_t* histogram;
  uint32_t* control;
};

// Caller validates descriptors, index arithmetic and all device spans, clears
// the histogram, and supplies one metadata entry per task. All launches use the
// supplied stream. Insufficient compact output capacity skips emission entirely;
// control[end+2] then contains the required capacity. Retry only after completion.
// Control layout follows the upstream independent token oracle:
// [contexts, segments, shards, tasks, thresholds..., context_map...,
//  collect_population, capacity, total, guarded_compact, scalar_dct8].
cudaError_t LaunchCudaAcTokenization(CudaTokenPointers p, uint32_t tasks,
                                     uint32_t groups, bool scalar_dct8,
                                     cudaStream_t stream);
cudaError_t LaunchCudaAcTokenEmit(CudaTokenPointers p, uint32_t tasks,
                                  bool scalar_dct8, cudaStream_t stream);
}  // namespace gjxl::cuda_internal
