// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "gpu/cuda/cuda_aq_exact_kernels.h"
#include <cstddef>

namespace gjxl::cuda_internal {

// Internal sink for resident policy comparisons. Batches is a host array whose
// anchor ranges partition [0, anchor_count); anchors is device memory.
// Validated nonoverlapping anchor rectangles cover every source pixel exactly
// once. Maxima has anchor_count floats. Block output, maxima, error, and
// comparison inputs/scratch must be mutually disjoint. No pixel distance map is
// returned.
struct CudaAqButteraugliReduction {
  const CudaAqAnchor *anchors = nullptr;
  const CudaAqExactBatch *batches = nullptr;
  size_t batch_count = 0;
  uint32_t anchor_count = 0;
  float *block_distance = nullptr;
  uint32_t block_stride = 0;
  float *maxima = nullptr;
  unsigned int *error = nullptr;
};

// One batch's composition/L16/max pass. Main/subscale inputs are preserved;
// maxima is indexed by global anchor index. Source extents/strides and anchor
// coverage are validated by the caller. Empty batches are no-ops. Error flags
// accumulate (8: invalid anchor, 16: invalid pixel, 32: invalid L16 result).
struct CudaAqComposeReductionPlan {
  const float *main_map;
  const float *sub_map;
  const CudaAqAnchor *anchors;
  float *block_map;
  float *maxima;
  unsigned int *error;
  uint32_t width, height, main_stride, sub_stride, block_stride;
  CudaAqExactBatch batch;
};

[[nodiscard]] cudaError_t
LaunchCudaAqComposeReduction(CudaAqComposeReductionPlan plan,
                             cudaStream_t stream);

} // namespace gjxl::cuda_internal
