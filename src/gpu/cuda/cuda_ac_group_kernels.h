// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

#include "gpu/cuda/cuda_aq_exact_kernels.h"

namespace gjxl::cuda_internal {

/// Copies one batch into packed active group/channel rows. Destination offsets
/// are indexed like anchors; channel strides follow the actual edge-group size.
/// Source and destination must not alias. No padded tails are written.
cudaError_t LaunchCudaPackAcGroups(
    const CudaAqAnchor* anchors, const uint64_t* destination_offsets,
    const int* source, int* destination, CudaAqExactBatch batch,
    uint32_t block_width, uint32_t block_height, cudaStream_t stream);

}  // namespace gjxl::cuda_internal
