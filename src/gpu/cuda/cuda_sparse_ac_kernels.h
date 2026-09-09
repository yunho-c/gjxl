// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace gjxl::cuda_internal {

// Each 64 logical values has one mask and an absolute payload offset.
// Within each word payloads retain coefficient order; tile ranges may arrive
// in any order. Source and payload must not overlap. Caller clears total and
// provides count * width payload bytes and ceil(count / 64) header entries.
cudaError_t LaunchCudaSparseAc(const void* source, uint32_t count,
    unsigned width, uint64_t* masks, uint32_t* offsets, void* values,
    uint32_t* total, cudaStream_t stream);

}  // namespace gjxl::cuda_internal
