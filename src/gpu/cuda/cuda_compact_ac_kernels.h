// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

namespace gjxl {
// Writes both representations; flags bit 0/1 report signed 8/16-bit overflow.
// Caller clears flags. Four-value padding and 4/8-byte alignment are required
// for byte/word destinations respectively; input and outputs must not overlap.
cudaError_t LaunchCudaCompactAc(const int32_t *source, uint32_t count,
                                uint32_t *bytes, uint32_t *words,
                                uint32_t *flags, cudaStream_t stream);
} // namespace gjxl
