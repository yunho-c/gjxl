// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "gpu/cuda/cuda_aq_exact_kernels.h"

namespace gjxl::cuda_internal {

// Packs one transform batch directly into active group/channel rows at both
// signed 8- and 16-bit widths. Caller clears flags before the first batch;
// bits 0/1 report signed 8/16-bit overflow. All batches accumulate into flags.
// Low bits are written even on overflow; only a fitting width may be consumed.
// Source, byte/word destinations, metadata and flags must not overlap.
// Destinations require 4/8-byte alignment and capacity N/2N bytes respectively,
// where N is the total active coefficient count. Coefficient counts and all
// destination offsets are multiples of four. Metadata/layout is prevalidated
// as for LaunchCudaPackAcGroups. Source and metadata remain unchanged, allowing
// ordinary dense group packing after int16 overflow. No padded tails are written.
cudaError_t LaunchCudaPackCompactAcGroups(
    const CudaAqAnchor* anchors, const uint64_t* destination_offsets,
    const int32_t* source, uint32_t* bytes, uint32_t* words, uint32_t* flags,
    CudaAqExactBatch batch, uint32_t block_width, uint32_t block_height,
    cudaStream_t stream);

}  // namespace gjxl::cuda_internal
