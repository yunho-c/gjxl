// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "gpu/cuda/cuda_aq_exact_kernels.h"

namespace gjxl::cuda_internal {

// Source is the final quantized batch/channel/anchor layout. Populations must
// be cleared, disjoint from source, and have kOrderPopulationCount uint32s.
// For pure DCT8, sampled flags use the batch's row-major anchor indexes; the
// host generated them in AC-group-first PRNG order. Other frames need no flags.
cudaError_t LaunchCudaCoefficientOrderPopulation(
  const int* source, const std::array<CudaAqExactBatch, 7>& batches,
  const uint8_t* sampled_dct8, uint32_t* populations, cudaStream_t stream);

}  // namespace gjxl::cuda_internal
