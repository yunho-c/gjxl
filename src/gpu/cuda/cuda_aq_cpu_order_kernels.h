// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

#include "gpu/cuda/cuda_aq_exact_kernels.h"

namespace gjxl::cuda_internal {
// Internal kernels: the caller validates buffers, strides, disjoint outputs,
// and anchor rectangles. All arithmetic executes on the GPU. The resident
// entries retain their separate, optimized arithmetic.
//
// Basis comes from the scalar CPU implementation, owned by the prepared
// evaluator's admitted device arena. No global device allocation is retained.
inline constexpr size_t kCudaCpuOrderBasisCount = 8 * 8 + 16 * 16 + 32 * 32;
inline constexpr size_t kCudaCpuOrderBasisWords =
    kCudaCpuOrderBasisCount * sizeof(double) / sizeof(int32_t);

[[nodiscard]] cudaError_t LaunchCudaCpuOrderInverseDct(
    const float* input, float* output, size_t count, unsigned width,
    unsigned height, const double* basis, cudaStream_t stream);
[[nodiscard]] cudaError_t LaunchCudaCpuOrderGaborish(
    std::array<const float*, 3> input, std::array<float*, 3> output,
    unsigned* error, CudaAqGaborishParams params, cudaStream_t stream);
[[nodiscard]] cudaError_t LaunchCudaCpuOrderOpsinToLinear(
    std::array<const float*, 3> input, std::array<float*, 3> output,
    unsigned* error, CudaAqColorParams params, float bias_cuberoot,
    cudaStream_t stream);
[[nodiscard]] cudaError_t LaunchCudaCpuOrderReduceButteraugli(
    const float* distance_map, uint32_t distance_stride,
    const CudaAqAnchor* anchors, float* block_distance, uint32_t block_stride,
    unsigned* error, uint32_t source_width, uint32_t source_height,
    CudaAqExactBatch batch, cudaStream_t stream);
}  // namespace gjxl::cuda_internal
