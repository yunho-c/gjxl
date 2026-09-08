// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cuda_runtime_api.h>

#include "gpu/ops/butteraugli.h"
#include "gpu/cuda/cuda_aq_butteraugli_kernels.h"

namespace gjxl::cuda_internal {

// Encodes a validated comparison into an already active CUDA submission.
// This is intentionally internal: the public prepared operation retains its
// synchronous, non-reentrant Compare contract.
[[nodiscard]] cudaError_t EncodePreparedCudaButteraugli(
    PreparedDeviceButteraugli& prepared,
    const DeviceButteraugliComparisonDescriptor& descriptor,
    cudaStream_t stream);

// Internal resident-policy path. The descriptor's distance_map is scratch;
// its composed pixel values are not observable by this consumer.
[[nodiscard]] cudaError_t EncodePreparedCudaButteraugliAndReduce(
    PreparedDeviceButteraugli& prepared,
    const DeviceButteraugliComparisonDescriptor& descriptor,
    const CudaAqButteraugliReduction& reduction, cudaStream_t stream);

}  // namespace gjxl::cuda_internal
