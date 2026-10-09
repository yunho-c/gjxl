// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "codestream/ac_tokenization_provider_internal.h"
#include "core/host_storage_bound.h"

namespace gjxl::cuda_internal {
// Bounds the provider for the encoder's geometry-selected context policies.
// Completed input coefficients are owned/accounted by the caller separately.
Status ComputeCudaTokenStoragePlan(
    Extent2D source, resource_budget_internal::HostStorageBound* out);
Status CreateCudaAcTokenizationProvider(
    GpuBackend& backend, const DeviceBuffer& coefficients, size_t offset,
    std::unique_ptr<codestream_internal::AcTokenizationProvider>* out);
// Private fixture observations; they never influence production decisions.
inline thread_local uint64_t cuda_token_provider_begin_count = 0;
inline thread_local uint64_t cuda_token_provider_retry_count = 0;
}  // namespace gjxl::cuda_internal
