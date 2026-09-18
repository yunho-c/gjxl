// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <array>

#include "gpu/cuda/cuda_profile_limits.h"
#include "gpu/ops/profile_storage_plan.h"

namespace gjxl::cuda_internal {

// Both diagnostic modes record launch metadata. Dispatch mode additionally
// retains two event handles per launch. Reserve its larger bound for either
// mode; the opaque CUDA driver allocations follow the driver exclusion.
inline Status ComputeCudaSubmissionProfileStoragePlan(
    const gpu_profile_internal::SubmissionProfileStorageOptions& options,
    gpu_profile_internal::SubmissionProfileStoragePlan* out) {
  if (out == nullptr) return Status::InvalidArgument("CUDA profile plan output is null");
  gpu_profile_internal::SubmissionProfileStoragePlan plan;
  Status status = gpu_profile_internal::ComputeSubmissionProfileStoragePlan(options, &plan);
  if (!status.ok()) return status;
  resource_budget_internal::HostStorageBound events;
  if (!events.AddVector<std::array<void*, 2>>(options.dispatches,
        resource_budget_internal::VectorCapacityPolicy::kGrowing) ||
      !plan.recorded.Add(events) || !plan.resolution.Add(events))
    return Status::OutOfMemory("CUDA profile event storage overflows");
  *out = plan;
  return Status::Ok();
}
} // namespace gjxl::cuda_internal
