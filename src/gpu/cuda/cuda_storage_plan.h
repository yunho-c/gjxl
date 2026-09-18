// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/geometry.h"
#include "core/host_storage_bound.h"
#include "gpu/ops/aq_evaluation.h"

namespace gjxl::cuda_internal {

struct CudaButteraugliStoragePlan {
  Extent2D working_extent;
  Extent2D sub_extent;
  size_t capacity_bytes = 0;
  size_t cached_reference_bytes = 0;
  size_t gaussian_kernel_bytes = 0;
  size_t peak_comparison_scratch_bytes = 0;
  resource_budget_internal::HostStorageBound host;
  bool operator==(const CudaButteraugliStoragePlan &) const = default;
};

// Allocation-free geometry recipes shared with the actual backing owners.
// Capacity includes every arena alignment gap. Driver metadata/page rounding
// and small owner/control objects are outside managed backing.
[[nodiscard]] Status
ComputeCudaButteraugliStoragePlan(Extent2D source,
                                  CudaButteraugliStoragePlan *out);
[[nodiscard]] Status ComputeCudaInputStoragePlan(Extent2D source,
                                                 size_t *capacity_bytes);

struct CudaResidentStorageOptions {
  Extent2D source;
  AqEvaluationOptions evaluation;
  bool resident_frontend = false;
  bool omit_initial_search_data = false;
  bool borrowed_input = false;
};

struct CudaResidentStoragePlan {
  size_t persistent_bytes = 0;
  size_t staging_bytes = 0;
  size_t sparse_header_bytes = 0;
  CudaButteraugliStoragePlan butteraugli;
  // Includes reconfiguration overlap and optional diagnostic readbacks. Native
  // AC ownership is separate so the workflow does not count moved payload
  // twice.
  resource_budget_internal::HostStorageBound host;
  resource_budget_internal::HostStorageBound native_ac;
  bool operator==(const CudaResidentStoragePlan &) const = default;
};

[[nodiscard]] Status
ComputeCudaResidentStoragePlan(const CudaResidentStorageOptions &options,
                               CudaResidentStoragePlan *out);

struct CudaCompatibilityStoragePlan {
  size_t persistent_bytes = 0;
  size_t staging_bytes = 0;
  CudaButteraugliStoragePlan butteraugli;
  // Prepared state, replacement metadata and diagnostic staging. Completed
  // frames and the CPU compatibility frontend are accounted by their callers.
  resource_budget_internal::HostStorageBound host;
};
[[nodiscard]] Status ComputeCudaFrameOnlyStoragePlan(
    Extent2D source, const AqEvaluationOptions &evaluation, bool borrowed_input,
    CudaCompatibilityStoragePlan *out);
[[nodiscard]] Status
ComputeCudaExactStoragePlan(Extent2D source,
                            const AqEvaluationOptions &evaluation,
                            CudaCompatibilityStoragePlan *out);

} // namespace gjxl::cuda_internal
