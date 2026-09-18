// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "codestream/cpu_workflow_storage_plan.h"
#include "gpu/cuda/cuda_storage_plan.h"

namespace gjxl::codestream_internal {
struct CudaWorkflowStoragePlan {
  Extent2D coding_extent;
  size_t maximum_attempts = 0;
  size_t score_count = 0;
  size_t cuda_idle_capacity = 0;
  cuda_internal::CudaResidentStoragePlan resident;
  cuda_internal::CudaCompatibilityStoragePlan compatibility;
  HostStorageBound frontend;
  HostStorageBound device;
  HostStorageBound completed;
  SerializerStoragePlan serializer;
  HostStorageBound output;
  HostStorageBound working;
};

// Complete CUDA workflow from borrowed RGB through publication. Compatibility
// routes compose the shared CPU-owner plan with CUDA arenas and host staging.
// One fixed-geometry reservation spans all target-size attempts. Moved native
// AC is counted once; device cache backing remains charged during
// serialization. Prior jobs' idle buffers are separate and reclaimable by
// admission. Planning allocates no backing; driver internals and small controls
// are excluded.
[[nodiscard]] Status
ComputeCudaWorkflowStoragePlan(Extent2D source,
                               const CpuWorkflowStorageOptions &options,
                               CudaWorkflowStoragePlan *out);
} // namespace gjxl::codestream_internal
