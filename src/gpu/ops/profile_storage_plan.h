// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/host_storage_bound.h"
#include "core/status.h"

namespace gjxl::gpu_profile_internal {

/// Counts are sums of each vector owner's maximum size/reserve history, not
/// just the largest logical graph observed at one instant. This includes
/// cleared owners, child graphs moved into a session, and generated ID strings.
/// Distinct simultaneous child/old-output graphs remain separate owners.
struct ProfileStorageShape {
  size_t wall_stages = 0;
  size_t submissions = 0;
  size_t stages = 0;
  size_t dispatches = 0;
  size_t maximum_id_length = 0;
};

/// Full nested GpuExecutionProfile backing, including vector/string growth.
/// This does not infer workflow stage/iteration/attempt counts from policy.
/// No managed backing allocation on success; failure preserves output.
[[nodiscard]] Status
ComputeProfileStorageBound(const ProfileStorageShape &shape,
                           resource_budget_internal::HostStorageBound *out);

struct SubmissionProfileStorageOptions {
  size_t stages = 0;
  size_t dispatches = 0; // Total across this submission's stages.
  size_t maximum_stage_id_length = 0;
  size_t maximum_group_id_length = 0;
  // Must include fallback stage_id + ".dispatch_" + decimal invocation, not
  // only the registered kernel-ID limit. Counts include all dispatched work,
  // even when dispatch timestamps are disabled in stage-profiling mode.
  size_t maximum_kernel_id_length = 0;
  size_t maximum_submission_id_length = 0;
};

struct SubmissionProfileStoragePlan {
  // Original MetalSubmission profile: exact stage reserve, growing dispatch
  // arrays/kernel labels. Submission ID is empty until a snapshot is resolved.
  resource_budget_internal::HostStorageBound recorded;
  // The completed snapshot plus its one-element GpuExecutionProfile wrapper.
  resource_budget_internal::HostStorageBound resolved_output;
  // Original retained graph coexists with the fresh deep copy and wrapper.
  resource_budget_internal::HostStorageBound resolution;
  bool operator==(const SubmissionProfileStoragePlan &) const = default;
};

/// Host profile backing of SubmitComputeProfiled/GetMetalSubmissionGpuProfile/
/// ResolveGpuSubmissionProfile. Backend encode-context/stage-input arrays,
/// caller/child profiles, session aggregation and old outputs are separate.
/// Opaque Metal counter buffers and NS::Data retain the driver exclusion.
/// Each call records/resolves a fresh graph; no arbitrary profile reuse history
/// is implied. Geometry/policy-to-count planning remains the caller's job.
[[nodiscard]] Status ComputeSubmissionProfileStoragePlan(
    const SubmissionProfileStorageOptions &options,
    SubmissionProfileStoragePlan *out);

} // namespace gjxl::gpu_profile_internal
