// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/ops/profile_storage_plan.h"

#include "gpu/ops/gpu_execution_profile_internal.h"

namespace gjxl::gpu_profile_internal {
namespace {
using resource_budget_internal::HostStorageBound;
using resource_budget_internal::StringCapacityPolicy;
using enum resource_budget_internal::VectorCapacityPolicy;

Status Overflow() {
  return Status::OutOfMemory("GPU profile storage bound overflows");
}
} // namespace

Status ComputeProfileStorageBound(const ProfileStorageShape &shape,
                                  HostStorageBound *out) {
  if (out == nullptr)
    return Status::InvalidArgument("GPU profile storage output is null");
  HostStorageBound bound;
  if (!bound.AddVector<GpuWallStageProfile>(shape.wall_stages, kGrowing) ||
      !bound.AddVector<GpuSubmissionProfile>(shape.submissions, kGrowing) ||
      !bound.AddVector<GpuStageProfile>(shape.stages, kGrowing) ||
      !bound.AddVector<GpuDispatchProfile>(shape.dispatches, kGrowing))
    return Overflow();
  for (size_t count : {shape.wall_stages, shape.submissions, shape.stages,
                       shape.stages, shape.dispatches})
    if (!bound.AddString(shape.maximum_id_length,
                         StringCapacityPolicy::kGrowing, count))
      return Overflow();
  *out = bound;
  return Status::Ok();
}

Status ComputeSubmissionProfileStoragePlan(
    const SubmissionProfileStorageOptions &options,
    SubmissionProfileStoragePlan *out) {
  if (out == nullptr || options.stages == 0)
    return Status::InvalidArgument("GPU submission profile shape is invalid");
  SubmissionProfileStoragePlan plan;
  for (bool copy : {false, true}) {
    auto &bound = copy ? plan.resolved_output : plan.recorded;
    if (!bound.AddVector<GpuStageProfile>(options.stages, kFreshExact) ||
        !bound.AddVector<GpuDispatchProfile>(options.dispatches,
                                             copy ? kFreshExact : kGrowing) ||
        !bound.AddString(options.maximum_stage_id_length,
                         StringCapacityPolicy::kFresh, options.stages) ||
        !bound.AddString(options.maximum_group_id_length,
                         StringCapacityPolicy::kFresh, options.stages) ||
        !bound.AddString(options.maximum_kernel_id_length,
                         copy ? StringCapacityPolicy::kFresh
                              : StringCapacityPolicy::kGrowing,
                         options.dispatches))
      return Overflow();
  }
  // Resolve assigns an ID into an initially empty string (which can use the
  // growth path), then moves the copy into a fresh one-submission vector.
  if (!plan.resolved_output.AddString(options.maximum_submission_id_length,
                                      StringCapacityPolicy::kGrowing) ||
      !plan.resolved_output.AddVector<GpuSubmissionProfile>(1, kFreshExact) ||
      !plan.resolution.Add(
          {plan.recorded.retained_bytes, plan.recorded.retained_bytes}) ||
      !plan.resolution.Add(plan.resolved_output))
    return Overflow();
  *out = plan;
  return Status::Ok();
}

} // namespace gjxl::gpu_profile_internal
