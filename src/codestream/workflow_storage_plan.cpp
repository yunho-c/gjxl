// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/workflow_storage_plan.h"

#include <algorithm>
#include <limits>

#include "codec/frontend_storage_plan.h"
#include "codestream/batch_workflow.h"
#include "codestream/compatibility_workflow_storage_plan.h"
#include "codestream/encoding_result_internal.h"
#include "codestream/resident_workflow_storage_plan.h"

namespace gjxl::codestream_internal {
namespace {
using enum resource_budget_internal::VectorCapacityPolicy;
Status Overflow() {
  return Status::OutOfMemory("Workflow admission storage bound overflows");
}
template <class Plan> void CopyBase(const Plan &base, WorkflowStoragePlan *p) {
  p->coding_extent = base.coding_extent;
  p->maximum_attempts = base.maximum_attempts;
  p->maximum_codestream_bytes = base.serializer.maximum_output_bytes;
  p->backend_working = base.working;
  p->output = base.output;
  if constexpr (requires { base.idle_pool_capacity; })
    p->idle_pool_capacity = base.idle_pool_capacity;
}
} // namespace

Status ComputeWorkflowStoragePlan(Extent2D source,
                                  const WorkflowStorageOptions &o,
                                  WorkflowStoragePlan *out) {
  if (out == nullptr ||
      (o.adapter != WorkflowStorageAdapter::kBorrowedLinearRgb &&
       o.adapter != WorkflowStorageAdapter::kPackedSrgbC) ||
      (o.adapter == WorkflowStorageAdapter::kPackedSrgbC &&
       (o.collect_timing || o.collect_profile || o.collect_gpu_profile)))
    return Status::InvalidArgument("Workflow storage adapter is invalid");
  const auto &e = o.encoding;
  const bool search =
      e.rate_control_mode == VarDctRateControlMode::kTargetBytes ||
      e.rate_control_mode == VarDctRateControlMode::kTargetBitsPerPixel;
  WorkflowStoragePlan p;
  p.adapter = o.adapter;
  p.collect_timing = o.collect_timing;
  CpuWorkflowStorageOptions plain{e, o.collect_timing, o.collect_profile};
  Status status;
  switch (o.route) {
  case WorkflowStorageRoute::kCpu: {
    if (o.collect_gpu_profile)
      return Status::InvalidArgument("CPU route cannot produce GPU profiles");
    CpuWorkflowStoragePlan cpu;
    status = ComputeCpuWorkflowStoragePlan(source, plain, &cpu);
    if (!status.ok())
      return status;
    CopyBase(cpu, &p);
    break;
  }
  case WorkflowStorageRoute::kMetal: {
    if ((e.backend != VarDctBackendPreference::kMetal &&
         e.backend != VarDctBackendPreference::kAutomatic) ||
        (e.backend == VarDctBackendPreference::kAutomatic &&
         (search ||
          e.rate_control_mode == VarDctRateControlMode::kMaximumError)))
      return Status::InvalidArgument(
          "Metal route conflicts with workflow policy");
    const bool resident =
        (e.metal_aq_mode == GpuAdaptiveQuantizationMode::kFullyResident ||
         e.metal_aq_mode == GpuAdaptiveQuantizationMode::kThroughput) &&
        e.rate_control_mode != VarDctRateControlMode::kMaximumError;
    if (resident) {
      ResidentWorkflowStoragePlan base;
      status = ComputeResidentWorkflowStoragePlan(
          source,
          {e, o.collect_timing, o.collect_profile, o.collect_gpu_profile},
          &base);
      if (!status.ok())
        return status;
      CopyBase(base, &p);
    } else {
      if (o.collect_gpu_profile)
        return Status::InvalidArgument(
            "Compatibility route cannot produce GPU profiles");
      plain.encoding.backend = VarDctBackendPreference::kMetal;
      MetalCompatibilityWorkflowStoragePlan base;
      status =
          ComputeMetalCompatibilityWorkflowStoragePlan(source, plain, &base);
      if (!status.ok())
        return status;
      CopyBase(base, &p);
    }
    break;
  }
  case WorkflowStorageRoute::kAutomaticExactSearch: {
    if (o.collect_gpu_profile)
      return Status::InvalidArgument(
          "Automatic searches cannot produce GPU profiles");
    AutomaticExactSearchStoragePlan base;
    status = ComputeAutomaticExactSearchStoragePlan(source, plain, &base);
    if (!status.ok())
      return status;
    CopyBase(base.cpu, &p);
    p.backend_working = base.working;
    p.output = base.output;
    p.idle_pool_capacity = base.metal.idle_pool_capacity;
    break;
  }
  default:
    return Status::InvalidArgument("Workflow storage route is invalid");
  }
  if (o.adapter == WorkflowStorageAdapter::kPackedSrgbC) {
    status =
        frontend_storage_internal::ComputeImage3FStorageBound(source, &p.input);
    if (!status.ok())
      return status;
    if (!p.publication.AddVector<uint8_t>(p.maximum_codestream_bytes,
                                          kFreshExact))
      return Overflow();
    // Only the copied C byte array is published. Internal scores and any
    // candidate backing are destroyed before the outer adapter returns.
    p.output = p.publication;
  }
  p.working = p.backend_working;
  if (!p.working.Add(p.input) || !p.working.Add(p.publication))
    return Overflow();
  *out = p;
  return Status::Ok();
}

Status
BatchWorkflowStorageAccumulator::AddRequest(const WorkflowStoragePlan *plan) {
  if (requests_ == std::numeric_limits<size_t>::max())
    return Overflow();
  if (plan != nullptr &&
      (plan->working.peak_bytes == 0 || !plan->collect_timing ||
       plan->adapter != WorkflowStorageAdapter::kBorrowedLinearRgb ||
       plan->output.retained_bytes > plan->working.peak_bytes))
    return Status::InvalidArgument(
        "Batch requires complete timed workflow plans");
  auto retained = retained_;
  if (plan != nullptr &&
      !retained.Add({plan->output.retained_bytes, plan->output.retained_bytes}))
    return Overflow();
  ++requests_;
  if (plan != nullptr) {
    ++encodable_;
    retained_ = retained;
    maximum_working_ = std::max(maximum_working_, plan->working.peak_bytes);
    for (size_t i = 0; i < idle_.size(); ++i)
      idle_[i] = std::max(idle_[i], plan->idle_pool_capacity[i]);
  }
  return Status::Ok();
}

Status
BatchWorkflowStorageAccumulator::Finish(size_t max_in_flight,
                                        size_t managed_limit,
                                        BatchWorkflowStoragePlan *out) const {
  if (out == nullptr || max_in_flight == 0)
    return Status::InvalidArgument(
        "Batch storage output or concurrency is invalid");
  const size_t limit =
      managed_limit == 0 ? std::numeric_limits<size_t>::max() : managed_limit;
  BatchWorkflowStoragePlan p;
  p.request_count = requests_;
  p.encodable_count = encodable_;
  p.work_slot_bytes = maximum_working_;
  p.retained_results = retained_;
  // These are the current driver's three request-sized owners, not an
  // invented per-image constant. No pre-admission plan vector is required.
  if (!p.result_metadata.AddVector<VarDctBatchEncodingResult>(requests_,
                                                              kFreshExact) ||
      !p.result_metadata.AddVector<OwnedEncodingResult>(requests_,
                                                        kFreshExact) ||
      !p.result_metadata.AddVector<
          std::array<resource_budget_internal::ResourceAllocation, 3>>(
          requests_, kFreshExact))
    return Overflow();
  HostStorageBound base = p.result_metadata;
  if (!base.Add(p.retained_results))
    return Overflow();
  if (base.peak_bytes > limit || maximum_working_ > limit - base.peak_bytes)
    return Status::OutOfMemory(
        "Retained batch results and largest work plan exceed managed limit");
  p.minimum_required_bytes = base.peak_bytes + maximum_working_;
  for (size_t bytes : idle_)
    if (!p.idle_pools.Add({bytes, bytes}))
      return Overflow();
  // Keep caches when the hard limit permits it; otherwise require trimming
  // before a completed worker reuses its slot. Never discard retained results.
  if (p.idle_pools.peak_bytes > limit - p.minimum_required_bytes) {
    p.trim_after_each_image = true;
    p.idle_pools = {};
  }
  if (!base.Add(p.idle_pools))
    return Overflow();
  if (encodable_ != 0) {
    p.in_flight = std::min({max_in_flight, encodable_,
                            (limit - base.peak_bytes) / maximum_working_});
    if (!base.Add({maximum_working_, maximum_working_}, p.in_flight))
      return Overflow();
  }
  p.working = base;
  *out = p;
  return Status::Ok();
}

} // namespace gjxl::codestream_internal
