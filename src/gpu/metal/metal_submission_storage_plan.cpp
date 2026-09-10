// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_submission_storage_plan.h"

#include <algorithm>
#include <limits>
#include <string_view>

#include "gpu/metal/metal_aq_evaluation_internal.h"

namespace gjxl::metal_internal {
namespace {
using enum resource_budget_internal::VectorCapacityPolicy;

Status Overflow() {
  return Status::OutOfMemory("Metal submission storage bound overflows");
}
} // namespace

Status ComputeAcSubmissionStoragePlan(const AcSubmissionStorageOptions &o,
                                      AcSubmissionStoragePlan *out) {
  if (out == nullptr || o.nonempty_batches > o.batches)
    return Status::InvalidArgument("Metal AC submission shape is invalid");
  if (o.batches > resource_budget_internal::ManagedAllocator<
                      MetalBackend::ValidatedAcStrategyBatch>{}.max_size())
    return Status::InvalidArgument("Too many AC-strategy candidate batches");
  AcSubmissionStoragePlan p;
  p.batch_capacity = o.batches;
  if (!p.input.AddVector<MetalBackend::ValidatedAcStrategyBatch>(o.batches,
                                                                 kFreshExact))
    return Overflow();
  if (o.profiling && o.nonempty_batches != 0) {
    p.stage_capacity = o.nonempty_batches;
    // Gather + forward transform + residual + inverse transform + final cost.
    // Each fused pair can reduce this count by one, never increase it.
    if (o.nonempty_batches > std::numeric_limits<size_t>::max() / 5)
      return Overflow();
    p.maximum_dispatches = 5 * o.nonempty_batches;
    if (!p.input.AddVector<MetalBackend::AcStrategyProfileContext>(
            p.stage_capacity, kFreshExact) ||
        !p.input.AddVector<MetalProfiledComputeStage>(p.stage_capacity,
                                                      kFreshExact))
      return Overflow();
    constexpr size_t maximum_stage_id = [] {
      size_t length = 0;
      // Include the unsupported fallback, even though validation rejects it.
      for (size_t i = 0; i <= static_cast<size_t>(AcStrategyType::kCount); ++i)
        length = std::max(length,
                          std::string_view(AcStrategyProfileStageId(
                                               static_cast<AcStrategyType>(i)))
                              .size());
      return length;
    }();
    constexpr size_t maximum_kernel_id =
        std::max(kMaximumKernelIdBytes - 1,
                 maximum_stage_id + sizeof(".dispatch_") - 1 +
                     std::numeric_limits<size_t>::digits10 + 1);
    const Status status =
        gpu_profile_internal::ComputeSubmissionProfileStoragePlan(
            {p.stage_capacity, p.maximum_dispatches, maximum_stage_id,
             sizeof(kAcStrategyProfileGroupId) - 1, maximum_kernel_id,
             o.maximum_submission_id_length},
            &p.profile);
    if (!status.ok())
      return status;
  }
  p.working = p.input;
  if (!p.working.Add(p.profile.recorded))
    return Overflow();
  p.working.peak_bytes =
      std::max(p.working.peak_bytes, p.profile.resolution.peak_bytes);
  p.working.retained_bytes =
      std::max(p.working.retained_bytes, p.profile.resolution.retained_bytes);
  *out = p;
  return Status::Ok();
}

Status ComputeResidentAqProfileInputStoragePlan(
    const ResidentAqProfileInputOptions &o,
    ResidentAqProfileInputStoragePlan *out) {
  if (out == nullptr || o.iterations > 4 || o.epf_iterations > 3 ||
      (o.iterations == 0 && !o.evaluate_final_field))
    return Status::InvalidArgument("Metal resident profile policy is invalid");
  ResidentAqProfileInputStoragePlan p;
  p.score_count = o.iterations + static_cast<size_t>(o.evaluate_final_field);
  const size_t stages_per_iteration =
      12 + 4 * kSupportedAqStrategies.size() +
      static_cast<size_t>(o.butteraugli_sinks) * 4 +
      (kMetalButteraugliPsychoProfiles.size() - 1) *
        (1 + static_cast<size_t>(o.butteraugli_sinks)) +
      static_cast<size_t>(o.gaborish) + o.epf_iterations;
  p.stage_capacity = p.score_count * stages_per_iteration +
                     static_cast<size_t>(!o.evaluate_final_field) *
                         (1 + kSupportedAqStrategies.size()) +
                     1;
  if (!p.input
           .AddVector<MetalPreparedAqEvaluation::ResidentProfileStageContext>(
               p.stage_capacity, kFreshExact) ||
      !p.input.AddVector<MetalProfiledComputeStage>(p.stage_capacity,
                                                    kFreshExact))
    return Overflow();
  *out = p;
  return Status::Ok();
}

} // namespace gjxl::metal_internal
