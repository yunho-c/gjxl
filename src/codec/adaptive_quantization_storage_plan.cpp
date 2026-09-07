// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/adaptive_quantization_storage_plan.h"

#include <algorithm>

#include "codec/adaptive_quantization_internal.h"
#include "codec/frontend_storage_plan.h"
#include "codec/perceptual_storage_plan.h"
#include "core/frame_geometry.h"

namespace gjxl::frontend_storage_internal {
namespace {
using enum resource_budget_internal::VectorCapacityPolicy;
Status Overflow() {
  return Status::OutOfMemory(
      "CPU adaptive-quantization storage bound overflows");
}
HostStorageBound Either(HostStorageBound a, HostStorageBound b) {
  return {std::max(a.retained_bytes, b.retained_bytes),
          std::max(a.peak_bytes, b.peak_bytes)};
}
} // namespace

Status ComputeAqPolicyStoragePlan(Extent2D blocks,
                                  AdaptiveQuantizationControlMode control,
                                  size_t iterations, bool collect_profile,
                                  AqPolicyStoragePlan *out) {
  size_t count = 0;
  const bool maximum =
      control == AdaptiveQuantizationControlMode::kMaximumError;
  if (out == nullptr || blocks.empty() || !blocks.try_area(&count) ||
      (!maximum && (control != AdaptiveQuantizationControlMode::kButteraugli ||
                    iterations > 4)))
    return Status::InvalidArgument("AQ policy storage shape is invalid");
  AqPolicyStoragePlan p;
  p.evaluations =
      maximum ? adaptive_quantization_internal::kMaximumErrorUpdateCount + 1
              : iterations + 1;
  HostStorageBound scores;
  if (!scores.AddVector<double>(p.evaluations, kFreshExact) ||
      !p.output.AddVector<float>(count, kFreshExact, 2) ||
      !p.output.Add(scores) ||
      (collect_profile &&
       !p.profile.AddVector<adaptive_quantization_internal::EvaluationProfile>(
           p.evaluations, kFreshExact)))
    return Overflow();
  // Live field and the previous evaluator distance map. Butteraugli retains
  // adjusted_initial; maximum error instead retains best_feasible_field.
  if (!p.working.AddVector<float>(count, kFreshExact, 2) ||
      !p.working.AddVector<float>(count,
                                  maximum ? kReusedExact : kFreshExact) ||
      // Initial adjustment uses one atomic array. Maximum-error updates use a
      // fresh destination plus UpdateMaximumErrorQuantField's atomic array.
      !p.working.AddVector<float>(count, kFreshExact, maximum ? 2 : 1) ||
      !p.working.Add(scores) || !p.working.Add(p.profile))
    return Overflow();
  *out = p;
  return Status::Ok();
}

Status ComputeCpuAqStoragePlan(Extent2D source, const CpuAqStorageOptions &o,
                               CpuAqStoragePlan *out) {
  if (out == nullptr || o.epf_iterations > 3)
    return Status::InvalidArgument("CPU AQ storage shape is invalid");
  FrameGeometry geometry;
  Status status = FrameGeometry::Create(source, &geometry);
  if (!status.ok())
    return status;
  const auto coding = geometry.padded_frame();
  const auto blocks = geometry.block_grid().blocks;
  size_t block_count = 0, source_pixels = 0;
  if (!blocks.try_area(&block_count) || !source.try_area(&source_pixels))
    return Overflow();
  CpuAqStoragePlan p;
  status = ComputeAqPolicyStoragePlan(blocks, o.control, o.iterations,
                                      o.collect_profile, &p.policy);
  if (!status.ok())
    return status;
  OwnedFrameStoragePlan frame;
  ColorCorrelationStoragePlan cfl;
  ColorTransformStoragePlan color;
  HostStorageBound image, padded_image, quantizer, reconstruction, filters,
      reduction;
  if (!(status = ComputeOwnedFrameStoragePlan(source, &frame)).ok() ||
      !(status = ComputeImage3FStorageBound(source, &image)).ok() ||
      !(status = ComputeImage3FStorageBound(coding, &padded_image)).ok() ||
      !(status = ComputeQuantizerSelectionStorageBound(blocks, &quantizer))
           .ok() ||
      !(status = ComputeColorCorrelationStoragePlan(
            coding, ColorCorrelationStorageMode::kTransform, &cfl))
           .ok() ||
      !(status = ComputeCoefficientReconstructionStorageBound(source,
                                                              &reconstruction))
           .ok() ||
      !(status = ComputeLoopFilterStorageBound(source, o.gaborish,
                                               o.epf_iterations, &filters))
           .ok() ||
      !(status = ComputeColorTransformStoragePlan(source, source, false,
                                                  o.cpu_thread_count, &color))
           .ok() ||
      !(status = ComputeBlockReductionStorageBound(blocks, &reduction)).ok())
    return status;
  p.evaluation_output = frame.output;
  if (!p.evaluation_output.Add(image) ||
      !p.evaluation_output.AddVector<float>(block_count, kFreshExact))
    return Overflow();
  p.evaluation_working = p.evaluation_output;
  // The two other source images are cropped reconstruction and filtered Opsin.
  if (!p.evaluation_working.Add(image, 2) ||
      !p.evaluation_working.Add(padded_image) ||
      !p.evaluation_working.AddVector<int32_t>(block_count, kFreshExact) ||
      !p.evaluation_working.AddVector<float>(block_count, kFreshExact))
    return Overflow();
  for (auto scratch : {quantizer, cfl.working, reconstruction, filters,
                       color.working, reduction})
    if (!p.evaluation_working.Add(scratch))
      return Overflow();
  // EPF sigma reduction and distance/error reduction use this scratch at
  // different stages; one reduction envelope above covers either consumer.
  if (o.control == AdaptiveQuantizationControlMode::kButteraugli) {
    NativeButteraugliStoragePlan metric;
    status = ComputeNativeButteraugliStoragePlan(source, &metric);
    if (!status.ok())
      return status;
    p.reference = o.prepared_reference
                      ? Either(metric.preparation, metric.comparison)
                      : metric.one_shot;
    if (!p.evaluation_working.AddVector<float>(source_pixels, kFreshExact))
      return Overflow();
  }
  p.working = p.evaluation_working;
  // CpuAdaptiveQuantizationEvaluator retains its prior frame and RGB until
  // the new evaluation completes. Its previous block map is in policy.working.
  for (auto owner : {p.policy.working, p.reference, frame.output, image})
    if (!p.working.Add(owner))
      return Overflow();
  *out = p;
  return Status::Ok();
}

} // namespace gjxl::frontend_storage_internal
