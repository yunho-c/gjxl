// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/cpu_workflow_storage_plan.h"

#include "codestream/workflow_publication_storage_plan.h"

#include <cmath>

#include "codec/ac_strategy_storage_plan.h"
#include "codec/frontend_storage_plan.h"
#include "codestream/rate_control_internal.h"
#include "codestream/workflow_internal.h"
#include "core/frame_geometry.h"

namespace gjxl::codestream_internal {
namespace {
using namespace frontend_storage_internal;
using enum resource_budget_internal::VectorCapacityPolicy;
Status Overflow() {
  return Status::OutOfMemory("CPU workflow storage bound overflows");
}
} // namespace

Status ComputeCpuWorkflowStoragePlan(Extent2D source,
                                     const CpuWorkflowStorageOptions &o,
                                     CpuWorkflowStoragePlan *out) {
  const auto &e = o.encoding;
  const bool search =
      e.rate_control_mode == VarDctRateControlMode::kTargetBytes ||
      e.rate_control_mode == VarDctRateControlMode::kTargetBitsPerPixel;
  const bool maximum =
      e.rate_control_mode == VarDctRateControlMode::kMaximumError;
  if (out == nullptr || e.effort < 1 || e.effort > 10 ||
      e.cpu_thread_count > kMaximumCpuThreadCount ||
      (e.backend != VarDctBackendPreference::kCpu &&
       e.backend != VarDctBackendPreference::kAutomatic) ||
      (e.density_mode != VarDctDensityMode::kDefault &&
       e.density_mode != VarDctDensityMode::kHighDensity) ||
      (e.compression_mode != VarDctCompressionMode::kAutomatic &&
       e.compression_mode != VarDctCompressionMode::kMaximumCompression) ||
      (!search && !maximum &&
       e.rate_control_mode != VarDctRateControlMode::kButteraugliTarget) ||
      (!search && !maximum &&
       (!std::isfinite(e.butteraugli_target) || e.butteraugli_target <= 0)) ||
      (search &&
       (e.target_size_maximum_attempts == 0 ||
        e.target_size_maximum_attempts > kMaximumTargetSizeEncodeAttempts)))
    return Status::InvalidArgument("CPU workflow storage shape is invalid");
  FrameGeometry geometry;
  Status status = FrameGeometry::Create(source, &geometry);
  if (!status.ok())
    return status;
  CpuWorkflowStoragePlan p;
  p.coding_extent = geometry.padded_frame();
  p.maximum_attempts = search ? e.target_size_maximum_attempts : 1;
  size_t blocks = 0, pixels = 0;
  if (!geometry.block_grid().blocks.try_area(&blocks) ||
      !p.coding_extent.try_area(&pixels))
    return Overflow();
  const auto filters = AdaptiveQuantizationOptions{}.profile.loop_filter;
  status = ComputeCpuAqStoragePlan(
      source,
      {.control = maximum ? AdaptiveQuantizationControlMode::kMaximumError
                          : AdaptiveQuantizationControlMode::kButteraugli,
       .iterations = AdaptiveQuantizationIterations(e),
       .cpu_thread_count = e.cpu_thread_count,
       .gaborish = filters.gaborish,
       .epf_iterations = filters.epf_options.iterations,
       .prepared_reference = true,
       // The workflow CPU profile does not request the nested AQ profile.
       .collect_profile = false},
      &p.aq);
  if (!status.ok())
    return status;
  p.score_count = p.aq.policy.evaluations;
  HostStorageBound source_image, coding_image, inverse;
  ColorTransformStoragePlan color;
  ColorCorrelationStoragePlan cfl;
  InitialQuantStoragePlan initial;
  ac_strategy_internal::SearchStoragePlan ac;
  if (!(status = ComputeImage3FStorageBound(source, &source_image)).ok() ||
      !(status = ComputeImage3FStorageBound(p.coding_extent, &coding_image))
           .ok() ||
      !(status = ComputeColorTransformStoragePlan(source, p.coding_extent, true,
                                                  e.cpu_thread_count, &color))
           .ok() ||
      !(status = ComputeLoopFilterStorageBound(p.coding_extent,
                                               filters.gaborish, 0, &inverse))
           .ok() ||
      !(status = ComputeColorCorrelationStoragePlan(
            p.coding_extent, ColorCorrelationStorageMode::kTransform, &cfl))
           .ok() ||
      !(status = ComputeInitialQuantStoragePlan(p.coding_extent,
                                                e.cpu_thread_count, &initial))
           .ok())
    return status;
  if (!UseFixedDct8Strategy(e)) {
    status = ac_strategy_internal::ComputeSearchStoragePlan(p.coding_extent, &ac);
    if (!status.ok()) return status;
  }
  // Workflow Opsin + prepared preprocessed Opsin; PipelineStorage owns a
  // separate source-sized RGB destination. No caller input is counted here.
  if (!p.frontend.Add(coding_image, 2) || !p.frontend.Add(source_image) ||
      // Prepared initial/strategy fields; four compatibility block fields.
      !p.frontend.AddVector<float>(blocks, kFreshExact, 6) ||
      // Prepared and compatibility pixel masks.
      !p.frontend.AddVector<float>(pixels, kFreshExact, 2) ||
      // Prepared sharpness and the prior strategy grid during replacement.
      // Fixed-grid replacement has no AC-search working plan to cover its
      // new grid while the previous attempt's grid is still alive.
      !p.frontend.AddVector<uint8_t>(blocks, kFreshExact,
                                     UseFixedDct8Strategy(e) ? 3 : 2))
    return Overflow();
  for (auto part :
       {color.working, inverse, cfl.working, initial.working, ac.working})
    if (!p.frontend.Add(part))
      return Overflow();
  status = ComputeSerializerStoragePlan(
      source,
      {.coding = {.entropy_behavior = ResolveEntropyBehavior(e),
                  .coefficient_order_behavior =
                      ResolveCoefficientOrderBehavior(e)},
       .cpu_thread_count = e.cpu_thread_count,
       .collect_profile = o.collect_profile},
      &p.serializer);
  if (!status.ok())
    return status;
  WorkflowPublicationStoragePlan publication;
  status = ComputeWorkflowPublicationStoragePlan(
      p.serializer.output, p.score_count, p.maximum_attempts, search,
      o.collect_timing, &publication, "CPU workflow storage bound overflows");
  if (!status.ok()) return status;
  p.search_control = publication.search_control;
  p.retained_best = publication.retained_best;
  p.output = publication.output;
  p.working = p.frontend;
  // AQ already includes the current score owner. The serializer bound already
  // includes current codestream output. Do not add either output again.
  for (auto part : {p.aq.working, p.serializer.working, publication.timing,
                    p.search_control, p.retained_best})
    if (!p.working.Add(part))
      return Overflow();
  *out = p;
  return Status::Ok();
}

} // namespace gjxl::codestream_internal
