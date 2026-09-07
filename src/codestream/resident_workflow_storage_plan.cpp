// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/resident_workflow_storage_plan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>

#include "codec/frontend_storage_plan.h"
#include "codestream/rate_control_internal.h"
#include "codestream/workflow_internal.h"
#include "gpu/metal/metal_aq_host_storage_plan.h"
#include "gpu/metal/metal_aq_profile_storage_plan.h"
#include "gpu/ops/ac_strategy_storage_plan.h"

namespace gjxl::codestream_internal {
namespace {
using namespace metal_internal;
using enum resource_budget_internal::VectorCapacityPolicy;

Status Overflow() {
  return Status::OutOfMemory("Resident workflow storage bound overflows");
}

HostStorageBound Either(HostStorageBound a, HostStorageBound b) {
  return {std::max(a.retained_bytes, b.retained_bytes),
          std::max(a.peak_bytes, b.peak_bytes)};
}

Status ProfilePlan(Extent2D source, Extent2D coding,
                   const ResidentAqProfileInputOptions &policy,
                   const AcSubmissionStoragePlan &ac,
                   ResidentWorkflowStoragePlan *p, HostStorageBound *output) {
  AqAuxiliaryProfileStoragePlan aux;
  Status status =
      ComputeAqAuxiliaryProfileStoragePlan({.source = source,
                                            .coding = coding,
                                            .resident_ac_strategy_inputs = true,
                                            .resident_initial_cfl = true,
                                            .gaborish = policy.gaborish},
                                           &aux);
  if (!status.ok())
    return status;
  ResidentAqProfileStoragePlan aq;
  status = ComputeResidentAqProfileStoragePlan(
      source, coding, policy, AqProfileFrameOutput::kCompleted, &aq);
  if (!status.ok())
    return status;
  // Six orchestration wall stages, four AC stages, three completed-output
  // stages. Reference, initial, AC, adjustment and resident policy each emit
  // one child submission. The input preparer does not emit a profile graph.
  // Max label includes registered kernel IDs and generated fallback IDs.
  p->profile_shape = {
      .wall_stages = 6 + 4 + 3,
      .submissions = 5,
      .stages = 3 + ac.stage_capacity + aq.metadata.stage_capacity,
      .dispatches = aux.reference_dispatches + aux.initial_dispatches +
                    ac.maximum_dispatches + aux.adjustment_dispatches +
                    aq.maximum_dispatches,
      .maximum_id_length =
          std::max(kMaximumKernelIdBytes - 1,
                   aq.maximum_id_length + sizeof(".dispatch_") - 1 +
                       std::numeric_limits<size_t>::digits10 + 1),
  };
  for (std::string_view id :
       {"frontend.prepare_evaluator", "frontend.initial_quantization",
        "frontend.reconfigure_aq", "frontend.quant_adjustment",
        "frontend.fixed_cfl", "resident.aq", "frontend.ac_strategy.prepare",
        "frontend.ac_strategy.wait", "frontend.ac_strategy.readback",
        "frontend.ac_strategy.merge"})
    p->profile_shape.maximum_id_length =
        std::max(p->profile_shape.maximum_id_length, id.size());
  status = gpu_profile_internal::ComputeProfileStorageBound(p->profile_shape,
                                                            output);
  if (!status.ok())
    return status;
  // The full parent bound includes every moved child, even ones not appended
  // yet. Add one complete child peak for current recording, snapshot copying,
  // callback inputs and parent/child outer-array overlap. No observed counts.
  HostStorageBound transient = ac.working;
  for (const auto child : {aux.reference.resolution, aux.initial.resolution,
                           aux.adjustment.resolution, aq.working})
    transient = Either(transient, child);
  p->diagnostics = *output;
  if (!p->diagnostics.Add(transient))
    return Overflow();
  return Status::Ok();
}
} // namespace

Status
ComputeResidentWorkflowStoragePlan(Extent2D source,
                                   const ResidentWorkflowStorageOptions &o,
                                   ResidentWorkflowStoragePlan *out) {
  const auto &e = o.encoding;
  const bool search =
      e.rate_control_mode == VarDctRateControlMode::kTargetBytes ||
      e.rate_control_mode == VarDctRateControlMode::kTargetBitsPerPixel;
  if (out == nullptr || e.effort < 1 || e.effort > 10 ||
      e.cpu_thread_count > kMaximumCpuThreadCount ||
      (e.backend != VarDctBackendPreference::kMetal &&
       e.backend != VarDctBackendPreference::kAutomatic) ||
      (e.metal_aq_mode != GpuAdaptiveQuantizationMode::kFullyResident &&
       e.metal_aq_mode != GpuAdaptiveQuantizationMode::kThroughput) ||
      (e.density_mode != VarDctDensityMode::kDefault &&
       e.density_mode != VarDctDensityMode::kHighDensity) ||
      (e.compression_mode != VarDctCompressionMode::kAutomatic &&
       e.compression_mode != VarDctCompressionMode::kMaximumCompression) ||
      (!search &&
       e.rate_control_mode != VarDctRateControlMode::kButteraugliTarget) ||
      ((search ||
        e.metal_aq_mode == GpuAdaptiveQuantizationMode::kThroughput) &&
       e.backend != VarDctBackendPreference::kMetal) ||
      (e.metal_aq_mode == GpuAdaptiveQuantizationMode::kThroughput &&
       e.density_mode == VarDctDensityMode::kHighDensity) ||
      (!search &&
       (!std::isfinite(e.butteraugli_target) || e.butteraugli_target <= 0)) ||
      (search &&
       (e.target_size_maximum_attempts == 0 ||
        e.target_size_maximum_attempts > kMaximumTargetSizeEncodeAttempts)) ||
      (o.collect_gpu_profile &&
       (search || e.backend != VarDctBackendPreference::kMetal)))
    return Status::InvalidArgument(
        "Unsupported resident workflow storage shape");
  FrameGeometry geometry;
  Status status = FrameGeometry::Create(source, &geometry);
  if (!status.ok())
    return status;
  ResidentWorkflowStoragePlan p;
  p.coding_extent = geometry.padded_frame();
  const auto coding = p.coding_extent;
  p.maximum_attempts = search ? e.target_size_maximum_attempts : 1;
  status = ValidateAqStorageGeometry(source, coding);
  if (!status.ok())
    return status;
  const auto filters = AdaptiveQuantizationOptions{}.profile.loop_filter;
  const size_t filter_images = std::min(
      size_t{2}, size_t(filters.gaborish) + filters.epf_options.iterations);
  const bool sinks = source.width >= 15 && source.height >= 15;
  const size_t iterations = AdaptiveQuantizationIterations(e);
  const bool final_score = e.collect_final_butteraugli_score || iterations == 0;
  p.score_count = iterations + size_t(final_score);
  AqHostStoragePlan host;
  status = ComputeAqHostStoragePlan({.source_extent = source,
                                     .coding_extent = coding,
                                     .resident_initial_quant = true,
                                     .resident_ac_strategy_inputs = true,
                                     .resident_quantization = true,
                                     .defer_final_transform_metadata = true,
                                     .reconfigure = true},
                                    &host);
  if (!status.ok())
    return status;
  // The checked AQ host geometry includes the tighter 32-bit coefficient bound.
  p.blocks = coding.width * coding.height / 64;
  size_t max_coefficients = 0;
  for (auto strategy : kSupportedAqStrategies) {
    const auto *info = GetAcStrategyInfo(strategy);
    if (info == nullptr)
      return Status::Internal("Resident AQ strategy disappeared");
    max_coefficients = std::max(max_coefficients, info->coefficient_count());
  }
  ResidentInputStoragePlan input;
  AqStoragePlan aq;
  ButteraugliStoragePlan butter;
  CompletedFrameStoragePlan completed;
  CompletedFrameHostStoragePlan completed_host;
  ac_strategy_search_internal::StoragePlan ac;
  ac_strategy_search_internal::HostStoragePlan ac_host;
  if (!(status = ComputeResidentInputStoragePlan(source, coding, &input))
           .ok() ||
      !(status =
            ComputeAqStoragePlan({.source_extent = source,
                                  .coding_extent = coding,
                                  .anchor_capacity_count = p.blocks,
                                  .maximum_coefficient_count = max_coefficients,
                                  .filter_scratch_image_count = filter_images,
                                  .borrowed_original_linear_rgb = true,
                                  .borrowed_coding_opsin = true,
                                  .needs_reconstructed = true,
                                  .frame_only_resident_initial_quant = true,
                                  .resident_quantization = true,
                                  .uses_butteraugli_sinks = sinks},
                                 &aq))
           .ok() ||
      !(status = ComputeButteraugliStoragePlan(
            source, sinks && filter_images == 2, &butter))
           .ok() ||
      !(status = ComputeCompletedFrameStoragePlan(source, coding, p.blocks,
                                                  &completed))
           .ok() ||
      !(status = ComputeCompletedFrameHostStoragePlan(source, coding, p.blocks,
                                                      &completed_host))
           .ok() ||
      !(status =
            ac_strategy_search_internal::ComputeStoragePlan(coding, true, &ac))
           .ok() ||
      !(status = ac_strategy_search_internal::ComputeHostStoragePlan(
            coding, true, search, &ac_host))
           .ok())
    return status;
  HostStorageBound common_device;
  for (size_t bytes : {input.capacity_bytes, aq.persistent_bytes,
                       aq.staging_bytes, butter.capacity_bytes})
    if (!common_device.Add({bytes, bytes}))
      return Overflow();
  HostStorageBound device_inventory = common_device;
  if (!device_inventory.Add(
          {completed.capacity_bytes, completed.capacity_bytes}) ||
      !device_inventory.Add({ac.device_bytes, ac.device_bytes}))
    return Overflow();
  p.device_bytes = device_inventory.peak_bytes;
  p.idle_pool_capacity = {input.capacity_bytes, aq.persistent_bytes,
                          aq.staging_bytes, butter.capacity_bytes};
  frontend_storage_internal::ColorCorrelationStoragePlan cfl;
  status = frontend_storage_internal::ComputeColorCorrelationStoragePlan(
      coding, frontend_storage_internal::ColorCorrelationStorageMode::kCopy,
      &cfl);
  if (!status.ok())
    return status;
  HostStorageBound common_frontend = host.working;
  // Prepared sharpness, selected/provisional grids; initial/strategy fields
  // and adjusted policy input. Count both CfL generations over retry
  // replacement. AC merge separately bounds its new output and export
  // temporary.
  if (!common_frontend.AddVector<uint8_t>(p.blocks, kFreshExact, 3) ||
      !common_frontend.AddVector<float>(p.blocks, kFreshExact, 3) ||
      !common_frontend.Add(cfl.working, search ? 2 : 1))
    return Overflow();
  p.frontend = common_frontend;
  if (!p.frontend.Add(ac_host.working) ||
      !p.frontend.Add(completed_host.working))
    return Overflow();
  AcSubmissionStoragePlan submission;
  const size_t nonempty =
      std::count_if(ac.stages.begin(), ac.stages.end(), [](const auto &stage) {
        return stage.candidate_count != 0;
      });
  status = ComputeAcSubmissionStoragePlan({.batches = ac.stages.size(),
                                           .nonempty_batches = nonempty,
                                           .profiling = o.collect_gpu_profile},
                                          &submission);
  if (!status.ok())
    return status;
  HostStorageBound profile_output;
  if (o.collect_gpu_profile) {
    status = ProfilePlan(source, coding,
                         {iterations, final_score, sinks, filters.gaborish,
                          filters.epf_options.iterations},
                         submission, &p, &profile_output);
    if (!status.ok())
      return status;
  } else if (!p.frontend.Add(submission.working))
    return Overflow();
  status = ComputeSerializerStoragePlan(
      source,
      {.coding = {.entropy_behavior = ResolveEntropyBehavior(e),
                  .coefficient_order_behavior =
                      ResolveCoefficientOrderBehavior(e)},
       .cpu_thread_count = e.cpu_thread_count,
       .collect_profile = o.collect_profile || o.collect_gpu_profile},
      &p.serializer);
  if (!status.ok())
    return status;
  HostStorageBound scores, timing;
  if (!scores.AddVector<double>(p.score_count, kFreshExact) ||
      (o.collect_timing && !timing.AddVector<VarDctEncodingAttemptTiming>(
                               p.maximum_attempts, kFreshExact)))
    return Overflow();
  if (search) {
    status = ComputeTargetSizeControlStorageBound(p.maximum_attempts,
                                                  &p.search_control);
    if (!status.ok())
      return status;
    if (p.maximum_attempts > 1 && (!p.retained_best.Add(p.serializer.output) ||
                                   !p.retained_best.Add(scores)))
      return Overflow();
  }
  p.output = p.serializer.output;
  if (!p.output.Add(scores) || !p.output.Add(timing) ||
      !p.output.Add(profile_output))
    return Overflow();
  HostStorageBound common = common_device;
  for (const auto part : {common_frontend, p.diagnostics, scores, timing,
                          p.search_control, p.retained_best})
    if (!common.Add(part))
      return Overflow();
  HostStorageBound ac_work = ac_host.working;
  if (!ac_work.Add({ac.device_bytes, ac.device_bytes}) ||
      (!o.collect_gpu_profile && !ac_work.Add(submission.working)))
    return Overflow();
  p.search_phase = common;
  p.completion_phase = common;
  if (!p.search_phase.Add(ac_work) ||
      !p.completion_phase.Add(
          {completed.capacity_bytes, completed.capacity_bytes}) ||
      !p.completion_phase.Add(completed_host.working) ||
      !p.completion_phase.Add(p.serializer.working) ||
      (p.maximum_attempts > 1 && !p.completion_phase.Add(ac_work)))
    return Overflow();
  // AQ/input buffers may enter idle pools before serialization, so retain
  // their charge in common. AC storage has no pool and really ends after the
  // final placement; it does not overlap that attempt's completed frame/tail.
  p.working = Either(p.search_phase, p.completion_phase);
  *out = p;
  return Status::Ok();
}

} // namespace gjxl::codestream_internal
