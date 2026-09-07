// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/compatibility_workflow_storage_plan.h"

#include <algorithm>

#include "codec/frontend_storage_plan.h"
#include "gpu/metal/metal_aq_host_storage_plan.h"
#include "gpu/metal/metal_aq_profile_storage_plan.h"
#include "gpu/ops/ac_strategy_storage_plan.h"

namespace gjxl::codestream_internal {
namespace {
using namespace frontend_storage_internal;
using namespace metal_internal;
using enum resource_budget_internal::VectorCapacityPolicy;

Status Overflow() {
  return Status::OutOfMemory("Metal compatibility workflow bound overflows");
}
bool IsSearch(VarDctRateControlMode mode) {
  return mode == VarDctRateControlMode::kTargetBytes ||
         mode == VarDctRateControlMode::kTargetBitsPerPixel;
}
} // namespace

Status ComputeMetalCompatibilityWorkflowStoragePlan(
    Extent2D source, const CpuWorkflowStorageOptions &o,
    MetalCompatibilityWorkflowStoragePlan *out) {
  const auto &e = o.encoding;
  const bool exact =
      e.metal_aq_mode == GpuAdaptiveQuantizationMode::kExactCoefficients;
  const bool frame_only =
      e.metal_aq_mode == GpuAdaptiveQuantizationMode::kMaximumThroughput;
  const bool maximum =
      e.rate_control_mode == VarDctRateControlMode::kMaximumError;
  const bool resident = !exact && !frame_only;
  if (out == nullptr || e.backend != VarDctBackendPreference::kMetal ||
      (resident &&
       ((e.metal_aq_mode != GpuAdaptiveQuantizationMode::kFullyResident &&
         e.metal_aq_mode != GpuAdaptiveQuantizationMode::kThroughput) ||
        !maximum)) ||
      (frame_only && maximum) ||
      ((frame_only || maximum) &&
       e.density_mode == VarDctDensityMode::kHighDensity))
    return Status::InvalidArgument("Unsupported Metal compatibility shape");

  // Shared conservative host preparation, serializer and search recipes.
  // The native CPU evaluator is NOT part of this Metal bound.
  auto cpu_options = o;
  cpu_options.encoding.backend = VarDctBackendPreference::kCpu;
  CpuWorkflowStoragePlan cpu;
  Status status = ComputeCpuWorkflowStoragePlan(source, cpu_options, &cpu);
  if (!status.ok())
    return status;
  MetalCompatibilityWorkflowStoragePlan p;
  p.coding_extent = cpu.coding_extent;
  p.maximum_attempts = cpu.maximum_attempts;
  p.score_count = frame_only ? 0 : cpu.score_count;
  p.frontend = cpu.frontend;
  p.serializer = cpu.serializer;
  p.search_control = cpu.search_control;
  const auto coding = p.coding_extent;
  const Extent2D blocks{coding.width / 8, coding.height / 8};
  AqHostStoragePlan host;
  status = ComputeAqHostStoragePlan(
      {.source_extent = source,
       .coding_extent = coding,
       .frame_only = frame_only,
       .resident_initial_quant = !exact,
       .resident_ac_strategy_inputs = resident,
       .resident_quantization = resident,
       .defer_final_transform_metadata = resident,
       .metric = maximum ? AqEvaluationMetric::kMaximumError
                         : AqEvaluationMetric::kButteraugli,
       .reconfigure = !frame_only,
       .exact_coefficients = exact,
       .reconstruct_exact_coefficients = exact,
       .initial_pixel_mask_readback = frame_only},
      &host);
  if (!status.ok())
    return status;
  // The host planner's checked 32-bit coefficient bound proves these products.
  const size_t block_count = blocks.width * blocks.height;
  const size_t pixels = coding.width * coding.height;
  const auto filters = AdaptiveQuantizationOptions{}.profile.loop_filter;
  const size_t filter_images =
      frame_only ? 0
                 : std::min(size_t{2}, size_t(filters.gaborish) +
                                           filters.epf_options.iterations);
  const bool sinks =
      !frame_only && !maximum && source.width >= 15 && source.height >= 15;
  size_t max_coefficients = 0;
  for (auto strategy : kSupportedAqStrategies) {
    const auto *info = GetAcStrategyInfo(strategy);
    if (info == nullptr)
      return Status::Internal("Metal AQ strategy disappeared");
    max_coefficients = std::max(max_coefficients, info->coefficient_count());
  }
  InitialQuantSortPlan sort;
  if (frame_only &&
      !(status = ComputeInitialQuantSortPlan(block_count, &sort)).ok())
    return status;
  AqStoragePlan device;
  status = ComputeAqStoragePlan(
      {.source_extent = source,
       .coding_extent = coding,
       .anchor_capacity_count = block_count,
       .maximum_coefficient_count = max_coefficients,
       .initial_quant_sort_count = sort.count,
       .filter_scratch_image_count = filter_images,
       .frame_only = frame_only,
       .borrowed_original_linear_rgb = resident,
       .borrowed_coding_opsin = resident,
       .needs_reconstructed = !frame_only || filters.gaborish,
       .frame_only_resident_initial_quant = !exact,
       .frame_only_resident_quantizer = frame_only,
       .resident_quantization = resident,
       .uses_butteraugli_sinks = sinks,
       .metric = maximum ? AqEvaluationMetric::kMaximumError
                         : AqEvaluationMetric::kButteraugli},
      &device);
  if (!status.ok())
    return status;
  p.evaluator = host.working;
  for (size_t bytes : {device.persistent_bytes, device.staging_bytes})
    if (!p.evaluator.Add({bytes, bytes}))
      return Overflow();
  if (resident) {
    ResidentInputStoragePlan input;
    status = ComputeResidentInputStoragePlan(source, coding, &input);
    if (!status.ok())
      return status;
    // Provisional strategy grid and adjusted policy input are additional to
    // the common prepared fields. Invariant CfL output is copied on the host.
    ColorCorrelationStoragePlan cfl;
    status = ComputeColorCorrelationStoragePlan(
        coding, ColorCorrelationStorageMode::kCopy, &cfl);
    if (!status.ok())
      return status;
    if (!p.frontend.Add({input.capacity_bytes, input.capacity_bytes}) ||
        !p.frontend.AddVector<uint8_t>(block_count, kFreshExact) ||
        !p.frontend.AddVector<float>(block_count, kFreshExact) ||
        !p.frontend.Add(cfl.working))
      return Overflow();
  }
  if (!frame_only && !maximum) {
    ButteraugliStoragePlan butter;
    status = ComputeButteraugliStoragePlan(source, sinks && filter_images == 2,
                                           &butter);
    if (!status.ok())
      return status;
    if (!p.evaluator.Add({butter.capacity_bytes, butter.capacity_bytes}))
      return Overflow();
  }
  OwnedFrameStoragePlan frame;
  status = ComputeOwnedFrameStoragePlan(source, &frame);
  if (!status.ok())
    return status;
  if (!p.evaluator.Add(frame.output))
    return Overflow();
  if (frame_only) {
    // RunGpuFrameOnlyQuantizationPipeline's fresh initial/strategy/final
    // fields, pixel mask, DCT8 grid and sharpness coexist with compatibility
    // destinations. Resident quantizer selection uses the planned device sort.
    if (!p.evaluator.AddVector<float>(block_count, kFreshExact, 3) ||
        !p.evaluator.AddVector<float>(pixels, kFreshExact) ||
        !p.evaluator.AddVector<uint8_t>(block_count, kFreshExact, 2))
      return Overflow();
  } else {
    // Shared policy retains its previous map; each GPU evaluation separately
    // constructs the current map. Only the final owned frame is retained.
    if (!p.evaluator.Add(cpu.aq.policy.working) ||
        !p.evaluator.AddVector<float>(block_count, kFreshExact))
      return Overflow();
    if (exact) {
      PreparedForwardStoragePlan forward;
      ColorCorrelationStoragePlan cfl;
      HostStorageBound quantizer, reduction;
      if (!(status = ComputePreparedForwardStoragePlan(
                coding, e.cpu_thread_count, &forward))
               .ok() ||
          !(status = ComputeColorCorrelationStoragePlan(
                coding, ColorCorrelationStorageMode::kTransform, &cfl))
               .ok() ||
          !(status = ComputeQuantizerSelectionStorageBound(blocks, &quantizer))
               .ok() ||
          !(status = ComputeBlockReductionStorageBound(blocks, &reduction))
               .ok())
        return status;
      // Exact input frame coexists with the final Metal-owned-frame export;
      // raw quant, sigma, CfL and packed forward coefficients survive Evaluate.
      for (auto owner :
           {forward.working, cfl.working, quantizer, reduction, frame.output})
        if (!p.evaluator.Add(owner))
          return Overflow();
      if (!p.evaluator.AddVector<int32_t>(block_count, kFreshExact) ||
          !p.evaluator.AddVector<float>(block_count, kFreshExact))
        return Overflow();
    }
    ac_strategy_search_internal::StoragePlan ac;
    ac_strategy_search_internal::HostStoragePlan ac_host;
    AcSubmissionStoragePlan submission;
    if (!(status = ac_strategy_search_internal::ComputeStoragePlan(
              coding, resident, &ac))
             .ok() ||
        !(status = ac_strategy_search_internal::ComputeHostStoragePlan(
              coding, resident, resident && IsSearch(e.rate_control_mode),
              &ac_host))
             .ok() ||
        !(status = ComputeAcSubmissionStoragePlan(
              {.batches = ac.stages.size(),
               .nonempty_batches = ac.stages.size()},
              &submission))
             .ok())
      return status;
    p.ac_search = ac_host.working;
    if (!p.ac_search.Add({ac.device_bytes, ac.device_bytes}) ||
        !p.ac_search.Add(submission.working))
      return Overflow();
  }
  HostStorageBound scores, timing;
  if (!scores.AddVector<double>(p.score_count, kFreshExact) ||
      (o.collect_timing && !timing.AddVector<VarDctEncodingAttemptTiming>(
                               p.maximum_attempts, kFreshExact)))
    return Overflow();
  if (p.maximum_attempts > 1 && (!p.retained_best.Add(p.serializer.output) ||
                                 !p.retained_best.Add(scores)))
    return Overflow();
  p.output = p.serializer.output;
  if (!p.output.Add(scores) || !p.output.Add(timing))
    return Overflow();
  p.working = p.frontend;
  for (auto owner : {p.evaluator, p.ac_search, p.serializer.working, timing,
                     p.search_control, p.retained_best})
    if (!p.working.Add(owner))
      return Overflow();
  *out = p;
  return Status::Ok();
}

Status
ComputeAutomaticExactSearchStoragePlan(Extent2D source,
                                       const CpuWorkflowStorageOptions &o,
                                       AutomaticExactSearchStoragePlan *out) {
  if (out == nullptr ||
      o.encoding.backend != VarDctBackendPreference::kAutomatic ||
      o.encoding.metal_aq_mode !=
          GpuAdaptiveQuantizationMode::kExactCoefficients ||
      !IsSearch(o.encoding.rate_control_mode))
    return Status::InvalidArgument("Unsupported automatic exact search shape");
  AutomaticExactSearchStoragePlan p;
  auto options = o;
  options.encoding.backend = VarDctBackendPreference::kCpu;
  Status status = ComputeCpuWorkflowStoragePlan(source, options, &p.cpu);
  if (!status.ok())
    return status;
  options.encoding.backend = VarDctBackendPreference::kMetal;
  status =
      ComputeMetalCompatibilityWorkflowStoragePlan(source, options, &p.metal);
  if (!status.ok())
    return status;
  p.working = p.cpu.working;
  if (!p.working.Add(p.metal.working))
    return Overflow();
  p.output = {
      std::max(p.cpu.output.retained_bytes, p.metal.output.retained_bytes),
      std::max(p.cpu.output.peak_bytes, p.metal.output.peak_bytes)};
  *out = p;
  return Status::Ok();
}

} // namespace gjxl::codestream_internal
