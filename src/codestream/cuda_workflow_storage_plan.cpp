// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/cuda_workflow_storage_plan.h"
#include "gpu/cuda/cuda_profile_storage_plan.h"

#include "codec/coefficient_order_population_internal.h"
#include "codec/frontend_storage_plan.h"
#include "codestream/workflow_internal.h"
#include "codestream/workflow_publication_storage_plan.h"
#include "core/frame_geometry.h"
#include "gpu/ops/ac_strategy_storage_plan.h"
#include <algorithm>
#include <cmath>
#include <string_view>

namespace gjxl::codestream_internal {
namespace {
using enum resource_budget_internal::VectorCapacityPolicy;
Status Overflow() {
  return Status::OutOfMemory("CUDA workflow storage bound overflows");
}

Status ComputeCompatibility(Extent2D source, const CudaWorkflowStorageOptions &o,
                            CudaWorkflowStoragePlan *out) {
  // These routes retain the shared CPU frontend/AQ owners (including exact
  // coefficients or final quality measurement). Compose their audited bound
  // with the concrete CUDA owners. No memory guard is relaxed for these modes.
  CpuWorkflowStorageOptions cpu_options{o.encoding, o.collect_timing, o.collect_profile};
  cpu_options.encoding.backend = VarDctBackendPreference::kCpu;
  cpu_options.encoding.adaptive_epf_sharpness = false;
  CpuWorkflowStoragePlan cpu;
  Status status = ComputeCpuWorkflowStoragePlan(source, cpu_options, &cpu);
  if (!status.ok())
    return status;
  const auto &e = o.encoding;
  const bool frame_only =
      e.gpu_aq_mode == GpuAdaptiveQuantizationMode::kMaximumThroughput;
  const bool exact =
      e.gpu_aq_mode == GpuAdaptiveQuantizationMode::kExactCoefficients;
  const bool maximum =
      e.rate_control_mode == VarDctRateControlMode::kMaximumError;
  CudaWorkflowStoragePlan p;
  p.coding_extent = cpu.coding_extent;
  p.maximum_attempts = cpu.maximum_attempts;
  p.score_count = frame_only ? 0 : cpu.score_count;
  p.frontend = cpu.frontend;
  p.serializer = cpu.serializer;
  p.output = cpu.output;
  p.working = cpu.working;
  AqEvaluationOptions evaluation;
  evaluation.dc_quantization = ResolveDcQuantization(e);
  evaluation.dc_prediction = e.dc_prediction;
  evaluation.profile.extra_dc_precision =
      evaluation.dc_quantization == DcQuantizationMode::kPredictionAware ? 1
                                                                         : 0;
  evaluation.profile.adaptive_dc_smoothing = ResolveAdaptiveDcSmoothing(e);
  evaluation.metric = maximum ? AqEvaluationMetric::kMaximumError
                              : AqEvaluationMetric::kButteraugli;
  evaluation.maximum_error = e.maximum_error;
  HostStorageBound cuda_host;
  if (frame_only || exact) {
    status = frame_only ? cuda_internal::ComputeCudaFrameOnlyStoragePlan(
                              source, evaluation, true, &p.compatibility)
                        : cuda_internal::ComputeCudaExactStoragePlan(
                              source, evaluation, &p.compatibility);
    if (!status.ok())
      return status;
    cuda_host = p.compatibility.host;
    if (!cuda_host.Add(p.compatibility.butteraugli.host))
      return Overflow();
    for (size_t bytes :
         {p.compatibility.persistent_bytes, p.compatibility.staging_bytes,
          p.compatibility.butteraugli.capacity_bytes})
      if (!p.device.Add({bytes, bytes}))
        return Overflow();
    if (frame_only) {
      size_t input_bytes = 0;
      status = cuda_internal::ComputeCudaInputStoragePlan(source, &input_bytes);
      if (!status.ok())
        return status;
      if (!p.device.Add({input_bytes, input_bytes}))
        return Overflow();
    }
  } else {
    status = cuda_internal::ComputeCudaResidentStoragePlan(
        {.source = source,
         .evaluation = evaluation,
         .resident_frontend = true,
         .omit_initial_search_data = UseFixedDct8Strategy(e),
         .borrowed_input = true},
        &p.resident);
    if (!status.ok())
      return status;
    cuda_host = p.resident.host;
    if (!cuda_host.Add(p.resident.native_ac) ||
        !cuda_host.Add(p.resident.butteraugli.host))
      return Overflow();
    size_t input_bytes = 0;
    status = cuda_internal::ComputeCudaInputStoragePlan(source, &input_bytes);
    if (!status.ok())
      return status;
    for (size_t bytes :
         {input_bytes, p.resident.persistent_bytes, p.resident.staging_bytes,
          p.resident.sparse_header_bytes,
          p.resident.butteraugli.capacity_bytes})
      if (!p.device.Add({bytes, bytes}))
        return Overflow();
  }
  if (!frame_only && !UseFixedDct8Strategy(e)) {
    ac_strategy_search_internal::StoragePlan ac;
    ac_strategy_search_internal::HostStoragePlan ac_host;
    status = ac_strategy_search_internal::ComputeStoragePlan(
        p.coding_extent, !exact, &ac, nullptr, UseDenseDct32Search(e));
    if (!status.ok())
      return status;
    status = ac_strategy_search_internal::ComputeHostStoragePlan(
        p.coding_extent, !exact, p.maximum_attempts > 1, &ac_host,
        UseDenseDct32Search(e));
    if (!status.ok())
      return status;
    if (!cuda_host.Add(ac_host.working) ||
        !p.device.Add({ac.input_arena_bytes, ac.input_arena_bytes}) ||
        !p.device.Add({ac.resource_arena_bytes, ac.resource_arena_bytes}))
      return Overflow();
  }
  // A direct exact reconstruction snapshots its incoming frame before
  // publication. Resident native ownership can also coexist with a prior frame.
  frontend_storage_internal::OwnedFrameStoragePlan frame;
  status =
      frontend_storage_internal::ComputeOwnedFrameStoragePlan(source, &frame);
  if (!status.ok())
    return status;
  p.completed = frame.output;
  if (!p.frontend.Add(cuda_host) || !p.working.Add(cuda_host) ||
      !p.working.Add(p.device) || !p.working.Add(p.completed))
    return Overflow();
  p.cuda_idle_capacity = p.device.peak_bytes;
  *out = p;
  return Status::Ok();
}
} // namespace

Status ComputeCudaWorkflowStoragePlan(Extent2D source,
                                      const CudaWorkflowStorageOptions &o,
                                      CudaWorkflowStoragePlan *out) {
  const auto &e = o.encoding;
  const bool search =
      e.rate_control_mode == VarDctRateControlMode::kTargetBytes ||
      e.rate_control_mode == VarDctRateControlMode::kTargetBitsPerPixel;
  if (out == nullptr || e.backend != VarDctBackendPreference::kCuda ||
      e.effort < 1 || e.effort > 10 ||
      e.cpu_thread_count > kMaximumCpuThreadCount ||
      (search &&
       (e.target_size_maximum_attempts == 0 ||
        e.target_size_maximum_attempts > kMaximumTargetSizeEncodeAttempts)) ||
      (!search &&
       e.rate_control_mode == VarDctRateControlMode::kButteraugliTarget &&
       (!std::isfinite(e.butteraugli_target) || e.butteraugli_target <= 0)))
    return Status::InvalidArgument("CUDA workflow storage options are invalid");
  if (o.collect_gpu_profile &&
      ((e.gpu_aq_mode != GpuAdaptiveQuantizationMode::kFullyResident &&
        e.gpu_aq_mode != GpuAdaptiveQuantizationMode::kThroughput) ||
       e.rate_control_mode != VarDctRateControlMode::kButteraugliTarget))
    return Status::InvalidArgument("CUDA GPU profiles require a resident target workflow");
  if (e.gpu_aq_mode == GpuAdaptiveQuantizationMode::kExactCoefficients ||
      e.gpu_aq_mode == GpuAdaptiveQuantizationMode::kMaximumThroughput ||
      e.rate_control_mode == VarDctRateControlMode::kMaximumError)
    return ComputeCompatibility(source, o, out);
  if (e.gpu_aq_mode != GpuAdaptiveQuantizationMode::kFullyResident &&
      e.gpu_aq_mode != GpuAdaptiveQuantizationMode::kThroughput)
    return Status::InvalidArgument("CUDA workflow AQ mode is invalid");
  if (!search &&
      e.rate_control_mode != VarDctRateControlMode::kButteraugliTarget)
    return Status::InvalidArgument(
        "CUDA workflow storage control mode is invalid");
  FrameGeometry geometry;
  Status status = FrameGeometry::Create(source, &geometry);
  if (!status.ok())
    return status;
  CudaWorkflowStoragePlan p;
  p.coding_extent = geometry.padded_frame();
  p.maximum_attempts = search ? e.target_size_maximum_attempts : 1;
  p.score_count = AdaptiveQuantizationIterations(e) +
                  size_t(e.collect_final_butteraugli_score);
  const bool fixed = UseFixedDct8Strategy(e);
  if (o.collect_gpu_profile) {
    // Reference preparation (unless evaluation-free), initial quantization,
    // optional AC search, encoding-policy setup, and the resident policy.
    // Sparse AC publication can submit one additional packing callback.
    // AQ iterations execute inside the policy callback, so they do not add
    // submissions. These bounds follow the call graph, not measured counts.
    const size_t submissions = 4 + size_t(!fixed) + size_t(p.score_count != 0);
    size_t id_length = cuda_internal::kCudaKernelProfileIdLength;
    for (std::string_view id : {
           "aq.prepare_reference", "aq.initial_quantization", "aq.prepare_encoding_policy",
           "aq.resident_policy", "ac_strategy.candidates", "frontend.ac_strategy",
           "frontend.prepare_evaluator", "frontend.initial_quantization",
           "frontend.reconfigure_aq", "frontend.quant_adjustment", "frontend.fixed_cfl",
           "resident.aq", "frontend.ac_strategy.prepare", "frontend.ac_strategy.wait",
           "frontend.ac_strategy.readback", "frontend.ac_strategy.merge"})
      id_length = std::max(id_length, id.size());
    // Launch bounds for the resident workflow (not arbitrary low-level
    // submissions). One batch per selected transform family, at most seven.
    // Quantizer selection has two deviations, four radix bytes, histogram +
    // bucket per byte, then finalization and raw quantization: 2*(1+4*2)+2.
    constexpr size_t quantizer = 20;
    const size_t families = fixed ? 1 : 7;
    // Psycho: opsin <=7, low/medium <=7 (including unfused variants), four
    // two-pass splits, one two-pass blur and suppression. Reference preparation
    // has at most two scales, expansion/subsampling and a three-launch mask.
    constexpr size_t psycho = 7 + 7 + 4 * 2 + 2 + 1;
    constexpr size_t reference = 3 + 2 * psycho + 3 + 3;
    // Difference: six Malta, two precompute+blur masks, one final kernel.
    // Compare includes both scales, expansion/subsampling, crop/composition,
    // <=4 radix-256 maximum reductions over a uint32 count, and one block
    // reduction/composition launch per family. Geometry changes grid sizes.
    constexpr size_t difference = 6 + 2 * 3 + 1;
    const size_t compare = 3 + 2 * (psycho + difference) + 3 + 1 + 1 + 4 + families;
    // Reconstruction selects/encodes coefficients and rebuilds LLF/inverse
    // per family, <=3 DC kernels and <=5 filter/color kernels. Forward/CfL
    // occurs once. A score iteration adds initialize/update (initialize once).
    const size_t reconstruction = quantizer + 4 * families + 3 + 5;
    const size_t frame_only = quantizer + 2 * families + 3;
    const size_t packing = 1 + families + 1; // population, AC packing, compact
    const size_t policy = families + quantizer + 1 + 1 +
      p.score_count * (reconstruction + compare + 1) +
      (e.collect_final_butteraugli_score ? 0 : frame_only) + packing;
    const size_t initial = fixed ? 2 : 4 + 3 + 1; // field, inverse Gaborish, CfL
    const size_t setup = families + 2; // adjustment, positive-range initialize/reduce
    const size_t ac = fixed ? 0 : 7 * 3; // norms, fused evaluation, final cost
    const size_t dispatches = (p.score_count == 0 ? 0 : reference) +
      initial + setup + ac + policy + 1; // optional separate sparse pack
    const size_t child_dispatches = std::max({reference, initial, setup, ac, policy});
    p.profile_shape = {.wall_stages = 6 + (fixed ? 0u : 4u),
                       .submissions = submissions, .stages = submissions,
                       .dispatches = dispatches,
                       .maximum_id_length = id_length};
    status = gpu_profile_internal::ComputeProfileStorageBound(p.profile_shape, &p.profile_output);
    if (!status.ok()) return status;
    // The parent can coexist with the largest in-flight operation capture
    // (policy + sparse packing), one fresh resolved snapshot, and two retained
    // submission recordings: policy's owner remains alive during sparse pack.
    HostStorageBound capture;
    status = gpu_profile_internal::ComputeProfileStorageBound(
      {.submissions = 2, .stages = 2, .dispatches = child_dispatches + 1,
       .maximum_id_length = id_length}, &capture);
    if (!status.ok()) return status;
    gpu_profile_internal::SubmissionProfileStoragePlan child;
    status = cuda_internal::ComputeCudaSubmissionProfileStoragePlan(
      {.stages = 1, .dispatches = child_dispatches,
       .maximum_stage_id_length = id_length,
       .maximum_kernel_id_length = cuda_internal::kCudaKernelProfileIdLength,
       .maximum_submission_id_length = id_length}, &child);
    if (!status.ok()) return status;
    p.diagnostics = p.profile_output;
    if (!p.diagnostics.Add(capture) || !p.diagnostics.Add(child.resolution) ||
        !p.diagnostics.Add(child.recorded)) return Overflow();
  }
  AqEvaluationOptions evaluation;
  evaluation.evaluation_free = p.score_count == 0;
  evaluation.dc_quantization = ResolveDcQuantization(e);
  evaluation.dc_prediction = e.dc_prediction;
  evaluation.profile.extra_dc_precision =
      evaluation.dc_quantization == DcQuantizationMode::kPredictionAware ? 1
                                                                         : 0;
  evaluation.profile.adaptive_dc_smoothing = ResolveAdaptiveDcSmoothing(e);
  status = cuda_internal::ComputeCudaResidentStoragePlan(
      {.source = source,
       .evaluation = evaluation,
       .resident_frontend = true,
       .omit_initial_search_data = fixed,
       .borrowed_input = true},
      &p.resident);
  if (!status.ok())
    return status;
  size_t input_bytes = 0;
  status = cuda_internal::ComputeCudaInputStoragePlan(source, &input_bytes);
  if (!status.ok())
    return status;
  for (size_t bytes :
       {input_bytes, p.resident.persistent_bytes, p.resident.staging_bytes,
        p.resident.sparse_header_bytes, p.resident.butteraugli.capacity_bytes})
    if (!p.device.Add({bytes, bytes}))
      return Overflow();
  p.frontend = p.resident.host;
  if (!p.frontend.Add(p.resident.butteraugli.host))
    return Overflow();
  ac_strategy_search_internal::StoragePlan ac;
  ac_strategy_search_internal::HostStoragePlan ac_host;
  if (!fixed) {
    status = ac_strategy_search_internal::ComputeStoragePlan(
        p.coding_extent, true, &ac, nullptr, UseDenseDct32Search(e));
    if (!status.ok())
      return status;
    status = ac_strategy_search_internal::ComputeHostStoragePlan(
        p.coding_extent, true, search, &ac_host, UseDenseDct32Search(e));
    if (!status.ok())
      return status;
    if (!p.frontend.Add(ac_host.working) ||
        !p.device.Add({ac.resource_arena_bytes, ac.resource_arena_bytes}))
      return Overflow();
  }
  // All retries retain fixed geometry and arena sizes. Buffers acquired from
  // the cache require exact capacities and remain inside this same reservation.
  p.cuda_idle_capacity = p.device.peak_bytes;
  const size_t blocks =
      geometry.block_grid().blocks.width * geometry.block_grid().blocks.height;
  frontend_storage_internal::ColorCorrelationStoragePlan cfl;
  status = frontend_storage_internal::ComputeColorCorrelationStoragePlan(
      p.coding_extent,
      frontend_storage_internal::ColorCorrelationStorageMode::kCopy, &cfl);
  if (!status.ok())
    return status;
  // Prepared sharpness, provisional/selected/replacement grids, compatibility
  // initial/strategy/adjusted fields and retry CfL copies.
  if (!p.frontend.AddVector<uint8_t>(blocks, kFreshExact, 4) ||
      !p.frontend.AddVector<float>(blocks, kFreshExact, 3) ||
      !p.frontend.Add(cfl.working, search ? 2 : 1))
    return Overflow();
  frontend_storage_internal::OwnedFrameStoragePlan frame;
  status =
      frontend_storage_internal::ComputeOwnedFrameStoragePlan(source, &frame);
  if (!status.ok())
    return status;
  p.completed = frame.output;
  const size_t dense_bytes = frame.ac_coefficients * sizeof(int32_t);
  p.completed.retained_bytes -= dense_bytes;
  p.completed.peak_bytes -= dense_bytes;
  if (!p.completed.Add(p.resident.native_ac) ||
      !p.completed.AddVector<size_t>(frame.ac_groups + 1, kFreshExact) ||
      !p.completed.AddVector<vardct_frame_internal::CoefficientOrderPopulation>(
          1, kFreshExact))
    return Overflow();
  status = ComputeSerializerStoragePlan(
      source,
      {.coding = {.entropy_behavior = ResolveEntropyBehavior(e),
                  .coefficient_order_behavior =
                      ResolveCoefficientOrderBehavior(e),
                  .dc_prediction = e.dc_prediction,
                  .dc_uint_search = UseDcUintSearch(e)},
       .cpu_thread_count = e.cpu_thread_count,
       .collect_profile = o.collect_profile || o.collect_gpu_profile},
      &p.serializer);
  if (!status.ok())
    return status;
  WorkflowPublicationStoragePlan publication;
  status = ComputeWorkflowPublicationStoragePlan(
      p.serializer.output, p.score_count, p.maximum_attempts, search,
      o.collect_timing, &publication,
      "CUDA workflow publication storage overflows");
  if (!status.ok())
    return status;
  p.output = publication.output;
  if (!p.output.Add(p.profile_output)) return Overflow();
  for (auto part : {p.frontend, p.device, p.completed, p.serializer.working,
                    p.diagnostics,
                    publication.scores, publication.timing,
                    publication.search_control, publication.retained_best})
    if (!p.working.Add(part))
      return Overflow();
  *out = p;
  return Status::Ok();
}
} // namespace gjxl::codestream_internal
