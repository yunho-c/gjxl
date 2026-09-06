// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_aq_profile_storage_plan.h"

#include <algorithm>
#include <limits>
#include <string_view>

namespace gjxl::metal_internal {
namespace {
using resource_budget_internal::HostStorageBound;

Status Geometry(Extent2D source, Extent2D coding, size_t *blocks) {
  Status status = ValidateAqStorageGeometry(source, coding);
  if (!status.ok())
    return status;
  size_t pixels = 0;
  if (!coding.try_area(&pixels) ||
      pixels > std::numeric_limits<uint32_t>::max() / size_t{3})
    return Status::InvalidArgument("AQ profile geometry exceeds shader limits");
  *blocks = pixels / 64;
  return Status::Ok();
}

constexpr size_t MaximumAqIdLength() {
  size_t length = 0;
  for (size_t i = 0; i <= static_cast<size_t>(AcStrategyType::kCount); ++i) {
    const auto strategy = static_cast<AcStrategyType>(i);
    for (const char *id : {AqReconstructionProfileStageId(strategy),
                           AqReconstructionCoefficientProfileStageId(strategy),
                           AqReconstructionScatterProfileStageId(strategy),
                           AqForwardCoefficientProfileStageId(strategy),
                           AqFinalFrameProfileStageId(strategy)})
      length = std::max(length, std::string_view(id).size());
  }
  for (std::string_view id : {"aq.reconstruction",
                              "aq.reconstruction.reset",
                              "aq.reconstruction.quantizer",
                              "aq.reconstruction.final_cfl",
                              "aq.policy_initialize",
                              "aq.policy_update",
                              "aq.gaborish",
                              "aq.epf.pass_0",
                              "aq.epf.pass_1",
                              "aq.epf.pass_2",
                              "aq.opsin_to_linear",
                              "aq.block_reduction",
                              "aq.final_frame",
                              "aq.final_frame.quantizer",
                              "butteraugli.psycho.sub",
                              "butteraugli.psycho.main",
                              "butteraugli.malta.sub",
                              "butteraugli.malta.main",
                              "butteraugli.mask_final.sub",
                              "butteraugli.mask_final.main",
                              "butteraugli.mask.main",
                              "butteraugli.l2.main",
                              "butteraugli.resident_reduction",
                              "butteraugli.score_reduction",
                              "resident.aq",
                              "resident.frame_output_prepare",
                              "resident.frame_mapping",
                              "resident.frame_assembly"})
    length = std::max(length, id.size());
  return length;
}

size_t KernelIdLength(size_t stage_length) {
  return std::max(kMaximumKernelIdBytes - 1,
                  stage_length + sizeof(".dispatch_") - 1 +
                      std::numeric_limits<size_t>::digits10 + 1);
}

Status
SingleSubmission(size_t dispatches, std::string_view id,
                 gpu_profile_internal::SubmissionProfileStoragePlan *out) {
  return gpu_profile_internal::ComputeSubmissionProfileStoragePlan(
      {1, dispatches, id.size(), id.size(), KernelIdLength(id.size()),
       id.size()},
      out);
}
} // namespace

Status ComputeInitialQuantSortPlan(size_t blocks, InitialQuantSortPlan *out) {
  if (out == nullptr || blocks == 0)
    return Status::InvalidArgument("Initial-quant sort shape is invalid");
  InitialQuantSortPlan p{.count = 1};
  size_t levels = 0;
  while (p.count < blocks) {
    if (p.count > std::numeric_limits<uint32_t>::max() / 2)
      return Status::InvalidArgument(
          "Resident initial-quant sort dimensions are too large");
    p.count *= 2;
    ++levels;
  }
  // The median and deviation each sort with 1 + ... + levels dispatches.
  // levels <= 31, so this multiplication is proven small on the host.
  p.dispatches = levels * (levels + 1);
  *out = p;
  return Status::Ok();
}

Status ComputeButteraugliDispatchPlan(Extent2D source, size_t anchors,
                                      size_t families,
                                      ButteraugliDispatchPlan *out) {
  if (out == nullptr)
    return Status::InvalidArgument("Butteraugli dispatch plan output is null");
  ButteraugliStoragePlan storage;
  Status status = ComputeButteraugliStoragePlan(source, false, &storage);
  if (!status.ok())
    return status;
  const size_t pixels = source.width * source.height;
  if (anchors > pixels || families > anchors ||
      families > kSupportedAqStrategies.size() ||
      ((anchors == 0) != (families == 0)))
    return Status::InvalidArgument("Butteraugli sink counts are invalid");
  constexpr size_t psycho = 1 + 1 + 2 * 2 + 2 + 1 + 2 * 2;
  constexpr size_t malta = 2 * kButteraugliMaltaAccumulationOrder.size();
  constexpr size_t mask_blur = 2;
  constexpr size_t reference = psycho + mask_blur + 1;     // Erosion.
  constexpr size_t difference = malta + 1 + mask_blur + 1; // L2 and final.
  const size_t reduction = ButteraugliReductionDispatchCount(pixels);
  ButteraugliDispatchPlan p;
  p.expanded = storage.expanded;
  p.multiscale = storage.multiscale;
  p.reference = reference + (p.expanded ? 3 : p.multiscale ? 3 + reference : 0);
  p.comparison = psycho + difference + reduction +
                 (p.expanded     ? 4
                  : p.multiscale ? 4 + psycho + difference
                                 : 0);
  if (p.multiscale && anchors != 0) {
    // Subsample + two psycho/Malta passes, two mask blurs, one subscale final.
    // L2/final on the main scale are fused into the resident sink dispatches.
    p.resident_comparison = 3 + 2 * (psycho + malta + mask_blur) + 1 +
                            families +
                            ButteraugliReductionDispatchCount(anchors);
  }
  *out = p;
  return Status::Ok();
}

Status
ComputeResidentAqProfileStoragePlan(Extent2D source, Extent2D coding,
                                    const ResidentAqProfileInputOptions &policy,
                                    AqProfileFrameOutput frame,
                                    ResidentAqProfileStoragePlan *out) {
  if (out == nullptr ||
      (frame != AqProfileFrameOutput::kNone &&
       frame != AqProfileFrameOutput::kOwned &&
       frame != AqProfileFrameOutput::kCompleted) ||
      (!policy.evaluate_final_field && frame == AqProfileFrameOutput::kNone))
    return Status::InvalidArgument(
        "Resident AQ profile output shape is invalid");
  size_t blocks = 0;
  Status status = Geometry(source, coding, &blocks);
  if (!status.ok())
    return status;
  const size_t families = std::min(blocks, kSupportedAqStrategies.size());
  ButteraugliDispatchPlan butter;
  status = ComputeButteraugliDispatchPlan(source, blocks, families, &butter);
  if (!status.ok())
    return status;
  if (policy.butteraugli_sinks != butter.multiscale)
    return Status::InvalidArgument(
        "Resident AQ profile sink geometry disagrees");
  ResidentAqProfileStoragePlan p;
  status = ComputeResidentAqProfileInputStoragePlan(policy, &p.metadata);
  if (!status.ok())
    return status;
  // Reset + 20-dispatch radix quantizer + four per-family coefficient/inverse/
  // scatter dispatches + filters + Opsin-to-linear + perceptual work + update.
  // First use also gathers/transforms each family, computes final CfL and
  // initializes the policy. Counting all first-use work bounds cached runs too.
  const size_t per_score = 1 + 20 + 4 * families + size_t(policy.gaborish) +
                           policy.epf_iterations + 1 + 1 +
                           (butter.multiscale ? butter.resident_comparison
                                              : butter.comparison + families);
  p.maximum_dispatches =
      p.metadata.score_count * per_score + 2 * families + 2 +
      size_t(!policy.evaluate_final_field) * (20 + 2 * families);
  p.maximum_id_length = MaximumAqIdLength();
  status = gpu_profile_internal::ComputeSubmissionProfileStoragePlan(
      {p.metadata.stage_capacity, p.maximum_dispatches, p.maximum_id_length,
       p.maximum_id_length, KernelIdLength(p.maximum_id_length),
       sizeof("resident.aq") - 1},
      &p.graph);
  if (!status.ok())
    return status;
  const size_t wall_count = frame == AqProfileFrameOutput::kNone    ? 0
                            : frame == AqProfileFrameOutput::kOwned ? 2
                                                                    : 3;
  status = gpu_profile_internal::ComputeProfileStorageBound(
      {wall_count, 0, 0, 0, p.maximum_id_length}, &p.wall);
  if (!status.ok())
    return status;
  p.working = p.metadata.input;
  if (!p.working.Add(p.graph.recorded))
    return Status::OutOfMemory("Resident AQ profile bound overflows");
  p.working.peak_bytes =
      std::max(p.working.peak_bytes, p.graph.resolution.peak_bytes);
  p.working.retained_bytes =
      std::max(p.working.retained_bytes, p.graph.resolution.retained_bytes);
  if (!p.working.Add(p.wall))
    return Status::OutOfMemory("Resident AQ wall profile bound overflows");
  *out = p;
  return Status::Ok();
}

Status
ComputeAqAuxiliaryProfileStoragePlan(const AqAuxiliaryProfileOptions &o,
                                     AqAuxiliaryProfileStoragePlan *out) {
  if (out == nullptr)
    return Status::InvalidArgument("Auxiliary AQ profile output is null");
  size_t blocks = 0;
  Status status = Geometry(o.source, o.coding, &blocks);
  if (!status.ok())
    return status;
  ButteraugliDispatchPlan butter;
  status = ComputeButteraugliDispatchPlan(o.source, 0, 0, &butter);
  if (!status.ok())
    return status;
  AqAuxiliaryProfileStoragePlan p;
  p.reference_dispatches = butter.reference;
  p.initial_dispatches =
      5 + size_t(o.resident_ac_strategy_inputs) +
      3 * size_t(o.resident_ac_strategy_inputs && o.gaborish) +
      size_t(o.resident_initial_cfl);
  if (o.frame_only_resident_quantizer) {
    status = ComputeInitialQuantSortPlan(blocks, &p.sort);
    if (!status.ok())
      return status;
    p.initial_dispatches += 5 + p.sort.dispatches;
  }
  p.adjustment_dispatches = 1 + std::min(blocks, kSupportedAqStrategies.size());
  status = SingleSubmission(p.reference_dispatches,
                            "frontend.prepare_aq.reference", &p.reference);
  if (!status.ok())
    return status;
  status = SingleSubmission(p.initial_dispatches,
                            "frontend.initial_quantization", &p.initial);
  if (!status.ok())
    return status;
  status = SingleSubmission(p.adjustment_dispatches,
                            "frontend.quant_adjustment", &p.adjustment);
  if (!status.ok())
    return status;
  *out = p;
  return Status::Ok();
}

} // namespace gjxl::metal_internal
