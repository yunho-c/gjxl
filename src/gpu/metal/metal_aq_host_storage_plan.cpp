// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_aq_host_storage_plan.h"

#include <algorithm>
#include <limits>

// Use the executing metadata types, not copied ABI-size constants. This planner
// belongs to the Metal library but does not initialize or call a backend.
#include "gpu/metal/metal_aq_evaluation_internal.h"
#include "gpu/metal/metal_storage_plan.h"

namespace gjxl::metal_internal {
namespace {
using resource_budget_internal::HostStorageBound;
using enum resource_budget_internal::VectorCapacityPolicy;
using vardct_frame_internal::QuantizedAcTransformLayout;

Status Overflow() {
  return Status::OutOfMemory("Metal AQ host storage bound overflows");
}

bool AddColorMetadata(size_t blocks, size_t tiles, HostStorageBound *bound) {
  // records and tile offsets survive the helper; positions and value offsets
  // coexist while it runs. All four vectors start empty on every call.
  return bound->AddVector<int32_t>(6 * blocks, kFreshExact) &&
         bound->AddVector<int32_t>(tiles + 1, kFreshExact, 2) &&
         bound->AddVector<size_t>(tiles, kFreshExact);
}
} // namespace

Status ComputeAqHostStoragePlan(const AqHostStorageOptions &o,
                                AqHostStoragePlan *out) {
  if (out == nullptr)
    return Status::InvalidArgument("Metal AQ host storage output is null");
  Status status = ValidateAqStorageGeometry(o.source_extent, o.coding_extent);
  if (!status.ok())
    return status;
  size_t pixels = 0;
  if (!o.coding_extent.try_area(&pixels) ||
      pixels > std::numeric_limits<uint32_t>::max() / size_t{3})
    return Status::InvalidArgument(
        "Metal AQ host coefficient count is too large");
  if ((o.metric != AqEvaluationMetric::kButteraugli &&
       o.metric != AqEvaluationMetric::kMaximumError) ||
      (o.resident_ac_strategy_inputs && !o.resident_initial_quant) ||
      (o.defer_final_transform_metadata &&
       (!o.resident_ac_strategy_inputs || !o.resident_quantization ||
        o.frame_only)) ||
      (o.frame_only &&
       (o.reconfigure || o.exact_coefficients || o.reconstructed_rgb_readback ||
        o.resident_quant_field_readback)) ||
      (o.reconstruct_exact_coefficients && !o.exact_coefficients) ||
      (o.resident_quant_field_readback && !o.resident_quantization) ||
      (o.initial_pixel_mask_readback && !o.resident_initial_quant))
    return Status::InvalidArgument(
        "Metal AQ host storage options are inconsistent");
  // Geometry and the 32-bit coefficient limit prove these products fit size_t.
  const size_t blocks = pixels / 64;
  const size_t tiles = ((o.coding_extent.width + 63) / 64) *
                       ((o.coding_extent.height + 63) / 64);
  const size_t groups = ((o.coding_extent.width + 255) / 256) *
                        ((o.coding_extent.height + 255) / 256);
  const size_t source_pixels = o.source_extent.width * o.source_extent.height;
  AqHostStoragePlan p;
  auto &prepared = p.prepared;
  if (!prepared.AddVector<uint8_t>(blocks, kFreshExact, 2) || // grid, sharpness
      !prepared.AddVector<int8_t>(tiles, kFreshExact, 2) ||
      !prepared.AddVector<AqAnchor>(blocks, kGrowing) ||
      (!o.defer_final_transform_metadata &&
       !prepared.AddVector<QuantizedAcTransformLayout>(blocks, kFreshExact)) ||
      (!o.frame_only &&
       !prepared.AddVector<float>(blocks, kFreshExact,
                                  4)) || // map + 3B maximum error
      (o.resident_initial_quant &&
       (!prepared.AddVector<float>(blocks, kFreshExact, 2) ||
        (!o.resident_ac_strategy_inputs &&
         !prepared.AddVector<float>(pixels, kFreshExact)))))
    return Overflow();
  HostStorageBound setup;
  if (!setup.AddVector<int32_t>(2 * blocks, kFreshExact) || // strategy records
      !setup.AddVector<int32_t>(2 * blocks, kGrowing) ||    // flattened anchors
      // Seven lists, but their logical counts SUM to at most blocks.
      !setup.AddVector<std::array<int32_t, 2>>(blocks, kGrowing) ||
      !setup.AddVector<float>(kAqQuantTableValueCount, kFreshExact) ||
      (o.resident_quantization && !o.defer_final_transform_metadata &&
       !AddColorMetadata(blocks, tiles, &setup)) ||
      (!o.frame_only && o.metric == AqEvaluationMetric::kButteraugli &&
       !setup.AddVector<float>(
           *std::max_element(kButteraugliKernelSizes.begin(),
                             kButteraugliKernelSizes.end()),
           kFreshExact)))
    return Overflow();
  p.preparation = prepared;
  if (!p.preparation.Add(setup))
    return Overflow();

  p.retained = {prepared.retained_bytes, prepared.retained_bytes};
  if ((o.reconfigure && o.defer_final_transform_metadata &&
       !p.retained.AddVector<QuantizedAcTransformLayout>(blocks,
                                                         kFreshExact)) ||
      // Lazy raw quant is fixed-size even across resident/nonresident calls.
      !p.retained.AddVector<int32_t>(blocks, kFreshExact) ||
      (o.exact_coefficients &&
       (!p.retained.AddVector<int32_t>(3 * pixels, kFreshExact) ||
        !p.retained.AddVector<int32_t>(3 * blocks, kFreshExact))) ||
      (o.reconstruct_exact_coefficients &&
       !p.retained.AddVector<float>(3 * pixels, kFreshExact)) ||
      (o.reconstructed_rgb_readback &&
       !p.retained.AddVector<float>(source_pixels, kFreshExact, 3)) ||
      (o.resident_quant_field_readback &&
       !p.retained.AddVector<float>(blocks, kFreshExact)) ||
      (o.initial_pixel_mask_readback && o.resident_ac_strategy_inputs &&
       !p.retained.AddVector<float>(pixels, kFreshExact)))
    return Overflow();
  HostStorageBound temporary;
  // Invariant CfL replacement: both old and fresh arrays are live at upload.
  if (!temporary.AddVector<int8_t>(tiles, kFreshExact, 2))
    return Overflow();
  if (o.resident_quantization) {
    HostStorageBound adjusted;
    if (!adjusted.AddVector<float>(blocks, kFreshExact))
      return Overflow();
    temporary.peak_bytes = std::max(temporary.peak_bytes, adjusted.peak_bytes);
    temporary.retained_bytes =
        std::max(temporary.retained_bytes, adjusted.retained_bytes);
  }
  if (o.exact_coefficients) {
    HostStorageBound offsets;
    if (!offsets.AddVector<size_t>(groups, kFreshExact))
      return Overflow();
    temporary.peak_bytes = std::max(temporary.peak_bytes, offsets.peak_bytes);
    temporary.retained_bytes =
        std::max(temporary.retained_bytes, offsets.retained_bytes);
  }
  if (o.reconfigure) {
    HostStorageBound replacement;
    if (!replacement.AddVector<int32_t>(2 * blocks, kFreshExact, 2) ||
        !replacement.AddVector<std::array<int32_t, 2>>(blocks, kGrowing) ||
        !replacement.AddVector<AqAnchor>(blocks, kFreshExact) ||
        !replacement.AddVector<uint8_t>(blocks, kFreshExact) ||
        !replacement.AddVector<QuantizedAcTransformLayout>(blocks,
                                                           kFreshExact) ||
        (o.resident_quantization &&
         !AddColorMetadata(blocks, tiles, &replacement)))
      return Overflow();
    temporary.peak_bytes =
        std::max(temporary.peak_bytes, replacement.peak_bytes);
    temporary.retained_bytes =
        std::max(temporary.retained_bytes, replacement.retained_bytes);
  }
  p.operation = p.retained;
  if (!p.operation.Add(temporary))
    return Overflow();
  p.working = {p.retained.retained_bytes,
               std::max(p.preparation.peak_bytes, p.operation.peak_bytes)};
  *out = p;
  return Status::Ok();
}

Status
ComputeCompletedFrameHostStoragePlan(Extent2D source, Extent2D coding,
                                     size_t anchor_count,
                                     CompletedFrameHostStoragePlan *out) {
  if (out == nullptr)
    return Status::InvalidArgument(
        "Completed frame host storage output is null");
  CompletedFrameStoragePlan device;
  Status status =
      ComputeCompletedFrameStoragePlan(source, coding, anchor_count, &device);
  if (!status.ok())
    return status;
  const size_t blocks = (coding.width / 8) * (coding.height / 8);
  const size_t tiles = ((coding.width + 63) / 64) * ((coding.height + 63) / 64);
  CompletedFrameHostStoragePlan p;
  if (!p.output.AddVector<uint8_t>(blocks, kFreshExact, 2) ||
      !p.output.AddVector<int32_t>(blocks, kFreshExact, 4) ||
      !p.output.AddVector<float>(blocks, kFreshExact, 3) ||
      !p.output.AddVector<int8_t>(tiles, kFreshExact, 2) ||
      !p.output.AddVector<size_t>(device.group_count, kFreshExact))
    return Overflow();
  p.working = p.output;
  if (!p.working.AddVector<uint32_t>(anchor_count, kFreshExact))
    return Overflow();
  *out = p;
  return Status::Ok();
}

} // namespace gjxl::metal_internal
