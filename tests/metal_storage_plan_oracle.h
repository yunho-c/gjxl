// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <limits>
#include <utility>

#include "gpu/metal/metal_storage_plan.h"

namespace gjxl::test::storage_plan_oracle {

// Frozen capacity recipe from dd5cd54, independent of the production planner.
// Deliberately retain these branches: a layout change needs an explicit oracle
// review, not automatic reuse of the new implementation's arithmetic.
inline Status AddPlannedPlane(DeviceElementType type, Extent2D extent,
                              size_t stride, size_t *capacity) {
  const size_t size =
      (type == DeviceElementType::kF32 || type == DeviceElementType::kI32) ? 4
                                                                           : 1;
  using Wide = __uint128_t;
  const Wide bytes = ((Wide{extent.height} - 1) * stride + extent.width) * size;
  const Wide aligned = ((Wide{*capacity} + 255) / 256) * 256;
  if (aligned + bytes > std::numeric_limits<size_t>::max())
    return Status::InvalidArgument("Frozen AQ capacity overflows");
  *capacity = static_cast<size_t>(aligned + bytes);
  return Status::Ok();
}

inline Status AqCapacity(const metal_internal::AqStoragePlanOptions &p,
                         std::pair<size_t, size_t> *capacities) {
  const Extent2D blocks{p.coding_extent.width / 8, p.coding_extent.height / 8};
  const Extent2D tiles{(p.coding_extent.width + 63) / 64,
                       (p.coding_extent.height + 63) / 64};
  const size_t block_count = blocks.width * blocks.height;
  const size_t coefficients =
      p.coding_extent.width * p.coding_extent.height * 3;
  Status status;
  size_t persistent_bytes = 0;
  if (!p.frame_only && !p.borrowed_original_linear_rgb) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = AddPlannedPlane(DeviceElementType::kF32, p.source_extent,
                               p.source_extent.width, &persistent_bytes);
      if (!status.ok())
        return status;
    }
  }
  if (!p.borrowed_coding_opsin) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = AddPlannedPlane(DeviceElementType::kF32, p.coding_extent,
                               p.coding_extent.width, &persistent_bytes);
      if (!status.ok())
        return status;
    }
  }
  if (p.needs_reconstructed) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = AddPlannedPlane(DeviceElementType::kF32, p.coding_extent,
                               p.coding_extent.width, &persistent_bytes);
      if (!status.ok())
        return status;
    }
  }
  if (!p.frame_only) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = AddPlannedPlane(DeviceElementType::kF32, p.source_extent,
                               p.source_extent.width, &persistent_bytes);
      if (!status.ok())
        return status;
    }
    status = AddPlannedPlane(DeviceElementType::kI32,
                             {blocks.width * 2, blocks.height},
                             blocks.width * 2, &persistent_bytes);
    if (!status.ok())
      return status;
  }
  status =
      AddPlannedPlane(DeviceElementType::kI32, {2 * p.anchor_capacity_count, 1},
                      2 * p.anchor_capacity_count, &persistent_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kU8, blocks, blocks.width,
                           &persistent_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kF32, {size_t{11904}, 1},
                           size_t{11904}, &persistent_bytes);
  if (!status.ok())
    return status;
  if (p.resident_quantization) {
    status = AddPlannedPlane(DeviceElementType::kI32, {6 * block_count, 1},
                             6 * block_count, &persistent_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kI32,
                             {tiles.width * tiles.height + 1, 1},
                             tiles.width * tiles.height + 1, &persistent_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kI8, tiles, tiles.width,
                             &persistent_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kI8, tiles, tiles.width,
                             &persistent_bytes);
    if (!status.ok())
      return status;
  }

  size_t staging_bytes = 0;
  status = AddPlannedPlane(DeviceElementType::kI32, blocks, blocks.width,
                           &staging_bytes);
  if (!status.ok())
    return status;
  if (!p.frame_only) {
    status = AddPlannedPlane(DeviceElementType::kF32, blocks, blocks.width,
                             &staging_bytes);
    if (!status.ok())
      return status;
  }
  if (p.frame_only_resident_initial_quant) {
    const Extent2D pre_erosion_extent{p.coding_extent.width / 4,
                                      p.coding_extent.height / 4};
    status = AddPlannedPlane(DeviceElementType::kF32, pre_erosion_extent,
                             pre_erosion_extent.width, &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kF32, p.coding_extent,
                             p.coding_extent.width, &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kF32, blocks, blocks.width,
                             &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kF32, blocks, blocks.width,
                             &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kF32, p.coding_extent,
                             p.coding_extent.width, &staging_bytes);
    if (!status.ok())
      return status;
    if (p.frame_only_resident_quantizer) {
      status = AddPlannedPlane(DeviceElementType::kF32,
                               {p.initial_quant_sort_count, 1},
                               p.initial_quant_sort_count, &staging_bytes);
      if (!status.ok())
        return status;
      status =
          AddPlannedPlane(DeviceElementType::kF32, {1, 1}, 1, &staging_bytes);
      if (!status.ok())
        return status;
      status =
          AddPlannedPlane(DeviceElementType::kI32, {2, 1}, 2, &staging_bytes);
      if (!status.ok())
        return status;
    }
  }
  if (p.resident_quantization) {
    status = AddPlannedPlane(DeviceElementType::kF32, blocks, blocks.width,
                             &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kF32, blocks, blocks.width,
                             &staging_bytes);
    if (!status.ok())
      return status;
    status =
        AddPlannedPlane(DeviceElementType::kF32, {5, 1}, 5, &staging_bytes);
    if (!status.ok())
      return status;
    status =
        AddPlannedPlane(DeviceElementType::kI32, {256, 1}, 256, &staging_bytes);
    if (!status.ok())
      return status;
    status =
        AddPlannedPlane(DeviceElementType::kI32, {3, 1}, 3, &staging_bytes);
    if (!status.ok())
      return status;
    status =
        AddPlannedPlane(DeviceElementType::kF32, {2, 1}, 2, &staging_bytes);
    if (!status.ok())
      return status;
    status =
        AddPlannedPlane(DeviceElementType::kI32, {2, 1}, 2, &staging_bytes);
    if (!status.ok())
      return status;
  }
  for (size_t image = 0; image < p.filter_scratch_image_count; ++image) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status = AddPlannedPlane(DeviceElementType::kF32, p.coding_extent,
                               p.coding_extent.width, &staging_bytes);
      if (!status.ok())
        return status;
    }
  }
  if (!p.resident_quantization) {
    status = AddPlannedPlane(DeviceElementType::kI8, tiles, tiles.width,
                             &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kI8, tiles, tiles.width,
                             &staging_bytes);
    if (!status.ok())
      return status;
  }
  if (!p.frame_only) {
    status = AddPlannedPlane(DeviceElementType::kF32, blocks, blocks.width,
                             &staging_bytes);
    if (!status.ok())
      return status;
    if (p.metric == AqEvaluationMetric::kButteraugli) {
      if (p.uses_butteraugli_sinks) {
        status = AddPlannedPlane(DeviceElementType::kF32,
                                 {p.anchor_capacity_count, 1},
                                 p.anchor_capacity_count, &staging_bytes);
        if (!status.ok())
          return status;
      }
      status =
          AddPlannedPlane(DeviceElementType::kF32, {1, 1}, 1, &staging_bytes);
      if (!status.ok())
        return status;
    } else {
      status = AddPlannedPlane(DeviceElementType::kF32, {3 * block_count, 1},
                               3 * block_count, &staging_bytes);
      if (!status.ok())
        return status;
    }
  }
  status = AddPlannedPlane(DeviceElementType::kF32, {coefficients, 1},
                           coefficients, &staging_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kF32, {coefficients, 1},
                           coefficients, &staging_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kI32, {coefficients, 1},
                           coefficients, &staging_bytes);
  if (!status.ok())
    return status;
  if (!p.frame_only) {
    status = AddPlannedPlane(DeviceElementType::kF32, {coefficients, 1},
                             coefficients, &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kF32, {3 * block_count, 1},
                             3 * block_count, &staging_bytes);
    if (!status.ok())
      return status;
  }
  status = AddPlannedPlane(DeviceElementType::kI32, {3 * block_count, 1},
                           3 * block_count, &staging_bytes);
  if (!status.ok())
    return status;
  status = AddPlannedPlane(DeviceElementType::kI32, {1, 1}, 1, &staging_bytes);
  if (!status.ok())
    return status;
  if (!p.frame_only) {
    status = AddPlannedPlane(DeviceElementType::kF32,
                             {p.maximum_coefficient_count, 1},
                             p.maximum_coefficient_count, &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kI32,
                             {p.maximum_coefficient_count, 1},
                             p.maximum_coefficient_count, &staging_bytes);
    if (!status.ok())
      return status;
    status = AddPlannedPlane(DeviceElementType::kF32,
                             {p.maximum_coefficient_count, 1},
                             p.maximum_coefficient_count, &staging_bytes);
    if (!status.ok())
      return status;
  }

  *capacities = {persistent_bytes, staging_bytes};
  return Status::Ok();
}

} // namespace gjxl::test::storage_plan_oracle
