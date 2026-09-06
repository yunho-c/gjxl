// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_storage_plan.h"

#include <algorithm>
#include <limits>

#include "codec/vardct_frame.h"

namespace gjxl::metal_internal {
Status ValidateAqStorageGeometry(Extent2D source, Extent2D coding) {
  if (source.empty() || coding.empty() ||
      coding.width % kJxlBlockDimension != 0 ||
      coding.height % kJxlBlockDimension != 0 || source.width > coding.width ||
      source.height > coding.height ||
      coding.width - source.width >= kJxlBlockDimension ||
      coding.height - source.height >= kJxlBlockDimension) {
    return Status::InvalidArgument(
        "Prepared AQ source and padded coding geometry are incompatible");
  }
  const Extent2D blocks{coding.width / kJxlBlockDimension,
                        coding.height / kJxlBlockDimension};
  constexpr size_t kShaderMaximum = std::numeric_limits<uint32_t>::max();
  size_t block_count = 0;
  if (source.width > kShaderMaximum || source.height > kShaderMaximum ||
      coding.width > kShaderMaximum || coding.height > kShaderMaximum ||
      blocks.width > kShaderMaximum / 2 || blocks.height > kShaderMaximum ||
      !blocks.try_area(&block_count) || block_count > kShaderMaximum) {
    return Status::InvalidArgument(
        "Prepared AQ geometry exceeds Metal shader limits");
  }
  return Status::Ok();
}

Status ComputeAqStoragePlan(const AqStoragePlanOptions &options,
                            AqStoragePlan *plan) {
  if (plan == nullptr)
    return Status::InvalidArgument("AQ storage-plan output is null");
  size_t pixel_count = 0;
  Status status =
      ValidateAqStorageGeometry(options.source_extent, options.coding_extent);
  if (!status.ok())
    return status;
  if (!options.coding_extent.try_area(&pixel_count) ||
      pixel_count > std::numeric_limits<uint32_t>::max() / size_t{3}) {
    return Status::InvalidArgument(
        "Prepared AQ coefficient storage exceeds Metal shader limits");
  }
  const Extent2D block_extent{options.coding_extent.width / 8,
                              options.coding_extent.height / 8};
  const Extent2D tile_extent{(options.coding_extent.width + 63) / 64,
                             (options.coding_extent.height + 63) / 64};
  const size_t block_count = pixel_count / 64;
  const size_t coefficient_value_count = 3 * pixel_count;
  if (options.anchor_capacity_count == 0 ||
      options.anchor_capacity_count > block_count ||
      options.filter_scratch_image_count > 2 ||
      (!options.frame_only && options.maximum_coefficient_count == 0) ||
      options.maximum_coefficient_count >
          std::numeric_limits<uint32_t>::max() ||
      (options.frame_only_resident_quantizer &&
       (options.initial_quant_sort_count < block_count ||
        options.initial_quant_sort_count >
            std::numeric_limits<uint32_t>::max())) ||
      (options.metric != AqEvaluationMetric::kButteraugli &&
       options.metric != AqEvaluationMetric::kMaximumError)) {
    return Status::InvalidArgument("AQ storage-plan policy counts are invalid");
  }
  AqStoragePlan candidate;
  DeviceScratchLayoutPlan persistent, staging;
  if (!options.frame_only && !options.borrowed_original_linear_rgb) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status =
          persistent.AddPlane(DeviceElementType::kF32, options.source_extent,
                              options.source_extent.width, kAqStorageAlignment,
                              &candidate.original[channel]);
      if (!status.ok())
        return status;
    }
  }
  if (!options.borrowed_coding_opsin) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status =
          persistent.AddPlane(DeviceElementType::kF32, options.coding_extent,
                              options.coding_extent.width, kAqStorageAlignment,
                              &candidate.coding[channel]);
      if (!status.ok())
        return status;
    }
  }
  if (options.needs_reconstructed) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status =
          persistent.AddPlane(DeviceElementType::kF32, options.coding_extent,
                              options.coding_extent.width, kAqStorageAlignment,
                              &candidate.reconstructed[channel]);
      if (!status.ok())
        return status;
    }
  }
  if (!options.frame_only) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status =
          persistent.AddPlane(DeviceElementType::kF32, options.source_extent,
                              options.source_extent.width, kAqStorageAlignment,
                              &candidate.reconstructed_linear[channel]);
      if (!status.ok())
        return status;
    }
    status = persistent.AddPlane(
        DeviceElementType::kI32, {block_extent.width * 2, block_extent.height},
        block_extent.width * 2, kAqStorageAlignment, &candidate.strategies);
    if (!status.ok())
      return status;
  }
  status = persistent.AddPlane(DeviceElementType::kI32,
                               {2 * options.anchor_capacity_count, 1},
                               2 * options.anchor_capacity_count,
                               kAqStorageAlignment, &candidate.anchors);
  if (!status.ok())
    return status;
  status = persistent.AddPlane(DeviceElementType::kU8, block_extent,
                               block_extent.width, kAqStorageAlignment,
                               &candidate.epf_sharpness);
  if (!status.ok())
    return status;
  status = persistent.AddPlane(
      DeviceElementType::kF32, {kAqQuantTableValueCount, 1},
      kAqQuantTableValueCount, kAqStorageAlignment, &candidate.quant_tables);
  if (!status.ok())
    return status;
  if (options.resident_quantization) {
    status = persistent.AddPlane(DeviceElementType::kI32, {6 * block_count, 1},
                                 6 * block_count, kAqStorageAlignment,
                                 &candidate.color_transform_records);
    if (!status.ok())
      return status;
    status =
        persistent.AddPlane(DeviceElementType::kI32,
                            {tile_extent.width * tile_extent.height + 1, 1},
                            tile_extent.width * tile_extent.height + 1,
                            kAqStorageAlignment, &candidate.color_tile_offsets);
    if (!status.ok())
      return status;
  }

  status = staging.AddPlane(DeviceElementType::kI32, block_extent,
                            block_extent.width, kAqStorageAlignment,
                            &candidate.raw_quant);
  if (!status.ok())
    return status;
  if (!options.frame_only) {
    status = staging.AddPlane(DeviceElementType::kF32, block_extent,
                              block_extent.width, kAqStorageAlignment,
                              &candidate.inverse_sigma);
    if (!status.ok())
      return status;
  }
  if (options.frame_only_resident_initial_quant) {
    const Extent2D pre_erosion_extent{options.coding_extent.width / 4,
                                      options.coding_extent.height / 4};
    status = staging.AddPlane(DeviceElementType::kF32, pre_erosion_extent,
                              pre_erosion_extent.width, kAqStorageAlignment,
                              &candidate.initial_quant_pre_erosion);
    if (!status.ok())
      return status;
    status = staging.AddPlane(DeviceElementType::kF32, options.coding_extent,
                              options.coding_extent.width, kAqStorageAlignment,
                              &candidate.initial_quant_unblurred_pixel_mask);
    if (!status.ok())
      return status;
    status = staging.AddPlane(DeviceElementType::kF32, block_extent,
                              block_extent.width, kAqStorageAlignment,
                              &candidate.initial_quant_field);
    if (!status.ok())
      return status;
    status = staging.AddPlane(DeviceElementType::kF32, block_extent,
                              block_extent.width, kAqStorageAlignment,
                              &candidate.initial_quant_strategy_mask);
    if (!status.ok())
      return status;
    status = staging.AddPlane(DeviceElementType::kF32, options.coding_extent,
                              options.coding_extent.width, kAqStorageAlignment,
                              &candidate.initial_quant_pixel_mask);
    if (!status.ok())
      return status;
    if (options.frame_only_resident_quantizer) {
      status = staging.AddPlane(
          DeviceElementType::kF32, {options.initial_quant_sort_count, 1},
          options.initial_quant_sort_count, kAqStorageAlignment,
          &candidate.initial_quant_sort);
      if (!status.ok())
        return status;
      status = staging.AddPlane(DeviceElementType::kF32, {1, 1}, 1,
                                kAqStorageAlignment,
                                &candidate.initial_quant_median);
      if (!status.ok())
        return status;
      status = staging.AddPlane(DeviceElementType::kI32, {2, 1}, 2,
                                kAqStorageAlignment,
                                &candidate.initial_quantizer_params);
      if (!status.ok())
        return status;
    }
  }
  if (options.resident_quantization) {
    status = staging.AddPlane(DeviceElementType::kF32, block_extent,
                              block_extent.width, kAqStorageAlignment,
                              &candidate.resident_quant_field);
    if (!status.ok())
      return status;
    status = staging.AddPlane(DeviceElementType::kF32, block_extent,
                              block_extent.width, kAqStorageAlignment,
                              &candidate.resident_policy_initial_field);
    if (!status.ok())
      return status;
    status = staging.AddPlane(DeviceElementType::kF32, {5, 1}, 5,
                              kAqStorageAlignment,
                              &candidate.resident_policy_scores);
    if (!status.ok())
      return status;
    status = staging.AddPlane(DeviceElementType::kI32, {256, 1}, 256,
                              kAqStorageAlignment,
                              &candidate.resident_quant_histogram);
    if (!status.ok())
      return status;
    status = staging.AddPlane(DeviceElementType::kI32, {3, 1}, 3,
                              kAqStorageAlignment,
                              &candidate.resident_quant_selection_state);
    if (!status.ok())
      return status;
    status = staging.AddPlane(DeviceElementType::kF32, {2, 1}, 2,
                              kAqStorageAlignment,
                              &candidate.resident_quant_statistics);
    if (!status.ok())
      return status;
    status = staging.AddPlane(DeviceElementType::kI32, {2, 1}, 2,
                              kAqStorageAlignment,
                              &candidate.resident_quantizer_params);
    if (!status.ok())
      return status;
  }
  for (size_t image = 0; image < options.filter_scratch_image_count; ++image) {
    for (size_t channel = 0; channel < 3; ++channel) {
      status =
          staging.AddPlane(DeviceElementType::kF32, options.coding_extent,
                           options.coding_extent.width, kAqStorageAlignment,
                           &candidate.filter_scratch[image][channel]);
      if (!status.ok())
        return status;
    }
  }
  DeviceScratchLayoutPlan &color_arena =
      options.resident_quantization ? persistent : staging;
  status = color_arena.AddPlane(DeviceElementType::kI8, tile_extent,
                                tile_extent.width, kAqStorageAlignment,
                                &candidate.y_to_x);
  if (!status.ok())
    return status;
  status = color_arena.AddPlane(DeviceElementType::kI8, tile_extent,
                                tile_extent.width, kAqStorageAlignment,
                                &candidate.y_to_b);
  if (!status.ok())
    return status;
  if (!options.frame_only) {
    status = staging.AddPlane(DeviceElementType::kF32, block_extent,
                              block_extent.width, kAqStorageAlignment,
                              &candidate.block_distance);
    if (!status.ok())
      return status;
    if (options.metric == AqEvaluationMetric::kButteraugli) {
      if (options.uses_butteraugli_sinks) {
        status = staging.AddPlane(
            DeviceElementType::kF32, {options.anchor_capacity_count, 1},
            options.anchor_capacity_count, kAqStorageAlignment,
            &candidate.score_partials);
        if (!status.ok())
          return status;
      }
      status = staging.AddPlane(DeviceElementType::kF32, {1, 1}, 1,
                                kAqStorageAlignment, &candidate.score);
      if (!status.ok())
        return status;
    } else {
      status = staging.AddPlane(DeviceElementType::kF32, {3 * block_count, 1},
                                3 * block_count, kAqStorageAlignment,
                                &candidate.transform_maximum_error);
      if (!status.ok())
        return status;
    }
  }
  status = staging.AddPlane(
      DeviceElementType::kF32, {coefficient_value_count, 1},
      coefficient_value_count, kAqStorageAlignment, &candidate.gathered_pixels);
  if (!status.ok())
    return status;
  status =
      staging.AddPlane(DeviceElementType::kF32, {coefficient_value_count, 1},
                       coefficient_value_count, kAqStorageAlignment,
                       &candidate.forward_coefficients);
  if (!status.ok())
    return status;
  status =
      staging.AddPlane(DeviceElementType::kI32, {coefficient_value_count, 1},
                       coefficient_value_count, kAqStorageAlignment,
                       &candidate.quantized_coefficients);
  if (!status.ok())
    return status;
  if (!options.frame_only) {
    status =
        staging.AddPlane(DeviceElementType::kF32, {coefficient_value_count, 1},
                         coefficient_value_count, kAqStorageAlignment,
                         &candidate.reconstruction_coefficients);
    if (!status.ok())
      return status;
    status =
        staging.AddPlane(DeviceElementType::kF32, {3 * block_count, 1},
                         3 * block_count, kAqStorageAlignment, &candidate.dc);
    if (!status.ok())
      return status;
  }
  status = staging.AddPlane(DeviceElementType::kI32, {3 * block_count, 1},
                            3 * block_count, kAqStorageAlignment,
                            &candidate.quantized_dc);
  if (!status.ok())
    return status;
  status =
      staging.AddPlane(DeviceElementType::kI32, {1, 1}, 1, kAqStorageAlignment,
                       &candidate.reconstruction_error);
  if (!status.ok())
    return status;
  if (!options.frame_only) {
    status = staging.AddPlane(
        DeviceElementType::kF32, {options.maximum_coefficient_count, 1},
        options.maximum_coefficient_count, kAqStorageAlignment,
        &candidate.quant_probe_input);
    if (!status.ok())
      return status;
    status = staging.AddPlane(
        DeviceElementType::kI32, {options.maximum_coefficient_count, 1},
        options.maximum_coefficient_count, kAqStorageAlignment,
        &candidate.quant_probe_quantized);
    if (!status.ok())
      return status;
    status = staging.AddPlane(
        DeviceElementType::kF32, {options.maximum_coefficient_count, 1},
        options.maximum_coefficient_count, kAqStorageAlignment,
        &candidate.quant_probe_dequantized);
    if (!status.ok())
      return status;
  }

  candidate.persistent_bytes = persistent.capacity_bytes();
  candidate.staging_bytes = staging.capacity_bytes();
  *plan = candidate;
  return Status::Ok();
}

Status ComputeResidentInputStoragePlan(Extent2D source_extent,
                                       Extent2D coding_extent,
                                       ResidentInputStoragePlan *plan) {
  if (plan == nullptr)
    return Status::InvalidArgument(
        "Resident input storage-plan output is null");
  Status status = ValidateAqStorageGeometry(source_extent, coding_extent);
  if (!status.ok())
    return status;
  ResidentInputStoragePlan candidate;
  DeviceScratchLayoutPlan arena;
  for (auto &plane : candidate.original) {
    status = arena.AddPlane(DeviceElementType::kF32, source_extent,
                            source_extent.width, kAqStorageAlignment, &plane);
    if (!status.ok())
      return status;
  }
  for (auto &plane : candidate.coding) {
    status = arena.AddPlane(DeviceElementType::kF32, coding_extent,
                            coding_extent.width, kAqStorageAlignment, &plane);
    if (!status.ok())
      return status;
  }
  status = arena.AddPlane(DeviceElementType::kI32, {4, 1}, 4,
                          kAqStorageAlignment, &candidate.result);
  if (!status.ok())
    return status;
  candidate.capacity_bytes = arena.capacity_bytes();
  *plan = candidate;
  return Status::Ok();
}

Status ComputeButteraugliStoragePlan(Extent2D requested, bool borrowing,
                                     ButteraugliStoragePlan *plan) {
  if (plan == nullptr || requested.empty() ||
      requested.width > std::numeric_limits<uint32_t>::max() ||
      requested.height > std::numeric_limits<uint32_t>::max()) {
    return Status::InvalidArgument(
        "Butteraugli storage-plan output or extent is invalid");
  }
  ButteraugliStoragePlan candidate;
  candidate.expanded = requested.width < 8 || requested.height < 8;
  candidate.working_extent =
      candidate.expanded ? Extent2D{std::max<size_t>(8, requested.width),
                                    std::max<size_t>(8, requested.height)}
                         : requested;
  candidate.xborder = requested.width < 8 ? (8 - requested.width) / 2 : 0;
  candidate.yborder = requested.height < 8 ? (8 - requested.height) / 2 : 0;
  candidate.multiscale =
      !candidate.expanded && requested.width >= 15 && requested.height >= 15;
  size_t requested_area = 0;
  if (!requested.try_area(&requested_area) ||
      requested_area > std::numeric_limits<uint32_t>::max()) {
    return Status::InvalidArgument(
        "Device Butteraugli scratch geometry overflows");
  }
  if (borrowing && !candidate.multiscale) {
    return Status::InvalidArgument(
        "Borrowed Butteraugli scratch requires an unexpanded multiscale image");
  }
  if (candidate.multiscale) {
    candidate.sub_extent = {requested.width / 2 + requested.width % 2,
                            requested.height / 2 + requested.height % 2};
  }
  Status status = ComputeDevicePlaneSizeBytes(
      DeviceElementType::kF32, candidate.working_extent,
      candidate.working_extent.width, &candidate.working_plane_bytes);
  if (!status.ok())
    return status;
  size_t sub_plane_bytes = 0;
  if (candidate.multiscale) {
    status = ComputeDevicePlaneSizeBytes(
        DeviceElementType::kF32, candidate.sub_extent,
        candidate.sub_extent.width, &sub_plane_bytes);
    if (!status.ok())
      return status;
  }
  const size_t partial_count =
      requested_area / kButteraugliReductionWidth +
      static_cast<size_t>(requested_area % kButteraugliReductionWidth != 0);
  DeviceScratchLayoutPlan arena;
  const auto add = [&](Extent2D extent, DevicePlaneLayout *plane) {
    return arena.AddPlane(DeviceElementType::kF32, extent, extent.width,
                          kButteraugliStorageAlignment, plane);
  };
  for (size_t index = 0; index < candidate.planes.size(); ++index) {
    if (borrowing && index >= kButteraugliBorrowedFirstPlane &&
        index < kButteraugliBorrowedFirstPlane + kButteraugliBorrowedPlaneCount)
      continue;
    const Extent2D extent =
        candidate.multiscale && index == kButteraugliFinalStagingPlane
            ? candidate.sub_extent
            : candidate.working_extent;
    status = add(extent, &candidate.planes[index]);
    if (!status.ok())
      return status;
  }
  if (candidate.multiscale) {
    for (auto &plane : candidate.reference_sub) {
      status = add(candidate.sub_extent, &plane);
      if (!status.ok())
        return status;
    }
    status = add(candidate.sub_extent, &candidate.reference_sub_mask);
    if (!status.ok())
      return status;
    status = add(candidate.sub_extent, &candidate.reference_sub_eroded_mask);
    if (!status.ok())
      return status;
  }
  for (auto &plane : candidate.reduction) {
    status = add({partial_count, 1}, &plane);
    if (!status.ok())
      return status;
  }
  for (size_t index = 0; index < candidate.kernels.size(); ++index) {
    status =
        add({kButteraugliKernelSizes[index], 1}, &candidate.kernels[index]);
    if (!status.ok())
      return status;
    candidate.gaussian_kernel_bytes +=
        kButteraugliKernelSizes[index] * sizeof(float);
  }
  candidate.capacity_bytes = arena.capacity_bytes();
  // These subsets cannot exceed the successfully checked complete arena size.
  candidate.cached_reference_bytes =
      (kButteraugliPsychoPlaneCount + 2) *
      (candidate.working_plane_bytes + sub_plane_bytes);
  candidate.peak_comparison_scratch_bytes =
      (kButteraugliWorkingPlaneCount - (kButteraugliPsychoPlaneCount + 2) -
       (borrowing ? kButteraugliBorrowedPlaneCount : 0)) *
          candidate.working_plane_bytes +
      2 * partial_count * sizeof(float);
  if (candidate.multiscale)
    candidate.peak_comparison_scratch_bytes -=
        candidate.working_plane_bytes - sub_plane_bytes;
  *plan = candidate;
  return Status::Ok();
}

Status ComputeCompletedFrameStoragePlan(Extent2D source, Extent2D coding,
                                        size_t anchor_count,
                                        CompletedFrameStoragePlan *plan) {
  if (plan == nullptr) {
    return Status::InvalidArgument(
        "Completed-frame storage-plan output is null");
  }
  Status status = ValidateAqStorageGeometry(source, coding);
  if (!status.ok())
    return status;
  const Extent2D blocks{coding.width / kJxlBlockDimension,
                        coding.height / kJxlBlockDimension};
  size_t block_count = 0;
  if (!blocks.try_area(&block_count) || anchor_count == 0 ||
      anchor_count > block_count) {
    return Status::InvalidArgument(
        "Completed-frame storage-plan anchor count is invalid");
  }
  CompletedFrameStoragePlan candidate;
  candidate.group_extent = blocks.ceil_div(kVarDctAcGroupBlockDimension);
  constexpr size_t kGroupCoefficients = 3 * kVarDctAcGroupCoefficientCapacity;
  if (!candidate.group_extent.try_area(&candidate.group_count) ||
      candidate.group_count >
          std::numeric_limits<uint32_t>::max() / kGroupCoefficients) {
    return Status::InvalidArgument("Completed Metal frame is too large");
  }
  candidate.coefficient_count = candidate.group_count * kGroupCoefficients;
  DeviceScratchLayoutPlan layout;
  status = layout.AddPlane(
      DeviceElementType::kI32, {candidate.coefficient_count, 1},
      candidate.coefficient_count, alignof(int32_t), &candidate.coefficients);
  if (!status.ok())
    return status;
  status =
      layout.AddPlane(DeviceElementType::kI32, {anchor_count, 1}, anchor_count,
                      alignof(int32_t), &candidate.destinations);
  if (!status.ok())
    return status;
  candidate.capacity_bytes = layout.capacity_bytes();
  *plan = candidate;
  return Status::Ok();
}

} // namespace gjxl::metal_internal
