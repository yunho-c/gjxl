// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>

#include "gpu/ops/aq_evaluation.h"
#include "gpu/scratch.h"

namespace gjxl::metal_internal {

inline constexpr size_t kAqStorageAlignment = 256;
inline constexpr size_t kAqQuantTableValueCount = 11904;

[[nodiscard]] Status ValidateAqStorageGeometry(Extent2D source,
                                               Extent2D coding);

/// Allocation-free input to the exact AQ arena layout. The caller resolves
/// policy flags and strategy counts; an upper bound on anchor capacity may be
/// used before strategy selection. Borrowed source planes need no backing here.
struct AqStoragePlanOptions {
  Extent2D source_extent;
  Extent2D coding_extent;
  size_t anchor_capacity_count = 0;
  size_t maximum_coefficient_count = 0;
  size_t initial_quant_sort_count = 0;
  size_t filter_scratch_image_count = 0;
  bool frame_only = false;
  bool borrowed_original_linear_rgb = false;
  bool borrowed_coding_opsin = false;
  bool needs_reconstructed = false;
  bool frame_only_resident_initial_quant = false;
  bool frame_only_resident_quantizer = false;
  bool resident_quantization = false;
  bool uses_butteraugli_sinks = false;
  AqEvaluationMetric metric = AqEvaluationMetric::kButteraugli;
};

struct AqStoragePlan {
  size_t persistent_bytes = 0;
  size_t staging_bytes = 0;
  std::array<DevicePlaneLayout, 3> original;
  std::array<DevicePlaneLayout, 3> coding;
  std::array<DevicePlaneLayout, 3> reconstructed;
  std::array<DevicePlaneLayout, 3> reconstructed_linear;
  DevicePlaneLayout strategies;
  DevicePlaneLayout anchors;
  DevicePlaneLayout epf_sharpness;
  DevicePlaneLayout quant_tables;
  DevicePlaneLayout color_transform_records;
  DevicePlaneLayout color_tile_offsets;
  DevicePlaneLayout raw_quant;
  DevicePlaneLayout inverse_sigma;
  DevicePlaneLayout initial_quant_pre_erosion;
  DevicePlaneLayout initial_quant_unblurred_pixel_mask;
  DevicePlaneLayout initial_quant_field;
  DevicePlaneLayout initial_quant_strategy_mask;
  DevicePlaneLayout initial_quant_pixel_mask;
  DevicePlaneLayout initial_quant_sort;
  DevicePlaneLayout initial_quant_median;
  DevicePlaneLayout initial_quantizer_params;
  DevicePlaneLayout resident_quant_field;
  DevicePlaneLayout resident_policy_initial_field;
  DevicePlaneLayout resident_policy_scores;
  DevicePlaneLayout resident_quant_histogram;
  DevicePlaneLayout resident_quant_selection_state;
  DevicePlaneLayout resident_quant_statistics;
  DevicePlaneLayout resident_quantizer_params;
  std::array<std::array<DevicePlaneLayout, 3>, 2> filter_scratch;
  DevicePlaneLayout y_to_x;
  DevicePlaneLayout y_to_b;
  DevicePlaneLayout block_distance;
  DevicePlaneLayout score_partials;
  DevicePlaneLayout score;
  DevicePlaneLayout transform_maximum_error;
  DevicePlaneLayout gathered_pixels;
  DevicePlaneLayout forward_coefficients;
  DevicePlaneLayout quantized_coefficients;
  DevicePlaneLayout reconstruction_coefficients;
  DevicePlaneLayout dc;
  DevicePlaneLayout quantized_dc;
  DevicePlaneLayout reconstruction_error;
  DevicePlaneLayout quant_probe_input;
  DevicePlaneLayout quant_probe_quantized;
  DevicePlaneLayout quant_probe_dequantized;
  bool operator==(const AqStoragePlan &) const = default;
};

/// Computes the sizes AND slices that actual allocation binds. Does no image
/// access or backend initialization. Successful planning allocates no heap
/// storage; an error Status may allocate its small diagnostic string. Failure
/// leaves output unchanged. Empty fields denote absent/borrowed planes.
[[nodiscard]] Status ComputeAqStoragePlan(const AqStoragePlanOptions &options,
                                          AqStoragePlan *plan);

struct ResidentInputStoragePlan {
  size_t capacity_bytes = 0;
  std::array<DevicePlaneLayout, 3> original;
  std::array<DevicePlaneLayout, 3> coding;
  DevicePlaneLayout result;
  bool operator==(const ResidentInputStoragePlan &) const = default;
};
[[nodiscard]] Status
ComputeResidentInputStoragePlan(Extent2D source_extent, Extent2D coding_extent,
                                ResidentInputStoragePlan *plan);

struct CompletedFrameStoragePlan {
  Extent2D group_extent;
  size_t group_count = 0;
  size_t coefficient_count = 0;
  size_t capacity_bytes = 0;
  DevicePlaneLayout coefficients;
  DevicePlaneLayout destinations;
  bool operator==(const CompletedFrameStoragePlan &) const = default;
};

/// The final group-major coefficients and destination table share one
/// independent allocation. Before strategy selection, block count bounds anchor
/// capacity; actual output generation supplies its authoritative final anchor
/// count.
[[nodiscard]] Status
ComputeCompletedFrameStoragePlan(Extent2D source_extent, Extent2D coding_extent,
                                 size_t anchor_count,
                                 CompletedFrameStoragePlan *plan);

inline constexpr size_t kButteraugliStorageAlignment = 64;
inline constexpr size_t kButteraugliReductionWidth = 256;
inline constexpr size_t kButteraugliWorkingPlaneCount = 33;
inline constexpr size_t kButteraugliPsychoPlaneCount = 10;
inline constexpr size_t kButteraugliBorrowedFirstPlane = 21;
inline constexpr size_t kButteraugliBorrowedPlaneCount = 9;
inline constexpr size_t kButteraugliFinalStagingPlane = 32;
inline constexpr std::array<size_t, 5> kButteraugliKernelSizes{5, 33, 15, 7,
                                                               13};

struct ButteraugliStoragePlan {
  size_t capacity_bytes = 0;
  Extent2D working_extent;
  Extent2D sub_extent;
  bool expanded = false;
  bool multiscale = false;
  size_t xborder = 0;
  size_t yborder = 0;
  size_t working_plane_bytes = 0;
  size_t cached_reference_bytes = 0;
  size_t gaussian_kernel_bytes = 0;
  size_t peak_comparison_scratch_bytes = 0;
  std::array<DevicePlaneLayout, kButteraugliWorkingPlaneCount> planes;
  std::array<DevicePlaneLayout, kButteraugliPsychoPlaneCount> reference_sub;
  DevicePlaneLayout reference_sub_mask;
  DevicePlaneLayout reference_sub_eroded_mask;
  std::array<DevicePlaneLayout, 2> reduction;
  std::array<DevicePlaneLayout, kButteraugliKernelSizes.size()> kernels;
  bool operator==(const ButteraugliStoragePlan &) const = default;
};

/// Borrowed planes are represented by absent slices and excluded from capacity.
/// The actual preparer must separately validate their backend, size and
/// aliasing.
[[nodiscard]] Status
ComputeButteraugliStoragePlan(Extent2D requested, bool borrowing,
                              ButteraugliStoragePlan *plan);

} // namespace gjxl::metal_internal
