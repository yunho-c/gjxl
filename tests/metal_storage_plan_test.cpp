// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "core/managed_allocator.h"
#include "gpu/metal/metal_storage_plan.h"
#include "metal_storage_plan_oracle.h"

namespace {
using namespace gjxl;
using namespace gjxl::metal_internal;
using namespace gjxl::resource_budget_internal;

bool Check(bool good, const char *message) {
  if (!good)
    std::cerr << message << '\n';
  return good;
}
bool Ok(const Status &status) {
  if (!status.ok())
    std::cerr << status.message() << '\n';
  return status.ok();
}

bool CheckSlices(std::vector<DevicePlaneLayout> planes, size_t alignment,
                 size_t capacity) {
  planes.erase(std::remove_if(planes.begin(), planes.end(),
                              [](const auto &p) {
                                return p.extent.empty() && p.size_bytes == 0;
                              }),
               planes.end());
  std::sort(planes.begin(), planes.end(), [](const auto &a, const auto &b) {
    return a.offset_bytes < b.offset_bytes;
  });
  size_t cursor = 0;
  for (const auto &plane : planes) {
    const size_t element = plane.element_type == DeviceElementType::kF32 ||
                                   plane.element_type == DeviceElementType::kI32
                               ? 4
                               : 1;
    const size_t expected_offset =
        (cursor + alignment - 1) / alignment * alignment;
    const size_t expected_bytes =
        ((plane.extent.height - 1) * plane.row_stride + plane.extent.width) *
        element;
    if (!Check(!plane.extent.empty() &&
                   plane.row_stride >= plane.extent.width &&
                   plane.offset_bytes == expected_offset &&
                   plane.size_bytes == expected_bytes,
               "Planned slices overlap, have a gap, or disagree with their "
               "geometry"))
      return false;
    cursor = plane.offset_bytes + plane.size_bytes;
  }
  return Check(cursor == capacity,
               "Planned slices do not cover the claimed arena capacity");
}

bool CheckAqSlices(const AqStoragePlanOptions &p, const AqStoragePlan &plan) {
  std::vector<DevicePlaneLayout> persistent, staging;
  for (const auto &plane : plan.original)
    persistent.push_back(plane);
  for (const auto &plane : plan.coding)
    persistent.push_back(plane);
  for (const auto &plane : plan.reconstructed)
    persistent.push_back(plane);
  for (const auto &plane : plan.reconstructed_linear)
    persistent.push_back(plane);
  persistent.push_back(plan.strategies);
  persistent.push_back(plan.anchors);
  persistent.push_back(plan.epf_sharpness);
  persistent.push_back(plan.quant_tables);
  persistent.push_back(plan.color_transform_records);
  persistent.push_back(plan.color_tile_offsets);
  staging.push_back(plan.raw_quant);
  staging.push_back(plan.inverse_sigma);
  staging.push_back(plan.initial_quant_pre_erosion);
  staging.push_back(plan.initial_quant_unblurred_pixel_mask);
  staging.push_back(plan.initial_quant_field);
  staging.push_back(plan.initial_quant_strategy_mask);
  staging.push_back(plan.initial_quant_pixel_mask);
  staging.push_back(plan.initial_quant_sort);
  staging.push_back(plan.initial_quant_median);
  staging.push_back(plan.initial_quantizer_params);
  staging.push_back(plan.resident_quant_field);
  staging.push_back(plan.resident_policy_initial_field);
  staging.push_back(plan.resident_policy_scores);
  staging.push_back(plan.resident_quant_histogram);
  staging.push_back(plan.resident_quant_selection_state);
  staging.push_back(plan.resident_quant_statistics);
  staging.push_back(plan.resident_quantizer_params);
  for (const auto &image : plan.filter_scratch)
    for (const auto &plane : image)
      staging.push_back(plane);
  (p.resident_quantization ? persistent : staging).push_back(plan.y_to_x);
  (p.resident_quantization ? persistent : staging).push_back(plan.y_to_b);
  staging.push_back(plan.block_distance);
  staging.push_back(plan.score_partials);
  staging.push_back(plan.score);
  staging.push_back(plan.transform_maximum_error);
  staging.push_back(plan.gathered_pixels);
  staging.push_back(plan.forward_coefficients);
  staging.push_back(plan.quantized_coefficients);
  staging.push_back(plan.reconstruction_coefficients);
  staging.push_back(plan.dc);
  staging.push_back(plan.quantized_dc);
  staging.push_back(plan.reconstruction_error);
  staging.push_back(plan.quant_probe_input);
  staging.push_back(plan.quant_probe_quantized);
  staging.push_back(plan.quant_probe_dequantized);

  return CheckSlices(std::move(persistent), 256, plan.persistent_bytes) &&
         CheckSlices(std::move(staging), 256, plan.staging_bytes);
}

bool CheckAqAndInputMatrix() {
  size_t cases = 0;
  for (const Extent2D source : {Extent2D{1, 1},
                                {17, 9},
                                {65, 63},
                                {1919, 1079},
                                {3839, 2159},
                                {3840, 2160}}) {
    const Extent2D coding{(source.width + 7) / 8 * 8,
                          (source.height + 7) / 8 * 8};
    const size_t blocks = coding.width * coding.height / 64;
    ResidentInputStoragePlan input;
    if (!Ok(ComputeResidentInputStoragePlan(source, coding, &input)))
      return false;
    std::vector<DevicePlaneLayout> input_planes;
    input_planes.insert(input_planes.end(), input.original.begin(),
                        input.original.end());
    input_planes.insert(input_planes.end(), input.coding.begin(),
                        input.coding.end());
    input_planes.push_back(input.result);
    size_t input_capacity = 0;
    const auto add = [&](size_t bytes) {
      input_capacity = (input_capacity + 255) / 256 * 256 + bytes;
    };
    for (size_t c = 0; c < 3; ++c)
      add(source.width * source.height * 4);
    for (size_t c = 0; c < 3; ++c)
      add(coding.width * coding.height * 4);
    add(16);
    if (!Check(input.capacity_bytes == input_capacity,
               "Resident input capacity differs from frozen recipe") ||
        !CheckSlices(std::move(input_planes), 256, input.capacity_bytes))
      return false;

    for (size_t flags = 0; flags < 1024; ++flags) {
      for (size_t filters = 0; filters <= 2; ++filters) {
        const AqStoragePlanOptions p{
            .source_extent = source,
            .coding_extent = coding,
            .anchor_capacity_count = flags & 256 ? (blocks + 1) / 2 : blocks,
            .maximum_coefficient_count = 1024,
            .initial_quant_sort_count = std::bit_ceil(blocks),
            .filter_scratch_image_count = filters,
            .frame_only = (flags & 1) != 0,
            .borrowed_original_linear_rgb = (flags & 2) != 0,
            .borrowed_coding_opsin = (flags & 4) != 0,
            .needs_reconstructed = (flags & 8) != 0,
            .frame_only_resident_initial_quant = (flags & 16) != 0,
            .frame_only_resident_quantizer = (flags & 32) != 0,
            .resident_quantization = (flags & 64) != 0,
            .uses_butteraugli_sinks = (flags & 128) != 0,
            .metric = flags & 512 ? AqEvaluationMetric::kMaximumError
                                  : AqEvaluationMetric::kButteraugli,
        };
        AqStoragePlan plan;
        std::pair<size_t, size_t> expected;
        if (!Ok(ComputeAqStoragePlan(p, &plan)) ||
            !Ok(test::storage_plan_oracle::AqCapacity(p, &expected)) ||
            !Check(
                plan.persistent_bytes == expected.first &&
                    plan.staging_bytes == expected.second,
                "AQ capacities differ from the frozen pre-extraction recipe") ||
            !CheckAqSlices(p, plan))
          return false;
        ++cases;
      }
    }
  }
  std::cout << "Frozen AQ capacity/layout cases: " << cases << '\n';
  return true;
}

bool CheckButteraugliMatrix() {
  for (const Extent2D extent : {Extent2D{1, 1},
                                {7, 19},
                                {8, 8},
                                {14, 15},
                                {15, 15},
                                {65, 63},
                                {1919, 1079},
                                {3839, 2159}}) {
    for (bool borrowing : {false, true}) {
      ButteraugliStoragePlan plan;
      plan.capacity_bytes = 123;
      const auto previous = plan;
      const auto status =
          ComputeButteraugliStoragePlan(extent, borrowing, &plan);
      const bool multiscale = extent.width >= 15 && extent.height >= 15;
      if (borrowing && !multiscale) {
        if (!Check(status.code() == StatusCode::kInvalidArgument &&
                       plan == previous,
                   "Invalid borrowed Butteraugli layout changed output"))
          return false;
        continue;
      }
      if (!Ok(status))
        return false;
      const size_t full = std::max<size_t>(8, extent.width) *
                          std::max<size_t>(8, extent.height) * 4;
      const size_t sub =
          multiscale ? ((extent.width + 1) / 2) * ((extent.height + 1) / 2) * 4
                     : 0;
      const size_t partials = (extent.width * extent.height + 255) / 256;
      size_t expected = 0;
      const auto add = [&](size_t bytes) {
        expected = (expected + 63) / 64 * 64 + bytes;
      };
      for (size_t i = 0; i < 33; ++i) {
        if (borrowing && i >= 21 && i < 30)
          continue;
        add(multiscale && i == 32 ? sub : full);
      }
      if (multiscale)
        for (size_t i = 0; i < 12; ++i)
          add(sub);
      add(partials * 4);
      add(partials * 4);
      for (size_t taps : {5, 33, 15, 7, 13})
        add(taps * 4);
      std::vector<DevicePlaneLayout> planes(plan.planes.begin(),
                                            plan.planes.end());
      planes.insert(planes.end(), plan.reference_sub.begin(),
                    plan.reference_sub.end());
      planes.push_back(plan.reference_sub_mask);
      planes.push_back(plan.reference_sub_eroded_mask);
      planes.insert(planes.end(), plan.reduction.begin(), plan.reduction.end());
      planes.insert(planes.end(), plan.kernels.begin(), plan.kernels.end());
      if (!Check(plan.capacity_bytes == expected &&
                     plan.cached_reference_bytes == 12 * (full + sub) &&
                     plan.gaussian_kernel_bytes == 292 &&
                     plan.peak_comparison_scratch_bytes ==
                         (21 - (borrowing ? 9 : 0)) * full + 8 * partials -
                             (multiscale ? full - sub : 0),
                 "Butteraugli capacity/metrics differ from frozen recipe") ||
          !CheckSlices(std::move(planes), 64, expected))
        return false;
      for (size_t i = 0; i < 33; ++i) {
        const bool absent = borrowing && i >= 21 && i < 30;
        if (!Check(plan.planes[i].extent.empty() == absent,
                   "Borrowed BA plane was allocated"))
          return false;
      }
    }
  }
  return true;
}

bool CheckCompletedFrames() {
  size_t cases = 0;
  for (size_t height : {size_t{1}, 8ul, 255ul, 256ul, 257ul, 1080ul, 2160ul}) {
    for (size_t width : {size_t{1}, 8ul, 255ul, 256ul, 257ul, 1920ul, 3840ul}) {
      const Extent2D source{width, height};
      const Extent2D coding{(width + 7) / 8 * 8, (height + 7) / 8 * 8};
      const size_t blocks = coding.width * coding.height / 64;
      for (size_t anchors : {size_t{1}, (blocks + 1) / 2, blocks}) {
        CompletedFrameStoragePlan plan;
        if (!Ok(ComputeCompletedFrameStoragePlan(source, coding, anchors,
                                                 &plan)))
          return false;
        const Extent2D groups{(width + 255) / 256, (height + 255) / 256};
        const size_t count = groups.width * groups.height;
        const size_t coefficients = count * 3 * 65536;
        if (!Check(plan.group_extent == groups && plan.group_count == count &&
                       plan.coefficient_count == coefficients &&
                       plan.capacity_bytes == (coefficients + anchors) * 4 &&
                       plan.destinations.offset_bytes == coefficients * 4 &&
                       plan.coefficients.extent == Extent2D{coefficients, 1} &&
                       plan.destinations.extent == Extent2D{anchors, 1},
                   "Completed frame layout differs from frozen group-major "
                   "recipe") ||
            !CheckSlices({plan.coefficients, plan.destinations}, 4,
                         plan.capacity_bytes))
          return false;
        ++cases;
      }
    }
  }
  CompletedFrameStoragePlan plan;
  // Greatest group count accepted by the unchanged shader indexing bound.
  const size_t maximum_width = 21845 * 256;
  if (!Ok(ComputeCompletedFrameStoragePlan({maximum_width, 256},
                                           {maximum_width, 256}, 1, &plan)))
    return false;
  const auto before = plan;
  for (Extent2D bad :
       {Extent2D{maximum_width + 256, 256}, {0, 256}, {65536, 65536}}) {
    if (!Check(ComputeCompletedFrameStoragePlan(bad, bad, 1, &plan).code() ==
                       StatusCode::kInvalidArgument &&
                   plan == before,
               "Completed frame indexing failure was not atomic"))
      return false;
  }
  for (size_t bad :
       {size_t{0}, size_t{2}, std::numeric_limits<size_t>::max()}) {
    if (!Check(!ComputeCompletedFrameStoragePlan({8, 8}, {8, 8}, bad, &plan)
                       .ok() &&
                   plan == before,
               "Completed frame anchor failure was not atomic"))
      return false;
  }
  if (!Check(!ComputeCompletedFrameStoragePlan({8, 8}, {8, 8}, 1, nullptr).ok(),
             "Null completed frame plan accepted"))
    return false;
  std::cout << "Frozen completed-frame layout cases: " << cases << '\n';
  return true;
}

bool CheckFailureAndNoBacking() {
  AqStoragePlanOptions p{.source_extent = {17, 9},
                         .coding_extent = {24, 16},
                         .anchor_capacity_count = 6,
                         .maximum_coefficient_count = 1024};
  AqStoragePlan plan;
  plan.persistent_bytes = 123;
  plan.original[0].size_bytes = 77;
  const auto previous = plan;
  for (size_t invalid = 0; invalid < 7; ++invalid) {
    auto bad = p;
    switch (invalid) {
    case 0:
      bad.coding_extent = {24, 15};
      break;
    case 1:
      bad.source_extent = {0, 9};
      break;
    case 2:
      bad.source_extent = {16, 9};
      break; // Padding is too large.
    case 3:
      bad.anchor_capacity_count = 7;
      break;
    case 4:
      bad.filter_scratch_image_count = 3;
      break;
    case 5:
      bad.maximum_coefficient_count = std::numeric_limits<size_t>::max();
      break;
    case 6:
      bad.coding_extent = {size_t{1} << 32, 8};
      break;
    }
    if (!Check(ComputeAqStoragePlan(bad, &plan).code() ==
                       StatusCode::kInvalidArgument &&
                   plan == previous,
               "AQ planning failure was not atomic"))
      return false;
  }
  if (!Check(ComputeAqStoragePlan(p, nullptr).code() ==
                 StatusCode::kInvalidArgument,
             "Null AQ plan was accepted"))
    return false;
  ResidentInputStoragePlan input;
  input.capacity_bytes = 99;
  const auto input_before = input;
  if (!Check(!ComputeResidentInputStoragePlan({17, 9}, {16, 16}, &input).ok() &&
                 input == input_before,
             "Resident input planning failure was not atomic"))
    return false;
  ButteraugliStoragePlan ba;
  CompletedFrameStoragePlan completed;
  ba.capacity_bytes = 33;
  const auto ba_before = ba;
  for (Extent2D invalid :
       {Extent2D{0, 1}, {size_t{1} << 32, 1}, {65536, 65536}})
    if (!Check(!ComputeButteraugliStoragePlan(invalid, false, &ba).ok() &&
                   ba == ba_before,
               "Butteraugli planning overflow/failure was not atomic"))
      return false;

  ResourceBudget budget(1);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(1, &job)))
    return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    ArmNextManagedHostAllocationFailureForTest();
    const bool success =
        ComputeAqStoragePlan(p, &plan).ok() &&
        ComputeResidentInputStoragePlan({3839, 2159}, {3840, 2160}, &input)
            .ok() &&
        ComputeButteraugliStoragePlan({3839, 2159}, true, &ba).ok() &&
        ComputeCompletedFrameStoragePlan({3839, 2159}, {3840, 2160}, 129600,
                                         &completed)
            .ok();
    const bool untouched = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!Check(success && untouched &&
                   budget.snapshot().peak_backing_bytes == 0,
               "Geometry planning consumed managed backing"))
      return false;
  }
  job.Reset();
  return Check(budget.snapshot().committed_bytes() == 0,
               "Planning retained a reservation");
}
} // namespace

int main() {
  return CheckAqAndInputMatrix() && CheckButteraugliMatrix() &&
                 CheckCompletedFrames() && CheckFailureAndNoBacking() &&
                 Check(DefaultResourceBudget().snapshot().peak_backing_bytes ==
                           0,
                       "Planner escaped to default domain")
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
