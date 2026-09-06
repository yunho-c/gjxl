// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/ops/butteraugli.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
#include "core/managed_allocator.h"

namespace gjxl {
using resource_budget_internal::ManagedVector;

namespace {

[[nodiscard]] bool ValidOptions(ButteraugliOptions options) noexcept {
  return std::isfinite(options.hf_asymmetry) &&
         options.hf_asymmetry > 0.0f &&
         std::isfinite(options.x_multiplier) &&
         options.x_multiplier > 0.0f &&
         std::isfinite(options.intensity_target) &&
         options.intensity_target > 0.0f;
}

[[nodiscard]] Status ValidateFloatImage(
  ConstDeviceImage3View image,
  BackendId backend_id) {

  Status status = ValidateDeviceImage3View(image, backend_id);
  if (!status.ok()) {
    return status;
  }
  if (image.plane[0].element_type != DeviceElementType::kF32) {
    return Status::InvalidArgument(
      "Device Butteraugli images must contain float32 planes");
  }
  return Status::Ok();
}

[[nodiscard]] bool ShouldInvalidate(const Status& status) noexcept {
  return !status.ok() && status.code() != StatusCode::kInvalidArgument;
}

[[nodiscard]] Status ReadScoreValue(
  GpuBackend& backend,
  ConstDevicePlaneView score_view,
  double* score) {

  if (score == nullptr) {
    return Status::InvalidArgument(
      "Device Butteraugli score output is null");
  }
  DeviceMemoryRange range;
  Status status = ComputeDevicePlaneRange(
    score_view, backend.id(), &range);
  if (!status.ok()) {
    return status;
  }
  if (score_view.element_type != DeviceElementType::kF32 ||
      score_view.extent != Extent2D{1, 1}) {
    return Status::InvalidArgument(
      "Device Butteraugli score must be one float32 value");
  }

  float candidate = 0.0f;
  status = backend.CopyDeviceToHost(
    *score_view.buffer, &candidate, sizeof(candidate),
    score_view.offset_bytes);
  if (!status.ok()) {
    return status;
  }
  if (!std::isfinite(candidate) || candidate < 0.0f) {
    return Status::Internal(
      "Device Butteraugli score is not finite and non-negative");
  }
  *score = static_cast<double>(candidate);
  return Status::Ok();
}

[[nodiscard]] Status ReadDistanceMapValue(
  GpuBackend& backend,
  ConstDevicePlaneView device_map,
  PlaneF32View host_map) {

  DeviceMemoryRange range;
  Status status = ComputeDevicePlaneRange(
    device_map, backend.id(), &range);
  if (!status.ok()) {
    return status;
  }
  if (device_map.element_type != DeviceElementType::kF32 ||
      !host_map.valid() || host_map.extent != device_map.extent) {
    return Status::InvalidArgument(
      "Device Butteraugli diagnostic map has invalid geometry or type");
  }

  size_t pixel_count = 0;
  if (!device_map.extent.try_area(&pixel_count)) {
    return Status::InvalidArgument(
      "Device Butteraugli diagnostic map dimensions overflow");
  }

  try {
    ManagedVector<float> candidate(pixel_count);
    const size_t row_bytes = device_map.extent.width * sizeof(float);
    for (size_t y = 0; y < device_map.extent.height; ++y) {
      const size_t device_offset = device_map.offset_bytes +
        y * device_map.row_stride * sizeof(float);
      status = backend.CopyDeviceToHost(
        *device_map.buffer,
        candidate.data() + y * device_map.extent.width,
        row_bytes,
        device_offset);
      if (!status.ok()) {
        return status;
      }
    }
    if (!std::ranges::all_of(candidate, [](float value) {
          return std::isfinite(value) && value >= 0.0f;
        })) {
      return Status::Internal(
        "Device Butteraugli map is not finite and non-negative");
    }
    for (size_t y = 0; y < host_map.extent.height; ++y) {
      std::copy_n(
        candidate.data() + y * host_map.extent.width,
        host_map.extent.width,
        host_map.Row(y));
    }
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate Device Butteraugli diagnostic readback");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Device Butteraugli diagnostic map is too large");
  }
  return Status::Ok();
}

}  // namespace

Status ValidateDeviceButteraugliPrepareDescriptor(
  const GpuBackend& backend,
  const DeviceButteraugliPrepareDescriptor& descriptor) {

  Status status = ValidateFloatImage(
    descriptor.reference_linear_rgb, backend.id());
  if (!status.ok()) {
    return status;
  }
  size_t pixel_count = 0;
  if (!descriptor.reference_linear_rgb.plane[0].extent.try_area(
        &pixel_count) ||
      !ValidOptions(descriptor.options)) {
    return Status::InvalidArgument(
      "Device Butteraugli reference extent or options are invalid");
  }
  return Status::Ok();
}

Status ValidateDeviceButteraugliComparisonDescriptor(
  const GpuBackend& backend,
  ConstDeviceImage3View reference_linear_rgb,
  Extent2D prepared_extent,
  const DeviceButteraugliComparisonDescriptor& descriptor) {

  if (prepared_extent.empty() ||
      reference_linear_rgb.plane[0].extent != prepared_extent) {
    return Status::InvalidArgument(
      "Device Butteraugli prepared extent is invalid");
  }
  Status status = ValidateFloatImage(
    reference_linear_rgb, backend.id());
  if (!status.ok()) {
    return status;
  }
  status = ValidateFloatImage(
    descriptor.distorted_linear_rgb, backend.id());
  if (!status.ok()) {
    return status;
  }
  if (descriptor.distorted_linear_rgb.plane[0].extent != prepared_extent) {
    return Status::InvalidArgument(
      "Device Butteraugli images have different extents");
  }

  DeviceMemoryRange distance_range;
  status = ComputeDevicePlaneRange(
    descriptor.distance_map, backend.id(), &distance_range);
  if (!status.ok()) {
    return status;
  }
  DeviceMemoryRange score_range;
  status = ComputeDevicePlaneRange(
    descriptor.score, backend.id(), &score_range);
  if (!status.ok()) {
    return status;
  }
  if (descriptor.distance_map.element_type != DeviceElementType::kF32 ||
      descriptor.distance_map.extent != prepared_extent ||
      descriptor.score.element_type != DeviceElementType::kF32 ||
      descriptor.score.extent != Extent2D{1, 1}) {
    return Status::InvalidArgument(
      "Device Butteraugli output geometry or type is invalid");
  }
  if (DeviceRangesOverlap(distance_range, score_range)) {
    return Status::InvalidArgument(
      "Device Butteraugli outputs overlap");
  }

  for (ConstDevicePlaneView input : reference_linear_rgb.plane) {
    DeviceMemoryRange input_range;
    status = ComputeDevicePlaneRange(input, backend.id(), &input_range);
    if (!status.ok()) {
      return status;
    }
    if (DeviceRangesOverlap(input_range, distance_range) ||
        DeviceRangesOverlap(input_range, score_range)) {
      return Status::InvalidArgument(
        "Device Butteraugli output overlaps the reference image");
    }
  }
  for (ConstDevicePlaneView input : descriptor.distorted_linear_rgb.plane) {
    DeviceMemoryRange input_range;
    status = ComputeDevicePlaneRange(input, backend.id(), &input_range);
    if (!status.ok()) {
      return status;
    }
    if (DeviceRangesOverlap(input_range, distance_range) ||
        DeviceRangesOverlap(input_range, score_range)) {
      return Status::InvalidArgument(
        "Device Butteraugli output overlaps the distorted image");
    }
  }
  return Status::Ok();
}

PreparedDeviceButteraugli::PreparedDeviceButteraugli(
  GpuBackend& backend,
  DeviceButteraugliPrepareDescriptor descriptor)
  : backend_(&backend),
    reference_linear_rgb_(descriptor.reference_linear_rgb),
    extent_(descriptor.reference_linear_rgb.plane[0].extent),
    options_(descriptor.options) {}

BackendId PreparedDeviceButteraugli::backend_id() const noexcept {
  return backend_->id();
}

Extent2D PreparedDeviceButteraugli::extent() const noexcept {
  return extent_;
}

ButteraugliOptions PreparedDeviceButteraugli::options() const noexcept {
  return options_;
}

bool PreparedDeviceButteraugli::valid() const noexcept {
  return valid_.load(std::memory_order_acquire);
}

GpuBackend& PreparedDeviceButteraugli::backend() const noexcept {
  return *backend_;
}

ConstDeviceImage3View
PreparedDeviceButteraugli::reference_linear_rgb() const noexcept {
  return reference_linear_rgb_;
}

bool PreparedDeviceButteraugli::BeginCall() noexcept {
  return !active_.test_and_set(std::memory_order_acquire);
}

void PreparedDeviceButteraugli::EndCall() noexcept {
  active_.clear(std::memory_order_release);
}

void PreparedDeviceButteraugli::Invalidate() noexcept {
  has_result_ = false;
  valid_.store(false, std::memory_order_release);
}

Status PreparedDeviceButteraugli::Compare(
  const DeviceButteraugliComparisonDescriptor& descriptor) {

  if (!valid()) {
    return Status::InvalidArgument(
      "Prepared Device Butteraugli state is invalid");
  }
  Status status = ValidateDeviceButteraugliComparisonDescriptor(
    *backend_, reference_linear_rgb_, extent_, descriptor);
  if (!status.ok()) {
    return status;
  }
  if (!BeginCall()) {
    return Status::InvalidArgument(
      "Prepared Device Butteraugli state is already in use");
  }
  if (!valid()) {
    EndCall();
    return Status::InvalidArgument(
      "Prepared Device Butteraugli state is invalid");
  }

  status = CompareValidated(descriptor);
  if (status.ok()) {
    last_distance_map_ = descriptor.distance_map;
    last_score_ = descriptor.score;
    has_result_ = true;
  } else if (ShouldInvalidate(status)) {
    Invalidate();
  }
  EndCall();
  return status;
}

Status PreparedDeviceButteraugli::ReadScore(double* score) {
  if (!valid()) {
    return Status::InvalidArgument(
      "Prepared Device Butteraugli has no readable result");
  }
  if (!BeginCall()) {
    return Status::InvalidArgument(
      "Prepared Device Butteraugli state is already in use");
  }
  if (!valid() || !has_result_) {
    EndCall();
    return Status::InvalidArgument(
      "Prepared Device Butteraugli has no readable result");
  }
  Status status = ReadScoreValue(*backend_, last_score_, score);
  if (ShouldInvalidate(status)) {
    Invalidate();
  }
  EndCall();
  return status;
}

Status PreparedDeviceButteraugli::ReadDistanceMap(
  PlaneF32View distance_map) {

  if (!valid()) {
    return Status::InvalidArgument(
      "Prepared Device Butteraugli has no readable result");
  }
  if (!BeginCall()) {
    return Status::InvalidArgument(
      "Prepared Device Butteraugli state is already in use");
  }
  if (!valid() || !has_result_) {
    EndCall();
    return Status::InvalidArgument(
      "Prepared Device Butteraugli has no readable result");
  }
  Status status = ReadDistanceMapValue(
    *backend_, last_distance_map_, distance_map);
  if (ShouldInvalidate(status)) {
    Invalidate();
  }
  EndCall();
  return status;
}

}  // namespace gjxl
