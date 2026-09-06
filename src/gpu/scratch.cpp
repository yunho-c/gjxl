// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/scratch.h"

#include <algorithm>
#include <limits>

namespace gjxl {

Status DeviceScratchLayoutPlan::AddPlane(
  DeviceElementType element_type, Extent2D extent, size_t row_stride,
  size_t alignment_bytes, DevicePlaneLayout* plane) {
  if (plane == nullptr || alignment_bytes == 0 ||
      (alignment_bytes & (alignment_bytes - 1)) != 0) {
    return Status::InvalidArgument("Device scratch plane plan is invalid");
  }
  const size_t element_size = DeviceElementSize(element_type);
  if (element_size == 0 || alignment_bytes < element_size) {
    return Status::InvalidArgument("Device scratch alignment is invalid");
  }
  size_t size_bytes = 0;
  Status status = ComputeDevicePlaneSizeBytes(element_type, extent, row_stride, &size_bytes);
  if (!status.ok()) return status;
  if (capacity_bytes_ > std::numeric_limits<size_t>::max() - (alignment_bytes - 1)) {
    return Status::InvalidArgument("Device scratch alignment overflows");
  }
  const size_t offset = (capacity_bytes_ + alignment_bytes - 1) & ~(alignment_bytes - 1);
  if (size_bytes > std::numeric_limits<size_t>::max() - offset) {
    return Status::InvalidArgument("Device scratch layout overflows");
  }
  *plane = {element_type, extent, row_stride, offset, size_bytes};
  capacity_bytes_ = offset + size_bytes;
  return Status::Ok();
}

Status DeviceScratchArena::Prepare(
  GpuBackend& backend,
  size_t capacity_bytes) {

  if (capacity_bytes == 0) {
    return Status::InvalidArgument(
      "Device scratch capacity is zero");
  }
  if (buffer_ != nullptr && backend_id_ != backend.id()) {
    return Status::InvalidArgument(
      "Device scratch belongs to another backend");
  }
  if (buffer_ != nullptr && capacity_bytes <= capacity_bytes_) {
    ResetLayout();
    return Status::Ok();
  }

  std::unique_ptr<DeviceBuffer> replacement;
  Status status = backend.Allocate(capacity_bytes, &replacement);
  if (!status.ok()) {
    return status;
  }
  buffer_ = std::move(replacement);
  backend_id_ = backend.id();
  capacity_bytes_ = capacity_bytes;
  layout_bytes_ = 0;
  peak_layout_bytes_ = 0;
  return Status::Ok();
}

void DeviceScratchArena::ResetLayout() noexcept {
  layout_bytes_ = 0;
}

Status DeviceScratchArena::AllocatePlane(
  DeviceElementType element_type,
  Extent2D extent,
  size_t row_stride,
  size_t alignment_bytes,
  DevicePlaneView* view) {

  DeviceScratchLayoutPlan plan(layout_bytes_);
  DevicePlaneLayout plane;
  const Status status = plan.AddPlane(element_type, extent, row_stride, alignment_bytes, &plane);
  if (!status.ok()) return status;
  return BindPlane(plane, view);
}

Status DeviceScratchArena::BindPlane(const DevicePlaneLayout& plane, DevicePlaneView* view) {
  if (view == nullptr || buffer_ == nullptr) {
    return Status::InvalidArgument("Device scratch plane request is invalid");
  }
  DevicePlaneView candidate{
    buffer_.get(), plane.offset_bytes, plane.element_type, plane.extent, plane.row_stride};
  DeviceMemoryRange range;
  const Status status = ComputeDevicePlaneRange(candidate, backend_id_, &range);
  if (!status.ok()) return status;
  if (range.size_bytes != plane.size_bytes) {
    return Status::InvalidArgument("Device scratch planned byte size is inconsistent");
  }
  // Range validation proves the end fits the actual backing, including addition.
  layout_bytes_ = std::max(layout_bytes_, range.offset_bytes + range.size_bytes);
  peak_layout_bytes_ = std::max(peak_layout_bytes_, layout_bytes_);
  *view = candidate;
  return Status::Ok();
}

}  // namespace gjxl
