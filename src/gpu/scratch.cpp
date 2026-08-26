// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/scratch.h"

#include <algorithm>
#include <limits>

namespace gjxl {

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

  if (view == nullptr || buffer_ == nullptr || extent.empty() ||
      row_stride < extent.width || alignment_bytes == 0 ||
      (alignment_bytes & (alignment_bytes - 1)) != 0) {
    return Status::InvalidArgument(
      "Device scratch plane request is invalid");
  }
  const size_t element_size = DeviceElementSize(element_type);
  if (element_size == 0 || alignment_bytes < element_size) {
    return Status::InvalidArgument(
      "Device scratch alignment is invalid");
  }
  if (layout_bytes_ >
      std::numeric_limits<size_t>::max() - (alignment_bytes - 1)) {
    return Status::InvalidArgument(
      "Device scratch alignment overflows");
  }
  const size_t aligned_offset =
    (layout_bytes_ + alignment_bytes - 1) & ~(alignment_bytes - 1);
  DevicePlaneView candidate{
    buffer_.get(), aligned_offset, element_type, extent, row_stride};
  DeviceMemoryRange range;
  Status status = ComputeDevicePlaneRange(candidate, backend_id_, &range);
  if (!status.ok()) {
    return status;
  }
  if (aligned_offset > std::numeric_limits<size_t>::max() - range.size_bytes) {
    return Status::InvalidArgument(
      "Device scratch layout overflows");
  }
  layout_bytes_ = aligned_offset + range.size_bytes;
  peak_layout_bytes_ = std::max(peak_layout_bytes_, layout_bytes_);
  *view = candidate;
  return Status::Ok();
}

}  // namespace gjxl
