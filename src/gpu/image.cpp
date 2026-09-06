// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/image.h"

#include <limits>

namespace gjxl {

Status ComputeDevicePlaneSizeBytes(
  DeviceElementType element_type, Extent2D extent, size_t row_stride,
  size_t* size_bytes) {
  if (size_bytes == nullptr || extent.empty() || row_stride < extent.width) {
    return Status::InvalidArgument("Device plane size output or geometry is invalid");
  }
  const size_t element_size = DeviceElementSize(element_type);
  if (element_size == 0) {
    return Status::InvalidArgument("Device plane element type is invalid");
  }
  if (extent.height - 1 > std::numeric_limits<size_t>::max() / row_stride) {
    return Status::InvalidArgument("Device plane row geometry overflows");
  }
  const size_t last_row = (extent.height - 1) * row_stride;
  if (extent.width > std::numeric_limits<size_t>::max() - last_row) {
    return Status::InvalidArgument("Device plane element range overflows");
  }
  const size_t count = last_row + extent.width;
  if (count > std::numeric_limits<size_t>::max() / element_size) {
    return Status::InvalidArgument("Device plane byte range overflows");
  }
  *size_bytes = count * element_size;
  return Status::Ok();
}

Status ComputeDevicePlaneRange(
  ConstDevicePlaneView view,
  BackendId expected_backend,
  DeviceMemoryRange* range) {

  if (range == nullptr) {
    return Status::InvalidArgument(
      "Device plane range output is null");
  }
  if (view.buffer == nullptr || view.extent.empty() ||
      view.row_stride < view.extent.width) {
    return Status::InvalidArgument(
      "Device plane buffer or geometry is invalid");
  }
  if (view.buffer->backend_id() != expected_backend) {
    return Status::InvalidArgument(
      "Device plane belongs to another backend");
  }

  const size_t element_size = DeviceElementSize(view.element_type);
  if (element_size == 0 || view.offset_bytes % element_size != 0) {
    return Status::InvalidArgument(
      "Device plane type or offset alignment is invalid");
  }
  size_t size_bytes = 0;
  const Status status = ComputeDevicePlaneSizeBytes(
    view.element_type, view.extent, view.row_stride, &size_bytes);
  if (!status.ok()) return status;
  if (view.offset_bytes > view.buffer->size_bytes() ||
      size_bytes > view.buffer->size_bytes() - view.offset_bytes) {
    return Status::InvalidArgument(
      "Device plane range exceeds its buffer");
  }

  *range = {view.buffer, view.offset_bytes, size_bytes};
  return Status::Ok();
}

Status ValidateDeviceImage3View(
  ConstDeviceImage3View view,
  BackendId expected_backend) {

  for (ConstDevicePlaneView plane : view.plane) {
    DeviceMemoryRange ignored;
    const Status status = ComputeDevicePlaneRange(
      plane, expected_backend, &ignored);
    if (!status.ok()) {
      return status;
    }
  }
  if (view.plane[0].extent != view.plane[1].extent ||
      view.plane[0].extent != view.plane[2].extent ||
      view.plane[0].element_type != view.plane[1].element_type ||
      view.plane[0].element_type != view.plane[2].element_type) {
    return Status::InvalidArgument(
      "Device image planes have mismatched geometry or type");
  }
  return Status::Ok();
}

bool DeviceRangesOverlap(
  DeviceMemoryRange left,
  DeviceMemoryRange right) noexcept {

  if (left.buffer == nullptr || right.buffer == nullptr ||
      left.buffer != right.buffer || left.size_bytes == 0 ||
      right.size_bytes == 0) {
    return false;
  }
  if (left.offset_bytes <= right.offset_bytes) {
    return right.offset_bytes - left.offset_bytes < left.size_bytes;
  }
  return left.offset_bytes - right.offset_bytes < right.size_bytes;
}

bool DeviceRangesEqual(
  DeviceMemoryRange left,
  DeviceMemoryRange right) noexcept {

  return left.buffer == right.buffer &&
         left.offset_bytes == right.offset_bytes &&
         left.size_bytes == right.size_bytes;
}

}  // namespace gjxl
