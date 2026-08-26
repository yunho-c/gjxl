// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/image.h"

#include <limits>

namespace gjxl {

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
  if (view.extent.height - 1 >
      std::numeric_limits<size_t>::max() / view.row_stride) {
    return Status::InvalidArgument(
      "Device plane row geometry overflows");
  }
  const size_t last_row = (view.extent.height - 1) * view.row_stride;
  if (view.extent.width >
      std::numeric_limits<size_t>::max() - last_row) {
    return Status::InvalidArgument(
      "Device plane element range overflows");
  }
  const size_t element_count = last_row + view.extent.width;
  if (element_count > std::numeric_limits<size_t>::max() / element_size) {
    return Status::InvalidArgument(
      "Device plane byte range overflows");
  }
  const size_t size_bytes = element_count * element_size;
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
