// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "core/geometry.h"
#include "core/status.h"
#include "gpu/buffer.h"

namespace gjxl {

enum class DeviceElementType : uint8_t {
  kF32,
  kI32,
  kI8,
  kU8,
};

[[nodiscard]] constexpr size_t DeviceElementSize(
  DeviceElementType type) noexcept {

  switch (type) {
    case DeviceElementType::kF32:
    case DeviceElementType::kI32:
      return 4;
    case DeviceElementType::kI8:
    case DeviceElementType::kU8:
      return 1;
  }
  return 0;
}

struct DeviceMemoryRange {
  const DeviceBuffer* buffer = nullptr;
  size_t offset_bytes = 0;
  size_t size_bytes = 0;
};

struct ConstDevicePlaneView {
  const DeviceBuffer* buffer = nullptr;
  size_t offset_bytes = 0;
  DeviceElementType element_type = DeviceElementType::kF32;
  Extent2D extent;
  size_t row_stride = 0;
};

struct DevicePlaneView {
  DeviceBuffer* buffer = nullptr;
  size_t offset_bytes = 0;
  DeviceElementType element_type = DeviceElementType::kF32;
  Extent2D extent;
  size_t row_stride = 0;

  [[nodiscard]] operator ConstDevicePlaneView() const noexcept {
    return {buffer, offset_bytes, element_type, extent, row_stride};
  }
};

struct ConstDeviceImage3View {
  std::array<ConstDevicePlaneView, 3> plane;
};

struct DeviceImage3View {
  std::array<DevicePlaneView, 3> plane;

  [[nodiscard]] operator ConstDeviceImage3View() const noexcept {
    return {{{plane[0], plane[1], plane[2]}}};
  }
};

/// Computes the minimal containing byte size without a buffer/backend. Shared
/// by pre-allocation planning and validation of actual device views. Failure
/// leaves the output unchanged; row padding after the final row is not included.
[[nodiscard]] Status ComputeDevicePlaneSizeBytes(
  DeviceElementType element_type, Extent2D extent, size_t row_stride,
  size_t* size_bytes);

/// Validates plane geometry and returns the minimal containing byte range.
[[nodiscard]] Status ComputeDevicePlaneRange(
  ConstDevicePlaneView view,
  BackendId expected_backend,
  DeviceMemoryRange* range);

[[nodiscard]] Status ValidateDeviceImage3View(
  ConstDeviceImage3View view,
  BackendId expected_backend);

[[nodiscard]] bool DeviceRangesOverlap(
  DeviceMemoryRange left,
  DeviceMemoryRange right) noexcept;

[[nodiscard]] bool DeviceRangesEqual(
  DeviceMemoryRange left,
  DeviceMemoryRange right) noexcept;

}  // namespace gjxl
