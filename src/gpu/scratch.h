// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <memory>

#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/image.h"

namespace gjxl {

struct DevicePlaneLayout {
  DeviceElementType element_type = DeviceElementType::kF32;
  Extent2D extent;
  size_t row_stride = 0;
  size_t offset_bytes = 0;
  size_t size_bytes = 0;

  bool operator==(const DevicePlaneLayout&) const = default;
};

/// Checked layout construction, heap-allocation-free on success, used by arena
/// slicing. A successful append commits both the plane and the new capacity;
/// failure leaves both unchanged. This owns neither views nor backing memory.
class DeviceScratchLayoutPlan {
public:
  explicit DeviceScratchLayoutPlan(size_t initial_bytes = 0) : capacity_bytes_(initial_bytes) {}
  [[nodiscard]] Status AddPlane(
    DeviceElementType element_type, Extent2D extent, size_t row_stride,
    size_t alignment_bytes, DevicePlaneLayout* plane);
  [[nodiscard]] size_t capacity_bytes() const noexcept { return capacity_bytes_; }
private:
  size_t capacity_bytes_ = 0;
};

/// One reusable device allocation with checked, non-owning slice planning.
class DeviceScratchArena {
public:
  DeviceScratchArena() = default;
  ~DeviceScratchArena() = default;

  DeviceScratchArena(const DeviceScratchArena&) = delete;
  DeviceScratchArena& operator=(const DeviceScratchArena&) = delete;
  DeviceScratchArena(DeviceScratchArena&&) noexcept = default;
  DeviceScratchArena& operator=(DeviceScratchArena&&) noexcept = default;

  /// Allocates or grows storage. Caller must first wait on every outstanding
  /// submission that references existing slices. A sufficient allocation is
  /// reused.
  [[nodiscard]] Status Prepare(GpuBackend& backend, size_t capacity_bytes);
  void ResetLayout() noexcept;

  [[nodiscard]] Status AllocatePlane(
    DeviceElementType element_type,
    Extent2D extent,
    size_t row_stride,
    size_t alignment_bytes,
    DevicePlaneView* view);

  /// Binds a precomputed slice without allocation. Validates its byte size and
  /// backing bounds; layout/peak usage become the maximum bound end. Planning
  /// owns ordering/non-overlap, and borrowed aliases do not need another slice.
  [[nodiscard]] Status BindPlane(const DevicePlaneLayout& plane, DevicePlaneView* view);

  [[nodiscard]] size_t capacity_bytes() const noexcept {
    return capacity_bytes_;
  }

  [[nodiscard]] size_t layout_bytes() const noexcept {
    return layout_bytes_;
  }

  [[nodiscard]] size_t peak_layout_bytes() const noexcept {
    return peak_layout_bytes_;
  }

  /// Exposes the allocation for backend-specific lifetime policy. The arena
  /// retains ownership and no outstanding submission may access it.
  [[nodiscard]] DeviceBuffer* backing_buffer() noexcept {
    return buffer_.get();
  }

private:
  std::unique_ptr<DeviceBuffer> buffer_;
  BackendId backend_id_ = 0;
  size_t capacity_bytes_ = 0;
  size_t layout_bytes_ = 0;
  size_t peak_layout_bytes_ = 0;
};

}  // namespace gjxl
