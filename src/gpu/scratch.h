// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <memory>

#include "core/status.h"
#include "gpu/backend.h"
#include "gpu/image.h"

namespace gjxl {

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
