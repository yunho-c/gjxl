// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>

namespace gjxl {

enum class BackendKind {
  kCuda,
  kMetal,
};

using BackendId = uint64_t;

class DeviceBuffer {
public:
  virtual ~DeviceBuffer() = default;

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] BackendKind backend() const noexcept {
    return backend_;
  }

  [[nodiscard]] size_t size_bytes() const noexcept {
    return size_bytes_;
  }

  [[nodiscard]] BackendId backend_id() const noexcept {
    return backend_id_;
  }

protected:
  DeviceBuffer(
    BackendKind backend,
    BackendId backend_id,
    size_t size_bytes)
    : backend_(backend),
      backend_id_(backend_id),
      size_bytes_(size_bytes) {}

private:
  BackendKind backend_;
  BackendId backend_id_;
  size_t size_bytes_;
};

}  // namespace gjxl
