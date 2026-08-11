// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

namespace gjxl {

enum class BackendKind {
  kCuda,
  kMetal,
};

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

protected:
  DeviceBuffer(BackendKind backend, size_t size_bytes)
    : backend_(backend),
      size_bytes_(size_bytes) {}

private:
  BackendKind backend_;
  size_t size_bytes_;
};

}
