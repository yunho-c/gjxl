// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <memory>
#include <string_view>

#include "core/status.h"
#include "gpu/buffer.h"

namespace gjxl {

struct Dct8Batch {
  const DeviceBuffer* input = nullptr;
  DeviceBuffer* output = nullptr;

  // Each block consists of exactly 64 contiguous floats.
  size_t block_count = 0;
};

class GpuBackend {
public:
  virtual ~GpuBackend() = default;

  GpuBackend(const GpuBackend&) = delete;
  GpuBackend& operator=(const GpuBackend&) = delete;

  [[nodiscard]] virtual BackendKind kind() const noexcept = 0;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;

  virtual Status Allocate(
    size_t size_bytes,
    std::unique_ptr<DeviceBuffer>* out) = 0;

  // Caller must ensure that the buffer is not simultaneously
  // being accessed by an outstanding GPU command.
  virtual Status CopyHostToDevice(
    DeviceBuffer& dst,
    const void* src,
    size_t size_bytes,
    size_t dst_offset_bytes = 0) = 0;

  // These enqueue work. They do not need to block the CPU.
  virtual Status ForwardDct8(
    const Dct8Batch& batch) = 0;

  virtual Status InverseDct8(
    const Dct8Batch& batch) = 0;

  // Explicit synchronization is useful for tests and benchmarks.
  virtual Status Synchronize() = 0;

protected:
  GpuBackend() = default;
};

}
