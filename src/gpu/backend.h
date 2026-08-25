// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <memory>
#include <string_view>

#include "core/status.h"
#include "gpu/buffer.h"

namespace gjxl {

template <size_t Dimension>
struct SquareDctBatch {
  static_assert(Dimension > 0);

  static constexpr size_t kDimension = Dimension;
  static constexpr size_t kElementsPerBlock = Dimension * Dimension;

  const DeviceBuffer* input = nullptr;
  DeviceBuffer* output = nullptr;

  // Blocks consist of Dimension * Dimension floats and use libjxl's scaled
  // square-DCT convention. Pixel blocks are row-major:
  // pixels[y * Dimension + x].
  // Coefficients use libjxl's square-transform layout:
  // coefficients[u * Dimension + v], with horizontal frequency first.
  // A constant pixel block therefore has that same value at coefficients[0].
  size_t block_count = 0;
};

using Dct8Batch = SquareDctBatch<8>;
using Dct16Batch = SquareDctBatch<16>;
using Dct32Batch = SquareDctBatch<32>;

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

  virtual Status CopyDeviceToHost(
    const DeviceBuffer& src,
    void* dst,
    size_t size_bytes,
    size_t src_offset_bytes = 0) = 0;

  // These enqueue work. They do not need to block the CPU.
  virtual Status ForwardDct8(
    const Dct8Batch& batch) = 0;

  virtual Status InverseDct8(
    const Dct8Batch& batch) = 0;

  virtual Status ForwardDct16(
    const Dct16Batch& batch) = 0;

  virtual Status InverseDct16(
    const Dct16Batch& batch) = 0;

  virtual Status ForwardDct32(
    const Dct32Batch& batch) = 0;

  virtual Status InverseDct32(
    const Dct32Batch& batch) = 0;

  // Explicit synchronization is useful for tests and benchmarks.
  virtual Status Synchronize() = 0;

protected:
  GpuBackend() = default;
};

}  // namespace gjxl
