// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "core/status.h"
#include "gpu/buffer.h"
#include "gpu/ops/primitives.h"
#include "gpu/ops/transform.h"

namespace gjxl {

struct GpuBackendStats {
  uint64_t successful_allocations = 0;
  uint64_t committed_submissions = 0;
};

class GpuBackend {
public:
  virtual ~GpuBackend() = default;

  GpuBackend(const GpuBackend&) = delete;
  GpuBackend& operator=(const GpuBackend&) = delete;

  [[nodiscard]] virtual BackendKind kind() const noexcept = 0;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;

  [[nodiscard]] BackendId id() const noexcept {
    return id_;
  }

  [[nodiscard]] bool owns(const DeviceBuffer& buffer) const noexcept {
    return buffer.backend_id() == id_;
  }

  [[nodiscard]] virtual GpuBackendStats stats() const noexcept = 0;

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
  virtual Status ForwardTransform(
    const TransformBatch& batch) = 0;

  virtual Status InverseTransform(
    const TransformBatch& batch) = 0;

  /// Validates and enqueues dependent primitive commands in one submission.
  virtual Status SubmitPrimitiveSequence(
    std::span<const PrimitiveCommand> commands) = 0;

  // Explicit synchronization is useful for tests and benchmarks.
  virtual Status Synchronize() = 0;

protected:
  GpuBackend()
    : id_(NextId()) {}

private:
  [[nodiscard]] static BackendId NextId() noexcept {
    static std::atomic<BackendId> next_id{1};
    return next_id.fetch_add(1, std::memory_order_relaxed);
  }

  BackendId id_;
};

}  // namespace gjxl
