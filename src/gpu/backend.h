// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

#include "core/status.h"
#include "gpu/buffer.h"
#include "gpu/ops/transform.h"
#include "gpu/submission.h"

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

  [[nodiscard]] GpuBackendStats stats() const noexcept {
    return {
      successful_allocations_.load(std::memory_order_relaxed),
      committed_submissions_.load(std::memory_order_relaxed),
    };
  }

  /// Releases idle perceptual-reference preparation capacity, if supported.
  /// Safe during independent encodes: active storage remains valid, but leases
  /// already acquired at the trim boundary cannot repopulate this cache.
  /// Does not initialize a device, wait for active work, or trim other pools.
  virtual Status TrimPreparationCache() { return Status::Ok(); }

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

  /// Enqueues transform work. A successful non-empty batch returns a non-null
  /// submission. A failed or empty batch leaves submission null.
  virtual Status ForwardTransform(
    const TransformBatch& batch,
    std::unique_ptr<GpuSubmission>* submission) = 0;

  virtual Status InverseTransform(
    const TransformBatch& batch,
    std::unique_ptr<GpuSubmission>* submission) = 0;

protected:
  GpuBackend()
    : id_(NextId()) {}

  void RecordSuccessfulAllocation() noexcept {
    successful_allocations_.fetch_add(1, std::memory_order_relaxed);
  }

  void RecordCommittedSubmission() noexcept {
    committed_submissions_.fetch_add(1, std::memory_order_relaxed);
  }

private:
  [[nodiscard]] static BackendId NextId() noexcept {
    static std::atomic<BackendId> next_id{1};
    return next_id.fetch_add(1, std::memory_order_relaxed);
  }

  BackendId id_;
  std::atomic<uint64_t> successful_allocations_{0};
  std::atomic<uint64_t> committed_submissions_{0};
};

}  // namespace gjxl
