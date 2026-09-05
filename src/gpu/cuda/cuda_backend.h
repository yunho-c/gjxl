// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include "core/status.h"
#include "gpu/backend.h"

namespace gjxl {

struct CudaBackendOptions {
  // CUDA runtime device ordinal. Device selection is explicit so callers can
  // construct independent backends for more than one GPU.
  int device_ordinal = 0;

  // Deterministic failure injection used by real-device backend tests.
  bool test_fail_submission = false;
  bool test_fail_completion = false;

  // Uses a library-owned stream-ordered pool when supported by the runtime
  // and device. False retains cudaMalloc/cudaFree. Unsupported devices also
  // retain the synchronous allocator.
  bool use_stream_ordered_allocation = true;

  // Cached physical device memory to retain across synchronization points.
  // Unset selects min(half of device memory, 4 GiB); zero disables retention.
  // Backends on the same device with the same threshold share a private pool.
  // This is a cache release threshold, not a limit on live allocations.
  std::optional<uint64_t> memory_pool_release_threshold_bytes;
};

/// Creates a CUDA backend on the requested runtime device.
///
/// The factory initializes a private non-blocking stream. It does not change
/// the calling thread's current CUDA device after returning.
[[nodiscard]] Status CreateCudaBackend(
  const CudaBackendOptions& options,
  std::unique_ptr<GpuBackend>* out);

[[nodiscard]] inline Status CreateCudaBackend(
  std::unique_ptr<GpuBackend>* out) {
  return CreateCudaBackend({}, out);
}

/// Synchronizes CUDA work on the selected device in the current context and
/// releases unused memory from gjxl's private pools. Live allocations remain
/// valid. Quiesce encoding first to reclaim all cached memory; concurrent
/// encodes may immediately grow it again. Other libraries' pools are untouched.
[[nodiscard]] Status TrimCudaDeviceMemory(int device_ordinal = 0);

/// Injects a failure into the next CUDA compute submission only.
[[nodiscard]] Status ArmNextCudaSubmissionFailureForTest(
  GpuBackend& backend,
  bool fail_submission,
  bool fail_completion);

}  // namespace gjxl
