// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <memory>

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

/// Injects a failure into the next CUDA compute submission only.
[[nodiscard]] Status ArmNextCudaSubmissionFailureForTest(
  GpuBackend& backend,
  bool fail_submission,
  bool fail_completion);

}  // namespace gjxl
