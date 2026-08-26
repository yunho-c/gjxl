// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/status.h"

namespace gjxl {

/// Owns one committed GPU submission and its completion result.
class GpuSubmission {
public:
  virtual ~GpuSubmission() = default;

  GpuSubmission(const GpuSubmission&) = delete;
  GpuSubmission& operator=(const GpuSubmission&) = delete;

  /// Waits for completion. Repeated and concurrent calls return the same
  /// cached status. Destruction does not implicitly wait.
  [[nodiscard]] virtual Status Wait() = 0;

protected:
  GpuSubmission() = default;
};

}  // namespace gjxl
