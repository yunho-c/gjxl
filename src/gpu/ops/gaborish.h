// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>

#include "core/image.h"
#include "core/status.h"
#include "gpu/backend.h"

namespace gjxl {

/// Applies the inverse Gaborish filter through the backend image primitives.
/// The output is committed only after all GPU work and validation succeed.
[[nodiscard]] Status ApplyGaborishInverseGpu(
  GpuBackend& gpu,
  ConstImage3FView input,
  std::array<float, 3> multipliers,
  Image3FView output);

}  // namespace gjxl
