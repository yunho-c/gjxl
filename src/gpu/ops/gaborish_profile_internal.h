// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "gpu/ops/gaborish.h"
#include "gpu/ops/gpu_execution_profile_internal.h"

namespace gjxl::gpu_profile_internal {

[[nodiscard]] Status ApplyGaborishInverseGpuProfiled(
  GpuBackend& gpu,
  ConstImage3FView input,
  std::array<float, 3> multipliers,
  Image3FView output,
  GpuProfilingSession* profiling_session);

}  // namespace gjxl::gpu_profile_internal
