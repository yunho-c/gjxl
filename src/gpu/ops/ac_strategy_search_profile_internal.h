// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "gpu/ops/ac_strategy_search.h"
#include "gpu/ops/gpu_execution_profile_internal.h"

namespace gjxl::gpu_profile_internal {

[[nodiscard]] Status FindAcStrategyGridGpuResidentProfiled(
  GpuBackend& gpu,
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  ResidentAcStrategySearchInputs resident,
  AcStrategySearchOptions options,
  AcStrategyGrid* out,
  PreparedAcStrategySearch* prepared,
  GpuProfilingSession* profiling_session,
  AcStrategyGpuSearchStats* stats = nullptr);

}  // namespace gjxl::gpu_profile_internal
