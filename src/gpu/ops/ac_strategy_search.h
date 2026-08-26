// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>

#include "codec/ac_strategy.h"
#include "gpu/backend.h"

namespace gjxl {

struct AcStrategyGpuSearchStats {
  std::array<size_t, kAcStrategyCount> candidate_counts{};
  size_t total_candidate_count = 0;
};

/// Selects an AC-strategy grid after staging every dependency-safe candidate
/// cost through one GPU submission. Search decisions and tie-breaking remain
/// on the CPU and are identical to FindAcStrategyGrid's traversal.
[[nodiscard]] Status FindAcStrategyGridGpu(
  GpuBackend& gpu,
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  AcStrategySearchOptions options,
  AcStrategyGrid* out,
  AcStrategyGpuSearchStats* stats = nullptr);

}  // namespace gjxl
