// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>

#include "codec/ac_strategy.h"
#include "gpu/backend.h"
#include "gpu/image.h"

namespace gjxl {

struct AcStrategyGpuSearchStats {
  std::array<size_t, kAcStrategyCount> candidate_counts{};
  size_t total_candidate_count = 0;
};

struct ResidentAcStrategySearchInputs {
  ConstDeviceImage3View opsin;
  ConstDevicePlaneView quant_field;
  ConstDevicePlaneView pixel_mask;
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

/// Runs the same CPU merge policy while candidate evaluation consumes the
/// prepared opsin, quant field, and pixel mask directly from device memory.
/// Host views remain the diagnostic/search-policy oracle and are never
/// uploaded by this operation.
[[nodiscard]] Status FindAcStrategyGridGpuResident(
  GpuBackend& gpu,
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  ResidentAcStrategySearchInputs resident,
  AcStrategySearchOptions options,
  AcStrategyGrid* out,
  AcStrategyGpuSearchStats* stats = nullptr);

}  // namespace gjxl
