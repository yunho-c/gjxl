// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <span>

#include "core/ac_strategy.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

/// Converts one transform's LLF region to one DC value per covered base block.
[[nodiscard]] Status ConvertLowFrequenciesToDc(
  AcStrategyType strategy,
  std::span<const float> coefficients,
  PlaneF32View dc);

/// Converts per-base-block DC values to one transform's LLF region.
[[nodiscard]] Status ConvertDcToLowFrequencies(
  AcStrategyType strategy,
  ConstPlaneF32View dc,
  std::span<float> coefficients);

}  // namespace gjxl
