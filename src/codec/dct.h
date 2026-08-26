// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <span>

#include "core/ac_strategy.h"
#include "core/status.h"

namespace gjxl {

/// Returns whether the scalar CPU DCT supports this complete AC strategy.
[[nodiscard]] bool SupportsCpuDct(AcStrategyType strategy) noexcept;

/// Computes one forward DCT using libjxl's scaling and coefficient layout.
[[nodiscard]] Status ForwardDctCpu(
  AcStrategyType strategy,
  std::span<const float> pixels,
  std::span<float> coefficients);

/// Computes one inverse DCT using libjxl's scaling and coefficient layout.
[[nodiscard]] Status InverseDctCpu(
  AcStrategyType strategy,
  std::span<const float> coefficients,
  std::span<float> pixels);

}  // namespace gjxl
