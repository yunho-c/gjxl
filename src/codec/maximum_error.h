// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>

#include "core/ac_strategy.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

struct MaximumErrorReduction {
  std::array<float, 3> channel_maximum{};
  float normalized_maximum = 0.0f;

  friend bool operator==(
    const MaximumErrorReduction&,
    const MaximumErrorReduction&) = default;
};

/// Reduces an XYB reconstruction error to one normalized maximum per selected
/// transform. The normalized value is copied over the complete transform
/// footprint. Pixels outside `source_extent` are ignored. Outputs are atomic.
[[nodiscard]] Status ReduceMaximumError(
  ConstImage3FView reference_opsin,
  ConstImage3FView reconstructed_opsin,
  Extent2D source_extent,
  const AcStrategyGrid& strategies,
  const std::array<float, 3>& limits,
  PlaneF32View block_error,
  MaximumErrorReduction* reduction);

/// Pinned maximum-error AQ multiplier: target normalized error in [0.5, 1].
[[nodiscard]] float MaximumErrorQuantFieldMultiplier(
  float normalized_error) noexcept;

/// Applies the pinned per-transform multiplier over each complete footprint.
/// Quant fields are clamped to the representable encoder-policy interval.
/// `upper_bound_limited` reports whether any requested increase hit the upper
/// bound. Failure leaves both outputs unchanged.
[[nodiscard]] Status UpdateMaximumErrorQuantField(
  const AcStrategyGrid& strategies,
  ConstPlaneF32View block_error,
  ConstPlaneF32View input,
  PlaneF32View output,
  bool* upper_bound_limited = nullptr);

}  // namespace gjxl
