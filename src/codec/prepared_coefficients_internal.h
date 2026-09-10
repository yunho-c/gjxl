// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "core/managed_allocator.h"
#include "core/ac_strategy.h"
#include "core/geometry.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl::prepared_coefficients_internal {

struct PreparedTransform {
  size_t block_x = 0;
  size_t block_y = 0;
  AcStrategyType strategy = AcStrategyType::kCount;
  size_t coefficient_offset = 0;
  size_t coefficient_count = 0;
};

/// CPU forward transforms shared by final CfL and exact coefficient coding.
/// Transform records are row-major; tile indices preserve that order.
struct PreparedForwardDctCoefficients {
  Extent2D pixel_extent;
  Extent2D block_extent;
  Extent2D color_tile_extent;
  resource_budget_internal::ManagedVector<PreparedTransform> transforms;
  resource_budget_internal::ManagedVector<size_t> color_tile_offsets;
  resource_budget_internal::ManagedVector<size_t> color_tile_transform_indices;
  std::array<resource_budget_internal::ManagedVector<float>, 3> coefficients;

  [[nodiscard]] bool valid() const noexcept;
};

[[nodiscard]] Status PrepareForwardDctCoefficients(
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  PreparedForwardDctCoefficients* out);

}  // namespace gjxl::prepared_coefficients_internal
