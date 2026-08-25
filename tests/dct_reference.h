// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

#include "core/geometry.h"

namespace gjxl::test {

/// Returns the linear coefficient offset used by libjxl's `ComputeScaledDCT`.
///
/// Coefficients are row-major `[v][u]` when `height < width` and row-major
/// `[u][v]` otherwise.
constexpr size_t LibjxlCoefficientIndex(
  Extent2D extent,
  size_t vertical_frequency,
  size_t horizontal_frequency) {

  if (extent.height < extent.width) {
    return vertical_frequency * extent.width + horizontal_frequency;
  }

  return horizontal_frequency * extent.height + vertical_frequency;
}

/// Computes batched forward DCT-II blocks using libjxl's scaled convention.
void ReferenceForwardDct(
  Extent2D extent,
  const float* pixels,
  double* coefficients,
  size_t block_count);

/// Computes batched inverse DCT-II blocks using libjxl's scaled convention.
void ReferenceInverseDct(
  Extent2D extent,
  const float* coefficients,
  double* pixels,
  size_t block_count);

}  // namespace gjxl::test
