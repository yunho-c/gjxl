// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

namespace gjxl::test {

/// Describes the spatial dimensions of one DCT block.
struct DctShape {
  /// Number of spatial rows or vertical-frequency samples.
  size_t rows;

  /// Number of spatial columns or horizontal-frequency samples.
  size_t cols;
};

/// Returns the linear coefficient offset used by libjxl's `ComputeScaledDCT`.
///
/// Coefficients are row-major `[v][u]` when `rows < cols` and row-major
/// `[u][v]` otherwise.
constexpr size_t LibjxlCoefficientIndex(
  DctShape shape,
  size_t vertical_frequency,
  size_t horizontal_frequency) {

  if (shape.rows < shape.cols) {
    return vertical_frequency * shape.cols + horizontal_frequency;
  }

  return horizontal_frequency * shape.rows + vertical_frequency;
}

/// Computes batched forward DCT-II blocks using libjxl's scaled convention.
void ReferenceForwardDct(
  DctShape shape,
  const float* pixels,
  double* coefficients,
  size_t block_count);

/// Computes batched inverse DCT-II blocks using libjxl's scaled convention.
void ReferenceInverseDct(
  DctShape shape,
  const float* coefficients,
  double* pixels,
  size_t block_count);

}  // namespace gjxl::test
