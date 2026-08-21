// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "dct_reference.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace gjxl::test {
namespace {

enum class BasisDirection {
  kForward,
  kInverse,
};

/// Builds a `length * length` row-major basis matrix indexed as
/// `[frequency][sample]` using the requested normalization direction.
std::vector<double> MakeBasis(
  size_t length,
  BasisDirection direction) {

  // An orthonormal 1-D DCT uses sqrt(2 / N). libjxl divides the
  // forward transform by sqrt(N), and its inverse restores that factor.
  // Folding both conventions into the basis gives sqrt(2) / N forward
  // and sqrt(2) inverse, with the usual extra 1 / sqrt(2) for DC.
  const double sqrt_two = std::sqrt(2.0);
  const double scale = direction == BasisDirection::kForward
    ? sqrt_two / static_cast<double>(length)
    : sqrt_two;

  std::vector<double> basis(length * length);

  for (size_t frequency = 0; frequency < length; ++frequency) {
    const double alpha = frequency == 0 ? 1.0 / sqrt_two : 1.0;

    for (size_t sample = 0; sample < length; ++sample) {
      const double angle =
        (static_cast<double>(sample) + 0.5) *
        static_cast<double>(frequency) *
        std::numbers::pi_v<double> /
        static_cast<double>(length);

      basis[frequency * length + sample] =
        scale * alpha * std::cos(angle);
    }
  }

  return basis;
}

}  // namespace

void ReferenceForwardDct(
  DctShape shape,
  const float* pixels,
  double* coefficients,
  size_t block_count) {

  assert(shape.rows > 0);
  assert(shape.cols > 0);

  const size_t block_size = shape.rows * shape.cols;
  const std::vector<double> vertical_basis =
    MakeBasis(shape.rows, BasisDirection::kForward);
  const std::vector<double> horizontal_basis =
    MakeBasis(shape.cols, BasisDirection::kForward);
  std::vector<double> horizontal_dct(block_size);

  for (size_t block = 0; block < block_count; ++block) {
    const float* source = pixels + block * block_size;
    double* destination = coefficients + block * block_size;

    // Apply the horizontal 1-D transform to every row. The intermediate
    // remains in natural [y][u] order.
    for (size_t y = 0; y < shape.rows; ++y) {
      for (size_t u = 0; u < shape.cols; ++u) {
        double result = 0.0;

        for (size_t x = 0; x < shape.cols; ++x) {
          result +=
            static_cast<double>(source[y * shape.cols + x]) *
            horizontal_basis[u * shape.cols + x];
        }

        horizontal_dct[y * shape.cols + u] = result;
      }
    }

    // Apply the vertical transform, then store the result in libjxl's
    // shape-dependent coefficient layout rather than natural [v][u] order.
    for (size_t v = 0; v < shape.rows; ++v) {
      for (size_t u = 0; u < shape.cols; ++u) {
        double result = 0.0;

        for (size_t y = 0; y < shape.rows; ++y) {
          result +=
            vertical_basis[v * shape.rows + y] *
            horizontal_dct[y * shape.cols + u];
        }

        destination[LibjxlCoefficientIndex(shape, v, u)] =
          result;
      }
    }
  }
}

void ReferenceInverseDct(
  DctShape shape,
  const float* coefficients,
  double* pixels,
  size_t block_count) {

  assert(shape.rows > 0);
  assert(shape.cols > 0);

  const size_t block_size = shape.rows * shape.cols;
  const std::vector<double> vertical_basis =
    MakeBasis(shape.rows, BasisDirection::kInverse);
  const std::vector<double> horizontal_basis =
    MakeBasis(shape.cols, BasisDirection::kInverse);
  std::vector<double> horizontal_idct(block_size);

  for (size_t block = 0; block < block_count; ++block) {
    const float* source = coefficients + block * block_size;
    double* destination = pixels + block * block_size;

    // Undo the horizontal-frequency axis while reading coefficients through
    // the shape-dependent layout. The intermediate is natural [v][x].
    for (size_t v = 0; v < shape.rows; ++v) {
      for (size_t x = 0; x < shape.cols; ++x) {
        double result = 0.0;

        for (size_t u = 0; u < shape.cols; ++u) {
          result +=
            static_cast<double>(
              source[LibjxlCoefficientIndex(shape, v, u)]) *
            horizontal_basis[u * shape.cols + x];
        }

        horizontal_idct[v * shape.cols + x] = result;
      }
    }

    // Undo the vertical-frequency axis and return row-major [y][x] pixels.
    for (size_t y = 0; y < shape.rows; ++y) {
      for (size_t x = 0; x < shape.cols; ++x) {
        double result = 0.0;

        for (size_t v = 0; v < shape.rows; ++v) {
          result +=
            vertical_basis[v * shape.rows + y] *
            horizontal_idct[v * shape.cols + x];
        }

        destination[y * shape.cols + x] = result;
      }
    }
  }
}

}  // namespace gjxl::test
