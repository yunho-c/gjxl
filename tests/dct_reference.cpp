// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "dct_reference.h"

#include <cmath>
#include <cstddef>

namespace gjxl::test {
namespace {

constexpr size_t kDctWidth = 8;
constexpr size_t kDctSize = kDctWidth * kDctWidth;

enum class Direction {
  kForward,
  kInverse,
};

double DctBasis(
  size_t frequency,
  size_t sample) {

  static const double kPi = std::acos(-1.0);

  const double scale =
    frequency == 0
      ? std::sqrt(1.0 / static_cast<double>(kDctWidth))
      : std::sqrt(2.0 / static_cast<double>(kDctWidth));

  return scale * std::cos(
    kPi *
    static_cast<double>((2 * sample + 1) * frequency) /
    static_cast<double>(2 * kDctWidth));
}

void ReferenceDct8(
  Direction direction,
  const float* input,
  float* output,
  size_t block_count) {

  for (size_t block = 0; block < block_count; ++block) {
    const float* source = input + block * kDctSize;
    float* destination = output + block * kDctSize;

    for (size_t row = 0; row < kDctWidth; ++row) {
      for (size_t col = 0; col < kDctWidth; ++col) {
        double result = 0.0;

        for (size_t i = 0; i < kDctWidth; ++i) {
          for (size_t j = 0; j < kDctWidth; ++j) {
            if (direction == Direction::kForward) {
              result +=
                DctBasis(row, i) *
                static_cast<double>(source[kDctWidth * i + j]) *
                DctBasis(col, j);
            } else {
              result +=
                DctBasis(i, row) *
                static_cast<double>(source[kDctWidth * i + j]) *
                DctBasis(j, col);
            }
          }
        }

        destination[kDctWidth * row + col] =
          static_cast<float>(result);
      }
    }
  }
}

}  // namespace

void ReferenceForwardDct8(
  const float* input,
  float* output,
  size_t block_count) {

  ReferenceDct8(
    Direction::kForward,
    input,
    output,
    block_count);
}

void ReferenceInverseDct8(
  const float* input,
  float* output,
  size_t block_count) {

  ReferenceDct8(
    Direction::kInverse,
    input,
    output,
    block_count);
}

}  // namespace gjxl::test
