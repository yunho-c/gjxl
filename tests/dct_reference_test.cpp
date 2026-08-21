// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates the CPU DCT oracle used by GPU tests.
///
/// Golden vector test against libjxl exercises normalization and coefficient layout.
/// Invariance and rectangular-shape tests exercise general correctness.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

#include "dct_reference.h"

namespace {

constexpr gjxl::test::DctShape kDct8Shape{
  .rows = 8,
  .cols = 8,
};

constexpr size_t kDctSize = kDct8Shape.rows * kDct8Shape.cols;

static_assert(
  gjxl::test::LibjxlCoefficientIndex({8, 16}, 3, 5) == 53);

static_assert(
  gjxl::test::LibjxlCoefficientIndex({16, 8}, 3, 5) == 83);

// Generated from libjxl e8ff09762481785938d8e4e01333ed3917571161 by
// applying DCTSlow<8> to an impulse at pixels[2][5], then transposing its
// conventional [v][u] output into ComputeScaledDCT<8, 8>'s [u][v] layout.
constexpr std::array<double, kDctSize> kLibjxlDct8ImpulseGolden = {
  0.015625,
  0.012276483724798474,
  -0.0084561890647843283,
  -0.021672497583158555,
  -0.015625000000000003,
  0.0043109278012959853,
  0.020415046326193381,
  0.018373056287802485,
  -0.012276483724798467,
  -0.0096455713692954713,
  0.0066439851153691888,
  0.017027972086744326,
  0.012276483724798471,
  -0.0033870742394490499,
  -0.016039998973729267,
  -0.014435617695488855,
  -0.0084561890647843405,
  -0.0066439851153692027,
  0.004576456543960202,
  0.011729071172433306,
  0.0084561890647843422,
  -0.0023330573140732105,
  -0.01104854345603982,
  -0.0099434264107252836,
  0.021672497583158555,
  0.017027972086744336,
  -0.011729071172433291,
  -0.030060617695488859,
  -0.021672497583158559,
  0.0059794286307045287,
  0.028316482698527743,
  0.02548416115152867,
  -0.015624999999999983,
  -0.01227648372479846,
  0.0084561890647843196,
  0.021672497583158531,
  0.015624999999999986,
  -0.0043109278012959801,
  -0.020415046326193356,
  -0.018373056287802465,
  -0.0043109278012959792,
  -0.0033870742394490469,
  0.0023330573140732036,
  0.0059794286307045209,
  0.0043109278012959801,
  -0.0011893823045111432,
  -0.0056324986094292732,
  -0.0050691148253352728,
  0.020415046326193384,
  0.016039998973729278,
  -0.011048543456039806,
  -0.028316482698527747,
  -0.020415046326193387,
  0.0056324986094292819,
  0.026673543456039804,
  0.02400555489723177,
  -0.018373056287802475,
  -0.014435617695488855,
  0.0099434264107252628,
  0.025484161151528656,
  0.018373056287802478,
  -0.0050691148253352771,
  -0.024005554897231753,
  -0.021604428630704529,
};

/// Checks DC/AC scaling and a float-quantized round trip for one shape.
bool TestReferenceShape(
  gjxl::test::DctShape shape,
  std::string_view shape_name) {

  const size_t block_size = shape.rows * shape.cols;
  constexpr float kConstantValue = 0.25f;
  constexpr double kExactTolerance = 1e-12;

  // Constant input check
  std::vector<float> constant_pixels(
    block_size,
    kConstantValue);
  std::vector<double> constant_coefficients(block_size);

  gjxl::test::ReferenceForwardDct(
    shape,
    constant_pixels.data(),
    constant_coefficients.data(),
    1);

  // libjxl's scaled convention preserves a constant block's value at DC.
  const size_t dc_index =
    gjxl::test::LibjxlCoefficientIndex(shape, 0, 0);

  if (std::abs(
      constant_coefficients[dc_index] -
      static_cast<double>(kConstantValue)) > kExactTolerance) {
    std::cerr
      << shape_name
      << " reference DC scaling check failed\n";

    return false;
  }

  for (size_t i = 0; i < block_size; ++i) {
    if (i != dc_index &&
        std::abs(constant_coefficients[i]) > kExactTolerance) {
      std::cerr
        << shape_name
        << " reference constant-block AC check failed at "
        << i
        << '\n';

      return false;
    }
  }

  // Round-trip check
  std::mt19937 rng(
    0xDC700000u ^
    static_cast<unsigned int>(shape.rows << 8) ^
    static_cast<unsigned int>(shape.cols));
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  std::vector<float> pixels(block_size);

  for (float& value : pixels) {
    value = distribution(rng);
  }

  std::vector<double> coefficients(block_size);

  gjxl::test::ReferenceForwardDct(
    shape,
    pixels.data(),
    coefficients.data(),
    1);

  // Match the float coefficient storage used at the GPU boundary before
  // checking the inverse reference.
  std::vector<float> rounded_coefficients(block_size);

  std::transform(
    coefficients.begin(),
    coefficients.end(),
    rounded_coefficients.begin(),
    [](double value) {
      return static_cast<float>(value);
    });

  std::vector<double> reconstructed(block_size);

  gjxl::test::ReferenceInverseDct(
    shape,
    rounded_coefficients.data(),
    reconstructed.data(),
    1);

  double max_error = 0.0;

  for (size_t i = 0; i < block_size; ++i) {
    max_error = std::max(
      max_error,
      std::abs(
        reconstructed[i] -
        static_cast<double>(pixels[i])));
  }

  if (max_error > 1e-6) {
    std::cerr
      << shape_name
      << " reference round-trip check failed: max error "
      << max_error
      << '\n';

    return false;
  }

  std::cout
    << shape_name
    << " CPU reference round-trip max error: "
    << max_error
    << '\n';

  return true;
}

/// Checks that transposing the spatial block and shape preserves libjxl's
/// linear coefficient order.
bool TestReferenceTransposePair(
  gjxl::test::DctShape shape,
  std::string_view shape_name) {

  const size_t block_size = shape.rows * shape.cols;
  const gjxl::test::DctShape transposed_shape{
    .rows = shape.cols,
    .cols = shape.rows,
  };
  std::mt19937 rng(
    0xDC710000u ^
    static_cast<unsigned int>(shape.rows << 8) ^
    static_cast<unsigned int>(shape.cols));
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  std::vector<float> pixels(block_size);
  std::vector<float> transposed_pixels(block_size);

  for (size_t y = 0; y < shape.rows; ++y) {
    for (size_t x = 0; x < shape.cols; ++x) {
      const float value = distribution(rng);
      pixels[y * shape.cols + x] = value;
      transposed_pixels[x * shape.rows + y] = value;
    }
  }

  std::vector<double> coefficients(block_size);
  std::vector<double> transposed_coefficients(block_size);

  gjxl::test::ReferenceForwardDct(
    shape,
    pixels.data(),
    coefficients.data(),
    1);

  gjxl::test::ReferenceForwardDct(
    transposed_shape,
    transposed_pixels.data(),
    transposed_coefficients.data(),
    1);

  // libjxl's layout makes a transform and its spatial transpose share the
  // same linear coefficient order. This checks layout independently of IDCT.
  double max_error = 0.0;

  for (size_t i = 0; i < block_size; ++i) {
    max_error = std::max(
      max_error,
      std::abs(
        coefficients[i] -
        transposed_coefficients[i]));
  }

  if (max_error > 1e-12) {
    std::cerr
      << shape_name
      << " reference transpose check failed: max error "
      << max_error
      << '\n';

    return false;
  }

  return true;
}

bool TestLibjxlGoldenVector() {
  std::array<float, kDctSize> pixels{};
  pixels[2 * kDct8Shape.cols + 5] = 1.0f;

  std::array<double, kDctSize> coefficients{};

  gjxl::test::ReferenceForwardDct(
    kDct8Shape,
    pixels.data(),
    coefficients.data(),
    1);

  double max_error = 0.0;

  for (size_t i = 0; i < kDctSize; ++i) {
    max_error = std::max(
      max_error,
      std::abs(
        coefficients[i] -
        kLibjxlDct8ImpulseGolden[i]));
  }

  if (max_error > 1e-14) {
    std::cerr
      << "Pinned libjxl DCT8 comparison failed: max error "
      << max_error
      << '\n';

    return false;
  }

  return true;
}

/// Runs the pinned libjxl parity check and generalized shape invariants.
bool TestReferenceContracts() {
  if (!TestLibjxlGoldenVector()) {
    return false;
  }

  struct ShapeCase {
    gjxl::test::DctShape shape;
    std::string_view name;
  };

  constexpr std::array kShapeCases{
    ShapeCase{{8, 8}, "8x8"},
    ShapeCase{{8, 16}, "8x16"},
    ShapeCase{{16, 8}, "16x8"},
    ShapeCase{{8, 64}, "8x64"},
    ShapeCase{{64, 8}, "64x8"},
  };

  for (const ShapeCase& shape_case : kShapeCases) {
    if (!TestReferenceShape(shape_case.shape, shape_case.name)) {
      return false;
    }
  }

  constexpr std::array kTransposeCases{
    ShapeCase{{8, 16}, "8x16/16x8"},
    ShapeCase{{8, 64}, "8x64/64x8"},
  };

  for (const ShapeCase& shape_case : kTransposeCases) {
    if (!TestReferenceTransposePair(shape_case.shape, shape_case.name)) {
      return false;
    }
  }

  return true;
}

}  // namespace

int main() {
  if (!TestReferenceContracts()) {
    return EXIT_FAILURE;
  }

  std::cout << "All DCT reference tests passed.\n";
  return EXIT_SUCCESS;
}
