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

constexpr gjxl::Extent2D kDct8Shape{
  .width = 8,
  .height = 8,
};

constexpr size_t kDctSize = kDct8Shape.width * kDct8Shape.height;

static_assert(
  gjxl::test::LibjxlCoefficientIndex({8, 16}, 3, 5) == 83);

static_assert(
  gjxl::test::LibjxlCoefficientIndex({16, 8}, 3, 5) == 53);

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

struct DctGoldenSample {
  size_t horizontal_frequency;
  size_t vertical_frequency;
  double coefficient;
};

// Selected coefficients generated from the same pinned libjxl revision and
// impulse as kLibjxlDct8ImpulseGolden. Sampling asymmetric frequency pairs
// keeps the larger golden data compact while still pinning scale and layout.
constexpr std::array kLibjxlDct16ImpulseGolden = {
  DctGoldenSample{0, 0, 0.00390625},
  DctGoldenSample{1, 0, 0.0026041236659286983},
  DctGoldenSample{0, 1, 0.0048719727069791849},
  DctGoldenSample{1, 2, 0.002046046835336717},
  DctGoldenSample{2, 1, -0.0038278843932731069},
  DctGoldenSample{3, 5, 0.0060100640370667697},
  DctGoldenSample{5, 3, 0.00048579230904686701},
  DctGoldenSample{7, 11, 0.0014387082011393941},
  DctGoldenSample{11, 7, -0.0057791006466050366},
  DctGoldenSample{15, 14, -0.0057288338417898314},
  DctGoldenSample{14, 15, -0.0030621254844484173},
  DctGoldenSample{8, 8, 0.0039062499999999965},
};

constexpr std::array kLibjxlDct32ImpulseGolden = {
  DctGoldenSample{0, 0, 0.0009765625},
  DctGoldenSample{1, 0, 0.0011845814776345782},
  DctGoldenSample{0, 1, 0.0013396790568295839},
  DctGoldenSample{1, 2, 0.0014774396488265572},
  DctGoldenSample{2, 1, 0.00089310462377957295},
  DctGoldenSample{3, 5, -3.2285940645431752e-05},
  DctGoldenSample{5, 3, -0.0013082263360325187},
  DctGoldenSample{7, 11, 0.0014181465012004183},
  DctGoldenSample{11, 7, -0.00026983048321276292},
  DctGoldenSample{13, 23, 0.0011623779772346932},
  DctGoldenSample{23, 13, -0.0019296582100345972},
  DctGoldenSample{31, 30, -0.00047333272657417282},
  DctGoldenSample{30, 31, -0.00041853395990601141},
  DctGoldenSample{16, 16, 0.00097656249999999913},
};

/// Checks DC/AC scaling and a float-quantized round trip for one shape.
bool TestReferenceShape(
  gjxl::Extent2D shape,
  std::string_view shape_name) {

  const size_t block_size = shape.width * shape.height;
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
    static_cast<unsigned int>(shape.height << 8) ^
    static_cast<unsigned int>(shape.width));
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
  gjxl::Extent2D shape,
  std::string_view shape_name) {

  const size_t block_size = shape.width * shape.height;
  const gjxl::Extent2D transposed_shape{
    .width = shape.height,
    .height = shape.width,
  };
  std::mt19937 rng(
    0xDC710000u ^
    static_cast<unsigned int>(shape.height << 8) ^
    static_cast<unsigned int>(shape.width));
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  std::vector<float> pixels(block_size);
  std::vector<float> transposed_pixels(block_size);

  for (size_t y = 0; y < shape.height; ++y) {
    for (size_t x = 0; x < shape.width; ++x) {
      const float value = distribution(rng);
      pixels[y * shape.width + x] = value;
      transposed_pixels[x * shape.height + y] = value;
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

bool TestLibjxlDct8GoldenVector() {
  std::array<float, kDctSize> pixels{};
  pixels[2 * kDct8Shape.width + 5] = 1.0f;

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

template <size_t Dimension, size_t SampleCount>
bool TestLibjxlGoldenSamples(
  std::string_view transform_name,
  const std::array<DctGoldenSample, SampleCount>& golden) {

  constexpr gjxl::Extent2D kShape{
    .width = Dimension,
    .height = Dimension,
  };
  constexpr size_t kBlockSize = Dimension * Dimension;
  std::array<float, kBlockSize> pixels{};
  pixels[2 * Dimension + 5] = 1.0f;
  std::array<double, kBlockSize> coefficients{};

  gjxl::test::ReferenceForwardDct(
    kShape,
    pixels.data(),
    coefficients.data(),
    1);

  double max_error = 0.0;

  for (const DctGoldenSample& sample : golden) {
    const size_t index =
      gjxl::test::LibjxlCoefficientIndex(
        kShape,
        sample.vertical_frequency,
        sample.horizontal_frequency);

    max_error = std::max(
      max_error,
      std::abs(
        coefficients[index] -
        sample.coefficient));
  }

  if (max_error > 1e-14) {
    std::cerr
      << "Pinned libjxl "
      << transform_name
      << " comparison failed: max error "
      << max_error
      << '\n';

    return false;
  }

  return true;
}

/// Runs the pinned libjxl parity check and generalized shape invariants.
bool TestReferenceContracts() {
  if (!TestLibjxlDct8GoldenVector()) {
    return false;
  }

  if (!TestLibjxlGoldenSamples<16>(
      "DCT16",
      kLibjxlDct16ImpulseGolden)) {
    return false;
  }

  if (!TestLibjxlGoldenSamples<32>(
      "DCT32",
      kLibjxlDct32ImpulseGolden)) {
    return false;
  }

  struct ShapeCase {
    gjxl::Extent2D shape;
    std::string_view name;
  };

  constexpr std::array kShapeCases{
    ShapeCase{{8, 8}, "8x8"},
    ShapeCase{{16, 16}, "16x16"},
    ShapeCase{{32, 32}, "32x32"},
    ShapeCase{{8, 16}, "8x16"},
    ShapeCase{{16, 8}, "16x8"},
    ShapeCase{{16, 32}, "16x32"},
    ShapeCase{{32, 16}, "32x16"},
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
    ShapeCase{{16, 32}, "16x32/32x16"},
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
