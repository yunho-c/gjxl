// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "util/fast_math.h"

namespace {

bool CheckFastLog2() {
  constexpr std::array<float, 7> kMantissas = {
    0.625f,
    0.75f,
    0.875f,
    1.0f,
    1.25f,
    1.5f,
    1.75f,
  };

  for (int exponent = -20; exponent <= 20; ++exponent) {
    for (float mantissa : kMantissas) {
      const float value = std::ldexp(mantissa, exponent);
      const float expected = std::log2(value);
      const float actual = gjxl::fast_math::FastLog2(value);
      if (!std::isfinite(actual) ||
          std::abs(actual - expected) > 5.0e-6f) {
        std::cerr << "FastLog2 exceeded its error bound\n";
        return false;
      }
    }
  }

  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  if (!std::isnan(gjxl::fast_math::FastLog2(0.0f)) ||
      !std::isnan(gjxl::fast_math::FastLog2(-1.0f)) ||
      !std::isnan(gjxl::fast_math::FastLog2(infinity)) ||
      !std::isnan(gjxl::fast_math::FastLog2(nan))) {
    std::cerr << "FastLog2 accepted an invalid input\n";
    return false;
  }

  return true;
}

bool CheckFastPow2() {
  for (int eighths = -160; eighths <= 160; ++eighths) {
    const float value = static_cast<float>(eighths) * 0.125f;
    const float expected = std::exp2(value);
    const float actual = gjxl::fast_math::FastPow2(value);
    const float relative_error = std::abs(actual - expected) / expected;
    if (!std::isfinite(actual) || relative_error > 5.0e-7f) {
      std::cerr << "FastPow2 exceeded its error bound\n";
      return false;
    }
  }

  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  if (!std::isnan(gjxl::fast_math::FastPow2(-127.0f)) ||
      !std::isnan(gjxl::fast_math::FastPow2(128.0f)) ||
      !std::isnan(gjxl::fast_math::FastPow2(infinity)) ||
      !std::isnan(gjxl::fast_math::FastPow2(nan))) {
    std::cerr << "FastPow2 accepted an invalid input\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!CheckFastLog2() || !CheckFastPow2()) {
    return EXIT_FAILURE;
  }

  std::cout << "All fast math tests passed.\n";
  return EXIT_SUCCESS;
}
