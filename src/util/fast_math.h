// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace gjxl::fast_math {

/// Approximates log2 with libjxl's rational polynomial.
/// Returns NaN for non-positive or non-finite inputs.
[[nodiscard]] inline float FastLog2(float value) {
  if (!std::isfinite(value) || value <= 0.0f) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  constexpr float kP0 = -1.8503833400518310e-06f;
  constexpr float kP1 = 1.4287160470083755f;
  constexpr float kP2 = 0.74245873327820566f;
  constexpr float kQ0 = 0.99032814277590719f;
  constexpr float kQ1 = 1.0096718572241148f;
  constexpr float kQ2 = 0.17409343003366853f;

  const uint32_t value_bits = std::bit_cast<uint32_t>(value);
  const int32_t shifted_exponent =
    static_cast<int32_t>(value_bits - 0x3f2aaaabu) >> 23;
  const uint32_t mantissa_bits =
    value_bits - (static_cast<uint32_t>(shifted_exponent) << 23);
  const float x = std::bit_cast<float>(mantissa_bits) - 1.0f;

  float numerator = std::fma(kP2, x, kP1);
  numerator = std::fma(numerator, x, kP0);
  float denominator = std::fma(kQ2, x, kQ1);
  denominator = std::fma(denominator, x, kQ0);
  return numerator / denominator + static_cast<float>(shifted_exponent);
}

/// Approximates 2^x with libjxl's rational polynomial.
/// Returns NaN outside the normal float exponent range [-126, 127].
[[nodiscard]] inline float FastPow2(float value) {
  if (!std::isfinite(value) || value < -126.0f || value > 127.0f) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  const float floor_value = std::floor(value);
  const int32_t exponent = static_cast<int32_t>(floor_value) + 127;
  const float exponent_value = std::bit_cast<float>(
    static_cast<uint32_t>(exponent) << 23);
  const float fraction = value - floor_value;

  float numerator = fraction + 1.01749063e+01f;
  numerator = std::fma(numerator, fraction, 4.88687798e+01f);
  numerator = std::fma(numerator, fraction, 9.85506591e+01f);
  numerator *= exponent_value;

  float denominator = std::fma(
    fraction,
    2.10242958e-01f,
    -2.22328856e-02f);
  denominator = std::fma(
    denominator,
    fraction,
    -1.94414990e+01f);
  denominator = std::fma(
    denominator,
    fraction,
    9.85506633e+01f);
  return numerator / denominator;
}

}  // namespace gjxl::fast_math
