// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

#include "codec/dct.h"
#include "dct_reference.h"

namespace {

constexpr std::array kStrategies = {
  gjxl::AcStrategyType::kDct8,
  gjxl::AcStrategyType::kDct16x16,
  gjxl::AcStrategyType::kDct32x32,
  gjxl::AcStrategyType::kDct16x8,
  gjxl::AcStrategyType::kDct8x16,
  gjxl::AcStrategyType::kDct32x16,
  gjxl::AcStrategyType::kDct16x32,
};

float Sample(size_t x, size_t y) {
  return
    0.35f * std::sin(static_cast<float>(x + 1) * 0.19f) +
    0.21f * std::cos(static_cast<float>(y + 2) * 0.31f) +
    0.003f * static_cast<float>((x * 17 + y * 13) % 23);
}

bool CheckStrategy(gjxl::AcStrategyType strategy) {
  const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(strategy);
  if (info == nullptr || !gjxl::SupportsCpuDct(strategy)) {
    return false;
  }

  const gjxl::Extent2D extent = info->pixel_extent();
  const size_t count = info->coefficient_count();
  std::vector<float> pixels(count);
  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      pixels[y * extent.width + x] = Sample(x, y);
    }
  }

  std::vector<double> reference_coefficients(count);
  gjxl::test::ReferenceForwardDct(
    extent,
    pixels.data(),
    reference_coefficients.data(),
    1);

  std::vector<float> coefficients(count);
  if (!gjxl::ForwardDctCpu(strategy, pixels, coefficients).ok()) {
    return false;
  }

  for (size_t i = 0; i < count; ++i) {
    if (std::abs(
          static_cast<double>(coefficients[i]) -
          reference_coefficients[i]) > 2.0e-6) {
      std::cerr << "CPU forward DCT differs from the independent oracle\n";
      return false;
    }
  }

  std::vector<double> reference_pixels(count);
  gjxl::test::ReferenceInverseDct(
    extent,
    coefficients.data(),
    reference_pixels.data(),
    1);

  std::vector<float> reconstructed(count);
  if (!gjxl::InverseDctCpu(
        strategy,
        coefficients,
        reconstructed).ok()) {
    return false;
  }

  for (size_t i = 0; i < count; ++i) {
    if (std::abs(
          static_cast<double>(reconstructed[i]) -
          reference_pixels[i]) > 2.0e-6 ||
        std::abs(reconstructed[i] - pixels[i]) > 3.0e-6f) {
      std::cerr << "CPU inverse DCT differs from the independent oracle\n";
      return false;
    }
  }

  return true;
}

bool CheckInvalidInputs() {
  std::array<float, 64> input{};
  std::array<float, 64> output{};
  if (gjxl::ForwardDctCpu(
        gjxl::AcStrategyType::kDct8,
        std::span<const float>(input).first(63),
        output).ok() ||
      gjxl::InverseDctCpu(
        gjxl::AcStrategyType::kDct8,
        input,
        std::span<float>(output).first(63)).ok() ||
      gjxl::ForwardDctCpu(
        gjxl::AcStrategyType::kAfv0,
        input,
        output).ok() ||
      gjxl::SupportsCpuDct(gjxl::AcStrategyType::kAfv0)) {
    std::cerr << "Invalid CPU DCT request was accepted\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  for (gjxl::AcStrategyType strategy : kStrategies) {
    if (!CheckStrategy(strategy)) {
      return EXIT_FAILURE;
    }
  }

  if (!CheckInvalidInputs()) {
    return EXIT_FAILURE;
  }

  std::cout << "All CPU DCT tests passed.\n";
  return EXIT_SUCCESS;
}
