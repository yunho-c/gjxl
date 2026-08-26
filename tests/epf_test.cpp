// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates EPF sharpness and reciprocal sigma field generation.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "codec/epf.h"

namespace {

bool CheckPinnedSigmaValues() {
  constexpr gjxl::Extent2D kExtent{4, 4};
  gjxl::AcStrategyGrid strategies;
  gjxl::Quantizer quantizer;
  if (!gjxl::AcStrategyGrid::Create(kExtent, &strategies).ok() ||
      !strategies.Set(0, 0, gjxl::AcStrategyType::kDct32x32).ok() ||
      !gjxl::Quantizer::Create({.global_scale = 3541, .quant_dc = 10},
                              &quantizer).ok()) {
    return false;
  }

  constexpr size_t kStride = 6;
  std::array<int32_t, kStride * kExtent.height> raw_quant;
  std::array<uint8_t, kStride * kExtent.height> sharpness;
  std::array<float, kStride * kExtent.height> inverse_sigma;
  raw_quant.fill(-777);
  sharpness.fill(255);
  inverse_sigma.fill(-777.0f);
  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      raw_quant[y * kStride + x] =
        static_cast<int32_t>(10 + x + y);
      sharpness[y * kStride + x] =
        static_cast<uint8_t>((y * kExtent.width + x) % 8);
    }
  }

  const gjxl::ConstPlaneI32View raw_view{
    raw_quant.data(), kExtent, kStride};
  const gjxl::ConstPlaneU8View sharpness_view{
    sharpness.data(), kExtent, kStride};
  const gjxl::PlaneF32View sigma_view{
    inverse_sigma.data(), kExtent, kStride};
  if (!gjxl::ComputeEpfInverseSigma(
        strategies,
        raw_view,
        quantizer,
        sharpness_view,
        {},
        sigma_view).ok()) {
    return false;
  }

  // All blocks use the transform anchor's raw quant value. The LUT samples
  // below are the pinned LoopFilter defaults at global scale 3541.
  constexpr std::array<float, 8> kGoldens = {
    -10000.0f,
    -9.632864952f,
    -4.816432476f,
    -3.210955381f,
    -2.408216238f,
    -1.926573157f,
    -1.605477691f,
    -1.376123667f,
  };
  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      const float expected = kGoldens[(y * kExtent.width + x) % 8];
      if (std::abs(sigma_view.Row(y)[x] - expected) > 2.0e-6f) {
        std::cerr << "EPF inverse sigma differs from pinned libjxl\n";
        return false;
      }
    }
    for (size_t x = kExtent.width; x < kStride; ++x) {
      if (inverse_sigma[y * kStride + x] != -777.0f) {
        std::cerr << "EPF inverse sigma overwrote row padding\n";
        return false;
      }
    }
  }
  return true;
}

bool CheckDefaultSharpnessAndValidation() {
  constexpr gjxl::Extent2D kExtent{2, 2};
  constexpr size_t kStride = 3;
  std::array<uint8_t, kStride * kExtent.height> sharpness;
  sharpness.fill(255);
  const gjxl::PlaneU8View sharpness_view{
    sharpness.data(), kExtent, kStride};
  if (!gjxl::FillDefaultEpfSharpness(sharpness_view).ok()) {
    return false;
  }
  for (size_t y = 0; y < kExtent.height; ++y) {
    if (sharpness_view.Row(y)[0] != 4 ||
        sharpness_view.Row(y)[1] != 4 ||
        sharpness[y * kStride + 2] != 255) {
      std::cerr << "Default EPF sharpness field is incorrect\n";
      return false;
    }
  }

  gjxl::AcStrategyGrid strategies;
  gjxl::Quantizer quantizer;
  if (!gjxl::AcStrategyGrid::Create(kExtent, &strategies).ok()) {
    return false;
  }
  strategies.fill_dct8();
  if (!gjxl::Quantizer::Create({.global_scale = 3541, .quant_dc = 10},
                              &quantizer).ok()) {
    return false;
  }
  std::array<int32_t, 4> raw_quant = {10, 10, 10, 10};
  std::array<float, 4> output = {-777.0f, -777.0f, -777.0f, -777.0f};
  const auto original_output = output;
  sharpness[0] = 8;
  if (gjxl::ComputeEpfInverseSigma(
        strategies,
        {raw_quant.data(), kExtent, kExtent.width},
        quantizer,
        {sharpness.data(), kExtent, kStride},
        {},
        {output.data(), kExtent, kExtent.width}).ok() ||
      output != original_output) {
    std::cerr << "Invalid EPF sharpness changed output\n";
    return false;
  }
  sharpness[0] = 4;
  raw_quant[3] = 0;
  if (gjxl::ComputeEpfInverseSigma(
        strategies,
        {raw_quant.data(), kExtent, kExtent.width},
        quantizer,
        {sharpness.data(), kExtent, kStride},
        {},
        {output.data(), kExtent, kExtent.width}).ok() ||
      output != original_output) {
    std::cerr << "Invalid EPF raw quantization changed output\n";
    return false;
  }
  auto bad_options = gjxl::EpfSigmaOptions{};
  bad_options.quant_multiplier =
    std::numeric_limits<float>::infinity();
  raw_quant[3] = 10;
  if (gjxl::ComputeEpfInverseSigma(
        strategies,
        {raw_quant.data(), kExtent, kExtent.width},
        quantizer,
        {sharpness.data(), kExtent, kStride},
        bad_options,
        {output.data(), kExtent, kExtent.width}).ok()) {
    std::cerr << "Invalid EPF options were accepted\n";
    return false;
  }
  return true;
}

bool CheckEpfFiltering() {
  constexpr gjxl::Extent2D kImageExtent{16, 16};
  constexpr gjxl::Extent2D kBlockExtent{2, 2};
  constexpr size_t kStride = kImageExtent.width + 3;
  std::array<std::vector<float>, 3> input;
  std::array<std::vector<float>, 3> output;
  std::array<std::vector<float>, 3> in_place;
  for (size_t channel = 0; channel < 3; ++channel) {
    input[channel].assign(kStride * kImageExtent.height, -111.0f);
    output[channel].assign(kStride * kImageExtent.height, -777.0f);
    for (size_t y = 0; y < kImageExtent.height; ++y) {
      std::fill_n(
        input[channel].data() + y * kStride,
        kImageExtent.width,
        0.0f);
    }
    input[channel][6 * kStride + 6] = static_cast<float>(channel + 1);
    in_place[channel] = input[channel];
  }
  const auto const_view = [&](const std::array<std::vector<float>, 3>& image) {
    return gjxl::ConstImage3FView{{
      gjxl::ConstPlaneF32View{
        image[0].data(), kImageExtent, kStride},
      gjxl::ConstPlaneF32View{
        image[1].data(), kImageExtent, kStride},
      gjxl::ConstPlaneF32View{
        image[2].data(), kImageExtent, kStride},
    }};
  };
  const auto view = [&](std::array<std::vector<float>, 3>& image) {
    return gjxl::Image3FView{{
      gjxl::PlaneF32View{image[0].data(), kImageExtent, kStride},
      gjxl::PlaneF32View{image[1].data(), kImageExtent, kStride},
      gjxl::PlaneF32View{image[2].data(), kImageExtent, kStride},
    }};
  };

  std::array<float, 4> inverse_sigma{};
  inverse_sigma.fill(-1.0f);
  gjxl::EpfFilterOptions options;
  options.iterations = 1;
  options.channel_scale = {0.0f, 0.0f, 0.0f};
  if (!gjxl::ApplyEpf(
        const_view(input),
        {inverse_sigma.data(), kBlockExtent, kBlockExtent.width},
        options,
        view(output)).ok()) {
    std::cerr << "EPF pass 1 failed\n";
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    const float expected = static_cast<float>(channel + 1) / 5.0f;
    if (std::abs(output[channel][6 * kStride + 6] - expected) > 1.0e-7f ||
        std::abs(output[channel][6 * kStride + 5] - expected) > 1.0e-7f) {
      std::cerr << "EPF cardinal impulse response is incorrect\n";
      return false;
    }
    for (size_t y = 0; y < kImageExtent.height; ++y) {
      for (size_t x = kImageExtent.width; x < kStride; ++x) {
        if (output[channel][y * kStride + x] != -777.0f) {
          std::cerr << "EPF overwrote row padding\n";
          return false;
        }
      }
    }
  }

  if (!gjxl::ApplyEpf(
        const_view(in_place),
        {inverse_sigma.data(), kBlockExtent, kBlockExtent.width},
        options,
        view(in_place)).ok()) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kImageExtent.height; ++y) {
      for (size_t x = 0; x < kImageExtent.width; ++x) {
        if (in_place[channel][y * kStride + x] !=
            output[channel][y * kStride + x]) {
          std::cerr << "In-place EPF differs\n";
          return false;
        }
      }
    }
  }

  inverse_sigma.fill(-10000.0f);
  options.iterations = 3;
  if (!gjxl::ApplyEpf(
        const_view(input),
        {inverse_sigma.data(), kBlockExtent, kBlockExtent.width},
        options,
        view(output)).ok()) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kImageExtent.height; ++y) {
      for (size_t x = 0; x < kImageExtent.width; ++x) {
        if (output[channel][y * kStride + x] !=
            input[channel][y * kStride + x]) {
          std::cerr << "Disabled EPF block changed pixels\n";
          return false;
        }
      }
    }
  }

  const auto original = output;
  options.iterations = 4;
  if (gjxl::ApplyEpf(
        const_view(input),
        {inverse_sigma.data(), kBlockExtent, kBlockExtent.width},
        options,
        view(output)).ok() ||
      output != original) {
    std::cerr << "Invalid EPF filtering was accepted or not atomic\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckPinnedSigmaValues() ||
      !CheckDefaultSharpnessAndValidation() ||
      !CheckEpfFiltering()) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
