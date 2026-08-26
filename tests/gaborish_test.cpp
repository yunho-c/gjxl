// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates CPU Gaborish filtering against pinned libjxl output.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "codec/gaborish.h"

namespace {

constexpr gjxl::Extent2D kExtent{11, 9};
constexpr size_t kInputStride = kExtent.width + 3;
constexpr size_t kOutputStride = kExtent.width + 5;
constexpr std::array<float, 3> kMultipliers = {
  0.92718927264540152f,
  0.7f,
  1.15f,
};

float InputSample(size_t channel, size_t x, size_t y) {
  return 0.2f * static_cast<float>(channel + 1) +
    0.013f * static_cast<float>(x * x + 3 * y) +
    std::sin(
      0.17f * static_cast<float>((channel + 2) * x + y));
}

struct ImageStorage {
  explicit ImageStorage(size_t stride, float padding)
    : stride(stride), padding(padding) {
    for (std::vector<float>& values : plane) {
      values.assign(stride * kExtent.height, padding);
    }
  }

  void FillInput() {
    for (size_t channel = 0; channel < plane.size(); ++channel) {
      for (size_t y = 0; y < kExtent.height; ++y) {
        for (size_t x = 0; x < kExtent.width; ++x) {
          plane[channel][y * stride + x] = InputSample(channel, x, y);
        }
      }
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{
      gjxl::ConstPlaneF32View{plane[0].data(), kExtent, stride},
      gjxl::ConstPlaneF32View{plane[1].data(), kExtent, stride},
      gjxl::ConstPlaneF32View{plane[2].data(), kExtent, stride},
    }};
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{
      gjxl::PlaneF32View{plane[0].data(), kExtent, stride},
      gjxl::PlaneF32View{plane[1].data(), kExtent, stride},
      gjxl::PlaneF32View{plane[2].data(), kExtent, stride},
    }};
  }

  [[nodiscard]] bool PaddingUntouched() const {
    for (const std::vector<float>& values : plane) {
      for (size_t y = 0; y < kExtent.height; ++y) {
        for (size_t x = kExtent.width; x < stride; ++x) {
          if (values[y * stride + x] != padding) {
            return false;
          }
        }
      }
    }
    return true;
  }

  size_t stride;
  float padding;
  std::array<std::vector<float>, 3> plane;
};

bool CheckPinnedOutput() {
  constexpr std::array<std::array<size_t, 2>, 9> kPoints = {{
    {0, 0}, {1, 0}, {5, 0}, {0, 1}, {3, 2},
    {7, 4}, {10, 7}, {0, 8}, {10, 8},
  }};
  constexpr std::array<std::array<float, 9>, 3> kGoldens = {{
    {{
      0.118733533f, 0.531602263f, 1.52510381f,
      0.367192984f, 1.38568616f, 1.0729866f,
      0.801902533f, 1.49319839f, 0.852656066f,
    }},
    {{
      0.33114028f, 0.903497994f, 1.30114675f,
      0.566082716f, 1.56910479f, 0.277131975f,
      2.04747653f, 1.69181454f, 2.28051662f,
    }},
    {{
      0.402221322f, 1.2897718f, 0.668636322f,
      0.674730301f, 1.54104948f, 0.576916695f,
      3.23237944f, 1.89854693f, 3.20546389f,
    }},
  }};

  ImageStorage input(kInputStride, -111.0f);
  ImageStorage output(kOutputStride, -777.0f);
  input.FillInput();
  const gjxl::Status status = gjxl::ApplyGaborishInverse(
    input.ConstView(), kMultipliers, output.View());
  if (!status.ok()) {
    std::cerr << "Gaborish filtering failed: " << status.message() << '\n';
    return false;
  }

  for (size_t channel = 0; channel < output.plane.size(); ++channel) {
    for (size_t i = 0; i < kPoints.size(); ++i) {
      const size_t x = kPoints[i][0];
      const size_t y = kPoints[i][1];
      const float actual = output.plane[channel][y * output.stride + x];
      if (std::abs(actual - kGoldens[channel][i]) > 3.0e-6f) {
        std::cerr
          << "Gaborish output differs from pinned libjxl at channel "
          << channel << " (" << x << ", " << y << "): "
          << actual << " versus " << kGoldens[channel][i] << '\n';
        return false;
      }
    }
  }

  if (!input.PaddingUntouched() || !output.PaddingUntouched()) {
    std::cerr << "Gaborish filtering overwrote row padding\n";
    return false;
  }
  return true;
}

bool CheckInPlaceAndFlatImage() {
  ImageStorage source(kInputStride, -111.0f);
  ImageStorage separate(kOutputStride, -777.0f);
  ImageStorage in_place(kInputStride, -111.0f);
  source.FillInput();
  in_place.FillInput();

  if (!gjxl::ApplyGaborishInverse(
        source.ConstView(), kMultipliers, separate.View()).ok() ||
      !gjxl::ApplyGaborishInverse(
        in_place.ConstView(), kMultipliers, in_place.View()).ok()) {
    std::cerr << "Valid in-place Gaborish filtering failed\n";
    return false;
  }

  for (size_t channel = 0; channel < source.plane.size(); ++channel) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        if (separate.plane[channel][y * separate.stride + x] !=
            in_place.plane[channel][y * in_place.stride + x]) {
          std::cerr << "In-place Gaborish output differs\n";
          return false;
        }
      }
    }
  }

  ImageStorage flat(kInputStride, -111.0f);
  for (std::vector<float>& plane : flat.plane) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      std::fill_n(plane.data() + y * flat.stride, kExtent.width, 0.75f);
    }
  }
  if (!gjxl::ApplyGaborishInverse(
        flat.ConstView(), kMultipliers, flat.View()).ok()) {
    return false;
  }
  for (const std::vector<float>& plane : flat.plane) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        if (std::abs(plane[y * flat.stride + x] - 0.75f) > 2.0e-6f) {
          std::cerr << "Gaborish filtering changed a flat image\n";
          return false;
        }
      }
    }
  }
  return true;
}

bool CheckInvalidInputsAreAtomic() {
  ImageStorage input(kInputStride, -111.0f);
  ImageStorage output(kOutputStride, -777.0f);
  input.FillInput();
  const auto original_output = output.plane;

  auto bad_multipliers = kMultipliers;
  bad_multipliers[1] = std::numeric_limits<float>::infinity();
  if (gjxl::ApplyGaborishInverse(
        input.ConstView(), bad_multipliers, output.View()).ok() ||
      output.plane != original_output) {
    std::cerr << "Invalid multiplier was accepted or changed output\n";
    return false;
  }

  input.plane[2][4 * input.stride + 3] =
    std::numeric_limits<float>::quiet_NaN();
  if (gjxl::ApplyGaborishInverse(
        input.ConstView(), kMultipliers, output.View()).ok() ||
      output.plane != original_output) {
    std::cerr << "Non-finite filtering result was accepted or changed output\n";
    return false;
  }

  gjxl::Image3FView mismatched = output.View();
  mismatched.plane[1].extent.width -= 1;
  if (gjxl::ApplyGaborishInverse(
        input.ConstView(), kMultipliers, mismatched).ok() ||
      output.plane != original_output) {
    std::cerr << "Mismatched output geometry was accepted\n";
    return false;
  }
  return true;
}

bool CheckDecoderGaborish() {
  ImageStorage impulse(kInputStride, -111.0f);
  ImageStorage output(kOutputStride, -777.0f);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      std::fill_n(
        impulse.plane[channel].data() + y * impulse.stride,
        kExtent.width,
        0.0f);
    }
    impulse.plane[channel][4 * impulse.stride + 5] =
      static_cast<float>(channel + 1);
  }

  const gjxl::GaborishOptions options;
  if (!gjxl::ApplyGaborish(
        impulse.ConstView(), options, output.View()).ok() ||
      !output.PaddingUntouched()) {
    std::cerr << "Decoder-side Gaborish failed\n";
    return false;
  }
  const float divisor =
    1.0f + 4.0f * (options.weight1[0] + options.weight2[0]);
  for (size_t channel = 0; channel < 3; ++channel) {
    const float amplitude = static_cast<float>(channel + 1);
    const float center = output.plane[channel][4 * output.stride + 5];
    const float axis = output.plane[channel][4 * output.stride + 4];
    const float diagonal = output.plane[channel][3 * output.stride + 4];
    if (std::abs(center - amplitude / divisor) > 2.0e-7f ||
        std::abs(axis - amplitude * options.weight1[channel] / divisor) >
          2.0e-7f ||
        std::abs(diagonal - amplitude * options.weight2[channel] / divisor) >
          2.0e-7f) {
      std::cerr << "Decoder-side Gaborish impulse response is incorrect\n";
      return false;
    }
  }

  ImageStorage in_place(kInputStride, -111.0f);
  in_place.plane = impulse.plane;
  if (!gjxl::ApplyGaborish(
        in_place.ConstView(), options, in_place.View()).ok()) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        if (in_place.plane[channel][y * in_place.stride + x] !=
            output.plane[channel][y * output.stride + x]) {
          std::cerr << "In-place decoder Gaborish differs\n";
          return false;
        }
      }
    }
  }

  auto invalid_options = options;
  invalid_options.weight1[0] = std::numeric_limits<float>::infinity();
  const auto original = output.plane;
  if (gjxl::ApplyGaborish(
        impulse.ConstView(), invalid_options, output.View()).ok() ||
      output.plane != original) {
    std::cerr << "Invalid decoder Gaborish was not atomic\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckPinnedOutput() ||
      !CheckInPlaceAndFlatImage() ||
      !CheckInvalidInputsAreAtomic() ||
      !CheckDecoderGaborish()) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
