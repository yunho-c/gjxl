// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates the scalar JPEG XL linear-RGB/XYB conversion contract.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "codec/color_transform.h"

namespace {

constexpr gjxl::Extent2D kExtent{9, 7};
constexpr size_t kStride = 12;

struct ImageStorage {
  std::array<std::vector<float>, 3> plane;

  explicit ImageStorage(float fill = -777.0f) {
    for (std::vector<float>& values : plane) {
      values.assign(kStride * kExtent.height, fill);
    }
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{
      gjxl::PlaneF32View{plane[0].data(), kExtent, kStride},
      gjxl::PlaneF32View{plane[1].data(), kExtent, kStride},
      gjxl::PlaneF32View{plane[2].data(), kExtent, kStride},
    }};
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{
      gjxl::ConstPlaneF32View{plane[0].data(), kExtent, kStride},
      gjxl::ConstPlaneF32View{plane[1].data(), kExtent, kStride},
      gjxl::ConstPlaneF32View{plane[2].data(), kExtent, kStride},
    }};
  }
};

void FillLinear(ImageStorage* image) {
  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      image->plane[0][y * kStride + x] =
        static_cast<float>(x) / static_cast<float>(kExtent.width - 1);
      image->plane[1][y * kStride + x] =
        static_cast<float>(y) / static_cast<float>(kExtent.height - 1);
      image->plane[2][y * kStride + x] =
        static_cast<float>((3 * x + 5 * y) % 17) / 16.0f;
    }
  }
}

bool PaddingIs(const ImageStorage& image, float value) {
  for (const std::vector<float>& plane : image.plane) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = kExtent.width; x < kStride; ++x) {
        if (plane[y * kStride + x] != value) {
          return false;
        }
      }
    }
  }
  return true;
}

bool CheckRoundTripAndInvariants() {
  ImageStorage linear;
  ImageStorage opsin;
  ImageStorage reconstructed;
  FillLinear(&linear);
  if (!gjxl::LinearRgbToOpsin(
        linear.ConstView(), 255.0f, opsin.View()).ok() ||
      !gjxl::OpsinToLinearRgb(
        opsin.ConstView(), 255.0f, reconstructed.View()).ok() ||
      !PaddingIs(opsin, -777.0f) ||
      !PaddingIs(reconstructed, -777.0f)) {
    return false;
  }

  float maximum_error = 0.0f;
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        maximum_error = std::max(
          maximum_error,
          std::abs(
            linear.plane[channel][y * kStride + x] -
            reconstructed.plane[channel][y * kStride + x]));
      }
    }
  }
  if (maximum_error > 2.0e-5f) {
    std::cerr << "Linear-RGB/XYB round trip is inaccurate: "
              << maximum_error << '\n';
    return false;
  }

  // Black maps to the XYB origin. Grayscale has X=0 and Y=B.
  if (std::abs(opsin.plane[0][0]) > 1.0e-8f ||
      std::abs(opsin.plane[1][0]) > 1.0e-7f ||
      std::abs(opsin.plane[2][0]) > 1.0e-7f) {
    std::cerr << "Black does not map to the XYB origin\n";
    return false;
  }
  ImageStorage gray;
  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      const float value = static_cast<float>(x + y + 1) / 17.0f;
      for (size_t channel = 0; channel < 3; ++channel) {
        gray.plane[channel][y * kStride + x] = value;
      }
    }
  }
  if (!gjxl::LinearRgbToOpsin(
        gray.ConstView(), 255.0f, gray.View()).ok()) {
    return false;
  }
  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      if (std::abs(gray.plane[0][y * kStride + x]) > 1.0e-6f ||
          std::abs(
            gray.plane[1][y * kStride + x] -
            gray.plane[2][y * kStride + x]) > 3.0e-5f) {
        std::cerr << "Grayscale XYB invariant failed\n";
        return false;
      }
    }
  }
  return true;
}

bool CheckValidationIsAtomic() {
  ImageStorage input;
  ImageStorage output(41.0f);
  FillLinear(&input);
  input.plane[1][3] = std::numeric_limits<float>::quiet_NaN();
  const auto original = output.plane;
  if (gjxl::LinearRgbToOpsin(
        input.ConstView(), 255.0f, output.View()).ok() ||
      output.plane != original ||
      gjxl::OpsinToLinearRgb(
        output.ConstView(), 0.0f, output.View()).ok() ||
      output.plane != original) {
    std::cerr << "Invalid color transform was not atomic\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckRoundTripAndInvariants() || !CheckValidationIsAtomic()) {
    return EXIT_FAILURE;
  }
  std::cout << "All color transform tests passed.\n";
  return EXIT_SUCCESS;
}
