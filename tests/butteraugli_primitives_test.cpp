// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates always-available native Butteraugli storage and blur primitives.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "butteraugli_test_tolerances.h"
#include "codec/butteraugli_internal.h"

namespace {

namespace bi = gjxl::butteraugli_internal;
namespace bt = gjxl::butteraugli_test;

constexpr float kPoison = -991.0f;

struct PlaneStorage {
  explicit PlaneStorage(gjxl::Extent2D image_extent, size_t padding = 4)
      : extent(image_extent), stride(image_extent.width + padding),
        values(stride * extent.height, kPoison) {}

  [[nodiscard]] gjxl::PlaneF32View View() {
    return {values.data(), extent, stride};
  }

  [[nodiscard]] gjxl::ConstPlaneF32View ConstView() const {
    return {values.data(), extent, stride};
  }

  [[nodiscard]] bool PaddingIsUntouched() const {
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = extent.width; x < stride; ++x) {
        if (values[y * stride + x] != kPoison) {
          return false;
        }
      }
    }
    return true;
  }

  gjxl::Extent2D extent;
  size_t stride;
  std::vector<float> values;
};

[[nodiscard]] bool IsInvalid(const gjxl::Status &status) {
  return status.code() == gjxl::StatusCode::kInvalidArgument;
}

[[nodiscard]] int64_t Mirror(int64_t coordinate, int64_t size) {
  while (coordinate < 0 || coordinate >= size) {
    coordinate = coordinate < 0 ? -coordinate - 1 : 2 * size - 1 - coordinate;
  }
  return coordinate;
}

[[nodiscard]] std::vector<double> GaussianKernel(float sigma) {
  const int radius = std::max(1, static_cast<int>(2.25f * std::abs(sigma)));
  const double scale = -1.0 / (2.0 * sigma * sigma);
  std::vector<double> kernel(2 * static_cast<size_t>(radius) + 1);
  for (int index = -radius; index <= radius; ++index) {
    kernel[static_cast<size_t>(index + radius)] =
        std::exp(scale * index * index);
  }
  return kernel;
}

[[nodiscard]] std::vector<float> ReferenceBlur(gjxl::ConstPlaneF32View input,
                                               float sigma) {

  const std::vector<double> kernel = GaussianKernel(sigma);
  const int64_t radius = static_cast<int64_t>(kernel.size() / 2);
  const int64_t width = static_cast<int64_t>(input.extent.width);
  const int64_t height = static_cast<int64_t>(input.extent.height);
  std::vector<double> horizontal(input.extent.width * input.extent.height);
  std::vector<float> result(horizontal.size());

  for (int64_t y = 0; y < height; ++y) {
    for (int64_t x = 0; x < width; ++x) {
      double sum = 0.0;
      double weight = 0.0;
      for (int64_t dx = -radius; dx <= radius; ++dx) {
        int64_t source_x = x + dx;
        if (radius == 2) {
          source_x = Mirror(source_x, width);
        } else if (source_x < 0 || source_x >= width) {
          continue;
        }
        const double kernel_weight = kernel[static_cast<size_t>(dx + radius)];
        sum += input.Row(static_cast<size_t>(y))[source_x] * kernel_weight;
        weight += kernel_weight;
      }
      horizontal[static_cast<size_t>(y) * input.extent.width +
                 static_cast<size_t>(x)] = sum / weight;
    }
  }

  for (int64_t y = 0; y < height; ++y) {
    for (int64_t x = 0; x < width; ++x) {
      double sum = 0.0;
      double weight = 0.0;
      for (int64_t dy = -radius; dy <= radius; ++dy) {
        int64_t source_y = y + dy;
        if (radius == 2) {
          source_y = Mirror(source_y, height);
        } else if (source_y < 0 || source_y >= height) {
          continue;
        }
        const double kernel_weight = kernel[static_cast<size_t>(dy + radius)];
        sum += horizontal[static_cast<size_t>(source_y) * input.extent.width +
                          static_cast<size_t>(x)] *
               kernel_weight;
        weight += kernel_weight;
      }
      result[static_cast<size_t>(y) * input.extent.width +
             static_cast<size_t>(x)] = static_cast<float>(sum / weight);
    }
  }
  return result;
}

[[nodiscard]] bool CheckStorage() {
  bi::OwnedPlaneF32 plane;
  if (plane.View().valid() || !plane.extent().empty() ||
      !plane.Resize({4, 3}).ok()) {
    return false;
  }
  gjxl::PlaneF32View plane_view = plane.View();
  if (!plane_view.valid() || plane_view.stride != 4 ||
      plane_view.Row(1) != plane_view.data + 4 || plane.size() != 12) {
    return false;
  }
  for (size_t index = 0; index < plane.size(); ++index) {
    plane_view.data[index] = static_cast<float>(index + 1);
  }
  float *const same_extent_address = plane_view.data;
  if (!plane.Resize({4, 3}).ok() || plane.View().data != same_extent_address ||
      plane.ConstView().data[7] != 8.0f || !plane.Resize({2, 5}).ok() ||
      plane.extent() != gjxl::Extent2D{2, 5}) {
    return false;
  }
  plane.View().data[0] = 17.0f;
  float *const resized_address = plane.View().data;
  if (!IsInvalid(plane.Resize({0, 5})) ||
      !IsInvalid(plane.Resize({std::numeric_limits<size_t>::max(), 2})) ||
      plane.extent() != gjxl::Extent2D{2, 5} ||
      plane.View().data != resized_address ||
      plane.ConstView().data[0] != 17.0f) {
    return false;
  }
  bi::OwnedPlaneF32 moved_plane = std::move(plane);
  if (!moved_plane.View().valid() || !plane.Resize({2, 5}).ok() ||
      !plane.View().valid()) {
    return false;
  }

  bi::OwnedImage3F image;
  if (image.View().valid() || !image.Resize({3, 2}).ok()) {
    return false;
  }
  gjxl::Image3FView image_view = image.View();
  if (!image_view.valid() || image.plane_size() != 6 ||
      image_view.plane[1].data != image_view.plane[0].data + 6 ||
      image_view.plane[2].data != image_view.plane[1].data + 6) {
    return false;
  }
  image_view.plane[2].data[4] = 23.0f;
  float *const image_address = image_view.plane[0].data;
  if (!image.Resize({3, 2}).ok() ||
      image.View().plane[0].data != image_address ||
      image.ConstView().plane[2].data[4] != 23.0f ||
      !image.Resize({2, 4}).ok() || image.plane_size() != 8 ||
      image.View().plane[1].data != image.View().plane[0].data + 8) {
    return false;
  }
  image.View().plane[2].data[7] = 29.0f;
  float *const resized_image_address = image.View().plane[0].data;
  if (!IsInvalid(image.Resize({3, 0})) ||
      !IsInvalid(
          image.Resize({std::numeric_limits<size_t>::max() / 2 + 1, 1})) ||
      image.extent() != gjxl::Extent2D{2, 4} ||
      image.View().plane[0].data != resized_image_address ||
      image.ConstView().plane[2].data[7] != 29.0f) {
    return false;
  }
  return true;
}

[[nodiscard]] bool CheckInvalidBlurRequests() {
  constexpr gjxl::Extent2D kExtent{8, 6};
  PlaneStorage input(kExtent);
  PlaneStorage output(kExtent);
  for (size_t y = 0; y < kExtent.height; ++y) {
    for (size_t x = 0; x < kExtent.width; ++x) {
      input.View().Row(y)[x] = static_cast<float>(x + 3 * y);
    }
  }
  const std::vector<float> original = output.values;
  bi::BlurScratch scratch;
  const auto check = [&](gjxl::ConstPlaneF32View source, float sigma,
                         bi::BlurScratch *requested_scratch,
                         gjxl::PlaneF32View destination) {
    const gjxl::Status status =
        bi::GaussianBlur(source, sigma, requested_scratch, destination);
    return IsInvalid(status) && output.values == original;
  };

  gjxl::ConstPlaneF32View invalid_input = input.ConstView();
  invalid_input.data = nullptr;
  if (!check(invalid_input, bi::kOpsinBlurSigma, &scratch, output.View())) {
    return false;
  }
  invalid_input = input.ConstView();
  invalid_input.stride = kExtent.width - 1;
  if (!check(invalid_input, bi::kOpsinBlurSigma, &scratch, output.View())) {
    return false;
  }
  invalid_input = input.ConstView();
  invalid_input.stride = std::numeric_limits<size_t>::max();
  if (!check(invalid_input, bi::kOpsinBlurSigma, &scratch, output.View())) {
    return false;
  }
  gjxl::PlaneF32View invalid_output = output.View();
  invalid_output.data = nullptr;
  if (!check(input.ConstView(), bi::kOpsinBlurSigma, &scratch,
             invalid_output)) {
    return false;
  }
  invalid_output = output.View();
  invalid_output.extent.width -= 1;
  if (!check(input.ConstView(), bi::kOpsinBlurSigma, &scratch,
             invalid_output)) {
    return false;
  }
  invalid_output = output.View();
  invalid_output.stride = kExtent.width - 1;
  if (!check(input.ConstView(), bi::kOpsinBlurSigma, &scratch,
             invalid_output)) {
    return false;
  }
  invalid_output = output.View();
  invalid_output.stride = std::numeric_limits<size_t>::max();
  if (!check(input.ConstView(), bi::kOpsinBlurSigma, &scratch,
             invalid_output) ||
      !check(input.ConstView(), bi::kOpsinBlurSigma, nullptr, output.View()) ||
      !check(input.ConstView(), 0.0f, &scratch, output.View()) ||
      !check(input.ConstView(), -1.2f, &scratch, output.View()) ||
      !check(input.ConstView(), std::numeric_limits<float>::infinity(),
             &scratch, output.View()) ||
      !check(input.ConstView(), std::numeric_limits<float>::quiet_NaN(),
             &scratch, output.View())) {
    return false;
  }
  for (float sigma : std::array<float, 4>{0.5f, 2.0f, 2.3f, 3.6f}) {
    if (!check(input.ConstView(), sigma, &scratch, output.View())) {
      return false;
    }
  }
  const gjxl::PlaneF32View alias{input.values.data(), kExtent, input.stride};
  return check(input.ConstView(), bi::kOpsinBlurSigma, &scratch, alias);
}

[[nodiscard]] bool CheckBlurBehavior() {
  bi::BlurScratch scratch;
  const std::array<gjxl::Extent2D, 4> extents = {
      gjxl::Extent2D{9, 7},
      gjxl::Extent2D{3, 7},
      gjxl::Extent2D{7, 3},
      gjxl::Extent2D{1, 1},
  };
  for (gjxl::Extent2D extent : extents) {
    PlaneStorage input(extent, 3);
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = 0; x < extent.width; ++x) {
        float value =
            0.013f * static_cast<float>(x + 7 * y) +
            0.07f * std::sin(0.41f * static_cast<float>(3 * x + 5 * y));
        if (x == 0 && y == std::min<size_t>(1, extent.height - 1)) {
          value += 1.0f;
        }
        if (x + 2 == extent.width && y + 1 == extent.height) {
          value -= 0.25f;
        }
        input.View().Row(y)[x] = value;
      }
    }
    const std::vector<float> original_input = input.values;
    for (float sigma : bi::kPinnedBlurSigmas) {
      PlaneStorage output(extent, 5);
      const std::vector<float> expected =
          ReferenceBlur(input.ConstView(), sigma);
      const gjxl::Status status =
          bi::GaussianBlur(input.ConstView(), sigma, &scratch, output.View());
      if (!status.ok() || !output.PaddingIsUntouched() ||
          input.values != original_input) {
        return false;
      }
      for (size_t y = 0; y < extent.height; ++y) {
        for (size_t x = 0; x < extent.width; ++x) {
          const float actual = output.ConstView().Row(y)[x];
          const float reference = expected[y * extent.width + x];
          if (!std::isfinite(actual) ||
              std::abs(actual - reference) >
                  bt::PrimitiveReferenceTolerance(reference)) {
            std::cerr << "Native blur boundary mismatch at " << extent.width
                      << 'x' << extent.height << " sigma=" << sigma << '\n';
            return false;
          }
        }
      }
    }
  }

  PlaneStorage constant({11, 6}, 2);
  for (size_t y = 0; y < constant.extent.height; ++y) {
    std::fill_n(constant.View().Row(y), constant.extent.width, 0.375f);
  }
  for (float sigma : bi::kPinnedBlurSigmas) {
    PlaneStorage output(constant.extent, 6);
    if (!bi::GaussianBlur(constant.ConstView(), sigma, &scratch, output.View())
             .ok() ||
        !output.PaddingIsUntouched()) {
      return false;
    }
    for (size_t y = 0; y < output.extent.height; ++y) {
      for (size_t x = 0; x < output.extent.width; ++x) {
        if (std::abs(output.ConstView().Row(y)[x] - 0.375f) >
            bt::kPrimitiveConstantTolerance) {
          std::cerr << "Native blur did not preserve a constant plane\n";
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace

int main() {
  if (!CheckStorage() || !CheckInvalidBlurRequests() || !CheckBlurBehavior()) {
    return EXIT_FAILURE;
  }
  std::cout << "All native Butteraugli primitive tests passed.\n";
  return EXIT_SUCCESS;
}
