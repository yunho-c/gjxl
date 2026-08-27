// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates decoder loop-filter ordering, aliasing, and atomicity.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "codec/epf.h"
#include "codec/gaborish.h"
#include "codec/loop_filter.h"
#include "core/image_buffer.h"
#include "core/image_ops.h"

namespace {

constexpr gjxl::Extent2D kExtent{16, 16};
constexpr gjxl::Extent2D kBlockExtent{2, 2};
constexpr size_t kStride = 19;

struct ImageStorage {
  std::array<std::vector<float>, 3> plane;

  explicit ImageStorage(float fill) {
    for (std::vector<float>& values : plane) {
      values.assign(kStride * kExtent.height, fill);
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{
      gjxl::ConstPlaneF32View{plane[0].data(), kExtent, kStride},
      gjxl::ConstPlaneF32View{plane[1].data(), kExtent, kStride},
      gjxl::ConstPlaneF32View{plane[2].data(), kExtent, kStride},
    }};
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{
      gjxl::PlaneF32View{plane[0].data(), kExtent, kStride},
      gjxl::PlaneF32View{plane[1].data(), kExtent, kStride},
      gjxl::PlaneF32View{plane[2].data(), kExtent, kStride},
    }};
  }
};

bool SameActivePixels(
  const ImageStorage& left,
  const ImageStorage& right) {

  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        if (left.plane[channel][y * kStride + x] !=
            right.plane[channel][y * kStride + x]) {
          return false;
        }
      }
    }
  }
  return true;
}

bool CheckComposition() {
  ImageStorage input(-111.0f);
  ImageStorage gaborished(-222.0f);
  ImageStorage expected(-333.0f);
  ImageStorage combined(-444.0f);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        input.plane[channel][y * kStride + x] =
          0.16f * std::sin(
            0.13f * static_cast<float>((channel + 1) * x + 2 * y)) +
          0.02f * static_cast<float>((7 * x + 5 * y) % 11);
      }
    }
  }
  std::array<float, 4> inverse_sigma = {-1.2f, -2.1f, -1.7f, -2.8f};
  const gjxl::ConstPlaneF32View sigma_view{
    inverse_sigma.data(), kBlockExtent, kBlockExtent.width};
  gjxl::LoopFilterOptions options;
  if (!gjxl::ApplyGaborish(
        input.ConstView(),
        options.gaborish_options,
        gaborished.View()).ok() ||
      !gjxl::ApplyEpf(
        gaborished.ConstView(),
        sigma_view,
        options.epf_options,
        expected.View()).ok() ||
      !gjxl::ApplyLoopFilters(
        input.ConstView(),
        sigma_view,
        options,
        combined.View()).ok()) {
    std::cerr << "Loop-filter composition failed\n";
    return false;
  }

  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        if (combined.plane[channel][y * kStride + x] !=
            expected.plane[channel][y * kStride + x]) {
          std::cerr << "Loop filters ran in the wrong order\n";
          return false;
        }
      }
    }
  }

  ImageStorage in_place = input;
  if (!gjxl::ApplyLoopFilters(
        in_place.ConstView(),
        sigma_view,
        options,
        in_place.View()).ok()) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        if (in_place.plane[channel][y * kStride + x] !=
            expected.plane[channel][y * kStride + x]) {
          std::cerr << "In-place loop filtering differs\n";
          return false;
        }
      }
    }
  }

  ImageStorage single_stage(-555.0f);
  options.epf_options.iterations = 0;
  if (!gjxl::ApplyLoopFilters(
        input.ConstView(), {}, options, single_stage.View()).ok() ||
      !SameActivePixels(single_stage, gaborished)) {
    std::cerr << "Gaborish-only loop filtering differs\n";
    return false;
  }

  ImageStorage epf_only(-666.0f);
  ImageStorage epf_expected(-777.0f);
  options.gaborish = false;
  options.epf_options.iterations = 2;
  if (!gjxl::ApplyEpf(
        input.ConstView(),
        sigma_view,
        options.epf_options,
        epf_expected.View()).ok() ||
      !gjxl::ApplyLoopFilters(
        input.ConstView(),
        sigma_view,
        options,
        epf_only.View()).ok() ||
      !SameActivePixels(epf_only, epf_expected)) {
    std::cerr << "EPF-only loop filtering differs\n";
    return false;
  }

  options.gaborish = false;
  options.epf_options.iterations = 0;
  if (!gjxl::ApplyLoopFilters(
        input.ConstView(), {}, options, combined.View()).ok()) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        if (combined.plane[channel][y * kStride + x] !=
            input.plane[channel][y * kStride + x]) {
          std::cerr << "Disabled loop filters did not copy input\n";
          return false;
        }
      }
    }
  }

  in_place = input;
  if (!gjxl::ApplyLoopFilters(
        in_place.ConstView(), {}, options, in_place.View()).ok() ||
      !SameActivePixels(in_place, input)) {
    std::cerr << "Disabled in-place loop filtering changed input\n";
    return false;
  }

  const auto original = combined.plane;
  options.epf_options.iterations = 4;
  if (gjxl::ApplyLoopFilters(
        input.ConstView(), sigma_view, options, combined.View()).ok() ||
      combined.plane != original) {
    std::cerr << "Invalid loop filtering was accepted or not atomic\n";
    return false;
  }
  return true;
}

bool CheckDeclaredImageBoundary() {
  constexpr gjxl::Extent2D kOddExtent{13, 9};
  constexpr gjxl::Extent2D kOddBlocks{2, 2};
  gjxl::Image3FBuffer input(kOddExtent);
  gjxl::Image3FBuffer output(kOddExtent);
  gjxl::Image3FBuffer in_place(kOddExtent);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kOddExtent.height; ++y) {
      for (size_t x = 0; x < kOddExtent.width; ++x) {
        input.plane(channel)[y * kOddExtent.width + x] =
          0.07f * static_cast<float>(channel + 1) +
          0.013f * static_cast<float>(x * x + 3 * y);
      }
    }
  }
  gjxl::CopyImage(input.const_view(), in_place.view());
  const std::array<float, 4> inverse_sigma = {
    -1.2f, -2.1f, -1.7f, -2.8f};
  const gjxl::ConstPlaneF32View sigma{
    inverse_sigma.data(), kOddBlocks, kOddBlocks.width};
  if (!gjxl::ApplyLoopFilters(
        input.const_view(), sigma, {}, output.view()).ok() ||
      !gjxl::ApplyLoopFilters(
        in_place.const_view(), sigma, {}, in_place.view()).ok()) {
    std::cerr << "Odd-sized loop filtering failed\n";
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t index = 0;
         index < kOddExtent.width * kOddExtent.height; ++index) {
      if (output.plane(channel)[index] != in_place.plane(channel)[index]) {
        std::cerr << "Odd-sized in-place loop filtering differs\n";
        return false;
      }
    }
  }

  const std::vector<float> original(
    output.plane(0).begin(), output.plane(0).end());
  const std::array<float, 1> wrong_sigma = {-1.0f};
  if (gjxl::ApplyLoopFilters(
        input.const_view(),
        {wrong_sigma.data(), {1, 1}, 1},
        {}, output.view()).ok() ||
      !std::equal(
        output.plane(0).begin(), output.plane(0).end(), original.begin())) {
    std::cerr << "Invalid odd-sized loop filtering was not atomic\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  return CheckComposition() && CheckDeclaredImageBoundary()
    ? EXIT_SUCCESS
    : EXIT_FAILURE;
}
