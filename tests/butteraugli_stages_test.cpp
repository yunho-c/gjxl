// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates always-available native Butteraugli opsin and frequency stages.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "butteraugli_test_tolerances.h"
#include "codec/butteraugli_internal.h"

namespace {

namespace bi = gjxl::butteraugli_internal;
namespace bt = gjxl::butteraugli_test;

constexpr float kPoison = -991.0f;

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D image_extent, size_t padding = 4)
      : extent(image_extent), stride(image_extent.width + padding) {
    for (std::vector<float> &values : plane) {
      values.assign(stride * extent.height, kPoison);
    }
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{{{plane[0].data(), extent, stride},
              {plane[1].data(), extent, stride},
              {plane[2].data(), extent, stride}}}};
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{{{plane[0].data(), extent, stride},
              {plane[1].data(), extent, stride},
              {plane[2].data(), extent, stride}}}};
  }

  [[nodiscard]] bool PaddingIsUntouched() const {
    for (const std::vector<float> &values : plane) {
      for (size_t y = 0; y < extent.height; ++y) {
        for (size_t x = extent.width; x < stride; ++x) {
          if (values[y * stride + x] != kPoison) {
            return false;
          }
        }
      }
    }
    return true;
  }

  gjxl::Extent2D extent;
  size_t stride;
  std::array<std::vector<float>, 3> plane;
};

[[nodiscard]] bool IsInvalid(const gjxl::Status &status) {
  return status.code() == gjxl::StatusCode::kInvalidArgument;
}

void FillImage(ImageStorage *image) {
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < image->extent.height; ++y) {
      for (size_t x = 0; x < image->extent.width; ++x) {
        image->plane[channel][y * image->stride + x] =
            0.03f * static_cast<float>(channel + 1) +
            0.011f * static_cast<float>(x + 3 * y) +
            0.07f *
                std::sin(0.37f * static_cast<float>(5 * x + 2 * y + channel));
      }
    }
  }
}

[[nodiscard]] std::vector<float> Snapshot(const bi::OwnedPsychoImage &image) {
  const std::array<gjxl::ConstPlaneF32View, 10> planes = {
      image.LowFrequencyView().plane[0],
      image.LowFrequencyView().plane[1],
      image.LowFrequencyView().plane[2],
      image.MediumFrequencyView().plane[0],
      image.MediumFrequencyView().plane[1],
      image.MediumFrequencyView().plane[2],
      image.HighFrequencyView(0),
      image.HighFrequencyView(1),
      image.UltraHighFrequencyView(0),
      image.UltraHighFrequencyView(1),
  };
  std::vector<float> values;
  values.reserve(10 * image.plane_size());
  for (gjxl::ConstPlaneF32View plane : planes) {
    for (size_t y = 0; y < plane.extent.height; ++y) {
      values.insert(values.end(), plane.Row(y),
                    plane.Row(y) + plane.extent.width);
    }
  }
  return values;
}

[[nodiscard]] bool CheckPsychoStorage() {
  bi::OwnedPsychoImage image;
  if (image.LowFrequencyView().valid() || !image.extent().empty() ||
      !image.Resize({4, 3}).ok()) {
    return false;
  }
  if (!image.LowFrequencyView().valid() ||
      !image.MediumFrequencyView().valid() ||
      !image.HighFrequencyView(0).valid() ||
      !image.UltraHighFrequencyView(1).valid() ||
      image.HighFrequencyView(2).valid() || image.plane_size() != 12) {
    return false;
  }
  float *const first = image.LowFrequencyView().plane[0].data;
  if (image.LowFrequencyView().plane[1].data != first + 12 ||
      image.MediumFrequencyView().plane[0].data != first + 36 ||
      image.HighFrequencyView(0).data != first + 72 ||
      image.UltraHighFrequencyView(1).data != first + 108) {
    return false;
  }
  image.UltraHighFrequencyView(1).data[11] = 19.0f;
  if (!image.Resize({4, 3}).ok() ||
      image.LowFrequencyView().plane[0].data != first ||
      image.UltraHighFrequencyView(1).data[11] != 19.0f ||
      !image.Resize({2, 5}).ok()) {
    return false;
  }
  image.LowFrequencyView().plane[0].data[0] = 23.0f;
  float *const resized = image.LowFrequencyView().plane[0].data;
  if (!IsInvalid(image.Resize({0, 5})) ||
      !IsInvalid(
          image.Resize({std::numeric_limits<size_t>::max() / 8 + 1, 1})) ||
      image.extent() != gjxl::Extent2D{2, 5} ||
      image.LowFrequencyView().plane[0].data != resized ||
      image.LowFrequencyView().plane[0].data[0] != 23.0f) {
    return false;
  }
  return true;
}

[[nodiscard]] bool CheckPointTransforms() {
  constexpr float kClampMultiplier = 0.724216145665f;
  const auto near = [](float actual, float expected) {
    return std::abs(actual - expected) <= 2.0e-6f;
  };
  return bi::RemoveRangeAroundZero(0.29f, 0.29f) == 0.0f &&
         bi::RemoveRangeAroundZero(-0.29f, 0.29f) == 0.0f &&
         near(bi::RemoveRangeAroundZero(0.5f, 0.29f), 0.21f) &&
         near(bi::RemoveRangeAroundZero(-0.5f, 0.29f), -0.21f) &&
         near(bi::AmplifyRangeAroundZero(0.1f, 0.1f), 0.2f) &&
         near(bi::AmplifyRangeAroundZero(-0.1f, 0.1f), -0.2f) &&
         near(bi::AmplifyRangeAroundZero(0.3f, 0.1f), 0.4f) &&
         near(bi::AmplifyRangeAroundZero(-0.3f, 0.1f), -0.4f) &&
         bi::MaximumClamp(3.0f, 3.0f) == 3.0f &&
         bi::MaximumClamp(-3.0f, 3.0f) == -3.0f &&
         near(bi::MaximumClamp(5.0f, 3.0f),
              std::fma(2.0f, kClampMultiplier, 3.0f)) &&
         near(bi::MaximumClamp(-5.0f, 3.0f),
              std::fma(-2.0f, kClampMultiplier, -3.0f)) &&
         bi::SuppressXByY(0.0f, 2.0f) == 2.0f &&
         std::abs(bi::SuppressXByY(1000.0f, 2.0f)) < 2.0f;
}

[[nodiscard]] bool CheckInvalidOpsinRequests() {
  constexpr gjxl::Extent2D kExtent{8, 6};
  ImageStorage input(kExtent);
  ImageStorage output(kExtent);
  FillImage(&input);
  const auto original_output = output.plane;
  bi::OpsinScratch scratch;
  const auto check = [&](gjxl::ConstImage3FView source, float intensity,
                         bi::OpsinScratch *requested_scratch,
                         gjxl::Image3FView destination) {
    return IsInvalid(bi::OpsinDynamicsImage(source, intensity,
                                            requested_scratch, destination)) &&
           output.plane == original_output;
  };

  gjxl::ConstImage3FView invalid_input = input.ConstView();
  invalid_input.plane[0].data = nullptr;
  if (!check(invalid_input, 80.0f, &scratch, output.View())) {
    return false;
  }
  invalid_input = input.ConstView();
  invalid_input.plane[1].extent.width -= 1;
  if (!check(invalid_input, 80.0f, &scratch, output.View())) {
    return false;
  }
  gjxl::Image3FView invalid_output = output.View();
  invalid_output.plane[2].stride = kExtent.width - 1;
  if (!check(input.ConstView(), 80.0f, &scratch, invalid_output) ||
      !check(input.ConstView(), 80.0f, nullptr, output.View()) ||
      !check(input.ConstView(), 0.0f, &scratch, output.View()) ||
      !check(input.ConstView(), -1.0f, &scratch, output.View()) ||
      !check(input.ConstView(), std::numeric_limits<float>::infinity(),
             &scratch, output.View()) ||
      !check(input.ConstView(), std::numeric_limits<float>::quiet_NaN(),
             &scratch, output.View())) {
    return false;
  }
  input.plane[1][2 * input.stride + 3] =
      std::numeric_limits<float>::quiet_NaN();
  if (!check(input.ConstView(), 80.0f, &scratch, output.View())) {
    return false;
  }
  input.plane[1][2 * input.stride + 3] = 0.2f;
  input.plane[0][0] = std::numeric_limits<float>::max();
  return check(input.ConstView(), std::numeric_limits<float>::max(), &scratch,
               output.View());
}

[[nodiscard]] bool CheckOpsinBehavior() {
  bi::OpsinScratch scratch;
  for (gjxl::Extent2D extent : std::array<gjxl::Extent2D, 4>{
           gjxl::Extent2D{1, 1}, gjxl::Extent2D{3, 7}, gjxl::Extent2D{7, 3},
           gjxl::Extent2D{9, 8}}) {
    ImageStorage input(extent, 3);
    ImageStorage output(extent, 7);
    FillImage(&input);
    const auto original_input = input.plane;
    if (!bi::OpsinDynamicsImage(input.ConstView(), 80.0f, &scratch,
                                output.View())
             .ok() ||
        input.plane != original_input || !input.PaddingIsUntouched() ||
        !output.PaddingIsUntouched()) {
      return false;
    }

    ImageStorage in_place = input;
    const gjxl::ConstImage3FView in_place_source = in_place.ConstView();
    const gjxl::Image3FView in_place_destination = in_place.View();
    if (!bi::OpsinDynamicsImage(in_place_source, 80.0f, &scratch,
                                in_place_destination)
             .ok() ||
        !in_place.PaddingIsUntouched()) {
      return false;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t y = 0; y < extent.height; ++y) {
        for (size_t x = 0; x < extent.width; ++x) {
          const float expected = output.plane[channel][y * output.stride + x];
          const float actual = in_place.plane[channel][y * in_place.stride + x];
          if (actual != expected || !std::isfinite(actual)) {
            return false;
          }
        }
      }
    }
  }

  ImageStorage black({5, 4});
  for (std::vector<float> &plane : black.plane) {
    for (size_t y = 0; y < black.extent.height; ++y) {
      std::fill_n(plane.data() + y * black.stride, black.extent.width, 0.0f);
    }
  }
  ImageStorage output(black.extent);
  if (!bi::OpsinDynamicsImage(black.ConstView(), 255.0f, &scratch,
                              output.View())
           .ok()) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    const float expected = output.plane[channel][0];
    for (size_t y = 0; y < output.extent.height; ++y) {
      for (size_t x = 0; x < output.extent.width; ++x) {
        if (output.plane[channel][y * output.stride + x] != expected) {
          return false;
        }
      }
    }
  }
  return output.plane[0][0] == 0.0f && output.plane[1][0] > 0.0f &&
         output.plane[2][0] > 0.0f;
}

[[nodiscard]] bool CheckFrequencyBehavior() {
  constexpr gjxl::Extent2D kExtent{7, 5};
  ImageStorage constant(kExtent);
  constexpr std::array<float, 3> kValues = {0.2f, -0.3f, 0.7f};
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      std::fill_n(constant.plane[channel].data() + y * constant.stride,
                  kExtent.width, kValues[channel]);
    }
  }
  const auto original_input = constant.plane;
  bi::FrequencyScratch scratch;
  bi::OwnedPsychoImage output;
  if (!bi::SeparateFrequencies(constant.ConstView(), &scratch, &output).ok() ||
      constant.plane != original_input || !constant.PaddingIsUntouched()) {
    return false;
  }
  const std::array<float, 3> expected_low = {
      kValues[0] * 33.832837186260f,
      kValues[1] * 14.458268100570f,
      std::fma(-0.362267051518f, kValues[1], kValues[2]) * 49.87984651440f,
  };
  const bi::OwnedPsychoImage &const_output = output;
  for (size_t channel = 0; channel < 3; ++channel) {
    const gjxl::ConstPlaneF32View low =
        const_output.LowFrequencyView().plane[channel];
    const gjxl::ConstPlaneF32View medium =
        const_output.MediumFrequencyView().plane[channel];
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        if (std::abs(low.Row(y)[x] - expected_low[channel]) >
                bt::MapTolerance(expected_low[channel]) ||
            std::abs(medium.Row(y)[x]) > bt::kPrimitiveConstantTolerance) {
          std::cerr << "Constant LF/MF mismatch: channel=" << channel
                    << " x=" << x << " y=" << y << " lf=" << low.Row(y)[x]
                    << " expected_lf=" << expected_low[channel]
                    << " mf=" << medium.Row(y)[x] << '\n';
          return false;
        }
      }
    }
  }
  for (size_t channel = 0; channel < 2; ++channel) {
    for (size_t y = 0; y < kExtent.height; ++y) {
      for (size_t x = 0; x < kExtent.width; ++x) {
        if (std::abs(output.HighFrequencyView(channel).Row(y)[x]) >
                bt::kPrimitiveConstantTolerance ||
            std::abs(output.UltraHighFrequencyView(channel).Row(y)[x]) >
                bt::kPrimitiveConstantTolerance) {
          std::cerr << "Constant HF/UHF mismatch: channel=" << channel
                    << " x=" << x << " y=" << y
                    << " hf=" << output.HighFrequencyView(channel).Row(y)[x]
                    << " uhf="
                    << output.UltraHighFrequencyView(channel).Row(y)[x] << '\n';
          return false;
        }
      }
    }
  }

  const std::vector<float> original_output = Snapshot(output);
  const gjxl::Extent2D original_extent = output.extent();
  gjxl::ConstImage3FView invalid = constant.ConstView();
  invalid.plane[0].data = nullptr;
  if (!IsInvalid(bi::SeparateFrequencies(invalid, &scratch, &output)) ||
      output.extent() != original_extent ||
      Snapshot(output) != original_output ||
      !IsInvalid(
          bi::SeparateFrequencies(constant.ConstView(), nullptr, &output)) ||
      Snapshot(output) != original_output ||
      !IsInvalid(
          bi::SeparateFrequencies(constant.ConstView(), &scratch, nullptr))) {
    return false;
  }
  constant.plane[2][0] = std::numeric_limits<float>::infinity();
  return IsInvalid(bi::SeparateFrequencies(constant.ConstView(), &scratch,
                                           &output)) &&
         output.extent() == original_extent &&
         Snapshot(output) == original_output;
}

[[nodiscard]] bool CheckFrequencyScratchReuse() {
  bi::FrequencyScratch scratch;
  bi::OwnedPsychoImage output;
  for (gjxl::Extent2D extent : std::array<gjxl::Extent2D, 5>{
           gjxl::Extent2D{9, 7}, gjxl::Extent2D{3, 7}, gjxl::Extent2D{7, 3},
           gjxl::Extent2D{1, 1}, gjxl::Extent2D{9, 7}}) {
    ImageStorage input(extent);
    FillImage(&input);
    if (!bi::SeparateFrequencies(input.ConstView(), &scratch, &output).ok() ||
        output.extent() != extent ||
        Snapshot(output).size() != 10 * extent.width * extent.height) {
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  const std::array<std::pair<const char *, bool (*)()>, 6> checks = {{
      {"psycho storage", CheckPsychoStorage},
      {"point transforms", CheckPointTransforms},
      {"invalid opsin requests", CheckInvalidOpsinRequests},
      {"opsin behavior", CheckOpsinBehavior},
      {"frequency behavior", CheckFrequencyBehavior},
      {"frequency scratch reuse", CheckFrequencyScratchReuse},
  }};
  for (const auto &[name, check] : checks) {
    if (!check()) {
      std::cerr << "Native Butteraugli check failed: " << name << '\n';
      return EXIT_FAILURE;
    }
  }
  std::cout << "All native Butteraugli opsin/frequency tests passed.\n";
  return EXIT_SUCCESS;
}
