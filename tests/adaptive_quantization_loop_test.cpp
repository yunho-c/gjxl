// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates iterative AQ updates around the full perceptual round trip.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "codec/adaptive_quantization.h"
#include "codec/color_transform.h"

namespace {

constexpr gjxl::Extent2D kOriginalExtent{21, 13};
constexpr gjxl::Extent2D kPaddedExtent{24, 16};
constexpr gjxl::Extent2D kBlockExtent{3, 2};

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D image_extent, float fill = -777.0f)
      : extent(image_extent), stride(image_extent.width + 3) {
    for (std::vector<float>& values : plane) {
      values.assign(stride * extent.height, fill);
    }
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{
      gjxl::PlaneF32View{plane[0].data(), extent, stride},
      gjxl::PlaneF32View{plane[1].data(), extent, stride},
      gjxl::PlaneF32View{plane[2].data(), extent, stride},
    }};
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{
      gjxl::ConstPlaneF32View{plane[0].data(), extent, stride},
      gjxl::ConstPlaneF32View{plane[1].data(), extent, stride},
      gjxl::ConstPlaneF32View{plane[2].data(), extent, stride},
    }};
  }

  gjxl::Extent2D extent;
  size_t stride;
  std::array<std::vector<float>, 3> plane;
};

void FillPaddedLinear(ImageStorage* padded, ImageStorage* original) {
  for (size_t y = 0; y < kPaddedExtent.height; ++y) {
    const size_t source_y = std::min(y, kOriginalExtent.height - 1);
    for (size_t x = 0; x < kPaddedExtent.width; ++x) {
      const size_t source_x = std::min(x, kOriginalExtent.width - 1);
      const float fx = static_cast<float>(source_x);
      const float fy = static_cast<float>(source_y);
      const std::array<float, 3> rgb = {
        std::clamp(
          0.12f + 0.025f * fx + 0.07f * std::sin(0.31f * fy),
          0.0f,
          1.0f),
        std::clamp(
          0.18f + 0.031f * fy + 0.05f * std::cos(0.27f * fx),
          0.0f,
          1.0f),
        ((source_x / 3 + source_y / 2) & 1u) == 0 ? 0.08f : 0.86f,
      };
      for (size_t channel = 0; channel < 3; ++channel) {
        padded->plane[channel][y * padded->stride + x] = rgb[channel];
        if (x < kOriginalExtent.width && y < kOriginalExtent.height) {
          original->plane[channel][y * original->stride + x] = rgb[channel];
        }
      }
    }
  }
}

struct AqStorage {
  static constexpr size_t kBlockStride = kBlockExtent.width + 2;

  explicit AqStorage(float fill = -777.0f)
      : reconstructed(kOriginalExtent, fill),
        quant_field(kBlockStride * kBlockExtent.height, fill),
        raw_quant(kBlockStride * kBlockExtent.height, -777),
        block_distance(kBlockStride * kBlockExtent.height, fill) {}

  [[nodiscard]] gjxl::AdaptiveQuantizationOutput Output() {
    return {
      .quant_field = {
        quant_field.data(), kBlockExtent, kBlockStride},
      .raw_quant_field = {
        raw_quant.data(), kBlockExtent, kBlockStride},
      .block_distance_map = {
        block_distance.data(), kBlockExtent, kBlockStride},
      .reconstructed_linear_rgb = reconstructed.View(),
      .quantizer = &quantizer,
      .color_correlation = &color_correlation,
      .score_history = &score_history,
    };
  }

  ImageStorage reconstructed;
  std::vector<float> quant_field;
  std::vector<int32_t> raw_quant;
  std::vector<float> block_distance;
  gjxl::Quantizer quantizer;
  gjxl::ColorCorrelationMap color_correlation;
  std::vector<double> score_history;
};

bool PaddingIsUntouched(const AqStorage& storage) {
  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = kBlockExtent.width; x < AqStorage::kBlockStride; ++x) {
      if (storage.quant_field[y * AqStorage::kBlockStride + x] != -777.0f ||
          storage.raw_quant[y * AqStorage::kBlockStride + x] != -777 ||
          storage.block_distance[y * AqStorage::kBlockStride + x] !=
            -777.0f) {
        return false;
      }
    }
  }
  for (const std::vector<float>& plane : storage.reconstructed.plane) {
    for (size_t y = 0; y < kOriginalExtent.height; ++y) {
      for (size_t x = kOriginalExtent.width;
           x < storage.reconstructed.stride;
           ++x) {
        if (plane[y * storage.reconstructed.stride + x] != -777.0f) {
          return false;
        }
      }
    }
  }
  return true;
}

bool CheckLoopAndUpdateRule() {
  ImageStorage original(kOriginalExtent);
  ImageStorage padded_linear(kPaddedExtent);
  ImageStorage opsin(kPaddedExtent);
  FillPaddedLinear(&padded_linear, &original);
  if (!gjxl::LinearRgbToOpsin(
        padded_linear.ConstView(), 255.0f, opsin.View()).ok()) {
    return false;
  }

  gjxl::AcStrategyGrid strategies;
  if (!gjxl::AcStrategyGrid::Create(kBlockExtent, &strategies).ok()) {
    return false;
  }
  strategies.fill_dct8();

  constexpr std::array<float, 6> kInitial = {
    0.41f, 0.44f, 0.47f,
    0.43f, 0.46f, 0.49f,
  };
  std::array<uint8_t, 6> sharpness{};
  sharpness.fill(4);
  const gjxl::ConstPlaneF32View initial{
    kInitial.data(), kBlockExtent, kBlockExtent.width};
  const gjxl::ConstPlaneU8View sharpness_view{
    sharpness.data(), kBlockExtent, kBlockExtent.width};

  AqStorage baseline;
  gjxl::AdaptiveQuantizationOptions baseline_options;
  baseline_options.butteraugli_target = 1.1f;
  baseline_options.iterations = 0;
  if (!gjxl::FindBestQuantization(
        original.ConstView(),
        opsin.ConstView(),
        strategies,
        initial,
        sharpness_view,
        baseline_options,
        baseline.Output()).ok() ||
      baseline.score_history.size() != 1 ||
      !baseline.quantizer.valid() ||
      !baseline.color_correlation.valid() ||
      !PaddingIsUntouched(baseline)) {
    std::cerr << "Zero-update AQ evaluation failed\n";
    return false;
  }

  const auto [minimum_it, maximum_it] = std::minmax_element(
    kInitial.begin(), kInitial.end());
  const float ratio = *maximum_it / *minimum_it;
  const float deviation = std::sqrt(250.0f / ratio);
  const float asymmetry = std::min(2.0f, deviation);
  const float lower = *minimum_it / (asymmetry * deviation);
  const float upper = *maximum_it * (deviation / asymmetry);
  std::array<float, 6> expected{};
  for (size_t index = 0; index < expected.size(); ++index) {
    const float difference =
      baseline.block_distance[
        (index / kBlockExtent.width) * AqStorage::kBlockStride +
        index % kBlockExtent.width] /
      baseline_options.butteraugli_target;
    if (difference <= 1.0f) {
      expected[index] = kInitial[index] * std::pow(difference, 0.2f);
    } else {
      expected[index] = kInitial[index] * difference;
      const long old_raw = std::lround(
        kInitial[index] * baseline.quantizer.inverse_global_scale());
      const long new_raw = std::lround(
        expected[index] * baseline.quantizer.inverse_global_scale());
      if (old_raw == new_raw) {
        expected[index] = kInitial[index] + baseline.quantizer.scale();
      }
    }
    expected[index] = std::clamp(expected[index], lower, upper);
  }

  AqStorage updated;
  gjxl::AdaptiveQuantizationOptions updated_options = baseline_options;
  updated_options.iterations = 1;
  const gjxl::Status status = gjxl::FindBestQuantization(
    original.ConstView(),
    opsin.ConstView(),
    strategies,
    initial,
    sharpness_view,
    updated_options,
    updated.Output());
  if (!status.ok() ||
      updated.score_history.size() != 2 ||
      !PaddingIsUntouched(updated)) {
    std::cerr << "One-update AQ evaluation failed: "
              << status.message() << '\n';
    return false;
  }
  for (size_t index = 0; index < expected.size(); ++index) {
    const size_t x = index % kBlockExtent.width;
    const size_t y = index / kBlockExtent.width;
    if (std::abs(
          updated.quant_field[y * AqStorage::kBlockStride + x] -
          expected[index]) > 2.0e-6f ||
        updated.raw_quant[y * AqStorage::kBlockStride + x] < 1 ||
        !std::isfinite(
          updated.block_distance[y * AqStorage::kBlockStride + x])) {
      std::cerr << "AQ field update differs from libjxl's rule\n";
      return false;
    }
  }
  return true;
}

bool CheckInvalidRequestIsAtomic() {
  ImageStorage original(kOriginalExtent, 0.25f);
  ImageStorage opsin(kPaddedExtent, 0.1f);
  gjxl::AcStrategyGrid strategies;
  if (!gjxl::AcStrategyGrid::Create(kBlockExtent, &strategies).ok()) {
    return false;
  }
  strategies.fill_dct8();
  std::array<float, 6> initial{};
  initial.fill(0.45f);
  std::array<uint8_t, 6> sharpness{};
  sharpness.fill(4);
  AqStorage output(31.0f);
  const auto original_quant = output.quant_field;
  const auto original_raw = output.raw_quant;
  const auto original_distance = output.block_distance;
  const auto original_image = output.reconstructed.plane;
  gjxl::AdaptiveQuantizationOptions options;
  options.iterations = 5;
  if (gjxl::FindBestQuantization(
        original.ConstView(),
        opsin.ConstView(),
        strategies,
        {initial.data(), kBlockExtent, kBlockExtent.width},
        {sharpness.data(), kBlockExtent, kBlockExtent.width},
        options,
        output.Output()).ok() ||
      output.quant_field != original_quant ||
      output.raw_quant != original_raw ||
      output.block_distance != original_distance ||
      output.reconstructed.plane != original_image ||
      output.quantizer.valid() ||
      output.color_correlation.valid() ||
      !output.score_history.empty()) {
    std::cerr << "Invalid AQ request changed output\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckLoopAndUpdateRule() || !CheckInvalidRequestIsAtomic()) {
    return EXIT_FAILURE;
  }
  std::cout << "All iterative adaptive-quantization tests passed.\n";
  return EXIT_SUCCESS;
}
