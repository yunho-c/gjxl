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
#include <iomanip>
#include <iostream>
#include <vector>

#include "butteraugli_test_tolerances.h"
#include "codec/adaptive_quantization.h"
#include "codec/adaptive_quantization_internal.h"
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
          expected[index]) >
          gjxl::butteraugli_test::kAqUpdateTolerance ||
        updated.raw_quant[y * AqStorage::kBlockStride + x] < 1 ||
        !std::isfinite(
          updated.block_distance[y * AqStorage::kBlockStride + x])) {
      std::cerr << "AQ field update differs from libjxl's rule\n";
      return false;
    }
  }
  AqStorage two_updates;
  updated_options.iterations = 2;
  const gjxl::Status two_update_status = gjxl::FindBestQuantization(
    original.ConstView(),
    opsin.ConstView(),
    strategies,
    initial,
    sharpness_view,
    updated_options,
    two_updates.Output());
  if (!two_update_status.ok() || two_updates.score_history.size() != 3 ||
      !PaddingIsUntouched(two_updates)) {
    std::cerr << "Two-update AQ evaluation failed\n";
    return false;
  }
  constexpr std::array<double, 3> kPinnedScoreHistory = {
    1.5871505737304688,
    1.4434140920639038,
    1.3173232078552246,
  };
  constexpr std::array<float, 6> kPinnedQuantField = {
    0.890399575f, 0.865948975f, 0.797086954f,
    1.00165439f, 0.905635715f, 0.859129727f,
  };
  constexpr std::array<int32_t, 6> kPinnedRawQuantField = {
    7, 7, 6,
    8, 7, 7,
  };
  constexpr std::array<float, 6> kPinnedBlockDistance = {
    1.45603883f, 1.42887533f, 1.43554032f,
    1.41945028f, 1.39149904f, 1.37905943f,
  };
  double maximum_score_error = 0.0;
  for (size_t index = 0; index < kPinnedScoreHistory.size(); ++index) {
    const double error = std::abs(two_updates.score_history[index] -
                                  kPinnedScoreHistory[index]);
    maximum_score_error = std::max(maximum_score_error, error);
    if (error >
          gjxl::butteraugli_test::kPinnedAqTolerance) {
      std::cerr << "Two-update AQ score history differs from the pin\n";
      return false;
    }
  }
  float maximum_quant_error = 0.0f;
  float maximum_block_distance_error = 0.0f;
  for (size_t index = 0; index < kPinnedQuantField.size(); ++index) {
    const size_t x = index % kBlockExtent.width;
    const size_t y = index / kBlockExtent.width;
    const size_t strided_index = y * AqStorage::kBlockStride + x;
    const float quant_error = std::abs(
      two_updates.quant_field[strided_index] - kPinnedQuantField[index]);
    const float block_distance_error = std::abs(
      two_updates.block_distance[strided_index] - kPinnedBlockDistance[index]);
    maximum_quant_error = std::max(maximum_quant_error, quant_error);
    maximum_block_distance_error =
      std::max(maximum_block_distance_error, block_distance_error);
    if (quant_error >
          gjxl::butteraugli_test::kPinnedAqTolerance ||
        two_updates.raw_quant[strided_index] !=
          kPinnedRawQuantField[index] ||
        block_distance_error >
          gjxl::butteraugli_test::kPinnedAqTolerance) {
      std::cerr << "Two-update AQ fields differ from the pin\n";
      return false;
    }
  }

  std::cout << std::setprecision(9)
            << "Two-update pin maximum errors: score="
            << maximum_score_error << " quant=" << maximum_quant_error
            << " block_distance=" << maximum_block_distance_error
            << " raw_quant=exact\n";

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

bool ColorCorrelationMapsEqual(
  const gjxl::ColorCorrelationMap& left,
  const gjxl::ColorCorrelationMap& right) {

  if (left.valid() != right.valid() ||
      left.tile_extent() != right.tile_extent()) {
    return false;
  }
  const auto left_x = left.y_to_x_map();
  const auto right_x = right.y_to_x_map();
  const auto left_b = left.y_to_b_map();
  const auto right_b = right.y_to_b_map();
  for (size_t y = 0; y < left.tile_extent().height; ++y) {
    if (!std::equal(
          left_x.Row(y),
          left_x.Row(y) + left.tile_extent().width,
          right_x.Row(y)) ||
        !std::equal(
          left_b.Row(y),
          left_b.Row(y) + left.tile_extent().width,
          right_b.Row(y))) {
      return false;
    }
  }
  return true;
}

bool CheckProfiledPath() {
  namespace aqi = gjxl::adaptive_quantization_internal;

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
  gjxl::AdaptiveQuantizationOptions options;
  options.butteraugli_target = 1.1f;
  options.iterations = 2;

  AqStorage ordinary;
  AqStorage profiled;
  aqi::AdaptiveQuantizationProfile profile;
  const gjxl::Status ordinary_status = gjxl::FindBestQuantization(
    original.ConstView(), opsin.ConstView(), strategies, initial,
    sharpness_view, options, ordinary.Output());
  const gjxl::Status profiled_status = aqi::FindBestQuantizationProfiled(
    original.ConstView(), opsin.ConstView(), strategies, initial,
    sharpness_view, options, profiled.Output(), &profile);
  if (!ordinary_status.ok() || !profiled_status.ok() ||
      ordinary.quant_field != profiled.quant_field ||
      ordinary.raw_quant != profiled.raw_quant ||
      ordinary.block_distance != profiled.block_distance ||
      ordinary.reconstructed.plane != profiled.reconstructed.plane ||
      ordinary.quantizer.params().global_scale !=
        profiled.quantizer.params().global_scale ||
      ordinary.quantizer.params().quant_dc !=
        profiled.quantizer.params().quant_dc ||
      !ColorCorrelationMapsEqual(
        ordinary.color_correlation, profiled.color_correlation) ||
      ordinary.score_history != profiled.score_history ||
      profile.evaluations.size() != options.iterations + 1) {
    std::cerr << "Profiled AQ differs from the production path\n";
    return false;
  }

  AqStorage zero_profiled;
  aqi::AdaptiveQuantizationProfile zero_profile;
  options.iterations = 0;
  if (!aqi::FindBestQuantizationProfiled(
        original.ConstView(), opsin.ConstView(), strategies, initial,
        sharpness_view, options, zero_profiled.Output(), &zero_profile).ok() ||
      zero_profile.evaluations.size() != 1) {
    std::cerr << "Zero-update profiled AQ has the wrong evaluation count\n";
    return false;
  }

  for (const aqi::EvaluationProfile& evaluation : profile.evaluations) {
    uint64_t measured_stages = 0;
    for (uint64_t stage : evaluation.stage_nanoseconds) {
      if (stage == 0) {
        std::cerr << "Profiled AQ omitted an evaluation stage\n";
        return false;
      }
      measured_stages += stage;
    }
    if (evaluation.total_nanoseconds < measured_stages) {
      std::cerr << "Profiled AQ stage time exceeds its evaluation total\n";
      return false;
    }
  }

  aqi::AdaptiveQuantizationProfile invalid_profile;
  invalid_profile.loop_setup_nanoseconds = 11;
  invalid_profile.quant_field_update_nanoseconds = 12;
  invalid_profile.output_commit_nanoseconds = 13;
  invalid_profile.evaluations.push_back({
    .total_nanoseconds = 14,
  });
  const aqi::AdaptiveQuantizationProfile original_profile = invalid_profile;
  AqStorage invalid_output(31.0f);
  const auto original_quant = invalid_output.quant_field;
  const auto original_raw = invalid_output.raw_quant;
  const auto original_distance = invalid_output.block_distance;
  const auto original_image = invalid_output.reconstructed.plane;
  options.iterations = 5;
  if (aqi::FindBestQuantizationProfiled(
        original.ConstView(), opsin.ConstView(), strategies, initial,
        sharpness_view, options, invalid_output.Output(),
        &invalid_profile).ok() ||
      invalid_profile != original_profile ||
      invalid_output.quant_field != original_quant ||
      invalid_output.raw_quant != original_raw ||
      invalid_output.block_distance != original_distance ||
      invalid_output.reconstructed.plane != original_image ||
      invalid_output.quantizer.valid() ||
      invalid_output.color_correlation.valid() ||
      !invalid_output.score_history.empty()) {
    std::cerr << "Invalid profiled AQ request changed caller state\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckLoopAndUpdateRule() || !CheckInvalidRequestIsAtomic() ||
      !CheckProfiledPath()) {
    return EXIT_FAILURE;
  }
  std::cout << "All iterative adaptive-quantization tests passed.\n";
  return EXIT_SUCCESS;
}
