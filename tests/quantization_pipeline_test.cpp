// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Exercises the complete CPU quantization pipeline on a varied corpus.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include "codec/color_transform.h"
#include "codec/quantization_pipeline.h"

namespace {

enum class Fixture {
  kGradient,
  kTexture,
  kEdge,
  kSaturated,
  kColorTileBoundary,
};

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

gjxl::Extent2D Padded(gjxl::Extent2D extent) {
  return {
    (extent.width + gjxl::kJxlBlockDimension - 1) /
      gjxl::kJxlBlockDimension * gjxl::kJxlBlockDimension,
    (extent.height + gjxl::kJxlBlockDimension - 1) /
      gjxl::kJxlBlockDimension * gjxl::kJxlBlockDimension,
  };
}

std::array<float, 3> Pixel(
  Fixture fixture,
  size_t x,
  size_t y,
  gjxl::Extent2D extent) {

  const float fx = static_cast<float>(x) /
    static_cast<float>(std::max<size_t>(1, extent.width - 1));
  const float fy = static_cast<float>(y) /
    static_cast<float>(std::max<size_t>(1, extent.height - 1));
  switch (fixture) {
    case Fixture::kGradient:
      return {0.05f + 0.85f * fx, 0.08f + 0.76f * fy,
              0.12f + 0.68f * (0.6f * fx + 0.4f * fy)};
    case Fixture::kTexture:
      return {
        0.45f + 0.28f * std::sin(0.73f * static_cast<float>(x + 2 * y)),
        0.48f + 0.24f * std::cos(0.51f * static_cast<float>(3 * x - y)),
        0.42f + 0.31f * std::sin(0.37f * static_cast<float>(x + 5 * y)),
      };
    case Fixture::kEdge: {
      const float value = x < extent.width / 2 ? 0.03f : 0.94f;
      return {value, 0.75f * value + 0.06f, 0.45f * value + 0.11f};
    }
    case Fixture::kSaturated:
      if ((x / 4 + y / 4) % 3 == 0) return {1.0f, 0.0f, 0.0f};
      if ((x / 4 + y / 4) % 3 == 1) return {0.0f, 1.0f, 0.0f};
      return {0.0f, 0.0f, 1.0f};
    case Fixture::kColorTileBoundary:
      return {
        x < 64 ? 0.18f + 0.25f * fy : 0.82f - 0.2f * fy,
        0.1f + 0.78f * fx,
        ((x / 8 + y / 5) & 1u) == 0 ? 0.14f : 0.77f,
      };
  }
  return {};
}

void FillImages(
  Fixture fixture,
  ImageStorage* original,
  ImageStorage* padded_linear) {

  for (size_t y = 0; y < padded_linear->extent.height; ++y) {
    const size_t source_y = std::min(y, original->extent.height - 1);
    for (size_t x = 0; x < padded_linear->extent.width; ++x) {
      const size_t source_x = std::min(x, original->extent.width - 1);
      const std::array<float, 3> pixel = Pixel(
        fixture, source_x, source_y, original->extent);
      for (size_t channel = 0; channel < 3; ++channel) {
        padded_linear->plane[channel][y * padded_linear->stride + x] =
          pixel[channel];
        if (x < original->extent.width && y < original->extent.height) {
          original->plane[channel][y * original->stride + x] =
            pixel[channel];
        }
      }
    }
  }
}

struct PipelineStorage {
  explicit PipelineStorage(
    gjxl::Extent2D original_extent,
    gjxl::Extent2D padded_extent,
    float fill = -777.0f)
      : block_extent{
          padded_extent.width / gjxl::kJxlBlockDimension,
          padded_extent.height / gjxl::kJxlBlockDimension},
        block_stride(block_extent.width + 2),
        pixel_stride(padded_extent.width + 3),
        initial_quant(block_stride * block_extent.height, fill),
        strategy_mask(block_stride * block_extent.height, fill),
        pixel_mask(pixel_stride * padded_extent.height, fill),
        final_quant(block_stride * block_extent.height, fill),
        raw_quant(block_stride * block_extent.height, -777),
        block_distance(block_stride * block_extent.height, fill),
        reconstructed(original_extent, fill),
        padded_extent(padded_extent) {}

  [[nodiscard]] gjxl::CpuQuantizationPipelineOutput Output() {
    return {
      .initial_quantization = {
        .quant_field = {
          initial_quant.data(), block_extent, block_stride},
        .strategy_mask = {
          strategy_mask.data(), block_extent, block_stride},
        .pixel_mask = {
          pixel_mask.data(), padded_extent, pixel_stride},
      },
      .adaptive_quantization = {
        .quant_field = {
          final_quant.data(), block_extent, block_stride},
        .raw_quant_field = {
          raw_quant.data(), block_extent, block_stride},
        .block_distance_map = {
          block_distance.data(), block_extent, block_stride},
        .reconstructed_linear_rgb = reconstructed.View(),
        .quantizer = &quantizer,
        .color_correlation = &color_correlation,
        .score_history = &score_history,
      },
      .strategies = &strategies,
    };
  }

  gjxl::Extent2D block_extent;
  size_t block_stride;
  size_t pixel_stride;
  std::vector<float> initial_quant;
  std::vector<float> strategy_mask;
  std::vector<float> pixel_mask;
  std::vector<float> final_quant;
  std::vector<int32_t> raw_quant;
  std::vector<float> block_distance;
  ImageStorage reconstructed;
  gjxl::Extent2D padded_extent;
  gjxl::Quantizer quantizer;
  gjxl::ColorCorrelationMap color_correlation;
  std::vector<double> score_history;
  gjxl::AcStrategyGrid strategies;
};

bool CheckResult(
  const PipelineStorage& result,
  size_t expected_score_count = 3) {

  if (!result.strategies.complete() ||
      !result.quantizer.valid() ||
      !result.color_correlation.valid() ||
      result.score_history.size() != expected_score_count) {
    return false;
  }
  for (double score : result.score_history) {
    if (!std::isfinite(score) || score < 0.0) {
      return false;
    }
  }
  for (size_t y = 0; y < result.block_extent.height; ++y) {
    for (size_t x = 0; x < result.block_extent.width; ++x) {
      const size_t index = y * result.block_stride + x;
      if (!std::isfinite(result.initial_quant[index]) ||
          result.initial_quant[index] <= 0.0f ||
          !std::isfinite(result.strategy_mask[index]) ||
          result.strategy_mask[index] <= 0.0f ||
          !std::isfinite(result.final_quant[index]) ||
          result.final_quant[index] <= 0.0f ||
          result.raw_quant[index] < 1 || result.raw_quant[index] > 256 ||
          !std::isfinite(result.block_distance[index]) ||
          result.block_distance[index] < 0.0f) {
        return false;
      }
    }
    for (size_t x = result.block_extent.width;
         x < result.block_stride;
         ++x) {
      const size_t index = y * result.block_stride + x;
      if (result.initial_quant[index] != -777.0f ||
          result.strategy_mask[index] != -777.0f ||
          result.final_quant[index] != -777.0f ||
          result.raw_quant[index] != -777 ||
          result.block_distance[index] != -777.0f) {
        return false;
      }
    }
  }
  for (size_t y = 0; y < result.padded_extent.height; ++y) {
    for (size_t x = 0; x < result.padded_extent.width; ++x) {
      if (!std::isfinite(result.pixel_mask[y * result.pixel_stride + x]) ||
          result.pixel_mask[y * result.pixel_stride + x] <= 0.0f) {
        return false;
      }
    }
    for (size_t x = result.padded_extent.width;
         x < result.pixel_stride;
         ++x) {
      if (result.pixel_mask[y * result.pixel_stride + x] != -777.0f) {
        return false;
      }
    }
  }
  for (const std::vector<float>& plane : result.reconstructed.plane) {
    for (size_t y = 0; y < result.reconstructed.extent.height; ++y) {
      for (size_t x = 0; x < result.reconstructed.extent.width; ++x) {
        if (!std::isfinite(
              plane[y * result.reconstructed.stride + x])) {
          return false;
        }
      }
      for (size_t x = result.reconstructed.extent.width;
           x < result.reconstructed.stride;
           ++x) {
        if (plane[y * result.reconstructed.stride + x] != -777.0f) {
          return false;
        }
      }
    }
  }
  return true;
}

bool RunFixture(
  Fixture fixture,
  std::string_view name,
  gjxl::Extent2D original_extent) {

  const gjxl::Extent2D padded_extent = Padded(original_extent);
  ImageStorage original(original_extent);
  ImageStorage padded_linear(padded_extent);
  ImageStorage opsin(padded_extent);
  FillImages(fixture, &original, &padded_linear);
  if (!gjxl::LinearRgbToOpsin(
        padded_linear.ConstView(), 255.0f, opsin.View()).ok()) {
    return false;
  }

  PipelineStorage output(original_extent, padded_extent);
  gjxl::CpuQuantizationPipelineOptions options;
  options.butteraugli_target = 1.2f;
  options.adaptive_quantization.iterations = 2;
  const gjxl::Status status = gjxl::RunCpuQuantizationPipeline(
    original.ConstView(),
    opsin.ConstView(),
    options,
    output.Output());
  if (!status.ok() || !CheckResult(output)) {
    std::cerr << "CPU quantization pipeline failed for " << name
              << ": " << status.message() << '\n';
    return false;
  }
  return true;
}

bool CheckInvalidRequestIsAtomic() {
  constexpr gjxl::Extent2D kExtent{16, 16};
  ImageStorage original(kExtent, 0.25f);
  ImageStorage opsin(kExtent, 0.1f);
  PipelineStorage output(kExtent, kExtent, 29.0f);
  const auto initial = output.initial_quant;
  const auto final = output.final_quant;
  const auto raw = output.raw_quant;
  const auto reconstructed = output.reconstructed.plane;
  gjxl::CpuQuantizationPipelineOptions options;
  options.butteraugli_target = 0.0f;
  if (gjxl::RunCpuQuantizationPipeline(
        original.ConstView(),
        opsin.ConstView(),
        options,
        output.Output()).ok() ||
      output.initial_quant != initial ||
      output.final_quant != final ||
      output.raw_quant != raw ||
      output.reconstructed.plane != reconstructed ||
      output.strategies.valid() ||
      output.quantizer.valid() ||
      output.color_correlation.valid() ||
      !output.score_history.empty()) {
    std::cerr << "Invalid CPU pipeline request changed output\n";
    return false;
  }
  return true;
}

bool CheckGaborishDisabledPath() {
  constexpr gjxl::Extent2D kExtent{16, 16};
  ImageStorage original(kExtent);
  ImageStorage padded_linear(kExtent);
  ImageStorage opsin(kExtent);
  FillImages(Fixture::kGradient, &original, &padded_linear);
  if (!gjxl::LinearRgbToOpsin(
        padded_linear.ConstView(), 255.0f, opsin.View()).ok()) {
    return false;
  }
  PipelineStorage output(kExtent, kExtent);
  gjxl::CpuQuantizationPipelineOptions options;
  options.butteraugli_target = 1.2f;
  options.adaptive_quantization.iterations = 0;
  options.adaptive_quantization.loop_filter.gaborish = false;
  const gjxl::Status status = gjxl::RunCpuQuantizationPipeline(
    original.ConstView(), opsin.ConstView(), options, output.Output());
  if (!status.ok() || !CheckResult(output, 1)) {
    std::cerr << "No-Gaborish CPU pipeline failed: "
              << status.message() << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!RunFixture(Fixture::kGradient, "odd gradient", {21, 13}) ||
      !RunFixture(Fixture::kTexture, "texture", {32, 24}) ||
      !RunFixture(Fixture::kEdge, "hard edge", {33, 17}) ||
      !RunFixture(Fixture::kSaturated, "saturated colors", {19, 25}) ||
      !RunFixture(
        Fixture::kColorTileBoundary,
        "64-pixel color-tile boundary",
        {69, 17}) ||
      !CheckGaborishDisabledPath() ||
      !CheckInvalidRequestIsAtomic()) {
    return EXIT_FAILURE;
  }
  std::cout << "All CPU quantization pipeline tests passed.\n";
  return EXIT_SUCCESS;
}
