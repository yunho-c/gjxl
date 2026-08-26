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
#include <iomanip>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#include "butteraugli_test_tolerances.h"
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
        0.48f + 0.24f * std::cos(
          0.51f *
          (3.0f * static_cast<float>(x) - static_cast<float>(y))),
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

struct PinnedPipelineExpectation {
  std::span<const double> score_history;
  std::span<const float> final_quant;
  std::span<const int32_t> raw_quant;
};

struct PipelinePinErrors {
  double score = 0.0;
  float final_quant = 0.0f;
};

PipelinePinErrors g_pipeline_pin_errors;

// Captured from the pre-integration pinned scalar libjxl facade. These pin the
// complete pipeline independently of the native implementation under test.
constexpr std::array<double, 3> kGradientScores = {
  0.75522822141647339, 0.66656208038330078, 0.8399806022644043,
};
constexpr std::array<float, 6> kGradientQuant = {
  0.472901613f, 0.472901613f, 0.42963475f,
  0.472901613f, 0.472901613f, 0.465390801f,
};
constexpr std::array<int32_t, 6> kGradientRaw = {5, 5, 5, 5, 5, 5};

constexpr std::array<double, 3> kTextureScores = {
  1.3783982992172241, 1.2329332828521729, 1.1630038022994995,
};
constexpr std::array<float, 12> kTextureQuant = {
  0.288514882f, 0.288514882f, 0.288514882f, 0.288514882f,
  0.288514882f, 0.288514882f, 0.288514882f, 0.288514882f,
  0.238204628f, 0.238204628f, 0.233531922f, 0.293218255f,
};
constexpr std::array<int32_t, 12> kTextureRaw = {
  5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 5,
};

constexpr std::array<double, 3> kEdgeScores = {
  0.38643217086791992, 0.36592763662338257, 0.44234001636505127,
};
constexpr std::array<float, 15> kEdgeQuant = {
  0.348677039f, 0.366792262f, 0.324879259f, 0.300253868f, 0.300253868f,
  0.348677039f, 0.366792262f, 0.324879259f, 0.300253868f, 0.300253868f,
  0.344846576f, 0.364931524f, 0.318393141f, 0.301782131f, 0.251067847f,
};
constexpr std::array<int32_t, 15> kEdgeRaw = {
  6, 6, 5, 5, 5, 6, 6, 5, 5, 5, 6, 6, 5, 5, 4,
};

constexpr std::array<double, 3> kSaturatedScores = {
  1.27321457862854, 0.77876514196395874, 1.3892008066177368,
};
constexpr std::array<float, 12> kSaturatedQuant = {
  0.346176445f, 0.346229523f, 0.340055197f, 0.336938828f,
  0.329504639f, 0.281445384f, 0.337511897f, 0.393181562f,
  0.34743312f, 0.305287272f, 0.341618329f, 0.363960683f,
};
constexpr std::array<int32_t, 12> kSaturatedRaw = {
  5, 5, 5, 5, 5, 4, 5, 6, 5, 5, 5, 5,
};

constexpr std::array<double, 3> kColorTileBoundaryScores = {
  0.68892323970794678, 1.0825774669647217, 1.0657584667205811,
};
constexpr std::array<float, 27> kColorTileBoundaryQuant = {
  0.536615729f, 0.517152071f, 0.530328691f, 0.510748625f, 0.519218326f,
  0.50656569f, 0.524777472f, 0.499126673f, 0.489810646f,
  0.499375999f, 0.506759942f, 0.530328691f, 0.510748625f, 0.492208749f,
  0.50656569f, 0.486722529f, 0.451566935f, 0.462795407f,
  0.445161372f, 0.444984674f, 0.422560602f, 0.401480108f, 0.387192398f,
  0.393088162f, 0.434787005f, 0.411924779f, 0.42122215f,
};
constexpr std::array<int32_t, 27> kColorTileBoundaryRaw = {
  6, 6, 6, 6, 6, 6, 6, 5, 5,
  5, 6, 6, 6, 5, 6, 5, 5, 5,
  5, 5, 5, 4, 4, 4, 5, 4, 5,
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

bool CheckPinnedResult(
  const PipelineStorage& result,
  PinnedPipelineExpectation expected) {

  const size_t block_count =
    result.block_extent.width * result.block_extent.height;
  if (result.score_history.size() != expected.score_history.size() ||
      block_count != expected.final_quant.size() ||
      block_count != expected.raw_quant.size()) {
    return false;
  }
  for (size_t index = 0; index < expected.score_history.size(); ++index) {
    const double error =
      std::abs(result.score_history[index] - expected.score_history[index]);
    g_pipeline_pin_errors.score =
      std::max(g_pipeline_pin_errors.score, error);
    if (error >
        gjxl::butteraugli_test::kPinnedAqTolerance) {
      return false;
    }
  }
  for (size_t index = 0; index < block_count; ++index) {
    const size_t x = index % result.block_extent.width;
    const size_t y = index / result.block_extent.width;
    const size_t strided_index = y * result.block_stride + x;
    const float quant_error = std::abs(
      result.final_quant[strided_index] - expected.final_quant[index]);
    g_pipeline_pin_errors.final_quant =
      std::max(g_pipeline_pin_errors.final_quant, quant_error);
    if (quant_error >
          gjxl::butteraugli_test::kPinnedAqTolerance ||
        result.raw_quant[strided_index] != expected.raw_quant[index]) {
      return false;
    }
  }
  return true;
}

bool RunFixture(
  Fixture fixture,
  std::string_view name,
  gjxl::Extent2D original_extent,
  PinnedPipelineExpectation expected) {

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
  if (!status.ok() || !CheckResult(output) ||
      !CheckPinnedResult(output, expected)) {
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

bool CheckInvalidOutputsAreRejected() {
  constexpr gjxl::Extent2D kExtent{16, 16};
  ImageStorage original(kExtent, 0.25f);
  ImageStorage opsin(kExtent, 0.1f);
  PipelineStorage storage(kExtent, kExtent, 29.0f);
  const auto initial = storage.initial_quant;
  const auto final = storage.final_quant;
  const auto raw = storage.raw_quant;
  const auto reconstructed = storage.reconstructed.plane;

  const auto rejected_atomically = [&](
    gjxl::CpuQuantizationPipelineOutput output) {
    const gjxl::Status status = gjxl::RunCpuQuantizationPipeline(
      original.ConstView(), opsin.ConstView(), {}, output);
    return !status.ok() &&
      storage.initial_quant == initial &&
      storage.final_quant == final &&
      storage.raw_quant == raw &&
      storage.reconstructed.plane == reconstructed &&
      !storage.strategies.valid() &&
      !storage.quantizer.valid() &&
      !storage.color_correlation.valid() &&
      storage.score_history.empty();
  };

  auto output = storage.Output();
  output.adaptive_quantization.quant_field.data = nullptr;
  if (!rejected_atomically(output)) return false;

  output = storage.Output();
  output.adaptive_quantization.raw_quant_field.data = nullptr;
  if (!rejected_atomically(output)) return false;

  output = storage.Output();
  output.adaptive_quantization.block_distance_map.data = nullptr;
  if (!rejected_atomically(output)) return false;

  output = storage.Output();
  output.adaptive_quantization.reconstructed_linear_rgb.plane[1].data = nullptr;
  if (!rejected_atomically(output)) return false;

  output = storage.Output();
  output.adaptive_quantization.quantizer = nullptr;
  if (!rejected_atomically(output)) return false;

  output = storage.Output();
  output.adaptive_quantization.color_correlation = nullptr;
  if (!rejected_atomically(output)) return false;

  output = storage.Output();
  output.adaptive_quantization.score_history = nullptr;
  if (!rejected_atomically(output)) return false;

  output = storage.Output();
  output.strategies = nullptr;
  if (!rejected_atomically(output)) return false;

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
  if (!RunFixture(Fixture::kGradient, "odd gradient", {21, 13},
                  {kGradientScores, kGradientQuant, kGradientRaw}) ||
      !RunFixture(Fixture::kTexture, "texture", {32, 24},
                  {kTextureScores, kTextureQuant, kTextureRaw}) ||
      !RunFixture(Fixture::kEdge, "hard edge", {33, 17},
                  {kEdgeScores, kEdgeQuant, kEdgeRaw}) ||
      !RunFixture(Fixture::kSaturated, "saturated colors", {19, 25},
                  {kSaturatedScores, kSaturatedQuant, kSaturatedRaw}) ||
      !RunFixture(
        Fixture::kColorTileBoundary,
        "64-pixel color-tile boundary",
        {69, 17},
        {kColorTileBoundaryScores,
         kColorTileBoundaryQuant,
         kColorTileBoundaryRaw}) ||
      !CheckGaborishDisabledPath() ||
      !CheckInvalidRequestIsAtomic() ||
      !CheckInvalidOutputsAreRejected()) {
    return EXIT_FAILURE;
  }
  std::cout << std::setprecision(9)
            << "Pipeline pin maximum errors: score="
            << g_pipeline_pin_errors.score
            << " final_quant=" << g_pipeline_pin_errors.final_quant
            << " raw_quant=exact\n"
            << "All CPU quantization pipeline tests passed.\n";
  return EXIT_SUCCESS;
}
