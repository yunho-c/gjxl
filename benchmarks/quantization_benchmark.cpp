// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Reports median and range CPU baselines for pre-GPU quantization stages.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "codec/ac_strategy.h"
#include "codec/adaptive_quantization.h"
#include "codec/chroma_from_luma.h"
#include "codec/color_transform.h"
#include "codec/epf.h"
#include "codec/gaborish.h"
#include "codec/quantization_pipeline.h"

namespace {

constexpr gjxl::Extent2D kPixelExtent{128, 96};
constexpr gjxl::Extent2D kBlockExtent{16, 12};
constexpr size_t kSamples = 5;

struct ImageStorage {
  std::array<std::vector<float>, 3> plane;

  ImageStorage() {
    for (std::vector<float>& values : plane) {
      values.resize(kPixelExtent.width * kPixelExtent.height);
    }
  }

  [[nodiscard]] gjxl::Image3FView View() {
    return {{
      gjxl::PlaneF32View{
        plane[0].data(), kPixelExtent, kPixelExtent.width},
      gjxl::PlaneF32View{
        plane[1].data(), kPixelExtent, kPixelExtent.width},
      gjxl::PlaneF32View{
        plane[2].data(), kPixelExtent, kPixelExtent.width},
    }};
  }

  [[nodiscard]] gjxl::ConstImage3FView ConstView() const {
    return {{
      gjxl::ConstPlaneF32View{
        plane[0].data(), kPixelExtent, kPixelExtent.width},
      gjxl::ConstPlaneF32View{
        plane[1].data(), kPixelExtent, kPixelExtent.width},
      gjxl::ConstPlaneF32View{
        plane[2].data(), kPixelExtent, kPixelExtent.width},
    }};
  }
};

struct StageOutput {
  std::vector<float> initial_quant = std::vector<float>(
    kBlockExtent.width * kBlockExtent.height);
  std::vector<float> strategy_mask = std::vector<float>(
    kBlockExtent.width * kBlockExtent.height);
  std::vector<float> pixel_mask = std::vector<float>(
    kPixelExtent.width * kPixelExtent.height);
  std::vector<float> final_quant = std::vector<float>(
    kBlockExtent.width * kBlockExtent.height);
  std::vector<int32_t> raw_quant = std::vector<int32_t>(
    kBlockExtent.width * kBlockExtent.height);
  std::vector<float> block_distance = std::vector<float>(
    kBlockExtent.width * kBlockExtent.height);
  ImageStorage reconstructed;
  gjxl::Quantizer quantizer;
  gjxl::ColorCorrelationMap color_correlation;
  std::vector<double> scores;
};

template <typename Function>
bool TimeStage(
  std::vector<double>* samples,
  Function&& function) {

  const auto start = std::chrono::steady_clock::now();
  const gjxl::Status status = function();
  const auto end = std::chrono::steady_clock::now();
  if (!status.ok()) {
    std::cerr << "Benchmark stage failed: " << status.message() << '\n';
    return false;
  }
  samples->push_back(
    std::chrono::duration<double, std::milli>(end - start).count());
  return true;
}

void PrintStats(
  std::string_view name,
  std::vector<double> samples) {

  std::sort(samples.begin(), samples.end());
  std::cout << std::left << std::setw(27) << name
            << std::right << std::fixed << std::setprecision(3)
            << " median " << std::setw(9) << samples[samples.size() / 2]
            << " ms  range [" << samples.front()
            << ", " << samples.back() << "] ms\n";
}

void FillLinear(ImageStorage* image) {
  for (size_t y = 0; y < kPixelExtent.height; ++y) {
    for (size_t x = 0; x < kPixelExtent.width; ++x) {
      const float fx = static_cast<float>(x) /
        static_cast<float>(kPixelExtent.width - 1);
      const float fy = static_cast<float>(y) /
        static_cast<float>(kPixelExtent.height - 1);
      image->plane[0][y * kPixelExtent.width + x] =
        std::clamp(
          0.08f + 0.72f * fx +
            0.13f * std::sin(0.47f * static_cast<float>(x + y)),
          0.0f,
          1.0f);
      image->plane[1][y * kPixelExtent.width + x] =
        std::clamp(
          0.1f + 0.68f * fy +
            0.16f * std::cos(
              0.39f *
              (2.0f * static_cast<float>(x) - static_cast<float>(y))),
          0.0f,
          1.0f);
      image->plane[2][y * kPixelExtent.width + x] =
        ((x / 7 + y / 5) & 1u) == 0 ? 0.12f : 0.84f;
    }
  }
}

}  // namespace

int main() {
  ImageStorage linear;
  ImageStorage opsin;
  ImageStorage preprocessed;
  FillLinear(&linear);
  if (!gjxl::LinearRgbToOpsin(
        linear.ConstView(), 255.0f, opsin.View()).ok() ||
      !gjxl::ApplyGaborishInverse(
        opsin.ConstView(), {1.0f, 1.0f, 1.0f},
        preprocessed.View()).ok()) {
    return EXIT_FAILURE;
  }

  StageOutput stage;
  const gjxl::InitialQuantFieldOutput initial_output{
    .quant_field = {
      stage.initial_quant.data(), kBlockExtent, kBlockExtent.width},
    .strategy_mask = {
      stage.strategy_mask.data(), kBlockExtent, kBlockExtent.width},
    .pixel_mask = {
      stage.pixel_mask.data(), kPixelExtent, kPixelExtent.width},
  };
  if (!gjxl::ComputeInitialQuantField(
        opsin.ConstView(), {.butteraugli_target = 1.2f},
        initial_output).ok()) {
    return EXIT_FAILURE;
  }
  gjxl::ColorCorrelationMap initial_color_correlation;
  if (!gjxl::ComputeInitialColorCorrelationMap(
        preprocessed.ConstView(), &initial_color_correlation).ok()) {
    return EXIT_FAILURE;
  }
  gjxl::AcStrategyGrid strategies;
  if (!gjxl::FindAcStrategyGrid(
        preprocessed.ConstView(),
        {stage.initial_quant.data(), kBlockExtent, kBlockExtent.width},
        {stage.pixel_mask.data(), kPixelExtent, kPixelExtent.width},
        initial_color_correlation,
        {.butteraugli_target = 1.2f},
        &strategies).ok()) {
    return EXIT_FAILURE;
  }
  std::vector<uint8_t> sharpness(
    kBlockExtent.width * kBlockExtent.height);
  if (!gjxl::FillDefaultEpfSharpness(
        {sharpness.data(), kBlockExtent, kBlockExtent.width}).ok()) {
    return EXIT_FAILURE;
  }

  const gjxl::AdaptiveQuantizationOutput adaptive_output{
    .quant_field = {
      stage.final_quant.data(), kBlockExtent, kBlockExtent.width},
    .raw_quant_field = {
      stage.raw_quant.data(), kBlockExtent, kBlockExtent.width},
    .block_distance_map = {
      stage.block_distance.data(), kBlockExtent, kBlockExtent.width},
    .reconstructed_linear_rgb = stage.reconstructed.View(),
    .quantizer = &stage.quantizer,
    .color_correlation = &stage.color_correlation,
    .score_history = &stage.scores,
  };
  gjxl::AdaptiveQuantizationOptions adaptive_options;
  adaptive_options.butteraugli_target = 1.2f;
  adaptive_options.iterations = 2;

  StageOutput pipeline_stage;
  gjxl::AcStrategyGrid pipeline_strategies;
  const gjxl::CpuQuantizationPipelineOutput pipeline_output{
    .initial_quantization = {
      .quant_field = {
        pipeline_stage.initial_quant.data(),
        kBlockExtent,
        kBlockExtent.width},
      .strategy_mask = {
        pipeline_stage.strategy_mask.data(),
        kBlockExtent,
        kBlockExtent.width},
      .pixel_mask = {
        pipeline_stage.pixel_mask.data(),
        kPixelExtent,
        kPixelExtent.width},
    },
    .adaptive_quantization = {
      .quant_field = {
        pipeline_stage.final_quant.data(),
        kBlockExtent,
        kBlockExtent.width},
      .raw_quant_field = {
        pipeline_stage.raw_quant.data(),
        kBlockExtent,
        kBlockExtent.width},
      .block_distance_map = {
        pipeline_stage.block_distance.data(),
        kBlockExtent,
        kBlockExtent.width},
      .reconstructed_linear_rgb = pipeline_stage.reconstructed.View(),
      .quantizer = &pipeline_stage.quantizer,
      .color_correlation = &pipeline_stage.color_correlation,
      .score_history = &pipeline_stage.scores,
    },
    .strategies = &pipeline_strategies,
  };
  gjxl::CpuQuantizationPipelineOptions pipeline_options;
  pipeline_options.butteraugli_target = 1.2f;
  pipeline_options.adaptive_quantization.iterations = 2;

  // Warm the code and data paths before collecting the small sample set.
  if (!gjxl::RunCpuQuantizationPipeline(
        linear.ConstView(), opsin.ConstView(), pipeline_options,
        pipeline_output).ok()) {
    return EXIT_FAILURE;
  }

  std::vector<double> initial_samples;
  std::vector<double> gaborish_samples;
  std::vector<double> cfl_samples;
  std::vector<double> search_samples;
  std::vector<double> adaptive_samples;
  std::vector<double> pipeline_samples;
  for (size_t sample = 0; sample < kSamples; ++sample) {
    if (!TimeStage(&initial_samples, [&] {
          return gjxl::ComputeInitialQuantField(
            opsin.ConstView(), {.butteraugli_target = 1.2f},
            initial_output);
        }) ||
        !TimeStage(&gaborish_samples, [&] {
          return gjxl::ApplyGaborishInverse(
            opsin.ConstView(), {1.0f, 1.0f, 1.0f},
            preprocessed.View());
        }) ||
        !TimeStage(&cfl_samples, [&] {
          return gjxl::ComputeInitialColorCorrelationMap(
            preprocessed.ConstView(), &initial_color_correlation);
        }) ||
        !TimeStage(&search_samples, [&] {
          return gjxl::FindAcStrategyGrid(
            preprocessed.ConstView(),
            {stage.initial_quant.data(), kBlockExtent, kBlockExtent.width},
            {stage.pixel_mask.data(), kPixelExtent, kPixelExtent.width},
            initial_color_correlation,
            {.butteraugli_target = 1.2f},
            &strategies);
        }) ||
        !TimeStage(&adaptive_samples, [&] {
          return gjxl::FindBestQuantization(
            linear.ConstView(),
            preprocessed.ConstView(),
            strategies,
            {stage.initial_quant.data(), kBlockExtent, kBlockExtent.width},
            {sharpness.data(), kBlockExtent, kBlockExtent.width},
            adaptive_options,
            adaptive_output);
        }) ||
        !TimeStage(&pipeline_samples, [&] {
          return gjxl::RunCpuQuantizationPipeline(
            linear.ConstView(),
            opsin.ConstView(),
            pipeline_options,
            pipeline_output);
        })) {
      return EXIT_FAILURE;
    }
  }

  std::cout << "CPU quantization baseline: 128x96, 5 samples, AQ=2 updates\n";
  PrintStats("initial_quant_field", std::move(initial_samples));
  PrintStats("gaborish_inverse", std::move(gaborish_samples));
  PrintStats("initial_cfl", std::move(cfl_samples));
  PrintStats("ac_strategy_search", std::move(search_samples));
  PrintStats("iterative_aq", std::move(adaptive_samples));
  PrintStats("complete_pipeline", std::move(pipeline_samples));
  return EXIT_SUCCESS;
}
