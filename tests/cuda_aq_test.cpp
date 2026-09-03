// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "codec/adaptive_quantization.h"
#include "codec/chroma_from_luma_internal.h"
#include "codec/color_transform.h"
#include "codec/epf.h"
#include "codec/gaborish.h"
#include "codec/quantization.h"
#include "codec/reconstruction.h"
#include "codestream/encoder.h"
#include "codestream/workflow_internal.h"
#include "core/frame_geometry.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/ops/adaptive_quantization.h"
#include "gpu/ops/aq_evaluation.h"

namespace {

constexpr gjxl::Extent2D kSourceExtent{257, 17};
constexpr gjxl::Extent2D kPaddedExtent{264, 24};

struct ImageStorage {
  explicit ImageStorage(gjxl::Extent2D extent, float fill = -991.0f)
    : extent(extent), stride(extent.width + 3) {
    for (auto& values : plane) values.assign(stride * extent.height, fill);
  }

  gjxl::Image3FView View() {
    return {{{
      {plane[0].data(), extent, stride},
      {plane[1].data(), extent, stride},
      {plane[2].data(), extent, stride},
    }}};
  }

  gjxl::ConstImage3FView View() const {
    return {{{
      {plane[0].data(), extent, stride},
      {plane[1].data(), extent, stride},
      {plane[2].data(), extent, stride},
    }}};
  }

  gjxl::Extent2D extent;
  size_t stride;
  std::array<std::vector<float>, 3> plane;
};

bool Check(gjxl::Status status, const char* operation) {
  if (status.ok()) return true;
  std::cerr << operation << " failed: " << status.message() << '\n';
  return false;
}

void FillLinear(ImageStorage* source, ImageStorage* padded) {
  for (size_t y = 0; y < kPaddedExtent.height; ++y) {
    const size_t sy = std::min(y, kSourceExtent.height - 1);
    for (size_t x = 0; x < kPaddedExtent.width; ++x) {
      const size_t sx = std::min(x, kSourceExtent.width - 1);
      const float fx = static_cast<float>(sx);
      const float fy = static_cast<float>(sy);
      const std::array<float, 3> rgb = {
        std::clamp(0.08f + 0.0028f * fx + 0.06f * std::sin(0.31f * fy),
                   0.0f, 1.0f),
        std::clamp(0.13f + 0.021f * fy + 0.04f * std::cos(0.071f * fx),
                   0.0f, 1.0f),
        ((sx / 7 + sy / 3) & 1u) == 0 ? 0.11f : 0.83f,
      };
      for (size_t channel = 0; channel < 3; ++channel) {
        padded->plane[channel][y * padded->stride + x] = rgb[channel];
        if (x < kSourceExtent.width && y < kSourceExtent.height) {
          source->plane[channel][y * source->stride + x] = rgb[channel];
        }
      }
    }
  }
}

double MaximumError(
  const std::vector<float>& expected,
  const std::vector<float>& actual) {
  double result = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    result = std::max(result, std::abs(
      static_cast<double>(expected[i]) - static_cast<double>(actual[i])));
  }
  return result;
}

bool CheckResidentFrontend(gjxl::GpuBackend& gpu, const ImageStorage& source,
                           const ImageStorage& opsin) {
  const gjxl::Extent2D blocks{kPaddedExtent.width / 8, kPaddedExtent.height / 8};
  const size_t block_count = blocks.width * blocks.height;
  const size_t pixel_count = kPaddedExtent.width * kPaddedExtent.height;
  gjxl::AcStrategyGrid strategies;
  std::vector<uint8_t> sharpness(block_count);
  if (!Check(gjxl::AcStrategyGrid::Create(blocks, &strategies),
             "Create CUDA AQ strategy grid") ||
      !Check(gjxl::FillDefaultEpfSharpness(
        {sharpness.data(), blocks, blocks.width}), "Fill CUDA AQ sharpness")) {
    return false;
  }
  strategies.fill_dct8();

  gjxl::AdaptiveQuantizationOptions aq_options;
  aq_options.butteraugli_target = 1.2f;
  const gjxl::InitialQuantizationOptions initial_options{
    .butteraugli_target = aq_options.profile.loop_filter.gaborish
      ? aq_options.butteraugli_target : 0.62f * aq_options.butteraugli_target,
    .rescale = 1.0f,
  };
  std::vector<float> expected_quant(block_count);
  std::vector<float> expected_strategy(block_count);
  std::vector<float> expected_pixel(pixel_count);
  if (!Check(gjxl::ComputeInitialQuantField(
      opsin.View(), initial_options,
      {
        .quant_field = {expected_quant.data(), blocks, blocks.width},
        .strategy_mask = {expected_strategy.data(), blocks, blocks.width},
        .pixel_mask = {expected_pixel.data(), kPaddedExtent, kPaddedExtent.width},
      }), "CPU initial quantization reference")) {
    return false;
  }

  std::vector<float> actual_quant(block_count, -5.0f);
  std::vector<float> actual_strategy(block_count, -5.0f);
  std::vector<float> actual_pixel(pixel_count, -5.0f);
  std::vector<float> final_quant(block_count, -5.0f);
  gjxl::VarDctEncoderFrame cuda_frame;
  if (!Check(gjxl::RunGpuFrameOnlyQuantizationResidentFrontend(
      gpu, source.View(), opsin.View(), strategies,
      {sharpness.data(), blocks, blocks.width}, initial_options, aq_options,
      {
        .quant_field = {actual_quant.data(), blocks, blocks.width},
        .strategy_mask = {actual_strategy.data(), blocks, blocks.width},
        .pixel_mask = {actual_pixel.data(), kPaddedExtent, kPaddedExtent.width},
      },
      {
        .quant_field = {final_quant.data(), blocks, blocks.width},
        .frame = &cuda_frame,
      }), "CUDA resident maximum-throughput frontend")) {
    return false;
  }
  const double quant_error = MaximumError(expected_quant, actual_quant);
  const double strategy_error = MaximumError(expected_strategy, actual_strategy);
  const double pixel_error = MaximumError(expected_pixel, actual_pixel);
  if (quant_error > 2.0e-6 || strategy_error > 2.0e-6 ||
      pixel_error > 2.0e-5 || actual_quant != final_quant ||
      !cuda_frame.valid()) {
    std::cerr << "CUDA initial frontend differs: quant=" << quant_error
              << " strategy=" << strategy_error << " pixel=" << pixel_error
              << '\n';
    return false;
  }

  float quant_dc = 0.0f;
  std::vector<int32_t> raw_quant(block_count);
  gjxl::Quantizer quantizer;
  gjxl::ColorCorrelationMap color;
  ImageStorage filtered(kPaddedExtent);
  gjxl::FrameGeometry geometry;
  if (!Check(gjxl::ComputeInitialQuantDc(aq_options.butteraugli_target, &quant_dc),
             "CPU DC quantization reference") ||
      !Check(gjxl::CreateQuantizerFromField(
        quant_dc, {expected_quant.data(), blocks, blocks.width},
        {raw_quant.data(), blocks, blocks.width}, &quantizer),
        "CPU quantizer reference") ||
      !Check(gjxl::chroma_from_luma_internal::ComputeInitialColorCorrelationMapFast(
        opsin.View(), &color), "CPU initial CfL reference") ||
      !Check(gjxl::ApplyGaborishInverse(
        opsin.View(), aq_options.profile.gaborish_inverse_multipliers,
        filtered.View()), "CPU inverse Gaborish reference") ||
      !Check(gjxl::FrameGeometry::Create(kSourceExtent, &geometry),
             "Create CPU frame geometry")) {
    return false;
  }
  gjxl::VarDctEncoderFrame cpu_frame;
  if (!Check(gjxl::ComputeQuantizedCoefficients(
      std::as_const(filtered).View(),
      {
        .geometry = geometry,
        .strategies = &strategies,
        .raw_quant_field = {raw_quant.data(), blocks, blocks.width},
        .quantizer = &quantizer,
        .color_correlation = &color,
        .epf_sharpness = {sharpness.data(), blocks, blocks.width},
      }, aq_options.profile, &cpu_frame), "CPU coefficient reference")) {
    return false;
  }
  std::vector<uint8_t> cpu_bytes;
  std::vector<uint8_t> cuda_bytes;
  if (!Check(gjxl::EncodeVarDctCodestream(cpu_frame, &cpu_bytes),
             "Encode CPU reference frame") ||
      !Check(gjxl::EncodeVarDctCodestream(cuda_frame, &cuda_bytes),
             "Encode CUDA frame") ||
      cpu_bytes != cuda_bytes) {
    std::cerr << "CUDA maximum-throughput frame differs from CPU bytes\n";
    return false;
  }
  return true;
}

bool CheckPreparedReuseAndFailure(
  gjxl::GpuBackend& gpu,
  const ImageStorage& source,
  const ImageStorage& opsin) {
  const gjxl::Extent2D blocks{kPaddedExtent.width / 8, kPaddedExtent.height / 8};
  const size_t block_count = blocks.width * blocks.height;
  const size_t pixel_count = kPaddedExtent.width * kPaddedExtent.height;
  gjxl::AcStrategyGrid strategies;
  std::vector<uint8_t> sharpness(block_count);
  if (!gjxl::AcStrategyGrid::Create(blocks, &strategies).ok() ||
      !gjxl::FillDefaultEpfSharpness(
        {sharpness.data(), blocks, blocks.width}).ok()) return false;
  strategies.fill_dct8();
  gjxl::AdaptiveQuantizationOptions aq_options;
  const gjxl::InitialQuantizationOptions initial_options{1.0f, 1.0f};
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!Check(gjxl::PrepareAqEvaluation(
      gpu,
      {
        .original_linear_rgb = source.View(),
        .coding_opsin = opsin.View(),
        .strategies = &strategies,
        .epf_sharpness = {sharpness.data(), blocks, blocks.width},
        .options = {aq_options.profile, aq_options.butteraugli},
        .frame_only = true,
        .frame_only_inverse_gaborish = true,
        .resident_initial_cfl = true,
        .frame_only_resident_initial_quant = true,
        .frame_only_resident_quantizer = true,
        .coefficient_decision_mode =
          gjxl::AcCoefficientDecisionMode::kAdjustedSharedQuant,
      }, &prepared), "Prepare reusable CUDA AQ operation")) {
    return false;
  }
  const gjxl::GpuBackendStats prepared_stats = gpu.stats();
  std::vector<float> quant(block_count, 19.0f);
  std::vector<float> strategy(block_count, 19.0f);
  std::vector<float> pixel(pixel_count, 19.0f);
  const auto output = gjxl::InitialQuantFieldOutput{
    .quant_field = {quant.data(), blocks, blocks.width},
    .strategy_mask = {strategy.data(), blocks, blocks.width},
    .pixel_mask = {pixel.data(), kPaddedExtent, kPaddedExtent.width},
  };
  gjxl::QuantizerParams params;
  float quant_dc = 0.0f;
  if (!gjxl::ComputeInitialQuantDc(1.0f, &quant_dc).ok() ||
      !gjxl::ArmNextCudaSubmissionFailureForTest(gpu, true, false).ok()) {
    return false;
  }
  const gjxl::Status failed = prepared->ComputeInitialQuantization(
    initial_options, output, &params, quant_dc);
  if (failed.ok() ||
      !std::all_of(quant.begin(), quant.end(), [](float v) { return v == 19.0f; }) ||
      !std::all_of(strategy.begin(), strategy.end(), [](float v) { return v == 19.0f; }) ||
      !std::all_of(pixel.begin(), pixel.end(), [](float v) { return v == 19.0f; })) {
    std::cerr << "CUDA initial-quantization failure was not atomic\n";
    return false;
  }
  if (!Check(prepared->ComputeInitialQuantization(
      initial_options, output, &params, quant_dc), "Retry CUDA initial quantization")) {
    return false;
  }
  gjxl::VarDctEncoderFrame first;
  gjxl::VarDctEncoderFrame second;
  if (!Check(prepared->EncodeFrame({.quantizer = params}, &first),
             "First reusable CUDA frame") ||
      !Check(prepared->EncodeFrame({.quantizer = params}, &second),
             "Second reusable CUDA frame") ||
      gpu.stats().successful_allocations != prepared_stats.successful_allocations) {
    std::cerr << "CUDA prepared AQ operation allocated during steady-state use\n";
    return false;
  }
  std::vector<uint8_t> first_bytes;
  std::vector<uint8_t> second_bytes;
  return gjxl::EncodeVarDctCodestream(first, &first_bytes).ok() &&
    gjxl::EncodeVarDctCodestream(second, &second_bytes).ok() &&
    first_bytes == second_bytes;
}

bool CheckPublicWorkflow(
  gjxl::GpuBackend& gpu,
  const ImageStorage& source) {
  std::vector<uint8_t> bytes;
  gjxl::VarDctEncodingSummary summary;
  const gjxl::VarDctEncodingOptions options{
    .butteraugli_target = 1.0f,
    .backend = gjxl::VarDctBackendPreference::kCuda,
    .gpu_aq_mode = gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput,
  };
  if (!Check(gjxl::codestream_internal::
      EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
        source.View(), options, &gpu, false, &bytes, &summary),
      "Forced CUDA public workflow") || bytes.empty() ||
      summary.execution_backend != gjxl::VarDctExecutionBackend::kCuda ||
      summary.gpu_aq_mode !=
        gjxl::GpuAdaptiveQuantizationMode::kMaximumThroughput ||
      !summary.score_history.empty()) {
    return false;
  }

  std::vector<uint8_t> failed_bytes{9, 7, 5};
  const std::vector<uint8_t> original_bytes = failed_bytes;
  gjxl::VarDctEncodingSummary failed_summary{
    .extent = {3, 2}, .encoded_bytes = 17, .score_history = {4.0}};
  const gjxl::VarDctEncodingSummary original_summary = failed_summary;
  if (!gjxl::ArmNextCudaSubmissionFailureForTest(gpu, true, false).ok()) {
    return false;
  }
  const gjxl::Status failed = gjxl::codestream_internal::
    EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      source.View(), options, &gpu, false, &failed_bytes, &failed_summary);
  if (failed.ok() || failed_bytes != original_bytes ||
      failed_summary != original_summary) {
    std::cerr << "Forced CUDA public workflow failure was not atomic\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  std::unique_ptr<gjxl::GpuBackend> gpu;
  const gjxl::Status factory = gjxl::CreateCudaBackend(&gpu);
  if (!factory.ok()) {
    if (factory.code() == gjxl::StatusCode::kUnavailable) {
      std::cout << "CUDA unavailable: " << factory.message() << '\n';
      return 77;
    }
    std::cerr << "CUDA factory failed: " << factory.message() << '\n';
    return EXIT_FAILURE;
  }
  ImageStorage source(kSourceExtent);
  ImageStorage padded(kPaddedExtent);
  ImageStorage opsin(kPaddedExtent);
  FillLinear(&source, &padded);
  if (!Check(gjxl::LinearRgbToOpsin(
             std::as_const(padded).View(), 255.0f, opsin.View()),
             "Prepare CUDA AQ opsin") ||
      !CheckResidentFrontend(*gpu, source, opsin) ||
      !CheckPreparedReuseAndFailure(*gpu, source, opsin) ||
      !CheckPublicWorkflow(*gpu, source)) {
    return EXIT_FAILURE;
  }
  std::cout << "CUDA maximum-throughput AQ matches CPU and is reusable.\n";
  return EXIT_SUCCESS;
}
