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

bool MakeExactStrategies(gjxl::AcStrategyGrid* strategies) {
  const gjxl::Extent2D blocks{kPaddedExtent.width / 8,
                              kPaddedExtent.height / 8};
  if (!Check(gjxl::AcStrategyGrid::Create(blocks, strategies),
             "Create CUDA exact strategy grid") ||
      !Check(strategies->Set(0, 0, gjxl::AcStrategyType::kDct16x16),
             "Place CUDA exact DCT16x16") ||
      !Check(strategies->Set(2, 1, gjxl::AcStrategyType::kDct16x8),
             "Place CUDA exact DCT16x8") ||
      !Check(strategies->Set(2, 0, gjxl::AcStrategyType::kDct8x16),
             "Place CUDA exact DCT8x16") ||
      !Check(strategies->Set(4, 0, gjxl::AcStrategyType::kDct16x32),
             "Place CUDA exact DCT16x32")) {
    return false;
  }
  strategies->fill_empty_dct8();
  return strategies->complete();
}

bool CheckResidentStrategyGridValidation(
    gjxl::GpuBackend& gpu, const ImageStorage& source,
    const ImageStorage& opsin) {
  const gjxl::Extent2D blocks{kPaddedExtent.width / 8,
                              kPaddedExtent.height / 8};
  const size_t block_count = blocks.width * blocks.height;
  gjxl::AcStrategyGrid crossing;
  gjxl::AcStrategyGrid valid;
  std::vector<uint8_t> sharpness(block_count);
  if (!Check(gjxl::AcStrategyGrid::Create(blocks, &crossing),
             "Create crossing CUDA resident strategy grid") ||
      !Check(crossing.Set(7, 0, gjxl::AcStrategyType::kDct16x16),
             "Place crossing CUDA resident strategy") ||
      !Check(gjxl::AcStrategyGrid::Create(blocks, &valid),
             "Create valid CUDA resident strategy grid") ||
      !Check(gjxl::FillDefaultEpfSharpness(
                 {sharpness.data(), blocks, blocks.width}),
             "Fill CUDA resident strategy-validation sharpness")) {
    return false;
  }
  crossing.fill_empty_dct8();
  valid.fill_dct8();

  gjxl::prepared_coefficients_internal::PreparedForwardDctCoefficients
      cpu_prepared;
  const gjxl::Status cpu_status =
      gjxl::prepared_coefficients_internal::PrepareForwardDctCoefficients(
          opsin.View(), crossing, &cpu_prepared);
  if (cpu_status.code() != gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "CPU reconstruction accepted a color-tile-crossing strategy\n";
    return false;
  }

  gjxl::AdaptiveQuantizationOptions options;
  const auto preparation = [&](const gjxl::AcStrategyGrid& strategies) {
    return gjxl::AqEvaluationPreparation{
        .original_linear_rgb = source.View(),
        .coding_opsin = opsin.View(),
        .strategies = &strategies,
        .epf_sharpness = {sharpness.data(), blocks, blocks.width},
        .options = {options.profile, options.butteraugli},
        .resident_quantization = true,
        .coefficient_decision_mode =
            gjxl::AcCoefficientDecisionMode::kAdjustedSharedQuant};
  };

  const gjxl::GpuBackendStats before_rejected_prepare = gpu.stats();
  std::unique_ptr<gjxl::PreparedAqEvaluation> rejected;
  const gjxl::Status rejected_prepare =
      gjxl::PrepareAqEvaluation(gpu, preparation(crossing), &rejected);
  const gjxl::GpuBackendStats after_rejected_prepare = gpu.stats();
  if (rejected_prepare.code() != gjxl::StatusCode::kInvalidArgument ||
      rejected != nullptr ||
      after_rejected_prepare.successful_allocations !=
          before_rejected_prepare.successful_allocations ||
      after_rejected_prepare.committed_submissions !=
          before_rejected_prepare.committed_submissions) {
    std::cerr << "CUDA resident preparation accepted or processed a "
                 "color-tile-crossing strategy\n";
    return false;
  }

  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!Check(gjxl::PrepareAqEvaluation(
                 gpu, preparation(valid), &prepared),
             "Prepare CUDA resident strategy-validation operation")) {
    return false;
  }
  const gjxl::GpuBackendStats before_reconfigure = gpu.stats();
  const gjxl::Status rejected_reconfigure = prepared->Reconfigure(
      crossing, {sharpness.data(), blocks, blocks.width});
  const gjxl::GpuBackendStats after_reconfigure = gpu.stats();
  if (rejected_reconfigure.code() != gjxl::StatusCode::kInvalidArgument ||
      after_reconfigure.successful_allocations !=
          before_reconfigure.successful_allocations ||
      after_reconfigure.committed_submissions !=
          before_reconfigure.committed_submissions ||
      !Check(prepared->Reconfigure(
                 valid, {sharpness.data(), blocks, blocks.width}),
             "Reuse CUDA resident operation after rejected reconfiguration")) {
    std::cerr << "CUDA resident reconfiguration accepted a "
                 "color-tile-crossing strategy or damaged prepared state\n";
    return false;
  }
  return true;
}

bool CheckExactWorkflow(gjxl::GpuBackend& gpu, const ImageStorage& source,
                        const ImageStorage& opsin) {
  const gjxl::Extent2D blocks{kPaddedExtent.width / 8,
                              kPaddedExtent.height / 8};
  const size_t block_count = blocks.width * blocks.height;
  gjxl::AcStrategyGrid strategies;
  std::vector<uint8_t> sharpness(block_count);
  std::vector<float> initial(block_count);
  if (!MakeExactStrategies(&strategies) ||
      !Check(gjxl::FillDefaultEpfSharpness(
                 {sharpness.data(), blocks, blocks.width}),
             "Fill CUDA exact sharpness")) {
    return false;
  }
  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      initial[y * blocks.width + x] =
          0.72f + 0.017f * static_cast<float>((5 * x + 3 * y) % 19);
    }
  }

  gjxl::AdaptiveQuantizationOptions options;
  options.butteraugli_target = 1.15f;
  options.iterations = 1;
  options.fast_color_correlation = false;
  options.profile.x_qm_scale = 3;
  options.profile.b_qm_scale = 1;
  options.profile.loop_filter.gaborish_options.weight1 =
      {0.071f, 0.093f, 0.057f};
  options.profile.loop_filter.gaborish_options.weight2 =
      {0.039f, 0.027f, 0.045f};
  options.profile.loop_filter.epf_options.iterations = 3;
  options.profile.loop_filter.epf_options.channel_scale =
      {31.0f, 7.0f, 4.25f};
  options.profile.loop_filter.epf_options.pass0_sigma_scale = 1.17f;
  options.profile.loop_filter.epf_options.pass2_sigma_scale = 4.75f;
  options.profile.loop_filter.epf_options.border_sad_multiplier = 0.81f;
  options.butteraugli = {0.91f, 1.07f, 80.0f};

  std::vector<float> cpu_quant(block_count);
  std::vector<float> cpu_block(block_count);
  std::vector<double> cpu_scores;
  ImageStorage cpu_reconstruction(kSourceExtent);
  gjxl::VarDctEncoderFrame cpu_frame;
  gjxl::MaximumErrorResult cpu_maximum;
  if (!Check(gjxl::FindBestQuantization(
                 source.View(), opsin.View(), strategies,
                 {initial.data(), blocks, blocks.width},
                 {sharpness.data(), blocks, blocks.width}, options,
                 {.quant_field = {cpu_quant.data(), blocks, blocks.width},
                  .block_distance_map =
                      {cpu_block.data(), blocks, blocks.width},
                  .reconstructed_linear_rgb = cpu_reconstruction.View(),
                  .frame = &cpu_frame,
                  .score_history = &cpu_scores,
                  .maximum_error_result = &cpu_maximum}),
             "CPU exact AQ oracle")) {
    return false;
  }

  constexpr float kPoison = -9876.0f;
  const size_t output_stride = blocks.width + 5;
  std::vector<float> cuda_quant(output_stride * blocks.height, kPoison);
  std::vector<float> cuda_block(output_stride * blocks.height, kPoison);
  std::vector<double> cuda_scores;
  ImageStorage cuda_reconstruction(kSourceExtent, kPoison);
  gjxl::VarDctEncoderFrame cuda_frame;
  gjxl::MaximumErrorResult cuda_maximum;
  const gjxl::GpuBackendStats before = gpu.stats();
  if (!Check(gjxl::RunGpuAdaptiveQuantization(
                 gpu, source.View(), opsin.View(), strategies,
                 {initial.data(), blocks, blocks.width},
                 {sharpness.data(), blocks, blocks.width}, options,
                 gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients,
                 {.quant_field = {cuda_quant.data(), blocks, output_stride},
                  .block_distance_map =
                      {cuda_block.data(), blocks, output_stride},
                  .reconstructed_linear_rgb = cuda_reconstruction.View(),
                  .frame = &cuda_frame,
                  .score_history = &cuda_scores,
                  .maximum_error_result = &cuda_maximum}),
             "CUDA exact AQ workflow")) {
    return false;
  }
  const gjxl::GpuBackendStats after = gpu.stats();
  const size_t evaluation_count = options.iterations + 1;
  if (after.successful_allocations != before.successful_allocations + 3 ||
      after.committed_submissions !=
          before.committed_submissions + 1 + 3 * evaluation_count) {
    std::cerr << "CUDA exact AQ resource accounting differs\n";
    return false;
  }

  double quant_error = 0.0;
  double block_error = 0.0;
  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      quant_error = std::max(
          quant_error,
          std::abs(static_cast<double>(cpu_quant[y * blocks.width + x]) -
                   cuda_quant[y * output_stride + x]));
      block_error = std::max(
          block_error,
          std::abs(static_cast<double>(cpu_block[y * blocks.width + x]) -
                   cuda_block[y * output_stride + x]));
    }
    for (size_t x = blocks.width; x < output_stride; ++x) {
      if (cuda_quant[y * output_stride + x] != kPoison ||
          cuda_block[y * output_stride + x] != kPoison) {
        std::cerr << "CUDA exact AQ changed host row padding\n";
        return false;
      }
    }
  }
  double score_error = 0.0;
  if (cpu_scores.size() != cuda_scores.size()) return false;
  for (size_t i = 0; i < cpu_scores.size(); ++i) {
    score_error = std::max(score_error,
                           std::abs(cpu_scores[i] - cuda_scores[i]));
  }
  double image_error = 0.0;
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kSourceExtent.height; ++y) {
      for (size_t x = 0; x < kSourceExtent.width; ++x) {
        image_error = std::max(
            image_error,
            std::abs(static_cast<double>(
                         cpu_reconstruction.plane[channel]
                                           [y * cpu_reconstruction.stride + x]) -
                     cuda_reconstruction.plane[channel]
                                                [y * cuda_reconstruction.stride +
                                                 x]));
      }
    }
  }
  if (!cpu_frame.valid() || !cuda_frame.valid() || quant_error > 2.0e-3 ||
      block_error > 2.0e-3 || score_error > 2.0e-3 ||
      image_error > 2.0e-3) {
    std::cerr << "CUDA exact AQ differs: quant=" << quant_error
              << " block=" << block_error << " score=" << score_error
              << " image=" << image_error << '\n';
    return false;
  }

  std::vector<float> inverse_sigma(block_count);
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!Check(gjxl::ComputeEpfInverseSigma(
                 strategies, cpu_frame.raw_quant_field(),
                 cpu_frame.quantizer(),
                 {sharpness.data(), blocks, blocks.width},
                 options.profile.epf_sigma,
                 {inverse_sigma.data(), blocks, blocks.width}),
             "Prepare CUDA exact failure EPF") ||
      !Check(gjxl::PrepareAqEvaluation(
                 gpu,
                 {.original_linear_rgb = source.View(),
                  .coding_opsin = opsin.View(),
                  .strategies = &strategies,
                  .epf_sharpness =
                      {sharpness.data(), blocks, blocks.width},
                  .options = {options.profile, options.butteraugli},
                  .coefficient_decision_mode =
                      gjxl::AcCoefficientDecisionMode::kAdjustedSharedQuant},
                 &prepared),
             "Prepare CUDA exact failure operation")) {
    return false;
  }
  std::vector<float> failed_map(block_count, kPoison);
  double failed_score = -1234.0;
  const gjxl::AqEvaluationInput failed_input{
      .raw_quant_field = cpu_frame.raw_quant_field(),
      .quantizer = cpu_frame.quantizer().params(),
      .y_to_x = cpu_frame.color_correlation().y_to_x_map(),
      .y_to_b = cpu_frame.color_correlation().y_to_b_map(),
      .epf_inverse_sigma =
          {inverse_sigma.data(), blocks, blocks.width},
      .exact_coefficients = &cpu_frame,
  };
  const gjxl::AqEvaluationOutput failed_output{
      .block_distance_map =
          {failed_map.data(), blocks, blocks.width},
      .score = &failed_score,
  };
  if (!Check(gjxl::ArmNextCudaSubmissionFailureForTest(gpu, false, true),
             "Arm CUDA exact completion failure")) {
    return false;
  }
  const gjxl::Status failed = prepared->Evaluate(failed_input, failed_output);
  const uint64_t failed_submissions = gpu.stats().committed_submissions;
  const gjxl::Status invalidated =
      prepared->Evaluate(failed_input, failed_output);
  if (failed.ok() ||
      invalidated.code() != gjxl::StatusCode::kFailedPrecondition ||
      gpu.stats().committed_submissions != failed_submissions ||
      failed_score != -1234.0 ||
      !std::ranges::all_of(failed_map,
                           [](float value) { return value == kPoison; })) {
    std::cerr << "CUDA exact completion failure was not atomic\n";
    return false;
  }
  return true;
}

bool CheckExactMaximumError(gjxl::GpuBackend& gpu,
                            const ImageStorage& source,
                            const ImageStorage& opsin) {
  const gjxl::Extent2D blocks{kPaddedExtent.width / 8,
                              kPaddedExtent.height / 8};
  const size_t block_count = blocks.width * blocks.height;
  gjxl::AcStrategyGrid strategies;
  std::vector<uint8_t> sharpness(block_count);
  std::vector<float> initial(block_count, 0.9f);
  if (!MakeExactStrategies(&strategies) ||
      !Check(gjxl::FillDefaultEpfSharpness(
                 {sharpness.data(), blocks, blocks.width}),
             "Fill CUDA maximum-error sharpness")) {
    return false;
  }
  gjxl::AdaptiveQuantizationOptions options;
  options.control_mode =
      gjxl::AdaptiveQuantizationControlMode::kMaximumError;
  options.maximum_error = {0.035f, 0.05f, 0.065f};
  options.fast_color_correlation = false;

  struct Result {
    explicit Result(size_t count)
        : quant(count), block(count), reconstruction(kSourceExtent) {}
    std::vector<float> quant;
    std::vector<float> block;
    ImageStorage reconstruction;
    gjxl::VarDctEncoderFrame frame;
    std::vector<double> scores;
    gjxl::MaximumErrorResult maximum;
  };
  Result cpu(block_count);
  Result cuda(block_count);
  const auto output = [&](Result& result) {
    return gjxl::AdaptiveQuantizationOutput{
        .quant_field = {result.quant.data(), blocks, blocks.width},
        .block_distance_map = {result.block.data(), blocks, blocks.width},
        .reconstructed_linear_rgb = result.reconstruction.View(),
        .frame = &result.frame,
        .score_history = &result.scores,
        .maximum_error_result = &result.maximum,
    };
  };
  if (!Check(gjxl::FindBestQuantization(
                 source.View(), opsin.View(), strategies,
                 {initial.data(), blocks, blocks.width},
                 {sharpness.data(), blocks, blocks.width}, options,
                 output(cpu)),
             "CPU maximum-error AQ oracle")) {
    return false;
  }
  const gjxl::GpuBackendStats before = gpu.stats();
  if (!Check(gjxl::RunGpuAdaptiveQuantization(
                 gpu, source.View(), opsin.View(), strategies,
                 {initial.data(), blocks, blocks.width},
                 {sharpness.data(), blocks, blocks.width}, options,
                 gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients,
                 output(cuda)),
             "CUDA exact maximum-error AQ")) {
    return false;
  }
  const gjxl::GpuBackendStats after = gpu.stats();
  if (after.successful_allocations != before.successful_allocations + 2 ||
      after.committed_submissions !=
          before.committed_submissions + cuda.maximum.evaluation_count ||
      cpu.maximum.evaluation_count != cuda.maximum.evaluation_count ||
      cpu.maximum.outcome != cuda.maximum.outcome ||
      cpu.scores.size() != cuda.scores.size()) {
    std::cerr << "CUDA maximum-error resource or policy result differs\n";
    return false;
  }

  double difference =
      std::abs(static_cast<double>(cpu.maximum.normalized_maximum) -
               cuda.maximum.normalized_maximum);
  for (size_t channel = 0; channel < 3; ++channel) {
    difference = std::max(
        difference,
        std::abs(static_cast<double>(cpu.maximum.achieved[channel]) -
                 cuda.maximum.achieved[channel]));
    for (size_t y = 0; y < kSourceExtent.height; ++y) {
      for (size_t x = 0; x < kSourceExtent.width; ++x) {
        difference = std::max(
            difference,
            std::abs(static_cast<double>(
                         cpu.reconstruction.plane[channel]
                                                 [y * cpu.reconstruction.stride +
                                                  x]) -
                     cuda.reconstruction.plane[channel]
                                                  [y * cuda.reconstruction.stride +
                                                   x]));
      }
    }
  }
  for (size_t index = 0; index < block_count; ++index) {
    difference = std::max(
        difference,
        std::abs(static_cast<double>(cpu.quant[index]) - cuda.quant[index]));
    difference = std::max(
        difference,
        std::abs(static_cast<double>(cpu.block[index]) - cuda.block[index]));
  }
  for (size_t index = 0; index < cpu.scores.size(); ++index) {
    difference =
        std::max(difference, std::abs(cpu.scores[index] - cuda.scores[index]));
  }
  std::vector<uint8_t> cpu_bytes;
  std::vector<uint8_t> cuda_bytes;
  if (!Check(gjxl::EncodeVarDctCodestream(cpu.frame, &cpu_bytes),
             "Encode CPU maximum-error frame") ||
      !Check(gjxl::EncodeVarDctCodestream(cuda.frame, &cuda_bytes),
             "Encode CUDA maximum-error frame") ||
      cpu_bytes != cuda_bytes || difference > 2.0e-4) {
    std::cerr << "CUDA exact maximum-error differs by " << difference << '\n';
    return false;
  }
  return true;
}

bool CheckFullyResident(gjxl::GpuBackend& gpu, const ImageStorage& source,
                        const ImageStorage& opsin) {
  const gjxl::Extent2D blocks{kPaddedExtent.width / 8,
                              kPaddedExtent.height / 8};
  const size_t block_count = blocks.width * blocks.height;
  gjxl::AcStrategyGrid strategies;
  std::vector<uint8_t> sharpness(block_count);
  std::vector<float> initial(block_count);
  if (!MakeExactStrategies(&strategies) ||
      !Check(gjxl::FillDefaultEpfSharpness(
                 {sharpness.data(), blocks, blocks.width}),
             "Fill CUDA resident sharpness")) {
    return false;
  }
  for (size_t index = 0; index < block_count; ++index) {
    initial[index] = 0.78f + 0.011f * static_cast<float>(index % 23);
  }
  struct Result {
    explicit Result(size_t count)
        : quant(count), block(count), reconstruction(kSourceExtent) {}
    std::vector<float> quant;
    std::vector<float> block;
    ImageStorage reconstruction;
    gjxl::VarDctEncoderFrame frame;
    std::vector<double> scores;
  };
  const auto output = [&](Result& result) {
    return gjxl::AdaptiveQuantizationOutput{
        .quant_field = {result.quant.data(), blocks, blocks.width},
        .block_distance_map = {result.block.data(), blocks, blocks.width},
        .reconstructed_linear_rgb = result.reconstruction.View(),
        .frame = &result.frame,
        .score_history = &result.scores};
  };

  gjxl::AdaptiveQuantizationOptions options;
  options.butteraugli_target = 1.1f;
  options.profile.x_qm_scale = 2;
  options.profile.b_qm_scale = 1;

  // With no policy update, the resident path is a useful exact oracle for the
  // strategy-aware quantization, coefficient, CfL, and reconstruction kernels.
  Result cpu(block_count);
  Result cuda(block_count);
  options.iterations = 0;
  if (!Check(gjxl::FindBestQuantization(
                 source.View(), opsin.View(), strategies,
                 {initial.data(), blocks, blocks.width},
                 {sharpness.data(), blocks, blocks.width}, options,
                 output(cpu)),
             "CPU resident AQ oracle") ||
      !Check(gjxl::RunGpuAdaptiveQuantization(
                 gpu, source.View(), opsin.View(), strategies,
                 {initial.data(), blocks, blocks.width},
                 {sharpness.data(), blocks, blocks.width}, options,
                 gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
                 output(cuda)),
             "CUDA fully resident AQ")) {
    return false;
  }
  double difference = std::max(MaximumError(cpu.quant, cuda.quant),
                               MaximumError(cpu.block, cuda.block));
  if (cpu.scores.size() != cuda.scores.size()) return false;
  for (size_t index = 0; index < cpu.scores.size(); ++index) {
    difference =
        std::max(difference, std::abs(cpu.scores[index] - cuda.scores[index]));
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kSourceExtent.height; ++y) {
      for (size_t x = 0; x < kSourceExtent.width; ++x) {
        difference = std::max(
            difference,
            std::abs(static_cast<double>(
                         cpu.reconstruction.plane[channel]
                                                 [y * cpu.reconstruction.stride +
                                                  x]) -
                     cuda.reconstruction.plane[channel]
                                                  [y * cuda.reconstruction.stride +
                                                   x]));
      }
    }
  }
  std::vector<uint8_t> cpu_bytes;
  std::vector<uint8_t> cuda_bytes;
  if (!Check(gjxl::EncodeVarDctCodestream(cpu.frame, &cpu_bytes),
             "Encode CPU resident frame") ||
      !Check(gjxl::EncodeVarDctCodestream(cuda.frame, &cuda_bytes),
             "Encode CUDA resident frame") ||
      cpu_bytes != cuda_bytes || difference > 2.0e-3) {
    std::cerr << "CUDA iteration-zero resident AQ differs by " << difference
              << '\n';
    return false;
  }

  // Once policy updates begin, fixed resident CfL deliberately differs from
  // the ordinary CPU evaluator. The public contract is deterministic bounded
  // and full materialization of the same resident policy result.
  options.iterations = 3;
  std::vector<float> bounded_quant(block_count);
  std::vector<float> bounded_block(block_count);
  std::vector<double> bounded_scores;
  Result full(block_count);
  const gjxl::GpuBackendStats before_bounded = gpu.stats();
  if (!Check(gjxl::RunGpuAdaptiveQuantizationPolicy(
                 gpu, source.View(), opsin.View(), strategies,
                 {initial.data(), blocks, blocks.width},
                 {sharpness.data(), blocks, blocks.width}, options,
                 gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
                 {.quant_field =
                      {bounded_quant.data(), blocks, blocks.width},
                  .block_distance_map =
                      {bounded_block.data(), blocks, blocks.width},
                  .score_history = &bounded_scores}),
             "CUDA bounded fully resident AQ") ||
      gpu.stats().successful_allocations !=
          before_bounded.successful_allocations + 3 ||
      gpu.stats().committed_submissions !=
          before_bounded.committed_submissions + 3) {
    std::cerr << "CUDA bounded resident resource count differs\n";
    return false;
  }
  const gjxl::GpuBackendStats before_full = gpu.stats();
  if (!Check(gjxl::RunGpuAdaptiveQuantization(
                 gpu, source.View(), opsin.View(), strategies,
                 {initial.data(), blocks, blocks.width},
                 {sharpness.data(), blocks, blocks.width}, options,
                 gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
                 output(full)),
             "CUDA full fully resident AQ") ||
      gpu.stats().successful_allocations !=
          before_full.successful_allocations + 3 ||
      gpu.stats().committed_submissions !=
          before_full.committed_submissions + 3 ||
      bounded_quant != full.quant || bounded_block != full.block ||
      bounded_scores != full.scores || full.scores.size() != 4 ||
      !full.frame.valid()) {
    std::cerr << "CUDA resident bounded and full results differ\n";
    return false;
  }
  for (double score : full.scores) {
    if (!std::isfinite(score) || score < 0.0) return false;
  }
  for (float value : full.quant) {
    if (!std::isfinite(value) || value <= 0.0f) return false;
  }
  for (float value : full.block) {
    if (!std::isfinite(value) || value < 0.0f) return false;
  }
  for (const std::vector<float>& plane : full.reconstruction.plane) {
    for (size_t y = 0; y < kSourceExtent.height; ++y) {
      for (size_t x = 0; x < kSourceExtent.width; ++x) {
        if (!std::isfinite(plane[y * full.reconstruction.stride + x])) {
          return false;
        }
      }
    }
  }
  std::vector<uint8_t> full_bytes;
  if (!Check(gjxl::EncodeVarDctCodestream(full.frame, &full_bytes),
             "Encode fully resident CUDA frame") ||
      full_bytes.empty()) {
    return false;
  }

  constexpr float kPoison = -4321.0f;
  std::vector<float> failed_quant(block_count, kPoison);
  std::vector<float> failed_block(block_count, kPoison);
  std::vector<double> failed_scores{17.0};
  ImageStorage failed_reconstruction(kSourceExtent, kPoison);
  gjxl::VarDctEncoderFrame failed_frame;
  if (!Check(gjxl::ArmNextCudaSubmissionFailureForTest(gpu, false, true),
             "Arm CUDA resident completion failure")) {
    return false;
  }
  const gjxl::Status failed = gjxl::RunGpuAdaptiveQuantization(
      gpu, source.View(), opsin.View(), strategies,
      {initial.data(), blocks, blocks.width},
      {sharpness.data(), blocks, blocks.width}, options,
      gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
      {.quant_field = {failed_quant.data(), blocks, blocks.width},
       .block_distance_map = {failed_block.data(), blocks, blocks.width},
       .reconstructed_linear_rgb = failed_reconstruction.View(),
       .frame = &failed_frame,
       .score_history = &failed_scores});
  if (failed.ok() ||
      !std::ranges::all_of(failed_quant,
                           [](float value) { return value == kPoison; }) ||
      !std::ranges::all_of(failed_block,
                           [](float value) { return value == kPoison; }) ||
      failed_scores != std::vector<double>{17.0} || failed_frame.valid()) {
    std::cerr << "CUDA resident failure was not atomic\n";
    return false;
  }
  for (const std::vector<float>& plane : failed_reconstruction.plane) {
    if (!std::ranges::all_of(plane,
                             [](float value) { return value == kPoison; })) {
      std::cerr << "CUDA resident failure changed reconstruction output\n";
      return false;
    }
  }
  return true;
}

bool CheckResidentInvariantColorCorrelationContract(
    gjxl::GpuBackend& gpu, const ImageStorage& source,
    const ImageStorage& opsin) {
  const gjxl::Extent2D blocks{kPaddedExtent.width / 8,
                              kPaddedExtent.height / 8};
  const size_t block_count = blocks.width * blocks.height;
  gjxl::AcStrategyGrid strategies;
  std::vector<uint8_t> sharpness(block_count);
  std::vector<float> prepared_field(block_count);
  std::vector<float> evaluation_field(block_count);
  if (!MakeExactStrategies(&strategies) ||
      !Check(gjxl::FillDefaultEpfSharpness(
                 {sharpness.data(), blocks, blocks.width}),
             "Fill CUDA resident invariant-CfL sharpness")) {
    return false;
  }
  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      const size_t index = y * blocks.width + x;
      prepared_field[index] = ((x + 3 * y) % 8) < 4 ? 0.24f : 3.4f;
      evaluation_field[index] = ((x + 5 * y) % 8) < 4 ? 3.1f : 0.31f;
    }
  }
  float prepared_quant_dc = 0.0f;
  if (!Check(gjxl::ComputeInitialQuantDc(1.0f, &prepared_quant_dc),
             "Prepare CUDA resident invariant-CfL DC")) {
    return false;
  }
  const float evaluation_quant_dc = prepared_quant_dc * 0.875f;

  std::vector<int32_t> prepared_raw(block_count);
  std::vector<int32_t> evaluation_raw(block_count);
  gjxl::Quantizer prepared_quantizer;
  gjxl::Quantizer evaluation_quantizer;
  gjxl::ColorCorrelationMap prepared_color;
  gjxl::ColorCorrelationMap evaluation_color;
  if (!Check(gjxl::CreateQuantizerFromField(
                 prepared_quant_dc,
                 {prepared_field.data(), blocks, blocks.width},
                 {prepared_raw.data(), blocks, blocks.width},
                 &prepared_quantizer),
             "Build CUDA resident invariant-CfL reference quantizer") ||
      !Check(gjxl::ComputeFinalColorCorrelationMap(
                 opsin.View(), strategies,
                 {prepared_raw.data(), blocks, blocks.width},
                 prepared_quantizer, true, &prepared_color),
             "Build CUDA resident invariant-CfL reference map") ||
      !Check(gjxl::CreateQuantizerFromField(
                 evaluation_quant_dc,
                 {evaluation_field.data(), blocks, blocks.width},
                 {evaluation_raw.data(), blocks, blocks.width},
                 &evaluation_quantizer),
             "Build CUDA resident evaluation reference quantizer") ||
      !Check(gjxl::ComputeFinalColorCorrelationMap(
                 opsin.View(), strategies,
                 {evaluation_raw.data(), blocks, blocks.width},
                 evaluation_quantizer, true, &evaluation_color),
             "Build CUDA resident evaluation reference map")) {
    return false;
  }
  const auto maps_equal = [](const gjxl::ColorCorrelationMap& left,
                             const gjxl::ColorCorrelationMap& right) {
    if (!left.valid() || !right.valid() ||
        left.tile_extent() != right.tile_extent()) {
      return false;
    }
    const auto left_x = left.y_to_x_map();
    const auto right_x = right.y_to_x_map();
    const auto left_b = left.y_to_b_map();
    const auto right_b = right.y_to_b_map();
    for (size_t y = 0; y < left.tile_extent().height; ++y) {
      if (!std::equal(left_x.Row(y),
                      left_x.Row(y) + left.tile_extent().width,
                      right_x.Row(y)) ||
          !std::equal(left_b.Row(y),
                      left_b.Row(y) + left.tile_extent().width,
                      right_b.Row(y))) {
        return false;
      }
    }
    return true;
  };
  if (maps_equal(prepared_color, evaluation_color)) {
    std::cerr << "CUDA resident invariant-CfL test maps are not distinct\n";
    return false;
  }

  gjxl::AdaptiveQuantizationOptions options;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!Check(gjxl::PrepareAqEvaluation(
                 gpu,
                 {.original_linear_rgb = source.View(),
                  .coding_opsin = opsin.View(),
                  .strategies = &strategies,
                  .epf_sharpness =
                      {sharpness.data(), blocks, blocks.width},
                  .options = {options.profile, options.butteraugli},
                  .resident_quantization = true,
                  .coefficient_decision_mode =
                      gjxl::AcCoefficientDecisionMode::kAdjustedSharedQuant},
                 &prepared),
             "Prepare CUDA resident invariant-CfL operation")) {
    return false;
  }
  const gjxl::GpuBackendStats before_invariant = gpu.stats();
  if (!Check(prepared->PrepareInvariantColorCorrelationResident(
                 {prepared_field.data(), blocks, blocks.width},
                 prepared_quant_dc),
             "Prepare CUDA resident invariant CfL")) {
    return false;
  }
  const gjxl::GpuBackendStats after_invariant = gpu.stats();
  if (after_invariant.successful_allocations !=
          before_invariant.successful_allocations ||
      after_invariant.committed_submissions !=
          before_invariant.committed_submissions) {
    std::cerr << "CUDA resident invariant CfL was not retained without a "
                 "submission or allocation\n";
    return false;
  }

  constexpr float kPoison = -7654.0f;
  std::vector<float> block_map(block_count, kPoison);
  double score = -1234.0;
  gjxl::QuantizerParams quantizer{1234, 5678};
  gjxl::VarDctEncoderFrame frame;
  gjxl::AqEvaluationOutput::Final final{.frame = &frame};
  const gjxl::AqEvaluationOutput output{
      .block_distance_map = {block_map.data(), blocks, blocks.width},
      .score = &score,
      .quantizer = &quantizer,
      .final = &final};
  if (!Check(prepared->Evaluate(
                 {.quant_field =
                      {evaluation_field.data(), blocks, blocks.width},
                  .quant_dc = evaluation_quant_dc},
                 output),
             "Evaluate CUDA resident invariant CfL with a changed field")) {
    return false;
  }
  if (!frame.valid() ||
      quantizer.global_scale !=
          evaluation_quantizer.params().global_scale ||
      quantizer.quant_dc != evaluation_quantizer.params().quant_dc ||
      !maps_equal(frame.color_correlation(), prepared_color) ||
      maps_equal(frame.color_correlation(), evaluation_color) ||
      !std::isfinite(score) || score < 0.0 ||
      !std::ranges::all_of(block_map, [](float value) {
        return std::isfinite(value) && value >= 0.0f;
      })) {
    std::cerr << "CUDA resident evaluation did not retain prepared invariant "
                 "CfL state\n";
    return false;
  }
  return true;
}

bool CheckResidentMaximumError(gjxl::GpuBackend& gpu,
                               const ImageStorage& source,
                               const ImageStorage& opsin) {
  const gjxl::Extent2D blocks{kPaddedExtent.width / 8,
                              kPaddedExtent.height / 8};
  const size_t block_count = blocks.width * blocks.height;
  gjxl::AcStrategyGrid strategies;
  std::vector<uint8_t> sharpness(block_count);
  std::vector<float> initial(block_count, 0.91f);
  if (!MakeExactStrategies(&strategies) ||
      !Check(gjxl::FillDefaultEpfSharpness(
                 {sharpness.data(), blocks, blocks.width}),
             "Fill CUDA resident maximum-error sharpness")) {
    return false;
  }
  gjxl::AdaptiveQuantizationOptions options;
  options.control_mode =
      gjxl::AdaptiveQuantizationControlMode::kMaximumError;
  options.maximum_error = {0.035f, 0.05f, 0.065f};
  std::vector<float> quant(block_count);
  std::vector<float> block(block_count);
  std::vector<double> scores;
  ImageStorage reconstruction(kSourceExtent);
  gjxl::VarDctEncoderFrame frame;
  gjxl::MaximumErrorResult maximum;
  if (!Check(gjxl::RunGpuAdaptiveQuantization(
                 gpu, source.View(), opsin.View(), strategies,
                 {initial.data(), blocks, blocks.width},
                 {sharpness.data(), blocks, blocks.width}, options,
                 gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
                 {.quant_field = {quant.data(), blocks, blocks.width},
                  .block_distance_map =
                      {block.data(), blocks, blocks.width},
                  .reconstructed_linear_rgb = reconstruction.View(),
                  .frame = &frame,
                  .score_history = &scores,
                  .maximum_error_result = &maximum}),
             "CUDA resident maximum-error AQ") ||
      !frame.valid() || scores.size() != maximum.evaluation_count ||
      scores.size() != 6 ||
      maximum.outcome == gjxl::MaximumErrorOutcome::kNotApplicable ||
      !std::isfinite(maximum.normalized_maximum) ||
      maximum.normalized_maximum < 0.0f) {
    std::cerr << "CUDA resident maximum-error result is invalid\n";
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    if (!std::isfinite(maximum.achieved[channel]) ||
        maximum.achieved[channel] < 0.0f) {
      return false;
    }
  }
  for (double score : scores) {
    if (!std::isfinite(score) || score < 0.0) return false;
  }
  std::vector<uint8_t> bytes;
  return Check(gjxl::EncodeVarDctCodestream(frame, &bytes),
               "Encode resident maximum-error frame") &&
         !bytes.empty();
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
  if (!gjxl::EncodeVarDctCodestream(first, &first_bytes).ok() ||
      !gjxl::EncodeVarDctCodestream(second, &second_bytes).ok() ||
      first_bytes != second_bytes) {
    return false;
  }

  const std::vector<float> quant_before_failure = quant;
  const std::vector<float> strategy_before_failure = strategy;
  const std::vector<float> pixel_before_failure = pixel;
  gjxl::QuantizerParams failed_params{1234, 5678};
  float changed_quant_dc = 0.0f;
  if (!gjxl::ComputeInitialQuantDc(1.25f, &changed_quant_dc).ok() ||
      !gjxl::ArmNextCudaSubmissionFailureForTest(
        gpu, false, true).ok()) {
    return false;
  }
  const gjxl::Status completion_failure =
    prepared->ComputeInitialQuantization(
      {1.25f, 1.0f}, output, &failed_params, changed_quant_dc);
  if (completion_failure.code() != gjxl::StatusCode::kDeviceError ||
      quant != quant_before_failure ||
      strategy != strategy_before_failure ||
      pixel != pixel_before_failure ||
      failed_params.global_scale != 1234 || failed_params.quant_dc != 5678) {
    std::cerr << "CUDA completion failure was not output-atomic\n";
    return false;
  }

  gjxl::VarDctEncoderFrame rejected;
  if (prepared->EncodeFrame({.quantizer = params}, &rejected).code() !=
        gjxl::StatusCode::kFailedPrecondition ||
      prepared->ComputeInitialQuantization(
        initial_options, output, &failed_params, quant_dc).code() !=
        gjxl::StatusCode::kFailedPrecondition) {
    std::cerr << "CUDA completion failure did not invalidate resident state\n";
    return false;
  }
  return true;
}

bool CheckPublicWorkflow(
  gjxl::GpuBackend& gpu,
  const ImageStorage& source) {
  const auto encode = [&](gjxl::GpuAdaptiveQuantizationMode mode,
                          bool collect_final_score,
                          std::vector<uint8_t>* bytes,
                          gjxl::VarDctEncodingSummary* summary) {
    return gjxl::codestream_internal::
      EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
        source.View(),
        {.butteraugli_target = 1.0f,
         .backend = gjxl::VarDctBackendPreference::kCuda,
         .gpu_aq_mode = mode,
         .collect_final_butteraugli_score = collect_final_score},
        &gpu, false, bytes, summary);
  };

  std::vector<uint8_t> resident_bytes;
  gjxl::VarDctEncodingSummary resident_summary;
  if (!Check(encode(gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
                    true, &resident_bytes, &resident_summary),
      "Forced resident CUDA public workflow") || resident_bytes.empty() ||
      resident_summary.execution_backend !=
        gjxl::VarDctExecutionBackend::kCuda ||
      resident_summary.gpu_aq_mode !=
        gjxl::GpuAdaptiveQuantizationMode::kFullyResident ||
      resident_summary.score_history.size() != 3 ||
      !resident_summary.final_butteraugli_score_evaluated) {
    return false;
  }

  std::vector<uint8_t> resident_default_bytes;
  gjxl::VarDctEncodingSummary resident_default_summary;
  std::vector<uint8_t> throughput_bytes;
  gjxl::VarDctEncodingSummary throughput_summary;
  std::vector<uint8_t> exact_bytes;
  gjxl::VarDctEncodingSummary exact_summary;
  if (!Check(encode(gjxl::GpuAdaptiveQuantizationMode::kFullyResident,
                    false, &resident_default_bytes,
                    &resident_default_summary),
             "Forced resident CUDA workflow without final score") ||
      !Check(encode(gjxl::GpuAdaptiveQuantizationMode::kThroughput,
                    false, &throughput_bytes, &throughput_summary),
             "Forced throughput CUDA public workflow") ||
      !Check(encode(gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients,
                    true, &exact_bytes, &exact_summary),
             "Forced exact CUDA public workflow") ||
      resident_default_bytes != resident_bytes ||
      resident_default_summary.score_history.size() != 2 ||
      resident_default_summary.final_butteraugli_score_evaluated ||
      throughput_bytes.empty() || throughput_summary.score_history.size() != 2 ||
      throughput_summary.final_butteraugli_score_evaluated ||
      throughput_summary.execution_backend !=
        gjxl::VarDctExecutionBackend::kCuda ||
      throughput_summary.gpu_aq_mode !=
        gjxl::GpuAdaptiveQuantizationMode::kThroughput ||
      exact_bytes.empty() || exact_summary.score_history.size() != 3 ||
      !exact_summary.final_butteraugli_score_evaluated ||
      exact_summary.execution_backend != gjxl::VarDctExecutionBackend::kCuda ||
      exact_summary.gpu_aq_mode !=
        gjxl::GpuAdaptiveQuantizationMode::kExactCoefficients) {
    std::cerr << "A forced CUDA public mode failed its output contract\n";
    return false;
  }

  std::vector<uint8_t> maximum_error_bytes;
  gjxl::VarDctEncodingSummary maximum_error_summary;
  if (!Check(gjxl::codestream_internal::
      EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
        source.View(),
        {.rate_control_mode = gjxl::VarDctRateControlMode::kMaximumError,
         .maximum_error = {0.05f, 0.05f, 0.05f},
         .backend = gjxl::VarDctBackendPreference::kCuda,
         .gpu_aq_mode = gjxl::GpuAdaptiveQuantizationMode::kFullyResident},
        &gpu, false, &maximum_error_bytes, &maximum_error_summary),
      "Forced resident CUDA maximum-error workflow") ||
      maximum_error_bytes.empty() ||
      maximum_error_summary.execution_backend !=
        gjxl::VarDctExecutionBackend::kCuda ||
      maximum_error_summary.maximum_error_evaluation_count != 6 ||
      !std::isfinite(maximum_error_summary.achieved_maximum_error_ratio)) {
    return false;
  }

  std::vector<uint8_t> target_bytes;
  gjxl::VarDctEncodingSummary target_summary;
  if (!Check(gjxl::codestream_internal::
      EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
        source.View(),
        {.rate_control_mode = gjxl::VarDctRateControlMode::kTargetBytes,
         .target_bytes = 512,
         .target_size_maximum_attempts = 2,
         .backend = gjxl::VarDctBackendPreference::kCuda,
         .gpu_aq_mode = gjxl::GpuAdaptiveQuantizationMode::kFullyResident},
        &gpu, false, &target_bytes, &target_summary),
      "Forced resident CUDA target-size workflow") || target_bytes.empty() ||
      target_summary.execution_backend != gjxl::VarDctExecutionBackend::kCuda ||
      target_summary.encode_attempt_count != 2) {
    return false;
  }

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
      !CheckExactWorkflow(*gpu, source, opsin) ||
      !CheckExactMaximumError(*gpu, source, opsin) ||
      !CheckResidentStrategyGridValidation(*gpu, source, opsin) ||
      !CheckFullyResident(*gpu, source, opsin) ||
      !CheckResidentInvariantColorCorrelationContract(*gpu, source, opsin) ||
      !CheckResidentMaximumError(*gpu, source, opsin) ||
      !CheckResidentFrontend(*gpu, source, opsin) ||
      !CheckPreparedReuseAndFailure(*gpu, source, opsin) ||
      !CheckPublicWorkflow(*gpu, source)) {
    return EXIT_FAILURE;
  }
  std::cout << "CUDA exact and maximum-throughput AQ match CPU.\n";
  return EXIT_SUCCESS;
}
