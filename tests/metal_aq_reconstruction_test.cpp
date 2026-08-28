// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/chroma_from_luma_internal.h"
#include "codec/epf.h"
#include "codec/quantization.h"
#include "codec/reconstruction.h"
#include "core/ac_strategy.h"
#include "core/quantizer.h"
#include "dct_reference.h"
#include "gpu/metal/metal_aq_evaluation_test.h"
#include "gpu/metal/metal_aq_reconstruction_test.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/aq_evaluation.h"

namespace {

constexpr gjxl::Extent2D kPixelExtent{96, 64};
constexpr gjxl::Extent2D kBlockExtent{12, 8};

bool CheckStatus(gjxl::Status status, std::string_view operation) {
  if (status.ok())
    return true;
  std::cerr << operation << " failed: " << status.message() << '\n';
  return false;
}

bool ExpectCode(gjxl::Status status, gjxl::StatusCode expected,
                std::string_view operation) {

  if (status.code() == expected)
    return true;
  std::cerr << operation << " returned " << static_cast<int>(status.code())
            << ", expected " << static_cast<int>(expected) << ": "
            << status.message() << '\n';
  return false;
}

bool Near(float actual, double expected, double absolute, double relative) {
  return std::abs(static_cast<double>(actual) - expected) <=
         absolute + relative * std::abs(expected);
}

struct HostImage {
  std::array<std::vector<float>, 3> plane;

  static HostImage Structured(bool flat) {
    HostImage image;
    std::mt19937 generator(0x8a317u);
    std::uniform_real_distribution<float> noise(-0.025f, 0.025f);
    for (size_t channel = 0; channel < 3; ++channel) {
      image.plane[channel].resize(kPixelExtent.width * kPixelExtent.height);
      for (size_t y = 0; y < kPixelExtent.height; ++y) {
        for (size_t x = 0; x < kPixelExtent.width; ++x) {
          float value = 0.19f + 0.11f * static_cast<float>(channel);
          if (!flat) {
            value += 0.0017f * static_cast<float>(x) -
                     0.0021f * static_cast<float>(y);
            value +=
                ((x / 7 + 3 * y / 5 + channel) % 2 == 0) ? 0.037f : -0.029f;
            value += noise(generator);
          }
          image.plane[channel][y * kPixelExtent.width + x] = value;
        }
      }
    }
    if (!flat) {
      image.plane[0][3 * kPixelExtent.width + 5] += 0.75f;
      image.plane[1][17 * kPixelExtent.width + 34] -= 0.63f;
      image.plane[2][51 * kPixelExtent.width + 77] += 0.91f;
    }
    return image;
  }

  gjxl::ConstImage3FView View() const {
    return {{{
        {plane[0].data(), kPixelExtent, kPixelExtent.width},
        {plane[1].data(), kPixelExtent, kPixelExtent.width},
        {plane[2].data(), kPixelExtent, kPixelExtent.width},
    }}};
  }
};

bool MakeMixedStrategies(gjxl::AcStrategyGrid *strategies) {
  if (!CheckStatus(gjxl::AcStrategyGrid::Create(kBlockExtent, strategies),
                   "strategy-grid creation") ||
      !CheckStatus(strategies->Set(0, 0, gjxl::AcStrategyType::kDct32x32),
                   "DCT32x32 placement") ||
      !CheckStatus(strategies->Set(4, 0, gjxl::AcStrategyType::kDct32x16),
                   "DCT32x16 placement") ||
      !CheckStatus(strategies->Set(6, 0, gjxl::AcStrategyType::kDct16x32),
                   "DCT16x32 placement") ||
      !CheckStatus(strategies->Set(10, 0, gjxl::AcStrategyType::kDct16x16),
                   "DCT16x16 placement") ||
      !CheckStatus(strategies->Set(6, 2, gjxl::AcStrategyType::kDct16x8),
                   "DCT16x8 placement") ||
      !CheckStatus(strategies->Set(7, 2, gjxl::AcStrategyType::kDct8x16),
                   "DCT8x16 placement")) {
    return false;
  }
  strategies->fill_empty_dct8();
  return strategies->complete();
}

bool MakeUniformStrategies(gjxl::AcStrategyType strategy,
                           gjxl::AcStrategyGrid *strategies) {
  const gjxl::AcStrategyInfo *info = gjxl::GetAcStrategyInfo(strategy);
  if (info == nullptr ||
      !CheckStatus(gjxl::AcStrategyGrid::Create(kBlockExtent, strategies),
                   "uniform strategy-grid creation")) {
    return false;
  }
  for (size_t y = 0; y < kBlockExtent.height;
       y += info->covered_blocks.height) {
    for (size_t x = 0; x < kBlockExtent.width;
         x += info->covered_blocks.width) {
      if (!CheckStatus(strategies->Set(x, y, strategy),
                       "uniform strategy placement")) {
        return false;
      }
    }
  }
  return strategies->complete();
}

struct InputStorage {
  std::vector<int32_t> raw_quant;
  std::vector<float> inverse_sigma;
  gjxl::ColorCorrelationMap color;
  gjxl::QuantizerParams params{8192, 48};

  static bool Make(const HostImage &image, InputStorage *out) {
    out->raw_quant.resize(kBlockExtent.width * kBlockExtent.height);
    out->inverse_sigma.resize(kBlockExtent.width * kBlockExtent.height);
    for (size_t y = 0; y < kBlockExtent.height; ++y) {
      for (size_t x = 0; x < kBlockExtent.width; ++x) {
        out->raw_quant[y * kBlockExtent.width + x] =
            1 + static_cast<int32_t>((37 * x + 19 * y) % 256);
        out->inverse_sigma[y * kBlockExtent.width + x] =
            -0.04f - 0.001f * static_cast<float>(x + y);
      }
    }
    return CheckStatus(
        gjxl::ComputeInitialColorCorrelationMap(image.View(), &out->color),
        "initial color-correlation map");
  }

  gjxl::AqEvaluationInput View() const {
    return {
        .raw_quant_field = {raw_quant.data(), kBlockExtent, kBlockExtent.width},
        .quantizer = params,
        .y_to_x = color.y_to_x_map(),
        .y_to_b = color.y_to_b_map(),
        .epf_inverse_sigma = {inverse_sigma.data(), kBlockExtent,
                              kBlockExtent.width},
    };
  }
};

gjxl::AqEvaluationOptions Options() {
  gjxl::AqEvaluationOptions options;
  options.profile.x_qm_scale = 3;
  options.profile.b_qm_scale = 1;
  return options;
}

gjxl::Status ComputeCpuFrame(
    const HostImage &image, const gjxl::AcStrategyGrid &strategies,
    const InputStorage &input, gjxl::VarDctEncoderFrame *frame) {
  gjxl::Quantizer quantizer;
  gjxl::Status status = gjxl::Quantizer::Create(input.params, &quantizer);
  if (!status.ok())
    return status;
  gjxl::FrameGeometry geometry;
  status = gjxl::FrameGeometry::Create(kPixelExtent, &geometry);
  if (!status.ok())
    return status;
  size_t block_count = 0;
  if (!kBlockExtent.try_area(&block_count)) {
    return gjxl::Status::InvalidArgument("Test block grid is too large");
  }
  std::vector<uint8_t> epf_sharpness(block_count, 4);
  return gjxl::ComputeQuantizedCoefficients(
      image.View(),
      {.geometry = geometry,
       .strategies = &strategies,
       .raw_quant_field = input.View().raw_quant_field,
       .quantizer = &quantizer,
       .color_correlation = &input.color,
       .epf_sharpness =
           {epf_sharpness.data(), kBlockExtent, kBlockExtent.width}},
      Options().profile,
      frame,
      gjxl::AcCoefficientDecisionMode::kFixedRawQuant);
}

gjxl::MetalBackendOptions SimdOptions() {
  gjxl::MetalBackendOptions options;
  using Impl = gjxl::MetalDctImplementation;
  options.forward_dct8 = Impl::kSimdgroupMatmul;
  options.inverse_dct8 = Impl::kSimdgroupMatmul;
  options.forward_dct16x16 = Impl::kSimdgroupMatmul;
  options.inverse_dct16x16 = Impl::kSimdgroupMatmul;
  options.forward_dct32x32 = Impl::kSimdgroupMatmul;
  options.inverse_dct32x32 = Impl::kSimdgroupMatmul;
  options.forward_dct16x8 = Impl::kSimdgroupMatmul;
  options.inverse_dct16x8 = Impl::kSimdgroupMatmul;
  options.forward_dct8x16 = Impl::kSimdgroupMatmul;
  options.inverse_dct8x16 = Impl::kSimdgroupMatmul;
  options.forward_dct32x16 = Impl::kSimdgroupMatmul;
  options.inverse_dct32x16 = Impl::kSimdgroupMatmul;
  options.forward_dct16x32 = Impl::kSimdgroupMatmul;
  options.inverse_dct16x32 = Impl::kSimdgroupMatmul;
  return options;
}

bool Prepare(gjxl::GpuBackend &gpu, const HostImage &image,
             const gjxl::AcStrategyGrid &strategies,
             std::unique_ptr<gjxl::PreparedAqEvaluation> *prepared,
             gjxl::AcCoefficientDecisionMode decision_mode =
                 gjxl::AcCoefficientDecisionMode::kFixedRawQuant) {

  const std::vector<uint8_t> sharpness(
      kBlockExtent.width * kBlockExtent.height, 4);
  const gjxl::AqEvaluationPreparation preparation{
      .original_linear_rgb = image.View(),
      .coding_opsin = image.View(),
      .strategies = &strategies,
      .epf_sharpness = {
          sharpness.data(), kBlockExtent, kBlockExtent.width},
      .options = Options(),
      .coefficient_decision_mode = decision_mode,
  };
  return CheckStatus(gjxl::PrepareAqEvaluation(gpu, preparation, prepared),
                     "prepared AQ reconstruction creation");
}

bool CompareForward(
    const HostImage &image,
    const gjxl::metal_internal::MetalAqReconstructionSnapshotForTesting
        &snapshot) {

  for (const auto &transform : snapshot.transforms) {
    const gjxl::AcStrategyInfo *info =
        gjxl::GetAcStrategyInfo(transform.strategy);
    if (info == nullptr)
      return false;
    const gjxl::Extent2D extent = info->pixel_extent();
    const size_t count = info->coefficient_count();
    std::vector<float> pixels(count);
    std::vector<double> expected(count);
    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t y = 0; y < extent.height; ++y) {
        const size_t source = (transform.block_y * 8 + y) * kPixelExtent.width +
                              transform.block_x * 8;
        std::copy_n(image.plane[channel].data() + source, extent.width,
                    pixels.data() + y * extent.width);
      }
      gjxl::test::ReferenceForwardDct(extent, pixels.data(), expected.data(),
                                      1);
      const double absolute =
          transform.strategy == gjxl::AcStrategyType::kDct8 ? 1.0e-5 : 2.0e-5;
      for (size_t index = 0; index < count; ++index) {
        if (!Near(transform.forward_coefficients[channel][index],
                  expected[index], absolute, 5.0e-5)) {
          std::cerr << "Forward coefficient mismatch for strategy "
                    << static_cast<int>(transform.strategy) << ", channel "
                    << channel << ", coefficient " << index << '\n';
          return false;
        }
      }
    }
  }
  return true;
}

bool CompareCoefficientOracle(
    const HostImage &image, const gjxl::AcStrategyGrid &strategies,
    const InputStorage &input,
    const gjxl::metal_internal::MetalAqReconstructionSnapshotForTesting
        &snapshot) {

  gjxl::VarDctEncoderFrame frame;
  if (!CheckStatus(ComputeCpuFrame(image, strategies, input, &frame),
                   "CPU coefficient oracle")) {
    return false;
  }
  if (frame.ac_group_count() != 1) {
    std::cerr << "Unexpected CPU oracle AC-group count\n";
    return false;
  }
  gjxl::VarDctAcGroupView group;
  if (!CheckStatus(frame.GetAcGroup(0, &group), "CPU oracle AC-group access")) {
    return false;
  }
  size_t transform_index = 0;
  size_t coefficient_offset = 0;
  bool coefficients_match = true;
  const gjxl::Status iteration_status = strategies.ForEachAnchor(
      [&](size_t block_x, size_t block_y, gjxl::AcStrategyType strategy) {
        const gjxl::AcStrategyInfo *info = gjxl::GetAcStrategyInfo(strategy);
        if (info == nullptr || transform_index >= snapshot.transforms.size()) {
          coefficients_match = false;
          return gjxl::Status::Ok();
        }
        const auto &actual = snapshot.transforms[transform_index];
        const size_t count = info->coefficient_count();
        if (actual.block_x != block_x || actual.block_y != block_y ||
            actual.strategy != strategy) {
          std::cerr << "Transform snapshot ordering mismatch\n";
          coefficients_match = false;
        }
        for (size_t channel = 0; channel < 3; ++channel) {
          const std::span<const int32_t> expected =
              group.coefficients[channel].subspan(coefficient_offset, count);
          if (actual.quantized_coefficients[channel].size() != count ||
              !std::equal(actual.quantized_coefficients[channel].begin(),
                          actual.quantized_coefficients[channel].end(),
                          expected.begin())) {
            std::cerr << "Quantized coefficient mismatch for transform "
                      << transform_index << ", channel " << channel << '\n';
            coefficients_match = false;
          }
        }
        coefficient_offset += count;
        ++transform_index;
        return gjxl::Status::Ok();
      });
  if (!iteration_status.ok() || !coefficients_match ||
      transform_index != snapshot.transforms.size() ||
      coefficient_offset != group.used_coefficient_count) {
    return false;
  }
  const size_t block_count = kBlockExtent.width * kBlockExtent.height;
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < kBlockExtent.height; ++y) {
      for (size_t x = 0; x < kBlockExtent.width; ++x) {
        const size_t index = y * kBlockExtent.width + x;
        if (!Near(snapshot.dc[channel][index],
                  frame.dc().plane[channel].Row(y)[x], 2.0e-4, 5.0e-5)) {
          std::cerr << "DC mismatch for channel " << channel << ", block "
                    << index << '/' << block_count << '\n';
          return false;
        }
      }
    }
  }

  std::array<std::vector<float>, 3> reconstructed;
  for (auto &plane : reconstructed)
    plane.resize(kPixelExtent.width * kPixelExtent.height);
  const gjxl::Image3FView output{{{
      {reconstructed[0].data(), kPixelExtent, kPixelExtent.width},
      {reconstructed[1].data(), kPixelExtent, kPixelExtent.width},
      {reconstructed[2].data(), kPixelExtent, kPixelExtent.width},
  }}};
  if (!CheckStatus(gjxl::ReconstructQuantizedCoefficients(frame, output),
                   "CPU reconstruction oracle")) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t index = 0; index < reconstructed[channel].size(); ++index) {
      const float actual = snapshot.reconstructed_opsin[channel][index];
      if (!std::isfinite(actual) ||
          !Near(actual, reconstructed[channel][index], 7.5e-4, 1.0e-4)) {
        std::cerr << "Reconstructed pixel mismatch for channel " << channel
                  << ", index " << index << ", actual " << actual
                  << ", expected " << reconstructed[channel][index] << '\n';
        return false;
      }
    }
  }
  return true;
}

bool CheckRoundTrip(gjxl::MetalBackendOptions backend_options,
                    const HostImage &image,
                    const gjxl::AcStrategyGrid &strategies,
                    bool check_repeats) {

  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!CheckStatus(
          gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, backend_options, &gpu),
          "Metal reconstruction backend creation")) {
    return false;
  }
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  InputStorage input;
  if (!InputStorage::Make(image, &input) ||
      !Prepare(*gpu, image, strategies, &prepared)) {
    return false;
  }
  const gjxl::GpuBackendStats before = gpu->stats();
  gjxl::metal_internal::MetalAqReconstructionSnapshotForTesting snapshot;
  const size_t repeats = check_repeats ? 3 : 1;
  for (size_t repeat = 0; repeat < repeats; ++repeat) {
    if (!CheckStatus(gjxl::metal_internal::RunMetalAqReconstructionForTesting(
                         *prepared, input.View(), &snapshot),
                     "Metal AQ reconstruction")) {
      return false;
    }
  }
  const gjxl::GpuBackendStats after = gpu->stats();
  if (after.successful_allocations != before.successful_allocations ||
      after.committed_submissions != before.committed_submissions + repeats) {
    std::cerr << "AQ reconstruction did not preserve prepared residency\n";
    return false;
  }
  return CompareForward(image, snapshot) &&
         CompareCoefficientOracle(image, strategies, input, snapshot);
}

bool CheckQuantizationProbe(const HostImage &image,
                            const gjxl::AcStrategyGrid &strategies) {

  std::unique_ptr<gjxl::GpuBackend> gpu;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "quantization-probe backend") ||
      !Prepare(*gpu, image, strategies, &prepared)) {
    return false;
  }
  constexpr std::array<gjxl::AcStrategyType, 7> strategies_to_probe = {
      gjxl::AcStrategyType::kDct8,     gjxl::AcStrategyType::kDct16x16,
      gjxl::AcStrategyType::kDct32x32, gjxl::AcStrategyType::kDct16x8,
      gjxl::AcStrategyType::kDct8x16,  gjxl::AcStrategyType::kDct32x16,
      gjxl::AcStrategyType::kDct16x32,
  };
  for (gjxl::AcStrategyType strategy : strategies_to_probe) {
    for (gjxl::XybChannel channel :
         {gjxl::XybChannel::kX, gjxl::XybChannel::kY, gjxl::XybChannel::kB}) {
      for (int32_t raw_quant : {1, 256}) {
        gjxl::QuantizationMatrixView matrix;
        if (!CheckStatus(
                gjxl::GetDefaultQuantizationMatrix(strategy, channel, &matrix),
                "probe quantization matrix")) {
          return false;
        }
        const float multiplier =
            channel == gjxl::XybChannel::kX
                ? 1.31f
                : (channel == gjxl::XybChannel::kB ? 0.77f : 1.0f);
        const gjxl::QuantizerParams params{8192, 48};
        gjxl::Quantizer quantizer;
        if (!CheckStatus(gjxl::Quantizer::Create(params, &quantizer),
                         "probe quantizer")) {
          return false;
      }
      std::vector<float> coefficients(matrix.dequant.size(), 0.0f);
      for (size_t y = 0; y < matrix.low_frequency_extent.height; ++y) {
        for (size_t x = 0; x < matrix.low_frequency_extent.width; ++x) {
          coefficients[y * matrix.coefficient_extent.width + x] =
              1000.0f + static_cast<float>(17 * y + x);
        }
      }
      const std::array<float, 9> scaled_values = {
            0.0f, 0.579f, 0.581f, -0.639f, -0.641f, 1.5f, 2.5f, -1.5f, -2.5f,
        };
        for (size_t i = 0; i < scaled_values.size(); ++i) {
          const size_t index = 5 * matrix.coefficient_extent.width + 5 + i;
          coefficients[index] =
              scaled_values[i] /
              (matrix.inverse_dequant[index] * quantizer.scale() *
               static_cast<float>(raw_quant) * multiplier);
        }
        std::vector<int32_t> expected_quantized(coefficients.size());
        std::vector<float> expected_dequantized(coefficients.size());
        if (!CheckStatus(
                gjxl::QuantizeAcBlock(
                    strategy, quantizer, raw_quant,
                    {.channel = channel, .matrix_multiplier = multiplier},
                    coefficients, expected_quantized),
                "CPU quantization probe") ||
            !CheckStatus(
                gjxl::DequantizeAcBlock(
                    strategy, quantizer, raw_quant,
                    {.channel = channel, .matrix_multiplier = multiplier},
                    expected_quantized, expected_dequantized),
                "CPU dequantization probe")) {
          return false;
        }
        std::vector<int32_t> actual_quantized;
        std::vector<float> actual_dequantized;
        const gjxl::metal_internal::MetalAqQuantizationProbeForTesting probe{
            .strategy = strategy,
            .channel = channel,
            .raw_quant = raw_quant,
            .quantizer = params,
            .matrix_multiplier = multiplier,
            .coefficients = coefficients,
        };
        if (!CheckStatus(
                gjxl::metal_internal::RunMetalAqQuantizationProbeForTesting(
                    *prepared, probe, &actual_quantized, &actual_dequantized),
                "Metal quantization probe") ||
            actual_quantized != expected_quantized) {
          std::cerr << "Direct quantization probe integer mismatch\n";
          return false;
        }
        for (size_t i = 0; i < actual_dequantized.size(); ++i) {
          if (!Near(actual_dequantized[i], expected_dequantized[i], 2.0e-6,
                    2.0e-6)) {
            std::cerr << "Direct dequantization probe mismatch at " << i
                      << '\n';
            return false;
          }
        }
      }
    }
  }

  std::vector<float> non_finite(64, 0.0f);
  non_finite[17] = std::numeric_limits<float>::infinity();
  std::vector<int32_t> quantized{123};
  std::vector<float> dequantized{456.0f};
  const gjxl::metal_internal::MetalAqQuantizationProbeForTesting bad{
      .strategy = gjxl::AcStrategyType::kDct8,
      .channel = gjxl::XybChannel::kY,
      .raw_quant = 256,
      .quantizer = {32768, 1},
      .matrix_multiplier = 1.0f,
      .coefficients = non_finite,
  };
  if (!ExpectCode(gjxl::metal_internal::RunMetalAqQuantizationProbeForTesting(
                      *prepared, bad, &quantized, &dequantized),
                  gjxl::StatusCode::kDeviceError,
                  "non-finite quantization probe") ||
      quantized != std::vector<int32_t>{123} ||
      dequantized != std::vector<float>{456.0f} ||
      !ExpectCode(gjxl::metal_internal::RunMetalAqQuantizationProbeForTesting(
                      *prepared, bad, &quantized, &dequantized),
                  gjxl::StatusCode::kFailedPrecondition,
                  "quantization probe reuse after numeric failure")) {
    return false;
  }

  if (!Prepare(*gpu, image, strategies, &prepared))
    return false;
  std::vector<float> overflowing(64, 0.0f);
  overflowing[23] = std::numeric_limits<float>::max();
  const gjxl::metal_internal::MetalAqQuantizationProbeForTesting overflow{
      .strategy = gjxl::AcStrategyType::kDct8,
      .channel = gjxl::XybChannel::kB,
      .raw_quant = 256,
      .quantizer = {32768, 1},
      .matrix_multiplier = 1.0f,
      .coefficients = overflowing,
  };
  return ExpectCode(gjxl::metal_internal::RunMetalAqQuantizationProbeForTesting(
                        *prepared, overflow, &quantized, &dequantized),
                    gjxl::StatusCode::kDeviceError,
                    "overflowing quantization probe") &&
         quantized == std::vector<int32_t>{123} &&
         dequantized == std::vector<float>{456.0f};
}

bool CheckResidentAdjustmentIntegration(
    const HostImage& image, const gjxl::AcStrategyGrid& strategies) {
  std::unique_ptr<gjxl::GpuBackend> gpu;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  InputStorage input;
  std::vector<uint8_t> sharpness(
      kBlockExtent.width * kBlockExtent.height);
  for (size_t index = 0; index < sharpness.size(); ++index) {
    sharpness[index] = static_cast<uint8_t>(index % 8);
  }
  if (!CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "resident-adjustment backend") ||
      !InputStorage::Make(image, &input) ||
      !Prepare(*gpu, image, strategies, &prepared,
               gjxl::AcCoefficientDecisionMode::kAdjustedSharedQuant) ||
      !CheckStatus(prepared->Reconfigure(
                       strategies,
                       {sharpness.data(), kBlockExtent,
                        kBlockExtent.width}),
                   "resident-adjustment reconfiguration")) {
    return false;
  }

  const gjxl::GpuBackendStats before = gpu->stats();
  gjxl::metal_internal::MetalAqReconstructionSnapshotForTesting snapshot;
  if (!CheckStatus(gjxl::metal_internal::RunMetalAqReconstructionForTesting(
                       *prepared, input.View(), &snapshot),
                   "resident adjustment integration")) {
    return false;
  }
  const gjxl::GpuBackendStats after = gpu->stats();
  if (after.successful_allocations != before.successful_allocations ||
      after.committed_submissions != before.committed_submissions + 1 ||
      snapshot.raw_quant.size() != input.raw_quant.size() ||
      snapshot.epf_inverse_sigma.size() != input.inverse_sigma.size()) {
    std::cerr << "Resident adjustment violated residency or readback shape\n";
    return false;
  }

  gjxl::Quantizer quantizer;
  if (!CheckStatus(gjxl::Quantizer::Create(input.params, &quantizer),
                   "resident-adjustment quantizer")) {
    return false;
  }
  const std::array<float, 3> matrix_multipliers = {
      gjxl::QuantizationMatrixMultiplier(Options().profile.x_qm_scale),
      1.0f,
      gjxl::QuantizationMatrixMultiplier(Options().profile.b_qm_scale),
  };
  bool changed_anchor = false;
  for (const auto& transform : snapshot.transforms) {
    const size_t raw_index =
        transform.block_y * kBlockExtent.width + transform.block_x;
    const std::array<std::span<const float>, 3> coefficients = {
        transform.forward_coefficients[0],
        transform.forward_coefficients[1],
        transform.forward_coefficients[2],
    };
    gjxl::AdjustedAcQuantization expected;
    std::vector<int32_t> expected_y(
        transform.forward_coefficients[1].size());
    if (!CheckStatus(gjxl::SelectAdjustedAcQuantization(
                         transform.strategy, quantizer,
                         input.raw_quant[raw_index], matrix_multipliers,
                         coefficients, &expected),
                     "resident-adjustment CPU decision") ||
        !CheckStatus(gjxl::QuantizeAdjustedYAcBlock(
                         transform.strategy, quantizer, expected,
                         coefficients[1], expected_y),
                     "resident-adjustment CPU Y quantization") ||
        snapshot.raw_quant[raw_index] != expected.raw_quant ||
        transform.quantized_coefficients[1] != expected_y) {
      std::cerr << "Resident adjustment differs at transform anchor "
                << transform.block_x << ',' << transform.block_y << '\n';
      return false;
    }
    changed_anchor |= expected.raw_quant != input.raw_quant[raw_index];
  }
  if (!changed_anchor) {
    std::cerr << "Resident adjustment integration fixture changed no anchor\n";
    return false;
  }
  for (size_t y = 0; y < kBlockExtent.height; ++y) {
    for (size_t x = 0; x < kBlockExtent.width; ++x) {
      gjxl::AcStrategyCell cell;
      if (!strategies.Get(x, y, &cell).ok() ||
          (!cell.is_anchor &&
           snapshot.raw_quant[y * kBlockExtent.width + x] !=
               input.raw_quant[y * kBlockExtent.width + x])) {
        std::cerr << "Resident adjustment changed a non-anchor raw quant\n";
        return false;
      }
    }
  }

  std::vector<float> expected_inverse_sigma(snapshot.raw_quant.size());
  if (!CheckStatus(gjxl::ComputeEpfInverseSigma(
                       strategies,
                       {snapshot.raw_quant.data(), kBlockExtent,
                        kBlockExtent.width},
                       quantizer,
                       {sharpness.data(), kBlockExtent, kBlockExtent.width},
                       Options().profile.epf_sigma,
                       {expected_inverse_sigma.data(), kBlockExtent,
                        kBlockExtent.width}),
                   "resident-adjustment EPF oracle")) {
    return false;
  }
  for (size_t index = 0; index < expected_inverse_sigma.size(); ++index) {
    if (!Near(snapshot.epf_inverse_sigma[index],
              expected_inverse_sigma[index], 2.0e-5, 2.0e-6)) {
      std::cerr << "Resident adjusted EPF sigma differs at block "
                << index << '\n';
      return false;
    }
  }
  return true;
}

enum class AdjustmentPattern {
  kFlat,
  kSparse,
  kActive,
  kHighFrequencyBorder,
  kThresholdTie,
  kQuantLimit,
};

void FillAdjustmentCoefficients(
    AdjustmentPattern pattern, size_t channel, int32_t raw_quant,
    float matrix_multiplier, const gjxl::Quantizer& quantizer,
    const gjxl::QuantizationMatrixView& matrix,
    std::vector<float>* coefficients) {

  std::fill(coefficients->begin(), coefficients->end(), 0.0f);
  const size_t width = matrix.coefficient_extent.width;
  const size_t height = matrix.coefficient_extent.height;
  switch (pattern) {
    case AdjustmentPattern::kFlat:
      return;
    case AdjustmentPattern::kSparse:
      for (size_t i = channel + 3; i < coefficients->size(); i += 97) {
        (*coefficients)[i] = (i & 1) == 0 ? 0.31f : -0.27f;
      }
      return;
    case AdjustmentPattern::kActive:
      for (size_t i = 0; i < coefficients->size(); ++i) {
        (*coefficients)[i] = static_cast<float>(static_cast<int32_t>(
          (i * 37 + channel * 13) % 101) - 50) * 0.019f;
      }
      return;
    case AdjustmentPattern::kHighFrequencyBorder:
    case AdjustmentPattern::kQuantLimit:
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
          if (x + 1 == width || y + 1 == height) {
            (*coefficients)[y * width + x] =
              ((x + y + channel) & 1) == 0 ? 0.83f : -0.71f;
          }
        }
      }
      return;
    case AdjustmentPattern::kThresholdTie: {
      const size_t index = width * height - 1;
      const size_t block_count = width * height / 64;
      const float threshold = std::max(
        0.54f,
        0.64f - std::clamp(0.003f * static_cast<float>(block_count),
                           0.0f, 0.08f));
      (*coefficients)[index] = threshold /
        (matrix.inverse_dequant[index] * quantizer.scale() *
         static_cast<float>(raw_quant) * matrix_multiplier);
      return;
    }
  }
}

bool CheckAdjustmentProbe(const HostImage& image,
                          const gjxl::AcStrategyGrid& strategies) {

  std::unique_ptr<gjxl::GpuBackend> gpu;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "adjustment-probe backend") ||
      !Prepare(*gpu, image, strategies, &prepared)) {
    return false;
  }
  const gjxl::QuantizerParams params{3541, 10};
  gjxl::Quantizer quantizer;
  if (!CheckStatus(gjxl::Quantizer::Create(params, &quantizer),
                   "adjustment-probe quantizer")) {
    return false;
  }
  const std::array<float, 3> matrix_multipliers = {
    gjxl::QuantizationMatrixMultiplier(1),
    1.0f,
    gjxl::QuantizationMatrixMultiplier(3),
  };
  constexpr std::array<gjxl::AcStrategyType, 7> kStrategies = {
    gjxl::AcStrategyType::kDct8,     gjxl::AcStrategyType::kDct16x16,
    gjxl::AcStrategyType::kDct32x32, gjxl::AcStrategyType::kDct16x8,
    gjxl::AcStrategyType::kDct8x16,  gjxl::AcStrategyType::kDct32x16,
    gjxl::AcStrategyType::kDct16x32,
  };
  constexpr std::array<AdjustmentPattern, 6> kPatterns = {
    AdjustmentPattern::kFlat,
    AdjustmentPattern::kSparse,
    AdjustmentPattern::kActive,
    AdjustmentPattern::kHighFrequencyBorder,
    AdjustmentPattern::kThresholdTie,
    AdjustmentPattern::kQuantLimit,
  };
  const gjxl::GpuBackendStats before = gpu->stats();
  size_t probe_count = 0;
  for (gjxl::AcStrategyType strategy : kStrategies) {
    std::array<gjxl::QuantizationMatrixView, 3> matrices;
    for (size_t channel = 0; channel < 3; ++channel) {
      constexpr std::array<gjxl::XybChannel, 3> kChannels = {
        gjxl::XybChannel::kX, gjxl::XybChannel::kY,
        gjxl::XybChannel::kB};
      if (!CheckStatus(gjxl::GetDefaultQuantizationMatrix(
            strategy, kChannels[channel], &matrices[channel]),
            "adjustment-probe matrix")) {
        return false;
      }
    }
    for (AdjustmentPattern pattern : kPatterns) {
      const int32_t raw_quant = pattern == AdjustmentPattern::kQuantLimit
        ? gjxl::kMaxRawQuant
        : 37;
      std::array<std::vector<float>, 3> coefficients;
      std::array<std::span<const float>, 3> coefficient_views;
      for (size_t channel = 0; channel < 3; ++channel) {
        coefficients[channel].resize(matrices[channel].dequant.size());
        FillAdjustmentCoefficients(
          pattern, channel, raw_quant, matrix_multipliers[channel], quantizer,
          matrices[channel], &coefficients[channel]);
        coefficient_views[channel] = coefficients[channel];
      }
      gjxl::AdjustedAcQuantization expected;
      std::vector<int32_t> expected_y(coefficients[1].size());
      if (!CheckStatus(gjxl::SelectAdjustedAcQuantization(
            strategy, quantizer, raw_quant, matrix_multipliers,
            coefficient_views, &expected), "CPU adjustment probe") ||
          !CheckStatus(gjxl::QuantizeAdjustedYAcBlock(
            strategy, quantizer, expected, coefficients[1], expected_y),
            "CPU adjusted-Y probe")) {
        return false;
      }
      gjxl::metal_internal::MetalAqAdjustmentResultForTesting actual;
      if (!CheckStatus(
            gjxl::metal_internal::RunMetalAqAdjustmentProbeForTesting(
              *prepared,
              {
                .strategy = strategy,
                .initial_raw_quant = raw_quant,
                .quantizer = params,
                .matrix_multipliers = matrix_multipliers,
                .coefficients = coefficient_views,
              },
              &actual),
            "Metal adjustment probe") ||
          actual.decision.raw_quant != expected.raw_quant ||
          actual.quantized_y != expected_y) {
        std::cerr << "Adjusted Metal decision differs for strategy "
                  << static_cast<int>(strategy) << ", pattern "
                  << static_cast<int>(pattern) << ": raw="
                  << actual.decision.raw_quant << ", expected="
                  << expected.raw_quant << '\n';
        return false;
      }
      for (size_t quadrant = 0; quadrant < 4; ++quadrant) {
        if (!Near(actual.decision.y_thresholds[quadrant],
                  expected.y_thresholds[quadrant], 2.0e-6, 2.0e-6)) {
          std::cerr << "Adjusted Y threshold differs for quadrant "
                    << quadrant << '\n';
          return false;
        }
      }
      ++probe_count;
    }
  }
  const gjxl::GpuBackendStats after = gpu->stats();
  if (after.successful_allocations != before.successful_allocations ||
      after.committed_submissions !=
        before.committed_submissions + probe_count) {
    std::cerr << "Adjustment probe violated prepared residency\n";
    return false;
  }

  std::array<std::vector<float>, 3> invalid_coefficients;
  std::array<std::span<const float>, 3> invalid_views;
  for (size_t channel = 0; channel < 3; ++channel) {
    invalid_coefficients[channel].assign(64, 0.0f);
    invalid_views[channel] = invalid_coefficients[channel];
  }
  invalid_coefficients[2][17] = std::numeric_limits<float>::infinity();
  invalid_views[2] = invalid_coefficients[2];
  gjxl::metal_internal::MetalAqAdjustmentResultForTesting sentinel{
    .decision = {
      .raw_quant = 123,
      .y_thresholds = {1.0f, 2.0f, 3.0f, 4.0f},
    },
    .quantized_y = {5, 6, 7},
  };
  const auto original = sentinel;
  return ExpectCode(
           gjxl::metal_internal::RunMetalAqAdjustmentProbeForTesting(
             *prepared,
             {
               .strategy = gjxl::AcStrategyType::kDct8,
               .initial_raw_quant = 37,
               .quantizer = params,
               .matrix_multipliers = matrix_multipliers,
               .coefficients = invalid_views,
             },
             &sentinel),
           gjxl::StatusCode::kDeviceError,
           "non-finite adjustment probe") &&
    sentinel.decision == original.decision &&
    sentinel.quantized_y == original.quantized_y;
}

gjxl::metal_internal::MetalAqReconstructionSnapshotForTesting
PoisonedSnapshot() {
  gjxl::metal_internal::MetalAqReconstructionSnapshotForTesting snapshot;
  snapshot.block_extent = {77, 88};
  snapshot.pixel_extent = {99, 111};
  snapshot.transforms.resize(1);
  snapshot.transforms[0].block_x = 123;
  snapshot.dc[0] = {456.0f};
  snapshot.reconstructed_opsin[2] = {789.0f};
  return snapshot;
}

bool IsPoisoned(
    const gjxl::metal_internal::MetalAqReconstructionSnapshotForTesting
        &snapshot) {

  return snapshot.block_extent == gjxl::Extent2D{77, 88} &&
         snapshot.pixel_extent == gjxl::Extent2D{99, 111} &&
         snapshot.transforms.size() == 1 &&
         snapshot.transforms[0].block_x == 123 &&
         snapshot.dc[0] == std::vector<float>{456.0f} &&
         snapshot.reconstructed_opsin[2] == std::vector<float>{789.0f};
}

bool CheckReconstructionFailure(gjxl::MetalBackendOptions backend_options,
                                gjxl::StatusCode expected,
                                const HostImage &image,
                                const gjxl::AcStrategyGrid &strategies) {

  std::unique_ptr<gjxl::GpuBackend> gpu;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  InputStorage input;
  if (!CheckStatus(
          gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
          "failure reconstruction backend") ||
      !InputStorage::Make(image, &input) ||
      !Prepare(*gpu, image, strategies, &prepared) ||
      !CheckStatus(gjxl::ArmNextMetalSubmissionFailureForTest(
                       *gpu, backend_options.test_fail_submission,
                       backend_options.test_fail_completion),
                   "AQ reconstruction failure injection")) {
    return false;
  }
  auto snapshot = PoisonedSnapshot();
  return ExpectCode(gjxl::metal_internal::RunMetalAqReconstructionForTesting(
                        *prepared, input.View(), &snapshot),
                    expected, "injected AQ reconstruction failure") &&
         IsPoisoned(snapshot) &&
         ExpectCode(gjxl::metal_internal::RunMetalAqReconstructionForTesting(
                        *prepared, input.View(), &snapshot),
                    gjxl::StatusCode::kFailedPrecondition,
                    "AQ reconstruction reuse after failure") &&
         IsPoisoned(snapshot);
}

bool CheckReconstructionValidationAndReadback(
    const HostImage &image, const gjxl::AcStrategyGrid &strategies) {

  std::unique_ptr<gjxl::GpuBackend> gpu;
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  InputStorage input;
  if (!CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "validation reconstruction backend") ||
      !InputStorage::Make(image, &input) ||
      !Prepare(*gpu, image, strategies, &prepared)) {
    return false;
  }
  auto snapshot = PoisonedSnapshot();
  InputStorage invalid = input;
  invalid.raw_quant[0] = 0;
  const uint64_t submissions = gpu->stats().committed_submissions;
  if (!ExpectCode(gjxl::metal_internal::RunMetalAqReconstructionForTesting(
                      *prepared, invalid.View(), &snapshot),
                  gjxl::StatusCode::kInvalidArgument,
                  "invalid AQ reconstruction input") ||
      !IsPoisoned(snapshot) ||
      gpu->stats().committed_submissions != submissions ||
      !CheckStatus(
          gjxl::metal_internal::FailNextMetalAqReadbackForTesting(*prepared),
          "AQ reconstruction readback failure injection") ||
      !ExpectCode(gjxl::metal_internal::RunMetalAqReconstructionForTesting(
                      *prepared, input.View(), &snapshot),
                  gjxl::StatusCode::kDeviceError,
                  "injected AQ reconstruction readback failure") ||
      !IsPoisoned(snapshot)) {
    return false;
  }
  return ExpectCode(gjxl::metal_internal::RunMetalAqReconstructionForTesting(
                        *prepared, input.View(), &snapshot),
                    gjxl::StatusCode::kFailedPrecondition,
                    "AQ reconstruction reuse after readback failure");
}

bool CheckConcurrentReconstruction(
    const HostImage &image, const gjxl::AcStrategyGrid &strategies) {

  std::unique_ptr<gjxl::GpuBackend> gpu;
  std::unique_ptr<gjxl::PreparedAqEvaluation> first;
  std::unique_ptr<gjxl::PreparedAqEvaluation> second;
  InputStorage input;
  if (!CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "concurrent reconstruction backend") ||
      !InputStorage::Make(image, &input) ||
      !Prepare(*gpu, image, strategies, &first) ||
      !Prepare(*gpu, image, strategies, &second)) {
    return false;
  }
  gjxl::metal_internal::MetalAqReconstructionSnapshotForTesting first_snapshot;
  gjxl::metal_internal::MetalAqReconstructionSnapshotForTesting second_snapshot;
  bool first_ok = false;
  bool second_ok = false;
  const gjxl::GpuBackendStats before = gpu->stats();
  std::thread first_thread([&] {
    first_ok = gjxl::metal_internal::RunMetalAqReconstructionForTesting(
                   *first, input.View(), &first_snapshot)
                   .ok();
  });
  std::thread second_thread([&] {
    second_ok = gjxl::metal_internal::RunMetalAqReconstructionForTesting(
                    *second, input.View(), &second_snapshot)
                    .ok();
  });
  first_thread.join();
  second_thread.join();
  const gjxl::GpuBackendStats after = gpu->stats();
  if (!first_ok || !second_ok ||
      after.successful_allocations != before.successful_allocations ||
      after.committed_submissions != before.committed_submissions + 2) {
    std::cerr << "Independent AQ reconstruction concurrency failed\n";
    return false;
  }
  return CompareCoefficientOracle(image, strategies, input, first_snapshot) &&
         CompareCoefficientOracle(image, strategies, input, second_snapshot);
}

bool CheckResidentInitialCfl(const HostImage& image,
                             const gjxl::AcStrategyGrid& strategies) {
  std::unique_ptr<gjxl::GpuBackend> gpu;
  InputStorage input;
  const std::vector<uint8_t> sharpness(
      kBlockExtent.width * kBlockExtent.height, 4);
  gjxl::ColorCorrelationMap expected;
  if (!CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "resident initial-CfL backend") ||
      !InputStorage::Make(image, &input) ||
      !CheckStatus(
          gjxl::chroma_from_luma_internal::
              ComputeInitialColorCorrelationMapFast(image.View(), &expected),
          "resident initial-CfL CPU oracle")) {
    return false;
  }

  const gjxl::AqEvaluationPreparation preparation{
      .original_linear_rgb = image.View(),
      .coding_opsin = image.View(),
      .strategies = &strategies,
      .epf_sharpness = {
          sharpness.data(), kBlockExtent, kBlockExtent.width},
      .options = Options(),
      .frame_only = true,
      .frame_only_resident_initial_cfl = true,
  };
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!CheckStatus(gjxl::PrepareAqEvaluation(
                       *gpu, preparation, &prepared),
                   "resident initial-CfL preparation")) {
    return false;
  }
  gjxl::AqEvaluationInput resident_input = input.View();
  resident_input.y_to_x = {};
  resident_input.y_to_b = {};
  gjxl::VarDctEncoderFrame frame;
  const gjxl::GpuBackendStats before = gpu->stats();
  if (!CheckStatus(prepared->EncodeFrame(resident_input, &frame),
                   "resident initial-CfL frame encoding")) {
    return false;
  }
  const gjxl::GpuBackendStats after = gpu->stats();
  if (after.successful_allocations != before.successful_allocations ||
      after.committed_submissions != before.committed_submissions + 1) {
    std::cerr << "Resident initial CfL escaped the prepared submission\n";
    return false;
  }
  const auto expected_x = expected.y_to_x_map();
  const auto expected_b = expected.y_to_b_map();
  const auto actual_x = frame.color_correlation().y_to_x_map();
  const auto actual_b = frame.color_correlation().y_to_b_map();
  for (size_t y = 0; y < expected.tile_extent().height; ++y) {
    for (size_t x = 0; x < expected.tile_extent().width; ++x) {
      if (actual_x.Row(y)[x] != expected_x.Row(y)[x] ||
          actual_b.Row(y)[x] != expected_b.Row(y)[x]) {
        std::cerr << "Resident initial CfL differs at tile " << x << ',' << y
                  << ": X " << static_cast<int>(actual_x.Row(y)[x]) << '/'
                  << static_cast<int>(expected_x.Row(y)[x]) << ", B "
                  << static_cast<int>(actual_b.Row(y)[x]) << '/'
                  << static_cast<int>(expected_b.Row(y)[x]) << '\n';
        return false;
      }
    }
  }
  return true;
}

double MaximumError(std::span<const float> left,
                    std::span<const float> right) {
  if (left.size() != right.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double error = 0.0;
  for (size_t index = 0; index < left.size(); ++index) {
    error = std::max(
        error, std::abs(static_cast<double>(left[index]) - right[index]));
  }
  return error;
}

bool CheckResidentInitialQuantization(const HostImage& image) {
  std::unique_ptr<gjxl::GpuBackend> gpu;
  gjxl::AcStrategyGrid strategies;
  if (!CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
                   "resident initial-quant backend") ||
      !MakeUniformStrategies(gjxl::AcStrategyType::kDct8, &strategies)) {
    return false;
  }
  const std::vector<uint8_t> sharpness(
      kBlockExtent.width * kBlockExtent.height, 4);
  const gjxl::AqEvaluationPreparation preparation{
      .original_linear_rgb = image.View(),
      .coding_opsin = image.View(),
      .strategies = &strategies,
      .epf_sharpness = {
          sharpness.data(), kBlockExtent, kBlockExtent.width},
      .options = Options(),
      .frame_only = true,
      .frame_only_resident_initial_quant = true,
      .frame_only_resident_quantizer = true,
  };
  std::unique_ptr<gjxl::PreparedAqEvaluation> prepared;
  if (!CheckStatus(gjxl::PrepareAqEvaluation(*gpu, preparation, &prepared),
                   "resident initial-quant preparation")) {
    return false;
  }

  const size_t block_count = kBlockExtent.width * kBlockExtent.height;
  const size_t pixel_count = kPixelExtent.width * kPixelExtent.height;
  for (const gjxl::InitialQuantizationOptions options :
       {gjxl::InitialQuantizationOptions{1.0f, 1.0f},
        gjxl::InitialQuantizationOptions{2.4f, 0.87f}}) {
    std::vector<float> expected_quant(block_count);
    std::vector<float> expected_strategy(block_count);
    std::vector<float> expected_pixel(pixel_count);
    std::vector<float> actual_quant(block_count, -1.0f);
    std::vector<float> actual_strategy(block_count, -1.0f);
    std::vector<float> actual_pixel(pixel_count, -1.0f);
    std::vector<int32_t> expected_raw(block_count);
    float quant_dc = 0.0f;
    gjxl::Quantizer expected_quantizer;
    if (!CheckStatus(
            gjxl::ComputeInitialQuantField(
                image.View(), options,
                {
                    .quant_field = {expected_quant.data(), kBlockExtent,
                                    kBlockExtent.width},
                    .strategy_mask = {expected_strategy.data(), kBlockExtent,
                                      kBlockExtent.width},
                    .pixel_mask = {expected_pixel.data(), kPixelExtent,
                                   kPixelExtent.width},
                }),
            "resident initial-quant CPU oracle")) {
      return false;
    }
    if (!CheckStatus(
            gjxl::ComputeInitialQuantDc(options.butteraugli_target, &quant_dc),
            "resident initial-quant DC oracle") ||
        !CheckStatus(
            gjxl::CreateQuantizerFromField(
                quant_dc,
                {expected_quant.data(), kBlockExtent, kBlockExtent.width},
                {expected_raw.data(), kBlockExtent, kBlockExtent.width},
                &expected_quantizer),
            "resident initial quantizer CPU oracle")) {
      return false;
    }
    const gjxl::GpuBackendStats before = gpu->stats();
    gjxl::QuantizerParams actual_quantizer;
    if (!CheckStatus(
            prepared->ComputeInitialQuantization(
                options,
                {
                    .quant_field = {actual_quant.data(), kBlockExtent,
                                    kBlockExtent.width},
                    .strategy_mask = {actual_strategy.data(), kBlockExtent,
                                      kBlockExtent.width},
                    .pixel_mask = {actual_pixel.data(), kPixelExtent,
                                   kPixelExtent.width},
                },
                &actual_quantizer, quant_dc),
            "resident initial-quant Metal execution")) {
      return false;
    }
    const gjxl::GpuBackendStats after = gpu->stats();
    const double quant_error = MaximumError(expected_quant, actual_quant);
    const double strategy_error =
        MaximumError(expected_strategy, actual_strategy);
    const double pixel_error = MaximumError(expected_pixel, actual_pixel);
    if (after.successful_allocations != before.successful_allocations ||
        after.committed_submissions != before.committed_submissions + 1 ||
        quant_error > 2.0e-6 || strategy_error > 2.0e-6 ||
        pixel_error > 2.0e-5 ||
        actual_quantizer.global_scale !=
            expected_quantizer.params().global_scale ||
        actual_quantizer.quant_dc != expected_quantizer.params().quant_dc) {
      std::cerr << "Resident initial quantization differs: quant="
                << quant_error << " strategy=" << strategy_error
                << " pixel=" << pixel_error << '\n';
      return false;
    }
    gjxl::ColorCorrelationMap color;
    gjxl::VarDctEncoderFrame frame;
    if (!CheckStatus(
            gjxl::ComputeInitialColorCorrelationMap(image.View(), &color),
            "resident initial-quant color map") ||
        !CheckStatus(
            prepared->EncodeFrame(
                {
                    .quantizer = actual_quantizer,
                    .y_to_x = color.y_to_x_map(),
                    .y_to_b = color.y_to_b_map(),
                },
                &frame),
            "resident initial-quant frame readback") ||
        !std::equal(expected_raw.begin(), expected_raw.end(),
                    frame.raw_quant_field().data) ||
        gpu->stats().successful_allocations != before.successful_allocations ||
        gpu->stats().committed_submissions !=
            before.committed_submissions + 2) {
      std::cerr << "Resident raw quantization differs from CPU oracle\n";
      return false;
    }
  }
  std::vector<float> failed_quant(block_count, -7.0f);
  std::vector<float> failed_strategy(block_count, -8.0f);
  std::vector<float> failed_pixel(pixel_count, -9.0f);
  const auto failure_output = gjxl::InitialQuantFieldOutput{
      .quant_field = {failed_quant.data(), kBlockExtent, kBlockExtent.width},
      .strategy_mask = {
          failed_strategy.data(), kBlockExtent, kBlockExtent.width},
      .pixel_mask = {failed_pixel.data(), kPixelExtent, kPixelExtent.width},
  };
  const gjxl::GpuBackendStats before_failure = gpu->stats();
  float failure_quant_dc = 0.0f;
  if (!CheckStatus(
          gjxl::ComputeInitialQuantDc(1.0f, &failure_quant_dc),
          "resident initial-quant failure DC") ||
      !CheckStatus(
          gjxl::metal_internal::FailNextMetalAqNumericForTesting(*prepared),
          "resident initial-quant numeric failure injection") ||
      !ExpectCode(prepared->ComputeInitialQuantization(
                      {}, failure_output, nullptr, failure_quant_dc),
                  gjxl::StatusCode::kDeviceError,
                  "resident initial-quant numeric failure") ||
      !std::ranges::all_of(failed_quant,
                           [](float value) { return value == -7.0f; }) ||
      !std::ranges::all_of(failed_strategy,
                           [](float value) { return value == -8.0f; }) ||
      !std::ranges::all_of(failed_pixel,
                           [](float value) { return value == -9.0f; }) ||
      gpu->stats().successful_allocations !=
          before_failure.successful_allocations ||
      gpu->stats().committed_submissions !=
          before_failure.committed_submissions + 1 ||
      !ExpectCode(prepared->ComputeInitialQuantization(
                      {}, failure_output, nullptr, failure_quant_dc),
                  gjxl::StatusCode::kFailedPrecondition,
                  "resident initial-quant reuse after numeric failure")) {
    return false;
  }
  return true;
}

} // namespace

int main() {
  gjxl::AcStrategyGrid strategies;
  if (!MakeMixedStrategies(&strategies))
    return EXIT_FAILURE;
  const HostImage structured = HostImage::Structured(false);
  const HostImage flat = HostImage::Structured(true);
  constexpr std::array<gjxl::AcStrategyType, 7> all_strategies = {
      gjxl::AcStrategyType::kDct8,     gjxl::AcStrategyType::kDct16x16,
      gjxl::AcStrategyType::kDct32x32, gjxl::AcStrategyType::kDct16x8,
      gjxl::AcStrategyType::kDct8x16,  gjxl::AcStrategyType::kDct32x16,
      gjxl::AcStrategyType::kDct16x32,
  };
  for (gjxl::AcStrategyType strategy : all_strategies) {
    gjxl::AcStrategyGrid uniform;
    if (!MakeUniformStrategies(strategy, &uniform) ||
        !CheckRoundTrip({}, structured, uniform, false) ||
        !CheckRoundTrip(SimdOptions(), structured, uniform, false)) {
      return EXIT_FAILURE;
    }
  }
  if (!CheckRoundTrip({}, structured, strategies, true) ||
      !CheckRoundTrip(SimdOptions(), structured, strategies, true) ||
      !CheckRoundTrip({}, flat, strategies, false) ||
      !CheckRoundTrip(SimdOptions(), flat, strategies, false) ||
      !CheckQuantizationProbe(structured, strategies) ||
      !CheckAdjustmentProbe(structured, strategies) ||
      !CheckResidentAdjustmentIntegration(structured, strategies) ||
      !CheckReconstructionFailure({.test_fail_submission = true},
                                  gjxl::StatusCode::kSubmissionFailed,
                                  structured, strategies) ||
      !CheckReconstructionFailure({.test_fail_completion = true},
                                  gjxl::StatusCode::kDeviceError, structured,
                                  strategies) ||
      !CheckReconstructionValidationAndReadback(structured, strategies) ||
      !CheckResidentInitialCfl(structured, strategies) ||
      !CheckResidentInitialCfl(flat, strategies) ||
      !CheckResidentInitialQuantization(structured) ||
      !CheckResidentInitialQuantization(flat) ||
      !CheckConcurrentReconstruction(structured, strategies)) {
    return EXIT_FAILURE;
  }
  std::cout << "Metal AQ Milestone 3 reconstruction tests passed\n";
  return EXIT_SUCCESS;
}
