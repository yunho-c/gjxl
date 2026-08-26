// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates the retained initial-codestream profile and serializer gate.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include "codec/chroma_from_luma.h"
#include "codec/reconstruction.h"
#include "codec/codestream.h"

namespace {

gjxl::Status MakeFrame(
  gjxl::SimpleVarDctCodestreamProfile profile,
  gjxl::VarDctEncoderFrame* frame,
  gjxl::QuantizerParams quantizer_params = {3541, 10}) {

  constexpr gjxl::Extent2D kPixels{8, 8};
  constexpr gjxl::Extent2D kBlocks{1, 1};
  std::array<std::vector<float>, 3> storage;
  for (size_t channel = 0; channel < storage.size(); ++channel) {
    storage[channel].resize(64);
    for (size_t index = 0; index < storage[channel].size(); ++index) {
      storage[channel][index] =
        0.01f * static_cast<float>((channel + 1) * (index + 3));
    }
  }
  const gjxl::ConstImage3FView opsin{{
    gjxl::ConstPlaneF32View{
      storage[0].data(), kPixels, kPixels.width},
    gjxl::ConstPlaneF32View{
      storage[1].data(), kPixels, kPixels.width},
    gjxl::ConstPlaneF32View{
      storage[2].data(), kPixels, kPixels.width},
  }};

  gjxl::FrameGeometry geometry;
  gjxl::AcStrategyGrid strategies;
  gjxl::Quantizer quantizer;
  gjxl::ColorCorrelationMap color_correlation;
  if (gjxl::Status status = gjxl::FrameGeometry::Create(kPixels, &geometry);
      !status.ok()) {
    return status;
  }
  if (gjxl::Status status = gjxl::AcStrategyGrid::Create(kBlocks, &strategies);
      !status.ok()) {
    return status;
  }
  strategies.fill_dct8();
  if (gjxl::Status status =
        gjxl::Quantizer::Create(quantizer_params, &quantizer);
      !status.ok()) {
    return status;
  }
  if (gjxl::Status status = gjxl::ComputeInitialColorCorrelationMap(
        opsin, &color_correlation);
      !status.ok()) {
    return status;
  }

  const int32_t raw_quant = 29;
  const uint8_t sharpness = 4;
  return gjxl::ComputeQuantizedCoefficients(
    opsin,
    {
      .geometry = geometry,
      .strategies = &strategies,
      .raw_quant_field = {&raw_quant, kBlocks, 1},
      .quantizer = &quantizer,
      .color_correlation = &color_correlation,
      .epf_sharpness = {&sharpness, kBlocks, 1},
    },
    profile,
    frame);
}

template <typename Mutator>
bool RejectsMutation(std::string_view name, Mutator mutate) {
  gjxl::SimpleVarDctCodestreamProfile profile;
  mutate(&profile);
  gjxl::VarDctEncoderFrame frame;
  const gjxl::Status build_status = MakeFrame(profile, &frame);
  const gjxl::Status validation_status =
    gjxl::ValidateSimpleCodestreamFrame(frame);
  if (!build_status.ok() || !frame.valid() ||
      validation_status.ok() ||
      validation_status.code() != gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "Profile mutation was not rejected: " << name
              << ", build=" << build_status.message()
              << ", validate=" << validation_status.message() << '\n';
    return false;
  }
  return true;
}

bool CheckDefaultAndQuantizerBoundary() {
  gjxl::VarDctEncoderFrame frame;
  gjxl::Status status = MakeFrame({}, &frame);
  if (!status.ok() || !frame.valid() ||
      frame.profile() != gjxl::SimpleVarDctCodestreamProfile{} ||
      !gjxl::ValidateSimpleCodestreamFrame(frame).ok()) {
    std::cerr << "Default profile did not pass validation: "
              << status.message() << '\n';
    return false;
  }

  status = MakeFrame(
    {}, &frame,
    {gjxl::kMaxEncoderGlobalScale, gjxl::kMaxQuantDc});
  if (!status.ok() || !gjxl::ValidateSimpleCodestreamFrame(frame).ok()) {
    std::cerr << "Encodable quantizer boundary was rejected: "
              << status.message() << '\n';
    return false;
  }
  return true;
}

bool CheckScaleMultipliers() {
  const float expected_min = static_cast<float>(std::pow(1.25, -2));
  const float expected_default = static_cast<float>(std::pow(1.25, 0));
  const float expected_max = static_cast<float>(std::pow(1.25, 5));
  if (gjxl::QuantizationMatrixMultiplier(0) != expected_min ||
      gjxl::QuantizationMatrixMultiplier(2) != expected_default ||
      gjxl::QuantizationMatrixMultiplier(7) != expected_max) {
    std::cerr << "Three-bit matrix scales derive incorrect multipliers\n";
    return false;
  }
  return true;
}

bool CheckUnsupportedProfileDimensions() {
  return
    RejectsMutation("non-linear sRGB", [](auto* p) {
      p->source_is_linear_srgb = false;
    }) &&
    RejectsMutation("integer samples", [](auto* p) {
      p->source_is_floating_point = false;
    }) &&
    RejectsMutation("intensity target", [](auto* p) {
      p->intensity_target = 254.0f;
    }) &&
    RejectsMutation("color channels", [](auto* p) {
      p->color_channel_count = 1;
    }) &&
    RejectsMutation("extra channels", [](auto* p) {
      p->extra_channel_count = 1;
    }) &&
    RejectsMutation("multiple passes", [](auto* p) {
      p->pass_count = 2;
    }) &&
    RejectsMutation("upsampling", [](auto* p) {
      p->upsampling = 2;
    }) &&
    RejectsMutation("custom matrices", [](auto* p) {
      p->quantization_matrix_mode = gjxl::QuantizationMatrixMode::kCustom;
    }) &&
    RejectsMutation("minimum X matrix scale", [](auto* p) {
      p->x_qm_scale = 0;
    }) &&
    RejectsMutation("maximum B matrix scale", [](auto* p) {
      p->b_qm_scale = 7;
    }) &&
    RejectsMutation("DC precision", [](auto* p) {
      p->extra_dc_precision = 1;
    }) &&
    RejectsMutation("DC CfL", [](auto* p) {
      p->dc_cfl_mode = gjxl::DcCflMode::kCustom;
    }) &&
    RejectsMutation("adaptive DC smoothing", [](auto* p) {
      p->adaptive_dc_smoothing = true;
    }) &&
    RejectsMutation("coefficient orders", [](auto* p) {
      p->coefficient_order_mode = gjxl::CoefficientOrderMode::kCustom;
    }) &&
    RejectsMutation("Gaborish disabled", [](auto* p) {
      p->loop_filter.gaborish = false;
    }) &&
    RejectsMutation("Gaborish inverse", [](auto* p) {
      p->gaborish_inverse_multipliers[0] = 1.01f;
    }) &&
    RejectsMutation("Gaborish weights", [](auto* p) {
      p->loop_filter.gaborish_options.weight1[1] += 0.01f;
    }) &&
    RejectsMutation("EPF sigma multiplier", [](auto* p) {
      p->epf_sigma.quant_multiplier = 0.5f;
    }) &&
    RejectsMutation("EPF sharpness table", [](auto* p) {
      p->epf_sigma.sharpness_lut[4] = 0.75f;
    }) &&
    RejectsMutation("EPF iterations", [](auto* p) {
      p->loop_filter.epf_options.iterations = 1;
    }) &&
    RejectsMutation("EPF channel scale", [](auto* p) {
      p->loop_filter.epf_options.channel_scale[2] = 4.0f;
    }) &&
    RejectsMutation("EPF pass 0", [](auto* p) {
      p->loop_filter.epf_options.pass0_sigma_scale = 1.0f;
    }) &&
    RejectsMutation("EPF pass 2", [](auto* p) {
      p->loop_filter.epf_options.pass2_sigma_scale = 6.0f;
    }) &&
    RejectsMutation("EPF border", [](auto* p) {
      p->loop_filter.epf_options.border_sad_multiplier = 0.75f;
    }) &&
    RejectsMutation("modular transforms", [](auto* p) {
      p->modular_transform_mode = gjxl::ModularTransformMode::kCustom;
    }) &&
    RejectsMutation("LZ77", [](auto* p) {
      p->lz77 = true;
    });
}

}  // namespace

int main() {
  gjxl::VarDctEncoderFrame empty;
  if (gjxl::ValidateSimpleCodestreamFrame(empty).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      !CheckDefaultAndQuantizerBoundary() ||
      !CheckScaleMultipliers() ||
      !CheckUnsupportedProfileDimensions()) {
    return EXIT_FAILURE;
  }
  std::cout << "All simple codestream-profile tests passed.\n";
  return EXIT_SUCCESS;
}
