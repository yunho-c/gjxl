// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/codestream.h"

#include <cmath>
#include <cstddef>

#include "codec/vardct_frame.h"
#include "core/quantizer.h"

namespace gjxl {
namespace {

constexpr size_t kMaximumJxlDimension = 0x3FFFFFFFu;

template <typename Enum>
bool IsKnown(Enum value, Enum first, Enum second) {
  return value == first || value == second;
}

bool ValidGaborish(const SimpleVarDctCodestreamProfile& profile) {
  for (float multiplier : profile.gaborish_inverse_multipliers) {
    if (!std::isfinite(multiplier)) {
      return false;
    }
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    const float weight1 = profile.loop_filter.gaborish_options.weight1[channel];
    const float weight2 = profile.loop_filter.gaborish_options.weight2[channel];
    const float divisor = 1.0f + 4.0f * (weight1 + weight2);
    if (!std::isfinite(weight1) || !std::isfinite(weight2) ||
        !std::isfinite(divisor) || std::abs(divisor) < 1.0e-8f) {
      return false;
    }
  }
  return true;
}

bool ValidEpf(const SimpleVarDctCodestreamProfile& profile) {
  if (!std::isfinite(profile.epf_sigma.quant_multiplier) ||
      profile.epf_sigma.quant_multiplier <= 0.0f ||
      profile.loop_filter.epf_options.iterations > 3 ||
      !std::isfinite(profile.loop_filter.epf_options.pass0_sigma_scale) ||
      profile.loop_filter.epf_options.pass0_sigma_scale <= 0.0f ||
      !std::isfinite(profile.loop_filter.epf_options.pass2_sigma_scale) ||
      profile.loop_filter.epf_options.pass2_sigma_scale <= 0.0f ||
      !std::isfinite(profile.loop_filter.epf_options.border_sad_multiplier) ||
      profile.loop_filter.epf_options.border_sad_multiplier <= 0.0f) {
    return false;
  }
  for (float value : profile.epf_sigma.sharpness_lut) {
    if (!std::isfinite(value) || value < 0.0f) {
      return false;
    }
  }
  for (float scale : profile.loop_filter.epf_options.channel_scale) {
    if (!std::isfinite(scale) || scale < 0.0f) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool SimpleVarDctCodestreamProfile::valid() const noexcept {
  const bool valid_upsampling =
    upsampling == 1 || upsampling == 2 ||
    upsampling == 4 || upsampling == 8;
  return std::isfinite(intensity_target) && intensity_target > 0.0f &&
    (color_channel_count == 1 || color_channel_count == 3) &&
    pass_count != 0 && pass_count <= 11 && valid_upsampling &&
    IsKnown(
      quantization_matrix_mode,
      QuantizationMatrixMode::kDefault,
      QuantizationMatrixMode::kCustom) &&
    x_qm_scale <= 7 && b_qm_scale <= 7 &&
    extra_dc_precision <= 3 &&
    IsKnown(dc_cfl_mode, DcCflMode::kDefault, DcCflMode::kCustom) &&
    IsKnown(
      coefficient_order_mode,
      CoefficientOrderMode::kDefault,
      CoefficientOrderMode::kCustom) &&
    IsKnown(
      modular_transform_mode,
      ModularTransformMode::kNone,
      ModularTransformMode::kCustom) &&
    ValidGaborish(*this) && ValidEpf(*this);
}

float QuantizationMatrixMultiplier(uint8_t scale) noexcept {
  return static_cast<float>(std::pow(
    1.25,
    static_cast<int>(scale) - 2));
}

Status ValidateSimpleCodestreamFrame(const VarDctEncoderFrame& frame) {
  if (!frame.valid()) {
    return Status::InvalidArgument("VarDCT frame is invalid");
  }

  const SimpleVarDctCodestreamProfile& profile = frame.profile();
  if (!profile.source_is_linear_srgb ||
      !profile.source_is_floating_point ||
      profile.intensity_target != 255.0f) {
    return Status::InvalidArgument(
      "Source metadata is outside the initial codestream profile");
  }
  if (profile.color_channel_count != 3 ||
      profile.extra_channel_count != 0 ||
      profile.pass_count != 1 ||
      profile.upsampling != 1) {
    return Status::InvalidArgument(
      "Channel, pass, or upsampling state is unsupported");
  }
  if (profile.quantization_matrix_mode !=
        QuantizationMatrixMode::kDefault) {
    return Status::InvalidArgument(
      "Quantization-matrix state is outside the initial profile");
  }

  const QuantizerParams quantizer = frame.quantizer().params();
  if (quantizer.global_scale == 0 ||
      quantizer.global_scale > kMaxEncoderGlobalScale ||
      quantizer.quant_dc == 0 || quantizer.quant_dc > kMaxQuantDc) {
    return Status::InvalidArgument(
      "Quantizer parameters cannot be encoded");
  }
  if (profile.extra_dc_precision != 0 ||
      profile.dc_cfl_mode != DcCflMode::kDefault ||
      profile.adaptive_dc_smoothing) {
    return Status::InvalidArgument(
      "DC coding state is outside the initial profile");
  }
  if (profile.coefficient_order_mode != CoefficientOrderMode::kDefault) {
    return Status::InvalidArgument(
      "Custom coefficient orders are unsupported");
  }

  const SimpleVarDctCodestreamProfile defaults;
  if (!profile.loop_filter.gaborish ||
      profile.gaborish_inverse_multipliers !=
        defaults.gaborish_inverse_multipliers ||
      profile.loop_filter.gaborish_options !=
        defaults.loop_filter.gaborish_options) {
    return Status::InvalidArgument(
      "Gaborish state is outside the initial profile");
  }
  if (profile.epf_sigma != defaults.epf_sigma ||
      profile.loop_filter.epf_options !=
        defaults.loop_filter.epf_options) {
    return Status::InvalidArgument(
      "EPF state is outside the initial profile");
  }
  if (profile.modular_transform_mode != ModularTransformMode::kNone ||
      profile.lz77) {
    return Status::InvalidArgument(
      "Modular transforms or LZ77 are unsupported");
  }

  const Extent2D extent = frame.geometry().frame();
  if (extent.width > kMaximumJxlDimension ||
      extent.height > kMaximumJxlDimension) {
    return Status::InvalidArgument(
      "Frame dimensions cannot be represented by the JPEG XL size header");
  }
  return Status::Ok();
}

}  // namespace gjxl
