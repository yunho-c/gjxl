// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/quantization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "codec/quantization_tables_generated.h"

namespace gjxl {
namespace {

constexpr float kQuantFieldTarget = 5.0f;
constexpr float kGlobalScaleNumerator = 4096.0f;

constexpr std::array<float, 4> kDefaultQuantBias = {
  1.0f - 0.05465007330715401f,
  1.0f - 0.07005449891748593f,
  1.0f - 0.049935103337343655f,
  0.145f,
};

bool ValidChannel(XybChannel channel) {
  return static_cast<uint8_t>(channel) <=
    static_cast<uint8_t>(XybChannel::kB);
}

size_t ChannelIndex(XybChannel channel) {
  return static_cast<size_t>(channel);
}

int32_t ClampRawQuant(float value) {
  return static_cast<int32_t>(
    std::clamp(value, 1.0f, static_cast<float>(kMaxRawQuant)));
}

Status ValidateQuantDc(float quant_dc) {
  if (!std::isfinite(quant_dc) ||
      quant_dc <= 0.0f ||
      quant_dc > static_cast<float>(kMaxQuantDc)) {

    return Status::InvalidArgument(
      "DC quantization must be finite and in (0, 65536]");
  }

  return Status::Ok();
}

Status ComputeQuantizerParams(
  float quant_dc,
  float quant_median,
  float quant_median_absolute_deviation,
  QuantizerParams* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Quantizer parameter output is null");
  }

  Status status = ValidateQuantDc(quant_dc);
  if (!status.ok()) {
    return status;
  }

  if (!std::isfinite(quant_median) ||
      !std::isfinite(quant_median_absolute_deviation) ||
      quant_median <= 0.0f ||
      quant_median_absolute_deviation < 0.0f) {

    return Status::InvalidArgument(
      "Quantization statistics must be finite and non-negative");
  }

  float scale =
    static_cast<float>(kQuantGlobalScaleDenominator) *
    (quant_median - quant_median_absolute_deviation) /
    kQuantFieldTarget;

  scale = std::clamp(
    scale,
    1.0f,
    static_cast<float>(kMaxEncoderGlobalScale));

  int32_t global_scale = static_cast<int32_t>(scale);
  const int32_t scaled_quant_dc = static_cast<int32_t>(
    static_cast<double>(quant_dc * kGlobalScaleNumerator) * 1.6);

  if (global_scale > scaled_quant_dc) {
    global_scale = std::max<int32_t>(1, scaled_quant_dc);
  }

  const float inverse_global_scale =
    static_cast<float>(kQuantGlobalScaleDenominator) /
    static_cast<float>(global_scale);

  const float quant_dc_value = std::min(
    static_cast<float>(kMaxQuantDc),
    quant_dc * inverse_global_scale + 0.5f);

  out->global_scale = static_cast<uint32_t>(global_scale);
  out->quant_dc = static_cast<uint32_t>(quant_dc_value);
  return Status::Ok();
}

template <size_t N>
QuantizationMatrixView MakeMatrixView(
  const std::array<float, N>& dequant,
  const std::array<float, N>& inverse_dequant,
  size_t channel,
  Extent2D coefficient_extent,
  Extent2D low_frequency_extent) {

  const size_t coefficient_count =
    coefficient_extent.width * coefficient_extent.height;
  const size_t offset = channel * coefficient_count;

  return {
    .dequant = std::span<const float>(dequant).subspan(
      offset,
      coefficient_count),
    .inverse_dequant = std::span<const float>(inverse_dequant).subspan(
      offset,
      coefficient_count),
    .coefficient_extent = coefficient_extent,
    .low_frequency_extent = low_frequency_extent,
  };
}

Status ValidateAcOperation(
  AcStrategyType strategy,
  const Quantizer& quantizer,
  int32_t raw_quant,
  AcQuantizationOptions options,
  size_t input_size,
  size_t output_size,
  QuantizationMatrixView* matrix) {

  if (!quantizer.valid()) {
    return Status::InvalidArgument(
      "Quantizer is not initialized");
  }

  if (raw_quant < 1 || raw_quant > kMaxRawQuant) {
    return Status::InvalidArgument(
      "Raw AC quantization must be in [1, 256]");
  }

  if (!std::isfinite(options.matrix_multiplier) ||
      options.matrix_multiplier <= 0.0f) {

    return Status::InvalidArgument(
      "Quantization matrix multiplier must be finite and positive");
  }

  Status status = GetDefaultQuantizationMatrix(
    strategy,
    options.channel,
    matrix);

  if (!status.ok()) {
    return status;
  }

  const size_t coefficient_count = matrix->dequant.size();
  if (input_size != coefficient_count || output_size != coefficient_count) {
    return Status::InvalidArgument(
      "AC coefficient span has the wrong size for its strategy");
  }

  return Status::Ok();
}

std::array<float, 4> QuantizationThresholds(
  XybChannel channel,
  size_t covered_block_count) {

  std::array<float, 4> thresholds = channel == XybChannel::kY
    ? std::array<float, 4>{0.58f, 0.64f, 0.64f, 0.64f}
    : std::array<float, 4>{0.58f, 0.62f, 0.62f, 0.62f};

  if (channel != XybChannel::kY && covered_block_count >= 4) {
    for (float& threshold : thresholds) {
      threshold = std::max(
        0.5f,
        threshold - 0.00744f *
          static_cast<float>(covered_block_count));
    }
  }

  return thresholds;
}

float AdjustQuantizationBias(int32_t quantized, XybChannel channel) {
  const float value = static_cast<float>(quantized);
  const float absolute_value = std::abs(value);

  if (absolute_value < 0.125f) {
    return 0.0f;
  }

  if (absolute_value < 1.125f) {
    return std::copysign(
      kDefaultQuantBias[ChannelIndex(channel)],
      value);
  }

  return value - kDefaultQuantBias[3] / value;
}

}  // namespace

Status CreateUniformQuantizer(
  float quant_dc,
  float quant_ac,
  PlaneI32View raw_quant_field,
  Quantizer* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Quantizer output is null");
  }

  if (!raw_quant_field.valid()) {
    return Status::InvalidArgument(
      "Raw quantization field is invalid");
  }

  if (!std::isfinite(quant_ac) ||
      quant_ac <= 0.0f ||
      quant_ac > static_cast<float>(kMaxQuantDc)) {

    return Status::InvalidArgument(
      "AC quantization must be finite and in (0, 65536]");
  }

  QuantizerParams params;
  Status status = ComputeQuantizerParams(
    quant_dc,
    quant_ac,
    0.0f,
    &params);

  if (!status.ok()) {
    return status;
  }

  Quantizer result;
  status = Quantizer::Create(params, &result);
  if (!status.ok()) {
    return status;
  }

  const int32_t raw_quant = ClampRawQuant(
    quant_ac * result.inverse_global_scale() + 0.5f);

  for (size_t y = 0; y < raw_quant_field.extent.height; ++y) {
    std::fill_n(
      raw_quant_field.Row(y),
      raw_quant_field.extent.width,
      raw_quant);
  }

  *out = result;
  return Status::Ok();
}

Status CreateQuantizerFromField(
  float quant_dc,
  ConstPlaneF32View quant_field,
  PlaneI32View raw_quant_field,
  Quantizer* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Quantizer output is null");
  }

  if (!quant_field.valid() ||
      !raw_quant_field.valid() ||
      quant_field.extent != raw_quant_field.extent) {

    return Status::InvalidArgument(
      "Quantization fields are invalid or have different extents");
  }

  Status status = ValidateQuantDc(quant_dc);
  if (!status.ok()) {
    return status;
  }

  const size_t value_count =
    quant_field.extent.width * quant_field.extent.height;
  std::vector<float> values;
  values.reserve(value_count);

  for (size_t y = 0; y < quant_field.extent.height; ++y) {
    const float* row = quant_field.Row(y);
    for (size_t x = 0; x < quant_field.extent.width; ++x) {
      if (!std::isfinite(row[x]) || row[x] <= 0.0f) {
        return Status::InvalidArgument(
          "Quantization field values must be finite and positive");
      }
      values.push_back(row[x]);
    }
  }

  const size_t median_index = values.size() / 2;
  std::nth_element(
    values.begin(),
    values.begin() + median_index,
    values.end());
  const float median = values[median_index];

  std::vector<float> deviations(values.size());
  std::transform(
    values.begin(),
    values.end(),
    deviations.begin(),
    [median](float value) {
      return std::abs(value - median);
    });

  std::nth_element(
    deviations.begin(),
    deviations.begin() + median_index,
    deviations.end());
  const float median_absolute_deviation = deviations[median_index];

  QuantizerParams params;
  status = ComputeQuantizerParams(
    quant_dc,
    median,
    median_absolute_deviation,
    &params);

  if (!status.ok()) {
    return status;
  }

  Quantizer result;
  status = Quantizer::Create(params, &result);
  if (!status.ok()) {
    return status;
  }

  for (size_t y = 0; y < quant_field.extent.height; ++y) {
    const float* source = quant_field.Row(y);
    int32_t* destination = raw_quant_field.Row(y);

    for (size_t x = 0; x < quant_field.extent.width; ++x) {
      destination[x] = ClampRawQuant(
        source[x] * result.inverse_global_scale() + 0.5f);
    }
  }

  *out = result;
  return Status::Ok();
}

Status GetDefaultQuantizationMatrix(
  AcStrategyType strategy,
  XybChannel channel,
  QuantizationMatrixView* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Quantization matrix output is null");
  }

  if (!ValidChannel(channel)) {
    return Status::InvalidArgument(
      "Unknown XYB channel");
  }

  const size_t channel_index = ChannelIndex(channel);
  switch (strategy) {
    case AcStrategyType::kDct8:
      *out = MakeMatrixView(
        quantization_internal::kDct8Dequant,
        quantization_internal::kDct8InverseDequant,
        channel_index,
        {8, 8},
        {1, 1});
      return Status::Ok();

    case AcStrategyType::kDct16x16:
      *out = MakeMatrixView(
        quantization_internal::kDct16Dequant,
        quantization_internal::kDct16InverseDequant,
        channel_index,
        {16, 16},
        {2, 2});
      return Status::Ok();

    case AcStrategyType::kDct32x32:
      *out = MakeMatrixView(
        quantization_internal::kDct32Dequant,
        quantization_internal::kDct32InverseDequant,
        channel_index,
        {32, 32},
        {4, 4});
      return Status::Ok();

    case AcStrategyType::kDct16x8:
    case AcStrategyType::kDct8x16:
      *out = MakeMatrixView(
        quantization_internal::kDct8x16Dequant,
        quantization_internal::kDct8x16InverseDequant,
        channel_index,
        {16, 8},
        {2, 1});
      return Status::Ok();

    case AcStrategyType::kDct32x16:
    case AcStrategyType::kDct16x32:
      *out = MakeMatrixView(
        quantization_internal::kDct16x32Dequant,
        quantization_internal::kDct16x32InverseDequant,
        channel_index,
        {32, 16},
        {4, 2});
      return Status::Ok();

    default:
      break;
  }

  if (GetAcStrategyInfo(strategy) == nullptr) {
    return Status::InvalidArgument(
      "Unknown AC strategy");
  }

  return Status::Unavailable(
    "Default quantization matrix is unavailable for this strategy");
}

Status QuantizeAcBlock(
  AcStrategyType strategy,
  const Quantizer& quantizer,
  int32_t raw_quant,
  AcQuantizationOptions options,
  std::span<const float> coefficients,
  std::span<int32_t> quantized) {

  QuantizationMatrixView matrix;
  Status status = ValidateAcOperation(
    strategy,
    quantizer,
    raw_quant,
    options,
    coefficients.size(),
    quantized.size(),
    &matrix);

  if (!status.ok()) {
    return status;
  }

  const AcStrategyInfo* strategy_info = GetAcStrategyInfo(strategy);
  if (strategy_info == nullptr) {
    return Status::Internal(
      "Validated AC strategy disappeared");
  }

  const size_t covered_block_count =
    strategy_info->covered_blocks.width *
    strategy_info->covered_blocks.height;
  const std::array<float, 4> thresholds =
    QuantizationThresholds(options.channel, covered_block_count);
  const float quantization_scale =
    quantizer.scale() *
    static_cast<float>(raw_quant) *
    options.matrix_multiplier;
  const size_t width = matrix.coefficient_extent.width;
  const size_t height = matrix.coefficient_extent.height;

  for (size_t index = 0; index < coefficients.size(); ++index) {
    if (!std::isfinite(coefficients[index])) {
      return Status::InvalidArgument(
        "AC coefficients must be finite");
    }

    const size_t x = index % width;
    const size_t y = index / width;
    const size_t threshold_index =
      static_cast<size_t>(y >= height / 2) * 2 +
      static_cast<size_t>(x >= width / 2);
    const float value =
      matrix.inverse_dequant[index] *
      quantization_scale *
      coefficients[index];

    if (!std::isfinite(value)) {
      return Status::InvalidArgument(
        "Scaled AC coefficient is not finite");
    }

    if (std::abs(value) < thresholds[threshold_index]) {
      quantized[index] = 0;
      continue;
    }

    const double rounded = std::nearbyint(static_cast<double>(value));
    if (rounded < static_cast<double>(std::numeric_limits<int32_t>::min()) ||
        rounded > static_cast<double>(std::numeric_limits<int32_t>::max())) {

      return Status::InvalidArgument(
        "Quantized AC coefficient exceeds int32 range");
    }

    quantized[index] = static_cast<int32_t>(rounded);
  }

  return Status::Ok();
}

Status DequantizeAcBlock(
  AcStrategyType strategy,
  const Quantizer& quantizer,
  int32_t raw_quant,
  AcQuantizationOptions options,
  std::span<const int32_t> quantized,
  std::span<float> coefficients) {

  QuantizationMatrixView matrix;
  Status status = ValidateAcOperation(
    strategy,
    quantizer,
    raw_quant,
    options,
    quantized.size(),
    coefficients.size(),
    &matrix);

  if (!status.ok()) {
    return status;
  }

  const float dequantization_scale =
    quantizer.inverse_quant_ac(raw_quant) /
    options.matrix_multiplier;

  for (size_t index = 0; index < quantized.size(); ++index) {
    coefficients[index] =
      AdjustQuantizationBias(quantized[index], options.channel) *
      (matrix.dequant[index] * dequantization_scale);

    if (!std::isfinite(coefficients[index])) {
      return Status::InvalidArgument(
        "Dequantized AC coefficient is not finite");
    }
  }

  return Status::Ok();
}

}  // namespace gjxl
