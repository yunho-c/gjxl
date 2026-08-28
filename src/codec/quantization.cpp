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

Status QuantizeAcBlockWithThresholds(
  AcStrategyType strategy,
  const Quantizer& quantizer,
  int32_t raw_quant,
  AcQuantizationOptions options,
  const std::array<float, 4>& thresholds,
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
  if (!std::ranges::all_of(thresholds, [](float threshold) {
        return std::isfinite(threshold) && threshold >= 0.0f;
      })) {
    return Status::InvalidArgument(
      "AC quantization thresholds must be finite and non-negative");
  }

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

Status AdjustQuantForChannel(
  const Quantizer& quantizer,
  size_t channel,
  float matrix_multiplier,
  AcStrategyType strategy,
  const QuantizationMatrixView& matrix,
  std::span<const float> coefficients,
  std::array<float, 4>* thresholds,
  int32_t* raw_quant) {

  const size_t xsize = matrix.coefficient_extent.width / kJxlBlockDimension;
  const size_t ysize = matrix.coefficient_extent.height / kJxlBlockDimension;
  const float qac = quantizer.scale() * static_cast<float>(*raw_quant);
  if (xsize > 1 || ysize > 1) {
    const float reduction = std::clamp(
      0.003f * static_cast<float>(xsize * ysize), 0.0f, 0.08f);
    for (float& threshold : *thresholds) {
      threshold = std::max(0.54f, threshold - reduction);
    }
  }

  float highest_frequency_border_sum = 0.0f;
  float error_sum = 0.0f;
  float value_sum = 0.0f;
  std::array<float, 4> high_frequency_nonzeros{};
  std::array<float, 4> high_frequency_max_error{};
  for (size_t y = 0; y < matrix.coefficient_extent.height; ++y) {
    for (size_t x = 0; x < matrix.coefficient_extent.width; ++x) {
      const size_t index = y * matrix.coefficient_extent.width + x;
      if (x < xsize && y < ysize) {
        continue;
      }
      const size_t quadrant =
        static_cast<size_t>(y >= matrix.coefficient_extent.height / 2) * 2 +
        static_cast<size_t>(x >= matrix.coefficient_extent.width / 2);
      const float value = coefficients[index] *
        (matrix.inverse_dequant[index] * qac * matrix_multiplier);
      if (!std::isfinite(value)) {
        return Status::InvalidArgument(
          "Adjusted AC coefficient is not finite");
      }
      const float quantized = std::abs(value) < (*thresholds)[quadrant]
        ? 0.0f
        : std::rint(value);
      const float error = std::abs(value - quantized);
      error_sum += error;
      value_sum += std::abs(quantized);
      if (channel == 1 && quantized == 0.0f) {
        high_frequency_max_error[quadrant] = std::max(
          high_frequency_max_error[quadrant], error);
      }
      if (quantized != 0.0f) {
        high_frequency_nonzeros[quadrant] += std::abs(quantized);
        const bool in_corner = y >= 7 * ysize && x >= 7 * xsize;
        const bool on_border =
          y + 1 == matrix.coefficient_extent.height ||
          x + 1 == matrix.coefficient_extent.width;
        const bool in_larger_corner = x >= 4 * xsize && y >= 4 * ysize;
        if (in_corner || (on_border && in_larger_corner)) {
          highest_frequency_border_sum += std::abs(value);
        }
      }
    }
  }

  if (channel == 1 &&
      value_sum * 8.0f < static_cast<float>(xsize * ysize)) {
    constexpr std::array<double, 4> kLimit = {
      0.46, 0.46, 0.46, 0.46};
    constexpr std::array<double, 4> kMultiplier = {
      0.9999, 0.9999, 0.9999, 0.9999};
    const int32_t original_quant = *raw_quant;
    int32_t new_quant = *raw_quant;
    for (size_t quadrant = 1; quadrant < 4; ++quadrant) {
      if (high_frequency_nonzeros[quadrant] == 0.0f &&
          high_frequency_max_error[quadrant] > kLimit[quadrant]) {
        new_quant = original_quant + 1;
        break;
      }
    }
    *raw_quant = new_quant;
    if (high_frequency_nonzeros[3] == 0.0f &&
        high_frequency_max_error[3] > kLimit[3]) {
      (*thresholds)[3] = static_cast<float>(
        kMultiplier[3] * high_frequency_max_error[3] * new_quant /
        original_quant);
    } else if ((high_frequency_nonzeros[1] == 0.0f &&
                high_frequency_max_error[1] > kLimit[1]) ||
               (high_frequency_nonzeros[2] == 0.0f &&
                high_frequency_max_error[2] > kLimit[2])) {
      (*thresholds)[1] = static_cast<float>(
        kMultiplier[1] * std::max(
          high_frequency_max_error[1], high_frequency_max_error[2]) *
        new_quant / original_quant);
      (*thresholds)[2] = (*thresholds)[1];
    } else if (high_frequency_nonzeros[0] == 0.0f &&
               high_frequency_max_error[0] > kLimit[0]) {
      (*thresholds)[0] = static_cast<float>(
        kMultiplier[0] * high_frequency_max_error[0] * new_quant /
        original_quant);
    }
  }

  const float all_nonzeros =
    high_frequency_nonzeros[0] + high_frequency_nonzeros[1] +
    high_frequency_nonzeros[2] + high_frequency_nonzeros[3] + 1.0f;
  constexpr std::array<float, 3> kBorderMultiplier = {70.0f, 30.0f, 60.0f};
  if (kBorderMultiplier[channel] * highest_frequency_border_sum >=
      all_nonzeros) {
    *raw_quant += kBorderMultiplier[channel] *
      highest_frequency_border_sum / all_nonzeros;
    if (*raw_quant >= kMaxRawQuant) {
      *raw_quant = kMaxRawQuant - 1;
    }
  }

  if (strategy == AcStrategyType::kDct8 &&
      high_frequency_nonzeros[0] + high_frequency_nonzeros[1] +
        high_frequency_nonzeros[2] + high_frequency_nonzeros[3] < 11.0f) {
    ++*raw_quant;
    if (*raw_quant >= kMaxRawQuant) {
      *raw_quant = kMaxRawQuant - 1;
    }
  }

  constexpr double kFirstMultiplier[4][3] = {
    {0.22080615753848404, 0.45797479824262011, 0.29859235095977965},
    {0.70109486510286834, 0.16185281305512639, 0.14387691730035473},
    {0.114985964456218638, 0.44656840441027695, 0.10587658215149048},
    {0.46849665264409396, 0.41239077937781954, 0.088667407767185444},
  };
  constexpr double kSecondMultiplier[4][3] = {
    {0.27450281941822197, 1.1255766549984996, 0.98950459134128388},
    {0.4652168675598285, 0.40945807983455818, 0.36581899811751367},
    {0.28034972424715715, 0.9182653201929738, 1.5581531543057416},
    {0.26873118114033728, 0.68863712390392484, 1.2082185408666786},
  };
  constexpr double kQuantNormalizer = 2.2942708343284721;
  error_sum *= kQuantNormalizer;
  value_sum *= kQuantNormalizer;
  if (static_cast<size_t>(strategy) >=
      static_cast<size_t>(AcStrategyType::kDct16x16)) {
    size_t strategy_class = 3;
    if (strategy == AcStrategyType::kDct32x16 ||
        strategy == AcStrategyType::kDct16x32) {
      strategy_class = 1;
    } else if (strategy == AcStrategyType::kDct16x16) {
      strategy_class = 0;
    } else if (strategy == AcStrategyType::kDct32x32) {
      strategy_class = 2;
    }
    const double threshold =
      kFirstMultiplier[strategy_class][channel] *
        xsize * ysize * kJxlBlockDimension * kJxlBlockDimension +
      kSecondMultiplier[strategy_class][channel] * value_sum;
    int32_t step = static_cast<int32_t>(error_sum / threshold);
    step = std::clamp(step, 0, 2);
    if (error_sum > threshold) {
      *raw_quant += step;
      if (*raw_quant >= kMaxRawQuant) {
        *raw_quant = kMaxRawQuant - 1;
      }
    }
  }

  const int32_t divisor = static_cast<int32_t>(xsize * ysize);
  const float minimum_nonzeros = *std::min_element(
    high_frequency_nonzeros.begin(), high_frequency_nonzeros.end());
  int32_t activity = 15;
  if (minimum_nonzeros < 15.0f * divisor) {
    activity =
      (static_cast<int32_t>(minimum_nonzeros) + divisor / 2) / divisor;
  }
  int32_t adjusted_quant = *raw_quant - activity;
  if (channel == 1) {
    for (size_t quadrant = 1; quadrant < 4; ++quadrant) {
      (*thresholds)[quadrant] += 0.01f * activity;
    }
  }
  const int32_t original_quant_limit = std::max(4, *raw_quant / 2);
  if (adjusted_quant < original_quant_limit) {
    adjusted_quant = original_quant_limit;
  }
  *raw_quant = adjusted_quant;
  return Status::Ok();
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

  size_t value_count = 0;
  if (!quant_field.extent.try_area(&value_count)) {
    return Status::InvalidArgument(
      "Quantization field dimensions are too large");
  }

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

  const AcStrategyInfo* strategy_info = GetAcStrategyInfo(strategy);
  if (strategy_info == nullptr) {
    return Status::InvalidArgument("Unknown AC strategy");
  }
  const size_t covered_block_count =
    strategy_info->covered_blocks.width *
    strategy_info->covered_blocks.height;
  const std::array<float, 4> thresholds =
    QuantizationThresholds(options.channel, covered_block_count);
  return QuantizeAcBlockWithThresholds(
    strategy, quantizer, raw_quant, options, thresholds,
    coefficients, quantized);
}

Status SelectAdjustedAcQuantization(
  AcStrategyType strategy,
  const Quantizer& quantizer,
  int32_t initial_raw_quant,
  const std::array<float, 3>& matrix_multipliers,
  const std::array<std::span<const float>, 3>& coefficients,
  AdjustedAcQuantization* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Adjusted AC quantization output is null");
  }
  constexpr std::array<XybChannel, 3> kAdjustmentChannels = {
    XybChannel::kX, XybChannel::kY, XybChannel::kB};
  std::array<QuantizationMatrixView, 3> matrices;
  for (size_t channel = 0; channel < 3; ++channel) {
    Status status = ValidateAcOperation(
      strategy,
      quantizer,
      initial_raw_quant,
      {
        .channel = kAdjustmentChannels[channel],
        .matrix_multiplier = matrix_multipliers[channel],
      },
      coefficients[channel].size(),
      coefficients[channel].size(),
      &matrices[channel]);
    if (!status.ok()) {
      return status;
    }
    if (!std::ranges::all_of(coefficients[channel], [](float coefficient) {
          return std::isfinite(coefficient);
        })) {
      return Status::InvalidArgument(
        "Adjusted AC coefficients must be finite");
    }
  }

  AdjustedAcQuantization result;
  result.y_thresholds = {0.58f, 0.64f, 0.64f, 0.64f};
  for (size_t channel : {size_t{1}, size_t{0}, size_t{2}}) {
    std::array<float, 4> thresholds = {0.58f, 0.64f, 0.64f, 0.64f};
    int32_t candidate_quant = initial_raw_quant;
    Status status = AdjustQuantForChannel(
      quantizer,
      channel,
      matrix_multipliers[channel],
      strategy,
      matrices[channel],
      coefficients[channel],
      &thresholds,
      &candidate_quant);
    if (!status.ok()) {
      return status;
    }
    if (channel == 1) {
      result.y_thresholds = thresholds;
    }
    result.raw_quant = std::max(result.raw_quant, candidate_quant);
  }
  if (result.raw_quant < 1 || result.raw_quant > kMaxRawQuant) {
    return Status::Internal(
      "Adjusted AC quantization is outside the encoder range");
  }
  *out = result;
  return Status::Ok();
}

Status QuantizeAdjustedYAcBlock(
  AcStrategyType strategy,
  const Quantizer& quantizer,
  const AdjustedAcQuantization& decision,
  std::span<const float> coefficients,
  std::span<int32_t> quantized) {

  return QuantizeAcBlockWithThresholds(
    strategy,
    quantizer,
    decision.raw_quant,
    {.channel = XybChannel::kY},
    decision.y_thresholds,
    coefficients,
    quantized);
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
