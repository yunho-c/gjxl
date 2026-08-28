// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <metal_stdlib>

using namespace metal;

struct AqAdjustedQuantization {
  int raw_quant;
  float4 y_thresholds;
};

static uint2 aq_quant_table_offsets(uint strategy) {
  switch (strategy) {
    case 0:  return uint2(0, 192);       // DCT8
    case 4:  return uint2(384, 1152);    // DCT16x16
    case 5:  return uint2(1920, 4992);   // DCT32x32
    case 6:  return uint2(8064, 8448);   // DCT16x8
    case 7:  return uint2(8064, 8448);   // DCT8x16
    case 10: return uint2(8832, 10368);  // DCT32x16
    case 11: return uint2(8832, 10368);  // DCT16x32
    default: return uint2(0, 192);
  }
}

[[maybe_unused]] static float aq_quantization_threshold(
  uint channel,
  uint covered_block_count,
  uint coefficient_index,
  uint coefficient_width,
  uint coefficient_height) {

  const uint x = coefficient_index % coefficient_width;
  const uint y = coefficient_index / coefficient_width;
  const uint quadrant = uint(y >= coefficient_height / 2u) * 2u +
    uint(x >= coefficient_width / 2u);
  float threshold;
  if (channel == 1u) {
    threshold = quadrant == 0u ? 0.58f : 0.64f;
  } else {
    threshold = quadrant == 0u ? 0.58f : 0.62f;
    if (covered_block_count >= 4u) {
      threshold = max(
        0.5f,
        threshold - 0.00744f * float(covered_block_count));
    }
  }
  return threshold;
}

[[maybe_unused]] static int aq_quantize_coefficient(
  float coefficient,
  float inverse_dequant,
  uint global_scale,
  int raw_quant,
  float matrix_multiplier,
  float threshold,
  device atomic_uint* error) {

  const float quantization_scale =
    (float(global_scale) * (1.0f / 65536.0f)) *
    float(raw_quant) * matrix_multiplier;
  const float value = inverse_dequant * quantization_scale * coefficient;
  if (!isfinite(coefficient) || !isfinite(value)) {
    atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    return 0;
  }
  if (abs(value) < threshold) {
    return 0;
  }
  const float rounded = rint(value);
  if (!isfinite(rounded) || rounded < -2147483648.0f ||
      rounded >= 2147483648.0f) {
    atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
    return 0;
  }
  return int(rounded);
}

[[maybe_unused]] static float aq_adjust_quantization_bias(
  int quantized,
  uint channel) {
  constexpr float kBias[3] = {
    1.0f - 0.05465007330715401f,
    1.0f - 0.07005449891748593f,
    1.0f - 0.049935103337343655f,
  };
  const float value = float(quantized);
  const float absolute_value = abs(value);
  if (absolute_value < 0.125f) {
    return 0.0f;
  }
  if (absolute_value < 1.125f) {
    return copysign(kBias[channel], value);
  }
  return value - 0.145f / value;
}

[[maybe_unused]] static float aq_dequantize_coefficient(
  int quantized,
  float dequant,
  uint global_scale,
  int raw_quant,
  float matrix_multiplier,
  uint channel,
  device atomic_uint* error) {

  const float dequantization_scale =
    (65536.0f / float(global_scale)) /
    (float(raw_quant) * matrix_multiplier);
  const float coefficient = aq_adjust_quantization_bias(quantized, channel) *
    (dequant * dequantization_scale);
  if (!isfinite(coefficient)) {
    atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
    return 0.0f;
  }
  return coefficient;
}

[[maybe_unused]] static int aq_adjust_quant_for_channel(
  device const float* coefficients,
  device const float* quant_tables,
  uint coefficient_count,
  uint channel_stride,
  uint coefficient_width,
  uint coefficient_height,
  uint strategy,
  uint channel,
  uint global_scale,
  int initial_raw_quant,
  float matrix_multiplier,
  thread float4& thresholds,
  device atomic_uint* error) {

  const uint xsize = coefficient_width / 8u;
  const uint ysize = coefficient_height / 8u;
  const uint2 table_offsets = aq_quant_table_offsets(strategy);
  const float qac =
    (float(global_scale) * (1.0f / 65536.0f)) * float(initial_raw_quant);
  if (xsize > 1u || ysize > 1u) {
    const float reduction = clamp(
      0.003f * float(xsize * ysize), 0.0f, 0.08f);
    for (uint quadrant = 0u; quadrant < 4u; ++quadrant) {
      thresholds[quadrant] = max(0.54f, thresholds[quadrant] - reduction);
    }
  }

  float highest_frequency_border_sum = 0.0f;
  float error_sum = 0.0f;
  float value_sum = 0.0f;
  float4 high_frequency_nonzeros = 0.0f;
  float4 high_frequency_max_error = 0.0f;
  for (uint y = 0u; y < coefficient_height; ++y) {
    for (uint x = 0u; x < coefficient_width; ++x) {
      if (x < xsize && y < ysize) continue;
      const uint index = y * coefficient_width + x;
      const uint quadrant = uint(y >= coefficient_height / 2u) * 2u +
        uint(x >= coefficient_width / 2u);
      const uint table = channel * coefficient_count + index;
      const float coefficient = coefficients[channel * channel_stride + index];
      const float value = coefficient *
        (quant_tables[table_offsets.y + table] * qac * matrix_multiplier);
      if (!isfinite(coefficient) || !isfinite(value)) {
        atomic_fetch_or_explicit(error, 32u, memory_order_relaxed);
        continue;
      }
      const float quantized = abs(value) < thresholds[quadrant]
        ? 0.0f
        : rint(value);
      const float quantization_error = abs(value - quantized);
      error_sum += quantization_error;
      value_sum += abs(quantized);
      if (channel == 1u && quantized == 0.0f) {
        high_frequency_max_error[quadrant] = max(
          high_frequency_max_error[quadrant], quantization_error);
      }
      if (quantized != 0.0f) {
        high_frequency_nonzeros[quadrant] += abs(quantized);
        const bool in_corner = y >= 7u * ysize && x >= 7u * xsize;
        const bool on_border =
          y + 1u == coefficient_height || x + 1u == coefficient_width;
        const bool in_larger_corner = x >= 4u * xsize && y >= 4u * ysize;
        if (in_corner || (on_border && in_larger_corner)) {
          highest_frequency_border_sum += abs(value);
        }
      }
    }
  }

  int raw_quant = initial_raw_quant;
  if (channel == 1u && value_sum * 8.0f < float(xsize * ysize)) {
    constexpr float kLimit = 0.46f;
    int new_quant = initial_raw_quant;
    for (uint quadrant = 1u; quadrant < 4u; ++quadrant) {
      if (high_frequency_nonzeros[quadrant] == 0.0f &&
          high_frequency_max_error[quadrant] > kLimit) {
        new_quant = initial_raw_quant + 1;
        break;
      }
    }
    raw_quant = new_quant;
    if (high_frequency_nonzeros[3] == 0.0f &&
        high_frequency_max_error[3] > kLimit) {
      thresholds[3] = 0.9999f * high_frequency_max_error[3] *
        float(new_quant) / float(initial_raw_quant);
    } else if ((high_frequency_nonzeros[1] == 0.0f &&
                high_frequency_max_error[1] > kLimit) ||
               (high_frequency_nonzeros[2] == 0.0f &&
                high_frequency_max_error[2] > kLimit)) {
      thresholds[1] = 0.9999f * max(
        high_frequency_max_error[1], high_frequency_max_error[2]) *
        float(new_quant) / float(initial_raw_quant);
      thresholds[2] = thresholds[1];
    } else if (high_frequency_nonzeros[0] == 0.0f &&
               high_frequency_max_error[0] > kLimit) {
      thresholds[0] = 0.9999f * high_frequency_max_error[0] *
        float(new_quant) / float(initial_raw_quant);
    }
  }

  const float all_nonzeros = high_frequency_nonzeros[0] +
    high_frequency_nonzeros[1] + high_frequency_nonzeros[2] +
    high_frequency_nonzeros[3] + 1.0f;
  constexpr float kBorderMultiplier[3] = {70.0f, 30.0f, 60.0f};
  if (kBorderMultiplier[channel] * highest_frequency_border_sum >=
      all_nonzeros) {
    raw_quant = int(
      float(raw_quant) + kBorderMultiplier[channel] *
        highest_frequency_border_sum / all_nonzeros);
    if (raw_quant >= 256) raw_quant = 255;
  }

  if (strategy == 0u &&
      high_frequency_nonzeros[0] + high_frequency_nonzeros[1] +
        high_frequency_nonzeros[2] + high_frequency_nonzeros[3] < 11.0f) {
    ++raw_quant;
    if (raw_quant >= 256) raw_quant = 255;
  }

  constexpr float kFirstMultiplier[4][3] = {
    {0.22080615753848404f, 0.45797479824262011f, 0.29859235095977965f},
    {0.70109486510286834f, 0.16185281305512639f, 0.14387691730035473f},
    {0.114985964456218638f, 0.44656840441027695f, 0.10587658215149048f},
    {0.46849665264409396f, 0.41239077937781954f, 0.088667407767185444f},
  };
  constexpr float kSecondMultiplier[4][3] = {
    {0.27450281941822197f, 1.1255766549984996f, 0.98950459134128388f},
    {0.4652168675598285f, 0.40945807983455818f, 0.36581899811751367f},
    {0.28034972424715715f, 0.9182653201929738f, 1.5581531543057416f},
    {0.26873118114033728f, 0.68863712390392484f, 1.2082185408666786f},
  };
  error_sum *= 2.2942708343284721f;
  value_sum *= 2.2942708343284721f;
  if (strategy >= 4u) {
    uint strategy_class = 3u;
    if (strategy == 10u || strategy == 11u) {
      strategy_class = 1u;
    } else if (strategy == 4u) {
      strategy_class = 0u;
    } else if (strategy == 5u) {
      strategy_class = 2u;
    }
    const float threshold =
      kFirstMultiplier[strategy_class][channel] *
        float(xsize * ysize * 64u) +
      kSecondMultiplier[strategy_class][channel] * value_sum;
    int step = clamp(int(error_sum / threshold), 0, 2);
    if (error_sum > threshold) {
      raw_quant += step;
      if (raw_quant >= 256) raw_quant = 255;
    }
  }

  const int divisor = int(xsize * ysize);
  const float minimum_nonzeros = min(
    min(high_frequency_nonzeros[0], high_frequency_nonzeros[1]),
    min(high_frequency_nonzeros[2], high_frequency_nonzeros[3]));
  int activity = 15;
  if (minimum_nonzeros < float(15 * divisor)) {
    activity = (int(minimum_nonzeros) + divisor / 2) / divisor;
  }
  int adjusted_quant = raw_quant - activity;
  if (channel == 1u) {
    for (uint quadrant = 1u; quadrant < 4u; ++quadrant) {
      thresholds[quadrant] += 0.01f * float(activity);
    }
  }
  const int original_quant_limit = max(4, raw_quant / 2);
  if (adjusted_quant < original_quant_limit) {
    adjusted_quant = original_quant_limit;
  }
  return adjusted_quant;
}

[[maybe_unused]] static AqAdjustedQuantization aq_select_adjusted_quantization(
  device const float* coefficients,
  device const float* quant_tables,
  uint coefficient_count,
  uint channel_stride,
  uint coefficient_width,
  uint coefficient_height,
  uint strategy,
  uint global_scale,
  int initial_raw_quant,
  float x_matrix_multiplier,
  float b_matrix_multiplier,
  device atomic_uint* error) {

  AqAdjustedQuantization result{
    0,
    float4(0.58f, 0.64f, 0.64f, 0.64f),
  };
  constexpr uint kChannels[3] = {1u, 0u, 2u};
  for (uint channel_index = 0u; channel_index < 3u; ++channel_index) {
    const uint channel = kChannels[channel_index];
    float4 thresholds(0.58f, 0.64f, 0.64f, 0.64f);
    const float matrix_multiplier = channel == 0u
      ? x_matrix_multiplier
      : (channel == 2u ? b_matrix_multiplier : 1.0f);
    const int candidate = aq_adjust_quant_for_channel(
      coefficients, quant_tables, coefficient_count, channel_stride,
      coefficient_width,
      coefficient_height, strategy, channel, global_scale,
      initial_raw_quant, matrix_multiplier, thresholds, error);
    if (channel == 1u) result.y_thresholds = thresholds;
    result.raw_quant = max(result.raw_quant, candidate);
  }
  return result;
}
