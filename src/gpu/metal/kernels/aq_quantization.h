// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <metal_stdlib>

using namespace metal;

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
