// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <metal_stdlib>

#include "aq_quantization.h"

using namespace metal;

struct AqReconstructionParams {
  uint coding_width;
  uint coding_height;
  uint coding_stride;
  uint block_width;
  uint block_height;
  uint raw_quant_stride;
  uint color_width;
  uint color_stride;
  uint anchor_offset;
  uint anchor_count;
  uint coefficient_offset;
  uint coefficient_count;
  uint pixel_width;
  uint pixel_height;
  uint covered_width;
  uint covered_height;
  uint strategy;
  uint global_scale;
  uint quant_dc;
  float x_matrix_multiplier;
  float b_matrix_multiplier;
  uint adjust_ac_quant;
  uint inverse_sigma_stride;
  uint epf_sharpness_stride;
  float epf_quant_multiplier;
  float epf_sharpness_lut[8];
  uint use_resident_quantizer;
};

struct AqResetParams {
  uint coefficient_value_count;
  uint dc_value_count;
  uint pixel_value_count;
  uint block_value_count;
  uint test_error_mask;
  uint preserve_error;
  uint preserve_forward_coefficients;
  uint poison_outputs;
};

struct AqResidentPolicyInitializeParams {
  uint block_width;
  uint block_height;
  uint quant_stride;
  uint initial_stride;
  uint score_count;
};

struct AqResidentPolicyUpdateParams {
  uint block_width;
  uint block_height;
  uint quant_stride;
  uint initial_stride;
  uint block_distance_stride;
  uint score_index;
  uint iteration;
  uint apply_update;
  float butteraugli_target;
  float lower_bound;
  float upper_bound;
};

struct AqInitialCflParams {
  uint width;
  uint height;
  uint coding_stride;
  uint tile_width;
  uint tile_height;
  uint color_stride;
};

struct AqFinalCflParams {
  uint tile_width;
  uint tile_height;
  uint color_stride;
  uint transform_count;
};

struct AqColorTransformRecord {
  uint coefficient_offset;
  uint channel_stride;
  uint coefficient_count;
  uint strategy;
  uint raw_quant_index;
  uint tile_value_offset;
};

struct AqInitialQuantGradientParams {
  uint width;
  uint height;
  uint coding_stride;
  uint pixel_mask_stride;
  uint pre_erosion_width;
  uint pre_erosion_stride;
  uint test_error_mask;
};

struct AqInitialQuantErosionParams {
  uint pre_erosion_width;
  uint pre_erosion_height;
  uint pre_erosion_stride;
  uint block_width;
  uint block_height;
  uint quant_stride;
  uint strategy_mask_stride;
  float weights[4];
};

struct AqInitialQuantModulationParams {
  uint coding_stride;
  uint block_width;
  uint block_height;
  uint quant_stride;
  float multiplier;
  float addend;
};

struct AqInitialQuantSelectionParams {
  uint value_count;
  uint padded_count;
  uint median_index;
  uint quant_width;
  uint quant_height;
  uint quant_stride;
  uint raw_quant_stride;
  uint scaled_quant_dc;
  float quant_dc;
};

struct AqInitialQuantSortParams {
  uint compare_distance;
  uint sequence_length;
  uint value_count;
};

struct AqQuantFieldAdjustmentParams {
  uint quant_stride;
  uint anchor_offset;
  uint anchor_count;
  uint covered_width;
  uint covered_height;
  float mean_max_mixer;
};

struct AqResidentQuantSelectionPass {
  uint shift;
  uint deviation;
};

struct AqQuantizationProbeParams {
  uint coefficient_count;
  uint strategy;
  uint channel;
  int raw_quant;
  uint global_scale;
  float matrix_multiplier;
};

struct AqAdjustmentProbeParams {
  uint coefficient_count;
  uint coefficient_width;
  uint coefficient_height;
  uint strategy;
  int initial_raw_quant;
  uint global_scale;
  float x_matrix_multiplier;
  float b_matrix_multiplier;
};

kernel void gjxl_aq_reset_exact_evaluation(
  device atomic_uint* error [[buffer(0)]],
  device float* block_distance [[buffer(1)]],
  constant AqResetParams& params [[buffer(2)]],
  uint index [[thread_position_in_grid]]) {

  constexpr uint kPoison = 0x7fc12345u;
  if (index == 0u && params.preserve_error == 0u) {
    atomic_store_explicit(
      error, params.test_error_mask, memory_order_relaxed);
  }
  if (index < params.block_value_count) {
    block_distance[index] = as_type<float>(kPoison);
  }
}

static uint aq_coefficient_index(
  constant AqReconstructionParams& params,
  uint vertical_frequency,
  uint horizontal_frequency) {

  return params.pixel_height < params.pixel_width
    ? vertical_frequency * params.pixel_width + horizontal_frequency
    : horizontal_frequency * params.pixel_height + vertical_frequency;
}

static float aq_downsample_scale(uint length, uint frequency) {
  if (length == 1u) return 1.0f;
  if (length == 2u) return frequency == 0u ? 1.0f : 0.901764195028874394f;
  constexpr float kScale[4] = {
    1.0f,
    0.974886821136879522f,
    0.901764195028874394f,
    0.787054918159101335f,
  };
  return kScale[frequency];
}

static float aq_upsample_scale(uint length, uint frequency) {
  if (length == 1u) return 1.0f;
  if (length == 2u) return frequency == 0u ? 1.0f : 1.108937353592731823f;
  constexpr float kScale[4] = {
    1.0f,
    1.025760096781116015f,
    1.108937353592731823f,
    1.270559368765487251f,
  };
  return kScale[frequency];
}

static float aq_forward_basis(uint length, uint frequency, uint sample) {
  const float alpha = frequency == 0u ? 0.7071067811865475244f : 1.0f;
  const float angle =
    (float(sample) + 0.5f) * float(frequency) * M_PI_F / float(length);
  return 1.4142135623730950488f * alpha * cos(angle) / float(length);
}

static float aq_inverse_basis(uint length, uint frequency, uint sample) {
  const float alpha = frequency == 0u ? 0.7071067811865475244f : 1.0f;
  const float angle =
    (float(sample) + 0.5f) * float(frequency) * M_PI_F / float(length);
  return 1.4142135623730950488f * alpha * cos(angle);
}

static int aq_round_dc(float value, device atomic_uint* error) {
  const float rounded = round(value);
  if (!isfinite(rounded) || rounded >= 2147483648.0f ||
      rounded < -2147483648.0f) {
    atomic_fetch_or_explicit(error, 16u, memory_order_relaxed);
    return 0;
  }
  return int(rounded);
}

kernel void gjxl_aq_reset_reconstruction(
  device float* gathered_pixels [[buffer(0)]],
  device float* forward_coefficients [[buffer(1)]],
  device int* quantized_coefficients [[buffer(2)]],
  device float* reconstruction_coefficients [[buffer(3)]],
  device float* dc [[buffer(4)]],
  device int* quantized_dc [[buffer(5)]],
  device float* reconstructed_x [[buffer(6)]],
  device float* reconstructed_y [[buffer(7)]],
  device float* reconstructed_b [[buffer(8)]],
  device atomic_uint* error [[buffer(9)]],
  device float* block_distance [[buffer(10)]],
  constant AqResetParams& params [[buffer(11)]],
  uint index [[thread_position_in_grid]]) {

  constexpr uint kPoison = 0x7fc12345u;
  if (index == 0u && params.preserve_error == 0u) {
    atomic_store_explicit(
      error, params.test_error_mask, memory_order_relaxed);
  }
  if (params.poison_outputs == 0u) return;
  if (index < params.coefficient_value_count) {
    gathered_pixels[index] = as_type<float>(kPoison);
    if (params.preserve_forward_coefficients == 0u) {
      forward_coefficients[index] = as_type<float>(kPoison);
    }
    quantized_coefficients[index] = int(0x81234567u);
    reconstruction_coefficients[index] = as_type<float>(kPoison);
  }
  if (index < params.dc_value_count) {
    dc[index] = as_type<float>(kPoison);
    quantized_dc[index] = int(0x81234567u);
  }
  if (index < params.pixel_value_count) {
    reconstructed_x[index] = as_type<float>(kPoison);
    reconstructed_y[index] = as_type<float>(kPoison);
    reconstructed_b[index] = as_type<float>(kPoison);
  }
  if (index < params.block_value_count) {
    block_distance[index] = as_type<float>(kPoison);
  }
}

static char aq_quantize_initial_cfl(float value) {
  constexpr float kTowardsZero = 2.6f;
  if (value >= kTowardsZero) {
    value -= kTowardsZero;
  } else if (value <= -kTowardsZero) {
    value += kTowardsZero;
  } else {
    value = 0.0f;
  }
  return char(clamp(round(value), -128.0f, 127.0f));
}

// The maximum-throughput encoder's initial CfL policy accumulates four CPU
// SIMD lanes independently. One Metal thread owns a complete 64x64 tile and
// preserves that order exactly; tiles remain independent and run in parallel.
kernel void gjxl_aq_initial_cfl(
  device const float* coding_x [[buffer(0)]],
  device const float* coding_y [[buffer(1)]],
  device const float* coding_b [[buffer(2)]],
  device char* y_to_x [[buffer(3)]],
  device char* y_to_b [[buffer(4)]],
  device atomic_uint* error [[buffer(5)]],
  constant AqInitialCflParams& params [[buffer(6)]],
  uint tile_index [[thread_position_in_grid]]) {

  const uint tile_count = params.tile_width * params.tile_height;
  if (tile_index >= tile_count) return;
  const uint tile_x = tile_index % params.tile_width;
  const uint tile_y = tile_index / params.tile_width;
  const uint x_begin = tile_x * 64u;
  const uint y_begin = tile_y * 64u;
  const uint x_end = min(x_begin + 64u, params.width);
  const uint y_end = min(y_begin + 64u, params.height);

  float sum_y[4] = {};
  float sum_x[4] = {};
  float sum_b[4] = {};
  uint sample_index = 0u;
  for (uint y = y_begin; y < y_end; ++y) {
    const uint row = y * params.coding_stride;
    for (uint x = x_begin; x < x_end; ++x, ++sample_index) {
      const uint lane = sample_index & 3u;
      const float value_y = coding_y[row + x];
      const float value_x = coding_x[row + x];
      const float value_b = coding_b[row + x];
      if (!isfinite(value_y) || !isfinite(value_x) || !isfinite(value_b)) {
        atomic_fetch_or_explicit(error, 1024u, memory_order_relaxed);
        return;
      }
      sum_y[lane] += value_y;
      sum_x[lane] += value_x;
      sum_b[lane] += value_b;
    }
  }
  if (sample_index == 0u) {
    atomic_fetch_or_explicit(error, 1024u, memory_order_relaxed);
    return;
  }
  const float sample_count = float(sample_index);
  const float mean_y = ((sum_y[0] + sum_y[1]) + (sum_y[2] + sum_y[3])) /
    sample_count;
  const float mean_x = ((sum_x[0] + sum_x[1]) + (sum_x[2] + sum_x[3])) /
    sample_count;
  const float mean_b = ((sum_b[0] + sum_b[1]) + (sum_b[2] + sum_b[3])) /
    sample_count;

  float quadratic[4] = {};
  float linear_x[4] = {};
  float linear_b[4] = {};
  sample_index = 0u;
  for (uint y = y_begin; y < y_end; ++y) {
    const uint row = y * params.coding_stride;
    for (uint x = x_begin; x < x_end; ++x, ++sample_index) {
      const uint lane = sample_index & 3u;
      const float centered_y = coding_y[row + x] - mean_y;
      const float a = centered_y * (1.0f / 84.0f);
      quadratic[lane] = fma(a, a, quadratic[lane]);
      linear_x[lane] = fma(
        a, -(coding_x[row + x] - mean_x), linear_x[lane]);
      linear_b[lane] = fma(
        a, centered_y - (coding_b[row + x] - mean_b), linear_b[lane]);
    }
  }
  const float quadratic_sum =
    (quadratic[0] + quadratic[1]) + (quadratic[2] + quadratic[3]);
  const float linear_x_sum =
    (linear_x[0] + linear_x[1]) + (linear_x[2] + linear_x[3]);
  const float linear_b_sum =
    (linear_b[0] + linear_b[1]) + (linear_b[2] + linear_b[3]);
  const float denominator = quadratic_sum + sample_count * 5.0e-10f;
  const uint color_index = tile_y * params.color_stride + tile_x;
  y_to_x[color_index] = aq_quantize_initial_cfl(-linear_x_sum / denominator);
  y_to_b[color_index] = aq_quantize_initial_cfl(-linear_b_sum / denominator);
}

static bool aq_final_cfl_layout(
  uint strategy,
  thread uint& coefficient_width,
  thread uint& coefficient_height,
  thread uint& low_frequency_width,
  thread uint& low_frequency_height) {

  switch (strategy) {
    case 0u:
      coefficient_width = 8u;
      coefficient_height = 8u;
      low_frequency_width = 1u;
      low_frequency_height = 1u;
      return true;
    case 4u:
      coefficient_width = 16u;
      coefficient_height = 16u;
      low_frequency_width = 2u;
      low_frequency_height = 2u;
      return true;
    case 5u:
      coefficient_width = 32u;
      coefficient_height = 32u;
      low_frequency_width = 4u;
      low_frequency_height = 4u;
      return true;
    case 6u:
    case 7u:
      coefficient_width = 16u;
      coefficient_height = 8u;
      low_frequency_width = 2u;
      low_frequency_height = 1u;
      return true;
    case 10u:
    case 11u:
      coefficient_width = 32u;
      coefficient_height = 16u;
      low_frequency_width = 4u;
      low_frequency_height = 2u;
      return true;
    default:
      return false;
  }
}

// Mirrors ComputeFinalColorCorrelationMapPrepared(..., fast=true). Four
// threads preserve its four independent accumulation lanes and their final
// pairwise reduction order.
kernel void gjxl_aq_final_cfl(
  device const AqColorTransformRecord* transforms [[buffer(0)]],
  device const uint* tile_offsets [[buffer(1)]],
  device const float* quant_tables [[buffer(2)]],
  device const float* forward_coefficients [[buffer(3)]],
  device const int* raw_quant [[buffer(4)]],
  device const uint* resident_quantizer [[buffer(5)]],
  device char* y_to_x [[buffer(6)]],
  device char* y_to_b [[buffer(7)]],
  device atomic_uint* error [[buffer(8)]],
  constant AqFinalCflParams& params [[buffer(9)]],
  uint tile_index [[threadgroup_position_in_grid]],
  uint lane [[thread_index_in_threadgroup]]) {

  if (tile_index >= params.tile_width * params.tile_height || lane >= 4u) {
    return;
  }
  const uint begin = tile_offsets[tile_index];
  const uint end = tile_offsets[tile_index + 1u];
  if (begin >= end || end > params.transform_count) {
    if (lane == 0u) {
      atomic_fetch_or_explicit(error, 2097152u, memory_order_relaxed);
    }
    return;
  }

  const uint global_scale = resident_quantizer[0];
  float quadratic_x = 0.0f;
  float linear_x = 0.0f;
  float quadratic_b = 0.0f;
  float linear_b = 0.0f;
  for (uint transform_index = begin; transform_index < end;
       ++transform_index) {
    const AqColorTransformRecord transform = transforms[transform_index];
    uint coefficient_width = 0u;
    uint coefficient_height = 0u;
    uint low_frequency_width = 0u;
    uint low_frequency_height = 0u;
    if (!aq_final_cfl_layout(
          transform.strategy, coefficient_width, coefficient_height,
          low_frequency_width, low_frequency_height) ||
        coefficient_width * coefficient_height !=
          transform.coefficient_count) {
      atomic_fetch_or_explicit(error, 2097152u, memory_order_relaxed);
      continue;
    }
    const int raw = raw_quant[transform.raw_quant_index];
    if (raw < 1 || raw > 256 || global_scale == 0u ||
        global_scale > 32768u) {
      atomic_fetch_or_explicit(error, 2097152u, memory_order_relaxed);
      continue;
    }
    const float quant_scale =
      (float(global_scale) * (1.0f / 65536.0f)) * 128.0f * float(raw);
    const uint2 table_offsets = aq_quant_table_offsets(transform.strategy);
    const uint first =
      (lane + 4u - (transform.tile_value_offset & 3u)) & 3u;
    for (uint coefficient = first;
         coefficient < transform.coefficient_count;
         coefficient += 4u) {
      const uint x = coefficient % coefficient_width;
      const uint y = coefficient / coefficient_width;
      if (x < low_frequency_width && y < low_frequency_height) {
        continue;
      }
      const uint base = transform.coefficient_offset + coefficient;
      const float coefficient_y =
        forward_coefficients[base + transform.channel_stride];
      const float coefficient_x = forward_coefficients[base];
      const float coefficient_b =
        forward_coefficients[base + 2u * transform.channel_stride];
      const float value_y_x = coefficient_y *
        quant_tables[table_offsets.y + coefficient] * quant_scale;
      const float value_x = coefficient_x *
        quant_tables[table_offsets.y + coefficient] * quant_scale;
      const float value_y_b = coefficient_y *
        quant_tables[
          table_offsets.y + 2u * transform.coefficient_count + coefficient] *
        quant_scale;
      const float value_b = coefficient_b *
        quant_tables[
          table_offsets.y + 2u * transform.coefficient_count + coefficient] *
        quant_scale;
      const float a_x = value_y_x * (1.0f / 84.0f);
      const float a_b = value_y_b * (1.0f / 84.0f);
      quadratic_x = fma(a_x, a_x, quadratic_x);
      linear_x = fma(a_x, -value_x, linear_x);
      quadratic_b = fma(a_b, a_b, quadratic_b);
      linear_b = fma(a_b, value_y_b - value_b, linear_b);
    }
  }

  threadgroup float quadratic_x_lanes[4];
  threadgroup float linear_x_lanes[4];
  threadgroup float quadratic_b_lanes[4];
  threadgroup float linear_b_lanes[4];
  quadratic_x_lanes[lane] = quadratic_x;
  linear_x_lanes[lane] = linear_x;
  quadratic_b_lanes[lane] = quadratic_b;
  linear_b_lanes[lane] = linear_b;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane != 0u) return;

  const AqColorTransformRecord last = transforms[end - 1u];
  const float sample_count =
    float(last.tile_value_offset + last.coefficient_count);
  const float quadratic_x_sum =
    (quadratic_x_lanes[0] + quadratic_x_lanes[1]) +
    (quadratic_x_lanes[2] + quadratic_x_lanes[3]);
  const float linear_x_sum =
    (linear_x_lanes[0] + linear_x_lanes[1]) +
    (linear_x_lanes[2] + linear_x_lanes[3]);
  const float quadratic_b_sum =
    (quadratic_b_lanes[0] + quadratic_b_lanes[1]) +
    (quadratic_b_lanes[2] + quadratic_b_lanes[3]);
  const float linear_b_sum =
    (linear_b_lanes[0] + linear_b_lanes[1]) +
    (linear_b_lanes[2] + linear_b_lanes[3]);
  const float result_x = -linear_x_sum /
    (quadratic_x_sum + sample_count * 5.0e-10f);
  const float result_b = -linear_b_sum /
    (quadratic_b_sum + sample_count * 5.0e-10f);
  if (!isfinite(result_x) || !isfinite(result_b)) {
    atomic_fetch_or_explicit(error, 2097152u, memory_order_relaxed);
    return;
  }
  const uint color_index =
    (tile_index / params.tile_width) * params.color_stride +
    tile_index % params.tile_width;
  y_to_x[color_index] = aq_quantize_initial_cfl(result_x);
  y_to_b[color_index] = aq_quantize_initial_cfl(result_b);
}

kernel void gjxl_aq_reset_initial_quant(
  device atomic_uint* error [[buffer(0)]],
  constant AqInitialQuantGradientParams& params [[buffer(1)]],
  uint index [[thread_position_in_grid]]) {

  if (index == 0u) {
    atomic_store_explicit(error, params.test_error_mask, memory_order_relaxed);
  }
}

template <bool Invert>
static float aq_initial_quant_gamma_ratio(float value) {
  constexpr float kEpsilon = 1.0e-2f;
  constexpr float kSgMul = 226.77216153508914f;
  constexpr float kSgMul2 = 1.0f / 73.377132366608819f;
  constexpr float kInverseLog2E = 0.6931471805599453f;
  constexpr float kSgReturnMul =
    kSgMul2 * 18.6580932135f * kInverseLog2E;
  constexpr float kSgOffset = 7.7825991679894591f;
  value = max(value, 0.0f);
  const float squared = value * value;
  const float numerator = fma(
    kSgReturnMul * 3.0f * kSgMul, squared, kEpsilon);
  const float denominator = fma(
    kInverseLog2E * kSgMul * value, squared,
    kSgOffset * kInverseLog2E + kEpsilon);
  return Invert ? numerator / denominator : denominator / numerator;
}

static float aq_initial_quant_masking_sqrt(float value) {
  constexpr float kLogOffset = 27.505837037000106f;
  constexpr float kMul = 211.66567973503678f;
  const float inner_scale = sqrt(kMul * 1.0e8f);
  return 0.25f * sqrt(fma(value, inner_scale, kLogOffset));
}

static float aq_initial_quant_log1p(float value) {
  const float sum = 1.0f + value;
  const float correction = (value - (sum - 1.0f)) / sum;
  return log(sum) + correction;
}

kernel void gjxl_aq_initial_quant_gradient(
  device const float* coding_y [[buffer(0)]],
  device float* pixel_mask [[buffer(1)]],
  device float* pre_erosion [[buffer(2)]],
  device atomic_uint* error [[buffer(3)]],
  constant AqInitialQuantGradientParams& params [[buffer(4)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.pre_erosion_width ||
      position.y >= params.height / 4u) return;
  constexpr float kMatchGammaOffset = 0.019f;
  constexpr float kDifferenceLimit = 0.2f;
  float row_differences[4] = {};
  const uint x_begin = position.x * 4u;
  const uint y_begin = position.y * 4u;
  for (uint row_index = 0u; row_index < 4u; ++row_index) {
    const uint y = y_begin + row_index;
    const uint top = y == 0u ? y : y - 1u;
    const uint bottom = min(y + 1u, params.height - 1u);
    for (uint column = 0u; column < 4u; ++column) {
      const uint x = x_begin + column;
      const uint left = x == 0u ? x : x - 1u;
      const uint right = min(x + 1u, params.width - 1u);
      const float center = coding_y[y * params.coding_stride + x];
      const float base = 0.25f * (
        coding_y[bottom * params.coding_stride + x] +
        coding_y[top * params.coding_stride + x] +
        coding_y[y * params.coding_stride + left] +
        coding_y[y * params.coding_stride + right]);
      const float gamma =
        aq_initial_quant_gamma_ratio<false>(center + kMatchGammaOffset);
      const float delta = gamma * (center - base);
      const float mask =
        1.0f / (aq_initial_quant_log1p(abs(delta)) + 0.01f);
      pixel_mask[y * params.pixel_mask_stride + x] = mask;
      float block_difference = min(delta * delta, kDifferenceLimit);
      block_difference = aq_initial_quant_masking_sqrt(block_difference);
      row_differences[column] += block_difference;
      if (!isfinite(mask) || mask <= 0.0f ||
          !isfinite(block_difference)) {
        atomic_fetch_or_explicit(error, 2048u, memory_order_relaxed);
      }
    }
  }
  const float value = (((row_differences[0] + row_differences[1]) +
                        row_differences[2]) + row_differences[3]) * 0.25f;
  pre_erosion[position.y * params.pre_erosion_stride + position.x] = value;
  if (!isfinite(value) || value <= 0.0f) {
    atomic_fetch_or_explicit(error, 2048u, memory_order_relaxed);
  }
}

static void aq_initial_quant_store_min4(
  float value,
  thread float& min0,
  thread float& min1,
  thread float& min2,
  thread float& min3) {

  if (value >= min3) return;
  if (value < min0) {
    min3 = min2;
    min2 = min1;
    min1 = min0;
    min0 = value;
  } else if (value < min1) {
    min3 = min2;
    min2 = min1;
    min1 = value;
  } else if (value < min2) {
    min3 = min2;
    min2 = value;
  } else {
    min3 = value;
  }
}

static void aq_initial_quant_sort4(thread float values[4]) {
  if (values[0] > values[1]) {
    const float value = values[0];
    values[0] = values[1];
    values[1] = value;
  }
  if (values[0] > values[2]) {
    const float value = values[0];
    values[0] = values[2];
    values[2] = value;
  }
  if (values[0] > values[3]) {
    const float value = values[0];
    values[0] = values[3];
    values[3] = value;
  }
  if (values[1] > values[2]) {
    const float value = values[1];
    values[1] = values[2];
    values[2] = value;
  }
  if (values[1] > values[3]) {
    const float value = values[1];
    values[1] = values[3];
    values[3] = value;
  }
  if (values[2] > values[3]) {
    const float value = values[2];
    values[2] = values[3];
    values[3] = value;
  }
}

static float aq_initial_quant_eroded_value(
  device const float* source,
  uint x,
  uint y,
  constant AqInitialQuantErosionParams& params) {

  const uint top = y == 0u ? y : y - 1u;
  const uint bottom = min(y + 1u, params.pre_erosion_height - 1u);
  const uint left = x == 0u ? x : x - 1u;
  const uint right = min(x + 1u, params.pre_erosion_width - 1u);
  const auto at = [&](uint sample_x, uint sample_y) {
    return source[sample_y * params.pre_erosion_stride + sample_x];
  };
  float minima[4] = {at(x, y), at(left, y), at(right, y), at(left, top)};
  aq_initial_quant_sort4(minima);
  aq_initial_quant_store_min4(
    at(x, top), minima[0], minima[1], minima[2], minima[3]);
  aq_initial_quant_store_min4(
    at(right, top), minima[0], minima[1], minima[2], minima[3]);
  aq_initial_quant_store_min4(
    at(left, bottom), minima[0], minima[1], minima[2], minima[3]);
  aq_initial_quant_store_min4(
    at(x, bottom), minima[0], minima[1], minima[2], minima[3]);
  aq_initial_quant_store_min4(
    at(right, bottom), minima[0], minima[1], minima[2], minima[3]);
  return ((params.weights[0] * minima[0] +
           params.weights[1] * minima[1]) +
          params.weights[2] * minima[2]) +
         params.weights[3] * minima[3];
}

kernel void gjxl_aq_initial_quant_fuzzy_erosion(
  device const float* pre_erosion [[buffer(0)]],
  device float* quant_field [[buffer(1)]],
  device float* strategy_mask [[buffer(2)]],
  device atomic_uint* error [[buffer(3)]],
  constant AqInitialQuantErosionParams& params [[buffer(4)]],
  uint2 block [[thread_position_in_grid]]) {

  if (block.x >= params.block_width || block.y >= params.block_height) return;
  const uint source_x = block.x * 2u;
  const uint source_y = block.y * 2u;
  float value = aq_initial_quant_eroded_value(
    pre_erosion, source_x, source_y, params);
  value += aq_initial_quant_eroded_value(
    pre_erosion, source_x + 1u, source_y, params);
  value += aq_initial_quant_eroded_value(
    pre_erosion, source_x, source_y + 1u, params);
  value += aq_initial_quant_eroded_value(
    pre_erosion, source_x + 1u, source_y + 1u, params);
  const uint quant_index = block.y * params.quant_stride + block.x;
  const float strategy = 1.0f / (value + 0.001f);
  quant_field[quant_index] = value;
  strategy_mask[block.y * params.strategy_mask_stride + block.x] = strategy;
  if (!isfinite(value) || value <= 0.0f ||
      !isfinite(strategy) || strategy <= 0.0f) {
    atomic_fetch_or_explicit(error, 4096u, memory_order_relaxed);
  }
}

static float aq_initial_quant_compute_mask(float value) {
  constexpr float kBase = -0.7647f;
  constexpr float kMul4 = 9.4708735624378946f;
  constexpr float kMul2 = 17.35036561631863f;
  constexpr float kOffset2 = 302.59587815579727f;
  constexpr float kMul3 = 6.7943250517376494f;
  constexpr float kOffset3 = 3.7179635626140772f;
  constexpr float kOffset4 = 0.25f * kOffset3;
  constexpr float kMul0 = 0.80061762862741759f;
  const float v1 = max(value * kMul0, 1.0e-3f);
  const float v2 = 1.0f / (v1 + kOffset2);
  const float v3 = 1.0f / fma(v1, v1, kOffset3);
  const float v4 = 1.0f / fma(v1, v1, kOffset4);
  return kBase + fma(kMul4, v4, fma(kMul2, v2, kMul3 * v3));
}

static float aq_initial_quant_fast_log2(float value) {
  constexpr float kP0 = -1.8503833400518310e-06f;
  constexpr float kP1 = 1.4287160470083755f;
  constexpr float kP2 = 0.74245873327820566f;
  constexpr float kQ0 = 0.99032814277590719f;
  constexpr float kQ1 = 1.0096718572241148f;
  constexpr float kQ2 = 0.17409343003366853f;
  const uint value_bits = as_type<uint>(value);
  const int shifted_exponent = int(value_bits - 0x3f2aaaabu) >> 23;
  const uint mantissa_bits =
    value_bits - (uint(shifted_exponent) << 23);
  const float x = as_type<float>(mantissa_bits) - 1.0f;
  float numerator = fma(kP2, x, kP1);
  numerator = fma(numerator, x, kP0);
  float denominator = fma(kQ2, x, kQ1);
  denominator = fma(denominator, x, kQ0);
  return numerator / denominator + float(shifted_exponent);
}

static float aq_initial_quant_fast_pow2(float value) {
  const float floor_value = floor(value);
  const int exponent = int(floor_value) + 127;
  const float exponent_value = as_type<float>(uint(exponent) << 23);
  const float fraction = value - floor_value;
  float numerator = fraction + 1.01749063e+01f;
  numerator = fma(numerator, fraction, 4.88687798e+01f);
  numerator = fma(numerator, fraction, 9.85506591e+01f);
  numerator *= exponent_value;
  float denominator = fma(fraction, 2.10242958e-01f, -2.22328856e-02f);
  denominator = fma(denominator, fraction, -1.94414990e+01f);
  denominator = fma(denominator, fraction, 9.85506633e+01f);
  return numerator / denominator;
}

kernel void gjxl_aq_initial_quant_modulation(
  device const float* coding_x [[buffer(0)]],
  device const float* coding_y [[buffer(1)]],
  device const float* coding_b [[buffer(2)]],
  device float* quant_field [[buffer(3)]],
  device atomic_uint* error [[buffer(4)]],
  constant AqInitialQuantModulationParams& params [[buffer(5)]],
  uint2 block [[thread_position_in_grid]]) {

  if (block.x >= params.block_width || block.y >= params.block_height) return;
  constexpr float kGammaBias = 0.16f;
  constexpr float kFrequencyLimit = 0.0206f;
  constexpr float kBlueLimit = 0.010474084867598155f;
  constexpr float kBlueOffset = 0.0031994768654636393f;
  float gamma_lanes[4] = {};
  float frequency_lanes[4] = {};
  float blue_lanes[4] = {};
  const uint pixel_x = block.x * 8u;
  const uint pixel_y = block.y * 8u;
  for (uint dy = 0u; dy < 8u; ++dy) {
    const uint row = (pixel_y + dy) * params.coding_stride + pixel_x;
    const uint next_row = dy + 1u < 8u ? row + params.coding_stride : row;
    for (uint dx = 0u; dx < 8u; ++dx) {
      const uint lane = dx & 3u;
      const float value_x = coding_x[row + dx];
      const float value_y = coding_y[row + dx];
      const float value_b = coding_b[row + dx];
      const float in_y = value_y + kGammaBias;
      gamma_lanes[lane] +=
        aq_initial_quant_gamma_ratio<true>(in_y - value_x);
      gamma_lanes[lane] +=
        aq_initial_quant_gamma_ratio<true>(in_y + value_x);
      if (dx + 1u < 8u) {
        frequency_lanes[lane] += min(
          kFrequencyLimit, abs(value_y - coding_y[row + dx + 1u]));
      }
      frequency_lanes[lane] += min(
        kFrequencyLimit, abs(value_y - coding_y[next_row + dx]));
      const float effective_y = value_y + kBlueOffset + abs(value_x);
      if (value_b > effective_y) {
        blue_lanes[lane] += min(value_b - effective_y, kBlueLimit);
      }
    }
  }
  const float gamma_overall =
    ((gamma_lanes[0] + gamma_lanes[1]) +
     (gamma_lanes[2] + gamma_lanes[3])) * (0.5f / 64.0f);
  const uint quant_index = block.y * params.quant_stride + block.x;
  const float mask = aq_initial_quant_compute_mask(quant_field[quant_index]);
  constexpr float kGamma = 0.1005613337192697f;
  const float gamma_value = fma(
    kGamma, aq_initial_quant_fast_log2(gamma_overall), mask);
  const float frequency_sum =
    (frequency_lanes[0] + frequency_lanes[1]) +
    (frequency_lanes[2] + frequency_lanes[3]);
  const float frequency_value =
    gamma_value + (frequency_sum * -0.38f + 0.42f);
  float blue_sum =
    (blue_lanes[0] + blue_lanes[1]) +
    (blue_lanes[2] + blue_lanes[3]);
  if (blue_sum >= 32.0f * kBlueLimit) {
    blue_sum = 64.0f * kBlueLimit - blue_sum;
  }
  blue_sum = min(blue_sum, 15.463398341612438f * kBlueLimit);
  const float blue_value = gamma_value + blue_sum * 0.90590804735610064f;
  const float exponent = min(frequency_value, blue_value);
  const float result = aq_initial_quant_fast_pow2(exponent * 1.442695041f) *
    params.multiplier + params.addend;
  quant_field[quant_index] = result;
  if (!isfinite(gamma_overall) || gamma_overall <= 0.0f ||
      !isfinite(result) || result <= 0.0f) {
    atomic_fetch_or_explicit(error, 8192u, memory_order_relaxed);
  }
}

kernel void gjxl_aq_initial_quant_sort_prepare(
  device const float* quant_field [[buffer(0)]],
  device float* sort_values [[buffer(1)]],
  constant AqInitialQuantSelectionParams& params [[buffer(2)]],
  uint index [[thread_position_in_grid]]) {

  if (index >= params.padded_count) return;
  if (index < params.value_count) {
    const uint y = index / params.quant_width;
    const uint x = index - y * params.quant_width;
    sort_values[index] = quant_field[y * params.quant_stride + x];
  } else {
    sort_values[index] = INFINITY;
  }
}

kernel void gjxl_aq_initial_quant_sort_step(
  device float* values [[buffer(0)]],
  constant AqInitialQuantSortParams& params [[buffer(1)]],
  uint index [[thread_position_in_grid]]) {

  if (index >= params.value_count) return;
  const uint partner = index ^ params.compare_distance;
  if (partner <= index || partner >= params.value_count) return;
  const bool ascending = (index & params.sequence_length) == 0u;
  const float left = values[index];
  const float right = values[partner];
  if ((ascending && left > right) || (!ascending && left < right)) {
    values[index] = right;
    values[partner] = left;
  }
}

kernel void gjxl_aq_initial_quant_capture_median(
  device const float* sort_values [[buffer(0)]],
  device float* median [[buffer(1)]],
  constant AqInitialQuantSelectionParams& params [[buffer(2)]],
  uint index [[thread_position_in_grid]]) {

  if (index == 0u) median[0] = sort_values[params.median_index];
}

kernel void gjxl_aq_initial_quant_deviation_prepare(
  device const float* quant_field [[buffer(0)]],
  device const float* median [[buffer(1)]],
  device float* sort_values [[buffer(2)]],
  constant AqInitialQuantSelectionParams& params [[buffer(3)]],
  uint index [[thread_position_in_grid]]) {

  if (index >= params.padded_count) return;
  if (index < params.value_count) {
    const uint y = index / params.quant_width;
    const uint x = index - y * params.quant_width;
    sort_values[index] =
      abs(quant_field[y * params.quant_stride + x] - median[0]);
  } else {
    sort_values[index] = INFINITY;
  }
}

kernel void gjxl_aq_initial_quant_finalize_quantizer(
  device const float* median [[buffer(0)]],
  device const float* sorted_deviations [[buffer(1)]],
  device uint* quantizer_params [[buffer(2)]],
  device atomic_uint* error [[buffer(3)]],
  constant AqInitialQuantSelectionParams& params [[buffer(4)]],
  uint index [[thread_position_in_grid]]) {

  if (index != 0u) return;
  const float median_value = median[0];
  const float deviation = sorted_deviations[params.median_index];
  float scale = 65536.0f * (median_value - deviation) / 5.0f;
  scale = clamp(scale, 1.0f, 32768.0f);
  uint global_scale = uint(scale);
  if (global_scale > params.scaled_quant_dc) {
    global_scale = max(1u, params.scaled_quant_dc);
  }
  const float inverse_global_scale = 65536.0f / float(global_scale);
  const float quant_dc = min(
    65536.0f, params.quant_dc * inverse_global_scale + 0.5f);
  quantizer_params[0] = global_scale;
  quantizer_params[1] = uint(quant_dc);
  if (!isfinite(median_value) || median_value <= 0.0f ||
      !isfinite(deviation) || deviation < 0.0f || global_scale == 0u ||
      global_scale > 32768u || quantizer_params[1] == 0u ||
      quantizer_params[1] > 65536u) {
    atomic_fetch_or_explicit(error, 32768u, memory_order_relaxed);
  }
}

kernel void gjxl_aq_initial_quant_raw_quant(
  device const float* quant_field [[buffer(0)]],
  device const uint* quantizer_params [[buffer(1)]],
  device int* raw_quant [[buffer(2)]],
  device atomic_uint* error [[buffer(3)]],
  constant AqInitialQuantSelectionParams& params [[buffer(4)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.quant_width ||
      position.y >= params.quant_height) return;
  const float inverse_global_scale =
    65536.0f / float(quantizer_params[0]);
  const float value = quant_field[
    position.y * params.quant_stride + position.x] *
    inverse_global_scale + 0.5f;
  const int raw = int(clamp(value, 1.0f, 256.0f));
  raw_quant[position.y * params.raw_quant_stride + position.x] = raw;
  if (!isfinite(value) || raw < 1 || raw > 256) {
    atomic_fetch_or_explicit(error, 65536u, memory_order_relaxed);
  }
}

kernel void gjxl_aq_adjust_quant_field(
  device const uint2* anchors [[buffer(0)]],
  device float* quant_field [[buffer(1)]],
  device atomic_uint* error [[buffer(2)]],
  constant AqQuantFieldAdjustmentParams& params [[buffer(3)]],
  uint anchor_index [[thread_position_in_grid]]) {

  if (anchor_index >= params.anchor_count) return;
  const uint2 anchor = anchors[params.anchor_offset + anchor_index];
  const uint covered_count = params.covered_width * params.covered_height;
  float maximum = quant_field[
    anchor.y * params.quant_stride + anchor.x];
  float mean = 0.0f;
  for (uint y = 0u; y < params.covered_height; ++y) {
    for (uint x = 0u; x < params.covered_width; ++x) {
      const float value = quant_field[
        (anchor.y + y) * params.quant_stride + anchor.x + x];
      maximum = max(maximum, value);
      mean += value;
    }
  }
  mean /= float(covered_count);
  const float result = covered_count < 4u
    ? maximum
    : maximum * params.mean_max_mixer +
        mean * (1.0f - params.mean_max_mixer);
  if (!isfinite(result) || result <= 0.0f) {
    atomic_fetch_or_explicit(error, 131072u, memory_order_relaxed);
    return;
  }
  for (uint y = 0u; y < params.covered_height; ++y) {
    for (uint x = 0u; x < params.covered_width; ++x) {
      quant_field[
        (anchor.y + y) * params.quant_stride + anchor.x + x] = result;
    }
  }
}

kernel void gjxl_aq_resident_quant_select_initialize(
  device uint* state [[buffer(0)]],
  device atomic_uint* histogram [[buffer(1)]],
  constant AqInitialQuantSelectionParams& params [[buffer(2)]],
  uint index [[thread_position_in_grid]]) {

  if (index == 0u) {
    state[0] = 0u;
    state[1] = 0u;
    state[2] = params.median_index;
  }
  if (index < 256u) {
    atomic_store_explicit(
      histogram + index, 0u, memory_order_relaxed);
  }
}

kernel void gjxl_aq_resident_quant_histogram(
  device const float* quant_field [[buffer(0)]],
  device const float* statistics [[buffer(1)]],
  device atomic_uint* histogram [[buffer(2)]],
  device const uint* state [[buffer(3)]],
  constant AqInitialQuantSelectionParams& params [[buffer(4)]],
  constant AqResidentQuantSelectionPass& pass [[buffer(5)]],
  uint index [[thread_position_in_grid]]) {

  if (index >= params.value_count) return;
  const uint y = index / params.quant_width;
  const uint x = index - y * params.quant_width;
  float value = quant_field[y * params.quant_stride + x];
  if (pass.deviation != 0u) value = abs(value - statistics[0]);
  const uint bits = as_type<uint>(value);
  if ((bits & state[1]) != state[0]) return;
  const uint bucket = (bits >> pass.shift) & 255u;
  atomic_fetch_add_explicit(
    histogram + bucket, 1u, memory_order_relaxed);
}

kernel void gjxl_aq_resident_quant_select_bucket(
  device atomic_uint* histogram [[buffer(0)]],
  device uint* state [[buffer(1)]],
  device float* statistics [[buffer(2)]],
  constant AqResidentQuantSelectionPass& pass [[buffer(3)]],
  uint index [[thread_position_in_grid]]) {

  if (index == 0u) {
    const uint rank = state[2];
    uint prefix_count = 0u;
    for (uint bucket = 0u; bucket < 256u; ++bucket) {
      const uint count = atomic_load_explicit(
        histogram + bucket, memory_order_relaxed);
      if (rank < prefix_count + count) {
        state[0] |= bucket << pass.shift;
        state[1] |= 255u << pass.shift;
        state[2] = rank - prefix_count;
        if (pass.shift == 0u) {
          statistics[pass.deviation != 0u ? 1u : 0u] =
            as_type<float>(state[0]);
        }
        break;
      }
      prefix_count += count;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (index < 256u) {
    atomic_store_explicit(
      histogram + index, 0u, memory_order_relaxed);
  }
}

kernel void gjxl_aq_resident_quant_finalize_quantizer(
  device const float* statistics [[buffer(0)]],
  device uint* quantizer_params [[buffer(1)]],
  device atomic_uint* error [[buffer(2)]],
  constant AqInitialQuantSelectionParams& params [[buffer(3)]],
  uint index [[thread_position_in_grid]]) {

  if (index != 0u) return;
  const float median_value = statistics[0];
  const float deviation = statistics[1];
  float scale = 65536.0f * (median_value - deviation) / 5.0f;
  scale = clamp(scale, 1.0f, 32768.0f);
  uint global_scale = uint(scale);
  if (global_scale > params.scaled_quant_dc) {
    global_scale = max(1u, params.scaled_quant_dc);
  }
  const float inverse_global_scale = 65536.0f / float(global_scale);
  const float quant_dc = min(
    65536.0f, params.quant_dc * inverse_global_scale + 0.5f);
  quantizer_params[0] = global_scale;
  quantizer_params[1] = uint(quant_dc);
  if (!isfinite(median_value) || median_value <= 0.0f ||
      !isfinite(deviation) || deviation < 0.0f || global_scale == 0u ||
      global_scale > 32768u || quantizer_params[1] == 0u ||
      quantizer_params[1] > 65536u) {
    atomic_fetch_or_explicit(error, 524288u, memory_order_relaxed);
  }
}

kernel void gjxl_aq_resident_policy_initialize(
  device const float* quant_field [[buffer(0)]],
  device float* initial_quant_field [[buffer(1)]],
  device float* scores [[buffer(2)]],
  device atomic_uint* error [[buffer(3)]],
  constant AqResidentPolicyInitializeParams& params [[buffer(4)]],
  uint index [[thread_position_in_grid]]) {

  constexpr uint kPoison = 0x7fc12345u;
  const uint block_count = params.block_width * params.block_height;
  if (index < block_count) {
    const uint y = index / params.block_width;
    const uint x = index - y * params.block_width;
    const float value = quant_field[y * params.quant_stride + x];
    if (!isfinite(value) || value <= 0.0f) {
      atomic_fetch_or_explicit(error, 1048576u, memory_order_relaxed);
    }
    initial_quant_field[y * params.initial_stride + x] = value;
  }
  if (index < params.score_count) {
    scores[index] = as_type<float>(kPoison);
  }
}

kernel void gjxl_aq_resident_policy_update(
  device float* quant_field [[buffer(0)]],
  device const float* initial_quant_field [[buffer(1)]],
  device const float* block_distance [[buffer(2)]],
  device const float* score [[buffer(3)]],
  device float* scores [[buffer(4)]],
  device const uint* quantizer_params [[buffer(5)]],
  device atomic_uint* error [[buffer(6)]],
  constant AqResidentPolicyUpdateParams& params [[buffer(7)]],
  uint index [[thread_position_in_grid]]) {

  const uint block_count = params.block_width * params.block_height;
  const uint global_scale = quantizer_params[0];
  if (index == 0u) {
    const float value = score[0];
    scores[params.score_index] = value;
    if (!isfinite(value) || value < 0.0f || global_scale == 0u ||
        global_scale > 32768u || quantizer_params[1] == 0u ||
        quantizer_params[1] > 65536u) {
      atomic_fetch_or_explicit(error, 1048576u, memory_order_relaxed);
    }
  }
  if (index >= block_count) return;
  const uint y = index / params.block_width;
  const uint x = index - y * params.block_width;
  const uint quant_index = y * params.quant_stride + x;
  float quant = quant_field[quant_index];
  const float initial =
    initial_quant_field[y * params.initial_stride + x];
  const float distance =
    block_distance[y * params.block_distance_stride + x];
  if (!isfinite(quant) || quant <= 0.0f ||
      !isfinite(initial) || initial <= 0.0f ||
      !isfinite(distance) || distance < 0.0f ||
      !isfinite(params.butteraugli_target) ||
      params.butteraugli_target <= 0.0f ||
      !isfinite(params.lower_bound) || params.lower_bound <= 0.0f ||
      !isfinite(params.upper_bound) ||
      params.upper_bound < params.lower_bound || global_scale == 0u) {
    atomic_fetch_or_explicit(error, 1048576u, memory_order_relaxed);
    return;
  }
  if (params.apply_update == 0u) return;

  if (params.iteration == 1u) {
    const float initial_clamp = 0.4f * quant + 0.6f * initial;
    if (quant < initial_clamp) {
      quant = clamp(
        initial_clamp, params.lower_bound, params.upper_bound);
    }
  }
  const float difference = distance / params.butteraugli_target;
  if (!isfinite(difference) || difference < 0.0f) {
    atomic_fetch_or_explicit(error, 1048576u, memory_order_relaxed);
    return;
  }
  if (difference <= 1.0f) {
    if (params.iteration < 2u) quant *= pow(difference, 0.2f);
  } else {
    const float old = quant;
    quant *= difference;
    const float inverse_global_scale = 65536.0f / float(global_scale);
    const float old_raw = floor(old * inverse_global_scale + 0.5f);
    const float new_raw = floor(quant * inverse_global_scale + 0.5f);
    if (old_raw == new_raw) {
      quant = old + float(global_scale) / 65536.0f;
    }
  }
  if (!isfinite(quant)) {
    atomic_fetch_or_explicit(error, 1048576u, memory_order_relaxed);
    return;
  }
  quant_field[quant_index] = clamp(
    quant, params.lower_bound, params.upper_bound);
}

kernel void gjxl_aq_reset_frame_encoding(
  device int* quantized_coefficients [[buffer(0)]],
  device int* quantized_dc [[buffer(1)]],
  device atomic_uint* error [[buffer(2)]],
  constant AqResetParams& params [[buffer(3)]],
  uint index [[thread_position_in_grid]]) {

  if (index == 0u && params.preserve_error == 0u) {
    atomic_store_explicit(
      error, params.test_error_mask, memory_order_relaxed);
  }
  if (index < params.coefficient_value_count) {
    quantized_coefficients[index] = int(0x81234567u);
  }
  if (index < params.dc_value_count) {
    quantized_dc[index] = int(0x81234567u);
  }
}

kernel void gjxl_aq_reset_exact_coefficients(
  device float* reconstructed_x [[buffer(0)]],
  device float* reconstructed_y [[buffer(1)]],
  device float* reconstructed_b [[buffer(2)]],
  device atomic_uint* error [[buffer(3)]],
  device float* block_distance [[buffer(4)]],
  constant AqResetParams& params [[buffer(5)]],
  uint index [[thread_position_in_grid]]) {

  constexpr uint kPoison = 0x7fc12345u;
  if (index == 0u && params.preserve_error == 0u) {
    atomic_store_explicit(
      error, params.test_error_mask, memory_order_relaxed);
  }
  if (index < params.pixel_value_count) {
    reconstructed_x[index] = as_type<float>(kPoison);
    reconstructed_y[index] = as_type<float>(kPoison);
    reconstructed_b[index] = as_type<float>(kPoison);
  }
  if (index < params.block_value_count) {
    block_distance[index] = as_type<float>(kPoison);
  }
}

kernel void gjxl_aq_gather_transform_pixels(
  device const float* coding_x [[buffer(0)]],
  device const float* coding_y [[buffer(1)]],
  device const float* coding_b [[buffer(2)]],
  device const uint2* anchors [[buffer(3)]],
  device float* gathered_pixels [[buffer(4)]],
  constant AqReconstructionParams& params [[buffer(5)]],
  uint index [[thread_position_in_grid]]) {

  const uint values_per_channel = params.anchor_count * params.coefficient_count;
  const uint value_count = 3u * values_per_channel;
  if (index >= value_count) return;
  const uint channel = index / values_per_channel;
  const uint channel_index = index - channel * values_per_channel;
  const uint anchor_index = channel_index / params.coefficient_count;
  const uint pixel_index = channel_index - anchor_index * params.coefficient_count;
  const uint x = pixel_index % params.pixel_width;
  const uint y = pixel_index / params.pixel_width;
  const uint2 anchor = anchors[params.anchor_offset + anchor_index];
  const uint source_index =
    (anchor.y * 8u + y) * params.coding_stride + anchor.x * 8u + x;
  const device float* source = channel == 0u
    ? coding_x
    : (channel == 1u ? coding_y : coding_b);
  gathered_pixels[params.coefficient_offset + index] = source[source_index];
}

kernel void gjxl_aq_encode_reconstruction_coefficients(
  device const uint2* anchors [[buffer(0)]],
  device const float* quant_tables [[buffer(1)]],
  device int* raw_quant [[buffer(2)]],
  device const char* y_to_x [[buffer(3)]],
  device const char* y_to_b [[buffer(4)]],
  device const float* forward_coefficients [[buffer(5)]],
  device int* quantized_coefficients [[buffer(6)]],
  device float* reconstruction_coefficients [[buffer(7)]],
  device float* dc [[buffer(8)]],
  device int* quantized_dc [[buffer(9)]],
  device atomic_uint* error [[buffer(10)]],
  constant AqReconstructionParams& params [[buffer(11)]],
  device float* inverse_sigma [[buffer(12)]],
  device const uchar* epf_sharpness [[buffer(13)]],
  device const uint* resident_quantizer [[buffer(14)]],
  uint anchor_index [[threadgroup_position_in_grid]],
  uint thread_index [[thread_index_in_threadgroup]],
  uint group_size [[threads_per_threadgroup]]) {

  if (anchor_index >= params.anchor_count) return;
  const uint thread_count = group_size;
  const uint global_scale = params.use_resident_quantizer != 0u
    ? resident_quantizer[0]
    : params.global_scale;
  const uint quant_dc = params.use_resident_quantizer != 0u
    ? resident_quantizer[1]
    : params.quant_dc;
  const uint2 anchor = anchors[params.anchor_offset + anchor_index];
  const uint block_count = params.block_width * params.block_height;
  const uint covered_count = params.covered_width * params.covered_height;
  const uint group_channel_stride = params.anchor_count * params.coefficient_count;
  const uint transform_offset =
    params.coefficient_offset + anchor_index * params.coefficient_count;
  const uint2 table_offsets = aq_quant_table_offsets(params.strategy);
  const uint coefficient_width = max(params.pixel_width, params.pixel_height);
  const uint coefficient_height = min(params.pixel_width, params.pixel_height);
  const uint raw_index = anchor.y * params.raw_quant_stride + anchor.x;
  const int initial_raw = raw_quant[raw_index];
  const uint color_index =
    (anchor.y / 8u) * params.color_stride + anchor.x / 8u;
  const float cfl_x = float(y_to_x[color_index]) * (1.0f / 84.0f);
  const float cfl_b = 1.0f + float(y_to_b[color_index]) * (1.0f / 84.0f);
  threadgroup int selected_raw = 0;
  threadgroup float selected_y_thresholds[4] = {};
  if (thread_index == 0u) {
    if (params.adjust_ac_quant != 0u) {
      const AqAdjustedQuantization decision = aq_select_adjusted_quantization(
        forward_coefficients + transform_offset, quant_tables,
        params.coefficient_count, group_channel_stride,
        coefficient_width, coefficient_height, params.strategy,
        global_scale, initial_raw, params.x_matrix_multiplier,
        params.b_matrix_multiplier, error);
      selected_raw = decision.raw_quant;
      for (uint quadrant = 0u; quadrant < 4u; ++quadrant) {
        selected_y_thresholds[quadrant] = decision.y_thresholds[quadrant];
      }
      raw_quant[raw_index] = decision.raw_quant;
    } else {
      selected_raw = initial_raw;
      selected_y_thresholds[0] = 0.58f;
      selected_y_thresholds[1] = 0.64f;
      selected_y_thresholds[2] = 0.64f;
      selected_y_thresholds[3] = 0.64f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int raw = selected_raw;

  if (params.adjust_ac_quant != 0u) {
    constexpr float kInverseSigmaNumerator = -1.1715728752538099024f;
    const float quantizer_scale =
      float(global_scale) * (1.0f / 65536.0f);
    const float sigma_quant = params.epf_quant_multiplier /
      (quantizer_scale * float(raw) * kInverseSigmaNumerator);
    for (uint block = thread_index; block < covered_count;
         block += thread_count) {
      const uint x = block % params.covered_width;
      const uint y = block / params.covered_width;
      const uint sharpness_index =
        (anchor.y + y) * params.epf_sharpness_stride + anchor.x + x;
      const uint sharpness = uint(epf_sharpness[sharpness_index]);
      if (sharpness >= 8u) {
        atomic_fetch_or_explicit(error, 64u, memory_order_relaxed);
        continue;
      }
      float sigma = sigma_quant * params.epf_sharpness_lut[sharpness];
      sigma = min(-1.0e-4f, sigma);
      const float value = 1.0f / sigma;
      if (!isfinite(value) || value >= 0.0f) {
        atomic_fetch_or_explicit(error, 128u, memory_order_relaxed);
        continue;
      }
      inverse_sigma[
        (anchor.y + y) * params.inverse_sigma_stride + anchor.x + x] = value;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  // DC is extracted from the preserved forward coefficients. The LLF portion
  // of dequantized Y is exactly zero, so this also equals post-CfL X/B DC.
  for (uint dc_task = thread_index;
       dc_task < 3u * covered_count;
       dc_task += thread_count) {
    const uint channel = dc_task / covered_count;
    const uint small = dc_task - channel * covered_count;
    const uint x = small % params.covered_width;
    const uint y = small / params.covered_width;
    float value = 0.0f;
    for (uint v = 0u; v < params.covered_height; ++v) {
      for (uint u = 0u; u < params.covered_width; ++u) {
        const uint coefficient = aq_coefficient_index(params, v, u);
        const float scaled = forward_coefficients[
          transform_offset + channel * group_channel_stride + coefficient] *
          aq_downsample_scale(params.covered_height, v) *
          aq_downsample_scale(params.covered_width, u);
        value += scaled *
          aq_inverse_basis(params.covered_height, v, y) *
          aq_inverse_basis(params.covered_width, u, x);
      }
    }
    if (!isfinite(value)) {
      atomic_fetch_or_explicit(error, 8u, memory_order_relaxed);
      value = 0.0f;
    }
    dc[channel * block_count +
      (anchor.y + y) * params.block_width + anchor.x + x] = value;
  }
  threadgroup_barrier(mem_flags::mem_device);

  // Mirror the simple codestream's modular DC quantization. Y is rounded and
  // reconstructed first; B then predicts that reconstructed Y with the
  // default DC CfL factor of one, while X has no DC prediction.
  for (uint small = thread_index; small < covered_count;
       small += thread_count) {
    const uint x = small % params.covered_width;
    const uint y = small / params.covered_width;
    const uint block_index =
      (anchor.y + y) * params.block_width + anchor.x + x;
    const float quant_scale =
      (float(global_scale) / 65536.0f) * float(quant_dc);
    const float inverse_x = 4096.0f * quant_scale;
    const float inverse_y = 512.0f * quant_scale;
    const float inverse_b = 256.0f * quant_scale;
    const int quantized_y = aq_round_dc(
      dc[block_count + block_index] * inverse_y, error);
    const float reconstructed_y = float(quantized_y) / inverse_y;
    const int quantized_x = aq_round_dc(
      dc[block_index] * inverse_x, error);
    const int quantized_b = aq_round_dc(
      (dc[2u * block_count + block_index] - reconstructed_y) * inverse_b,
      error);
    quantized_dc[block_index] = quantized_x;
    quantized_dc[block_count + block_index] = quantized_y;
    quantized_dc[2u * block_count + block_index] = quantized_b;
    dc[block_index] = float(quantized_x) / inverse_x;
    dc[block_count + block_index] = reconstructed_y;
    dc[2u * block_count + block_index] =
      float(quantized_b) / inverse_b + reconstructed_y;
  }
  threadgroup_barrier(mem_flags::mem_device);

  // Quantize/dequantize Y first because X/B prediction consumes rounded Y.
  for (uint coefficient = thread_index;
       coefficient < params.coefficient_count;
       coefficient += thread_count) {
    const uint channel = 1u;
    const uint offset = transform_offset + channel * group_channel_stride + coefficient;
    const uint x = coefficient % coefficient_width;
    const uint y = coefficient / coefficient_width;
    const uint quadrant = uint(y >= coefficient_height / 2u) * 2u +
      uint(x >= coefficient_width / 2u);
    const float threshold = params.adjust_ac_quant != 0u
      ? selected_y_thresholds[quadrant]
      : aq_quantization_threshold(
          channel, covered_count, coefficient,
          coefficient_width, coefficient_height);
    const uint table = channel * params.coefficient_count + coefficient;
    const int quantized = aq_quantize_coefficient(
      forward_coefficients[offset],
      quant_tables[table_offsets.y + table],
      global_scale,
      raw,
      1.0f,
      threshold,
      error);
    quantized_coefficients[offset] = quantized;
    reconstruction_coefficients[offset] = aq_dequantize_coefficient(
      quantized,
      quant_tables[table_offsets.x + table],
      global_scale,
      raw,
      1.0f,
      channel,
      error);
  }
  threadgroup_barrier(mem_flags::mem_device);

  for (uint coefficient = thread_index;
       coefficient < params.coefficient_count;
       coefficient += thread_count) {
    const float reconstructed_y = reconstruction_coefficients[
      transform_offset + group_channel_stride + coefficient];
    for (uint channel : {0u, 2u}) {
      const uint offset = transform_offset + channel * group_channel_stride + coefficient;
      const float factor = channel == 0u ? cfl_x : cfl_b;
      const float multiplier = channel == 0u
        ? params.x_matrix_multiplier
        : params.b_matrix_multiplier;
      const float predicted = forward_coefficients[offset] - factor * reconstructed_y;
      const float threshold = aq_quantization_threshold(
        channel, covered_count, coefficient, coefficient_width, coefficient_height);
      const uint table = channel * params.coefficient_count + coefficient;
      const int quantized = aq_quantize_coefficient(
        predicted,
        quant_tables[table_offsets.y + table],
        global_scale,
        raw,
        multiplier,
        threshold,
        error);
      quantized_coefficients[offset] = quantized;
      reconstruction_coefficients[offset] = aq_dequantize_coefficient(
        quantized,
        quant_tables[table_offsets.x + table],
        global_scale,
        raw,
        multiplier,
        channel,
        error);
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  for (uint coefficient = thread_index;
       coefficient < params.coefficient_count;
       coefficient += thread_count) {
    const float reconstructed_y = reconstruction_coefficients[
      transform_offset + group_channel_stride + coefficient];
    reconstruction_coefficients[transform_offset + coefficient] += cfl_x * reconstructed_y;
    reconstruction_coefficients[
      transform_offset + 2u * group_channel_stride + coefficient] +=
        cfl_b * reconstructed_y;
  }
  threadgroup_barrier(mem_flags::mem_device);

  for (uint llf_task = thread_index;
       llf_task < 3u * covered_count;
       llf_task += thread_count) {
    const uint channel = llf_task / covered_count;
    const uint small = llf_task - channel * covered_count;
    const uint u = small % params.covered_width;
    const uint v = small / params.covered_width;
    float value = 0.0f;
    for (uint y = 0u; y < params.covered_height; ++y) {
      for (uint x = 0u; x < params.covered_width; ++x) {
        value += dc[channel * block_count +
          (anchor.y + y) * params.block_width + anchor.x + x] *
          aq_forward_basis(params.covered_height, v, y) *
          aq_forward_basis(params.covered_width, u, x);
      }
    }
    value *= aq_upsample_scale(params.covered_height, v) *
      aq_upsample_scale(params.covered_width, u);
    reconstruction_coefficients[
      transform_offset + channel * group_channel_stride +
      aq_coefficient_index(params, v, u)] = value;
  }
}

kernel void gjxl_aq_encode_frame_coefficients(
  device const uint2* anchors [[buffer(0)]],
  device const float* quant_tables [[buffer(1)]],
  device int* raw_quant [[buffer(2)]],
  device const char* y_to_x [[buffer(3)]],
  device const char* y_to_b [[buffer(4)]],
  device const float* forward_coefficients [[buffer(5)]],
  device int* quantized_coefficients [[buffer(6)]],
  device int* quantized_dc [[buffer(7)]],
  device atomic_uint* error [[buffer(8)]],
  constant AqReconstructionParams& params [[buffer(9)]],
  uint anchor_index [[threadgroup_position_in_grid]],
  uint thread_index [[thread_index_in_threadgroup]]) {

  if (anchor_index >= params.anchor_count) return;
  threadgroup float dc[3 * 16];
  threadgroup float reconstructed_y[32 * 32];
  const uint2 anchor = anchors[params.anchor_offset + anchor_index];
  const uint block_count = params.block_width * params.block_height;
  const uint covered_count = params.covered_width * params.covered_height;
  const uint group_channel_stride = params.anchor_count * params.coefficient_count;
  const uint transform_offset =
    params.coefficient_offset + anchor_index * params.coefficient_count;
  const uint2 table_offsets = aq_quant_table_offsets(params.strategy);
  const uint coefficient_width = max(params.pixel_width, params.pixel_height);
  const uint coefficient_height = min(params.pixel_width, params.pixel_height);
  const uint raw_index = anchor.y * params.raw_quant_stride + anchor.x;
  const int initial_raw = raw_quant[raw_index];
  const uint color_index =
    (anchor.y / 8u) * params.color_stride + anchor.x / 8u;
  const float cfl_x = float(y_to_x[color_index]) * (1.0f / 84.0f);
  const float cfl_b = 1.0f + float(y_to_b[color_index]) * (1.0f / 84.0f);
  threadgroup int selected_raw = 0;
  threadgroup float selected_y_thresholds[4] = {};
  if (thread_index == 0u) {
    if (params.adjust_ac_quant != 0u) {
      const AqAdjustedQuantization decision = aq_select_adjusted_quantization(
        forward_coefficients + transform_offset, quant_tables,
        params.coefficient_count, group_channel_stride,
        coefficient_width, coefficient_height, params.strategy,
        params.global_scale, initial_raw, params.x_matrix_multiplier,
        params.b_matrix_multiplier, error);
      selected_raw = decision.raw_quant;
      for (uint quadrant = 0u; quadrant < 4u; ++quadrant) {
        selected_y_thresholds[quadrant] = decision.y_thresholds[quadrant];
      }
      raw_quant[raw_index] = decision.raw_quant;
    } else {
      selected_raw = initial_raw;
      selected_y_thresholds[0] = 0.58f;
      selected_y_thresholds[1] = 0.64f;
      selected_y_thresholds[2] = 0.64f;
      selected_y_thresholds[3] = 0.64f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int raw = selected_raw;

  for (uint dc_task = thread_index;
       dc_task < 3u * covered_count;
       dc_task += 256u) {
    const uint channel = dc_task / covered_count;
    const uint small = dc_task - channel * covered_count;
    const uint x = small % params.covered_width;
    const uint y = small / params.covered_width;
    float value = 0.0f;
    for (uint v = 0u; v < params.covered_height; ++v) {
      for (uint u = 0u; u < params.covered_width; ++u) {
        const uint coefficient = aq_coefficient_index(params, v, u);
        const float scaled = forward_coefficients[
          transform_offset + channel * group_channel_stride + coefficient] *
          aq_downsample_scale(params.covered_height, v) *
          aq_downsample_scale(params.covered_width, u);
        value += scaled *
          aq_inverse_basis(params.covered_height, v, y) *
          aq_inverse_basis(params.covered_width, u, x);
      }
    }
    if (!isfinite(value)) {
      atomic_fetch_or_explicit(error, 8u, memory_order_relaxed);
      value = 0.0f;
    }
    dc[dc_task] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint small = thread_index; small < covered_count; small += 256u) {
    const uint x = small % params.covered_width;
    const uint y = small / params.covered_width;
    const uint block_index =
      (anchor.y + y) * params.block_width + anchor.x + x;
    const float quant_scale =
      (float(params.global_scale) / 65536.0f) * float(params.quant_dc);
    const float inverse_x = 4096.0f * quant_scale;
    const float inverse_y = 512.0f * quant_scale;
    const float inverse_b = 256.0f * quant_scale;
    const int quantized_y = aq_round_dc(
      dc[covered_count + small] * inverse_y, error);
    const float reconstructed_dc_y = float(quantized_y) / inverse_y;
    const int quantized_x = aq_round_dc(
      dc[small] * inverse_x, error);
    const int quantized_b = aq_round_dc(
      (dc[2u * covered_count + small] - reconstructed_dc_y) * inverse_b,
      error);
    quantized_dc[block_index] = quantized_x;
    quantized_dc[block_count + block_index] = quantized_y;
    quantized_dc[2u * block_count + block_index] = quantized_b;
  }

  for (uint coefficient = thread_index;
       coefficient < params.coefficient_count;
       coefficient += 256u) {
    const uint channel = 1u;
    const uint offset =
      transform_offset + channel * group_channel_stride + coefficient;
    const uint x = coefficient % coefficient_width;
    const uint y = coefficient / coefficient_width;
    const uint quadrant = uint(y >= coefficient_height / 2u) * 2u +
      uint(x >= coefficient_width / 2u);
    const float threshold = params.adjust_ac_quant != 0u
      ? selected_y_thresholds[quadrant]
      : aq_quantization_threshold(
          channel, covered_count, coefficient, coefficient_width,
          coefficient_height);
    const uint table = channel * params.coefficient_count + coefficient;
    const int quantized = aq_quantize_coefficient(
      forward_coefficients[offset],
      quant_tables[table_offsets.y + table],
      params.global_scale,
      raw,
      1.0f,
      threshold,
      error);
    quantized_coefficients[offset] = quantized;
    reconstructed_y[coefficient] = aq_dequantize_coefficient(
      quantized,
      quant_tables[table_offsets.x + table],
      params.global_scale,
      raw,
      1.0f,
      channel,
      error);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint coefficient = thread_index;
       coefficient < params.coefficient_count;
       coefficient += 256u) {
    for (uint channel : {0u, 2u}) {
      const uint offset =
        transform_offset + channel * group_channel_stride + coefficient;
      const float factor = channel == 0u ? cfl_x : cfl_b;
      const float multiplier = channel == 0u
        ? params.x_matrix_multiplier
        : params.b_matrix_multiplier;
      const float predicted =
        forward_coefficients[offset] - factor * reconstructed_y[coefficient];
      const float threshold = aq_quantization_threshold(
        channel, covered_count, coefficient, coefficient_width,
        coefficient_height);
      const uint table = channel * params.coefficient_count + coefficient;
      quantized_coefficients[offset] = aq_quantize_coefficient(
        predicted,
        quant_tables[table_offsets.y + table],
        params.global_scale,
        raw,
        multiplier,
        threshold,
        error);
    }
  }
}

kernel void gjxl_aq_scatter_reconstructed_pixels(
  device const uint2* anchors [[buffer(0)]],
  device const float* reconstructed_pixels [[buffer(1)]],
  device float* reconstructed_x [[buffer(2)]],
  device float* reconstructed_y [[buffer(3)]],
  device float* reconstructed_b [[buffer(4)]],
  constant AqReconstructionParams& params [[buffer(5)]],
  uint index [[thread_position_in_grid]]) {

  const uint values_per_channel = params.anchor_count * params.coefficient_count;
  const uint value_count = 3u * values_per_channel;
  if (index >= value_count) return;
  const uint channel = index / values_per_channel;
  const uint channel_index = index - channel * values_per_channel;
  const uint anchor_index = channel_index / params.coefficient_count;
  const uint pixel_index = channel_index - anchor_index * params.coefficient_count;
  const uint x = pixel_index % params.pixel_width;
  const uint y = pixel_index / params.pixel_width;
  const uint2 anchor = anchors[params.anchor_offset + anchor_index];
  const uint destination_index =
    (anchor.y * 8u + y) * params.coding_stride + anchor.x * 8u + x;
  device float* destination = channel == 0u
    ? reconstructed_x
    : (channel == 1u ? reconstructed_y : reconstructed_b);
  destination[destination_index] = reconstructed_pixels[
    params.coefficient_offset + index];
}

kernel void gjxl_aq_quantization_probe(
  device const float* coefficients [[buffer(0)]],
  device const float* quant_tables [[buffer(1)]],
  device int* quantized [[buffer(2)]],
  device float* dequantized [[buffer(3)]],
  device atomic_uint* error [[buffer(4)]],
  constant AqQuantizationProbeParams& params [[buffer(5)]],
  uint index [[thread_position_in_grid]]) {

  if (index >= params.coefficient_count) return;
  const uint2 table_offsets = aq_quant_table_offsets(params.strategy);
  const uint coefficient_width = params.strategy == 0u
    ? 8u
    : (params.strategy == 4u ? 16u :
       (params.strategy == 5u ? 32u :
        ((params.strategy == 6u || params.strategy == 7u) ? 16u : 32u)));
  const uint coefficient_height = params.coefficient_count / coefficient_width;
  const uint covered_count = params.coefficient_count / 64u;
  const uint table = params.channel * params.coefficient_count + index;
  const float threshold = aq_quantization_threshold(
    params.channel,
    covered_count,
    index,
    coefficient_width,
    coefficient_height);
  const int value = aq_quantize_coefficient(
    coefficients[index],
    quant_tables[table_offsets.y + table],
    params.global_scale,
    params.raw_quant,
    params.matrix_multiplier,
    threshold,
    error);
  quantized[index] = value;
  dequantized[index] = aq_dequantize_coefficient(
    value,
    quant_tables[table_offsets.x + table],
    params.global_scale,
    params.raw_quant,
    params.matrix_multiplier,
    params.channel,
    error);
}

kernel void gjxl_aq_adjustment_probe(
  device const float* coefficients [[buffer(0)]],
  device const float* quant_tables [[buffer(1)]],
  device int* quantized_y [[buffer(2)]],
  device int* adjusted_raw_quant [[buffer(3)]],
  device float* adjusted_y_thresholds [[buffer(4)]],
  device atomic_uint* error [[buffer(5)]],
  constant AqAdjustmentProbeParams& params [[buffer(6)]],
  uint index [[thread_position_in_grid]]) {

  if (index != 0u) return;
  const AqAdjustedQuantization decision = aq_select_adjusted_quantization(
    coefficients, quant_tables, params.coefficient_count,
    params.coefficient_count,
    params.coefficient_width, params.coefficient_height, params.strategy,
    params.global_scale, params.initial_raw_quant,
    params.x_matrix_multiplier, params.b_matrix_multiplier, error);
  adjusted_raw_quant[0] = decision.raw_quant;
  for (uint quadrant = 0u; quadrant < 4u; ++quadrant) {
    adjusted_y_thresholds[quadrant] = decision.y_thresholds[quadrant];
  }

  const uint2 table_offsets = aq_quant_table_offsets(params.strategy);
  for (uint coefficient = 0u;
       coefficient < params.coefficient_count; ++coefficient) {
    const uint x = coefficient % params.coefficient_width;
    const uint y = coefficient / params.coefficient_width;
    const uint quadrant = uint(y >= params.coefficient_height / 2u) * 2u +
      uint(x >= params.coefficient_width / 2u);
    const uint table = params.coefficient_count + coefficient;
    quantized_y[coefficient] = aq_quantize_coefficient(
      coefficients[params.coefficient_count + coefficient],
      quant_tables[table_offsets.y + table], params.global_scale,
      decision.raw_quant, 1.0f, decision.y_thresholds[quadrant], error);
  }
}
