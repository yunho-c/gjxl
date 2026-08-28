// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <metal_stdlib>

using namespace metal;

struct AcStrategyCandidate {
  uint block_x;
  uint block_y;
  float quant_norm;
  float entropy_multiplier;
  float cfl_x;
  float cfl_b;
};

struct AcStrategyBatchParams {
  uint pixel_width;
  uint pixel_height;
  uint opsin_row_stride;
  uint pixel_mask_row_stride;
  uint quant_field_row_stride;
  uint candidate_count;
  uint coefficient_count;
  uint transform_width;
  uint transform_height;
  uint covered_block_width;
  uint covered_block_height;
  uint covered_block_count;
  uint use_device_quant_norm;
  float info_loss_multiplier;
  float zeros_multiplier;
  float cost_delta;
};

struct ChannelRate {
  float magnitude;
  uint nonzero_count;
};

constant float kMaskOffset[3] = {12.0f, 0.0f, 4.0f};
constant float kChannelMultiplier[3] = {
  2.0441408586549744e7f,
  1.0f,
  1.266770081387616f,
};

inline uint CeilLog2Nonzero(uint value) {
  return value <= 1 ? 0 : 32 - clz(value - 1);
}

inline float RoundAwayFromZero(float value) {
  return copysign(floor(abs(value) + 0.5f), value);
}

inline float FastLog2(float value) {
  const uint value_bits = as_type<uint>(value);
  const int shifted_exponent = int(value_bits - 0x3f2aaaabu) >> 23;
  const uint mantissa_bits =
    value_bits - (uint(shifted_exponent) << 23);
  const float x = as_type<float>(mantissa_bits) - 1.0f;
  float numerator = fma(0.74245873327820566f, x, 1.4287160470083755f);
  numerator = fma(numerator, x, -1.8503833400518310e-06f);
  float denominator = fma(0.17409343003366853f, x, 1.0096718572241148f);
  denominator = fma(denominator, x, 0.99032814277590719f);
  return numerator / denominator + float(shifted_exponent);
}

inline float FastPow2(float value) {
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

inline float ComputeQuantNorm(
  device const float* quant_field,
  AcStrategyCandidate candidate,
  constant AcStrategyBatchParams& params) {

  if (params.use_device_quant_norm == 0u) return candidate.quant_norm;
  if (params.covered_block_count == 1u) {
    return quant_field[
      candidate.block_y * params.quant_field_row_stride + candidate.block_x];
  }
  if (params.covered_block_count == 2u) {
    const float first = quant_field[
      candidate.block_y * params.quant_field_row_stride + candidate.block_x];
    const uint second_x = candidate.block_x +
      (params.covered_block_width == 2u ? 1u : 0u);
    const uint second_y = candidate.block_y +
      (params.covered_block_height == 2u ? 1u : 0u);
    const float second = quant_field[
      second_y * params.quant_field_row_stride + second_x];
    return max(first, second);
  }
  float sum = 0.0f;
  for (uint dy = 0; dy < params.covered_block_height; ++dy) {
    for (uint dx = 0; dx < params.covered_block_width; ++dx) {
      float value = quant_field[
        (candidate.block_y + dy) * params.quant_field_row_stride +
        candidate.block_x + dx];
      value *= value;
      value *= value;
      value *= value;
      sum += value * value;
    }
  }
  sum /= float(params.covered_block_count);
  return FastPow2(FastLog2(sum) * (1.0f / 16.0f));
}

kernel void gjxl_ac_strategy_gather(
  device const float* opsin_x [[buffer(0)]],
  device const float* opsin_y [[buffer(1)]],
  device const float* opsin_b [[buffer(2)]],
  device const AcStrategyCandidate* candidates [[buffer(3)]],
  device float* packed_pixels [[buffer(4)]],
  constant AcStrategyBatchParams& params [[buffer(5)]],
  uint index [[thread_position_in_grid]]) {

  const uint channel_stride = params.coefficient_count;
  const uint candidate_stride = 3 * channel_stride;
  const uint element_count = params.candidate_count * candidate_stride;
  if (index >= element_count) {
    return;
  }

  const uint candidate_index = index / candidate_stride;
  const uint candidate_element = index % candidate_stride;
  const uint channel = candidate_element / channel_stride;
  const uint element = candidate_element % channel_stride;
  const uint row = element / params.transform_width;
  const uint column = element % params.transform_width;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const bool candidate_valid =
    candidate.block_x <=
      (params.pixel_width - params.transform_width) / 8 &&
    candidate.block_y <=
      (params.pixel_height - params.transform_height) / 8 &&
    isfinite(candidate.quant_norm) && candidate.quant_norm > 0.0f &&
    isfinite(candidate.entropy_multiplier) &&
    candidate.entropy_multiplier > 0.0f &&
    isfinite(candidate.cfl_x) && isfinite(candidate.cfl_b);
  if (!candidate_valid) {
    packed_pixels[index] = NAN;
    return;
  }
  const uint pixel_x = candidate.block_x * 8 + column;
  const uint pixel_y = candidate.block_y * 8 + row;

  const uint source_index = pixel_y * params.opsin_row_stride + pixel_x;
  packed_pixels[index] = channel == 0u ? opsin_x[source_index] :
    channel == 1u ? opsin_y[source_index] : opsin_b[source_index];
}

kernel void gjxl_ac_strategy_residual(
  device const float* coefficients [[buffer(0)]],
  device const float* matrices [[buffer(1)]],
  device const AcStrategyCandidate* candidates [[buffer(2)]],
  device const float* quant_field [[buffer(3)]],
  device float* residual_coefficients [[buffer(4)]],
  device ChannelRate* channel_rates [[buffer(5)]],
  constant AcStrategyBatchParams& params [[buffer(6)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 group_position [[threadgroup_position_in_grid]]) {

  const uint transform_index = group_position.x;
  const uint candidate_index = transform_index / 3;
  const uint channel = transform_index % 3;
  const uint base = transform_index * params.coefficient_count;
  const uint y_base =
    (candidate_index * 3 + 1) * params.coefficient_count;
  const uint matrix_base = channel * params.coefficient_count;
  const uint inverse_matrix_base =
    (3 + channel) * params.coefficient_count;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const float quant_norm = ComputeQuantNorm(quant_field, candidate, params);
  const float cfl_factor =
    channel == 0 ? candidate.cfl_x :
    channel == 2 ? candidate.cfl_b : 0.0f;

  const float decorrelated =
    coefficients[base + tid] - coefficients[y_base + tid] * cfl_factor;
  const float scaled =
    decorrelated * matrices[inverse_matrix_base + tid] *
    quant_norm;
  const float rounded = RoundAwayFromZero(scaled);
  residual_coefficients[base + tid] =
    matrices[matrix_base + tid] * (scaled - rounded);

  threadgroup float magnitude_reduction[1024];
  threadgroup uint nonzero_reduction[1024];
  magnitude_reduction[tid] = sqrt(abs(rounded));
  nonzero_reduction[tid] = rounded != 0.0f ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = params.coefficient_count / 2;
       stride != 0;
       stride /= 2) {
    if (tid < stride) {
      magnitude_reduction[tid] += magnitude_reduction[tid + stride];
      nonzero_reduction[tid] += nonzero_reduction[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    channel_rates[transform_index] = {
      magnitude_reduction[0],
      nonzero_reduction[0],
    };
  }
}

kernel void gjxl_ac_strategy_cost(
  device const float* residual_pixels [[buffer(0)]],
  device const float* pixel_mask [[buffer(1)]],
  device const AcStrategyCandidate* candidates [[buffer(2)]],
  device const ChannelRate* channel_rates [[buffer(3)]],
  device float* costs [[buffer(4)]],
  device const float* quant_field [[buffer(5)]],
  constant AcStrategyBatchParams& params [[buffer(6)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 group_position [[threadgroup_position_in_grid]]) {

  const uint candidate_index = group_position.x;
  const AcStrategyCandidate candidate = candidates[candidate_index];
  const uint row = tid / params.transform_width;
  const uint column = tid % params.transform_width;
  const bool candidate_fits =
    candidate.block_x <=
      (params.pixel_width - params.transform_width) / 8 &&
    candidate.block_y <=
      (params.pixel_height - params.transform_height) / 8;
  if (!candidate_fits) {
    if (tid == 0) {
      costs[candidate_index] = NAN;
    }
    return;
  }
  const uint pixel_x = candidate.block_x * 8 + column;
  const uint pixel_y = candidate.block_y * 8 + row;
  const uint mask_index =
    pixel_y * params.pixel_mask_row_stride + pixel_x;
  const float mask = pixel_mask[mask_index];

  threadgroup float loss_reduction[1024];
  float entropy = 0.0f;
  float loss = 0.0f;

  for (uint channel = 0; channel < 3; ++channel) {
    const uint transform_index = candidate_index * 3 + channel;
    const uint base = transform_index * params.coefficient_count;
    float weighted =
      (mask + kMaskOffset[channel]) * residual_pixels[base + tid];
    weighted *= weighted;
    weighted *= weighted;
    weighted *= weighted;
    loss_reduction[tid] =
      isfinite(mask) && mask > 0.0f ? weighted : NAN;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.coefficient_count / 2;
         stride != 0;
         stride /= 2) {
      if (tid < stride) {
        loss_reduction[tid] += loss_reduction[tid + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
      const ChannelRate rate = channel_rates[transform_index];
      entropy += params.cost_delta * rate.magnitude;
      const uint nonzero_bits = CeilLog2Nonzero(rate.nonzero_count + 1) + 1;
      entropy += params.zeros_multiplier * float(
        CeilLog2Nonzero(nonzero_bits + 17) + nonzero_bits);
      loss += loss_reduction[0] * kChannelMultiplier[channel];

      if (channel == 0 && params.covered_block_count >= 2) {
        const float weight = 1.0f + min(
          3.0f,
          float(params.covered_block_count) / 8.0f);
        entropy *= weight;
        loss *= weight;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    const float quant_norm = ComputeQuantNorm(
      quant_field, candidate, params);
    const float normalized_loss = loss / float(params.coefficient_count);
    const float loss_cost =
      powr(normalized_loss, 0.125f) * float(params.coefficient_count) /
      quant_norm;
    const float result =
      entropy * candidate.entropy_multiplier +
      params.info_loss_multiplier * loss_cost;
    costs[candidate_index] =
      isfinite(result) && result >= 0.0f ? result : NAN;
  }
}
