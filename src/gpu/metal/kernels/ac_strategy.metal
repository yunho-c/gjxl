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
  uint opsin_plane_stride;
  uint pixel_mask_row_stride;
  uint candidate_count;
  uint coefficient_count;
  uint transform_width;
  uint transform_height;
  uint covered_block_count;
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

kernel void gjxl_ac_strategy_gather(
  device const float* opsin [[buffer(0)]],
  device const AcStrategyCandidate* candidates [[buffer(1)]],
  device float* packed_pixels [[buffer(2)]],
  constant AcStrategyBatchParams& params [[buffer(3)]],
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

  const uint source_index =
    channel * params.opsin_plane_stride +
    pixel_y * params.opsin_row_stride +
    pixel_x;
  packed_pixels[index] = opsin[source_index];
}

kernel void gjxl_ac_strategy_residual(
  device const float* coefficients [[buffer(0)]],
  device const float* matrices [[buffer(1)]],
  device const AcStrategyCandidate* candidates [[buffer(2)]],
  device float* residual_coefficients [[buffer(3)]],
  device ChannelRate* channel_rates [[buffer(4)]],
  constant AcStrategyBatchParams& params [[buffer(5)]],
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
  const float cfl_factor =
    channel == 0 ? candidate.cfl_x :
    channel == 2 ? candidate.cfl_b : 0.0f;

  const float decorrelated =
    coefficients[base + tid] - coefficients[y_base + tid] * cfl_factor;
  const float scaled =
    decorrelated * matrices[inverse_matrix_base + tid] *
    candidate.quant_norm;
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
  constant AcStrategyBatchParams& params [[buffer(5)]],
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
    const float normalized_loss = loss / float(params.coefficient_count);
    const float loss_cost =
      powr(normalized_loss, 0.125f) * float(params.coefficient_count) /
      candidate.quant_norm;
    const float result =
      entropy * candidate.entropy_multiplier +
      params.info_loss_multiplier * loss_cost;
    costs[candidate_index] =
      isfinite(result) && result >= 0.0f ? result : NAN;
  }
}
