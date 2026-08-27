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
};

struct AqResetParams {
  uint coefficient_value_count;
  uint dc_value_count;
  uint pixel_value_count;
  uint block_value_count;
  uint test_error_mask;
};

struct AqQuantizationProbeParams {
  uint coefficient_count;
  uint strategy;
  uint channel;
  int raw_quant;
  uint global_scale;
  float matrix_multiplier;
};

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
  if (index == 0u) {
    atomic_store_explicit(
      error, params.test_error_mask, memory_order_relaxed);
  }
  if (index < params.coefficient_value_count) {
    gathered_pixels[index] = as_type<float>(kPoison);
    forward_coefficients[index] = as_type<float>(kPoison);
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
  device const int* raw_quant [[buffer(2)]],
  device const char* y_to_x [[buffer(3)]],
  device const char* y_to_b [[buffer(4)]],
  device const float* forward_coefficients [[buffer(5)]],
  device int* quantized_coefficients [[buffer(6)]],
  device float* reconstruction_coefficients [[buffer(7)]],
  device float* dc [[buffer(8)]],
  device int* quantized_dc [[buffer(9)]],
  device atomic_uint* error [[buffer(10)]],
  constant AqReconstructionParams& params [[buffer(11)]],
  uint anchor_index [[threadgroup_position_in_grid]],
  uint thread_index [[thread_index_in_threadgroup]]) {

  if (anchor_index >= params.anchor_count) return;
  const uint2 anchor = anchors[params.anchor_offset + anchor_index];
  const uint block_count = params.block_width * params.block_height;
  const uint covered_count = params.covered_width * params.covered_height;
  const uint group_channel_stride = params.anchor_count * params.coefficient_count;
  const uint transform_offset =
    params.coefficient_offset + anchor_index * params.coefficient_count;
  const uint2 table_offsets = aq_quant_table_offsets(params.strategy);
  const uint coefficient_width = max(params.pixel_width, params.pixel_height);
  const uint coefficient_height = min(params.pixel_width, params.pixel_height);
  const int raw = raw_quant[anchor.y * params.raw_quant_stride + anchor.x];
  const uint color_index =
    (anchor.y / 8u) * params.color_stride + anchor.x / 8u;
  const float cfl_x = float(y_to_x[color_index]) * (1.0f / 84.0f);
  const float cfl_b = 1.0f + float(y_to_b[color_index]) * (1.0f / 84.0f);

  // DC is extracted from the preserved forward coefficients. The LLF portion
  // of dequantized Y is exactly zero, so this also equals post-CfL X/B DC.
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
    dc[channel * block_count +
      (anchor.y + y) * params.block_width + anchor.x + x] = value;
  }
  threadgroup_barrier(mem_flags::mem_device);

  // Mirror the simple codestream's modular DC quantization. Y is rounded and
  // reconstructed first; B then predicts that reconstructed Y with the
  // default DC CfL factor of one, while X has no DC prediction.
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
       coefficient += 256u) {
    const uint channel = 1u;
    const uint offset = transform_offset + channel * group_channel_stride + coefficient;
    const float threshold = aq_quantization_threshold(
      channel, covered_count, coefficient, coefficient_width, coefficient_height);
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
    reconstruction_coefficients[offset] = aq_dequantize_coefficient(
      quantized,
      quant_tables[table_offsets.x + table],
      params.global_scale,
      raw,
      1.0f,
      channel,
      error);
  }
  threadgroup_barrier(mem_flags::mem_device);

  for (uint coefficient = thread_index;
       coefficient < params.coefficient_count;
       coefficient += 256u) {
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
        params.global_scale,
        raw,
        multiplier,
        threshold,
        error);
      quantized_coefficients[offset] = quantized;
      reconstruction_coefficients[offset] = aq_dequantize_coefficient(
        quantized,
        quant_tables[table_offsets.x + table],
        params.global_scale,
        raw,
        multiplier,
        channel,
        error);
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  for (uint coefficient = thread_index;
       coefficient < params.coefficient_count;
       coefficient += 256u) {
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
       llf_task += 256u) {
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
