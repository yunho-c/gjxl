// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <metal_stdlib>

using namespace metal;

struct AqGaborishParams {
  uint width;
  uint height;
  uint input_stride;
  uint output_stride;
  float center_weight[3];
  float axis_weight[3];
  float diagonal_weight[3];
};

struct AqEpfParams {
  uint width;
  uint height;
  uint input_stride;
  uint output_stride;
  uint inverse_sigma_stride;
  uint pass;
  float sigma_scale;
  float border_sad_multiplier;
  float channel_scale[3];
};

struct AqOpsinToLinearParams {
  uint width;
  uint height;
  uint input_stride;
  uint output_stride;
  float scale;
};

constant int2 kEpfPlusOffsets[5] = {
  int2(0, 0), int2(0, -1), int2(-1, 0), int2(0, 1), int2(1, 0),
};

constant int2 kEpfPass0Offsets[12] = {
  int2(0, -2), int2(-1, -1), int2(0, -1), int2(1, -1),
  int2(-2, 0), int2(-1, 0), int2(1, 0), int2(2, 0),
  int2(-1, 1), int2(0, 1), int2(1, 1), int2(0, 2),
};

constant int2 kEpfCardinalOffsets[4] = {
  int2(0, -1), int2(-1, 0), int2(1, 0), int2(0, 1),
};

static uint aq_mirror_offset(uint coordinate, int delta, uint size) {
  if (delta < 0) {
    const uint distance = uint(-delta);
    return coordinate >= distance
      ? coordinate - distance
      : distance - coordinate - 1u;
  }
  const uint distance = uint(delta);
  const uint remaining = size - coordinate - 1u;
  return distance <= remaining
    ? coordinate + distance
    : size - (distance - remaining);
}

static float aq_sample(
  device const float* plane,
  uint stride,
  uint width,
  uint height,
  uint x,
  uint y,
  int dx,
  int dy) {

  const uint sample_x = aq_mirror_offset(x, dx, width);
  const uint sample_y = aq_mirror_offset(y, dy, height);
  return plane[sample_y * stride + sample_x];
}

kernel void gjxl_aq_gaborish_f32(
  device const float* input_x [[buffer(0)]],
  device const float* input_y [[buffer(1)]],
  device const float* input_b [[buffer(2)]],
  device float* output_x [[buffer(3)]],
  device float* output_y [[buffer(4)]],
  device float* output_b [[buffer(5)]],
  device atomic_uint* error [[buffer(6)]],
  constant AqGaborishParams& params [[buffer(7)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  device const float* inputs[3] = {input_x, input_y, input_b};
  device float* outputs[3] = {output_x, output_y, output_b};
  const uint output_index = position.y * params.output_stride + position.x;
  for (uint channel = 0u; channel < 3u; ++channel) {
    device const float* input = inputs[channel];
    const float axes =
      (aq_sample(input, params.input_stride, params.width, params.height,
                 position.x, position.y, -1, 0) +
       aq_sample(input, params.input_stride, params.width, params.height,
                 position.x, position.y, 1, 0)) +
      (aq_sample(input, params.input_stride, params.width, params.height,
                 position.x, position.y, 0, -1) +
       aq_sample(input, params.input_stride, params.width, params.height,
                 position.x, position.y, 0, 1));
    const float diagonals =
      (aq_sample(input, params.input_stride, params.width, params.height,
                 position.x, position.y, -1, -1) +
       aq_sample(input, params.input_stride, params.width, params.height,
                 position.x, position.y, 1, -1)) +
      (aq_sample(input, params.input_stride, params.width, params.height,
                 position.x, position.y, -1, 1) +
       aq_sample(input, params.input_stride, params.width, params.height,
                 position.x, position.y, 1, 1));
    float value = params.center_weight[channel] *
      aq_sample(input, params.input_stride, params.width, params.height,
                position.x, position.y, 0, 0);
    value += params.axis_weight[channel] * axes;
    value += params.diagonal_weight[channel] * diagonals;
    if (!isfinite(value)) {
      atomic_fetch_or_explicit(error, 16u, memory_order_relaxed);
      value = 0.0f;
    }
    outputs[channel][output_index] = value;
  }
}

static float aq_epf_patch_sad(
  device const float* input_x,
  device const float* input_y,
  device const float* input_b,
  constant AqEpfParams& params,
  uint x,
  uint y,
  int dx,
  int dy) {

  device const float* inputs[3] = {input_x, input_y, input_b};
  float sad = 0.0f;
  for (uint channel = 0u; channel < 3u; ++channel) {
    float channel_sad = 0.0f;
    for (uint index = 0u; index < 5u; ++index) {
      const int2 offset = kEpfPlusOffsets[index];
      channel_sad += fabs(
        aq_sample(inputs[channel], params.input_stride, params.width,
                  params.height, x, y, offset.x, offset.y) -
        aq_sample(inputs[channel], params.input_stride, params.width,
                  params.height, x, y, dx + offset.x, dy + offset.y));
    }
    sad = fma(channel_sad, params.channel_scale[channel], sad);
  }
  return sad;
}

static float aq_epf_pixel_sad(
  device const float* input_x,
  device const float* input_y,
  device const float* input_b,
  constant AqEpfParams& params,
  uint x,
  uint y,
  int dx,
  int dy) {

  device const float* inputs[3] = {input_x, input_y, input_b};
  float sad = 0.0f;
  for (uint channel = 0u; channel < 3u; ++channel) {
    sad = fma(
      fabs(
        aq_sample(inputs[channel], params.input_stride, params.width,
                  params.height, x, y, 0, 0) -
        aq_sample(inputs[channel], params.input_stride, params.width,
                  params.height, x, y, dx, dy)),
      params.channel_scale[channel], sad);
  }
  return sad;
}

kernel void gjxl_aq_epf_f32(
  device const float* input_x [[buffer(0)]],
  device const float* input_y [[buffer(1)]],
  device const float* input_b [[buffer(2)]],
  device const float* inverse_sigma [[buffer(3)]],
  device float* output_x [[buffer(4)]],
  device float* output_y [[buffer(5)]],
  device float* output_b [[buffer(6)]],
  device atomic_uint* error [[buffer(7)]],
  constant AqEpfParams& params [[buffer(8)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  device const float* inputs[3] = {input_x, input_y, input_b};
  device float* outputs[3] = {output_x, output_y, output_b};
  const uint input_index = position.y * params.input_stride + position.x;
  const uint output_index = position.y * params.output_stride + position.x;
  const float block_inverse_sigma = inverse_sigma[
    (position.y / 8u) * params.inverse_sigma_stride + position.x / 8u];
  if (block_inverse_sigma < -3.905242919921875f) {
    for (uint channel = 0u; channel < 3u; ++channel) {
      outputs[channel][output_index] = inputs[channel][input_index];
    }
    return;
  }

  const bool block_border =
    position.x % 8u == 0u || position.x % 8u == 7u ||
    position.y % 8u == 0u || position.y % 8u == 7u;
  const float scaled_inverse_sigma = block_inverse_sigma *
    params.sigma_scale *
    (block_border ? params.border_sad_multiplier : 1.0f);
  float sum[3] = {
    input_x[input_index], input_y[input_index], input_b[input_index],
  };
  float weight_sum = 1.0f;

  const uint candidate_count = params.pass == 0u ? 12u : 4u;
  for (uint index = 0u; index < candidate_count; ++index) {
    const int2 offset = params.pass == 0u
      ? kEpfPass0Offsets[index]
      : kEpfCardinalOffsets[index];
    const float sad = params.pass == 2u
      ? aq_epf_pixel_sad(input_x, input_y, input_b, params,
                         position.x, position.y, offset.x, offset.y)
      : aq_epf_patch_sad(input_x, input_y, input_b, params,
                         position.x, position.y, offset.x, offset.y);
    const float weight = max(0.0f, fma(sad, scaled_inverse_sigma, 1.0f));
    weight_sum += weight;
    for (uint channel = 0u; channel < 3u; ++channel) {
      sum[channel] = fma(
        weight,
        aq_sample(inputs[channel], params.input_stride, params.width,
                  params.height, position.x, position.y, offset.x, offset.y),
        sum[channel]);
    }
  }

  for (uint channel = 0u; channel < 3u; ++channel) {
    float value = sum[channel] / weight_sum;
    if (!isfinite(value)) {
      atomic_fetch_or_explicit(error, 32u, memory_order_relaxed);
      value = 0.0f;
    }
    outputs[channel][output_index] = value;
  }
}

kernel void gjxl_aq_opsin_to_linear_rgb_f32(
  device const float* input_x [[buffer(0)]],
  device const float* input_y [[buffer(1)]],
  device const float* input_b [[buffer(2)]],
  device float* output_r [[buffer(3)]],
  device float* output_g [[buffer(4)]],
  device float* output_b [[buffer(5)]],
  device atomic_uint* error [[buffer(6)]],
  constant AqOpsinToLinearParams& params [[buffer(7)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  constexpr float kOpsinBias = 0.0037930732552754493f;
  constexpr float kBiasCuberoot = 0.15595419704914093f;
  constexpr float kInverseOpsinMatrix[9] = {
    11.031566901960783f, -9.866943921568629f, -0.16462299647058826f,
    -3.254147380392157f, 4.418770392156863f, -0.16462299647058826f,
    -3.6588512862745097f, 2.7129230470588235f, 1.9459282392156863f,
  };
  const uint input_index = position.y * params.input_stride + position.x;
  const uint output_index = position.y * params.output_stride + position.x;
  const float gamma[3] = {
    input_y[input_index] + input_x[input_index] + kBiasCuberoot,
    input_y[input_index] - input_x[input_index] + kBiasCuberoot,
    input_b[input_index] + kBiasCuberoot,
  };
  float mixed[3];
  for (uint channel = 0u; channel < 3u; ++channel) {
    mixed[channel] = gamma[channel] * gamma[channel] * gamma[channel] -
      kOpsinBias;
  }
  device float* outputs[3] = {output_r, output_g, output_b};
  for (uint row = 0u; row < 3u; ++row) {
    float value = params.scale * kInverseOpsinMatrix[3u * row] * mixed[0];
    value = fma(
      params.scale * kInverseOpsinMatrix[3u * row + 1u], mixed[1], value);
    value = fma(
      params.scale * kInverseOpsinMatrix[3u * row + 2u], mixed[2], value);
    if (!isfinite(value)) {
      atomic_fetch_or_explicit(error, 64u, memory_order_relaxed);
      value = 0.0f;
    }
    outputs[row][output_index] = value;
  }
}
