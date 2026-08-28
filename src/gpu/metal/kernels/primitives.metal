// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <metal_stdlib>

using namespace metal;

struct AffineParams {
  uint width;
  uint height;
  uint input_stride;
  uint output_stride;
  float scale;
  float bias;
};

struct ConvolutionParams {
  uint width;
  uint height;
  uint input_stride;
  uint output_stride;
  uint kernel_size;
};

struct Symmetric5Params {
  uint width;
  uint height;
  uint input_stride;
  uint output_stride;
  float distance0;
  float distance1;
  float distance2;
  float distance4;
  float distance8;
  float distance5;
};

struct ReductionParams {
  uint width;
  uint input_stride;
  uint input_count;
};

kernel void gjxl_pointwise_affine_f32(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  constant AffineParams& params [[buffer(2)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) {
    return;
  }
  output[position.y * params.output_stride + position.x] =
    input[position.y * params.input_stride + position.x] * params.scale +
    params.bias;
}

kernel void gjxl_convolve_horizontal_f32(
  device const float* input [[buffer(0)]],
  device const float* weights [[buffer(1)]],
  device float* output [[buffer(2)]],
  constant ConvolutionParams& params [[buffer(3)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) {
    return;
  }
  const int radius = static_cast<int>(params.kernel_size / 2);
  const int center = static_cast<int>(position.x);
  float sum = 0.0f;
  float weight_sum = 0.0f;
  for (int delta = -radius; delta <= radius; ++delta) {
    const int source_x = center + delta;
    if (source_x < 0 || source_x >= static_cast<int>(params.width)) {
      continue;
    }
    const float weight = weights[delta + radius];
    sum += input[position.y * params.input_stride +
                 static_cast<uint>(source_x)] * weight;
    weight_sum += weight;
  }
  output[position.y * params.output_stride + position.x] = sum / weight_sum;
}

kernel void gjxl_convolve_vertical_f32(
  device const float* input [[buffer(0)]],
  device const float* weights [[buffer(1)]],
  device float* output [[buffer(2)]],
  constant ConvolutionParams& params [[buffer(3)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) {
    return;
  }
  const int radius = static_cast<int>(params.kernel_size / 2);
  const int center = static_cast<int>(position.y);
  float sum = 0.0f;
  float weight_sum = 0.0f;
  for (int delta = -radius; delta <= radius; ++delta) {
    const int source_y = center + delta;
    if (source_y < 0 || source_y >= static_cast<int>(params.height)) {
      continue;
    }
    const float weight = weights[delta + radius];
    sum += input[static_cast<uint>(source_y) * params.input_stride +
                 position.x] * weight;
    weight_sum += weight;
  }
  output[position.y * params.output_stride + position.x] = sum / weight_sum;
}

static uint gjxl_mirror_radius2(int coordinate, uint size) {
  if (size == 1u) return 0u;
  if (coordinate < 0) return uint(-coordinate - 1);
  if (coordinate >= int(size)) return 2u * size - 1u - uint(coordinate);
  return uint(coordinate);
}

static float gjxl_symmetric5_weighted_row(
  device const float* input,
  int x,
  int y,
  constant Symmetric5Params& params,
  float center_weight,
  float near_weight,
  float far_weight) {

  const uint source_y = gjxl_mirror_radius2(y, params.height);
  const uint far_left = gjxl_mirror_radius2(x - 2, params.width);
  const uint near_left = gjxl_mirror_radius2(x - 1, params.width);
  const uint center = gjxl_mirror_radius2(x, params.width);
  const uint near_right = gjxl_mirror_radius2(x + 1, params.width);
  const uint far_right = gjxl_mirror_radius2(x + 2, params.width);
  device const float* row = input + source_y * params.input_stride;
  const float far = far_weight * (row[far_left] + row[far_right]);
  const float near = near_weight * (row[near_left] + row[near_right]);
  const float center_value = center_weight * row[center];
  return far + (near + center_value);
}

kernel void gjxl_convolve_symmetric5_f32(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  constant Symmetric5Params& params [[buffer(2)]],
  uint2 position [[thread_position_in_grid]]) {

  if (position.x >= params.width || position.y >= params.height) return;
  const int x = int(position.x);
  const int y = int(position.y);
  float sum0 = gjxl_symmetric5_weighted_row(
    input, x, y, params, params.distance0, params.distance1,
    params.distance2);
  sum0 += gjxl_symmetric5_weighted_row(
    input, x, y - 2, params, params.distance2, params.distance5,
    params.distance8);
  float sum1 = gjxl_symmetric5_weighted_row(
    input, x, y + 2, params, params.distance2, params.distance5,
    params.distance8);
  sum0 += gjxl_symmetric5_weighted_row(
    input, x, y - 1, params, params.distance1, params.distance4,
    params.distance5);
  sum1 += gjxl_symmetric5_weighted_row(
    input, x, y + 1, params, params.distance1, params.distance4,
    params.distance5);
  output[position.y * params.output_stride + position.x] = sum0 + sum1;
}

kernel void gjxl_reduce_max_f32(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  constant ReductionParams& params [[buffer(2)]],
  uint thread_index [[thread_index_in_threadgroup]],
  uint3 group_position [[threadgroup_position_in_grid]]) {

  constexpr uint kReductionWidth = 256;
  threadgroup float values[kReductionWidth];
  const uint index = group_position.x * kReductionWidth + thread_index;
  float value = -INFINITY;
  if (index < params.input_count) {
    const uint y = index / params.width;
    const uint x = index - y * params.width;
    value = input[y * params.input_stride + x];
  }
  values[thread_index] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = kReductionWidth / 2; step != 0; step /= 2) {
    if (thread_index < step) {
      values[thread_index] = max(values[thread_index],
                                 values[thread_index + step]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (thread_index == 0) {
    output[group_position.x] = values[0];
  }
}
