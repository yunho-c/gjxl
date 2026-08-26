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
