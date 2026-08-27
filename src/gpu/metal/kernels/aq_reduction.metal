// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <metal_stdlib>

using namespace metal;

struct AqBlockReductionParams {
  uint source_width;
  uint source_height;
  uint distance_stride;
  uint block_stride;
  uint anchor_offset;
  uint anchor_count;
  uint pixel_width;
  uint pixel_height;
  uint covered_width;
  uint covered_height;
};

kernel void gjxl_aq_reduce_block_distance_f32(
  device const float* distance_map [[buffer(0)]],
  device const uint2* anchors [[buffer(1)]],
  device float* block_distance [[buffer(2)]],
  device atomic_uint* error [[buffer(3)]],
  constant AqBlockReductionParams& params [[buffer(4)]],
  uint anchor_index [[threadgroup_position_in_grid]],
  uint thread_index [[thread_index_in_threadgroup]]) {

  threadgroup float partial[256];
  if (anchor_index >= params.anchor_count) return;

  const uint2 anchor = anchors[params.anchor_offset + anchor_index];
  const uint x_begin = anchor.x * 8u;
  const uint y_begin = anchor.y * 8u;
  if (x_begin >= params.source_width || y_begin >= params.source_height) {
    if (thread_index == 0u) {
      atomic_fetch_or_explicit(error, 64u, memory_order_relaxed);
    }
    return;
  }
  const uint valid_width =
    min(params.pixel_width, params.source_width - x_begin);
  const uint valid_height =
    min(params.pixel_height, params.source_height - y_begin);
  const uint pixel_count = valid_width * valid_height;

  float sum = 0.0f;
  for (uint index = thread_index; index < pixel_count; index += 256u) {
    const uint x = index % valid_width;
    const uint y = index / valid_width;
    float value = distance_map[
      (y_begin + y) * params.distance_stride + x_begin + x];
    if (!isfinite(value) || value < 0.0f) {
      atomic_fetch_or_explicit(error, 128u, memory_order_relaxed);
      value = 0.0f;
    }
    value *= value;
    value *= value;
    value *= value;
    value *= value;
    sum += value;
  }
  partial[thread_index] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint width = 128u; width != 0u; width >>= 1u) {
    if (thread_index < width) {
      partial[thread_index] += partial[thread_index + width];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (thread_index == 0u) {
    const float mean = partial[0] / float(pixel_count);
    const float reduced = 1.2f * pow(mean, 1.0f / 16.0f);
    if (!isfinite(reduced) || reduced < 0.0f) {
      atomic_fetch_or_explicit(error, 256u, memory_order_relaxed);
      return;
    }
    for (uint dy = 0u; dy < params.covered_height; ++dy) {
      for (uint dx = 0u; dx < params.covered_width; ++dx) {
        block_distance[
          (anchor.y + dy) * params.block_stride + anchor.x + dx] = reduced;
      }
    }
  }
}
