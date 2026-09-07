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

// Pass specialization removes dynamic neighborhood selection. The tiled
// variants load one mirrored neighborhood and process several rows per lane.
// The scalar operation and accumulation order intentionally match the control.
inline float aq_dataflow_sample(device const float* plane,
    constant AqEpfParams& params, uint2 position, uint2 local,
    uint tile_stride, int2 delta) {
  return aq_sample(plane, params.input_stride, params.width, params.height,
                   position.x, position.y, delta.x, delta.y);
}
inline float aq_dataflow_sample(threadgroup const float* plane,
    constant AqEpfParams& params, uint2 position, uint2 local,
    uint tile_stride, int2 delta) {
  return plane[uint(int(local.y) + delta.y) * tile_stride +
               uint(int(local.x) + delta.x)];
}

template <uint Pass, typename PlanePointer>
inline void AqEpfDataflowPixel(PlanePointer input_x, PlanePointer input_y,
    PlanePointer input_b, device const float* inverse_sigma,
    device float* output_x, device float* output_y, device float* output_b,
    device atomic_uint* error, constant AqEpfParams& params,
    uint2 position, uint2 local, uint tile_stride) {
  if (position.x >= params.width || position.y >= params.height) return;
  PlanePointer inputs[3] = {input_x, input_y, input_b};
  device float* outputs[3] = {output_x, output_y, output_b};
  const uint output_index = position.y * params.output_stride + position.x;
  const float block_inverse_sigma = inverse_sigma[
    (position.y / 8u) * params.inverse_sigma_stride + position.x / 8u];
  float center[3];
  for (uint channel = 0u; channel < 3u; ++channel) {
    center[channel] = aq_dataflow_sample(inputs[channel], params, position,
                                         local, tile_stride, int2(0));
  }
  if (block_inverse_sigma < -3.905242919921875f) {
    for (uint channel = 0u; channel < 3u; ++channel) {
      outputs[channel][output_index] = center[channel];
    }
    return;
  }
  const bool block_border =
    position.x % 8u == 0u || position.x % 8u == 7u ||
    position.y % 8u == 0u || position.y % 8u == 7u;
  const float scaled_inverse_sigma = block_inverse_sigma * params.sigma_scale *
    (block_border ? params.border_sad_multiplier : 1.0f);
  float sum[3] = {center[0], center[1], center[2]};
  float weight_sum = 1.0f;
  for (uint index = 0u; index < (Pass == 0u ? 12u : 4u); ++index) {
    const int2 offset = Pass == 0u ? kEpfPass0Offsets[index] : kEpfCardinalOffsets[index];
    float sad = 0.0f;
    for (uint channel = 0u; channel < 3u; ++channel) {
      if (Pass == 2u) {
        sad = fma(fabs(center[channel] - aq_dataflow_sample(inputs[channel],
          params, position, local, tile_stride, offset)), params.channel_scale[channel], sad);
      } else {
        float channel_sad = 0.0f;
        for (uint patch = 0u; patch < 5u; ++patch) {
          const int2 delta = kEpfPlusOffsets[patch];
          channel_sad += fabs(
            aq_dataflow_sample(inputs[channel], params, position, local, tile_stride, delta) -
            aq_dataflow_sample(inputs[channel], params, position, local, tile_stride, offset + delta));
        }
        sad = fma(channel_sad, params.channel_scale[channel], sad);
      }
    }
    const float weight = max(0.0f, fma(sad, scaled_inverse_sigma, 1.0f));
    weight_sum += weight;
    for (uint channel = 0u; channel < 3u; ++channel) {
      sum[channel] = fma(weight, aq_dataflow_sample(inputs[channel], params,
        position, local, tile_stride, offset), sum[channel]);
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

#define GJXL_EPF_ARGUMENTS \
  device const float* input_x [[buffer(0)]], \
  device const float* input_y [[buffer(1)]], \
  device const float* input_b [[buffer(2)]], \
  device const float* inverse_sigma [[buffer(3)]], \
  device float* output_x [[buffer(4)]], \
  device float* output_y [[buffer(5)]], \
  device float* output_b [[buffer(6)]], \
  device atomic_uint* error [[buffer(7)]], \
  constant AqEpfParams& params [[buffer(8)]]

#define GJXL_EPF_DIRECT(name, pass) \
kernel void name(GJXL_EPF_ARGUMENTS, uint2 position [[thread_position_in_grid]]) { \
  AqEpfDataflowPixel<pass>(input_x, input_y, input_b, inverse_sigma, \
    output_x, output_y, output_b, error, params, position, uint2(0), 0); \
}
GJXL_EPF_DIRECT(gjxl_aq_epf_pass0_direct, 0)
GJXL_EPF_DIRECT(gjxl_aq_epf_pass1_direct, 1)
GJXL_EPF_DIRECT(gjxl_aq_epf_pass2_direct, 2)
#undef GJXL_EPF_DIRECT

// Repeated reflection also bounds cooperative loads made outside the last
// partial tile. No lane returns until the neighborhood has been published.
inline uint aq_epf_mirror_coordinate(int coordinate, uint size) {
  const int period = int(2u * size);
  int reflected = coordinate % period;
  if (reflected < 0) reflected += period;
  return uint(reflected < int(size) ? reflected : period - reflected - 1);
}

template <uint Pass, uint Width, uint Height, uint PixelsPerThread>
inline void AqEpfDataflowTile(device const float* input_x,
    device const float* input_y, device const float* input_b,
    device const float* inverse_sigma, device float* output_x,
    device float* output_y, device float* output_b, device atomic_uint* error,
    constant AqEpfParams& params, threadgroup float* tile,
    uint2 thread_position, uint2 group_position) {
  constexpr uint kHalo = Pass == 0 ? 3 : Pass == 1 ? 2 : 1;
  constexpr uint kStride = Width + 2 * kHalo;
  constexpr uint kTileHeight = Height * PixelsPerThread;
  constexpr uint kPlane = kStride * (kTileHeight + 2 * kHalo);
  const uint2 origin = group_position * uint2(Width, kTileHeight);
  const uint tid = thread_position.y * Width + thread_position.x;
  const bool interior = origin.x >= kHalo && origin.y >= kHalo &&
    origin.x + Width + kHalo <= params.width &&
    origin.y + kTileHeight + kHalo <= params.height;
  for (uint i = tid; i < kPlane; i += Width * Height) {
    uint x = origin.x + i % kStride - kHalo;
    uint y = origin.y + i / kStride - kHalo;
    if (!interior) {
      x = aq_epf_mirror_coordinate(int(origin.x + i % kStride) - int(kHalo), params.width);
      y = aq_epf_mirror_coordinate(int(origin.y + i / kStride) - int(kHalo), params.height);
    }
    const uint input_index = y * params.input_stride + x;
    tile[i] = input_x[input_index];
    tile[kPlane + i] = input_y[input_index];
    tile[2 * kPlane + i] = input_b[input_index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint pixel = 0; pixel < PixelsPerThread; ++pixel) {
    const uint2 local = thread_position + uint2(kHalo, kHalo + pixel * Height);
    const uint2 position = origin + thread_position + uint2(0, pixel * Height);
    AqEpfDataflowPixel<Pass>(static_cast<threadgroup const float*>(tile),
      static_cast<threadgroup const float*>(tile + kPlane),
      static_cast<threadgroup const float*>(tile + 2 * kPlane), inverse_sigma,
      output_x, output_y, output_b, error, params, position, local, kStride);
  }
}

#define GJXL_EPF_TILED(name, pass, width, height, pixels) \
kernel void name(GJXL_EPF_ARGUMENTS, \
  uint2 thread_position [[thread_position_in_threadgroup]], \
  uint2 group_position [[threadgroup_position_in_grid]]) { \
  constexpr uint halo = pass == 0 ? 3 : pass == 1 ? 2 : 1; \
  threadgroup float tile[3 * (width + 2 * halo) * (height * pixels + 2 * halo)]; \
  AqEpfDataflowTile<pass, width, height, pixels>(input_x, input_y, input_b, \
    inverse_sigma, output_x, output_y, output_b, error, params, tile, \
    thread_position, group_position); \
}
#define GJXL_EPF_VARIANTS(pass) \
  GJXL_EPF_TILED(gjxl_aq_epf_pass##pass##_tile16x8_p1, pass, 16, 8, 1) \
  GJXL_EPF_TILED(gjxl_aq_epf_pass##pass##_tile16x8_p2, pass, 16, 8, 2) \
  GJXL_EPF_TILED(gjxl_aq_epf_pass##pass##_tile16x8_p4, pass, 16, 8, 4) \
  GJXL_EPF_TILED(gjxl_aq_epf_pass##pass##_tile32x4_p2, pass, 32, 4, 2) \
  GJXL_EPF_TILED(gjxl_aq_epf_pass##pass##_tile32x4_p4, pass, 32, 4, 4)
GJXL_EPF_VARIANTS(0)
GJXL_EPF_VARIANTS(1)
GJXL_EPF_VARIANTS(2)
#undef GJXL_EPF_VARIANTS
#undef GJXL_EPF_TILED
#undef GJXL_EPF_ARGUMENTS
