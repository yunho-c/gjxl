// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// General transformed-plane arithmetic follows the pinned scalar predictor.
#include "modular_adaptive_params.h"
#include "histogram_sample_rows_params.h"
#include "../../../codec/modular/weighted_feature_math.h"
#include <metal_stdlib>
using namespace metal;
using namespace gjxl_modular_adaptive;
namespace {
struct Neighborhood {
  long left, top, top_left, top_right, left_left, top_top, top_right_right;
};
Neighborhood neighbors(device const int *plane, ulong stride, uint width, uint x, uint y) {
  const ulong row = ulong(y) * stride;
  const long left = x ? plane[row + x - 1] : y ? plane[row - stride + x] : 0;
  const long top = y ? plane[row - stride + x] : left;
  const long right = y && x + 1 < width ? plane[row - stride + x + 1] : top;
  return {left,
          top,
          x && y ? plane[row - stride + x - 1] : left,
          right,
          x > 1 ? plane[row + x - 2] : left,
          y > 1 ? plane[row - 2 * stride + x] : top,
          y && x + 2 < width ? plane[row - stride + x + 2] : right};
}
long predict(uint p, Neighborhood n, long weighted) {
  switch (p) {
  case 0:
    return 0;
  case 1:
    return n.left;
  case 2:
    return n.top;
  case 3:
    return (n.left + n.top) / 2;
  case 4:
    return abs(n.top - n.top_left) < abs(n.left - n.top_left) ? n.left : n.top;
  case 5:
    return clamp(n.left + n.top - n.top_left, min(n.left, n.top), max(n.left, n.top));
  case 6:
    return weighted;
  case 7:
    return n.top_right;
  case 8:
    return n.top_left;
  case 9:
    return n.left_left;
  case 10:
    return (n.left + n.top_left) / 2;
  case 11:
    return (n.top_left + n.top) / 2;
  case 12:
    return (n.top + n.top_right) / 2;
  case 13:
    return (6 * n.top - 2 * n.top_top + 7 * n.left + n.left_left + n.top_right_right +
            3 * n.top_right + 8) /
           16;
  default:
    return 0; // Host validation rejects unknown predictors.
  }
}
long property(int p, Neighborhood n, uint channel, uint stream, uint x, uint y,
              device const int *plane, ulong stride, uint width, long weighted) {
  switch (p) {
  case 0:
    return long(channel);
  case 1:
    return long(stream);
  case 2:
    return long(y);
  case 3:
    return long(x);
  case 4:
    return long(abs(n.top));
  case 5:
    return long(abs(n.left));
  case 6:
    return n.top;
  case 7:
    return n.left;
  case 8: {
    long previous = 0;
    if (x) {
      const auto prior = neighbors(plane, stride, width, x - 1, y);
      previous = prior.left + prior.top - prior.top_left;
    }
    return n.left - previous;
  }
  case 9:
    return n.left + n.top - n.top_left;
  case 10:
    return n.left - n.top_left;
  case 11:
    return n.top_left - n.top;
  case 12:
    return n.top - n.top_right;
  case 13:
    return n.top - n.top_top;
  case 14:
    return n.left - n.left_left;
  case 15:
    return weighted;
  default:
    return 0; // Host validation rejects unknown properties.
  }
}
template <bool Diagnose>
void tokenize(device const int *planes, device const WeightedFeature *features,
              device uchar *output, device uchar *dispatch, constant Params &p, uint2 id,
              device Diagnostic *trace) {
  if (id.y >= p.slice_count)
    return;
  const Slice s = reinterpret_cast<device const Slice *>(dispatch)[id.y];
  if (id.x >= ulong(s.width) * s.height)
    return;
  const ulong token = s.token_offset + id.x;
  const uint x = id.x % s.width, y = id.x / s.width;
  device const int *plane = planes + s.plane_offset_elements;
  const Neighborhood n = neighbors(plane, s.plane_stride_elements, s.width, x, y);
  const WeightedFeature weighted = p.has_weighted ? features[token] : WeightedFeature{0, 0};
  device const Node *nodes = reinterpret_cast<device const Node *>(dispatch + p.nodes_offset_bytes);
  if (p.reserved)
    nodes += s.reserved[0]; // Independent stream tree, with local child/context indices.
  uint leaf = 0;
  while (nodes[leaf].property != -1) {
    const Node node = nodes[leaf];
    const long value = property(node.property, n, s.property_channel, s.stream_id, x, y, plane,
                                s.plane_stride_elements, s.width, weighted.property15);
    leaf = value > node.split ? node.left : node.right;
  }
  const Node node = nodes[leaf];
  const int sample = plane[ulong(y) * s.plane_stride_elements + x];
  const long prediction = predict(node.predictor, n, weighted.prediction);
  const long adjusted = long(sample) - prediction - long(node.offset);
  const long quotient = adjusted / long(node.multiplier);
  if (adjusted % long(node.multiplier) || quotient < -2147483648L || quotient > 2147483647L) {
    atomic_fetch_or_explicit(
        reinterpret_cast<device atomic_uint *>(dispatch + p.status_offset_bytes),
        uint(gjxl_modular_tokens::kResidualError), memory_order_relaxed);
    return;
  }
  const int residual = int(quotient);
  const uint value = (uint(residual) << 1) ^ ((uint(~residual) >> 31) - 1);
  reinterpret_cast<device uint *>(output)[token] = value;
  reinterpret_cast<device ushort *>(output + p.contexts_offset_bytes)[token] = ushort(node.context);
  if (Diagnose) {
    for (int i = 0; i < 16; ++i)
      trace[token].properties[i] = property(i, n, s.property_channel, s.stream_id, x, y, plane,
                                            s.plane_stride_elements, s.width, weighted.property15);
    for (uint i = 0; i < 14; ++i)
      trace[token].predictions[i] = predict(i, n, weighted.prediction);
    trace[token].selected_prediction = prediction;
    trace[token].leaf = leaf;
    trace[token].context = node.context;
    trace[token].sample = sample;
    trace[token].residual = residual;
    trace[token].value = value;
    trace[token].reserved = 0;
  }
}
} // namespace
// One independent selected row per thread. Sampling/cuts are supplied by the
// native preparer; the existing wide predictor/property arithmetic is shared
// with tokenization, including the slice-local previous-gradient boundary.
kernel void gjxl_histogram_sample_rows(
    device const int *planes [[buffer(0)]],
    device const WeightedFeature *features [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uchar *dispatch [[buffer(3)]],
    constant gjxl_histogram_rows::Params &p [[buffer(4)]],
    uint2 id [[thread_position_in_grid]]) {
  if (id.y >= p.slices)
    return;
  const auto s = reinterpret_cast<device const gjxl_histogram_rows::Slice *>(
      dispatch + p.slices_offset)[id.y];
  if (id.x >= s.row_count)
    return;
  const ulong row = ulong(s.first_row) + id.x;
  const uint pixel = reinterpret_cast<device const uint *>(dispatch + p.positions_offset)[row];
  if (pixel >= ulong(s.source.width) * s.source.height) {
    atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(dispatch + p.status_offset),
                             2u, memory_order_relaxed);
    return;
  }
  device const auto &m = *reinterpret_cast<device const gjxl_histogram_rows::Mapping *>(dispatch);
  const uint x = pixel % s.source.width, y = pixel / s.source.width;
  device const int *plane = planes + s.source.plane_offset_elements;
  const auto n = neighbors(plane, s.source.plane_stride_elements, s.source.width, x, y);
  const auto weighted = features[s.source.token_offset + pixel];
  for (uint i = 0; i < p.statics + p.dynamics; ++i) {
    const long value = property(m.properties[i], n, s.channel, s.group, x, y,
                                plane, s.source.plane_stride_elements, s.source.width,
                                weighted.property15);
    const uint index = uint(clamp(value, -511L, 511L) + 511);
    if (i < p.statics)
      reinterpret_cast<device uint *>(output)[ulong(i) * p.rows + row] = m.statics[i][index];
    else
      output[ulong(4 * p.statics + i - p.statics) * p.rows + row] = m.dynamics[i - p.statics][index];
  }
  const long sample = plane[ulong(y) * s.source.plane_stride_elements + x];
  for (uint i = 0; i < p.predictors; ++i) {
    const long residual = sample - predict(m.predictors[i], n, weighted.prediction);
    if (residual < -2147483648L || residual > 2147483647L) {
      atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(dispatch + p.status_offset),
                               1u, memory_order_relaxed);
      return;
    }
    const uint value = (uint(residual) << 1) ^ uint(int(residual) >> 31);
    uint symbol = value, extra = 0;
    if (value >= 16) {
      const uint nbits = 31 - clz(value), mantissa = value - (1u << nbits);
      symbol = 16 + ((nbits - 4) << 3) + ((mantissa >> (nbits - 1)) << 2) + (mantissa & 3);
      extra = nbits - 3;
    }
    const ulong offset = ulong(4 * p.statics + p.dynamics) * p.rows + 2 * (ulong(i) * p.rows + row);
    output[offset] = uchar(symbol);
    output[offset + 1] = uchar(extra);
  }
}
kernel void gjxl_modular_adaptive_tokens(device const int *planes [[buffer(0)]],
                                         device const WeightedFeature *features [[buffer(1)]],
                                         device uchar *output [[buffer(2)]],
                                         device uchar *dispatch [[buffer(3)]],
                                         constant Params &p [[buffer(4)]],
                                         uint2 id [[thread_position_in_grid]]) {
  tokenize<false>(planes, features, output, dispatch, p, id, nullptr);
}
kernel void gjxl_modular_adaptive_diagnostic(device const int *planes [[buffer(0)]],
                                             device const WeightedFeature *features [[buffer(1)]],
                                             device uchar *output [[buffer(2)]],
                                             device uchar *dispatch [[buffer(3)]],
                                             constant Params &p [[buffer(4)]],
                                             device Diagnostic *trace [[buffer(5)]],
                                             uint2 id [[thread_position_in_grid]]) {
  tokenize<true>(planes, features, output, dispatch, p, id, trace);
}

// One independent slice per group. All lanes reach every barrier, including
// inactive rows and lanes downstream of a numerical failure. Four banks retain
// dependencies at t-1, t-2 and t-3 for t=x+2*y; no whole-image error map is needed.
kernel void gjxl_modular_weighted_features(
    device const int *planes [[buffer(0)]], device WeightedFeature *features [[buffer(1)]],
    device const Slice *slices [[buffer(2)]], device atomic_uint *status [[buffer(3)]],
    constant gjxl_weighted_features::Parameters *parameters [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using namespace gjxl_weighted_features;
  threadgroup State frontier[4 * 256];
  const Slice s = slices[group];
  device const int *plane = planes + s.plane_offset_elements;
  for (uint step = 0; step < s.width + 2 * s.height - 2; ++step) {
    for (uint y = lane; y < s.height; y += 128) {
      if (step < 2 * y)
        continue;
      const uint x = step - 2 * y;
      if (x >= s.width)
        continue;
      const State west = x ? frontier[((step - 1) & 3) * 256 + y] : State{};
      const State westwest = x > 1 ? frontier[((step - 2) & 3) * 256 + y] : State{};
      const State north = y ? frontier[((step - 2) & 3) * 256 + y - 1] : State{};
      const State northeast =
          y ? (x + 1 < s.width ? frontier[((step - 1) & 3) * 256 + y - 1] : north) : State{};
      const State northwest = y ? (x ? frontier[((step - 3) & 3) * 256 + y - 1] : north) : State{};
      const auto n = neighbors(plane, s.plane_stride_elements, s.width, x, y);
      const auto result =
          PredictAndUpdate(x, s.width, {n.top, n.left, n.top_right, n.top_left, n.top_top},
                           plane[ulong(y) * s.plane_stride_elements + x], parameters[s.reserved[0]], west,
                           westwest, north, northeast, northwest);
      features[s.token_offset + ulong(y) * s.width + x] = {result.prediction, result.property15};
      frontier[(step & 3) * 256 + y] = result.state;
      if (!result.valid)
        atomic_store_explicit(status, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
