// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "modular_histogram_fitting_params.h"
#include <metal_stdlib>
using namespace metal;
using namespace gjxl_histogram_fitting;

namespace {
constant float kMaximumCost = 0x1.fffffep127f;
device atomic_uint *StatusWord(device uchar *dispatch, constant Params &p) {
  return reinterpret_cast<device atomic_uint *>(dispatch + p.status_offset);
}
bool Failed(device uchar *dispatch, constant Params &p) {
  return atomic_load_explicit(StatusWord(dispatch, p), memory_order_relaxed);
}
void Error(device uchar *dispatch, constant Params &p, uint error) {
  atomic_fetch_or_explicit(StatusWord(dispatch, p), error,
                           memory_order_relaxed);
}
TaskDescriptor Task(device uchar *dispatch, constant Params &p, uint slot) {
  return reinterpret_cast<device const TaskDescriptor *>(
      dispatch + p.tasks_offset)[p.first_task + slot];
}
float4 NativeLog2(float4 x) {
  const uint4 bits = as_type<uint4>(x);
  const int4 exponent = as_type<int4>(bits - uint4(0x3f2aaaab)) >> 23;
  const float4 mantissa =
      as_type<float4>(bits - (as_type<uint4>(exponent) << 23));
  const float4 z = mantissa - 1.0f;
  float4 a =
      fma(float4(7.4245873327820566E-01f), z, float4(1.4287160470083755E+00f));
  float4 b =
      fma(float4(1.7409343003366853E-01f), z, float4(1.0096718572241148E+00f));
  a = fma(a, z, float4(-1.8503833400518310E-06f));
  b = fma(b, z, float4(9.9032814277590719E-01f));
  return a / b + float4(exponent);
}
// Compiled with fast math and implicit contraction disabled. Four independent
// lanes accumulate symbols in the CPU order, followed by its pairwise sum.
float Entropy(device const uint *lower, device const int *parent, uint symbols,
              bool high) {
  uint total = 0;
  for (uint s = 0; s < symbols; ++s)
    total += high ? uint(parent[s]) - lower[s] : lower[s];
  const float inverse = 1.0f / float(total);
  float4 bits = 0.0f;
  for (uint s = 0; s < symbols; s += 4) {
    uint4 counts(lower[s], lower[s + 1], lower[s + 2], lower[s + 3]);
    if (high)
      counts = uint4(parent[s], parent[s + 1], parent[s + 2], parent[s + 3]) -
               counts;
    const float4 f = float4(counts);
    const float4 probability = max(f * inverse, float4(1.0f / 4096));
    const float4 log =
        select(NativeLog2(probability), float4(0), counts == total);
    bits = bits - f * log;
  }
  return (bits.x + bits.y) + (bits.z + bits.w);
}
bool Threshold(device const uint *occupancy, uint b, uint rows) {
  return occupancy[b] != (b ? occupancy[b - 1] : 0) && occupancy[b] < rows;
}
} // namespace

// Validate membership and the CPU parent summary even for all-constant nodes.
// The 64-bit population reduction cannot wrap for malformed weighted multisets;
// uint histogram overflow on such a request is harmless because no result is
// published and all subsequent kernels observe the status before reading it.
kernel void gjxl_histogram_validate(device const uchar *prepared [[buffer(0)]],
                                    device uchar *dispatch [[buffer(1)]],
                                    device PropertyResult *output [[buffer(2)]],
                                    constant Params &p [[buffer(4)]],
                                    uint node_index
                                    [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_threadgroup]]) {
  const NodeDescriptor n =
      reinterpret_cast<device const NodeDescriptor *>(dispatch)[node_index];
  const FitDescriptor f =
      reinterpret_cast<device const FitDescriptor *>(prepared)[n.fit];
  device const uint *indices =
      reinterpret_cast<device const uint *>(dispatch + n.indices_offset);
  device const ushort *weights =
      reinterpret_cast<device const ushort *>(prepared + f.weights_offset);
  device const int *parent =
      reinterpret_cast<device const int *>(dispatch + n.parent_counts_offset);
  device const uint *parent_extra =
      reinterpret_cast<device const uint *>(dispatch + n.parent_extra_offset);
  threadgroup atomic_uint counts[14 * 256], extra[14];
  threadgroup ulong population[128];
  for (uint i = lane; i < f.predictors * n.symbols; i += 128)
    atomic_store_explicit(&counts[i], 0, memory_order_relaxed);
  if (lane < f.predictors)
    atomic_store_explicit(&extra[lane], 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong sum = 0;
  for (uint i = lane; i < n.rows; i += 128) {
    const uint row = indices[i];
    if (row >= f.rows) {
      Error(dispatch, p, kRowIndexError);
      continue;
    }
    const uint weight = weights[row];
    sum += weight;
    for (uint pred = 0; pred < f.predictors; ++pred) {
      device const uchar2 *tokens = reinterpret_cast<device const uchar2 *>(
          prepared + f.residual_offsets[pred]);
      const uchar2 token = tokens[row];
      if (token.x >= n.symbols) {
        Error(dispatch, p, kParentSummaryError);
        continue;
      }
      atomic_fetch_add_explicit(&counts[pred * n.symbols + token.x], weight,
                                memory_order_relaxed);
      atomic_fetch_add_explicit(&extra[pred], weight * uint(token.y),
                                memory_order_relaxed);
    }
  }
  population[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 64; stride; stride >>= 1) {
    if (lane < stride)
      population[lane] += population[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0 && (population[0] > f.population || population[0] > (1u << 24)))
    Error(dispatch, p, kPopulationError);
  for (uint i = lane; i < f.predictors * n.symbols; i += 128)
    if (atomic_load_explicit(&counts[i], memory_order_relaxed) !=
        uint(parent[i]))
      Error(dispatch, p, kParentSummaryError);
  if (lane < f.predictors &&
      atomic_load_explicit(&extra[lane], memory_order_relaxed) !=
          parent_extra[lane])
    Error(dispatch, p, kParentSummaryError);
  if (lane < f.properties) {
    PropertyResult result;
    result.source_id = f.source_id;
    result.node_id = n.node_id;
    result.property_index = lane;
    result.is_constant =
        (n.inherited_constants & (1u << lane)) != 0 || f.bins[lane] == 1;
    for (uint c = 0; c < kCategories; ++c)
      result.winners[c] = {0, 0, 0, 0, kMaximumCost, kMaximumCost, 0, 0};
    output[n.result_first + lane] = result;
  }
}

// Independent integer shards have deterministic sums. Occupancy counts row
// entries; histogram and extra-bit increments use the positive row weights.
kernel void gjxl_histogram_collect(device const uchar *prepared [[buffer(0)]],
                                   device uchar *dispatch [[buffer(1)]],
                                   device uchar *scratch [[buffer(3)]],
                                   constant Params &p [[buffer(4)]],
                                   uint3 group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_threadgroup]]) {
  if (Failed(dispatch, p))
    return;
  const uint shard = group.x % p.shards, tile = group.x / p.shards;
  const uint pred = group.y, slot = group.z;
  const TaskDescriptor task = Task(dispatch, p, slot);
  const NodeDescriptor n =
      reinterpret_cast<device const NodeDescriptor *>(dispatch)[task.node];
  const FitDescriptor f =
      reinterpret_cast<device const FitDescriptor *>(prepared)[n.fit];
  const uint bins = f.bins[task.property];
  const uint tile_bins = min(256u, kHistogramTileCells / n.symbols), first = tile * tile_bins;
  if (pred >= f.predictors || first >= bins)
    return;
  const uint width = min(tile_bins, bins - first);
  // Bounded local atomics avoid serializing every row increment through device
  // memory. Tiles rescan only membership/property values; residual/weight reads
  // are restricted to the tile. Each shard writes disjoint complete output,
  // including zeros, so the arena needs no separate clear between waves.
  threadgroup atomic_uint counts[kHistogramTileCells], bit_counts[256], occupancy[256];
  for (uint i = lane; i < width * n.symbols; i += 128)
    atomic_store_explicit(&counts[i], 0, memory_order_relaxed);
  for (uint i = lane; i < width; i += 128) {
    atomic_store_explicit(&bit_counts[i], 0, memory_order_relaxed);
    atomic_store_explicit(&occupancy[i], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device const uint *indices =
      reinterpret_cast<device const uint *>(dispatch + n.indices_offset);
  device const ushort *weights =
      reinterpret_cast<device const ushort *>(prepared + f.weights_offset);
  device const uchar2 *tokens = reinterpret_cast<device const uchar2 *>(
      prepared + f.residual_offsets[pred]);
  device const uchar *property = prepared + f.property_offsets[task.property];
  for (uint i = shard * 128 + lane; i < n.rows; i += p.shards * 128) {
    const uint row = indices[i];
    const uint bin = task.property < f.static_properties
                         ? reinterpret_cast<device const uint *>(property)[row]
                         : property[row];
    if (bin < first || bin >= first + width)
      continue;
    const uint local = bin - first, weight = weights[row];
    const uchar2 code = tokens[row];
    atomic_fetch_add_explicit(&counts[local * n.symbols + code.x], weight,
                              memory_order_relaxed);
    atomic_fetch_add_explicit(&bit_counts[local], weight * uint(code.y),
                              memory_order_relaxed);
    if (pred == 0)
      atomic_fetch_add_explicit(&occupancy[local], 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device uchar *base = scratch + slot * p.slot_bytes;
  device uint *hist =
      reinterpret_cast<device uint *>(base + p.shard_histograms);
  device uint *extra = reinterpret_cast<device uint *>(base + p.shard_extra);
  device uint *occupied =
      reinterpret_cast<device uint *>(base + p.shard_occupancy);
  for (uint i = lane; i < width * n.symbols; i += 128)
    hist[((shard * f.predictors + pred) * bins + first) * n.symbols + i] =
        atomic_load_explicit(&counts[i], memory_order_relaxed);
  for (uint i = lane; i < width; i += 128) {
    extra[(shard * f.predictors + pred) * bins + first + i] =
        atomic_load_explicit(&bit_counts[i], memory_order_relaxed);
    if (pred == 0)
      occupied[shard * bins + first + i] =
          atomic_load_explicit(&occupancy[i], memory_order_relaxed);
  }
}

kernel void gjxl_histogram_prefix(device const uchar *prepared [[buffer(0)]],
                                  device uchar *dispatch [[buffer(1)]],
                                  device uchar *scratch [[buffer(3)]],
                                  constant Params &p [[buffer(4)]],
                                  uint2 index [[thread_position_in_grid]]) {
  if (Failed(dispatch, p))
    return;
  const uint axis = index.x, slot = index.y;
  const TaskDescriptor task = Task(dispatch, p, slot);
  const NodeDescriptor n =
      reinterpret_cast<device const NodeDescriptor *>(dispatch)[task.node];
  const FitDescriptor f =
      reinterpret_cast<device const FitDescriptor *>(prepared)[n.fit];
  const uint bins = f.bins[task.property];
  device uchar *base = scratch + slot * p.slot_bytes;
  device const uint *sh =
      reinterpret_cast<device const uint *>(base + p.shard_histograms);
  device const uint *se =
      reinterpret_cast<device const uint *>(base + p.shard_extra);
  device const uint *so =
      reinterpret_cast<device const uint *>(base + p.shard_occupancy);
  device uint *hist = reinterpret_cast<device uint *>(base + p.histograms);
  device uint *extra = reinterpret_cast<device uint *>(base + p.extra);
  device uint *occupied = reinterpret_cast<device uint *>(base + p.occupancy);
  if (axis < f.predictors * n.symbols) {
    const uint pred = axis / n.symbols, sym = axis % n.symbols;
    uint sum = 0;
    for (uint b = 0; b < bins; ++b) {
      for (uint shard = 0; shard < p.shards; ++shard)
        sum += sh[((shard * f.predictors + pred) * bins + b) * n.symbols + sym];
      hist[(pred * bins + b) * n.symbols + sym] = sum;
    }
    device const int *parent =
        reinterpret_cast<device const int *>(dispatch + n.parent_counts_offset);
    if (sum != uint(parent[axis]))
      Error(dispatch, p, kArithmeticError);
  }
  if (axis < f.predictors) {
    uint sum = 0;
    for (uint b = 0; b < bins; ++b) {
      for (uint shard = 0; shard < p.shards; ++shard)
        sum += se[(shard * f.predictors + axis) * bins + b];
      extra[axis * bins + b] = sum;
    }
    device const uint *parent =
        reinterpret_cast<device const uint *>(dispatch + n.parent_extra_offset);
    if (sum != parent[axis])
      Error(dispatch, p, kArithmeticError);
  }
  if (axis == 0) {
    uint sum = 0;
    for (uint b = 0; b < bins; ++b) {
      for (uint shard = 0; shard < p.shards; ++shard)
        sum += so[shard * bins + b];
      occupied[b] = sum;
    }
    if (sum != n.rows)
      Error(dispatch, p, kArithmeticError);
  }
}

kernel void gjxl_histogram_costs(device const uchar *prepared [[buffer(0)]],
                                 device uchar *dispatch [[buffer(1)]],
                                 device uchar *scratch [[buffer(3)]],
                                 constant Params &p [[buffer(4)]],
                                 uint2 index [[thread_position_in_grid]]) {
  if (Failed(dispatch, p))
    return;
  const uint axis = index.x, slot = index.y;
  const TaskDescriptor task = Task(dispatch, p, slot);
  const NodeDescriptor n =
      reinterpret_cast<device const NodeDescriptor *>(dispatch)[task.node];
  const FitDescriptor f =
      reinterpret_cast<device const FitDescriptor *>(prepared)[n.fit];
  const uint bins = f.bins[task.property];
  if (axis >= bins * f.predictors)
    return;
  const uint pred = axis / bins, bin = axis % bins;
  device uchar *base = scratch + slot * p.slot_bytes;
  device const uint *occupied =
      reinterpret_cast<device const uint *>(base + p.occupancy);
  if (!Threshold(occupied, bin, n.rows))
    return;
  device const uint *counts =
      reinterpret_cast<device const uint *>(base + p.histograms) +
      axis * n.symbols;
  device const int *parent =
      reinterpret_cast<device const int *>(dispatch + n.parent_counts_offset) +
      pred * n.symbols;
  device const uint *parent_extra =
      reinterpret_cast<device const uint *>(dispatch + n.parent_extra_offset);
  device const uint *extra =
      reinterpret_cast<device const uint *>(base + p.extra);
  device CandidateCosts *costs =
      reinterpret_cast<device CandidateCosts *>(base + p.candidate_costs);
  const float low =
      Entropy(counts, parent, n.symbols, false) + float(extra[axis]);
  const float high =
      (Entropy(counts, parent, n.symbols, true) + float(parent_extra[pred])) -
      float(extra[axis]);
  costs[axis] = {low, high};
}

kernel void gjxl_histogram_select(device const uchar *prepared [[buffer(0)]],
                                  device uchar *dispatch [[buffer(1)]],
                                  device PropertyResult *output [[buffer(2)]],
                                  device uchar *scratch [[buffer(3)]],
                                  constant Params &p [[buffer(4)]],
                                  uint slot [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_threadgroup]]) {
  // No kernel writes the error word below, so this check is group-uniform.
  if (Failed(dispatch, p))
    return;
  const TaskDescriptor task = Task(dispatch, p, slot);
  const NodeDescriptor n =
      reinterpret_cast<device const NodeDescriptor *>(dispatch)[task.node];
  const FitDescriptor f =
      reinterpret_cast<device const FitDescriptor *>(prepared)[n.fit];
  const uint bins = f.bins[task.property];
  device uchar *base = scratch + slot * p.slot_bytes;
  device const uint *occupied =
      reinterpret_cast<device const uint *>(base + p.occupancy);
  device const CandidateCosts *costs =
      reinterpret_cast<device const CandidateCosts *>(base + p.candidate_costs);
  device const float *penalties = reinterpret_cast<device const float *>(
      dispatch + n.predictor_penalties_offset);
  device SideWinner *sides =
      reinterpret_cast<device SideWinner *>(base + p.side_winners);
  for (uint b = lane; b < bins; b += 128) {
    if (!Threshold(occupied, b, n.rows))
      continue;
    SideWinner low{kMaximumCost, 0, 0}, high{kMaximumCost, 0, 0};
    for (uint pred = 0; pred < f.predictors; ++pred) {
      const CandidateCosts c = costs[pred * bins + b];
      const float penalty = penalties[pred];
      if (c.low + penalty < low.cost + low.penalty)
        low = {c.low, penalty, f.predictor_ids[pred]};
      if (c.high + penalty < high.cost + high.penalty)
        high = {c.high, penalty, f.predictor_ids[pred]};
    }
    sides[2 * b] = low;
    sides[2 * b + 1] = high;
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (lane == 0) {
    PropertyResult result = output[n.result_first + task.property];
    uint thresholds = 0;
    for (uint b = 0; b < bins; ++b) {
      if (!Threshold(occupied, b, n.rows))
        continue;
      ++thresholds;
      const SideWinner low = sides[2 * b], high = sides[2 * b + 1];
      const bool weighted = f.property_ids[task.property] == 15 ||
                            low.predictor == 6 || high.predictor == 6;
      const uint category =
          f.property_ids[task.property] < 2
              ? (low.cost == 0 || high.cost == 0 ? kStaticConstant : kStatic)
              : (weighted ? kNonstatic : kNonstaticWithoutWeighted);
      const Candidate current = result.winners[category];
      if (low.cost + high.cost < current.left_cost + current.right_cost)
        result.winners[category] = {b,
                                    occupied[b],
                                    low.predictor,
                                    high.predictor,
                                    low.cost,
                                    high.cost,
                                    1,
                                    0};
    }
    result.is_constant = thresholds == 0;
    output[n.result_first + task.property] = result;
  }
}
