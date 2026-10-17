// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Included by the exact histogram translation unit, sharing its NativeLog2,
// Entropy and Threshold arithmetic and strict floating-point compiler flags.
#include "../../ops/modular_resident_histogram_layout.h"
namespace resident_impl {
using namespace gjxl_histogram_resident;
struct Split { uint property; Candidate candidate; };
float Cost(Split s) { return s.candidate.left_cost + s.candidate.right_cost; }
uint Property(device const uchar *prepared, device const FitDescriptor &f, uint prop, uint row) {
  device const uchar *column = prepared + f.property_offsets[prop];
  return prop < f.static_properties ? reinterpret_cast<device const uint *>(column)[row]
                                     : column[row];
}
// Pair the ascending misplaced positions in the two halves. This is the exact
// native two-pointer swap permutation, with parallel discovery and swaps.
void Misplaced(device const uchar *prepared, device const FitDescriptor &f,
               device const uint *order, device uint *positions,
               device uint &count, uint begin, uint end, uint property,
               uint bin, bool low, uint lane, threadgroup uint *groups) {
  for (uint first = begin; first < end; first += 128) {
    const uint index = first + lane;
    const bool bad = index < end &&
        ((Property(prepared, f, property, order[index]) <= bin) != low);
    const uint prefix = simd_prefix_exclusive_sum(uint(bad));
    const uint population = simd_sum(uint(bad));
    if ((lane & 31) == 0) groups[lane >> 5] = population;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint offset = 0;
    for (uint i = 0; i < (lane >> 5); ++i) offset += groups[i];
    if (bad) positions[begin + count + offset + prefix] = index;
    threadgroup_barrier(mem_flags::mem_device);
    if (!lane) count += groups[0] + groups[1] + groups[2] + groups[3];
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
}
} // namespace resident_impl

kernel void gjxl_histogram_resident_fit(
    device const uchar *prepared [[buffer(0)]],
    device uchar *work [[buffer(1)]],
    uint fit_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  using namespace resident_impl;
  device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[fit_index];
  const Fit d = reinterpret_cast<device const Fit *>(work)[fit_index];
  device State &state = *reinterpret_cast<device State *>(work + d.state);
  device Node *nodes = reinterpret_cast<device Node *>(work + d.nodes);
  device uint *stack = reinterpret_cast<device uint *>(work + d.stack);
  device uint *order = reinterpret_cast<device uint *>(work + d.order);
  device uint *partition = reinterpret_cast<device uint *>(work + d.partition);
  device uint *parent = reinterpret_cast<device uint *>(work + d.parent);
  device uint *parent_extra = reinterpret_cast<device uint *>(work + d.parent_extra);
  device uint *hist = reinterpret_cast<device uint *>(work + d.histogram);
  device uint *extra = reinterpret_cast<device uint *>(work + d.extra);
  device uint *occupancy = reinterpret_cast<device uint *>(work + d.occupancy);
  device SideWinner *sides = reinterpret_cast<device SideWinner *>(work + d.sides);
  device const float *penalties = reinterpret_cast<device const float *>(work + d.penalties);
  device const ushort *weights = reinterpret_cast<device const ushort *>(prepared + f.weights_offset);
  threadgroup atomic_uint parent_counts[14 * 256], parent_bits[14], max_token;
  threadgroup uint groups[4];
  threadgroup bool running;
  threadgroup Split best[4], chosen;
  if (!lane) {
    state = {0, 1, 1, 0, 0, 0, 0, 0, 0};
    nodes[0] = {-1, 0, 0, 0, f.predictor_ids[0], 0, f.rows, 1, 0};
    stack[0] = 0;
  }
  for (uint row = lane; row < f.rows; row += 128) order[row] = row;
  threadgroup_barrier(mem_flags::mem_device);
  for (;;) {
    if (!lane) {
      running = state.pending && !state.status;
      if (running) state.active = stack[--state.pending];
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (!running) break;
    device Node &node = nodes[state.active];
    if (node.begin == node.end) continue;
    for (uint i = lane; i < f.predictors * 256; i += 128)
      atomic_store_explicit(&parent_counts[i], 0, memory_order_relaxed);
    if (lane < f.predictors) atomic_store_explicit(&parent_bits[lane], 0, memory_order_relaxed);
    if (!lane) atomic_store_explicit(&max_token, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint maximum = 0;
    for (uint i = node.begin + lane; i < node.end; i += 128) {
      const uint row = order[i], weight = weights[row];
      for (uint pred = 0; pred < f.predictors; ++pred) {
        const uchar2 token = reinterpret_cast<device const uchar2 *>(prepared + f.residual_offsets[pred])[row];
        maximum = max(maximum, uint(token.x) + 1);
        atomic_fetch_add_explicit(&parent_counts[pred * 256 + token.x], weight, memory_order_relaxed);
        atomic_fetch_add_explicit(&parent_bits[pred], weight * uint(token.y), memory_order_relaxed);
      }
    }
    atomic_fetch_max_explicit(&max_token, maximum, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < f.predictors * 256; i += 128)
      parent[i] = atomic_load_explicit(&parent_counts[i], memory_order_relaxed);
    if (lane < f.predictors) parent_extra[lane] = atomic_load_explicit(&parent_bits[lane], memory_order_relaxed);
    if (!lane) {
      state.symbols = (atomic_load_explicit(&max_token, memory_order_relaxed) + 3) & ~3u;
      ++state.visited;
      for (uint c = 0; c < 4; ++c)
        best[c] = {0, {0, 0, 0, 0, kMaximumCost, kMaximumCost, 0, 0}};
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint parent_index = 0;
    while (f.predictor_ids[parent_index] != node.predictor) ++parent_index;
    if (!lane)
      state.base_cost = Entropy(parent + parent_index * 256,
          reinterpret_cast<device const int *>(parent + parent_index * 256), state.symbols, false) +
          float(parent_extra[parent_index]);
    threadgroup_barrier(mem_flags::mem_device);
    if (state.base_cost > d.split_threshold) {
      for (uint prop = 0; prop < f.properties; ++prop) {
        const uint bins = f.bins[prop], symbols = state.symbols;
        if ((node.constants & (1u << prop)) || bins == 1) {
          if (!lane) node.constants |= 1u << prop;
          threadgroup_barrier(mem_flags::mem_device);
          continue;
        }
        for (uint b = lane; b < bins; b += 128) {
          occupancy[b] = 0;
          sides[2 * b] = sides[2 * b + 1] = {kMaximumCost, 0, 0};
        }
        for (uint pred = 0; pred < f.predictors; ++pred) {
          for (uint i = lane; i < bins * symbols; i += 128) hist[i] = 0;
          for (uint b = lane; b < bins; b += 128) extra[b] = 0;
          threadgroup_barrier(mem_flags::mem_device);
          device const uchar2 *tokens = reinterpret_cast<device const uchar2 *>(prepared + f.residual_offsets[pred]);
          for (uint i = node.begin + lane; i < node.end; i += 128) {
            const uint row = order[i], weight = weights[row];
            const uint bin = Property(prepared, f, prop, row);
            const uchar2 token = tokens[row];
            atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(hist + bin * symbols + token.x), weight, memory_order_relaxed);
            atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(extra + bin), weight * uint(token.y), memory_order_relaxed);
            if (!pred) atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(occupancy + bin), 1, memory_order_relaxed);
          }
          threadgroup_barrier(mem_flags::mem_device);
          for (uint sym = lane; sym < symbols; sym += 128) {
            uint sum = 0;
            for (uint b = 0; b < bins; ++b) { sum += hist[b * symbols + sym]; hist[b * symbols + sym] = sum; }
          }
          if (!lane) {
            uint sum = 0, used = 0;
            for (uint b = 0; b < bins; ++b) {
              sum += extra[b]; extra[b] = sum;
              if (!pred) { used += occupancy[b]; occupancy[b] = used; }
            }
            if (!pred) {
              bool varying = false;
              for (uint b = 0; b < bins; ++b) varying |= Threshold(occupancy, b, node.end - node.begin);
              if (!varying) node.constants |= 1u << prop;
            }
          }
          threadgroup_barrier(mem_flags::mem_device);
          if (node.constants & (1u << prop)) break;
          const float penalty = penalties[parent_index * f.predictors + pred];
          for (uint b = lane; b < bins; b += 128) {
            if (!Threshold(occupancy, b, node.end - node.begin)) continue;
            const float low = Entropy(hist + b * symbols, reinterpret_cast<device const int *>(parent + pred * 256), symbols, false) + float(extra[b]);
            const float high = Entropy(hist + b * symbols, reinterpret_cast<device const int *>(parent + pred * 256), symbols, true) + float(parent_extra[pred]) - float(extra[b]);
            if (low + penalty < sides[2 * b].cost + sides[2 * b].penalty)
              sides[2 * b] = {low, penalty, f.predictor_ids[pred]};
            if (high + penalty < sides[2 * b + 1].cost + sides[2 * b + 1].penalty)
              sides[2 * b + 1] = {high, penalty, f.predictor_ids[pred]};
          }
          threadgroup_barrier(mem_flags::mem_device);
        }
        if (!lane && !(node.constants & (1u << prop))) {
          for (uint b = 0; b < bins; ++b) {
            if (!Threshold(occupancy, b, node.end - node.begin)) continue;
            const SideWinner low = sides[2 * b], high = sides[2 * b + 1];
            const bool wp = f.property_ids[prop] == 15 || low.predictor == 6 || high.predictor == 6;
            const uint category = f.property_ids[prop] < 2 ? (low.cost == 0 || high.cost == 0 ? 0 : 1) : (wp ? 2 : 3);
            if (low.cost + high.cost < Cost(best[category]))
              best[category] = {prop, {b, occupancy[b], low.predictor, high.predictor, low.cost, high.cost, 1, 0}};
          }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
      }
    }
    if (!lane) {
      chosen = best[2];
      if (Cost(best[3]) + d.split_threshold < state.base_cost && Cost(best[3]) <= d.fast_decode_multiplier * Cost(chosen)) chosen = best[3];
      if (Cost(best[1]) + d.split_threshold < state.base_cost && Cost(best[1]) <= d.fast_decode_multiplier * Cost(chosen)) chosen = best[1];
      if (Cost(best[0]) + d.split_threshold < state.base_cost) chosen = best[0];
      if (!(Cost(chosen) + d.split_threshold < state.base_cost)) chosen.candidate.valid = 0;
      if (chosen.candidate.valid) {
        if (state.size + 2 > d.maximum_nodes || node.depth >= d.maximum_depth) state.status = kCapacityError;
        else {
          const uint cut = node.begin + chosen.candidate.low_rows;
          node.property = int(f.property_ids[chosen.property]);
          node.split = reinterpret_cast<device const int *>(work + d.thresholds[chosen.property])[chosen.candidate.bin];
          node.left = state.size; node.right = state.size + 1;
          nodes[state.size++] = {-1, 0, 0, 0, chosen.candidate.right_predictor, cut, node.end, node.depth + 1, node.constants};
          nodes[state.size++] = {-1, 0, 0, 0, chosen.candidate.left_predictor, node.begin, cut, node.depth + 1, node.constants};
          state.low_count = state.high_count = 0;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (state.status) break;
    if (!chosen.candidate.valid) continue;
    const uint cut = node.begin + chosen.candidate.low_rows;
    Misplaced(prepared, f, order, partition, state.low_count, node.begin, cut,
              chosen.property, chosen.candidate.bin, true, lane, groups);
    Misplaced(prepared, f, order, partition, state.high_count, cut, node.end,
              chosen.property, chosen.candidate.bin, false, lane, groups);
    if (!lane && state.low_count != state.high_count) state.status = kPartitionError;
    threadgroup_barrier(mem_flags::mem_device);
    if (state.status) break;
    for (uint i = lane; i < state.low_count; i += 128) {
      const uint low = partition[node.begin + i], high = partition[cut + i];
      const uint temp = order[low]; order[low] = order[high]; order[high] = temp;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (!lane) { stack[state.pending++] = node.right; stack[state.pending++] = node.left; }
    threadgroup_barrier(mem_flags::mem_device);
  }
}
