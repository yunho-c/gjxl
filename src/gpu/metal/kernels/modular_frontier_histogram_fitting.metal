// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Included after the depth-first resident executor; shares exact arithmetic
// and the native two-pointer partition permutation helper.
namespace frontier_impl {
using namespace resident_impl;
void Fail(device FrontierControl &control, uint error) {
  atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(&control.status), error, memory_order_relaxed);
}
}

kernel void gjxl_histogram_frontier_init(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::FrontierParams &p [[buffer(2)]],
    uint fit [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using namespace frontier_impl;
  device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[fit];
  const Fit d = reinterpret_cast<device const Fit *>(work)[fit];
  if (!lane) {
    *reinterpret_cast<device State *>(work + d.state) = {0, 1, 0, 0, 0, 0, 0, 0, 0};
    reinterpret_cast<device Node *>(work + d.nodes)[0] = {-1, 0, 0, 0, f.predictor_ids[0], 0, f.rows, 1, 0};
    reinterpret_cast<device NodeRef *>(work + p.queue)[fit] = {fit, 0};
    if (!fit) *reinterpret_cast<device FrontierControl *>(work + p.control) = {0, 0, p.fits, 0, 0, 0, 0, 0, 0, {1, 1, 1}};
  }
  device uint *order = reinterpret_cast<device uint *>(work + d.order);
  for (uint i = lane; i < f.rows; i += 128) order[i] = i;
}

kernel void gjxl_histogram_frontier_begin(
    device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::FrontierParams &p [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]) {
  using namespace frontier_impl;
  if (lane) return;
  device FrontierControl &c = *reinterpret_cast<device FrontierControl *>(work + p.control);
  c.first = c.head;
  c.count = c.status ? 0 : min(p.active_nodes, c.tail - c.head);
  c.maximum_ready = max(c.maximum_ready, c.tail - c.head);
  c.maximum_active = max(c.maximum_active, c.count);
  uint counts[64] = {};
  device const NodeRef *queue = reinterpret_cast<device const NodeRef *>(work + p.queue);
  for (uint i = 0; i < c.count; ++i)
    c.maximum_same_fit = max(c.maximum_same_fit, ++counts[queue[c.first + i].fit]);
  c.head += c.count;
  c.rounds += c.count != 0;
  c.dispatch[0] = max(1u, c.count); c.dispatch[1] = c.dispatch[2] = 1;
}

// One group per READY NODE, including siblings from the same tree. No group
// waits on another group: the next dispatch is the only producer/consumer
// boundary. Scratch is indexed by active slot, not retained per tree node.
kernel void gjxl_histogram_frontier_nodes(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::FrontierParams &p [[buffer(2)]],
    uint slot [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using namespace frontier_impl;
  device FrontierControl &control = *reinterpret_cast<device FrontierControl *>(work + p.control);
  if (slot >= control.count) return;
  device NodeRef *queue = reinterpret_cast<device NodeRef *>(work + p.queue);
  const NodeRef ref = queue[control.first + slot];
  device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[ref.fit];
  Fit d = reinterpret_cast<device const Fit *>(work)[ref.fit];
  device State &fit_state = *reinterpret_cast<device State *>(work + d.state);
  const uint base = p.scratch + slot * p.slot.bytes;
  d.state = base + p.slot.state;
  d.parent = base + p.slot.parent; d.parent_extra = base + p.slot.parent_extra;
  d.histogram = base + p.slot.histogram; d.extra = base + p.slot.extra;
  d.occupancy = base + p.slot.occupancy; d.sides = base + p.slot.sides;
  device State &state = *reinterpret_cast<device State *>(work + d.state);
  device Node *nodes = reinterpret_cast<device Node *>(work + d.nodes);
  device Node &node = nodes[ref.node];
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
  // After parent publication all 3584 counters become tile scratch: 3072
  // joint cells plus 256 extra-bit and 256 occupancy bins. Keep the footprint
  // below 15 KiB, including room for Metal shader-validation instrumentation.
  threadgroup atomic_uint *bin_bits = parent_counts + kHistogramTileCells;
  threadgroup atomic_uint *bin_occupancy = bin_bits + 256;
  threadgroup uint groups[4];
  threadgroup Split best[4], chosen;
  if (!lane) state = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  threadgroup_barrier(mem_flags::mem_device);
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
      atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(&fit_state.visited), 1, memory_order_relaxed);
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
          // Reuse the parent threadgroup histogram after its immutable copy.
          // Bounded tiles replace contended device-memory increments; integer
          // sums and the subsequent exact scoring order are unchanged.
          device const uchar2 *tokens = reinterpret_cast<device const uchar2 *>(prepared + f.residual_offsets[pred]);
          const uint tile_bins = min(256u, kHistogramTileCells / symbols);
          for (uint first = 0; first < bins; first += tile_bins) {
            const uint width = min(tile_bins, bins - first);
            for (uint i = lane; i < width * symbols; i += 128)
              atomic_store_explicit(&parent_counts[i], 0, memory_order_relaxed);
            for (uint i = lane; i < width; i += 128) {
              atomic_store_explicit(&bin_bits[i], 0, memory_order_relaxed);
              atomic_store_explicit(&bin_occupancy[i], 0, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = node.begin + lane; i < node.end; i += 128) {
              const uint row = order[i];
              const uint bin = Property(prepared, f, prop, row);
              if (bin < first || bin >= first + width) continue;
              const uint local = bin - first, weight = weights[row];
              const uchar2 token = tokens[row];
              atomic_fetch_add_explicit(&parent_counts[local * symbols + token.x], weight, memory_order_relaxed);
              atomic_fetch_add_explicit(&bin_bits[local], weight * uint(token.y), memory_order_relaxed);
              if (!pred) atomic_fetch_add_explicit(&bin_occupancy[local], 1, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < width * symbols; i += 128)
              hist[first * symbols + i] = atomic_load_explicit(&parent_counts[i], memory_order_relaxed);
            for (uint i = lane; i < width; i += 128) {
              extra[first + i] = atomic_load_explicit(&bin_bits[i], memory_order_relaxed);
              if (!pred) occupancy[first + i] = atomic_load_explicit(&bin_occupancy[i], memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
          }
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
        if (node.depth >= d.maximum_depth) state.status = kCapacityError;
        else {
          const uint first = atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(&fit_state.size), 2, memory_order_relaxed);
          if (first + 2 > d.maximum_nodes) state.status = kCapacityError;
          else {
            const uint cut = node.begin + chosen.candidate.low_rows;
            node.property = int(f.property_ids[chosen.property]);
            node.split = reinterpret_cast<device const int *>(work + d.thresholds[chosen.property])[chosen.candidate.bin];
            node.left = first; node.right = first + 1;
            nodes[first] = {-1, 0, 0, 0, chosen.candidate.right_predictor, cut, node.end, node.depth + 1, node.constants};
            nodes[first + 1] = {-1, 0, 0, 0, chosen.candidate.left_predictor, node.begin, cut, node.depth + 1, node.constants};
          }
        }
        if (state.status) Fail(control, state.status);
      }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (state.status || !chosen.candidate.valid) return;
    const uint cut = node.begin + chosen.candidate.low_rows;
    Misplaced(prepared, f, order, partition, state.low_count, node.begin, cut,
              chosen.property, chosen.candidate.bin, true, lane, groups);
    Misplaced(prepared, f, order, partition, state.high_count, cut, node.end,
              chosen.property, chosen.candidate.bin, false, lane, groups);
    if (!lane && state.low_count != state.high_count) {
      state.status = kPartitionError; Fail(control, state.status);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (state.status) return;
    for (uint i = lane; i < state.low_count; i += 128) {
      const uint low = partition[node.begin + i], high = partition[cut + i];
      const uint temp = order[low]; order[low] = order[high]; order[high] = temp;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (!lane) {
      const uint first = atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(&control.tail), 2, memory_order_relaxed);
      if (first + 2 > p.queue_capacity) { Fail(control, kCapacityError); return; }
      queue[first] = {ref.fit, node.left}; queue[first + 1] = {ref.fit, node.right};
    }
}
