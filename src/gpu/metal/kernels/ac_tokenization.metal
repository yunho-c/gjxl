// SPDX-License-Identifier: Apache-2.0
// Experimental exact AC token preparation. One SIMD group owns a
// transform/channel.
#include <metal_stdlib>
using namespace metal;
struct AcTokenAnchor {
  uint source, group, packed, reserved;
};
struct AcTokenGroup {
  uint first, anchors, output, map, width, height, tokens, reserved;
};
struct AcTokenStrategy {
  uint count, covered, width, height, order0, order1, order2, log2covered;
};
struct AcTokenMetadata {
  uint nonzeros, last, count, output;
};
constant ushort kCoefficientFrequencyContext[64] = {
    0xBAD, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15,    15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22,
    23,    23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26,
    27,    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
};
constant ushort kCoefficientNonzeroContext[64] = {
    0xBAD, 0,   31,  62,  62,  93,  93,  93,  93,  123, 123, 123, 123,
    152,   152, 152, 152, 152, 152, 152, 152, 180, 180, 180, 180, 180,
    180,   180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
    206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
};
inline uint ac_token_channel(uint task) {
  const uint c = task % 3;
  return c < 2 ? c ^ 1u : 2u;
}
inline uint ac_token_order(AcTokenStrategy s, uint c) {
  return c == 0 ? s.order0 : c == 1 ? s.order1 : s.order2;
}
inline uint ac_token_signed(int value) {
  return (uint(value) << 1) ^ uint(value >> 31);
}

kernel void
gjxl_ac_token_metadata(device const int *coefficients [[buffer(0)]],
                       device const AcTokenAnchor *anchors [[buffer(1)]],
                       device const AcTokenGroup *groups [[buffer(2)]],
                       device const AcTokenStrategy *strategies [[buffer(3)]],
                       device const uint *orders [[buffer(4)]],
                       device AcTokenMetadata *metadata [[buffer(5)]],
                       device uchar *maps [[buffer(6)]],
                       constant uint &tasks [[buffer(7)]],
                       uint index [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
  const uint task = index / 32;
  if (task >= tasks)
    return;
  const AcTokenAnchor a = anchors[task / 3];
  const AcTokenStrategy s = strategies[(a.packed >> 10) & 31u];
  const uint channel = ac_token_channel(task),
             order = ac_token_order(s, channel);
  uint count = 0, last = 0;
  for (uint j = s.covered + lane; j < s.count; j += 32) {
    const int value =
        coefficients[a.source + channel * 65536u + orders[order + j]];
    count += value != 0;
    if (value != 0)
      last = j;
  }
  count = simd_sum(count);
  last = simd_max(last);
  if (lane == 0)
    metadata[task] = {count, last, count == 0 ? 1u : last - s.covered + 2u, 0u};
  const AcTokenGroup g = groups[a.group];
  const uint x = a.packed & 31u, y = (a.packed >> 5) & 31u;
  if (lane < s.covered)
    maps[g.map + channel * g.width * g.height + (y + lane / s.width) * g.width +
         x + lane % s.width] = uchar((count + s.covered - 1) / s.covered);
}

kernel void gjxl_ac_token_offsets(device AcTokenGroup *groups [[buffer(0)]],
                                  device AcTokenMetadata *metadata
                                  [[buffer(1)]],
                                  uint group [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  const AcTokenGroup g = groups[group];
  uint offset = 0;
  for (uint first = 0; first < 3 * g.anchors; first += 32) {
    const uint local = first + lane, task = 3 * g.first + local;
    const uint count = local < 3 * g.anchors ? metadata[task].count : 0;
    const uint prefix = simd_prefix_exclusive_sum(count);
    if (local < 3 * g.anchors)
      metadata[task].output = offset + prefix;
    offset += simd_sum(count);
  }
  if (lane == 0)
    groups[group].tokens = offset;
}

kernel void
gjxl_ac_token_group_offsets(device AcTokenGroup *groups [[buffer(0)]],
                            device uint *control [[buffer(1)]],
                            constant uint &group_count [[buffer(2)]],
                            uint lane [[thread_index_in_simdgroup]]) {
  uint offset = 0;
  for (uint first = 0; first < group_count; first += 32) {
    const uint group = first + lane;
    const uint count = group < group_count ? groups[group].tokens : 0;
    const uint prefix = simd_prefix_exclusive_sum(count);
    if (group < group_count)
      groups[group].output = offset + prefix;
    offset += simd_sum(count);
  }
  const uint end = 4 + control[1] - 1 + 3 * 13 * control[1];
  if (lane == 0)
    control[end + 2] = offset;
}

inline void ac_token_population(uint value, uint context, uint shard,
                                uint contexts, device atomic_uint *histogram) {
  uint symbol = value;
  if (value >= 16) {
    const uint exponent = 31 - clz(value);
    symbol = 16 + ((exponent - 4) << 2) +
             ((value - (1u << exponent)) >> (exponent - 2));
  }
  atomic_fetch_add_explicit(histogram + (shard * contexts + context) * 128 +
                                symbol,
                            1u, memory_order_relaxed);
}

kernel void
gjxl_ac_token_emit(device const int *coefficients [[buffer(0)]],
                   device const AcTokenAnchor *anchors [[buffer(1)]],
                   device const AcTokenGroup *groups [[buffer(2)]],
                   device const AcTokenStrategy *strategies [[buffer(3)]],
                   device const uint *orders [[buffer(4)]],
                   device const AcTokenMetadata *metadata [[buffer(5)]],
                   device const uchar *maps [[buffer(6)]],
                   device uint *values [[buffer(7)]],
                   device ushort *contexts_out [[buffer(8)]],
                   device atomic_uint *histogram [[buffer(9)]],
                   device const uint *control [[buffer(10)]],
                   uint index [[thread_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]]) {
  const uint task = index / 32;
  if (task >= control[3])
    return;
  const uint control_end = 4 + control[1] - 1 + 3 * 13 * control[1];
  // A speculative compact output is published only if the whole stream fits.
  if (control[control_end + 3] &&
      control[control_end + 2] > control[control_end + 1])
    return;
  const AcTokenAnchor a = anchors[task / 3];
  const uint strategy = (a.packed >> 10) & 31u;
  if (strategy == 0 && control[control_end + 4])
    return;
  const AcTokenStrategy s = strategies[strategy];
  const AcTokenGroup g = groups[a.group];
  const AcTokenMetadata m = metadata[task];
  const uint channel = ac_token_channel(task),
             order = ac_token_order(s, channel);
  const uint x = a.packed & 31u, y = (a.packed >> 5) & 31u;
  const uint num_contexts = control[0], segments = control[1],
             shards = control[2];
  const uint raw_quant = ((a.packed >> 15) & 255u) + 1u;
  uint segment = 0;
  while (segment + 1 < segments && raw_quant > control[4 + segment])
    ++segment;
  const uint family = strategy == 0                    ? 0
                      : strategy == 4                  ? 2
                      : strategy == 5                  ? 3
                      : strategy == 6 || strategy == 7 ? 4
                                                       : 6;
  const uint channel_row = channel < 2 ? channel ^ 1u : 2u;
  const uint block_context =
      control[4 + segments - 1 + (channel_row * 13 + family) * segments +
              segment];
  const uint shard = (task / 3) & (shards - 1);
  if (lane == 0) {
    const uint base = g.map + channel * g.width * g.height;
    const uint prediction =
        x == 0   ? (y == 0 ? 32u : uint(maps[base + (y - 1) * g.width]))
        : y == 0 ? uint(maps[base + x - 1])
                 : (uint(maps[base + (y - 1) * g.width + x]) +
                    uint(maps[base + y * g.width + x - 1]) + 1) /
                       2;
    const uint bucket = prediction < 8     ? prediction
                        : prediction >= 64 ? 36
                                           : 4 + prediction / 2;
    const uint context = bucket * num_contexts + block_context;
    values[g.output + m.output] = m.nonzeros;
    contexts_out[g.output + m.output] = ushort(context);
    if (control[4 + segments - 1 + 3 * 13 * segments])
      ac_token_population(m.nonzeros, context, shard, num_contexts * 495,
                          histogram);
  }
  uint consumed = 0, previous = m.nonzeros > s.count / 16 ? 0u : 1u;
  for (uint first = s.covered; m.nonzeros != 0 && first <= m.last;
       first += 32) {
    const uint j = first + lane;
    const bool active = j <= m.last;
    const int coefficient =
        active ? coefficients[a.source + channel * 65536u + orders[order + j]]
               : 0;
    const uint nz = uint(coefficient != 0);
    const uint before = simd_prefix_exclusive_sum(nz);
    const uint remaining = m.nonzeros - consumed - before;
    const uint prev = lane == 0 ? previous : simd_shuffle_up(nz, 1);
    if (active) {
      const uint local =
          (kCoefficientNonzeroContext[(remaining + s.covered - 1) >>
                                      s.log2covered] +
           kCoefficientFrequencyContext[j >> s.log2covered]) *
              2 +
          prev;
      const uint context = num_contexts * 37 + 458 * block_context + local;
      const uint value = ac_token_signed(coefficient);
      values[g.output + m.output + 1 + j - s.covered] = value;
      contexts_out[g.output + m.output + 1 + j - s.covered] = ushort(context);
      if (control[4 + segments - 1 + 3 * 13 * segments])
        ac_token_population(value, context, shard, num_contexts * 495,
                            histogram);
    }
    consumed += simd_sum(nz);
    previous = simd_shuffle(nz, 31);
  }
}

// DCT8 often emits only a few coefficients. One lane per channel/transform
// avoids reserving 32 lanes for that short serial state recurrence.
kernel void gjxl_ac_token_emit_scalar_dct8(
    device const int *coefficients [[buffer(0)]],
    device const AcTokenAnchor *anchors [[buffer(1)]],
    device const AcTokenGroup *groups [[buffer(2)]],
    device const AcTokenStrategy *strategies [[buffer(3)]],
    device const uint *orders [[buffer(4)]],
    device const AcTokenMetadata *metadata [[buffer(5)]],
    device const uchar *maps [[buffer(6)]], device uint *values [[buffer(7)]],
    device ushort *contexts_out [[buffer(8)]],
    device atomic_uint *histogram [[buffer(9)]],
    device const uint *control [[buffer(10)]],
    uint task [[thread_position_in_grid]]) {
  if (task >= control[3])
    return;
  const uint end = 4 + control[1] - 1 + 3 * 13 * control[1];
  if (control[end + 3] && control[end + 2] > control[end + 1])
    return;
  const AcTokenAnchor a = anchors[task / 3];
  if (((a.packed >> 10) & 31u) != 0)
    return;
  const AcTokenStrategy s = strategies[0];
  const AcTokenGroup g = groups[a.group];
  const AcTokenMetadata m = metadata[task];
  const uint channel = ac_token_channel(task),
             order = ac_token_order(s, channel);
  const uint x = a.packed & 31u, y = (a.packed >> 5) & 31u;
  const uint num_contexts = control[0], segments = control[1],
             shards = control[2];
  const uint raw_quant = ((a.packed >> 15) & 255u) + 1u;
  uint segment = 0;
  while (segment + 1 < segments && raw_quant > control[4 + segment])
    ++segment;
  const uint channel_row = channel < 2 ? channel ^ 1u : 2u;
  const uint block_context =
      control[4 + segments - 1 + (channel_row * 13) * segments + segment];
  const uint shard = (task / 3) & (shards - 1);
  const uint base = g.map + channel * g.width * g.height;
  const uint prediction =
      x == 0   ? (y == 0 ? 32u : uint(maps[base + (y - 1) * g.width]))
      : y == 0 ? uint(maps[base + x - 1])
               : (uint(maps[base + (y - 1) * g.width + x]) +
                  uint(maps[base + y * g.width + x - 1]) + 1) /
                     2;
  const uint bucket = prediction < 8     ? prediction
                      : prediction >= 64 ? 36
                                         : 4 + prediction / 2;
  const uint context = bucket * num_contexts + block_context;
  const uint output = g.output + m.output;
  values[output] = m.nonzeros;
  contexts_out[output] = ushort(context);
  if (control[end])
    ac_token_population(m.nonzeros, context, shard, num_contexts * 495,
                        histogram);
  uint remaining = m.nonzeros, previous = remaining > 4 ? 0u : 1u;
  for (uint j = 1; remaining != 0 && j <= m.last; ++j) {
    const int coefficient =
        coefficients[a.source + channel * 65536u + orders[order + j]];
    const uint local = (kCoefficientNonzeroContext[remaining] +
                        kCoefficientFrequencyContext[j]) *
                           2 +
                       previous;
    const uint coefficient_context =
        num_contexts * 37 + 458 * block_context + local;
    const uint value = ac_token_signed(coefficient);
    values[output + j] = value;
    contexts_out[output + j] = ushort(coefficient_context);
    if (control[end])
      ac_token_population(value, coefficient_context, shard, num_contexts * 495,
                          histogram);
    previous = uint(coefficient != 0);
    remaining -= previous;
  }
}

// Pure-DCT8 experiment: one threadgroup owns an AC group. Atomic allocation
// only chooses each group's physical address; logical group/token order stays
// fixed in the CPU's group-indexed views. No intra-stream atomic append occurs.
kernel void
gjxl_ac_token_group_dct8(device const int *coefficients [[buffer(0)]],
                         device const AcTokenAnchor *anchors [[buffer(1)]],
                         device AcTokenGroup *groups [[buffer(2)]],
                         device const AcTokenStrategy *strategies [[buffer(3)]],
                         device const uint *orders [[buffer(4)]],
                         device uint *values [[buffer(7)]],
                         device ushort *contexts_out [[buffer(8)]],
                         device atomic_uint *histogram [[buffer(9)]],
                         device uint *control [[buffer(10)]],
                         uint group [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint group_size [[threads_per_threadgroup]]) {
  threadgroup ushort local_metadata[3072];
  threadgroup uint simd_totals[8];
  threadgroup uint group_output, group_tokens;
  AcTokenGroup g = groups[group];
  const AcTokenStrategy s = strategies[0];
  const uint count = 3 * g.anchors, width = group_size;
  const uint simd = tid / 32;
  for (uint local = simd; local < count; local += width / 32) {
    const uint task = 3 * g.first + local;
    const AcTokenAnchor a = anchors[task / 3];
    const uint channel = ac_token_channel(task);
    const uint order = ac_token_order(s, channel);
    uint nz = 0, last = 0;
    for (uint j = 1 + lane; j < 64; j += 32) {
      const int v =
          coefficients[a.source + channel * 65536u + orders[order + j]];
      nz += v != 0;
      if (v != 0)
        last = j;
    }
    nz = simd_sum(nz);
    last = simd_max(last);
    if (lane == 0)
      local_metadata[local] = ushort(nz | (last << 6));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint chunk = (count + width - 1) / width;
  const uint first = min(tid * chunk, count), stop = min(first + chunk, count);
  uint subtotal = 0;
  for (uint local = first; local < stop; ++local)
    subtotal += 1 + (uint(local_metadata[local]) >> 6);
  uint prefix = simd_prefix_exclusive_sum(subtotal);
  const uint simd_total = simd_sum(subtotal);
  if (lane == 0)
    simd_totals[simd] = simd_total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0; i < simd; ++i)
    prefix += simd_totals[i];
  const uint end = 4 + control[1] - 1 + 3 * 13 * control[1];
  if (tid == 0) {
    uint total = 0;
    for (uint i = 0; i < width / 32; ++i)
      total += simd_totals[i];
    group_tokens = total;
    group_output = atomic_fetch_add_explicit(
        reinterpret_cast<device atomic_uint *>(control + end + 2), total,
        memory_order_relaxed);
    groups[group].tokens = total;
    groups[group].output = group_output;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (group_output > control[end + 1] ||
      group_tokens > control[end + 1] - group_output)
    return;
  g.output = group_output;
  for (uint local = first; local < stop; ++local) {
    const uint task = 3 * g.first + local;
    const AcTokenAnchor a = anchors[task / 3];
    const uint packed = local_metadata[local];
    const AcTokenMetadata m = {packed & 63u, packed >> 6, 1 + (packed >> 6),
                               prefix};
    const uint channel = ac_token_channel(task),
               order = ac_token_order(s, channel);
    const uint x = a.packed & 31u, y = (a.packed >> 5) & 31u;
    const uint num_contexts = control[0], segments = control[1],
               shards = control[2];
    const uint raw_quant = ((a.packed >> 15) & 255u) + 1u;
    uint segment = 0;
    while (segment + 1 < segments && raw_quant > control[4 + segment])
      ++segment;
    const uint channel_row = channel < 2 ? channel ^ 1u : 2u;
    const uint block_context =
        control[4 + segments - 1 + (channel_row * 13) * segments + segment];
    const uint shard = (task / 3) & (shards - 1);
    const uint prediction =
        x == 0
            ? (y == 0 ? 32u : uint(local_metadata[local - 3 * g.width] & 63u))
        : y == 0 ? uint(local_metadata[local - 3] & 63u)
                 : (uint(local_metadata[local - 3 * g.width] & 63u) +
                    uint(local_metadata[local - 3] & 63u) + 1) /
                       2;
    const uint bucket = prediction < 8     ? prediction
                        : prediction >= 64 ? 36
                                           : 4 + prediction / 2;
    const uint context = bucket * num_contexts + block_context;
    const uint output = g.output + m.output;
    values[output] = m.nonzeros;
    contexts_out[output] = ushort(context);
    if (control[end])
      ac_token_population(m.nonzeros, context, shard, num_contexts * 495,
                          histogram);
    uint remaining = m.nonzeros, previous = remaining > 4 ? 0u : 1u;
    for (uint j = 1; remaining != 0 && j <= m.last; ++j) {
      const int coefficient =
          coefficients[a.source + channel * 65536u + orders[order + j]];
      const uint local = (kCoefficientNonzeroContext[remaining] +
                          kCoefficientFrequencyContext[j]) *
                             2 +
                         previous;
      const uint coefficient_context =
          num_contexts * 37 + 458 * block_context + local;
      const uint value = ac_token_signed(coefficient);
      values[output + j] = value;
      contexts_out[output + j] = ushort(coefficient_context);
      if (control[end])
        ac_token_population(value, coefficient_context, shard,
                            num_contexts * 495, histogram);
      previous = uint(coefficient != 0);
      remaining -= previous;
    }
    prefix += m.count;
  }
}
