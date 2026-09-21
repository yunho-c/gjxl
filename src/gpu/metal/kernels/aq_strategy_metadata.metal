// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include <metal_stdlib>
using namespace metal;

struct MetadataParams { uint width, height, chunks, tiles_x, tiles_y, groups_x, groups_y; };
// AQ family order, deliberately different from candidate-scoring order.
constant uint strategies[7] = {0, 4, 5, 6, 7, 10, 11};
constant uint widths[7] = {1, 2, 4, 1, 2, 2, 4};
constant uint heights[7] = {1, 2, 4, 2, 1, 4, 2};
constant uint population_families[7] = {0, 2, 3, 4, 4, 6, 6};
uint Family(uint strategy) {
  for (uint f = 0; f < 7; ++f) if (strategies[f] == strategy) return f;
  return 7;
}

bool ValidCell(device const uchar* cells, constant MetadataParams& p, uint i, uint f) {
  if (f == 7) return false;
  const uint x = i % p.width, y = i / p.width;
  const uint tx = x / 8 * 8, ty = y / 8 * 8;
  const uint w = widths[f], h = heights[f];
  const uint anchor = 2 * strategies[f] + 1;
  if (cells[i] & 1) {
    if (x + w > min(tx + 8, p.width) || y + h > min(ty + 8, p.height)) return false;
    for (uint dy = 0; dy < h; ++dy) for (uint dx = 0; dx < w; ++dx) {
      if (cells[(y + dy) * p.width + x + dx] !=
          (anchor - uint(dx != 0 || dy != 0))) return false;
    }
  }
  // Also rejects orphan cells and overlapping equal-family rectangles.
  uint owners = 0;
  for (uint ay = max(int(ty), int(y) - int(h) + 1); ay <= y; ++ay)
    for (uint ax = max(int(tx), int(x) - int(w) + 1); ax <= x; ++ax)
      owners += cells[ay * p.width + ax] == anchor;
  return owners == 1;
}

kernel void gjxl_aq_metadata_reset(device atomic_uint* control [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  if (i < 4) atomic_store_explicit(control + i, 0u, memory_order_relaxed);
}

kernel void gjxl_aq_metadata_count(
    device const uchar* cells [[buffer(0)]], device uint* ranks [[buffer(1)]],
    device uint* chunks [[buffer(2)]], device atomic_uint* control [[buffer(5)]],
    constant MetadataParams& p [[buffer(11)]], uint i [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
  threadgroup uint partial[7 * 8];
  const uint count = p.width * p.height;
  const uint cell = i < count ? cells[i] : 0;
  const uint f = i < count ? Family(cell >> 1) : 7;
  const bool anchor = i < count && (cell & 1) && f < 7;
  if (i < count && (!ValidCell(cells, p, i, f) ||
      cells[count + (i / p.width / 8) * p.tiles_x + (i % p.width / 8)] != 0))
    atomic_fetch_or_explicit(control, 1u, memory_order_relaxed);
  uint local_rank = 0;
  for (uint family = 0; family < 7; ++family) {
    const uint member = uint(anchor && f == family);
    const uint prefix = simd_prefix_exclusive_sum(member);
    const uint total = simd_sum(member);
    if (f == family) local_rank = prefix;
    if (tid % 32 == 0) partial[family * 8 + tid / 32] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (anchor) {
    for (uint sg = 0; sg < tid / 32; ++sg) local_rank += partial[f * 8 + sg];
  }
  if (i < count) ranks[i] = anchor ? local_rank : 0;
  if (tid < 7) {
    uint total = 0;
    for (uint sg = 0; sg < 8; ++sg) total += partial[tid * 8 + sg];
    chunks[group * 7 + tid] = total;
  }
}

kernel void gjxl_aq_metadata_tile_count(device const uchar* cells [[buffer(0)]],
    device uint* tiles [[buffer(3)]], constant MetadataParams& p [[buffer(11)]],
    uint tile [[thread_position_in_grid]]) {
  if (tile >= p.tiles_x * p.tiles_y) return;
  uint count = 0;
  const uint x0 = tile % p.tiles_x * 8, y0 = tile / p.tiles_x * 8;
  for (uint y = y0; y < min(y0 + 8, p.height); ++y)
    for (uint x = x0; x < min(x0 + 8, p.width); ++x) count += cells[y * p.width + x] & 1;
  tiles[tile] = count;
}

kernel void gjxl_aq_metadata_prefix(device uint* chunks [[buffer(2)]],
    device const uint* tiles [[buffer(3)]], device uint* families [[buffer(4)]],
    device atomic_uint* control [[buffer(5)]], device uint* tile_offsets [[buffer(9)]],
    constant MetadataParams& p [[buffer(11)]], uint tid [[thread_index_in_threadgroup]]) {
  // Keep the validity word in the same explicitly indexed array. A separate
  // threadgroup bool produced inconsistent totals in the shader-validation build on
  // the qualification toolchain; both builds are covered by the CPU oracle.
  threadgroup uint totals[8];
  if (tid < 7) {
    uint prefix = 0;
    for (uint c = 0; c < p.chunks; ++c) {
      const uint n = chunks[c * 7 + tid];
      chunks[c * 7 + tid] = prefix;
      prefix += n;
    }
    totals[tid] = prefix;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    ulong area = 0;
    uint anchors = 0, mask = 0;
    for (uint f = 0; f < 7; ++f) {
      area += ulong(totals[f]) * widths[f] * heights[f];
      anchors += totals[f];
      if (totals[f]) mask |= 1u << population_families[f];
    }
    const bool valid = atomic_load_explicit(control, memory_order_relaxed) == 0 &&
      area == ulong(p.width) * p.height;
    totals[7] = uint(valid);
    if (!valid) atomic_fetch_or_explicit(control, 2u, memory_order_relaxed);
    atomic_store_explicit(control + 1, valid ? anchors : 0u, memory_order_relaxed);
    atomic_store_explicit(control + 2, valid ? mask : 0u, memory_order_relaxed);
    atomic_store_explicit(control + 3, valid ? uint(area) * 192u : 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool valid = totals[7] != 0;
  if (tid < 7) {
    uint anchor_offset = 0, coefficient_offset = 0;
    for (uint f = 0; f < tid; ++f) {
      anchor_offset += totals[f];
      coefficient_offset += totals[f] * widths[f] * heights[f] * 192u;
    }
    families[tid * 5] = strategies[tid];
    families[tid * 5 + 1] = valid ? anchor_offset : 0u;
    families[tid * 5 + 2] = valid ? totals[tid] : 0u;
    families[tid * 5 + 3] = valid ? coefficient_offset : 0u;
    families[tid * 5 + 4] = widths[tid] * heights[tid] * 64u;
  }
  if (tid == 7) {
    uint offset = 0;
    for (uint t = 0; t < p.tiles_x * p.tiles_y; ++t) {
      tile_offsets[t] = offset;
      offset += valid ? tiles[t] : 0u;
    }
    tile_offsets[p.tiles_x * p.tiles_y] = offset;
  }
}

kernel void gjxl_aq_metadata_scatter(device const uchar* cells [[buffer(0)]],
    device uint* ranks [[buffer(1)]], device const uint* chunks [[buffer(2)]],
    device const uint* families [[buffer(4)]], device atomic_uint* control [[buffer(5)]],
    device uint* strategy_map [[buffer(6)]], device uint* anchors [[buffer(7)]],
    constant MetadataParams& p [[buffer(11)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.width * p.height) return;
  if (atomic_load_explicit(control, memory_order_relaxed) != 0) {
    strategy_map[2 * i] = 0; strategy_map[2 * i + 1] = 1;
    return;
  }
  const uint cell = cells[i], f = Family(cell >> 1);
  strategy_map[2 * i] = cell >> 1; strategy_map[2 * i + 1] = cell & 1;
  if (!(cell & 1)) return;
  const uint rank = chunks[(i / 256) * 7 + f] + ranks[i];
  ranks[i] = rank;
  const uint index = families[f * 5 + 1] + rank;
  anchors[2 * index] = i % p.width; anchors[2 * index + 1] = i / p.width;
}

kernel void gjxl_aq_metadata_cfl(device const uchar* cells [[buffer(0)]],
    device const uint* ranks [[buffer(1)]], device const uint* families [[buffer(4)]],
    device atomic_uint* control [[buffer(5)]], device uint* records [[buffer(8)]],
    device const uint* offsets [[buffer(9)]], constant MetadataParams& p [[buffer(11)]],
    uint tile [[thread_position_in_grid]]) {
  if (tile >= p.tiles_x * p.tiles_y || atomic_load_explicit(control, memory_order_relaxed)) return;
  uint record = offsets[tile], values = 0;
  const uint x0 = tile % p.tiles_x * 8, y0 = tile / p.tiles_x * 8;
  for (uint y = y0; y < min(y0 + 8, p.height); ++y) {
    for (uint x = x0; x < min(x0 + 8, p.width); ++x) {
      const uint i = y * p.width + x, cell = cells[i];
      if (!(cell & 1)) continue;
      const uint f = Family(cell >> 1), size = families[f * 5 + 4];
      records[6 * record] = families[f * 5 + 3] + ranks[i] * size;
      records[6 * record + 1] = families[f * 5 + 2] * size;
      records[6 * record + 2] = size;
      records[6 * record + 3] = cell >> 1;
      records[6 * record + 4] = i;
      records[6 * record + 5] = values;
      values += size; ++record;
    }
  }
}

kernel void gjxl_aq_metadata_destinations(device const uchar* cells [[buffer(0)]],
    device const uint* ranks [[buffer(1)]], device const uint* families [[buffer(4)]],
    device atomic_uint* control [[buffer(5)]], device uint* destinations [[buffer(10)]],
    constant MetadataParams& p [[buffer(11)]], uint group [[thread_position_in_grid]]) {
  if (group >= p.groups_x * p.groups_y || atomic_load_explicit(control, memory_order_relaxed)) return;
  uint used = 0;
  const uint x0 = group % p.groups_x * 32, y0 = group / p.groups_x * 32;
  for (uint y = y0; y < min(y0 + 32, p.height); ++y) {
    for (uint x = x0; x < min(x0 + 32, p.width); ++x) {
      const uint i = y * p.width + x, cell = cells[i];
      if (!(cell & 1)) continue;
      const uint f = Family(cell >> 1);
      const uint index = families[f * 5 + 1] + ranks[i];
      destinations[index] = group * (3u * 65536u) + used;
      used += families[f * 5 + 4];
    }
  }
}

#include "aq_strategy_dispatch.h"
// Templates contain only geometry/policy values. Selection-dependent fields
// and every indirect group count are overwritten from the validated family bank.
kernel void gjxl_aq_strategy_dispatch(
    device const uint* families [[buffer(0)]],
    constant gjxl_aq_dispatch::Record* templates [[buffer(1)]],
    device gjxl_aq_dispatch::Record* records [[buffer(2)]],
    uint f [[thread_position_in_grid]]) {
  if (f >= 7) return;
  using namespace gjxl_aq_dispatch;
  Record r = templates[f];
  const uint a = families[5*f+1], n = families[5*f+2];
  const uint c = families[5*f+3], size = families[5*f+4];
  r.reconstruction[8] = a;
  r.reconstruction[9] = n;
  r.reconstruction[10] = c;
  r.completed_reconstruction[8] = a;
  r.completed_reconstruction[9] = n;
  r.completed_reconstruction[10] = c;
  r.forward[0] = r.inverse[0] = a;
  r.forward[1] = r.inverse[1] = n;
  r.forward[2] = r.inverse[2] = c;
  r.adjustment[1] = a;
  r.adjustment[2] = n;
  r.population[0] = a;
  r.population[1] = n;
  r.population[3] = families[2] == r.reconstruction[3]*r.reconstruction[4] ? 1u : 0u;
  r.block_reduction[4] = a;
  r.block_reduction[5] = n;
  const bool parallel = size >= 128 && n <= 256;
  for (uint d = 0; d < kDispatchCount; ++d) {
    r.groups[d][0] = 0;
    r.groups[d][1] = r.groups[d][2] = 1;
  }
  r.groups[kTransforms][0] = n;
  r.groups[kDct][0] = 3*n;
  r.groups[kAdjustedScalar][0] = parallel ? 0 : (n+255)/256;
  r.groups[kAdjustedParallel][0] = parallel ? n : 0;
  r.groups[kQuantField][0] = (n+255)/256;
  r.groups[kPopulation][0] = size/32;
  r.groups[kPopulation][1] = (n+63)/64;
  r.groups[kLlf][0] = (3*n*(size/64)+255)/256;
  records[f] = r;
}

// Import selector/metadata failure after the enclosing AQ reset. Later policy
// resets preserve this flag, including the zero-update final-frame path.
kernel void gjxl_aq_metadata_error(device const uint* control [[buffer(0)]],
                                 device atomic_uint* error [[buffer(1)]]) {
  if (control[0]) atomic_fetch_or_explicit(error, 0x10000000u, memory_order_relaxed);
}
