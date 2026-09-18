// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include <metal_stdlib>
using namespace metal;
namespace ac_selection_internal {
uint Mantissa(uint b) {
  return (b & 0x7fffffu) | (((b >> 23) & 255) ? 0x800000u : 0u);
}
uint Exponent(uint b) {
  uint e = (b >> 23) & 255;
  return e ? e - 1 : 0;
}
uint PolicyBits(float raw, float multiplier) {
  uint bits = as_type<uint>(raw), a = bits & 0x7fffffffu;
  if (a < 0x1000000u) {
    // The DCT8 multiplier lies in [0.714...,1]. Below twice FLT_MIN,
    // form the correctly rounded product in units of 2^-149; hardware
    // multiplication would flush subnormal inputs or results on this GPU.
    uint b = as_type<uint>(multiplier), shift = 149 - Exponent(a) - Exponent(b);
    ulong product = ulong(Mantissa(a)) * Mantissa(b), units = product >> shift;
    ulong tail = product & ((1ul << shift) - 1), mid = 1ul << (shift - 1);
    units += tail > mid || (tail == mid && (units & 1));
    return (bits & 0x80000000u) | uint(units);
  }
  return as_type<uint>(raw * multiplier);
}
constant uchar greedy_width[7] = {1, 1, 2, 2, 2, 4, 4};
constant uchar greedy_height[7] = {1, 2, 1, 2, 4, 2, 4};
constant uchar greedy_strategy[7] = {0, 6, 7, 4, 10, 11, 5};
struct GreedyTile {
  uint cost[64];
  uchar priority[64], grid[64];
  uint w, h;
  device const float *family[7];
  uint base[7], columns[7], step[7];
  float multiplier;
};
uint GreedyCost(thread GreedyTile &g, uint s, uint x, uint y) {
  float raw =
      g.family[s][g.base[s] + (y / g.step[s]) * g.columns[s] + x / g.step[s]];
  return (s == 0 ? PolicyBits(raw, g.multiplier) : as_type<uint>(raw)) &
         0x7fffffffu;
}
uint GreedyAdd(uint a, uint b) {
  if (!a)
    return b;
  if (!b)
    return a;
  if (a >= 0x800000u && b >= 0x800000u)
    return as_type<uint>(as_type<float>(a) + as_type<float>(b));
  // Metal FP32 arithmetic flushes subnormals. Preserve CPU round-to-nearest,
  // ties-to-even with three guard/sticky bits only on this exceptional path.
  if (a < b) {
    uint t = a;
    a = b;
    b = t;
  }
  if (a >= 0x7f800000u)
    return a;
  uint e = Exponent(a), delta = e - Exponent(b);
  ulong small = ulong(Mantissa(b)) << 3;
  small = delta >= 64 ? ulong(small != 0)
                      : (delta ? ((small >> delta) |
                                  ulong((small & ((1ul << delta) - 1)) != 0))
                               : small);
  ulong sum = (ulong(Mantissa(a)) << 3) + small;
  if (sum >= (1ul << 27)) {
    sum = (sum >> 1) | (sum & 1);
    ++e;
  }
  uint m = uint(sum >> 3), tail = uint(sum & 7);
  m += tail > 4 || (tail == 4 && (m & 1));
  if (m >= 0x1000000u) {
    m >>= 1;
    ++e;
  }
  if (e >= 254)
    return 0x7f800000u;
  return m < 0x800000u ? m : ((e + 1) << 23) | (m & 0x7fffffu);
}
// Keep the mutating helpers out of line. The tested Metal compiler produced
// incorrect maps with inlining on; shader validation masked that failure.
// Both ordinary and instrumented kernels are checked against the CPU policy.
__attribute__((noinline)) void GreedySet(thread GreedyTile &g, uint s, uint x,
                                         uint y, uint cost) {
  for (uint dy = 0; dy < greedy_height[s]; ++dy)
    for (uint dx = 0; dx < greedy_width[s]; ++dx) {
      uint i = (y + dy) * 8 + x + dx;
      g.grid[i] = (s << 1) | uint(dx == 0 && dy == 0);
      g.cost[i] = 0;
    }
  g.cost[y * 8 + x] = cost;
}
bool GreedyCrossH(thread GreedyTile &g, uint start, uint y, uint end) {
  if (start >= g.w || y >= g.h || y == 0)
    return false;
  while (start && !(g.grid[y * 8 + start] & 1))
    --start;
  for (uint x = start; x < min(end, g.w);) {
    uchar c = g.grid[y * 8 + x];
    if (!(c & 1))
      return true;
    x += greedy_width[c >> 1];
  }
  return false;
}
bool GreedyCrossV(thread GreedyTile &g, uint x, uint start, uint end) {
  if (x >= g.w || start >= g.h || x == 0)
    return false;
  while (start && !(g.grid[start * 8 + x] & 1))
    --start;
  for (uint y = start; y < min(end, g.h);) {
    uchar c = g.grid[y * 8 + x];
    if (!(c & 1))
      return true;
    y += greedy_height[c >> 1];
  }
  return false;
}
__attribute__((noinline)) void GreedyDivision(thread GreedyTile &g, uint blocks,
                                              uint x, uint y) {
  if (GreedyCrossH(g, x, y, x + blocks) ||
      GreedyCrossH(g, x, y + blocks, x + blocks) ||
      GreedyCrossV(g, x, y, y + blocks) ||
      GreedyCrossV(g, x + blocks, y, y + blocks))
    return;
  uint split_size = blocks / 2, vertical = blocks == 2 ? 1 : 4,
       horizontal = blocks == 2 ? 2 : 5, square = blocks == 2 ? 3 : 6;
  bool av = !GreedyCrossV(g, x + split_size, y, y + blocks),
       ah = !GreedyCrossH(g, x, y + split_size, x + blocks);
  uint q[4] = {0, 0, 0, 0};
  for (uint dy = 0; dy < blocks; ++dy)
    for (uint dx = 0; dx < blocks; ++dx) {
      uint i = (dy / split_size) * 2 + dx / split_size;
      q[i] = GreedyAdd(q[i], g.cost[(y + dy) * 8 + x + dx]);
    }
  uint vl = 0x7f7fffffu, vr = 0x7f7fffffu, ht = 0x7f7fffffu, hb = 0x7f7fffffu;
  if (av) {
    if ((g.grid[y * 8 + x] >> 1) != vertical)
      vl = GreedyCost(g, vertical, x, y);
    if ((g.grid[y * 8 + x + split_size] >> 1) != vertical)
      vr = GreedyCost(g, vertical, x + split_size, y);
  }
  if (ah) {
    if ((g.grid[y * 8 + x] >> 1) != horizontal)
      ht = GreedyCost(g, horizontal, x, y);
    if ((g.grid[(y + split_size) * 8 + x] >> 1) != horizontal)
      hb = GreedyCost(g, horizontal, x, y + split_size);
  }
  uint sq = GreedyCost(g, square, x, y);
  uint q02 = GreedyAdd(q[0], q[2]), q13 = GreedyAdd(q[1], q[3]);
  uint q01 = GreedyAdd(q[0], q[1]), q23 = GreedyAdd(q[2], q[3]);
  uint vc = GreedyAdd(min(vl, q02), min(vr, q13));
  uint hc = GreedyAdd(min(ht, q01), min(hb, q23));
  if (sq < vc && sq < hc)
    GreedySet(g, square, x, y, sq);
  else if (vc < hc) {
    if (vl < q02)
      GreedySet(g, vertical, x, y, vl);
    if (vr < q13)
      GreedySet(g, vertical, x + split_size, y, vr);
  } else {
    if (ht < q01)
      GreedySet(g, horizontal, x, y, ht);
    if (hb < q23)
      GreedySet(g, horizontal, x, y + split_size, hb);
  }
}
void GreedyMerge(thread GreedyTile &g, uint s, uint x, uint y, uint priority) {
  uint current = 0;
  for (uint dy = 0; dy < greedy_height[s]; ++dy)
    for (uint dx = 0; dx < greedy_width[s]; ++dx) {
      uint i = (y + dy) * 8 + x + dx;
      if (g.priority[i] >= priority)
        return;
      current = GreedyAdd(current, g.cost[i]);
    }
  uint candidate = GreedyCost(g, s, x, y);
  if (candidate >= current)
    return;
  GreedySet(g, s, x, y, candidate);
  for (uint dy = 0; dy < greedy_height[s]; ++dy)
    for (uint dx = 0; dx < greedy_width[s]; ++dx)
      g.priority[(y + dy) * 8 + x + dx] = priority;
}

struct SelectionParams {
  uint width, height, tiles_x, tiles_y;
  float multiplier;
};
uint Positions(uint width, uint covered, uint step) {
  return width < covered ? 0 : (width - covered) / step + 1;
}
} // namespace ac_selection_internal

kernel void gjxl_ac_strategy_select_greedy(
    device const float *c0 [[buffer(0)]], device const float *c1 [[buffer(1)]],
    device const float *c2 [[buffer(2)]], device const float *c3 [[buffer(3)]],
    device const float *c4 [[buffer(4)]], device const float *c5 [[buffer(5)]],
    device const float *c6 [[buffer(6)]], device uchar *output [[buffer(7)]],
    constant ac_selection_internal::SelectionParams &p [[buffer(8)]],
    uint tile [[thread_position_in_grid]]) {
  using namespace ac_selection_internal;
  if (tile >= p.tiles_x * p.tiles_y)
    return;
  uint tx = tile % p.tiles_x, ty = tile / p.tiles_x;
  GreedyTile g;
  g.w = min(8u, p.width - tx * 8);
  g.h = min(8u, p.height - ty * 8);
  g.multiplier = p.multiplier;
  g.family[0] = c0;
  g.family[1] = c1;
  g.family[2] = c2;
  g.family[3] = c3;
  g.family[4] = c4;
  g.family[5] = c5;
  g.family[6] = c6;
  bool bad = false;
  for (uint s = 0; s < 7; ++s) {
    uint step = s < 4 ? 1 : 2, w = greedy_width[s], h = greedy_height[s];
    uint full_x = Positions(8, w, step), full_y = Positions(8, h, step);
    uint total_x = (p.width / 8) * full_x + Positions(p.width % 8, w, step);
    uint nx = Positions(g.w, w, step), ny = Positions(g.h, h, step);
    g.base[s] = ty * full_y * total_x + tx * full_x * ny;
    g.columns[s] = nx;
    g.step[s] = step;
    for (uint i = 0; i < nx * ny; ++i) {
      uint bits = as_type<uint>(g.family[s][g.base[s] + i]);
      bad |= (bits & 0x7fffffffu) > 0x7f7fffffu ||
             ((bits >> 31) && (bits & 0x7fffffffu));
    }
  }
  output[p.width * p.height + tile] = bad;
  if (bad) {
    // Keep dependent metadata construction in bounds even when the final
    // completion boundary will reject this tile's numeric error.
    for (uint y = 0; y < g.h; ++y)
      for (uint x = 0; x < g.w; ++x)
        output[(ty * 8 + y) * p.width + tx * 8 + x] = 1;
    return;
  }
  for (uint i = 0; i < 64; ++i) {
    g.cost[i] = 0;
    g.priority[i] = 0;
    g.grid[i] = 255;
  }
  for (uint y = 0; y < g.h; ++y)
    for (uint x = 0; x < g.w; ++x) {
      g.cost[y * 8 + x] = GreedyCost(g, 0, x, y);
      g.grid[y * 8 + x] = 1;
    }
  const uint stages[4] = {1, 2, 5, 4};
  for (uint stage = 0; stage < 4; ++stage) {
    uint s = stages[stage];
    for (uint y = 0; y + greedy_height[s] <= g.h; y += greedy_height[s])
      for (uint x = 0; x + greedy_width[s] <= g.w; x += greedy_width[s]) {
        if (y + 3 < g.h && x + 3 < g.w) {
          if (s == 5) {
            if (((y | x) % 4) == 0)
              GreedyDivision(g, 4, x, y);
            continue;
          }
          if (s == 4)
            continue;
        }
        if ((s == 5 && y % 4 != 0) || (s == 4 && x % 4 != 0))
          continue;
        if (y + 1 < g.h && x + 1 < g.w) {
          if (s == 2) {
            if (((y | x) % 2) == 0)
              GreedyDivision(g, 2, x, y);
            continue;
          }
          if (s == 1)
            continue;
        }
        if ((s == 2 && y % 2 == 1) || (s == 1 && x % 2 == 1))
          continue;
        GreedyMerge(g, s, x, y, stage < 2 ? 2 : 4);
      }
  }
  for (uint y = 0; y + 1 < g.h; ++y)
    for (uint x = 0; x + 1 < g.w; ++x)
      if ((y | x) % 2 != 0)
        GreedyDivision(g, 2, x, y);
  for (uint y = 0; y + 3 < g.h; y += 2)
    for (uint x = 0; x + 3 < g.w; x += 2)
      if ((y | x) % 4 != 0)
        GreedyDivision(g, 4, x, y);

  for (uint y = 0; y < g.h; ++y)
    for (uint x = 0; x < g.w; ++x) {
      uchar cell = g.grid[y * 8 + x];
      output[(ty * 8 + y) * p.width + tx * 8 + x] =
          (greedy_strategy[cell >> 1] << 1) | (cell & 1);
    }
}
