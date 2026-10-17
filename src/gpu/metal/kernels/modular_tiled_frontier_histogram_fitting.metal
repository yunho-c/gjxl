// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// GPU-owned row tiling, compact histogram admission, scoring and partition.
// Dispatch barriers are the only cross-threadgroup producer/consumer edges.
namespace tiled_frontier_impl {
using namespace frontier_impl;
device TiledControl &Control(device uchar *work, constant TiledParams &p) {
  return *reinterpret_cast<device TiledControl *>(work + p.frontier.control);
}
device uchar *SlotBytes(device uchar *work, constant TiledParams &p, uint slot) {
  return work + p.frontier.scratch + slot * p.frontier.slot.bytes;
}
device TiledSlot &Slot(device uchar *work, constant TiledParams &p, uint slot) {
  return *reinterpret_cast<device TiledSlot *>(SlotBytes(work, p, slot));
}
void Add(device uint *value, uint amount) {
  if (amount) atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(value), amount, memory_order_relaxed);
}
uint FindTask(device const TiledTask *tasks, uint count, uint group) {
  uint lo = 0, hi = count;
  while (lo < hi) {
    const uint mid = lo + (hi - lo) / 2;
    if (group >= tasks[mid].end) lo = mid + 1; else hi = mid;
  }
  return lo;
}
// Exclusive prefixes and totals for two integer counts over all 128 lanes.
// The final barrier permits the caller to reuse the four shared SIMD totals.
uint4 GroupPrefix(uint2 value, uint lane, threadgroup uint2 *groups) {
  const uint2 sum(simd_prefix_inclusive_sum(value.x), simd_prefix_inclusive_sum(value.y));
  if ((lane & 31) == 31) groups[lane / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint2 offset = sum - value, total = 0;
  for (uint i = 0; i < 4; ++i) {
    if (i < lane / 32) offset += groups[i];
    total += groups[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return uint4(offset, total);
}
// Reuse lower-histogram loads between children. Each side retains Entropy's
// native four-lane arithmetic and final pairwise sum, without reassociation.
template <typename Histogram>
float2 EntropyPair(Histogram lower, device const uint *parent, uint symbols) {
  uint low_total = 0, high_total = 0;
  for (uint sym = 0; sym < symbols; ++sym) {
    const uint low = lower[sym];
    low_total += low;
    high_total += parent[sym] - low;
  }
  const float low_inverse = 1.0f / float(low_total), high_inverse = 1.0f / float(high_total);
  float4 low_bits = 0.0f, high_bits = 0.0f;
  for (uint sym = 0; sym < symbols; sym += 4) {
    const uint4 low(lower[sym], lower[sym + 1], lower[sym + 2], lower[sym + 3]);
    const uint4 high = uint4(parent[sym], parent[sym + 1], parent[sym + 2], parent[sym + 3]) - low;
    const float4 low_f = float4(low), high_f = float4(high);
    const float4 low_probability = max(low_f * low_inverse, float4(1.0f / 4096));
    const float4 high_probability = max(high_f * high_inverse, float4(1.0f / 4096));
    low_bits = low_bits - low_f * select(NativeLog2(low_probability), float4(0), low == low_total);
    high_bits = high_bits - high_f * select(NativeLog2(high_probability), float4(0), high == high_total);
  }
  return float2((low_bits.x + low_bits.y) + (low_bits.z + low_bits.w),
                (high_bits.x + high_bits.y) + (high_bits.z + high_bits.w));
}
Candidate EmptyCandidate() { return {0, 0, 0, 0, kMaximumCost, kMaximumCost, 0, 0}; }
// Histogram reads finish before winners are shared for the final ordered scan.
// Nonempty nodes have at least four symbol slots plus one extra-bit column,
// bounding compact bins and keeping this scratch below 5 KiB.
union SmallJobScratch {
  uint histogram[kTiledSmallCells];
  SideWinner sides[2 * (kTiledSmallCells / 5)];
};
}

kernel void gjxl_histogram_tiled_init(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::TiledParams &p [[buffer(2)]],
    uint fit [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using namespace tiled_frontier_impl;
  device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[fit];
  device const Fit &d = reinterpret_cast<device const Fit *>(work)[fit];
  if (!lane) {
    *reinterpret_cast<device State *>(work + d.state) = {0, 1, 0, 0, 0, 0, 0, 0, 0};
    reinterpret_cast<device Node *>(work + d.nodes)[0] = {-1, 0, 0, 0, f.predictor_ids[0], 0, f.rows, 1, 0};
    reinterpret_cast<device NodeRef *>(work + p.frontier.queue)[fit] = {fit, 0};
    if (!fit) {
      Control(work, p) = {};
      Control(work, p).frontier.tail = p.frontier.fits;
    }
  }
  if (!fit)
    for (uint i = lane; i < p.frontier.active_nodes; i += 128) Slot(work, p, i).stage = kTiledFree;
  device uint *order = reinterpret_cast<device uint *>(work + d.order);
  for (uint i = lane; i < f.rows; i += 128) order[i] = i;
}

kernel void gjxl_histogram_tiled_begin(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::TiledParams &p [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]) {
  using namespace tiled_frontier_impl;
  device TiledControl &c = Control(work, p);
  if (c.frontier.status) {
    if (!lane) { c.row_tasks = 0; c.stats_dispatch[0] = c.stats_dispatch[1] = c.stats_dispatch[2] = 1; }
    return;
  }
  // Contiguous ranges per lane preserve the original slot/queue order. The
  // admitted maximum is 1024 slots, so each lane owns at most eight slots.
  const uint owned = (p.frontier.active_nodes + 127) / 128, first = lane * owned;
  const uint head = c.frontier.head, ready = c.frontier.tail - head;
  threadgroup uint2 groups[4];
  threadgroup atomic_uint counts[64];
  if (lane < 64) atomic_store_explicit(&counts[lane], 0, memory_order_relaxed);
  uint free_slots[8] = {}, tiles[8] = {}, free_count = 0;
  #pragma unroll
  for (uint i = 0; i < 8; ++i) {
    if (i < owned && first + i < p.frontier.active_nodes)
      free_slots[i] = Slot(work, p, first + i).stage == kTiledFree;
    free_count += free_slots[i];
  }
  const uint4 free_prefix = GroupPrefix(uint2(free_count, 0), lane, groups);
  const uint admitted = min(ready, free_prefix.z);
  device const NodeRef *queue = reinterpret_cast<device const NodeRef *>(work + p.frontier.queue);
  uint rank = free_prefix.x, new_nodes = 0, row_groups = 0;
  #pragma unroll
  for (uint i = 0; i < 8; ++i) {
    if (i < owned && first + i < p.frontier.active_nodes) {
      device TiledSlot &s = Slot(work, p, first + i);
      if (free_slots[i] && rank < admitted) {
        s = {queue[head + rank], kTiledNew, 0, 0, 0, 0, 0, 0};
        device const Fit &d = reinterpret_cast<device const Fit *>(work)[s.ref.fit];
        device const Node &n = reinterpret_cast<device const Node *>(work + d.nodes)[s.ref.node];
        tiles[i] = (n.end - n.begin + kTiledRows - 1) / kTiledRows;
        ++new_nodes; row_groups += tiles[i];
      }
      rank += free_slots[i];
      if (s.stage != kTiledFree)
        atomic_fetch_add_explicit(&counts[s.ref.fit], 1, memory_order_relaxed);
    }
  }
  const uint4 task_prefix = GroupPrefix(uint2(new_nodes, row_groups), lane, groups);
  device TiledTask *tasks = reinterpret_cast<device TiledTask *>(work + p.row_tasks);
  uint task = task_prefix.x, group = task_prefix.y;
  #pragma unroll
  for (uint i = 0; i < 8; ++i) if (tiles[i]) {
    tasks[task++] = {first + i, 0, group, group + tiles[i]}; group += tiles[i];
  }
  const uint same_fit = simd_max(lane < 64 ? atomic_load_explicit(&counts[lane], memory_order_relaxed) : 0);
  if (!(lane & 31))
    atomic_fetch_max_explicit(reinterpret_cast<device atomic_uint *>(&c.frontier.maximum_same_fit), same_fit, memory_order_relaxed);
  if (!lane) {
    c.frontier.maximum_ready = max(c.frontier.maximum_ready, ready + c.live);
    c.frontier.head = head + admitted;
    c.live += admitted;
    c.frontier.maximum_active = max(c.frontier.maximum_active, c.live);
    c.frontier.rounds += c.live != 0;
    c.row_tasks = task_prefix.z;
    c.stats_dispatch[0] = max(1u, task_prefix.w); c.stats_dispatch[1] = c.stats_dispatch[2] = 1;
  }
}

kernel void gjxl_histogram_tiled_clear_stats(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::TiledParams &p [[buffer(2)]],
    uint slot [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using namespace tiled_frontier_impl;
  if (Control(work, p).frontier.status) return;
  device TiledSlot &s = Slot(work, p, slot);
  if (s.stage != kTiledNew) return;
  device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[s.ref.fit];
  device uchar *base = SlotBytes(work, p, slot);
  device uint *parent = reinterpret_cast<device uint *>(base + p.parent);
  device uint *extra = reinterpret_cast<device uint *>(base + p.parent_extra);
  device uint *occupancy = reinterpret_cast<device uint *>(base + p.occupancy);
  for (uint i = lane; i < f.predictors * 256; i += 128) parent[i] = 0;
  if (lane < f.predictors) extra[lane] = 0;
  for (uint i = lane; i < f.properties * 256; i += 128) occupancy[i] = 0;
}

kernel void gjxl_histogram_tiled_stats(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::TiledParams &p [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using namespace tiled_frontier_impl;
  device TiledControl &c = Control(work, p);
  if (c.frontier.status) return;
  device const TiledTask *tasks = reinterpret_cast<device const TiledTask *>(work + p.row_tasks);
  const uint task_index = FindTask(tasks, c.row_tasks, group);
  if (task_index == c.row_tasks) return;
  const TiledTask task = tasks[task_index];
  device TiledSlot &s = Slot(work, p, task.slot);
  device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[s.ref.fit];
  device const Fit &d = reinterpret_cast<device const Fit *>(work)[s.ref.fit];
  const Node n = reinterpret_cast<device const Node *>(work + d.nodes)[s.ref.node];
  device const uint *order = reinterpret_cast<device const uint *>(work + d.order);
  device const ushort *weights = reinterpret_cast<device const ushort *>(prepared + f.weights_offset);
  device uchar *base = SlotBytes(work, p, task.slot);
  const uint first = n.begin + (group - task.first) * kTiledRows, end = min(n.end, first + kTiledRows);
  threadgroup atomic_uint counts[256], extra;
  uint maximum = 0;
  for (uint pred = 0; pred < f.predictors; ++pred) {
    for (uint i = lane; i < 256; i += 128) atomic_store_explicit(&counts[i], 0, memory_order_relaxed);
    if (!lane) atomic_store_explicit(&extra, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uchar2 *tokens = reinterpret_cast<device const uchar2 *>(prepared + f.residual_offsets[pred]);
    for (uint i = first + lane; i < end; i += 128) {
      const uint row = order[i], weight = weights[row]; const uchar2 t = tokens[row];
      atomic_fetch_add_explicit(&counts[t.x], weight, memory_order_relaxed);
      if (t.y) atomic_fetch_add_explicit(&extra, weight * uint(t.y), memory_order_relaxed);
      maximum = max(maximum, uint(t.x) + 1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint *parent = reinterpret_cast<device uint *>(base + p.parent) + pred * 256;
    for (uint i = lane; i < 256; i += 128) Add(parent + i, atomic_load_explicit(&counts[i], memory_order_relaxed));
    if (!lane) Add(reinterpret_cast<device uint *>(base + p.parent_extra) + pred, atomic_load_explicit(&extra, memory_order_relaxed));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  maximum = simd_max(maximum);
  if (!(lane & 31))
    atomic_fetch_max_explicit(reinterpret_cast<device atomic_uint *>(&s.symbols), maximum, memory_order_relaxed);
  for (uint prop = 0; prop < f.properties; ++prop) {
    if ((n.constants & (1u << prop)) || f.bins[prop] == 1) continue;
    for (uint i = lane; i < 256; i += 128) atomic_store_explicit(&counts[i], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = first + lane; i < end; i += 128)
      atomic_fetch_add_explicit(&counts[Property(prepared, f, prop, order[i])], 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint *occupancy = reinterpret_cast<device uint *>(base + p.occupancy) + prop * 256;
    for (uint i = lane; i < f.bins[prop]; i += 128) Add(occupancy + i, atomic_load_explicit(&counts[i], memory_order_relaxed));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void gjxl_histogram_tiled_summarize(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::TiledParams &p [[buffer(2)]],
    uint slot [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using namespace tiled_frontier_impl;
  if (Control(work, p).frontier.status) return;
  device TiledSlot &s = Slot(work, p, slot);
  if (s.stage != kTiledNew) return;
  device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[s.ref.fit];
  device const Fit &d = reinterpret_cast<device const Fit *>(work)[s.ref.fit];
  device Node &n = reinterpret_cast<device Node *>(work + d.nodes)[s.ref.node];
  device uchar *base = SlotBytes(work, p, slot);
  device const uint *parent = reinterpret_cast<device const uint *>(base + p.parent);
  device const uint *extra = reinterpret_cast<device const uint *>(base + p.parent_extra);
  if (!lane) {
    s.symbols = (s.symbols + 3) & ~3u;
    while (f.predictor_ids[s.parent_index] != n.predictor) ++s.parent_index;
    s.base_cost = Entropy(parent + s.parent_index * 256,
        reinterpret_cast<device const int *>(parent + s.parent_index * 256), s.symbols, false) + float(extra[s.parent_index]);
  }
  threadgroup_barrier(mem_flags::mem_device);
  device TiledProperty *properties = reinterpret_cast<device TiledProperty *>(base + p.properties);
  // Each SIMD group owns whole properties. Only integer occupancy/bin scans
  // move across lanes; the parent entropy above retains its exact float order.
  const uint local = lane & 31, inherited = n.constants;
  uint constants = 0, needed = 0;
  for (uint prop = lane / 32; prop < f.properties; prop += 4) {
    device TiledProperty &q = properties[prop];
    if (!local) q.bins = q.histogram = 0;
    if (local < 4) q.winners[local] = EmptyCandidate();
    if (!(s.base_cost > d.split_threshold)) continue;
    if ((inherited & (1u << prop)) || f.bins[prop] == 1) { constants |= 1u << prop; continue; }
    device uint *occupancy = reinterpret_cast<device uint *>(base + p.occupancy) + prop * 256;
    device uint *map = reinterpret_cast<device uint *>(base + p.bin_map) + prop * 256;
    uint used = 0, sum = 0;
    for (uint first = 0; first < f.bins[prop]; first += 32) {
      const uint b = first + local, count = b < f.bins[prop] ? occupancy[b] : 0;
      const uint2 prefix(simd_prefix_inclusive_sum(uint(count != 0)), simd_prefix_inclusive_sum(count));
      if (b < f.bins[prop]) {
        map[b] = used + prefix.x - uint(count != 0);
        occupancy[b] = sum + prefix.y;
      }
      used += simd_broadcast(prefix.x, 31);
      sum += simd_broadcast(prefix.y, 31);
    }
    if (!local) {
      if (used <= 1) constants |= 1u << prop;
      else {
        q.bins = used;
        if (n.end - n.begin <= kTiledRows && used * (s.symbols + 1) <= kTiledSmallCells)
          q.histogram = kTiledLocalHistogram;
        else
          needed += used * f.predictors * (s.symbols + 1) * 4;
      }
    }
  }
  threadgroup uint2 summaries[4];
  if (!local) summaries[lane / 32] = uint2(constants, needed);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (!lane) {
    s.needed = 0;
    for (uint i = 0; i < 4; ++i) { n.constants |= summaries[i].x; s.needed += summaries[i].y; }
    s.stage = kTiledPending;
  }
}

kernel void gjxl_histogram_tiled_schedule(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::TiledParams &p [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]) {
  using namespace tiled_frontier_impl;
  device TiledControl &c = Control(work, p);
  // The common case admits every pending node. Build each node's task list
  // independently, after a small prefix over counts. The bounded serial path
  // below retains the original first-fit policy when the pool is too small.
  // At most 1024 * 16 property jobs fit below bit 16. Pack the selected-node
  // count above them to keep this array at 12 KiB, leaving validation room
  // within the device's 32 KiB shared-memory limit.
  threadgroup packed_uint3 offsets[1024];
  threadgroup uint4 total;
  for (uint i = lane; i < p.frontier.active_nodes; i += 128) {
    device const TiledSlot &s = Slot(work, p, i);
    uint3 count = 0;
    if (!c.frontier.status && s.stage == kTiledPending) {
      device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[s.ref.fit];
      device const Fit &d = reinterpret_cast<device const Fit *>(work)[s.ref.fit];
      device const Node &n = reinterpret_cast<device const Node *>(work + d.nodes)[s.ref.node];
      device const TiledProperty *properties = reinterpret_cast<device const TiledProperty *>(SlotBytes(work, p, i) + p.properties);
      uint jobs = 0, device_jobs = 0;
      for (uint prop = 0; prop < f.properties; ++prop) {
        jobs += properties[prop].bins != 0;
        device_jobs += properties[prop].bins && properties[prop].histogram != kTiledLocalHistogram;
      }
      count = uint3(s.needed, jobs | (1u << 16), device_jobs * ((n.end - n.begin + kTiledRows - 1) / kTiledRows) * f.predictors);
    }
    offsets[i] = count;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!lane) {
    uint3 sum = 0;
    ulong bytes = 0;
    for (uint i = 0; i < p.frontier.active_nodes; ++i) {
      const uint3 count = offsets[i]; offsets[i] = sum;
      bytes += count.x;
      sum += uint3(0, count.yz);
      // Only the all-admitted path uses byte offsets. Saturation avoids
      // overflow for a wide frontier whose combined histograms exceed 4 GiB.
      sum.x = uint(min(bytes, ulong(p.histogram_bytes) + 1));
    }
    total = uint4(sum.x, sum.y & 65535u, sum.z, sum.y >> 16);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (total.x <= p.histogram_bytes) {
    device TiledTask *tasks = reinterpret_cast<device TiledTask *>(work + p.jobs);
    device uint *selected = reinterpret_cast<device uint *>(work + p.selected);
    for (uint i = lane; i < p.frontier.active_nodes; i += 128) {
      device TiledSlot &s = Slot(work, p, i);
      if (c.frontier.status || s.stage != kTiledPending) continue;
      const uint3 packed = offsets[i];
      const uint4 offset = uint4(packed.x, packed.y & 65535u, packed.z, packed.y >> 16);
      s.stage = kTiledSelected; selected[offset.w] = i;
      device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[s.ref.fit];
      device const Fit &d = reinterpret_cast<device const Fit *>(work)[s.ref.fit];
      device const Node &n = reinterpret_cast<device const Node *>(work + d.nodes)[s.ref.node];
      device TiledProperty *properties = reinterpret_cast<device TiledProperty *>(SlotBytes(work, p, i) + p.properties);
      const uint tiles = (n.end - n.begin + kTiledRows - 1) / kTiledRows;
      uint used = offset.x, job = offset.y, groups = offset.z;
      for (uint prop = 0; prop < f.properties; ++prop) {
        device TiledProperty &q = properties[prop]; if (!q.bins) continue;
        const uint count = q.histogram == kTiledLocalHistogram ? 0 : tiles * f.predictors;
        if (count) {
          q.histogram = p.histogram + used;
          used += q.bins * f.predictors * (s.symbols + 1) * 4;
        }
        tasks[job++] = {i, prop, groups, groups + count};
        groups += count;
      }
    }
    if (!lane) {
      c.jobs = total.y; c.selected = total.w;
      c.histogram_peak = max(c.histogram_peak, total.x);
      c.collect_dispatch[0] = max(1u, total.z); c.collect_dispatch[1] = c.collect_dispatch[2] = 1;
      c.jobs_dispatch[0] = max(1u, c.jobs); c.jobs_dispatch[1] = c.jobs_dispatch[2] = 1;
      c.finish_dispatch[0] = max(1u, c.selected); c.finish_dispatch[1] = c.finish_dispatch[2] = 1;
    }
    return;
  }
  if (lane) return;
  c.jobs = c.selected = 0; uint used = 0, groups = 0;
  device TiledTask *tasks = reinterpret_cast<device TiledTask *>(work + p.jobs);
  device uint *selected = reinterpret_cast<device uint *>(work + p.selected);
  if (!c.frontier.status) for (uint i = 0; i < p.frontier.active_nodes; ++i) {
    device TiledSlot &s = Slot(work, p, i);
    if (s.stage != kTiledPending || s.needed > p.histogram_bytes - used) continue;
    s.stage = kTiledSelected; selected[c.selected++] = i;
    device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[s.ref.fit];
    device const Fit &d = reinterpret_cast<device const Fit *>(work)[s.ref.fit];
    device const Node &n = reinterpret_cast<device const Node *>(work + d.nodes)[s.ref.node];
    device TiledProperty *properties = reinterpret_cast<device TiledProperty *>(SlotBytes(work, p, i) + p.properties);
    const uint tiles = (n.end - n.begin + kTiledRows - 1) / kTiledRows;
    for (uint prop = 0; prop < f.properties; ++prop) {
      device TiledProperty &q = properties[prop];
      if (!q.bins) continue;
      const uint count = q.histogram == kTiledLocalHistogram ? 0 : tiles * f.predictors;
      if (count) {
        q.histogram = p.histogram + used;
        used += q.bins * f.predictors * (s.symbols + 1) * 4;
      }
      tasks[c.jobs++] = {i, prop, groups, groups + count};
      groups += count;
    }
  }
  if (!c.frontier.status && c.live && !c.selected) Fail(c.frontier, kCapacityError);
  c.histogram_peak = max(c.histogram_peak, used);
  c.maximum_deferred = max(c.maximum_deferred, c.live - c.selected);
  c.collect_dispatch[0] = max(1u, groups); c.collect_dispatch[1] = c.collect_dispatch[2] = 1;
  c.jobs_dispatch[0] = max(1u, c.jobs); c.jobs_dispatch[1] = c.jobs_dispatch[2] = 1;
  c.finish_dispatch[0] = max(1u, c.selected); c.finish_dispatch[1] = c.finish_dispatch[2] = 1;
}

kernel void gjxl_histogram_tiled_clear_histogram(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::TiledParams &p [[buffer(2)]],
    uint job [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using namespace tiled_frontier_impl;
  device TiledControl &c = Control(work, p);
  if (c.frontier.status || job >= c.jobs) return;
  const TiledTask task = reinterpret_cast<device const TiledTask *>(work + p.jobs)[job];
  device TiledSlot &s = Slot(work, p, task.slot);
  device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[s.ref.fit];
  device const TiledProperty &q = reinterpret_cast<device const TiledProperty *>(SlotBytes(work, p, task.slot) + p.properties)[task.property];
  if (q.histogram == kTiledLocalHistogram) return;
  device uint *hist = reinterpret_cast<device uint *>(work + q.histogram);
  for (uint i = lane; i < q.bins * f.predictors * (s.symbols + 1); i += 128) hist[i] = 0;
}

kernel void gjxl_histogram_tiled_collect(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::TiledParams &p [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using namespace tiled_frontier_impl;
  device TiledControl &c = Control(work, p);
  if (c.frontier.status) return;
  device const TiledTask *tasks = reinterpret_cast<device const TiledTask *>(work + p.jobs);
  const uint job = FindTask(tasks, c.jobs, group); if (job == c.jobs) return;
  const TiledTask task = tasks[job];
  device TiledSlot &s = Slot(work, p, task.slot);
  device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[s.ref.fit];
  device const Fit &d = reinterpret_cast<device const Fit *>(work)[s.ref.fit];
  const Node n = reinterpret_cast<device const Node *>(work + d.nodes)[s.ref.node];
  device uchar *base = SlotBytes(work, p, task.slot);
  device const TiledProperty &q = reinterpret_cast<device const TiledProperty *>(base + p.properties)[task.property];
  const uint tiles = (n.end - n.begin + kTiledRows - 1) / kTiledRows;
  const uint pred = (group - task.first) / tiles, tile = (group - task.first) % tiles;
  const uint first = n.begin + tile * kTiledRows, end = min(n.end, first + kTiledRows), stride = s.symbols + 1;
  device const uint *order = reinterpret_cast<device const uint *>(work + d.order);
  device const uint *map = reinterpret_cast<device const uint *>(base + p.bin_map) + task.property * 256;
  device const uchar2 *tokens = reinterpret_cast<device const uchar2 *>(prepared + f.residual_offsets[pred]);
  device const ushort *weights = reinterpret_cast<device const ushort *>(prepared + f.weights_offset);
  device uint *hist = reinterpret_cast<device uint *>(work + q.histogram) + pred * q.bins * stride;
  // At most 512 distinct row keys enter 1024 slots. CAS insertion never waits
  // for another group, and cannot exhaust this table. Counts are integer sums.
  threadgroup atomic_uint keys[kTiledHashEntries], counts[kTiledHashEntries], extra[256];
  const uint cells = q.bins * stride;
  if (cells <= kTiledHashEntries) {
    // Compact histograms fit directly in the existing count storage. Include
    // the extra-bit column, avoiding hash setup and probing for these groups.
    for (uint i = lane; i < cells; i += 128)
      atomic_store_explicit(&counts[i], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = first + lane; i < end; i += 128) {
      const uint row = order[i], weight = weights[row];
      const uint bin = map[Property(prepared, f, task.property, row)];
      const uchar2 token = tokens[row];
      atomic_fetch_add_explicit(&counts[bin * stride + token.x], weight, memory_order_relaxed);
      if (token.y) atomic_fetch_add_explicit(&counts[bin * stride + s.symbols], weight * uint(token.y), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < cells; i += 128)
      Add(hist + i, atomic_load_explicit(&counts[i], memory_order_relaxed));
    return;
  }
  for (uint i = lane; i < kTiledHashEntries; i += 128) {
    atomic_store_explicit(&keys[i], 0xffffffffu, memory_order_relaxed);
    atomic_store_explicit(&counts[i], 0, memory_order_relaxed);
  }
  for (uint i = lane; i < q.bins; i += 128) atomic_store_explicit(&extra[i], 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = first + lane; i < end; i += 128) {
    const uint row = order[i], weight = weights[row], bin = map[Property(prepared, f, task.property, row)];
    const uchar2 token = tokens[row]; const uint key = bin * stride + token.x;
    uint bucket = ((key * 2654435761u) ^ (key >> 10)) & (kTiledHashEntries - 1);
    for (;;) {
      uint expected = 0xffffffffu;
      if (atomic_compare_exchange_weak_explicit(&keys[bucket], &expected, key, memory_order_relaxed, memory_order_relaxed) || expected == key) break;
      bucket = (bucket + 1) & (kTiledHashEntries - 1);
    }
    atomic_fetch_add_explicit(&counts[bucket], weight, memory_order_relaxed);
    if (token.y) atomic_fetch_add_explicit(&extra[bin], weight * uint(token.y), memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = lane; i < kTiledHashEntries; i += 128) {
    const uint key = atomic_load_explicit(&keys[i], memory_order_relaxed);
    if (key != 0xffffffffu) Add(hist + key, atomic_load_explicit(&counts[i], memory_order_relaxed));
  }
  for (uint i = lane; i < q.bins; i += 128) Add(hist + i * stride + s.symbols, atomic_load_explicit(&extra[i], memory_order_relaxed));
}

kernel void gjxl_histogram_tiled_score(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::TiledParams &p [[buffer(2)]],
    uint job [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using namespace tiled_frontier_impl;
  device TiledControl &c = Control(work, p); if (c.frontier.status || job >= c.jobs) return;
  const TiledTask task = reinterpret_cast<device const TiledTask *>(work + p.jobs)[job];
  device TiledSlot &s = Slot(work, p, task.slot);
  device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[s.ref.fit];
  device const Fit &d = reinterpret_cast<device const Fit *>(work)[s.ref.fit];
  const Node n = reinterpret_cast<device const Node *>(work + d.nodes)[s.ref.node];
  device uchar *base = SlotBytes(work, p, task.slot);
  device TiledProperty &q = reinterpret_cast<device TiledProperty *>(base + p.properties)[task.property];
  if (q.histogram == kTiledLocalHistogram) return;
  device const uint *parent = reinterpret_cast<device const uint *>(base + p.parent);
  device const uint *parent_extra = reinterpret_cast<device const uint *>(base + p.parent_extra);
  device const uint *occupancy = reinterpret_cast<device const uint *>(base + p.occupancy) + task.property * 256;
  device const uint *map = reinterpret_cast<device const uint *>(base + p.bin_map) + task.property * 256;
  device const float *penalties = reinterpret_cast<device const float *>(work + d.penalties) + s.parent_index * f.predictors;
  const uint bins = f.bins[task.property], symbols = s.symbols, stride = symbols + 1;
  threadgroup SideWinner sides[512];
  for (uint b = lane; b < bins; b += 128) sides[2 * b] = sides[2 * b + 1] = {kMaximumCost, 0, 0};
  for (uint pred = 0; pred < f.predictors; ++pred) {
    device uint *hist = reinterpret_cast<device uint *>(work + q.histogram) + pred * q.bins * stride;
    // Prefix independent integer columns using every SIMD lane. The native
    // floating-point scoring order below is unchanged.
    for (uint sym = lane / 32; sym < stride; sym += 4) {
      uint carry = 0;
      for (uint first = 0; first < q.bins; first += 32) {
        const uint b = first + (lane & 31);
        const uint value = b < q.bins ? hist[b * stride + sym] : 0;
        const uint sum = carry + simd_prefix_inclusive_sum(value);
        if (b < q.bins) hist[b * stride + sym] = sum;
        carry = simd_broadcast(sum, 31);
      }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    const float penalty = penalties[pred];
    for (uint b = lane; b < bins; b += 128) {
      if (!Threshold(occupancy, b, n.end - n.begin)) continue;
      device const uint *low_hist = hist + map[b] * stride;
      const float2 costs = EntropyPair(low_hist, parent + pred * 256, symbols);
      const float low = costs.x + float(low_hist[symbols]);
      const float high = costs.y + float(parent_extra[pred]) - float(low_hist[symbols]);
      if (low + penalty < sides[2 * b].cost + sides[2 * b].penalty) sides[2 * b] = {low, penalty, f.predictor_ids[pred]};
      if (high + penalty < sides[2 * b + 1].cost + sides[2 * b + 1].penalty) sides[2 * b + 1] = {high, penalty, f.predictor_ids[pred]};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!lane) for (uint b = 0; b < bins; ++b) {
    if (!Threshold(occupancy, b, n.end - n.begin)) continue;
    const SideWinner low = sides[2 * b], high = sides[2 * b + 1];
    const bool wp = f.property_ids[task.property] == 15 || low.predictor == 6 || high.predictor == 6;
    const uint category = f.property_ids[task.property] < 2 ?
        (low.cost == 0 || high.cost == 0 ? 0 : 1) : (wp ? 2 : 3);
    device Candidate &best = q.winners[category];
    if (low.cost + high.cost < best.left_cost + best.right_cost)
      best = {b, occupancy[b], low.predictor, high.predictor, low.cost, high.cost, 1, 0};
  }
}

// Single-tile compact jobs never allocate, clear or publish a device histogram.
// Keep this separate from collection/scoring of large jobs so its shared storage
// and register needs do not change those kernels' resource requirements.
kernel void gjxl_histogram_tiled_collect_score(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::TiledParams &p [[buffer(2)]],
    uint job [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using namespace tiled_frontier_impl;
  device TiledControl &c = Control(work, p); if (c.frontier.status || job >= c.jobs) return;
  const TiledTask task = reinterpret_cast<device const TiledTask *>(work + p.jobs)[job];
  device TiledSlot &s = Slot(work, p, task.slot);
  device uchar *base = SlotBytes(work, p, task.slot);
  device TiledProperty &q = reinterpret_cast<device TiledProperty *>(base + p.properties)[task.property];
  if (q.histogram != kTiledLocalHistogram) return;
  device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[s.ref.fit];
  device const Fit &d = reinterpret_cast<device const Fit *>(work)[s.ref.fit];
  const Node n = reinterpret_cast<device const Node *>(work + d.nodes)[s.ref.node];
  device const uint *order = reinterpret_cast<device const uint *>(work + d.order);
  device const ushort *weights = reinterpret_cast<device const ushort *>(prepared + f.weights_offset);
  device const uint *parent = reinterpret_cast<device const uint *>(base + p.parent);
  device const uint *parent_extra = reinterpret_cast<device const uint *>(base + p.parent_extra);
  device const uint *occupancy = reinterpret_cast<device const uint *>(base + p.occupancy) + task.property * 256;
  device const uint *map = reinterpret_cast<device const uint *>(base + p.bin_map) + task.property * 256;
  device const float *penalties = reinterpret_cast<device const float *>(work + d.penalties) + s.parent_index * f.predictors;
  const uint bins = f.bins[task.property], symbols = s.symbols, stride = symbols + 1, cells = q.bins * stride;
  threadgroup SmallJobScratch scratch;
  threadgroup uint *hist = scratch.histogram;
  SideWinner lows[2], highs[2];
  for (uint i = 0; i < 2; ++i) lows[i] = highs[i] = {kMaximumCost, 0, 0};
  threadgroup atomic_uint *counts = reinterpret_cast<threadgroup atomic_uint *>(hist);
  // Preserve canonical predictor preference. Each iteration owns the entire
  // node histogram and finishes all reads before reusing it for the next one.
  for (uint pred = 0; pred < f.predictors; ++pred) {
    for (uint i = lane; i < cells; i += 128) hist[i] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uchar2 *tokens = reinterpret_cast<device const uchar2 *>(prepared + f.residual_offsets[pred]);
    for (uint i = n.begin + lane; i < n.end; i += 128) {
      const uint row = order[i], weight = weights[row];
      const uint bin = map[Property(prepared, f, task.property, row)];
      const uchar2 token = tokens[row];
      atomic_fetch_add_explicit(&counts[bin * stride + token.x], weight, memory_order_relaxed);
      if (token.y) atomic_fetch_add_explicit(&counts[bin * stride + symbols], weight * uint(token.y), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint sym = lane / 32; sym < stride; sym += 4) {
      uint carry = 0;
      for (uint first = 0; first < q.bins; first += 32) {
        const uint b = first + (lane & 31);
        const uint value = b < q.bins ? hist[b * stride + sym] : 0;
        const uint sum = carry + simd_prefix_inclusive_sum(value);
        if (b < q.bins) hist[b * stride + sym] = sum;
        carry = simd_broadcast(sum, 31);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float penalty = penalties[pred];
    #pragma unroll
    for (uint i = 0; i < 2; ++i) {
      // Every compact bin except the last is a valid split threshold. Assign
      // these densely to lanes; the ordered scan below restores original bins.
      const uint compact = lane + 128 * i;
      if (compact + 1 >= q.bins) continue;
      threadgroup const uint *low_hist = hist + compact * stride;
      const float2 costs = EntropyPair(low_hist, parent + pred * 256, symbols);
      const float low = costs.x + float(low_hist[symbols]);
      const float high = costs.y + float(parent_extra[pred]) - float(low_hist[symbols]);
      if (low + penalty < lows[i].cost + lows[i].penalty) lows[i] = {low, penalty, f.predictor_ids[pred]};
      if (high + penalty < highs[i].cost + highs[i].penalty) highs[i] = {high, penalty, f.predictor_ids[pred]};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint compact = lane; compact + 1 < q.bins; compact += 128) {
    scratch.sides[2 * compact] = lows[compact / 128];
    scratch.sides[2 * compact + 1] = highs[compact / 128];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!lane) for (uint b = 0; b < bins; ++b) {
    if (!Threshold(occupancy, b, n.end - n.begin)) continue;
    const SideWinner low = scratch.sides[2 * map[b]], high = scratch.sides[2 * map[b] + 1];
    const bool wp = f.property_ids[task.property] == 15 || low.predictor == 6 || high.predictor == 6;
    const uint category = f.property_ids[task.property] < 2 ?
        (low.cost == 0 || high.cost == 0 ? 0 : 1) : (wp ? 2 : 3);
    device Candidate &best = q.winners[category];
    if (low.cost + high.cost < best.left_cost + best.right_cost)
      best = {b, occupancy[b], low.predictor, high.predictor, low.cost, high.cost, 1, 0};
  }
}

kernel void gjxl_histogram_tiled_finish(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::TiledParams &p [[buffer(2)]],
    uint index [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using namespace tiled_frontier_impl;
  device TiledControl &c = Control(work, p);
  // Do not inspect a concurrently updated global error here: every selected
  // group follows its own uniform path and publication checks the final error.
  if (index >= c.selected) return;
  const uint slot = reinterpret_cast<device const uint *>(work + p.selected)[index];
  device TiledSlot &s = Slot(work, p, slot);
  device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[s.ref.fit];
  device const Fit &d = reinterpret_cast<device const Fit *>(work)[s.ref.fit];
  device State &fit_state = *reinterpret_cast<device State *>(work + d.state);
  device Node *nodes = reinterpret_cast<device Node *>(work + d.nodes);
  device Node &node = nodes[s.ref.node];
  device const TiledProperty *properties = reinterpret_cast<device const TiledProperty *>(SlotBytes(work, p, slot) + p.properties);
  threadgroup Split chosen;
  threadgroup uint groups[4], error;
  if (!lane) {
    error = 0;
    Split best[4]; for (uint i = 0; i < 4; ++i) best[i] = {0, EmptyCandidate()};
    for (uint prop = 0; prop < f.properties; ++prop)
      for (uint cat = 0; cat < 4; ++cat) {
        const Candidate v = properties[prop].winners[cat];
        if (v.left_cost + v.right_cost < Cost(best[cat])) best[cat] = {prop, v};
      }
    chosen = best[2];
    if (Cost(best[3]) + d.split_threshold < s.base_cost && Cost(best[3]) <= d.fast_decode_multiplier * Cost(chosen)) chosen = best[3];
    if (Cost(best[1]) + d.split_threshold < s.base_cost && Cost(best[1]) <= d.fast_decode_multiplier * Cost(chosen)) chosen = best[1];
    if (Cost(best[0]) + d.split_threshold < s.base_cost) chosen = best[0];
    if (!(Cost(chosen) + d.split_threshold < s.base_cost)) chosen.candidate.valid = 0;
    Add(&fit_state.visited, 1);
    if (chosen.candidate.valid) {
      if (node.depth >= d.maximum_depth) error = kCapacityError;
      else {
        const uint first = atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(&fit_state.size), 2, memory_order_relaxed);
        if (first + 2 > d.maximum_nodes) error = kCapacityError;
        else {
          const uint cut = node.begin + chosen.candidate.low_rows;
          node.property = int(f.property_ids[chosen.property]);
          node.split = reinterpret_cast<device const int *>(work + d.thresholds[chosen.property])[chosen.candidate.bin];
          node.left = first; node.right = first + 1;
          nodes[first] = {-1, 0, 0, 0, chosen.candidate.right_predictor, cut, node.end, node.depth + 1, node.constants};
          nodes[first + 1] = {-1, 0, 0, 0, chosen.candidate.left_predictor, node.begin, cut, node.depth + 1, node.constants};
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (!error && chosen.candidate.valid) {
    const uint cut = node.begin + chosen.candidate.low_rows;
    device uint *order = reinterpret_cast<device uint *>(work + d.order);
    device uint *partition = reinterpret_cast<device uint *>(work + d.partition);
    Misplaced(prepared, f, order, partition, s.low_count, node.begin, cut, chosen.property, chosen.candidate.bin, true, lane, groups);
    Misplaced(prepared, f, order, partition, s.high_count, cut, node.end, chosen.property, chosen.candidate.bin, false, lane, groups);
    if (!lane && s.low_count != s.high_count) error = kPartitionError;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (!error) {
      for (uint i = lane; i < s.low_count; i += 128) {
        const uint low = partition[node.begin + i], high = partition[cut + i];
        const uint temp = order[low]; order[low] = order[high]; order[high] = temp;
      }
      threadgroup_barrier(mem_flags::mem_device);
      if (!lane) {
        const uint first = atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(&c.frontier.tail), 2, memory_order_relaxed);
        if (first + 2 > p.frontier.queue_capacity) error = kCapacityError;
        else {
          device NodeRef *queue = reinterpret_cast<device NodeRef *>(work + p.frontier.queue);
          queue[first] = {s.ref.fit, node.left}; queue[first + 1] = {s.ref.fit, node.right};
        }
      }
    }
  }
  if (!lane) {
    if (error) Fail(c.frontier, error);
    s.stage = kTiledFree;
    Add(&c.completed, 1);
    atomic_fetch_sub_explicit(reinterpret_cast<device atomic_uint *>(&c.live), 1, memory_order_relaxed);
  }
}
