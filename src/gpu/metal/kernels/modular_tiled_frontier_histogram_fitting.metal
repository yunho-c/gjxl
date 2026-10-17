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
Candidate EmptyCandidate() { return {0, 0, 0, 0, kMaximumCost, kMaximumCost, 0, 0}; }
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
  if (lane) return;
  device TiledControl &c = Control(work, p);
  c.row_tasks = 0;
  uint groups = 0;
  if (!c.frontier.status) {
    c.frontier.maximum_ready = max(c.frontier.maximum_ready, c.frontier.tail - c.frontier.head + c.live);
    device const NodeRef *queue = reinterpret_cast<device const NodeRef *>(work + p.frontier.queue);
    device TiledTask *tasks = reinterpret_cast<device TiledTask *>(work + p.row_tasks);
    uint counts[64] = {};
    for (uint i = 0; i < p.frontier.active_nodes; ++i) {
      device TiledSlot &s = Slot(work, p, i);
      if (s.stage == kTiledFree && c.frontier.head < c.frontier.tail) {
        s = {queue[c.frontier.head++], kTiledNew, 0, 0, 0, 0, 0, 0};
        ++c.live;
        device const Fit &d = reinterpret_cast<device const Fit *>(work)[s.ref.fit];
        device const Node &n = reinterpret_cast<device const Node *>(work + d.nodes)[s.ref.node];
        const uint tiles = (n.end - n.begin + kTiledRows - 1) / kTiledRows;
        tasks[c.row_tasks++] = {i, 0, groups, groups + tiles}; groups += tiles;
      }
      if (s.stage != kTiledFree)
        c.frontier.maximum_same_fit = max(c.frontier.maximum_same_fit, ++counts[s.ref.fit]);
    }
    c.frontier.maximum_active = max(c.frontier.maximum_active, c.live);
    c.frontier.rounds += c.live != 0;
  }
  c.stats_dispatch[0] = max(1u, groups); c.stats_dispatch[1] = c.stats_dispatch[2] = 1;
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
      atomic_fetch_add_explicit(&extra, weight * uint(t.y), memory_order_relaxed);
      maximum = max(maximum, uint(t.x) + 1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint *parent = reinterpret_cast<device uint *>(base + p.parent) + pred * 256;
    for (uint i = lane; i < 256; i += 128) Add(parent + i, atomic_load_explicit(&counts[i], memory_order_relaxed));
    if (!lane) Add(reinterpret_cast<device uint *>(base + p.parent_extra) + pred, atomic_load_explicit(&extra, memory_order_relaxed));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
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
  if (s.stage != kTiledNew || lane) return;
  device const FitDescriptor &f = reinterpret_cast<device const FitDescriptor *>(prepared)[s.ref.fit];
  device const Fit &d = reinterpret_cast<device const Fit *>(work)[s.ref.fit];
  device Node &n = reinterpret_cast<device Node *>(work + d.nodes)[s.ref.node];
  device uchar *base = SlotBytes(work, p, slot);
  device const uint *parent = reinterpret_cast<device const uint *>(base + p.parent);
  device const uint *extra = reinterpret_cast<device const uint *>(base + p.parent_extra);
  s.symbols = (s.symbols + 3) & ~3u;
  while (f.predictor_ids[s.parent_index] != n.predictor) ++s.parent_index;
  s.base_cost = Entropy(parent + s.parent_index * 256,
      reinterpret_cast<device const int *>(parent + s.parent_index * 256), s.symbols, false) + float(extra[s.parent_index]);
  device TiledProperty *properties = reinterpret_cast<device TiledProperty *>(base + p.properties);
  for (uint prop = 0; prop < f.properties; ++prop) {
    device TiledProperty &q = properties[prop]; q.bins = 0;
    for (uint c = 0; c < 4; ++c) q.winners[c] = EmptyCandidate();
    if (!(s.base_cost > d.split_threshold)) continue;
    if ((n.constants & (1u << prop)) || f.bins[prop] == 1) { n.constants |= 1u << prop; continue; }
    device uint *occupancy = reinterpret_cast<device uint *>(base + p.occupancy) + prop * 256;
    device uint *map = reinterpret_cast<device uint *>(base + p.bin_map) + prop * 256;
    uint sum = 0;
    for (uint b = 0; b < f.bins[prop]; ++b) {
      const uint count = occupancy[b];
      map[b] = q.bins; q.bins += count != 0;
      sum += count; occupancy[b] = sum;
    }
    if (q.bins <= 1) { n.constants |= 1u << prop; q.bins = 0; }
    s.needed += q.bins * f.predictors * (s.symbols + 1) * 4;
  }
  s.stage = kTiledPending;
}

kernel void gjxl_histogram_tiled_schedule(
    device const uchar *prepared [[buffer(0)]], device uchar *work [[buffer(1)]],
    constant gjxl_histogram_resident::TiledParams &p [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]) {
  using namespace tiled_frontier_impl;
  if (lane) return;
  device TiledControl &c = Control(work, p);
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
      q.histogram = p.histogram + used;
      used += q.bins * f.predictors * (s.symbols + 1) * 4;
      tasks[c.jobs++] = {i, prop, groups, groups + tiles * f.predictors};
      groups += tiles * f.predictors;
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
    atomic_fetch_add_explicit(&extra[bin], weight * uint(token.y), memory_order_relaxed);
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
    for (uint sym = lane; sym < stride; sym += 128) {
      uint sum = 0;
      for (uint b = 0; b < q.bins; ++b) { sum += hist[b * stride + sym]; hist[b * stride + sym] = sum; }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    const float penalty = penalties[pred];
    for (uint b = lane; b < bins; b += 128) {
      if (!Threshold(occupancy, b, n.end - n.begin)) continue;
      device const uint *low_hist = hist + map[b] * stride;
      const float low = Entropy(low_hist, reinterpret_cast<device const int *>(parent + pred * 256), symbols, false) + float(low_hist[symbols]);
      const float high = Entropy(low_hist, reinterpret_cast<device const int *>(parent + pred * 256), symbols, true) + float(parent_extra[pred]) - float(low_hist[symbols]);
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
