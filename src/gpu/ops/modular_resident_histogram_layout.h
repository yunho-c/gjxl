// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "modular_histogram_fitting_layout.h"
namespace gjxl_histogram_resident {
using gjxl_histogram_fitting::Word;
enum : Word { kCapacityError = 1, kPartitionError = 2 };
// Every node owns its range and inherited state. The first executor uses a DFS
// index stack; a later frontier executor can queue these same records.
struct Node {
  int property, split;
  Word left, right, predictor;
  Word begin, end, depth, constants;
};
struct State {
  Word status, size, pending, visited;
  Word active, symbols, low_count, high_count;
  float base_cost;
};
// Byte offsets are relative to the resident work arena. Prepared sample
// columns use the existing immutable FitDescriptor and remain untouched.
struct Fit {
  Word state, nodes, stack, order, partition;
  Word parent, parent_extra, histogram, extra, occupancy, sides;
  Word thresholds[16], penalties;
  Word maximum_nodes, maximum_depth, maximum_bins;
  float split_threshold, fast_decode_multiplier;
};
static_assert(sizeof(Node) == 36);
static_assert(sizeof(State) == 36);
static_assert(sizeof(Fit) == 132);
// Append-only ready queue. Entries are consumed only after the dispatch that
// produced them completes, so active ranges are disjoint even within a tree.
struct NodeRef { Word fit, node; };
struct FrontierControl {
  Word status, head, tail, first, count;
  Word rounds, maximum_ready, maximum_active, maximum_same_fit;
  Word dispatch[3];
};
struct FrontierScratch {
  Word state, parent, parent_extra, histogram, extra, occupancy, sides, bytes;
};
struct FrontierParams {
  Word fits, queue, control, scratch, active_nodes, queue_capacity;
  FrontierScratch slot;
};
static_assert(sizeof(NodeRef) == 8);
static_assert(sizeof(FrontierControl) == 48);
static_assert(sizeof(FrontierParams) == 56);
// The tiled executor retains summaries in bounded active slots while a shared
// histogram pool admits a subset. No node is rescanned just because the pool
// was full. Only the original-bin dimension is compacted; symbol indices and
// the four-lane floating-point entropy reduction remain unchanged.
enum : Word { kTiledFree = 0, kTiledNew = 1, kTiledPending = 2, kTiledSelected = 3,
              kTiledRows = 512, kTiledHashEntries = 1024 };
struct TiledSlot {
  NodeRef ref;
  Word stage, symbols, parent_index, needed, low_count, high_count;
  float base_cost;
};
struct TiledProperty {
  Word bins, histogram;
  gjxl_histogram_fitting::Candidate winners[4];
};
struct TiledTask { Word slot, property, first, end; };
struct TiledControl {
  FrontierControl frontier; // common diagnostic counters/status/ready queue
  Word live, completed, row_tasks, jobs, selected, histogram_peak, maximum_deferred;
  Word stats_dispatch[3], collect_dispatch[3], jobs_dispatch[3], finish_dispatch[3];
};
struct TiledParams {
  FrontierParams frontier;
  Word row_tasks, jobs, selected, histogram, histogram_bytes;
  Word parent, parent_extra, occupancy, bin_map, properties;
};
static_assert(sizeof(TiledSlot) == 36);
static_assert(sizeof(TiledProperty) == 136);
static_assert(sizeof(TiledTask) == 16);
static_assert(sizeof(TiledControl) == 124);
static_assert(sizeof(TiledParams) == 96);
} // namespace gjxl_histogram_resident
