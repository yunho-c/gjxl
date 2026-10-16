// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "../../ops/modular_histogram_fitting_layout.h"
namespace gjxl_histogram_fitting {
// Scratch offsets are relative to one reusable property slot. Tasks index the
// request's node descriptors; a wave maps consecutive tasks to those slots.
struct Params {
  Word nodes, tasks_offset, status_offset, shards;
  Word first_task, tasks, slot_bytes, maximum_bins;
  Word maximum_predictors, maximum_symbols;
  Word shard_histograms, shard_extra, shard_occupancy;
  Word histograms, extra, occupancy, candidate_costs, side_winners;
};
} // namespace gjxl_histogram_fitting
