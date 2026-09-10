// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>

#include "codec/ac_strategy_search_policy.h"
#include "codec/ac_strategy_storage_plan.h"
#include "core/geometry.h"
#include "core/status.h"

namespace gjxl::ac_strategy_search_internal {

struct StageStoragePlan {
  size_t candidate_count = 0;
  size_t candidate_bytes = 0;
  // Host matrices exist even for an empty family; its device buffers do not.
  size_t matrix_bytes = 0;
  size_t cost_bytes = 0;
  bool operator==(const StageStoragePlan &) const = default;
};

struct StoragePlan {
  Extent2D block_extent;
  Extent2D tile_extent;
  size_t pixel_count = 0;
  size_t block_count = 0;
  size_t opsin_bytes = 0;
  size_t mask_bytes = 0;
  size_t block_cost_bytes_per_stage = 0;
  std::array<StageStoragePlan, ac_strategy_internal::kCandidateStages.size()>
      stages;
  size_t maximum_packed_bytes = 0; // Two separate backings of this size.
  size_t maximum_rate_bytes = 0;
  size_t device_bytes = 0;
  bool operator==(const StoragePlan &) const = default;
};

/// Exact fresh device allocation requests, independent of image values and CPU
/// placement decisions. Reused buffers/vector capacities and replacement
/// overlap still need admission-aware handling; device_bytes is not a
/// whole-work bound. No backend access or heap allocation on success (an error
/// Status may allocate its small diagnostic string); failure leaves output
/// unchanged.
[[nodiscard]] Status ComputeStoragePlan(Extent2D coding, bool resident,
                                        StoragePlan *out);

struct HostStoragePlan {
  // Candidates, quantization matrices, readback costs and dense cost tables.
  resource_budget_internal::HostStorageBound prepared;
  // Packed Opsin/mask survive through merge in the nonresident path only.
  resource_budget_internal::HostStorageBound staging;
  ac_strategy_internal::SearchStoragePlan merge;
  // Complete host owner inventory, including prepared/staging/merge output.
  resource_budget_internal::HostStorageBound working;
  bool operator==(const HostStoragePlan &) const = default;
};

/// Backend-independent search host backing. Device buffers, backend submission
/// metadata/profiles, borrowed inputs and old output grids are separate. With
/// reuse_prepared=false, state must start empty. With true, coding must bound
/// BOTH dimensions of EVERY search since state was empty; a bound on pixel area
/// alone is insufficient. Costs resize with geometric growth; candidates use
/// explicit reserve and dense tables use assign. This does not transfer an old
/// prepared owner's charges to a new reservation. No backing allocation on
/// success; failure leaves output unchanged.
[[nodiscard]] Status ComputeHostStoragePlan(Extent2D coding, bool resident,
                                            bool reuse_prepared,
                                            HostStoragePlan *out);

} // namespace gjxl::ac_strategy_search_internal
