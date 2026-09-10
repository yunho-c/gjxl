// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/geometry.h"
#include "core/host_storage_bound.h"
#include "core/status.h"

namespace gjxl::ac_strategy_internal {

struct SearchStoragePlan {
  resource_budget_internal::HostStorageBound output;
  // Includes output and the temporary placement grid, which overlap at export.
  // The retained field is an owner-inventory bound, not phase-exit retention.
  resource_budget_internal::HostStorageBound working;
  bool operator==(const SearchStoragePlan &) const = default;
};

/// CPU search, with either computed or supplied candidate costs. Search is
/// serial; transform, cost and tile scratch are stack-only. Caller
/// image/fields, supplied cost tables and an old output grid are separate
/// owners. Geometry is padded pixels. No backing allocation on success; failure
/// is atomic.
[[nodiscard]] Status ComputeSearchStoragePlan(Extent2D coding,
                                              SearchStoragePlan *out);

} // namespace gjxl::ac_strategy_internal
