// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/geometry.h"
#include "core/host_storage_bound.h"

namespace gjxl::frontend_storage_internal {

/// Native CPU Butteraugli only, not Metal arenas. Caller RGB/map storage and
/// any previous prepared owner are separate. Small pimpl/control objects and
/// stacks follow the common managed-boundary exclusions.
struct NativeButteraugliStoragePlan {
  Extent2D working_extent;
  Extent2D sub_extent; // Empty when there is no half-resolution scale.
  resource_budget_internal::HostStorageBound prepared;
  // Complete fresh preparation, including the returned prepared owner.
  resource_budget_internal::HostStorageBound preparation;
  // Complete comparison, INCLUDING the prepared owner and temporary replacement
  // psychoimage. Repeated comparisons at the prepared geometry fit this bound.
  resource_budget_internal::HostStorageBound comparison;
  // Complete public ComputeButteraugliDistance scratch. Also covers internal
  // NativeButteraugliScratch reuse at this SAME requested geometry, including
  // main/subscale backing replacement and retained staging after failures.
  resource_budget_internal::HostStorageBound one_shot;
  bool operator==(const NativeButteraugliStoragePlan &) const = default;
};

/// Pure checked O(1) plan, atomic on invalid inputs/overflow. No public-domain
/// installation or complete workflow admission is implied by this component.
[[nodiscard]] Status
ComputeNativeButteraugliStoragePlan(Extent2D requested,
                                    NativeButteraugliStoragePlan *out);

} // namespace gjxl::frontend_storage_internal
