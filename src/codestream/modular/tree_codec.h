// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "codestream/entropy_storage_plan.h"

namespace gjxl::modular_internal {

inline constexpr size_t kTreeContextCount = 6;

struct GlobalTreeStoragePlan {
  size_t maximum_bits = 0;
  // Includes the optimizer's resulting model and all write scratch, but not
  // input tokens or the caller's destination writer/history.
  resource_budget_internal::HostStorageBound scratch;
  bool operator==(const GlobalTreeStoragePlan&) const = default;
};

// Checked, allocation-free planning; failure leaves *out unchanged.
[[nodiscard]] Status ComputeGlobalTreeStoragePlan(
    size_t maximum_tokens, GlobalTreeStoragePlan* out);

// The adapter supplies resolved, valid tree tokens (wire contexts 0..5).
// This is emission, not tree validation or learning. No frame/group knowledge.
// The caller owns the atomic temporary/allotment and catches allocation errors.
[[nodiscard]] Status WriteGlobalTreeInTransaction(
    std::span<const EntropyToken> tokens, BitWriter* writer);

}  // namespace gjxl::modular_internal
