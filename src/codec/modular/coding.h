// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/tree.h"

namespace gjxl::modular_internal {
struct ModularCodingPolicy {
  DecisionTree tree;
  WeightedPredictorParameters weighted;
  uint8_t rct = 0; // 0 is identity; 1..41 affect only the first three color planes.
  bool operator==(const ModularCodingPolicy &) const = default;
};
[[nodiscard]] inline Status ValidateCodingPolicy(const ModularCodingPolicy &policy) {
  TreeLayout layout;
  if (!policy.weighted.valid() || policy.rct >= 42)
    return Status::InvalidArgument("Invalid Modular coding policy");
  return ValidateTree(policy.tree, &layout);
}
} // namespace gjxl::modular_internal
