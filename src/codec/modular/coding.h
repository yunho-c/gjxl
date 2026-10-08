// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/tree.h"
#include "codec/modular/transform/transform.h"

namespace gjxl::modular_internal {
struct ModularCodingPolicy {
  DecisionTree tree;
  WeightedPredictorParameters weighted;
  uint8_t rct = 0; // 0 is identity; 1..41 affect only the first three color planes.
  TransformSequence transforms;
  bool operator==(const ModularCodingPolicy &) const = default;
};
[[nodiscard]] inline Status ValidateCodingPolicy(const ModularCodingPolicy &policy) {
  TreeLayout layout;
  if (!policy.weighted.valid() || policy.rct >= 42)
    return Status::InvalidArgument("Invalid Modular coding policy");
  if (auto s = ValidateTransforms(policy.transforms); !s.ok())
    return s;
  return ValidateTree(policy.tree, &layout);
}
} // namespace gjxl::modular_internal
