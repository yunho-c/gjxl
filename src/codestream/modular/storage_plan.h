// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/entropy_storage_plan.h"
#include "codestream/modular/tokenization.h"

namespace gjxl::modular_internal {
struct ModularWorkflowStoragePlan {
  ModularFrameGeometry geometry;
  size_t tokens = 0, streams = 0;
  size_t maximum_global_bits = 0, maximum_group_bits = 0;
  size_t maximum_codestream_bytes = 0;
  resource_budget_internal::HostStorageBound preparation, modeling, emission;
  resource_budget_internal::HostStorageBound assembly, publication, output, working;
  bool operator==(const ModularWorkflowStoragePlan &) const = default;
};
// Shape-only, allocation-free; bounds fresh work and output. An existing managed
// result stays charged separately in its domain during replacement.
[[nodiscard]] Status ComputeModularWorkflowStoragePlan(Extent2D extent, EntropyCodingMode mode,
                                                       ModularWorkflowStoragePlan *out);
} // namespace gjxl::modular_internal
