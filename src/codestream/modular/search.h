// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/storage_plan.h"
#include "codec/modular/tree_learning.h"
namespace gjxl::modular_internal {
// Deterministic uniform sampling in canonical stream/channel/raster order.
// Outputs are a learned single leaf and at most one split (two leaves).
[[nodiscard]] Status LearnModularPolicies(const ModularEncoderFrame &frame,
                                          const ModularStreamPlan &plan, uint8_t rct,
                                          ModularCodingPolicy *single, ModularCodingPolicy *split);
[[nodiscard]] Status ComputeModularSearchStoragePlan(Extent2D extent, PackedModularFormat format,
                                                     EntropyCodingMode mode,
                                                     ModularWorkflowStoragePlan *out);
} // namespace gjxl::modular_internal
