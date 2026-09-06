// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/ac_strategy_storage_plan.h"

#include "core/block_grid.h"

namespace gjxl::ac_strategy_internal {

Status ComputeSearchStoragePlan(Extent2D coding, SearchStoragePlan *out) {
  size_t blocks = 0;
  if (out == nullptr || !BlockGrid::IsPaddedPixelExtent(coding) ||
      !BlockGrid::FromPaddedPixelExtent(coding).blocks.try_area(&blocks))
    return Status::InvalidArgument("AC-search storage geometry is invalid");
  SearchStoragePlan plan;
  if (!plan.output.AddVector<uint8_t>(
          blocks,
          resource_budget_internal::VectorCapacityPolicy::kFreshExact) ||
      !plan.working.Add(plan.output, 2))
    return Status::OutOfMemory("AC-search host storage bound overflows");
  *out = plan;
  return Status::Ok();
}

} // namespace gjxl::ac_strategy_internal
