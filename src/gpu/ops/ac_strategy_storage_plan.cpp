// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/ops/ac_strategy_storage_plan.h"

#include <algorithm>
#include <limits>

#include "codec/chroma_from_luma.h"
#include "gpu/ops/ac_strategy.h"

namespace gjxl::ac_strategy_search_internal {
namespace {
constexpr size_t kTileBlocks = kColorTileDimension / kJxlBlockDimension;
static_assert(kTileBlocks == 8);

// The old tile-by-tile sum is separable: sum_y sum_x (nx * ny) equals
// (sum_x nx) * (sum_y ny). Count full tiles plus a possible partial final tile.
size_t AxisPositions(size_t blocks, size_t covered, size_t step) {
  const auto in_tile = [=](size_t width) {
    return width < covered ? size_t{0} : (width - covered) / step + 1;
  };
  return (blocks / kTileBlocks) * in_tile(kTileBlocks) +
         in_tile(blocks % kTileBlocks);
}

bool Multiply(size_t a, size_t b, size_t *out) {
  if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
    return false;
  *out = a * b;
  return true;
}
bool Add(size_t value, size_t *total) {
  if (value > std::numeric_limits<size_t>::max() - *total)
    return false;
  *total += value;
  return true;
}
} // namespace

Status ComputeStoragePlan(Extent2D coding, bool resident, StoragePlan *out) {
  if (out == nullptr || coding.empty() ||
      coding.width % kJxlBlockDimension != 0 ||
      coding.height % kJxlBlockDimension != 0) {
    return Status::InvalidArgument(
        "GPU AC-strategy search requires a padded opsin image and plan output");
  }
  StoragePlan plan;
  plan.block_extent = {coding.width / kJxlBlockDimension,
                       coding.height / kJxlBlockDimension};
  plan.tile_extent = coding.ceil_div(kColorTileDimension);
  constexpr size_t kMax = std::numeric_limits<uint32_t>::max();
  if (coding.width > kMax || coding.height > kMax ||
      !coding.try_area(&plan.pixel_count) || plan.pixel_count > kMax ||
      !plan.block_extent.try_area(&plan.block_count)) {
    return Status::InvalidArgument(
        "GPU AC-strategy search exceeds 32-bit indexing limits");
  }
  if (!Multiply(plan.block_count, sizeof(float),
                &plan.block_cost_bytes_per_stage) ||
      (!resident &&
       (!Multiply(plan.pixel_count, 3 * sizeof(float), &plan.opsin_bytes) ||
        !Multiply(plan.pixel_count, sizeof(float), &plan.mask_bytes)))) {
    return Status::InvalidArgument("GPU AC-strategy input storage overflows");
  }
  for (size_t i = 0; i < plan.stages.size(); ++i) {
    const auto &stage = ac_strategy_internal::kCandidateStages[i];
    const auto *info = GetAcStrategyInfo(stage.strategy);
    if (info == nullptr || stage.anchor_step == 0 ||
        info->covered_blocks.empty() ||
        info->covered_blocks.width > kTileBlocks ||
        info->covered_blocks.height > kTileBlocks) {
      return Status::Internal("GPU AC-strategy candidate policy is invalid");
    }
    auto &family = plan.stages[i];
    size_t packed_values = 0, packed_bytes = 0, rate_bytes = 0;
    if (!Multiply(AxisPositions(plan.block_extent.width,
                                info->covered_blocks.width, stage.anchor_step),
                  AxisPositions(plan.block_extent.height,
                                info->covered_blocks.height, stage.anchor_step),
                  &family.candidate_count) ||
        family.candidate_count > kMax ||
        !Multiply(family.candidate_count, sizeof(AcStrategyCandidate),
                  &family.candidate_bytes) ||
        !Multiply(info->coefficient_count(),
                  kAcStrategyCostMatrixCount * sizeof(float),
                  &family.matrix_bytes) ||
        !Multiply(family.candidate_count, sizeof(float), &family.cost_bytes) ||
        !Multiply(family.candidate_count, 3 * info->coefficient_count(),
                  &packed_values) ||
        !Multiply(packed_values, sizeof(float), &packed_bytes) ||
        !Multiply(family.candidate_count,
                  3 * kAcStrategyRateScratchBytesPerChannel, &rate_bytes)) {
      return Status::InvalidArgument("GPU AC-strategy storage plan overflows");
    }
    plan.maximum_packed_bytes =
        std::max(plan.maximum_packed_bytes, packed_bytes);
    plan.maximum_rate_bytes = std::max(plan.maximum_rate_bytes, rate_bytes);
    if (family.candidate_count != 0 &&
        (!Add(family.candidate_bytes, &plan.device_bytes) ||
         !Add(family.matrix_bytes, &plan.device_bytes) ||
         !Add(family.cost_bytes, &plan.device_bytes))) {
      return Status::InvalidArgument(
          "GPU AC-strategy device storage sum overflows");
    }
  }
  if (!Add(plan.opsin_bytes, &plan.device_bytes) ||
      !Add(plan.mask_bytes, &plan.device_bytes) ||
      !Add(plan.maximum_packed_bytes, &plan.device_bytes) ||
      !Add(plan.maximum_packed_bytes, &plan.device_bytes) ||
      !Add(plan.maximum_rate_bytes, &plan.device_bytes)) {
    return Status::InvalidArgument(
        "GPU AC-strategy device storage sum overflows");
  }
  *out = plan;
  return Status::Ok();
}

Status ComputeHostStoragePlan(Extent2D coding, bool resident,
                              bool reuse_prepared, HostStoragePlan *out) {
  if (out == nullptr)
    return Status::InvalidArgument("AC-search host plan output is null");
  StoragePlan device;
  Status status = ComputeStoragePlan(coding, resident, &device);
  if (!status.ok())
    return status;
  HostStoragePlan plan;
  status = ac_strategy_internal::ComputeSearchStoragePlan(coding, &plan.merge);
  if (!status.ok())
    return status;
  using enum resource_budget_internal::VectorCapacityPolicy;
  const auto reserved = reuse_prepared ? kReusedExact : kFreshExact;
  const auto resized = reuse_prepared ? kGrowing : kFreshExact;
  for (const auto &stage : device.stages) {
    // Matrices exist even for empty families. Their size is constant across
    // searches, so resize never grows an already initialized matrix owner.
    if (!plan.prepared.AddVector<AcStrategyCandidate>(stage.candidate_count,
                                                      reserved) ||
        !plan.prepared.AddVector<float>(stage.matrix_bytes / sizeof(float),
                                        kFreshExact) ||
        !plan.prepared.AddVector<float>(stage.candidate_count, resized) ||
        !plan.prepared.AddVector<float>(device.block_count, reserved))
      return Status::OutOfMemory("AC-search host storage bound overflows");
  }
  // PackOpsin is one contiguous three-channel vector, not three vectors.
  if ((!resident &&
       (!plan.staging.AddVector<float>(device.opsin_bytes / sizeof(float),
                                       kFreshExact) ||
        !plan.staging.AddVector<float>(device.pixel_count, kFreshExact))) ||
      !plan.working.Add(plan.prepared) || !plan.working.Add(plan.staging) ||
      !plan.working.Add(plan.merge.working))
    return Status::OutOfMemory("AC-search host storage bound overflows");
  *out = plan;
  return Status::Ok();
}
} // namespace gjxl::ac_strategy_search_internal
