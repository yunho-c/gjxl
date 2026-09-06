// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/token_storage_plan.h"

#include <algorithm>
#include <array>
#include <limits>

#include "codec/vardct_frame.h"
#include "codestream/ac_group.h"
#include "codestream/dc_group.h"
#include "codestream/entropy_internal.h"

namespace gjxl::codestream_internal {
namespace {
bool Multiply(size_t a, size_t b, size_t *out) {
  if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
    return false;
  *out = a * b;
  return true;
}

// Interior, right strip, bottom strip, corner. Group boundaries are multiples
// of color-tile dimensions, so local metadata tile counts sum correctly too.
template <typename Function>
Status ForGroupClasses(Extent2D blocks, size_t dimension, Function &&function) {
  const std::array<size_t, 2> widths{dimension, blocks.width % dimension};
  const std::array<size_t, 2> heights{dimension, blocks.height % dimension};
  const std::array<size_t, 2> nx{blocks.width / dimension, 1};
  const std::array<size_t, 2> ny{blocks.height / dimension, 1};
  for (size_t y = 0; y < 2; ++y) {
    for (size_t x = 0; x < 2; ++x) {
      if (widths[x] == 0 || heights[y] == 0 || nx[x] == 0 || ny[y] == 0)
        continue;
      size_t count = 0;
      if (!Multiply(nx[x], ny[y], &count))
        return Status::OutOfMemory("Token group class count overflows");
      Status status = function(Extent2D{widths[x], heights[y]}, count);
      if (!status.ok())
        return status;
    }
  }
  return Status::Ok();
}

Status Overflow() {
  return Status::OutOfMemory("Aggregate token storage bound overflows");
}
} // namespace

Status ComputeTokenizationStoragePlan(Extent2D blocks,
                                      const TokenizationStorageOptions &options,
                                      TokenizationStoragePlan *out) {
  size_t block_count = 0;
  TokenizationStoragePlan plan;
  if (out == nullptr || blocks.empty() || !blocks.try_area(&block_count) ||
      options.context_count == 0 ||
      options.context_count >= std::numeric_limits<uint16_t>::max() ||
      options.workers == 0 || options.map_count == 0 ||
      (options.order_count != 1 && options.order_count != 2) ||
      (!options.exhaustive &&
       (options.order_count != 1 || options.map_count != 1)) ||
      (options.exhaustive && options.collect_fixed_populations) ||
      !blocks.ceil_div(kVarDctAcGroupBlockDimension)
           .try_area(&plan.ac_group_count) ||
      !blocks.ceil_div(kSimpleDcGroupBlockDimension)
           .try_area(&plan.dc_group_count)) {
    return Status::InvalidArgument("Token storage plan arguments are invalid");
  }
  using enum resource_budget_internal::VectorCapacityPolicy;
  const auto add_dc = [&](Extent2D extent, size_t count) {
    DcGroupTokenStoragePlan group;
    Status status = ComputeDcGroupTokenStoragePlan(
        extent, extent.width * extent.height, &group);
    if (!status.ok())
      return status;
    return plan.dc.Add(group.output, count) ? Status::Ok() : Overflow();
  };
  Status status = ForGroupClasses(blocks, kSimpleDcGroupBlockDimension, add_dc);
  if (!status.ok())
    return status;
  DcGroupTokenStoragePlan maximum_dc;
  const Extent2D dc_extent{
      std::min(blocks.width, kSimpleDcGroupBlockDimension),
      std::min(blocks.height, kSimpleDcGroupBlockDimension)};
  status = ComputeDcGroupTokenStoragePlan(
      dc_extent, dc_extent.width * dc_extent.height, &maximum_dc);
  if (!status.ok())
    return status;
  // DC tokenization is serial; all group outputs survive into entropy coding.
  if (!plan.dc.Add(maximum_dc.scratch) ||
      !plan.dc.AddVector<SimpleDcGroupTokenStreams>(plan.dc_group_count,
                                                    kFreshExact))
    return Overflow();

  size_t variants = 0, template_tasks = 0, context_tasks = 0;
  if (!Multiply(options.order_count, options.map_count, &variants) ||
      !Multiply(options.order_count, plan.ac_group_count, &template_tasks) ||
      !Multiply(variants, plan.ac_group_count, &context_tasks))
    return Overflow();
  const auto add_ac = [&](Extent2D extent, size_t count) {
    AcGroupTokenStoragePlan group;
    Status group_status = ComputeAcGroupTokenStoragePlan(
        extent, extent.width * extent.height, options.context_count,
        options.collect_fixed_populations, &group);
    if (!group_status.ok())
      return group_status;
    if (!options.exhaustive)
      return plan.ac.Add(group.direct_output, count) ? Status::Ok()
                                                     : Overflow();
    size_t templates = 0, contexts = 0;
    if (!Multiply(count, options.order_count, &templates) ||
        !Multiply(count, variants, &contexts) ||
        !plan.ac.Add(group.template_output, templates) ||
        !plan.ac.Add(group.context_output, contexts))
      return Overflow();
    return Status::Ok();
  };
  status = ForGroupClasses(blocks, kVarDctAcGroupBlockDimension, add_ac);
  if (!status.ok())
    return status;
  AcGroupTokenStoragePlan maximum_ac;
  const Extent2D ac_extent{
      std::min(blocks.width, kVarDctAcGroupBlockDimension),
      std::min(blocks.height, kVarDctAcGroupBlockDimension)};
  status = ComputeAcGroupTokenStoragePlan(
      ac_extent, ac_extent.width * ac_extent.height, options.context_count,
      options.collect_fixed_populations, &maximum_ac);
  if (!status.ok())
    return status;
  if (!options.exhaustive) {
    HostStorageBound natural_orders;
    status = ComputeAcNaturalOrderStorageBound(&natural_orders);
    if (!status.ok())
      return status;
    if (!plan.ac.Add(natural_orders) ||
        !plan.ac.Add(maximum_ac.direct_scratch,
                     std::min(options.workers, plan.ac_group_count)) ||
        !plan.ac.AddVector<SimpleAcGroupTokenData>(plan.ac_group_count,
                                                   kFreshExact) ||
        (options.collect_fixed_populations &&
         !plan.ac.AddVector<PreparedFixedAnsCluster>(options.context_count,
                                                     kFreshExact)))
      return Overflow();
  } else {
    // Template construction and context materialization are joined phases.
    // Include every retained output in both bounds, but not simultaneous worker
    // scratch for the two nonoverlapping phases.
    HostStorageBound template_scratch, context_scratch;
    if (!template_scratch.Add(maximum_ac.template_scratch,
                              std::min(options.workers, template_tasks)) ||
        !context_scratch.Add(maximum_ac.context_scratch,
                             std::min(options.workers, context_tasks)) ||
        !plan.ac.Add({std::max(template_scratch.retained_bytes,
                               context_scratch.retained_bytes),
                      std::max(template_scratch.peak_bytes,
                               context_scratch.peak_bytes)}) ||
        !plan.ac.AddVector<SimpleAcGroupTokenTemplate>(
            plan.ac_group_count, kFreshExact, options.order_count) ||
        !plan.ac.AddVector<Storage<uint16_t>>(plan.ac_group_count, kFreshExact,
                                              variants))
      return Overflow();
  }
  *out = plan;
  return Status::Ok();
}
} // namespace gjxl::codestream_internal
