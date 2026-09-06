// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

#include "core/geometry.h"
#include "core/host_storage_bound.h"
#include "core/status.h"

namespace gjxl::codestream_internal {

using resource_budget_internal::HostStorageBound;

struct AcGroupTokenCounts {
  size_t block_count = 0;
  size_t coefficient_count = 0; // Per channel, including LLF coefficients.
  size_t block_context_keys = 0;
  size_t token_capacity = 0; // Existing reservation, not emitted token count.
  bool operator==(const AcGroupTokenCounts &) const = default;
};

struct DcGroupTokenCounts {
  size_t block_count = 0;
  size_t color_tile_count = 0;
  size_t dc_tokens = 0;
  size_t metadata_tokens = 0;
  bool operator==(const DcGroupTokenCounts &) const = default;
};

/// The runtime passes its validated anchor count. Before placement, pass the
/// block count as the anchor upper bound. These functions validate geometry and
/// counts, not strategy coverage, coefficient values or metadata.
[[nodiscard]] Status ComputeAcGroupTokenCounts(Extent2D blocks, size_t anchors,
                                               AcGroupTokenCounts *out);
[[nodiscard]] Status ComputeDcGroupTokenCounts(Extent2D blocks, size_t anchors,
                                               DcGroupTokenCounts *out);

struct AcGroupTokenStoragePlan {
  AcGroupTokenCounts counts;
  HostStorageBound direct_output;
  // Worker scratch may be reused only within this plan's geometry, anchor and
  // context-count upper bounds (not merely its current group's logical sizes).
  HostStorageBound direct_scratch;
  HostStorageBound template_output;
  HostStorageBound template_scratch;
  HostStorageBound context_output;
  HostStorageBound context_scratch;
  bool operator==(const AcGroupTokenStoragePlan &) const = default;
};

struct DcGroupTokenStoragePlan {
  DcGroupTokenCounts counts;
  HostStorageBound output;
  HostStorageBound scratch;
  bool operator==(const DcGroupTokenStoragePlan &) const = default;
};

/// Bounds owned backing inside the group tokenizers. Borrowed frame/custom
/// orders/maps and old output backings are excluded. Direct tokenization's
/// shared natural-order table is returned separately by the function below.
/// Success does not allocate; failure preserves output (Status may allocate).
[[nodiscard]] Status ComputeAcGroupTokenStoragePlan(
    Extent2D blocks, size_t anchors, size_t context_count,
    bool collect_fixed_populations, AcGroupTokenStoragePlan *out);
[[nodiscard]] Status
ComputeDcGroupTokenStoragePlan(Extent2D blocks, size_t anchors,
                               DcGroupTokenStoragePlan *out);
[[nodiscard]] Status ComputeAcNaturalOrderStorageBound(HostStorageBound *out);

struct TokenizationStorageOptions {
  bool exhaustive = false;
  bool collect_fixed_populations = true;
  size_t context_count = 0;
  size_t order_count = 1; // Exhaustive: natural plus optional custom order.
  size_t map_count = 1;   // Exhaustive: independently retained context maps.
  size_t workers = 1;     // Upper bound on simultaneous tokenizer participants.
};

struct TokenizationStoragePlan {
  size_t ac_group_count = 0;
  size_t dc_group_count = 0;
  HostStorageBound dc;
  HostStorageBound ac;
  bool operator==(const TokenizationStoragePlan &) const = default;
};

/// Geometry-only upper bound for all retained DC/AC token owners, direct fixed
/// population reduction, and tokenizer worker scratch. Includes outer token
/// containers and maximum simultaneous order/map variants; does NOT include
/// frame backing, coefficient-order search/tokens, block-map/candidate objects,
/// borrowed stream tables, dispatch/status/profile arrays, entropy or writers.
/// This component is not whole-serializer/workflow admission. Four rectangular
/// group classes are summed, without iterating over the image's group count.
[[nodiscard]] Status
ComputeTokenizationStoragePlan(Extent2D blocks,
                               const TokenizationStorageOptions &options,
                               TokenizationStoragePlan *out);

} // namespace gjxl::codestream_internal
