// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codestream/encoder.h"
#include "core/geometry.h"
#include "core/host_storage_bound.h"

namespace gjxl::codestream_internal {

using resource_budget_internal::HostStorageBound;

struct CoefficientOrderStoragePlan {
  size_t ac_group_count = 0;
  size_t maximum_participants = 0;
  size_t maximum_order_elements = 0; // All families and three channels.
  size_t maximum_tokens = 0;
  HostStorageBound orders;
  HostStorageBound tokens;
  // COMPLETE Compute...ForEncoder + Tokenize... envelope, including both
  // outputs. Includes capacity in cleared, natural-order family vectors.
  HostStorageBound working;
  bool operator==(const CoefficientOrderStoragePlan &) const = default;
};

/// Geometry-only bound for the existing full/sampled order selection and its
/// tokenization. `workers` bounds runtime participants (1..8), including the
/// caller. A sampled plan also covers the full fallback for mixed strategies.
/// The maximum-compression caller must pass its normalized kFull behavior.
[[nodiscard]] Status ComputeCoefficientOrderStoragePlan(
    Extent2D blocks, VarDctCoefficientOrderBehavior behavior, size_t workers,
    CoefficientOrderStoragePlan *out);

struct BlockContextMapStoragePlan {
  size_t maximum_maps = 0;
  size_t maximum_thresholds = 0;
  size_t maximum_map_entries = 0;
  size_t maximum_block_contexts = 0;
  size_t maximum_ac_contexts = 0;
  // Fresh copy of any generated map; excludes inline SimpleBlockContextMap.
  HostStorageBound map;
  // One returned map, or the returned exhaustive vector and all nested maps.
  HostStorageBound output;
  // COMPLETE selection envelope, including output and adaptive-map scratch.
  HostStorageBound working;
  bool operator==(const BlockContextMapStoragePlan &) const = default;
};

[[nodiscard]] Status
ComputeBlockContextMapStoragePlan(Extent2D blocks, bool exhaustive,
                                  BlockContextMapStoragePlan *out);

// These bounds cover initially empty internal destinations and valid borrowed
// frames. Previous outputs, frame backing, public compatibility adapters,
// encoder candidate copies, entropy, section writers and thread stacks/control
// objects are excluded. Successful planning allocates nothing and is O(1) in
// image size; failure preserves *out. They are components, not whole-workflow
// admission. Plans are defined beside the actual policies/private types.

} // namespace gjxl::codestream_internal
