// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codestream/entropy_storage_plan.h"
#include "codestream/representation_storage_plan.h"
#include "codestream/token_storage_plan.h"

namespace gjxl::codestream_internal {

inline constexpr size_t kSerializerMaximumSectionWorkers = 8;

struct SerializerStorageOptions {
  VarDctCodestreamOptions coding;
  // Match the runtime EncodeScope. Zero covers hardware-auto, INCLUDING its
  // current nested DC-measurement workers. This plan does not install a scope.
  size_t cpu_thread_count = 0;
  bool collect_profile = false;
};

struct SerializerStoragePlan {
  size_t ac_group_count = 0;
  size_t dc_group_count = 0;
  size_t maximum_order_variants = 0;
  size_t maximum_ac_candidates = 0;
  size_t maximum_ac_tokens = 0;
  size_t maximum_dc_tokens = 0;
  size_t maximum_output_bytes = 0;
  // Separately exposed for retained-result admission. The COMPLETE working
  // envelope already includes this output; do not add it twice.
  HostStorageBound output;
  HostStorageBound working;
  bool operator==(const SerializerStoragePlan &) const = default;
};

/// Complete owned-backing bound for EncodeVarDctCodestreamToBuffer on a valid
/// borrowed simple frame with this pixel extent and these normalized options.
/// Includes validation, representation/token/model/search/measurement storage,
/// worker arrays, headers, section writers, assembly and fresh published bytes.
/// Excludes the borrowed frame, prior output, caller-owned profile, allocator
/// headers, stacks and small runtime control objects. Successful planning is
/// allocation-free and O(1) in image size; failure preserves output.
///
/// This is a conservative capacity bound, not expected usage or RSS. Named
/// phase envelopes may overlap conservatively. It does NOT admit a workflow,
/// account frontend/device/retry lifetimes, or enforce aggregate CPU limits.
[[nodiscard]] Status
ComputeSerializerStoragePlan(Extent2D frame_extent,
                             const SerializerStorageOptions &options,
                             SerializerStoragePlan *out);

// Composition helpers: kept beside private types/format constants in the
// implementation they bound. Inputs are checked component plans/counts from
// ComputeSerializerStoragePlan, not arbitrary externally supplied models.
struct SerializerHeaderStoragePlan {
  size_t frame_prefix_bits = 0;
  size_t dc_global_bits = 0;
  size_t ac_global_bits = 0;
  EntropyModelStoragePlan dc_model, ac_model, order_model;
  HostStorageBound frame_scratch, dc_global_scratch, ac_global_scratch;
};

[[nodiscard]] Status ComputeSerializerHeaderStoragePlan(
    size_t ac_groups, size_t dc_groups, const BlockContextMapStoragePlan &maps,
    size_t order_tokens, SerializerHeaderStoragePlan *out);

[[nodiscard]] Status
ComputeSerializerControlStorageBound(const SerializerStoragePlan &counts,
                                     const SerializerStorageOptions &options,
                                     size_t maximum_maps, bool has_orders,
                                     HostStorageBound *out);

} // namespace gjxl::codestream_internal
