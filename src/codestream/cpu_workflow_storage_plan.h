// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/adaptive_quantization_storage_plan.h"
#include "codestream/serializer_storage_plan.h"
#include "codestream/workflow.h"

namespace gjxl::codestream_internal {

struct CpuWorkflowStorageOptions {
  VarDctEncodingOptions encoding;
  bool collect_timing = false;
  bool collect_profile = false;
};

struct CpuWorkflowStoragePlan {
  Extent2D coding_extent;
  size_t maximum_attempts = 0;
  size_t score_count = 0;
  HostStorageBound frontend;
  frontend_storage_internal::CpuAqStoragePlan aq;
  SerializerStoragePlan serializer;
  HostStorageBound search_control;
  HostStorageBound retained_best;
  HostStorageBound output;
  HostStorageBound working;
  bool operator==(const CpuWorkflowStoragePlan &) const = default;
};

/// Complete native CPU workflow bound from borrowed RGB to outer publication,
/// including compatibility destinations and forced-CPU size search. Fixed
/// geometry, initially empty prepared workflow, existing effort/entropy policy.
/// Automatic requests require proof that EVERY attempt stays on CPU; this does
/// not select a backend or bound mixed CPU/Metal automatic-exact searches.
///
/// One reservation spans all attempts. Caller input/old/published output, input
/// conversion adapters, retained batch output, immutable backend/code, driver
/// internals, stacks and small controls/allocator overhead are separate. No
/// public admission, RSS estimate or request validation is implied. A checked
/// conservative sum, O(1), allocation-free on success and atomic on failure.
[[nodiscard]] Status
ComputeCpuWorkflowStoragePlan(Extent2D source,
                              const CpuWorkflowStorageOptions &options,
                              CpuWorkflowStoragePlan *out);

} // namespace gjxl::codestream_internal
