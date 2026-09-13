// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>

#include "codestream/cpu_workflow_storage_plan.h"

namespace gjxl::codestream_internal {

struct MetalCompatibilityWorkflowStoragePlan {
  Extent2D coding_extent;
  size_t maximum_attempts = 0;
  size_t score_count = 0;
  HostStorageBound frontend;
  HostStorageBound evaluator;
  HostStorageBound ac_search;
  HostStorageBound search_control;
  HostStorageBound retained_best;
  SerializerStoragePlan serializer;
  HostStorageBound output;
  HostStorageBound working;
  // Same order as ResidentWorkflowStoragePlan: input, persistent, staging,
  // Butteraugli, completed frame (unused on compatibility routes). This is a
  // pool-capacity bound, not additional active backing.
  std::array<size_t, 5> idle_pool_capacity{};
  bool
  operator==(const MetalCompatibilityWorkflowStoragePlan &) const = default;
};

/// Complete forced-Metal exact-coefficient, resident maximum-error and
/// maximum-throughput workflow envelopes. Uses the same timing/CPU-profile
/// options as the CPU planner; GPU profiles are not supported by these public
/// workflow modes. Ordinary resident Butteraugli uses its separate phase plan.
///
/// Initially empty workflow, fixed geometry and mode through all attempts, one
/// reservation spanning the search. Includes host/device evaluator backing,
/// owned frame output, CPU tail, and at most one retained best result. The
/// frontend deliberately uses the conservative shared host-preparation bound,
/// including compatibility destinations even when a particular mode omits them.
/// Caller input, old/published output, other jobs' idle caches, input adapters,
/// retained batch results, immutable setup, driver internals and small controls
/// remain separate. O(1), allocation-free on success, atomic on failure. Does
/// not select a backend, validate the entire request or implement public
/// admission.
[[nodiscard]] Status ComputeMetalCompatibilityWorkflowStoragePlan(
    Extent2D source, const CpuWorkflowStorageOptions &options,
    MetalCompatibilityWorkflowStoragePlan *out);

struct AutomaticExactSearchStoragePlan {
  CpuWorkflowStoragePlan cpu;
  MetalCompatibilityWorkflowStoragePlan metal;
  HostStorageBound output;
  HostStorageBound working;
  bool operator==(const AutomaticExactSearchStoragePlan &) const = default;
};

/// Automatic exact-coefficient byte/bpp search can select CPU or Metal on each
/// attempt. Sum the two complete bounds, not their maximum: prepared/native
/// reference and GPU evaluator/cache backing can survive an attempt on the
/// other backend. Shared owners are conservatively budgeted twice, but actual
/// accounting still charges each backing once. Retained best output is bounded
/// independently of attempt count. This neither changes nor predicts selection.
[[nodiscard]] Status
ComputeAutomaticExactSearchStoragePlan(Extent2D source,
                                       const CpuWorkflowStorageOptions &options,
                                       AutomaticExactSearchStoragePlan *out);

} // namespace gjxl::codestream_internal
