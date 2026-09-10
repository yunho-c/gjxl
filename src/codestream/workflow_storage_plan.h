// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>

#include "codestream/cpu_workflow_storage_plan.h"

namespace gjxl::codestream_internal {

enum class WorkflowStorageRoute { kCpu, kMetal, kAutomaticExactSearch };
enum class WorkflowStorageAdapter { kBorrowedLinearRgb, kPackedSrgbC };

struct WorkflowStorageOptions {
  VarDctEncodingOptions encoding;
  WorkflowStorageRoute route = WorkflowStorageRoute::kCpu;
  WorkflowStorageAdapter adapter = WorkflowStorageAdapter::kBorrowedLinearRgb;
  bool collect_timing = false;
  bool collect_profile = false;
  bool collect_gpu_profile = false;
};

struct WorkflowStoragePlan {
  Extent2D coding_extent;
  WorkflowStorageAdapter adapter = WorkflowStorageAdapter::kBorrowedLinearRgb;
  size_t maximum_attempts = 0;
  size_t maximum_codestream_bytes = 0;
  bool collect_timing = false;
  HostStorageBound backend_working;
  HostStorageBound input;
  HostStorageBound publication;
  HostStorageBound output;
  HostStorageBound working;
  // One production backend's four idle pools: resident input, AQ persistent,
  // AQ staging, Butteraugli. They are already included in backend_working.
  std::array<size_t, 4> idle_pool_capacity{};
  bool operator==(const WorkflowStoragePlan &) const = default;
};

/// Compose the existing policy recipe and outer adapter. Options and caller
/// buffers must first pass normal entry-point validation. route records actual
/// selection: kCpu requires proof every attempt stays on CPU; kMetal requires
/// a selected Metal backend; kAutomaticExactSearch covers both possibilities.
/// This function never initializes a backend or changes its selection policy.
/// Packed C conversion includes three source F32 planes and the final
/// byte-array copy while the internal codestream remains owned; it has no
/// diagnostics. All other exclusions are inherited from the component plans.
/// Successful planning is allocation-free and O(1); failure leaves output
/// unchanged.
[[nodiscard]] Status
ComputeWorkflowStoragePlan(Extent2D source,
                           const WorkflowStorageOptions &options,
                           WorkflowStoragePlan *out);

struct BatchWorkflowStoragePlan {
  size_t request_count = 0;
  size_t encodable_count = 0;
  size_t in_flight = 0;
  size_t work_slot_bytes = 0;
  size_t minimum_required_bytes = 0;
  // If keeping idle pools would make even one work slot inadmissible, a caller
  // may use this plan only if each completed worker trims before reusing its
  // slot. In-flight jobs' idle buffers remain covered by their work envelopes.
  bool trim_after_each_image = false;
  HostStorageBound result_metadata;
  HostStorageBound retained_results;
  HostStorageBound idle_pools;
  HostStorageBound working;
  bool operator==(const BatchWorkflowStoragePlan &) const = default;
};

/// Streaming preflight: no plan vector or other request-sized allocation is
/// needed before admission. Add each request once; nullptr represents a request
/// already rejected by ordinary validation (its result slot is still counted).
/// Successful plans must be unmodified ComputeWorkflowStoragePlan results for
/// borrowed input with attempt timing, matching the existing batch driver.
/// All requests share one production backend and one admission domain.
class BatchWorkflowStorageAccumulator {
public:
  [[nodiscard]] Status AddRequest(const WorkflowStoragePlan *plan);
  // Zero managed_limit is unlimited. Select no more than max_in_flight work
  // slots, preserving a full largest-image slot and all retained output credit.
  // A finite limit may reduce concurrency, never encoding effort or candidates.
  // The executor must enforce in_flight and trim_after_each_image; this helper
  // does not change the current batch driver's behavior by itself.
  [[nodiscard]] Status Finish(size_t max_in_flight, size_t managed_limit,
                              BatchWorkflowStoragePlan *out) const;

private:
  size_t requests_ = 0;
  size_t encodable_ = 0;
  size_t maximum_working_ = 0;
  HostStorageBound retained_;
  std::array<size_t, 4> idle_{};
};

} // namespace gjxl::codestream_internal
