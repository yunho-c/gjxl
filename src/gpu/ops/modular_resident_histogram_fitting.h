// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "gpu/ops/modular_histogram_fitting.h"
#include "gpu/ops/modular_resident_histogram_layout.h"

namespace gjxl::histogram_gpu_internal {
inline constexpr size_t kMaximumResidentHistogramFits = 64;
inline constexpr size_t kMaximumResidentHistogramRowsPerFit = 131072;
enum class ResidentHistogramExecutor : uint8_t { kDepthFirst, kFrontier, kTiledFrontier };
struct ResidentHistogramRequest {
  HistogramFittingSource source;
  modular_internal::HistogramTreeLearningOptions learning;
};
struct ResidentHistogramInput {
  std::span<const ResidentHistogramRequest> requests;
  // Diagnostic only; normal publication copies trees, not row permutations.
  bool capture_row_order = false;
  ResidentHistogramExecutor executor = ResidentHistogramExecutor::kDepthFirst;
  // Frontier scheduling/storage controls, independent of learned policy.
  size_t maximum_active_nodes = 128;
  size_t rounds_per_submission = 16;
  size_t maximum_scratch_bytes = 64 * 1024 * 1024;
};
struct ResidentHistogramPlan {
  size_t fits = 0, rows = 0, work_bytes = 0;
  HistogramFittingPreparationPlan prepared;
  std::array<gjxl_histogram_resident::Fit, kMaximumResidentHistogramFits> layout{};
  gjxl_histogram_resident::FrontierParams frontier{};
  gjxl_histogram_resident::TiledParams tiled{};
  size_t scratch_bytes = 0, maximum_nodes = 0;
  resource_budget_internal::HostStorageBound publication;
  resource_budget_internal::HostStorageBound output, working;
};
struct ResidentHistogramResult {
  resource_budget_internal::ManagedVector<modular_internal::DecisionTree> trees;
  resource_budget_internal::ManagedVector<uint32_t> row_order;
  resource_budget_internal::ManagedVector<size_t> row_offsets;
  size_t visited_nodes = 0, device_bytes = 0, planned_peak_bytes = 0;
  uint64_t gpu_nanoseconds = 0;
  size_t submissions = 0, frontier_rounds = 0, maximum_ready_nodes = 0,
         maximum_active_nodes = 0, maximum_same_fit_nodes = 0;
  size_t maximum_histogram_bytes = 0, maximum_deferred_nodes = 0;
};
// Complete fitting of identical prepared host samples, including validation,
// immutable upload, GPU setup/parent summaries/selection/partition/traversal,
// and atomic CPU tree publication. No CPU node decisions or CPU subtrees.
// V1 rejects empty samples, CPU execution_stats, multiplier ranges, >256 bins
// and >131072 rows per fit before submission. Capacity/depth overflow fails the
// whole operation; no truncation.
// Old output and caller-owned samples are excluded from the admission bound.
[[nodiscard]] Status ComputeResidentHistogramPlan(
    const ResidentHistogramInput &input, ResidentHistogramPlan *out,
    const HistogramFittingDeviceLimits *limits = nullptr);
} // namespace gjxl::histogram_gpu_internal
