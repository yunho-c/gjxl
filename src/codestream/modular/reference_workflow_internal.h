// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "gpu/ops/modular_resident_histogram_fitting.h"
#include "codestream/modular/reference_workflow.h"
#include <chrono>

namespace gjxl::modular_internal::reference_workflow_internal {
template <typename Function>
Status Stage(const resource_budget_internal::ResourceBudget &budget, size_t bytes,
             ReferenceWorkflowStats *stats, Function &&function) {
  using namespace resource_budget_internal;
  ResourceReservation reservation;
  if (auto s = budget.TryReserve(std::max(size_t{1}, bytes), &reservation); !s.ok())
    return s;
  stats->maximum_stage_bytes = std::max(stats->maximum_stage_bytes, bytes);
  auto context = CurrentResourceContext();
  context.reservation = &reservation;
  context.resource_class = ResourceClass::kPreparation;
  ResourceContextScope scope(context);
  return function();
}
class Seconds {
  using Clock = std::chrono::steady_clock;

public:
  explicit Seconds(double &out) : out_(out), start_(Clock::now()) {}
  ~Seconds() { out_ += std::chrono::duration<double>(Clock::now() - start_).count(); }
  Seconds(const Seconds &) = delete;
  Seconds &operator=(const Seconds &) = delete;

private:
  double &out_;
  Clock::time_point start_;
};
using GpuTokenOwners = resource_budget_internal::ManagedVector<
    std::unique_ptr<modular_gpu_internal::PreparedGpuModularTokens>>;
struct HistogramFitRequest {
  std::span<const HistogramTrainingImage> images;
  ReferenceModularStream *record = nullptr;
  size_t maximum_leaves = 0;
  const HistogramRowSource *row_source = nullptr;
};
// Returns handled=false only before preparation/partitioning begins, allowing
// the original CPU scheduler to use tighter stage admission. Once work starts,
// allocation/launch/device errors propagate and all workers/submissions drain.
[[nodiscard]] Status TryLearnHistogramBatch(
    std::span<const HistogramFitRequest> requests,
    HistogramSamplingOptions sampling, size_t participants,
    const ReferenceWorkflowOptions &options,
    const resource_budget_internal::ResourceBudget &budget,
    ReferenceWorkflowStats *stats, bool *handled, bool measure_learning = true);
// Shared complete resident operation for cohorts and the continuous scheduler.
// admission_limited is true only when reservation fails before GPU work starts.
[[nodiscard]] Status FitResidentHistogramBatch(
    std::span<const histogram_gpu_internal::ResidentHistogramRequest> requests,
    const ReferenceWorkflowOptions &options,
    const resource_budget_internal::ResourceBudget &budget,
    ReferenceWorkflowStats *stats,
    histogram_gpu_internal::ResidentHistogramResult *out,
    bool *admission_limited = nullptr);
// Select independent unpublished records, retaining each record's original
// candidate order. Input views stay alive until all jobs have joined. Records
// may be partially updated on failure; the owning frame is published later.
[[nodiscard]] Status SelectReferenceModularScopes(
    std::span<ReferenceModularStream> records, ReferenceSelectionOptions options,
    size_t participants, const resource_budget_internal::ResourceBudget &budget,
    ReferenceWorkflowStats *stats);
// Records and stream views are unpublished and remain at stable addresses.
// CPU learning policy is shared by both routes. Hybrid token owners outlive
// the completed borrows installed into streams; feature owners never escape.
[[nodiscard]] Status
LearnReferenceModularScopes(std::span<ReferenceModularStream> records,
                            std::span<PreparedModularStreamView> streams, uint64_t population,
                            const ReferenceWorkflowOptions &options,
                            const resource_budget_internal::ResourceBudget &budget,
                            ReferenceWorkflowStats *stats, GpuTokenOwners *tokens);
} // namespace gjxl::modular_internal::reference_workflow_internal
