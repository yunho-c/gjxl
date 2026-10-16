// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/histogram_tree_fitter.h"
#include "codec/modular/histogram_tree_learning.h"
#include "core/host_storage_bound.h"
#include "gpu/backend.h"
#include "gpu/ops/modular_histogram_fitting_layout.h"
#include <array>
#include <span>

namespace gjxl::histogram_gpu_internal {
inline constexpr size_t kMaximumHistogramFittingFits = 256;
inline constexpr size_t kMaximumHistogramFittingRows = 8 * 1024 * 1024;
inline constexpr size_t kHistogramFittingAlignment = 256;
inline constexpr size_t kMaximumHistogramFittingShards = 32;
using HistogramPropertyResult = gjxl_histogram_fitting::PropertyResult;

struct HistogramFittingShape {
  uint32_t source_id = 0;
  size_t rows = 0, population = 0, predictors = 0, properties = 0,
         static_properties = 0;
  std::array<modular_internal::Predictor, 14> predictor_ids{};
  std::array<uint8_t, 16> property_ids{};
  std::array<size_t, 16> bins{};
  bool operator==(const HistogramFittingShape &) const = default;
};
struct HistogramFittingSource {
  uint32_t source_id = 0;
  const modular_internal::HistogramTrainingSamples *samples = nullptr;
};
struct HistogramFittingDeviceLimits {
  size_t maximum_fits = 0, maximum_rows = 0, maximum_nodes = 0;
  size_t maximum_buffer_bytes = 0, maximum_scratch_bytes = 0;
};
struct HistogramFittingColumnLayout {
  size_t weights_offset = 0;
  std::array<size_t, 14> residual_offsets{};
  std::array<size_t, 16> property_offsets{};
  size_t bytes = 0;
  bool operator==(const HistogramFittingColumnLayout &) const = default;
};
struct HistogramFittingPreparationPlan {
  size_t fits = 0, rows = 0, descriptors_bytes = 0, columns_offset = 0;
  size_t storage_bytes = 0, host_shapes_bytes = 0;
  resource_budget_internal::HostStorageBound staging, row_order;
  // Device columns + immutable host shapes; row_order is caller-owned and
  // excluded here. Preparation adds one fresh host upload arena.
  resource_budget_internal::HostStorageBound retained, preparation;
  bool operator==(const HistogramFittingPreparationPlan &) const = default;
};
struct HistogramFittingNodeShape {
  size_t fit = 0, rows = 0, symbols = 0;
  uint32_t node_id = 0, inherited_constants = 0;
  modular_internal::Predictor parent_predictor =
      modular_internal::Predictor::kZero;
  float change_predictor_penalty = 0;
  bool operator==(const HistogramFittingNodeShape &) const = default;
};
struct HistogramFittingNodeInput {
  HistogramFittingNodeShape shape;
  std::span<const uint32_t> rows;
  // Predictor-major, exactly predictors*symbols nonnegative int32 counts.
  std::span<const int32_t> parent_counts;
  std::span<const uint32_t> parent_extra_bits;
};
struct HistogramFittingExecutionOptions {
  // Storage choices, not learned-policy or automatic GPU crossover controls.
  size_t histogram_shards = 4, maximum_active_properties = 64;
  size_t maximum_scratch_bytes = 64 * 1024 * 1024;
};
enum class HistogramFittingProfileMode : uint8_t { kDisabled, kWall, kStage };
// Caller-owned diagnostics. Wall phases partition the operation interior up to
// transient release, excluding final inline publication. Wait overlaps device
// execution and concurrent CPU fits. Stage mode adds encoder
// boundaries/counters and is attribution only.
inline constexpr std::array<const char *, 8> kHistogramFittingWallPhases = {
    "validate_plan", "allocate",        "pack",   "record_submit", "wait",
    "readback",      "resolve_profile", "release"};
inline constexpr std::array<const char *, 5> kHistogramFittingGpuStages = {
    "validate", "collect", "prefix", "costs", "select"};
struct HistogramFittingBatchProfile {
  size_t nodes = 0, rows = 0, tasks = 0, active_properties = 0, waves = 0;
  size_t maximum_bins = 0, maximum_symbols = 0, scratch_bytes = 0;
  std::array<uint64_t, 8> wall_nanoseconds{};
  std::array<uint64_t, 5> gpu_nanoseconds{}, dispatches{};
  uint64_t command_buffer_gpu_nanoseconds = 0;
  bool operator==(const HistogramFittingBatchProfile &) const = default;
};
struct HistogramFittingProfile {
  HistogramFittingProfileMode mode = HistogramFittingProfileMode::kDisabled;
  std::span<HistogramFittingBatchProfile> batches;
  size_t written = 0, dropped = 0;
};
struct HistogramFittingScratchLayout {
  size_t shard_histograms_offset = 0, shard_extra_offset = 0,
         shard_occupancy_offset = 0;
  size_t histograms_offset = 0, extra_offset = 0, occupancy_offset = 0;
  size_t candidate_costs_offset = 0, side_winners_offset = 0, bytes = 0;
  bool operator==(const HistogramFittingScratchLayout &) const = default;
};
struct HistogramFittingEvaluationPlan {
  size_t prepared_bytes = 0, row_order_bytes = 0;
  size_t nodes = 0, rows = 0, properties = 0, tasks = 0;
  size_t maximum_bins = 0, maximum_predictors = 0, maximum_symbols = 0;
  size_t descriptors_bytes = 0, indices_offset = 0, parent_counts_offset = 0;
  size_t parent_extra_offset = 0, predictor_penalties_offset = 0,
         tasks_offset = 0, status_offset = 0, dispatch_bytes = 0;
  size_t histogram_shards = 0, active_properties = 0, scratch_bytes = 0,
         output_bytes = 0;
  HistogramFittingScratchLayout scratch;
  resource_budget_internal::HostStorageBound staging, retained, evaluation;
  bool operator==(const HistogramFittingEvaluationPlan &) const = default;
};

// No allocation, device access or submission. All outputs are failure-atomic.
// Shape planning reads no row contents; source validation scans native rows.
// Empty fits are omitted by the caller, not submitted as learning jobs.
[[nodiscard]] Status
ResolveHistogramFittingSource(const HistogramFittingSource &source,
                              HistogramFittingShape *out);
[[nodiscard]] Status
ComputeHistogramFittingColumnLayout(const HistogramFittingShape &shape,
                                    HistogramFittingColumnLayout *out);
[[nodiscard]] Status ComputeHistogramFittingPreparationPlan(
    std::span<const HistogramFittingShape> shapes,
    HistogramFittingPreparationPlan *out,
    const HistogramFittingDeviceLimits *limits = nullptr);
[[nodiscard]] Status ValidateHistogramFittingSources(
    std::span<const HistogramFittingSource> sources,
    HistogramFittingPreparationPlan *out,
    const HistogramFittingDeviceLimits *limits = nullptr);
// Native bias expressions include a double literal before conversion back to
// float. The host packer computes these once per node; GPU kernels consume the
// exact resulting float bits instead of re-evaluating the biases in float.
[[nodiscard]] Status ComputeHistogramFittingPredictorPenalties(
    std::span<const modular_internal::Predictor> predictors,
    modular_internal::Predictor parent, float change_penalty,
    std::span<float> out);
// One ready node per fit, in increasing fit-index order. Inherited constant
// bits index the prepared property order; their semantic truth is established
// by the CPU parent. Summary and row contents are checked separately below.
[[nodiscard]] Status ComputeHistogramFittingEvaluationPlan(
    std::span<const HistogramFittingShape> shapes,
    std::span<const HistogramFittingNodeShape> nodes,
    HistogramFittingExecutionOptions options,
    HistogramFittingEvaluationPlan *out,
    const HistogramFittingDeviceLimits *limits = nullptr);
// Compose unmodified plans from the same cohort with the CPU row-order vector.
// Excludes borrowed CPU samples/trees/features, old outputs, backend/driver
// memory and inline controls. Those remain separately charged by the workflow.
[[nodiscard]] Status ComputeHistogramFittingWorkingSetBound(
    const HistogramFittingPreparationPlan &preparation,
    const HistogramFittingEvaluationPlan &evaluation,
    resource_budget_internal::HostStorageBound *out);

class PreparedGpuHistogramRows {
public:
  virtual ~PreparedGpuHistogramRows() = default;
  PreparedGpuHistogramRows(const PreparedGpuHistogramRows &) = delete;
  PreparedGpuHistogramRows &
  operator=(const PreparedGpuHistogramRows &) = delete;
  [[nodiscard]] virtual BackendId backend_id() const noexcept = 0;
  // Completed immutable metadata; these views allocate/copy nothing.
  [[nodiscard]] virtual std::span<const HistogramFittingShape>
  shapes() const noexcept = 0;

protected:
  PreparedGpuHistogramRows() = default;
};
struct HistogramFittingInput {
  const PreparedGpuHistogramRows *prepared = nullptr;
  std::span<const HistogramFittingNodeInput> nodes;
  HistogramFittingExecutionOptions options;
  HistogramFittingProfileMode profile_mode =
      HistogramFittingProfileMode::kDisabled;
  // Published together with results, only on success. Never retained.
  HistogramFittingBatchProfile *profile = nullptr;
};
// Checks ownership metadata, shapes and parent summaries without reading row
// indices or native GPU buffers. The backend must additionally verify its
// native owner type and allocation bounds. Device validation checks indices,
// bounded weighted populations and agreement with the supplied parent summary
// before histogram/scoring results may be published. Learning membership is a
// permutation subset; the evaluator also permits bounded weighted multisets.
[[nodiscard]] Status ValidateHistogramFittingInput(
    const HistogramFittingInput &input, BackendId backend,
    HistogramFittingEvaluationPlan *out,
    const HistogramFittingDeviceLimits *limits = nullptr);
[[nodiscard]] Status CheckHistogramFittingDeviceStatus(uint32_t status);

// Bridge the resumable native fitter and fixed-width operation results. Views
// borrow the ready node until Apply; all helpers allocate nothing. The CPU
// evaluator uses the same arithmetic as ordinary native fitting, exporting
// exact costs and category winners for backend differential checks.
[[nodiscard]] HistogramFittingNodeInput
HistogramFittingNodeView(size_t fit,
                         const modular_internal::HistogramNodeEvaluation &node);
[[nodiscard]] Status
EvaluateHistogramNodeOnCpu(modular_internal::HistogramTreeFitter &fitter,
                           uint32_t source_id,
                           std::span<HistogramPropertyResult> out);
[[nodiscard]] Status
ApplyHistogramNodeEvaluation(modular_internal::HistogramTreeFitter &fitter,
                             uint32_t source_id,
                             std::span<const HistogramPropertyResult> results);

class PreparedGpuHistogramEvaluation {
public:
  virtual ~PreparedGpuHistogramEvaluation() = default;
  PreparedGpuHistogramEvaluation(const PreparedGpuHistogramEvaluation &) =
      delete;
  PreparedGpuHistogramEvaluation &
  operator=(const PreparedGpuHistogramEvaluation &) = delete;
  [[nodiscard]] virtual BackendId backend_id() const noexcept = 0;
  // Complete host-readable results: request-major then every property in its
  // original order, including constants. No copy/map/second storage charge.
  [[nodiscard]] virtual std::span<const HistogramPropertyResult>
  results() const noexcept = 0;

protected:
  PreparedGpuHistogramEvaluation() = default;
};
// Optional private capability, implemented by Metal.
// Preparation snapshots native rows; later CPU partitions must track original
// indices separately. Exact native arithmetic and category winners are
// required; tree preferences, unquantization, partitioning and publication stay
// on CPU. Both calls consume ambient admission, complete synchronously,
// validate all device status, and publish only on success. Drain pending work
// on failure. No CPU retry after execution starts; no mutation of a borrowed
// row list while it is in use. The backend outlives its prepared/result owners.
class GpuModularHistogramFitting {
public:
  virtual ~GpuModularHistogramFitting() = default;
  [[nodiscard]] virtual HistogramFittingDeviceLimits
  histogram_fitting_limits() const noexcept = 0;
  // Additional transient host backing for the optional diagnostic path.
  // Opaque Metal counter/driver allocations retain the driver exclusion.
  [[nodiscard]] virtual Status HistogramFittingProfileStorageBound(
      const HistogramFittingEvaluationPlan &plan,
      HistogramFittingProfileMode mode,
      resource_budget_internal::HostStorageBound *out) const = 0;
  [[nodiscard]] virtual Status
  PrepareHistogramRows(std::span<const HistogramFittingSource> sources,
                       std::unique_ptr<PreparedGpuHistogramRows> *out) = 0;
  [[nodiscard]] virtual Status EvaluateHistogramNodes(
      const HistogramFittingInput &input,
      std::unique_ptr<PreparedGpuHistogramEvaluation> *out) = 0;
};
[[nodiscard]] inline GpuModularHistogramFitting *
QueryGpuModularHistogramFitting(GpuBackend &backend) noexcept {
  return dynamic_cast<GpuModularHistogramFitting *>(&backend);
}
} // namespace gjxl::histogram_gpu_internal
