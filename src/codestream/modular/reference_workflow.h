// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/histogram_sample_preparation.h"
#include "codec/modular/reference_selection.h"
#include "codestream/modular/prepared_streams.h"
#include "gpu/ops/modular_stream_tokens.h"
#include "gpu/ops/modular_histogram_fitting.h"

namespace gjxl::modular_internal {
enum class ReferenceBuffering : uint8_t { kAutomatic, kBuffered };
enum class ReferencePredictors : uint8_t { kBest, kAll };
// Private qualification controls. Public routing remains on the ordinary CPU
// fitter until complete-call evidence supports an automatic GPU preference.
enum class ReferenceFittingBackend : uint8_t { kCpu, kResumableCpu, kGpu, kAuto };
struct ReferenceWorkflowOptions {
  ReferenceBuffering buffering = ReferenceBuffering::kAutomatic;
  ReferencePredictors predictors = ReferencePredictors::kBest;
  size_t maximum_samples = kHistogramMaximumSamples;
  PreparedModularEntropyOptions entropy{
      PreparedModularEntropyPolicy::kDirectRateAns,
      AnsSelectionMode::kEstimated, 64};
  PreparedModularTokenizationOptions tokenization;
  // Optional private hybrid route. Null selects CPU features and tokens.
  // Backend outlives a prepared frame and its completed token owners.
  GpuBackend *gpu = nullptr;
  WeightedFeatureBackend feature_producer = WeightedFeatureBackend::kAuto;
  size_t feature_cpu_participants = 4;
  // Local prepare-and-fit jobs, or property evaluation within a shared fit.
  // Zero uses the caller's CPU allowance; one is the serial control. Workers
  // require CPU participation and managed-memory admission before launch.
  size_t learning_cpu_participants = 0;
  // Independent local palette/RCT/WP selection jobs. Zero inherits the
  // admitted CPU allowance; one retains serial selection. Global selection
  // still precedes stream placement and always runs as one job.
  size_t selection_cpu_participants = 0;
  ReferenceFittingBackend fitting_backend = ReferenceFittingBackend::kCpu;
  // Private experiments use the measured 16K-row cutoff; GPU fitting remains
  // disabled by default. This is not a guaranteed CPU/GPU speed crossover.
  size_t fitting_minimum_rows = 16384, fitting_batch_fits = 64;
  histogram_gpu_internal::HistogramFittingExecutionOptions fitting_execution;
  // Caller-owned, coordinator-only sink; disabled for ordinary encoding.
  histogram_gpu_internal::HistogramFittingProfile *fitting_gpu_profile = nullptr;

  // Private sample-row experiment. GPU batching is initially local-scope only;
  // global scopes use selected CPU rows. No public policy/default changes.
  HistogramRowBackend sample_rows = HistogramRowBackend::kRasterCpu;
  bool use_unique_row_proof = false;
  bool defer_gpu_row_materialization = false;
  bool gpu_row_diagnostics = false;
  HistogramRowGenerator *row_generator = nullptr;
  const HistogramRowSource *row_source =
      nullptr; // Borrowed for a joined local batch.
};
struct ReferenceLearningStats {
  size_t first_group = 0, stop_group = 0;
  HistogramPreparationStats preparation;
  size_t distinct_samples = 0, tree_nodes = 0;
};
struct ReferenceWorkflowStats {
  size_t selection_peak_participants = 0, selection_waves = 0,
         selection_memory_limited_waves = 0;
  // Joined caller wall versus summed job wall; work is not an additive stage.
  double selection_seconds = 0, selection_work_seconds = 0;
  size_t entropy_cpu_participants = 1, entropy_memory_reductions = 0;
  size_t tokenization_cpu_participants = 1, tokenization_memory_reductions = 0;
  bool local_trees = false;
  size_t selection_calls = 0, learning_calls = 0, fits = 0,
         empty_learning_calls = 0;
  size_t scanned_pixels = 0, selected_samples = 0, distinct_samples = 0,
         tree_nodes = 0;
  // Largest individual stage reservation, excluding backing retained by other
  // stages. The caller's ResourceBudget observes the complete managed peak.
  size_t maximum_stage_bytes = 0;
  size_t feature_batches = 0, feature_samples = 0, gpu_feature_batches = 0,
         gpu_feature_samples = 0, cpu_feature_batches = 0;
  // Local fitting cohorts can retain several feature owners under one shared
  // sample ceiling. Admission pressure flushes a nonempty cohort early.
  size_t feature_cohorts = 0, feature_memory_limited_cohorts = 0;
  size_t token_batches = 0, token_samples = 0;
  size_t peak_feature_samples = 0, peak_feature_bytes = 0;
  // A learning wave is now one joined ready-queue invocation, which can
  // consume more jobs than its participant grant. Fields retain their names
  // for existing private diagnostic readers.
  size_t learning_peak_participants = 0, learning_waves = 0,
         learning_memory_limited_waves = 0;
  // Sums of per-job wall time (overlap when local jobs run concurrently).
  // Preparation includes sample planning; fitting includes compaction.
  double sample_preparation_seconds = 0, fitting_seconds = 0;
  HistogramRowProfile rows;
  // Caller-wall intervals. Learning includes admission, preparation, fitting,
  // joins and diagnostic reduction. Features and GPU tokens are separate.
  double learning_seconds = 0, feature_seconds = 0, token_seconds = 0;
  size_t histogram_gpu_preparations = 0, histogram_gpu_batches = 0,
         histogram_gpu_nodes = 0, histogram_gpu_rows = 0,
         histogram_gpu_properties = 0, histogram_cpu_nodes = 0,
         histogram_admission_fallbacks = 0;
  size_t histogram_prepared_bytes = 0, histogram_dispatch_bytes = 0,
         histogram_peak_scratch_bytes = 0;
  // Synchronous caller-wall operation times, including copies/allocations and
  // waits. CPU fitting_seconds excludes these GPU operations in a batched fit.
  double histogram_preparation_seconds = 0, histogram_evaluation_seconds = 0;
};
// A lossless source has at most four normal planes. One reference palette pass
// adds at most four metadata planes; local passes start without global meta.
inline constexpr size_t kReferenceStreamChannels = 8;
struct ReferenceModularStream {
  size_t id = 0, channel_count = 0, metadata_channels = 0;
  Extent2D learning_extent;
  std::array<ModularChannelView, kReferenceStreamChannels> channels{};
  ReferenceSelectionResult selection;
  DecisionTree tree;
  ReferenceLearningStats learning;
  [[nodiscard]] std::span<const ModularChannelView> training_channels() const {
    return std::span(channels).first(channel_count);
  }
};
// Owns all samples/trees and optional GPU token backing behind view(). A GPU
// backend must outlive this frame. No feature maps survive preparation.
// Channel spans and local-tree pointers
// point into fixed managed allocations, so moves preserve their addresses.
// The global learning image retains excluded descriptors, while view() exposes
// only the coded global prefix. Empty canonical stream-ID slots are preserved
// during global preparation; local learning uses [0,1).
class ReferenceModularFrame {
public:
  ReferenceModularFrame() = default;
  ReferenceModularFrame(const ReferenceModularFrame &) = delete;
  ReferenceModularFrame &operator=(const ReferenceModularFrame &) = delete;
  ReferenceModularFrame(ReferenceModularFrame &&) noexcept = default;
  ReferenceModularFrame &operator=(ReferenceModularFrame &&) noexcept = default;
  [[nodiscard]] PreparedModularStreamsView view() const;
  [[nodiscard]] std::span<const ReferenceModularStream> streams() const {
    return records_;
  }
  [[nodiscard]] const ReferenceWorkflowStats &stats() const { return stats_; }

private:
  friend Status
  PrepareReferenceModularFrame(PackedModularImageView,
                               const ReferenceWorkflowOptions &,
                               const resource_budget_internal::ResourceBudget &,
                               ReferenceModularFrame *);
  ModularEncoderFrame source_;
  Extent2D extent_;
  PackedModularFormat format_ = PackedModularFormat::kRgb8;
  resource_budget_internal::ManagedVector<ReferenceModularStream> records_;
  resource_budget_internal::ManagedVector<
      std::unique_ptr<modular_gpu_internal::PreparedGpuModularTokens>> gpu_tokens_;
  resource_budget_internal::ManagedVector<PreparedModularStreamView> streams_;
  ReferenceWorkflowStats stats_;
};

// Private ordinary lossless e9 profile: fixed 256-pixel groups, preserved
// invisible RGB, no responsive transforms, patches, custom trees or extra
// channels beyond unassociated alpha. Automatic mode uses local trees above
// eight groups and skips global palettes. Buffered mode retains global
// palettes and one shared tree. Each selected scope is prepared once and fit
// once; empty learning calls produce a trivial tree without invoking the
// fitter. Caps reject rather than truncate. The result owns its samples and
// does not borrow the packed input.
//
// Staged, nonblocking admission in the caller's managed-memory domain. Before
// every allocating stage, its existing planner bounds the fresh backing and
// TryReserve checks capacity alongside all retained backing. No whole-workflow
// bound is claimed. Busy capacity returns Unavailable; failures preserve *out.
// The caller must account for borrowed packed bytes and previous outputs too.
[[nodiscard]] Status PrepareReferenceModularFrame(
    PackedModularImageView input, const ReferenceWorkflowOptions &options,
    const resource_budget_internal::ResourceBudget &budget,
    ReferenceModularFrame *out);
[[nodiscard]] Status EncodeReferenceModularImage(
    PackedModularImageView input, const ReferenceWorkflowOptions &options,
    const resource_budget_internal::ResourceBudget &budget,
    codestream_internal::CodestreamBuffer *out,
    ReferenceWorkflowStats *stats = nullptr);
} // namespace gjxl::modular_internal
