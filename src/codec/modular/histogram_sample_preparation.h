// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/histogram_tree_learning.h"
#include "codec/modular/weighted_features.h"

namespace gjxl::modular_internal {
class HistogramTreeFitter;
struct HistogramFittingProfile;

inline constexpr size_t kHistogramMaximumPopulation = size_t{1} << 28;
// One reference learning image, normally a canonical Modular stream. Empty
// entries preserve holes in group IDs; channel numbering restarts per image.
// Channels and their pixels are borrowed for the synchronous preparation call.
struct HistogramTrainingImage {
  Extent2D extent;
  std::span<const ModularChannelView> channels;
  size_t metadata_channels = 0;
  // Applies to this stream's channels; predictor state still resets for each
  // channel. These must be the parameters used when tokenizing the same data.
  WeightedPredictorParameters weighted;
  // Optional parallel array of channel-local completed maps. Empty array uses
  // scalar WP for every channel; an empty entry uses scalar WP for that channel.
  // Borrowed maps preserve all sampling, quantization and deduplication order.
  std::span<const WeightedFeatureView> weighted_features;
};
struct HistogramSamplingOptions {
  float sampling_fraction = 0.5f * 1.3f;
  float split_threshold = 89.0f;
  float fast_decode_multiplier = 1.001f;
  size_t maximum_samples = kMaximumTrainingSamples;
  size_t maximum_property_values = 256;
  size_t maximum_channel_size = 0xFFFFFF;
  uint32_t first_group = 0;
  size_t predictor_count = 2, property_count = kPropertyCount;
  std::array<Predictor, kHistogramPredictors> predictors{Predictor::kWeighted,
                                                         Predictor::kGradient};
  std::array<uint8_t, kPropertyCount> properties{0,  1, 15, 9, 10, 11, 12, 13,
                                                 14, 2, 3,  4, 5,  6,  7,  8};
  // Previous-channel properties are outside this profile. Multiplier boxes
  // are copied into the prepared result.
  std::span<const HistogramMultiplierRange> multipliers;
};
struct HistogramPreparationStats {
  size_t scanned_pixels = 0, population = 0, selected_samples = 0;
  size_t quantization_samples = 0, channels = 0;
};

// Finalized rows are immutable to callers. Fitting may reorder them internally.
// A default-constructed object is not a successfully prepared sample set.
class PreparedHistogramSamples {
public:
  PreparedHistogramSamples() = default;
  PreparedHistogramSamples(const PreparedHistogramSamples &) = delete;
  PreparedHistogramSamples &
  operator=(const PreparedHistogramSamples &) = delete;
  PreparedHistogramSamples(PreparedHistogramSamples &&) noexcept = default;
  PreparedHistogramSamples &
  operator=(PreparedHistogramSamples &&) noexcept = default;
  [[nodiscard]] const HistogramTrainingSamples &samples() const {
    return samples_;
  }
  [[nodiscard]] const HistogramPreparationStats &stats() const {
    return stats_;
  }
  [[nodiscard]] HistogramTreeLearningOptions tree_options() const {
    HistogramTreeLearningOptions options;
    options.split_threshold = threshold_;
    options.fast_decode_multiplier = fast_decode_;
    options.static_range = range_;
    options.multipliers = multipliers_;
    options.maximum_samples = maximum_samples_;
    return options;
  }

private:
  friend struct HistogramSampleBuilder;
  friend Status PrepareHistogramSamples(std::span<const HistogramTrainingImage>,
                                        PreparedHistogramSamples *,
                                        HistogramSamplingOptions);
  friend Status LearnPreparedHistogramTree(PreparedHistogramSamples &,
                                           DecisionTree *, size_t, size_t, size_t,
                                           HistogramTreeLearningStats *);
  friend Status InitializePreparedHistogramTreeFitter(
      PreparedHistogramSamples &, HistogramTreeFitter *, size_t, size_t,
      HistogramFittingProfile *);
  HistogramTrainingSamples samples_;
  resource_budget_internal::ManagedVector<HistogramMultiplierRange>
      multipliers_;
  HistogramPreparationStats stats_;
  HistogramStaticRange range_{};
  float threshold_ = 0, fast_decode_ = 1;
  size_t maximum_samples_ = 0;
};

// Borrows prepared columns/options until the fit is published or destroyed.
// Do not move or reuse the prepared owner during that lifetime.
[[nodiscard]] Status InitializePreparedHistogramTreeFitter(
    PreparedHistogramSamples &samples, HistogramTreeFitter *fitter,
    size_t maximum_leaves = kMaximumTreeLeaves,
    size_t maximum_depth = kMaximumTreeDepth,
    HistogramFittingProfile *profile = nullptr);

// Allocation-free and pixel-read-free; replays the geometry-dependent RNG to
// count admission exactly. Does not truncate samples to meet an admission cap.
// Bounds all new preparation/result backing, excluding borrowed pixels and old
// output and borrowed feature maps. Includes worst-case deduplication state
// and one live WP channel state for uncovered channels.
[[nodiscard]] Status ComputeHistogramSamplePreparationStorageBound(
    std::span<const HistogramTrainingImage> images,
    HistogramSamplingOptions options,
    resource_budget_internal::HostStorageBound *out);
// Native quantization, reference sampling, residual proxy and weighted-row
// deduplication. Both result and counters publish only on complete success.
[[nodiscard]] Status
PrepareHistogramSamples(std::span<const HistogramTrainingImage> images,
                        PreparedHistogramSamples *out,
                        HistogramSamplingOptions options = {});
[[nodiscard]] Status
LearnPreparedHistogramTree(PreparedHistogramSamples &samples, DecisionTree *out,
                           size_t maximum_leaves = kMaximumTreeLeaves,
                           size_t maximum_depth = kMaximumTreeDepth,
                           size_t cpu_participants = 1,
                           HistogramTreeLearningStats *execution_stats = nullptr);
} // namespace gjxl::modular_internal
