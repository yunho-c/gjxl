// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "codec/modular/tree_learning.h"

namespace gjxl::modular_internal {

inline constexpr size_t kHistogramPredictors = 14;
inline constexpr size_t kHistogramStaticProperties = 2;
inline constexpr size_t kHistogramMaximumStaticBins = 65536;
// Private compact-data admission. Existing/public learning budgets retain their
// 1M ceiling. Larger replay inputs require an explicit maximum_samples request.
inline constexpr size_t kHistogramMaximumSamples = size_t{1} << 24;

struct HistogramResidualToken {
  uint8_t symbol = 0;
  uint8_t extra_bits = 0;
  bool operator==(const HistogramResidualToken &) const = default;
};

// Quantized, weighted rows in predictor/property column order. Property IDs are
// the native 0..15 wire properties; static properties must be a leading 0,1
// prefix. Nonstatic column i corresponds to property i + static_property_count.
// thresholds[p][bin] is the original split value for that quantized bin.
// Row storage is caller-owned and is partitioned in place by fitting; ordering
// is scratch, including after a fitting failure. Weights/rows remain
// associated. This representation has no libjxl, entropy-writer or stream-plan
// dependency.
struct HistogramTrainingSamples {
  size_t total_samples = 0;
  size_t predictor_count = 0, property_count = 0, static_property_count = 0;
  std::array<Predictor, kHistogramPredictors> predictors{};
  std::array<uint8_t, kPropertyCount> properties{};
  resource_budget_internal::ManagedVector<uint16_t> weights;
  std::array<resource_budget_internal::ManagedVector<HistogramResidualToken>,
             kHistogramPredictors>
      residuals;
  std::array<resource_budget_internal::ManagedVector<uint32_t>,
             kHistogramStaticProperties>
      static_values;
  std::array<resource_budget_internal::ManagedVector<uint8_t>, kPropertyCount>
      values;
  std::array<resource_budget_internal::ManagedVector<int32_t>, kPropertyCount>
      thresholds;
  void Swap(size_t a, size_t b);
};

using HistogramStaticRange =
    std::array<std::array<uint32_t, 2>, kHistogramStaticProperties>;
struct HistogramMultiplierRange {
  HistogramStaticRange range;
  uint32_t multiplier = 1;
};
struct HistogramTreeLearningStats {
  size_t participants = 1, parallel_nodes = 0;
};
struct HistogramTreeLearningOptions {
  // Effective threshold, after the caller scales for its sampling fraction.
  float split_threshold = 89.0f;
  float fast_decode_multiplier = 1.001f;
  HistogramStaticRange static_range{{{0, UINT32_MAX}, {0, UINT32_MAX}}};
  std::span<const HistogramMultiplierRange> multipliers;
  size_t maximum_samples = kMaximumTrainingSamples;
  size_t maximum_leaves = kMaximumTreeLeaves;
  size_t maximum_depth = kMaximumTreeDepth;
  // Opt-in property parallelism. Zero uses the current CPU allowance; private
  // calls without participation and nested parallel scopes remain serial.
  size_t cpu_participants = 1;
  HistogramTreeLearningStats *execution_stats = nullptr;
};

// Shape/context-only upper limit; the actual CPU grant can be smaller. Small
// sample sets and shapes with fewer than two varying properties stay serial.
[[nodiscard]] size_t HistogramTreeLearningParticipants(
    const HistogramTrainingSamples &samples, HistogramTreeLearningOptions options);

// Allocation-free input/option validation. Rejects unsupported properties,
// malformed bins/weights and arithmetic/resource overflows before any mutation.
[[nodiscard]] Status
ValidateHistogramTrainingSamples(const HistogramTrainingSamples &samples,
                                 HistogramTreeLearningOptions options = {});
// Bounds new fitting scratch and result backing, excluding caller-owned sample
// columns and old output. Shape validation only; does not scan the sample rows.
[[nodiscard]] Status ComputeHistogramTreeLearningStorageBound(
    const HistogramTrainingSamples &samples,
    HistogramTreeLearningOptions options,
    resource_budget_internal::HostStorageBound *out);
// Capacity-only upper bound for admitting independent prepare-and-fit jobs
// before their sample columns exist. maximum_bins includes every static and
// dynamic property's bins. Excludes sample columns and tree compaction.
[[nodiscard]] Status ComputeHistogramTreeWorkspaceStorageBound(
    size_t predictor_count, size_t maximum_bins, size_t maximum_leaves,
    size_t maximum_depth, resource_budget_internal::HostStorageBound *out);
// Threshold-driven, depth-first reference-style growth. Capacity/depth limits
// reject the fit; they never silently truncate it. All allocation is completed
// before row partitioning; only *out publication is atomic. The caller provides
// an image-dependent maximum_leaves limit when fitting a global coding tree.
[[nodiscard]] Status
LearnHistogramTree(HistogramTrainingSamples &samples, DecisionTree *out,
                   HistogramTreeLearningOptions options = {});

} // namespace gjxl::modular_internal
