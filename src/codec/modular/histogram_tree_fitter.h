// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "codec/modular/histogram_tree_learning.h"
#include <limits>
#include <memory>

namespace gjxl::modular_internal {
struct HistogramFittingProfile;

// Native evaluation boundary. Category order is static/zero-entropy-side,
// other static, nonstatic using Weighted/property 15, and nonstatic without WP.
// Costs exclude predictor penalties; low_rows counts entries, not weights.
struct HistogramSplitCandidate {
  uint32_t bin = 0;
  size_t low_rows = 0;
  Predictor low_predictor = Predictor::kZero, high_predictor = Predictor::kZero;
  float low_cost = std::numeric_limits<float>::max();
  float high_cost = std::numeric_limits<float>::max();
  bool valid = false;
  bool operator==(const HistogramSplitCandidate &) const = default;
};
struct HistogramPropertyEvaluation {
  bool constant = false;
  std::array<HistogramSplitCandidate, 4> winners;
  bool operator==(const HistogramPropertyEvaluation &) const = default;
};
struct HistogramNodeEvaluation {
  uint32_t node_id = 0, inherited_constants = 0;
  size_t begin = 0, end = 0, symbols = 0;
  Predictor parent_predictor = Predictor::kZero;
  float change_predictor_penalty = 0;
  std::span<const uint32_t> original_rows;
  std::span<const int32_t> parent_counts;
  std::span<const uint32_t> parent_extra_bits;
};
namespace histogram_fitting_internal {
struct State;
struct DeleteState {
  void operator()(State *) const noexcept;
};
} // namespace histogram_fitting_internal

// A single depth-first fit that can pause before independent property work.
// Sample columns, options.multipliers and any diagnostic sink outlive the fit.
// Concurrent/interleaved fits require separate diagnostic sinks. Fit/node
// envelopes include paused time; named CPU phase timers include only that work.
// The caller exclusively owns this state; sequential calls may migrate between
// admitted workers. CPU evaluation is serial: outer batching owns parallelism.
// Initialize allocates all tree/CPU scratch/row-order storage before
// partitioning. Advance handles CPU-only/forced nodes until a ready node or
// completion. Only Apply may consume a ready node; it validates every result
// before mutation. Publish is failure-atomic. Rows are scratch, including after
// a later failure.
class HistogramTreeFitter {
public:
  HistogramTreeFitter() = default;
  ~HistogramTreeFitter();
  HistogramTreeFitter(HistogramTreeFitter &&) noexcept;
  HistogramTreeFitter &operator=(HistogramTreeFitter &&) noexcept;
  HistogramTreeFitter(const HistogramTreeFitter &) = delete;
  HistogramTreeFitter &operator=(const HistogramTreeFitter &) = delete;

  [[nodiscard]] Status Initialize(HistogramTrainingSamples &samples,
                                  HistogramTreeLearningOptions options = {},
                                  HistogramFittingProfile *profile = nullptr);
  [[nodiscard]] Status Advance();
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] bool complete() const noexcept;
  [[nodiscard]] HistogramNodeEvaluation node() const noexcept;
  [[nodiscard]] std::span<const uint32_t> row_order() const noexcept;
  [[nodiscard]] Status EvaluateCpu(std::span<HistogramPropertyEvaluation> out);
  [[nodiscard]] Status
  Apply(std::span<const HistogramPropertyEvaluation> results);
  [[nodiscard]] Status Publish(DecisionTree *out);

private:
  std::unique_ptr<histogram_fitting_internal::State,
                  histogram_fitting_internal::DeleteState>
      state_;
};

[[nodiscard]] Status ComputeHistogramTreeFitterStorageBound(
    const HistogramTrainingSamples &samples,
    HistogramTreeLearningOptions options,
    resource_budget_internal::HostStorageBound *out);
} // namespace gjxl::modular_internal
