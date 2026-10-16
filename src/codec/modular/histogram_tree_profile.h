// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>

namespace gjxl::modular_internal {
// Private opt-in diagnostics for the native histogram fitter, independent of
// the original learner's profile. Output backing is caller-owned; property
// counters live in admitted fitter scratch with no per-node allocations.
// Property stage times sum task wall intervals,
// overlapping within a parallel fit as well as across independent fits. Fit
// and node times remain elapsed spans; stage sums are not elapsed encode time.
enum class HistogramFitStage {
  kValidation,
  kSetup,
  kSymbolRange,
  kParentHistograms,
  kPropertyPopulation,
  kJointHistograms,
  kCandidateEvaluation,
  kSplitSelection,
  kPartition,
  kOutput,
  kCount
};
inline constexpr std::array<const char *, 10> kHistogramFitStageNames{
    "validation",
    "setup",
    "symbol_range",
    "parent_histograms",
    "property_population",
    "joint_histograms",
    "candidate_evaluation",
    "split_selection",
    "partition",
    "output"};
using HistogramFitSeconds =
    std::array<double, size_t(HistogramFitStage::kCount)>;
// Bucket zero is 0..1; bucket i>0 is (2^(i-1), 2^i]. The input contract caps
// populations at 2^24 and bins at 2^16. Last buckets are saturating, not lost.
inline constexpr size_t kHistogramRowBuckets = 25, kHistogramBinBuckets = 17;
inline size_t HistogramProfileBucket(size_t value, size_t buckets) {
  const size_t index = value <= 1 ? 0 : std::bit_width(value - 1);
  return index < buckets ? index : buckets - 1;
}
struct HistogramNodeWork {
  uint64_t nodes = 0, splits = 0, forced_splits = 0, rows = 0;
  uint64_t symbol_row_visits = 0, parent_histogram_row_visits = 0;
  uint64_t property_row_visits = 0, joint_histogram_row_visits = 0;
  // Occupancy and first-predictor histograms share these visits. Physical
  // passes = property_row_visits + joint_histogram_row_visits - fused visits.
  uint64_t fused_property_row_visits = 0, skipped_constant_properties = 0;
  uint64_t properties = 0, constant_properties = 0, active_bins = 0,
           bin_span = 0;
  uint64_t candidate_thresholds = 0, entropy_evaluations = 0, symbol_terms = 0;
  uint64_t partition_rows = 0, row_swaps = 0;
  bool operator==(const HistogramNodeWork &) const = default;
};
struct HistogramNodeBucket {
  HistogramNodeWork work;
  HistogramFitSeconds seconds{};
  double node_seconds = 0;
};
struct HistogramDepthBucket {
  uint64_t nodes = 0, rows = 0;
  double seconds = 0;
};
struct HistogramFitRecord {
  size_t rows = 0, population = 0, predictors = 0, properties = 0;
  size_t nodes = 0, maximum_depth = 0;
  bool success = false;
  double seconds = 0;
  HistogramFitSeconds stages{};
};
struct HistogramFittingProfile {
  HistogramFitSeconds seconds{};
  double fit_seconds = 0;
  uint64_t calls = 0, successes = 0;
  size_t peak_property_participants = 1, parallel_property_nodes = 0;
  std::array<HistogramNodeBucket, kHistogramRowBuckets> by_rows{};
  // Depth is one-based. The final entry also records any larger depth.
  std::array<HistogramDepthBucket, 2048> by_depth{};
  // Padded maximum symbol+1, as actually scanned by EstimateBits. This is
  // symbol-axis length, NOT the number of nonzero residual-symbol bins.
  std::array<uint64_t, 257> nodes_by_symbol_span{};
  std::array<uint64_t, kHistogramBinBuckets> properties_by_active_bins{};
  std::array<uint64_t, kHistogramBinBuckets> properties_by_bin_span{};
  std::span<HistogramFitRecord> records;
  size_t records_written = 0, records_dropped = 0;
};
inline void MergeHistogramNodeWork(HistogramNodeWork &to,
                                   const HistogramNodeWork &from) {
  for (auto member :
       {&HistogramNodeWork::nodes, &HistogramNodeWork::splits,
        &HistogramNodeWork::forced_splits, &HistogramNodeWork::rows,
        &HistogramNodeWork::symbol_row_visits,
        &HistogramNodeWork::parent_histogram_row_visits,
        &HistogramNodeWork::property_row_visits,
        &HistogramNodeWork::joint_histogram_row_visits,
        &HistogramNodeWork::fused_property_row_visits,
        &HistogramNodeWork::skipped_constant_properties,
        &HistogramNodeWork::properties, &HistogramNodeWork::constant_properties,
        &HistogramNodeWork::active_bins, &HistogramNodeWork::bin_span,
        &HistogramNodeWork::candidate_thresholds,
        &HistogramNodeWork::entropy_evaluations,
        &HistogramNodeWork::symbol_terms, &HistogramNodeWork::partition_rows,
        &HistogramNodeWork::row_swaps})
    to.*member += from.*member;
}
// Reduce only after joining workers, in canonical stream order. Each source
// owns its counters/records; no hot-loop atomic updates or shared sink writes.
inline void MergeHistogramFittingProfile(HistogramFittingProfile &to,
                                         const HistogramFittingProfile &from) {
  to.fit_seconds += from.fit_seconds;
  to.calls += from.calls;
  to.successes += from.successes;
  to.peak_property_participants =
      std::max(to.peak_property_participants, from.peak_property_participants);
  to.parallel_property_nodes += from.parallel_property_nodes;
  for (size_t i = 0; i < to.seconds.size(); ++i)
    to.seconds[i] += from.seconds[i];
  for (size_t i = 0; i < to.by_rows.size(); ++i) {
    auto &a = to.by_rows[i];
    const auto &b = from.by_rows[i];
    a.node_seconds += b.node_seconds;
    for (size_t j = 0; j < a.seconds.size(); ++j)
      a.seconds[j] += b.seconds[j];
    MergeHistogramNodeWork(a.work, b.work);
  }
  for (size_t i = 0; i < to.by_depth.size(); ++i) {
    to.by_depth[i].nodes += from.by_depth[i].nodes;
    to.by_depth[i].rows += from.by_depth[i].rows;
    to.by_depth[i].seconds += from.by_depth[i].seconds;
  }
  for (size_t i = 0; i < to.nodes_by_symbol_span.size(); ++i)
    to.nodes_by_symbol_span[i] += from.nodes_by_symbol_span[i];
  for (size_t i = 0; i < to.properties_by_active_bins.size(); ++i) {
    to.properties_by_active_bins[i] += from.properties_by_active_bins[i];
    to.properties_by_bin_span[i] += from.properties_by_bin_span[i];
  }
  to.records_dropped += from.records_dropped;
  for (size_t i = 0; i < from.records_written; ++i) {
    if (to.records_written < to.records.size())
      to.records[to.records_written++] = from.records[i];
    else
      ++to.records_dropped;
  }
}
inline thread_local HistogramFittingProfile *active_histogram_fitting_profile =
    nullptr;
class HistogramFittingProfileSession {
public:
  explicit HistogramFittingProfileSession(HistogramFittingProfile *profile)
      : previous_(active_histogram_fitting_profile) {
    active_histogram_fitting_profile = profile;
  }
  ~HistogramFittingProfileSession() {
    active_histogram_fitting_profile = previous_;
  }
  HistogramFittingProfileSession(const HistogramFittingProfileSession &) =
      delete;
  HistogramFittingProfileSession &
  operator=(const HistogramFittingProfileSession &) = delete;

private:
  HistogramFittingProfile *previous_;
};

namespace histogram_profile_internal {
using Clock = std::chrono::steady_clock;
inline double Elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
// Compile-time disabled probes remove clocks/counters from the hot loops.
template <bool Enabled> class Phase {
public:
  template <class Profile>
  Phase(Profile *profile, HistogramFitStage stage,
        HistogramNodeBucket *node = nullptr)
      : seconds_(profile ? &profile->seconds : nullptr), node_(node),
        stage_(stage) {
    if constexpr (Enabled) {
      if (seconds_)
        start_ = Clock::now();
    }
  }
  ~Phase() { Stop(); }
  void Stop() {
    if constexpr (Enabled) {
      if (!seconds_)
        return;
      const double seconds = Elapsed(start_);
      (*seconds_)[size_t(stage_)] += seconds;
      if (node_)
        node_->seconds[size_t(stage_)] += seconds;
      seconds_ = nullptr;
    }
  }
  Phase(const Phase &) = delete;
  Phase &operator=(const Phase &) = delete;

private:
  HistogramFitSeconds *seconds_;
  HistogramNodeBucket *node_;
  HistogramFitStage stage_;
  Clock::time_point start_{};
};
template <bool Enabled> class Node {
public:
  Node(HistogramFittingProfile *profile, size_t rows, size_t depth,
       HistogramFitRecord *record) {
    if constexpr (Enabled) {
      bucket =
          &profile->by_rows[HistogramProfileBucket(rows, kHistogramRowBuckets)];
      depth_ = &profile->by_depth[depth < profile->by_depth.size()
                                      ? depth
                                      : profile->by_depth.size() - 1];
      ++bucket->work.nodes;
      bucket->work.rows += rows;
      ++depth_->nodes;
      depth_->rows += rows;
      if (record && record->maximum_depth < depth)
        record->maximum_depth = depth;
      start_ = Clock::now();
    }
  }
  ~Node() {
    if constexpr (Enabled) {
      const double seconds = Elapsed(start_);
      bucket->node_seconds += seconds;
      depth_->seconds += seconds;
    }
  }
  HistogramNodeBucket *bucket = nullptr;

private:
  HistogramDepthBucket *depth_ = nullptr;
  Clock::time_point start_{};
};
// Declared before fitting storage so the envelope includes its destruction.
class Fit {
public:
  explicit Fit(HistogramFittingProfile *profile) : profile_(profile) {
    if (!profile_)
      return;
    start_ = Clock::now();
    before_ = profile_->seconds;
    ++profile_->calls;
    if (profile_->records_written < profile_->records.size()) {
      record = &profile_->records[profile_->records_written++];
      *record = {};
    } else {
      ++profile_->records_dropped;
    }
  }
  ~Fit() {
    if (!profile_)
      return;
    const double seconds = Elapsed(start_);
    profile_->fit_seconds += seconds;
    if (record) {
      record->seconds = seconds;
      for (size_t i = 0; i < before_.size(); ++i)
        record->stages[i] = profile_->seconds[i] - before_[i];
    }
  }
  void Success(size_t nodes) {
    if (!profile_)
      return;
    ++profile_->successes;
    if (record) {
      record->success = true;
      record->nodes = nodes;
    }
  }
  HistogramFitRecord *record = nullptr;

private:
  HistogramFittingProfile *profile_;
  HistogramFitSeconds before_{};
  Clock::time_point start_{};
};
} // namespace histogram_profile_internal
} // namespace gjxl::modular_internal
