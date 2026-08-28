// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/rate_control_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gjxl::codestream_internal {
namespace {

struct SearchInterval {
  float lower = 0.0f;
  float upper = 0.0f;
};

[[nodiscard]] size_t AbsoluteByteError(
  size_t encoded_bytes,
  size_t target_bytes) noexcept {

  return encoded_bytes < target_bytes
    ? target_bytes - encoded_bytes
    : encoded_bytes - target_bytes;
}

[[nodiscard]] bool IsWithinTolerance(
  size_t encoded_bytes,
  const TargetSizeSearchOptions& options) noexcept {

  if (options.selection == TargetSizeSelectionPolicy::kClosestAbsolute) {
    return AbsoluteByteError(encoded_bytes, options.target_bytes) <=
      options.tolerance_bytes;
  }
  return encoded_bytes <= options.target_bytes &&
    options.target_bytes - encoded_bytes <= options.tolerance_bytes;
}

[[nodiscard]] double FinalScore(
  const VarDctEncodingSummary& summary) noexcept {

  return summary.score_history.empty()
    ? std::numeric_limits<double>::infinity()
    : summary.score_history.back();
}

[[nodiscard]] bool BetterCandidate(
  const TargetSizeSearchOptions& options,
  float candidate_target,
  const std::vector<uint8_t>& candidate,
  const VarDctEncodingSummary& candidate_summary,
  const TargetSizeSearchResult& best) noexcept {

  if (best.codestream.empty()) {
    return true;
  }
  const bool candidate_is_under =
    candidate.size() <= options.target_bytes;
  const bool best_is_under = best.codestream.size() <= options.target_bytes;
  if (options.selection == TargetSizeSelectionPolicy::kClosestAbsolute) {
    const size_t candidate_error =
      AbsoluteByteError(candidate.size(), options.target_bytes);
    const size_t best_error =
      AbsoluteByteError(best.codestream.size(), options.target_bytes);
    if (candidate_error != best_error) {
      return candidate_error < best_error;
    }
    if (candidate_is_under != best_is_under) {
      return candidate_is_under;
    }
  } else {
    if (candidate_is_under != best_is_under) {
      return candidate_is_under;
    }
    if (candidate.size() != best.codestream.size()) {
      return candidate_is_under
        ? candidate.size() > best.codestream.size()
        : candidate.size() < best.codestream.size();
    }
  }
  const double candidate_score = FinalScore(candidate_summary);
  const double best_score = FinalScore(best.summary);
  if (candidate_score != best_score) {
    return candidate_score < best_score;
  }
  return candidate_target < best.summary.selected_butteraugli_target;
}

[[nodiscard]] Status ValidateSearchOptions(
  const TargetSizeSearchOptions& options,
  const TargetSizeEvaluator& evaluator,
  const TargetSizeSearchResult* result) {

  switch (options.selection) {
    case TargetSizeSelectionPolicy::kLargestAtOrBelow:
    case TargetSizeSelectionPolicy::kClosestAbsolute:
      break;
    default:
      return Status::InvalidArgument(
        "Target-size selection policy is invalid");
  }
  if (result == nullptr || !evaluator || options.target_bytes == 0 ||
      options.maximum_attempts == 0 ||
      options.maximum_attempts > kMaximumTargetSizeEncodeAttempts ||
      !std::isfinite(options.minimum_butteraugli_target) ||
      !std::isfinite(options.maximum_butteraugli_target) ||
      options.minimum_butteraugli_target <= 0.0f ||
      options.maximum_butteraugli_target <=
        options.minimum_butteraugli_target) {
    return Status::InvalidArgument(
      "Target-size search options are invalid");
  }
  return Status::Ok();
}

/// Evaluator failures are candidate-local. An invalid successful result is a
/// search-contract violation and remains terminal.
[[nodiscard]] Status EvaluateCandidate(
  float butteraugli_target,
  const TargetSizeSearchOptions& options,
  const TargetSizeEvaluator& evaluator,
  TargetSizeSearchResult* best,
  Status* first_failure) {

  std::vector<uint8_t> codestream;
  VarDctEncodingSummary summary;
  ++best->attempt_count;
  Status status = evaluator(
    butteraugli_target, &codestream, &summary);
  if (!status.ok()) {
    ++best->failed_attempt_count;
    if (first_failure->ok()) {
      *first_failure = std::move(status);
    }
    return Status::Ok();
  }
  if (codestream.empty() || summary.encoded_bytes != codestream.size() ||
      summary.rate_control_mode !=
        VarDctRateControlMode::kButteraugliTarget ||
      summary.selected_butteraugli_target != butteraugli_target ||
      (!summary.score_history.empty() &&
       !std::isfinite(summary.score_history.back()))) {
    return Status::Internal(
      "Target-size evaluator returned an invalid candidate");
  }

  if (BetterCandidate(
        options, butteraugli_target, codestream, summary, *best)) {
    best->codestream = std::move(codestream);
    best->summary = std::move(summary);
  }
  return Status::Ok();
}

[[nodiscard]] size_t WidestInterval(
  const std::vector<SearchInterval>& intervals) noexcept {

  size_t best = 0;
  for (size_t index = 1; index < intervals.size(); ++index) {
    const float width = intervals[index].upper - intervals[index].lower;
    const float best_width =
      intervals[best].upper - intervals[best].lower;
    if (width > best_width ||
        (width == best_width &&
         intervals[index].lower < intervals[best].lower)) {
      best = index;
    }
  }
  return best;
}

}  // namespace

Status SearchTargetSize(
  const TargetSizeSearchOptions& options,
  const TargetSizeEvaluator& evaluator,
  TargetSizeSearchResult* result) {

  Status status = ValidateSearchOptions(options, evaluator, result);
  if (!status.ok()) {
    return status;
  }

  try {
    TargetSizeSearchResult candidate;
    Status first_failure;
    const float lower = options.minimum_butteraugli_target;
    const float upper = options.maximum_butteraugli_target;
    status = EvaluateCandidate(
      lower, options, evaluator, &candidate, &first_failure);
    if (!status.ok()) {
      return status;
    }
    if (candidate.attempt_count < options.maximum_attempts &&
        (candidate.codestream.empty() ||
         !IsWithinTolerance(candidate.codestream.size(), options))) {
      status = EvaluateCandidate(
        upper, options, evaluator, &candidate, &first_failure);
      if (!status.ok()) {
        return status;
      }
    }

    std::vector<SearchInterval> intervals;
    intervals.push_back({lower, upper});
    while (candidate.attempt_count < options.maximum_attempts &&
           (candidate.codestream.empty() ||
            !IsWithinTolerance(candidate.codestream.size(), options)) &&
           !intervals.empty()) {
      const size_t index = WidestInterval(intervals);
      const SearchInterval interval = intervals[index];
      intervals.erase(intervals.begin() + static_cast<std::ptrdiff_t>(index));
      const float midpoint = interval.lower +
        0.5f * (interval.upper - interval.lower);
      if (midpoint == interval.lower || midpoint == interval.upper) {
        continue;
      }
      status = EvaluateCandidate(
        midpoint, options, evaluator, &candidate, &first_failure);
      if (!status.ok()) {
        return status;
      }
      intervals.push_back({interval.lower, midpoint});
      intervals.push_back({midpoint, interval.upper});
    }

    if (candidate.codestream.empty()) {
      return first_failure.ok()
        ? Status::Internal(
            "Target-size search produced no valid candidate")
        : first_failure;
    }
    candidate.target_size_met =
      IsWithinTolerance(candidate.codestream.size(), options);
    candidate.search_exhausted = !candidate.target_size_met &&
      (candidate.attempt_count == options.maximum_attempts ||
       intervals.empty());
    *result = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate target-size search storage");
  } catch (const std::length_error&) {
    return Status::OutOfMemory(
      "Target-size search storage is too large");
  }
  return Status::Ok();
}

}  // namespace gjxl::codestream_internal
