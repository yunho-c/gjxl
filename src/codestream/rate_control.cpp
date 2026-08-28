// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/rate_control_internal.h"

#include <cmath>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gjxl::codestream_internal {
namespace {

[[nodiscard]] bool IsWithinTolerance(
  size_t encoded_bytes,
  size_t target_bytes,
  size_t tolerance_bytes) noexcept {

  return encoded_bytes <= target_bytes &&
    target_bytes - encoded_bytes <= tolerance_bytes;
}

[[nodiscard]] double FinalScore(
  const VarDctEncodingSummary& summary) noexcept {

  return summary.score_history.back();
}

[[nodiscard]] bool BetterCandidate(
  size_t target_bytes,
  float candidate_target,
  const std::vector<uint8_t>& candidate,
  const VarDctEncodingSummary& candidate_summary,
  const TargetSizeSearchResult& best) noexcept {

  if (best.codestream.empty()) {
    return true;
  }
  const bool candidate_is_under = candidate.size() <= target_bytes;
  const bool best_is_under = best.codestream.size() <= target_bytes;
  if (candidate_is_under != best_is_under) {
    return candidate_is_under;
  }
  if (candidate.size() != best.codestream.size()) {
    return candidate_is_under
      ? candidate.size() > best.codestream.size()
      : candidate.size() < best.codestream.size();
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

[[nodiscard]] Status EvaluateCandidate(
  float butteraugli_target,
  const TargetSizeSearchOptions& options,
  const TargetSizeEvaluator& evaluator,
  TargetSizeSearchResult* best,
  size_t* encoded_bytes) {

  std::vector<uint8_t> codestream;
  VarDctEncodingSummary summary;
  ++best->attempt_count;
  Status status = evaluator(
    butteraugli_target, &codestream, &summary);
  if (!status.ok()) {
    return status;
  }
  if (codestream.empty() || summary.encoded_bytes != codestream.size() ||
      summary.rate_control_mode !=
        VarDctRateControlMode::kButteraugliTarget ||
      summary.selected_butteraugli_target != butteraugli_target ||
      summary.score_history.empty() ||
      !std::isfinite(summary.score_history.back())) {
    return Status::Internal(
      "Target-size evaluator returned an invalid candidate");
  }

  *encoded_bytes = codestream.size();
  if (BetterCandidate(
        options.target_bytes,
        butteraugli_target,
        codestream,
        summary,
        *best)) {
    best->codestream = std::move(codestream);
    best->summary = std::move(summary);
  }
  return Status::Ok();
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
    float lower_target = options.minimum_butteraugli_target;
    float upper_target = options.maximum_butteraugli_target;
    size_t lower_size = 0;
    status = EvaluateCandidate(
      lower_target, options, evaluator, &candidate, &lower_size);
    if (!status.ok()) {
      return status;
    }
    if (IsWithinTolerance(
          lower_size, options.target_bytes, options.tolerance_bytes) ||
        lower_size <= options.target_bytes ||
        candidate.attempt_count == options.maximum_attempts) {
      candidate.target_size_met = IsWithinTolerance(
        candidate.codestream.size(),
        options.target_bytes,
        options.tolerance_bytes);
      *result = std::move(candidate);
      return Status::Ok();
    }

    size_t upper_size = 0;
    status = EvaluateCandidate(
      upper_target, options, evaluator, &candidate, &upper_size);
    if (!status.ok()) {
      return status;
    }
    if (IsWithinTolerance(
          upper_size, options.target_bytes, options.tolerance_bytes) ||
        upper_size > options.target_bytes ||
        candidate.attempt_count == options.maximum_attempts) {
      candidate.target_size_met = IsWithinTolerance(
        candidate.codestream.size(),
        options.target_bytes,
        options.tolerance_bytes);
      *result = std::move(candidate);
      return Status::Ok();
    }

    while (candidate.attempt_count < options.maximum_attempts) {
      const float midpoint = lower_target +
        0.5f * (upper_target - lower_target);
      if (midpoint == lower_target || midpoint == upper_target) {
        break;
      }
      size_t midpoint_size = 0;
      status = EvaluateCandidate(
        midpoint, options, evaluator, &candidate, &midpoint_size);
      if (!status.ok()) {
        return status;
      }
      if (IsWithinTolerance(
            midpoint_size, options.target_bytes, options.tolerance_bytes)) {
        break;
      }
      if (midpoint_size > options.target_bytes) {
        lower_target = midpoint;
        lower_size = midpoint_size;
      } else {
        upper_target = midpoint;
        upper_size = midpoint_size;
      }
      if (lower_size == upper_size) {
        break;
      }
    }

    candidate.target_size_met = IsWithinTolerance(
      candidate.codestream.size(),
      options.target_bytes,
      options.tolerance_bytes);
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
