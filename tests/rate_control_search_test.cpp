// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include "codestream/rate_control_internal.h"
#include "codestream/workflow.h"
#include "core/status.h"

namespace {

using SearchOptions = gjxl::codestream_internal::TargetSizeSearchOptions;
using SearchResult = gjxl::codestream_internal::TargetSizeSearchResult;
using Evaluator = gjxl::codestream_internal::TargetSizeEvaluator;

[[nodiscard]] gjxl::VarDctEncodingSummary MakeSummary(
  float target,
  size_t encoded_bytes,
  double score) {

  gjxl::VarDctEncodingSummary summary;
  summary.extent = {8, 8};
  summary.encoded_bytes = encoded_bytes;
  summary.rate_control_mode =
    gjxl::VarDctRateControlMode::kButteraugliTarget;
  summary.achieved_bits_per_pixel =
    static_cast<double>(encoded_bytes);
  summary.selected_butteraugli_target = target;
  summary.encode_attempt_count = 1;
  summary.score_history = {score};
  return summary;
}

[[nodiscard]] Evaluator MakeEvaluator(
  std::function<size_t(float)> size_for_target,
  std::function<double(float)> score_for_target =
    [](float target) { return static_cast<double>(target); }) {

  return [size_for_target = std::move(size_for_target),
          score_for_target = std::move(score_for_target)](
           float target,
           std::vector<uint8_t>* codestream,
           gjxl::VarDctEncodingSummary* summary) {
    const size_t size = size_for_target(target);
    std::vector<uint8_t> candidate(size, 0x5a);
    *codestream = std::move(candidate);
    *summary = MakeSummary(target, size, score_for_target(target));
    return gjxl::Status::Ok();
  };
}

[[nodiscard]] size_t MonotonicSize(float target) {
  const size_t reduction = std::min<size_t>(
    1000,
    static_cast<size_t>(target * 100.0f));
  return 2000 - reduction;
}

bool CheckBracketedSearch() {
  SearchResult result;
  const gjxl::Status status =
    gjxl::codestream_internal::SearchTargetSize(
      {
        .target_bytes = 1500,
        .tolerance_bytes = 0,
        .maximum_attempts = 12,
      },
      MakeEvaluator(MonotonicSize),
      &result);
  if (!status.ok() || result.codestream.size() != 1500 ||
      result.summary.encoded_bytes != 1500 ||
      result.attempt_count != 3 || !result.target_size_met) {
    std::cerr << "Bracketed target-size search failed\n";
    return false;
  }
  return true;
}

bool CheckInfeasibleTargets() {
  SearchResult oversized;
  gjxl::Status status = gjxl::codestream_internal::SearchTargetSize(
    {
      .target_bytes = 2500,
      .maximum_attempts = 12,
    },
    MakeEvaluator(MonotonicSize),
    &oversized);
  if (!status.ok() || oversized.codestream.size() != 1999 ||
      oversized.attempt_count != 1 || oversized.target_size_met ||
      oversized.summary.selected_butteraugli_target !=
        gjxl::codestream_internal::kMinimumTargetSizeButteraugliTarget) {
    std::cerr << "Oversized target-size request was mishandled\n";
    return false;
  }

  SearchResult undersized;
  status = gjxl::codestream_internal::SearchTargetSize(
    {
      .target_bytes = 900,
      .maximum_attempts = 12,
    },
    MakeEvaluator(MonotonicSize),
    &undersized);
  if (!status.ok() || undersized.codestream.size() != 1000 ||
      undersized.attempt_count != 2 || undersized.target_size_met ||
      undersized.summary.selected_butteraugli_target !=
        gjxl::codestream_internal::kMaximumTargetSizeButteraugliTarget) {
    std::cerr << "Undersized target-size request was mishandled\n";
    return false;
  }
  return true;
}

bool CheckAttemptLimitAndPlateau() {
  SearchResult limited;
  gjxl::Status status = gjxl::codestream_internal::SearchTargetSize(
    {
      .target_bytes = 1500,
      .maximum_attempts = 2,
    },
    MakeEvaluator(MonotonicSize),
    &limited);
  if (!status.ok() || limited.codestream.size() != 1000 ||
      limited.attempt_count != 2 || limited.target_size_met) {
    std::cerr << "Target-size attempt limit was not honored\n";
    return false;
  }

  SearchResult plateau;
  status = gjxl::codestream_internal::SearchTargetSize(
    {
      .target_bytes = 900,
      .maximum_attempts = 12,
    },
    MakeEvaluator([](float) { return size_t{1000}; }),
    &plateau);
  if (!status.ok() || plateau.codestream.size() != 1000 ||
      plateau.attempt_count != 2 || plateau.target_size_met ||
      plateau.summary.selected_butteraugli_target !=
        gjxl::codestream_internal::kMinimumTargetSizeButteraugliTarget) {
    std::cerr << "Target-size plateau selection is invalid\n";
    return false;
  }
  return true;
}

bool CheckNonMonotonicSelection() {
  const auto size_for_target = [](float target) {
    if (target > 7.0f) {
      return size_t{1000};
    }
    if (target > 4.0f) {
      return size_t{1400};
    }
    if (target > 2.0f) {
      // Deliberate local reversal: this higher-quality point is smaller.
      return size_t{1300};
    }
    if (target > 1.0f) {
      return size_t{1500};
    }
    return size_t{2000};
  };
  SearchResult result;
  const gjxl::Status status =
    gjxl::codestream_internal::SearchTargetSize(
      {
        .target_bytes = 1500,
        .maximum_attempts = 8,
      },
      MakeEvaluator(size_for_target),
      &result);
  if (!status.ok() || result.codestream.size() != 1500 ||
      result.attempt_count != 5 || !result.target_size_met) {
    std::cerr << "Non-monotonic target-size selection failed\n";
    return false;
  }
  return true;
}

bool CheckAtomicFailures() {
  SearchResult original;
  original.codestream = {1, 2, 3};
  original.summary = MakeSummary(1.0f, 3, 1.0);
  original.attempt_count = 7;
  original.target_size_met = true;

  SearchResult failed = original;
  const Evaluator failure = [](
    float,
    std::vector<uint8_t>*,
    gjxl::VarDctEncodingSummary*) {
    return gjxl::Status::SubmissionFailed("injected search failure");
  };
  gjxl::Status status = gjxl::codestream_internal::SearchTargetSize(
    {
      .target_bytes = 1500,
      .maximum_attempts = 12,
    },
    failure,
    &failed);
  if (status.code() != gjxl::StatusCode::kSubmissionFailed ||
      failed != original) {
    std::cerr << "Target-size evaluator failure was not atomic\n";
    return false;
  }

  const Evaluator invalid_candidate = [](
    float target,
    std::vector<uint8_t>* codestream,
    gjxl::VarDctEncodingSummary* summary) {
    *codestream = {1, 2, 3};
    *summary = MakeSummary(target, 4, 1.0);
    return gjxl::Status::Ok();
  };
  failed = original;
  status = gjxl::codestream_internal::SearchTargetSize(
    {
      .target_bytes = 1500,
      .maximum_attempts = 12,
    },
    invalid_candidate,
    &failed);
  if (status.code() != gjxl::StatusCode::kInternal || failed != original) {
    std::cerr << "Invalid target-size candidate was not rejected atomically\n";
    return false;
  }
  return true;
}

bool CheckInvalidOptions() {
  const Evaluator evaluator = MakeEvaluator(MonotonicSize);
  SearchResult result;
  const auto rejected = [&](SearchOptions options) {
    SearchResult candidate = result;
    return gjxl::codestream_internal::SearchTargetSize(
      options, evaluator, &candidate).code() ==
        gjxl::StatusCode::kInvalidArgument && candidate == result;
  };

  if (!rejected({.target_bytes = 0, .maximum_attempts = 12}) ||
      !rejected({.target_bytes = 1000, .maximum_attempts = 0}) ||
      !rejected({
        .target_bytes = 1000,
        .maximum_attempts =
          gjxl::codestream_internal::kMaximumTargetSizeEncodeAttempts + 1,
      }) ||
      !rejected({
        .target_bytes = 1000,
        .maximum_attempts = 12,
        .minimum_butteraugli_target = 1.0f,
        .maximum_butteraugli_target = 1.0f,
      }) ||
      !rejected({
        .target_bytes = 1000,
        .maximum_attempts = 12,
        .minimum_butteraugli_target =
          std::numeric_limits<float>::quiet_NaN(),
      }) ||
      gjxl::codestream_internal::SearchTargetSize(
        {.target_bytes = 1000, .maximum_attempts = 12}, {}, &result).code() !=
          gjxl::StatusCode::kInvalidArgument ||
      gjxl::codestream_internal::SearchTargetSize(
        {.target_bytes = 1000, .maximum_attempts = 12}, evaluator, nullptr)
          .code() != gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "Invalid target-size search options were accepted\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckBracketedSearch() ||
      !CheckInfeasibleTargets() ||
      !CheckAttemptLimitAndPlateau() ||
      !CheckNonMonotonicSelection() ||
      !CheckAtomicFailures() ||
      !CheckInvalidOptions()) {
    return EXIT_FAILURE;
  }
  std::cout << "All target-size search tests passed.\n";
  return EXIT_SUCCESS;
}
