// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "codestream/workflow.h"
#include "core/status.h"

namespace gjxl::codestream_internal {

inline constexpr float kMinimumTargetSizeButteraugliTarget = 0.01f;
inline constexpr float kMaximumTargetSizeButteraugliTarget = 10.0f;
inline constexpr size_t kMaximumTargetSizeEncodeAttempts = 64;

struct TargetSizeSearchOptions {
  size_t target_bytes = 0;
  size_t tolerance_bytes = 0;
  size_t maximum_attempts = 0;
  TargetSizeSelectionPolicy selection =
    TargetSizeSelectionPolicy::kLargestAtOrBelow;
  float minimum_butteraugli_target =
    kMinimumTargetSizeButteraugliTarget;
  float maximum_butteraugli_target =
    kMaximumTargetSizeButteraugliTarget;
};

struct TargetSizeSearchResult {
  std::vector<uint8_t> codestream;
  VarDctEncodingSummary summary;
  size_t attempt_count = 0;
  size_t failed_attempt_count = 0;
  bool target_size_met = false;
  bool search_exhausted = false;

  friend bool operator==(
    const TargetSizeSearchResult&,
    const TargetSizeSearchResult&) = default;
};

using TargetSizeEvaluator = std::function<Status(
  float butteraugli_target,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary)>;

/// Performs a bounded target-size search over complete encodes controlled by a
/// Butteraugli-target-like scalar. Score history is optional; when present its
/// final value is an equal-size tie-break. Failure leaves `result` unchanged.
[[nodiscard]] Status SearchTargetSize(
  const TargetSizeSearchOptions& options,
  const TargetSizeEvaluator& evaluator,
  TargetSizeSearchResult* result);

}  // namespace gjxl::codestream_internal
