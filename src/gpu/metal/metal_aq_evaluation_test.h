// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/geometry.h"
#include "gpu/ops/aq_evaluation.h"

namespace gjxl::metal_internal {

struct MetalAqReadbackStatsForTesting {
  size_t control_bytes = 0;
  size_t score_history_bytes = 0;
  size_t maximum_error_bytes = 0;
  size_t quantizer_bytes = 0;
  size_t quant_field_bytes = 0;
  size_t block_distance_map_bytes = 0;
  size_t frame_bytes = 0;
  size_t reconstructed_rgb_bytes = 0;

  [[nodiscard]] size_t total_bytes() const noexcept {
    return control_bytes + score_history_bytes + maximum_error_bytes +
      quantizer_bytes + quant_field_bytes + block_distance_map_bytes +
      frame_bytes + reconstructed_rgb_bytes;
  }
};

/// Leaves one production evaluation outstanding so tests can exercise
/// lifetime and non-reentrancy behavior. Finish must precede ordinary reuse.
[[nodiscard]] Status SubmitMetalAqEvaluationForTesting(
  PreparedAqEvaluation& prepared,
  AqEvaluationInput input);

[[nodiscard]] Status FinishMetalAqEvaluationForTesting(
  PreparedAqEvaluation& prepared,
  AqEvaluationOutput output);

/// Runs only the strategy-aware AQ reduction against a controlled pixel map.
[[nodiscard]] Status RunMetalAqBlockReductionForTesting(
  PreparedAqEvaluation& prepared,
  ConstPlaneF32View distance_map,
  PlaneF32View block_distance_map);

/// Injects a failure at the next production input upload boundary.
[[nodiscard]] Status FailNextMetalAqUploadForTesting(
  PreparedAqEvaluation& prepared);

/// Injects a device numeric flag into the next production submission.
[[nodiscard]] Status FailNextMetalAqNumericForTesting(
  PreparedAqEvaluation& prepared);

/// Injects a failure after successful GPU completion but before readback.
[[nodiscard]] Status FailNextMetalAqReadbackForTesting(
  PreparedAqEvaluation& prepared);

/// Injects an allocation failure before optional resident host staging.
[[nodiscard]] Status FailNextMetalAqResidentStagingForTesting(
  PreparedAqEvaluation& prepared);

/// Records whether this prepared object waits for an outstanding submission.
[[nodiscard]] Status SetMetalAqWaitObserverForTesting(
  PreparedAqEvaluation& prepared,
  bool* observed);

/// Returns the byte classes transferred by the last successful evaluation or
/// fused policy.
[[nodiscard]] Status GetMetalAqReadbackStatsForTesting(
  PreparedAqEvaluation& prepared,
  MetalAqReadbackStatsForTesting* stats);

/// Exercises checked geometry limits without requiring correspondingly large
/// host allocations.
[[nodiscard]] Status ValidateMetalAqGeometryForTesting(
  Extent2D source_extent,
  Extent2D coding_extent);

}  // namespace gjxl::metal_internal
