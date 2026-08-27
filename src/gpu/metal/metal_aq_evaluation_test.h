// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/geometry.h"
#include "gpu/ops/aq_evaluation.h"

namespace gjxl::metal_internal {

/// Runs the transitional Milestone 2 upload/dispatch/readback contract probe.
/// This is deliberately not part of the production AQ operation.
[[nodiscard]] Status RunMetalAqContractProbeForTesting(
  PreparedAqEvaluation& prepared,
  AqEvaluationInput input,
  AqEvaluationOutput output);

/// Leaves one probe submission outstanding so tests can exercise lifetime and
/// non-reentrancy behavior. Finish must be called before ordinary reuse.
[[nodiscard]] Status SubmitMetalAqContractProbeForTesting(
  PreparedAqEvaluation& prepared,
  AqEvaluationInput input);

[[nodiscard]] Status FinishMetalAqContractProbeForTesting(
  PreparedAqEvaluation& prepared,
  AqEvaluationOutput output);

/// Injects a failure after successful GPU completion but before readback.
[[nodiscard]] Status FailNextMetalAqReadbackForTesting(
  PreparedAqEvaluation& prepared);

/// Records whether this prepared object waits for an outstanding submission.
[[nodiscard]] Status SetMetalAqWaitObserverForTesting(
  PreparedAqEvaluation& prepared,
  bool* observed);

/// Exercises checked geometry limits without requiring correspondingly large
/// host allocations.
[[nodiscard]] Status ValidateMetalAqGeometryForTesting(
  Extent2D source_extent,
  Extent2D coding_extent);

}  // namespace gjxl::metal_internal
