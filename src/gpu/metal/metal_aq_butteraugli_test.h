// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <vector>

#include "core/geometry.h"
#include "core/status.h"
#include "gpu/ops/aq_evaluation.h"

namespace gjxl::metal_internal {

struct MetalAqButteraugliSnapshotForTesting {
  Extent2D source_extent;
  std::vector<float> distance_map;
  double score = 0.0;
};

/// Runs resident reconstruction, postprocessing, and prepared Butteraugli.
/// Only the final map and scalar are read for this diagnostic.
[[nodiscard]] Status
RunMetalAqButteraugliForTesting(PreparedAqEvaluation &prepared,
                                AqEvaluationInput input,
                                MetalAqButteraugliSnapshotForTesting *snapshot);

/// Injects a failure after postprocessing and before/during Butteraugli.
[[nodiscard]] Status
FailNextMetalAqButteraugliForTesting(PreparedAqEvaluation &prepared,
                                     bool fail_submission, bool fail_completion,
                                     bool fail_readback);

} // namespace gjxl::metal_internal
