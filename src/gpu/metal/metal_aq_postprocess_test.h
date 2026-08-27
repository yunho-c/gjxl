// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu/ops/aq_evaluation.h"

namespace gjxl::metal_internal {

struct MetalAqPostprocessSnapshotForTesting {
  Extent2D coding_extent;
  Extent2D source_extent;
  std::array<std::vector<float>, 3> reconstructed_opsin;
  std::array<std::vector<float>, 3> filtered_opsin;
  std::array<std::vector<float>, 3> reconstructed_linear;
};

struct MetalAqPostprocessPlanForTesting {
  size_t filter_scratch_images = 0;
  uint32_t gaborish_dispatches = 0;
  uint32_t epf_dispatches = 0;
  uint32_t color_dispatches = 0;
  uint32_t copy_dispatches = 0;
};

/// Uploads controlled reconstructed opsin and executes only Milestone 4.
[[nodiscard]] Status
RunMetalAqPostprocessForTesting(PreparedAqEvaluation &prepared,
                                ConstImage3FView reconstructed_opsin,
                                ConstPlaneF32View epf_inverse_sigma,
                                MetalAqPostprocessSnapshotForTesting *snapshot);

/// Executes Milestones 3 and 4 in one submission with no intermediate readback.
[[nodiscard]] Status RunMetalAqReconstructionAndPostprocessForTesting(
    PreparedAqEvaluation &prepared, AqEvaluationInput input,
    MetalAqPostprocessSnapshotForTesting *snapshot);

/// Returns the immutable filter routing selected during preparation.
[[nodiscard]] Status
GetMetalAqPostprocessPlanForTesting(PreparedAqEvaluation &prepared,
                                    MetalAqPostprocessPlanForTesting *plan);

} // namespace gjxl::metal_internal
