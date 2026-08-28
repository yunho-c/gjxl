// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstdint>

#include "gpu/ops/aq_evaluation.h"

namespace gjxl::metal_internal {

/// Diagnostic host/device timing for one production prepared evaluation.
struct MetalAqEvaluationProfile {
  uint64_t input_upload_bytes = 0;
  uint64_t input_upload_nanoseconds = 0;
  uint64_t submission_nanoseconds = 0;
  uint64_t completion_wait_nanoseconds = 0;
  uint64_t command_buffer_gpu_nanoseconds = 0;
  uint64_t bounded_readback_nanoseconds = 0;
  uint64_t final_readback_nanoseconds = 0;
  uint64_t output_commit_nanoseconds = 0;
};

[[nodiscard]] Status EvaluateMetalAqProfiled(
  PreparedAqEvaluation& prepared,
  AqEvaluationInput input,
  AqEvaluationOutput output,
  MetalAqEvaluationProfile* profile);

}  // namespace gjxl::metal_internal
