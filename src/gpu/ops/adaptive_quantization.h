// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <vector>

#include "codec/adaptive_quantization.h"
#include "gpu/backend.h"

namespace gjxl {

struct GpuAdaptiveQuantizationPolicyOutput {
  PlaneF32View quant_field;
  PlaneF32View block_distance_map;
  std::vector<double>* score_history = nullptr;
};

/// Runs the bounded adaptive-quantization policy with a prepared GPU evaluator.
///
/// The optional AQ capability is required; this operation never silently falls
/// back to CPU evaluation. Caller-visible output is committed only after every
/// requested evaluation succeeds.
[[nodiscard]] Status RunGpuAdaptiveQuantizationPolicy(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuAdaptiveQuantizationPolicyOutput output);

/// Runs GPU adaptive quantization and materializes the final resident
/// reconstruction and encoder frame from the last evaluation.
///
/// Prepared AQ support is required; this operation never silently falls back
/// to CPU evaluation. All caller-visible outputs are committed atomically.
[[nodiscard]] Status RunGpuAdaptiveQuantization(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  AdaptiveQuantizationOutput output);

}  // namespace gjxl
