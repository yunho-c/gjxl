// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "gpu/ops/adaptive_quantization.h"
#include "gpu/ops/gpu_execution_profile_internal.h"

namespace gjxl::adaptive_quantization_gpu_internal {

[[nodiscard]] Status RunPreparedGpuAdaptiveQuantizationProfiled(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneF32View initial_quant_field,
  ConstPlaneU8View epf_sharpness,
  AdaptiveQuantizationOptions options,
  GpuAdaptiveQuantizationMode mode,
  PreparedAdaptiveQuantization* prepared,
  AdaptiveQuantizationOutput output,
  AdaptiveQuantizationMaterialization materialization,
  gpu_profile_internal::GpuProfilingMode profiling_mode,
  gpu_profile_internal::GpuExecutionProfile* profile);

}  // namespace gjxl::adaptive_quantization_gpu_internal
