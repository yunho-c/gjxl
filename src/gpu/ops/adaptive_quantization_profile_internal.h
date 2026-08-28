// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "gpu/ops/adaptive_quantization.h"
#include "gpu/ops/quantization_pipeline_profile_internal.h"

namespace gjxl::adaptive_quantization_gpu_internal {

[[nodiscard]] Status RunGpuFrameOnlyQuantizationResidentFrontendProfiled(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneU8View epf_sharpness,
  InitialQuantizationOptions initial_options,
  AdaptiveQuantizationOptions options,
  InitialQuantFieldOutput initial_output,
  GpuFrameOnlyQuantizationOutput output,
  quantization_pipeline_internal::GpuFrameOnlyPipelineProfile* profile);

}  // namespace gjxl::adaptive_quantization_gpu_internal
