// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "gpu/ops/gpu_execution_profile_internal.h"
#include "gpu/ops/quantization_pipeline.h"

namespace gjxl::quantization_pipeline_internal {

[[nodiscard]] Status RunPreparedGpuQuantizationPipelineForEncodingProfiled(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared,
  CpuQuantizationPipelineOptions options,
  GpuAdaptiveQuantizationMode aq_mode,
  GpuEncodingQuantizationPipelineOutput output,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
    prepared_aq,
  gpu_profile_internal::GpuProfilingMode profiling_mode,
  gpu_profile_internal::GpuExecutionProfile* profile);

}  // namespace gjxl::quantization_pipeline_internal
