// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/profile_timing_internal.h"
#include "gpu/ops/aq_evaluation_profile_internal.h"
#include "gpu/ops/quantization_pipeline.h"

namespace gjxl::quantization_pipeline_internal {

struct GpuFrameOnlyPipelineProfile {
  profile_internal::HostInterval initial_field_preparation;
  profile_internal::HostInterval quantization_parameter_preparation;
  profile_internal::HostInterval prepared_evaluation_setup;
  gpu_profile_internal::FrameEncodingProfile initial_quantization;
  gpu_profile_internal::FrameEncodingProfile frame_encoding;
  profile_internal::HostInterval output_commit;
};

[[nodiscard]] Status RunGpuFrameOnlyQuantizationPipelineProfiled(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  GpuFrameOnlyPipelineOutput output,
  GpuFrameOnlyPipelineProfile* profile);

}  // namespace gjxl::quantization_pipeline_internal
