// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/quantization_pipeline.h"

namespace gjxl::quantization_pipeline_internal {

class AdaptiveQuantizationProvider {
public:
  virtual ~AdaptiveQuantizationProvider() = default;

  [[nodiscard]] virtual Status Find(
    ConstImage3FView original_linear_rgb,
    ConstImage3FView opsin,
    const AcStrategyGrid& strategies,
    ConstPlaneF32View initial_quant_field,
    ConstPlaneU8View epf_sharpness,
    AdaptiveQuantizationOptions options,
    AdaptiveQuantizationOutput output) = 0;

protected:
  AdaptiveQuantizationProvider() = default;
};

[[nodiscard]] Status RunQuantizationPipelineWithProviders(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  AcStrategySearchProvider& strategy_search,
  AdaptiveQuantizationProvider& adaptive_quantization,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output);

}  // namespace gjxl::quantization_pipeline_internal
