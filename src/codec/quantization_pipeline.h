// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/ac_strategy.h"
#include "codec/adaptive_quantization.h"
#include "core/ac_strategy.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

struct CpuQuantizationPipelineOptions {
  float butteraugli_target = 1.0f;
  float initial_quant_rescale = 1.0f;
  AdaptiveQuantizationOptions adaptive_quantization;
};

struct CpuQuantizationPipelineOutput {
  InitialQuantFieldOutput initial_quantization;
  AdaptiveQuantizationOutput adaptive_quantization;
};

/// Supplies AC-strategy selection without coupling codec orchestration to a
/// particular CPU or GPU implementation.
class AcStrategySearchProvider {
public:
  virtual ~AcStrategySearchProvider() = default;

  [[nodiscard]] virtual Status Find(
    ConstImage3FView opsin,
    ConstPlaneF32View quant_field,
    ConstPlaneF32View pixel_mask,
    const ColorCorrelationMap& color_correlation,
    AcStrategySearchOptions options,
    AcStrategyGrid* out) = 0;

protected:
  AcStrategySearchProvider() = default;
};

/// Runs the complete pipeline using an injected AC-strategy implementation.
///
/// The provider is invoked exactly once after initial AQ, Gaborish, and
/// first-pass CfL. All caller-visible outputs remain atomic across later AQ.
[[nodiscard]] Status RunQuantizationPipeline(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  AcStrategySearchProvider& strategy_search,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output);

/// Runs the complete pipeline with the built-in CPU AC-strategy search.
///
/// `opsin` contains the padded, pre-Gaborish XYB input. Initial AQ samples it
/// directly; subsequent stages consume an internally inverse-filtered copy.
/// All caller-visible outputs are committed only after the pipeline succeeds.
[[nodiscard]] Status RunCpuQuantizationPipeline(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output);

}  // namespace gjxl
