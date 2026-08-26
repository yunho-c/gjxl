// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>

#include "codec/adaptive_quantization.h"
#include "core/ac_strategy.h"
#include "core/image.h"
#include "core/status.h"

namespace gjxl {

struct CpuQuantizationPipelineOptions {
  float butteraugli_target = 1.0f;
  float initial_quant_rescale = 1.0f;
  std::array<float, 3> gaborish_inverse_multipliers = {
    1.0f, 1.0f, 1.0f};
  AdaptiveQuantizationOptions adaptive_quantization;
};

struct CpuQuantizationPipelineOutput {
  InitialQuantFieldOutput initial_quantization;
  AdaptiveQuantizationOutput adaptive_quantization;
  AcStrategyGrid* strategies = nullptr;
};

/// Runs the complete CPU quantization reference pipeline before GPU porting.
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
