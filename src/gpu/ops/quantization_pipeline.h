// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/quantization_pipeline.h"
#include "gpu/backend.h"
#include "gpu/ops/ac_strategy_search.h"

namespace gjxl {

/// Runs the complete quantization pipeline with GPU AC search and prepared AQ.
///
/// Both optional GPU capabilities are required; this operation never silently
/// falls back to CPU AQ. Caller-visible pipeline outputs and optional search
/// statistics are committed only if the complete pipeline succeeds.
[[nodiscard]] Status RunGpuQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output,
  AcStrategyGpuSearchStats* stats = nullptr);

}  // namespace gjxl
