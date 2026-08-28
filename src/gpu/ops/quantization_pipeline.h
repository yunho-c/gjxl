// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/quantization_pipeline.h"
#include "gpu/backend.h"
#include "gpu/ops/ac_strategy_search.h"
#include "gpu/ops/adaptive_quantization.h"

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

/// Runs the complete GPU pipeline with an explicit AQ evaluation mode.
/// Fully resident mode is experimental and may produce different quantization
/// decisions and codestream bytes than the CPU reference.
[[nodiscard]] Status RunGpuQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  GpuAdaptiveQuantizationMode aq_mode,
  CpuQuantizationPipelineOutput output,
  AcStrategyGpuSearchStats* stats = nullptr);

namespace quantization_pipeline_internal {

struct PreparedQuantizationPipeline;

/// Reuses target-invariant host preparation across complete GPU attempts.
[[nodiscard]] Status RunPreparedGpuQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared,
  CpuQuantizationPipelineOptions options,
  GpuAdaptiveQuantizationMode aq_mode,
  CpuQuantizationPipelineOutput output,
  AcStrategyGpuSearchStats* stats = nullptr,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
    prepared_aq = nullptr);

}  // namespace quantization_pipeline_internal

}  // namespace gjxl
