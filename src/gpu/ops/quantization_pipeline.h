// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codec/quantization_pipeline.h"
#include "gpu/backend.h"
#include "gpu/ops/ac_strategy_search.h"
#include "gpu/ops/adaptive_quantization.h"

namespace gjxl {

struct GpuFrameOnlyPipelineOutput {
  InitialQuantFieldOutput initial_quantization;
  PlaneF32View quant_field;
  VarDctEncoderFrame* frame = nullptr;
};

/// Builds a DCT8-only resident frame from the initial quant field without AC
/// search, inverse reconstruction, or perceptual scoring. This explicit path
/// is intended only for the maximum-throughput workflow policy.
[[nodiscard]] Status RunGpuFrameOnlyQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  GpuFrameOnlyPipelineOutput output);

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
/// Fully resident mode may produce different quantization decisions and
/// codestream bytes than the CPU reference.
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

struct GpuEncodingQuantizationPipelineOutput {
  VarDctEncoderFrame* frame = nullptr;
  std::vector<double>* score_history = nullptr;
  MaximumErrorResult* maximum_error_result = nullptr;
  bool collect_final_butteraugli_score = true;
};

/// Reuses maximum-throughput prepared device storage across compatible
/// rate-control attempts. This encoding-only entry point ignores diagnostic
/// plane outputs and commits only the frame.
[[nodiscard]] Status RunPreparedGpuFrameOnlyQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared_pipeline,
  CpuQuantizationPipelineOptions options,
  GpuFrameOnlyPipelineOutput output,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization* prepared);

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

/// Runs a Metal pipeline for codestream encoding without materializing
/// diagnostic quant fields, block maps, or reconstructed RGB.
[[nodiscard]] Status RunPreparedGpuQuantizationPipelineForEncoding(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared,
  CpuQuantizationPipelineOptions options,
  GpuAdaptiveQuantizationMode aq_mode,
  GpuEncodingQuantizationPipelineOutput output,
  AcStrategyGpuSearchStats* stats = nullptr,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
    prepared_aq = nullptr);

}  // namespace quantization_pipeline_internal

}  // namespace gjxl
