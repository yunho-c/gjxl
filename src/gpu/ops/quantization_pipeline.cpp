// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/ops/quantization_pipeline.h"

#include "codec/quantization_pipeline_internal.h"
#include "gpu/ops/adaptive_quantization.h"
#include "gpu/ops/aq_evaluation.h"

namespace gjxl {
namespace {

class GpuAcStrategySearchProvider final : public AcStrategySearchProvider {
public:
  explicit GpuAcStrategySearchProvider(GpuBackend& gpu)
    : gpu_(gpu) {}

  Status Find(
    ConstImage3FView opsin,
    ConstPlaneF32View quant_field,
    ConstPlaneF32View pixel_mask,
    const ColorCorrelationMap& color_correlation,
    AcStrategySearchOptions options,
    AcStrategyGrid* out) override {

    return FindAcStrategyGridGpu(
      gpu_,
      opsin,
      quant_field,
      pixel_mask,
      color_correlation,
      options,
      out,
      &stats_);
  }

  [[nodiscard]] const AcStrategyGpuSearchStats& stats() const noexcept {
    return stats_;
  }

private:
  GpuBackend& gpu_;
  AcStrategyGpuSearchStats stats_;
};

class GpuAdaptiveQuantizationProvider final
    : public quantization_pipeline_internal::AdaptiveQuantizationProvider {
public:
  explicit GpuAdaptiveQuantizationProvider(GpuBackend& gpu) : gpu_(gpu) {}

  Status Find(
    ConstImage3FView original_linear_rgb,
    ConstImage3FView opsin,
    const AcStrategyGrid& strategies,
    ConstPlaneF32View initial_quant_field,
    ConstPlaneU8View epf_sharpness,
    AdaptiveQuantizationOptions options,
    AdaptiveQuantizationOutput output) override {

    return RunGpuAdaptiveQuantization(
      gpu_, original_linear_rgb, opsin, strategies, initial_quant_field,
      epf_sharpness, options, output);
  }

private:
  GpuBackend& gpu_;
};

}  // namespace

Status RunGpuQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output,
  AcStrategyGpuSearchStats* stats) {

  if (QueryGpuAqEvaluation(gpu) == nullptr) {
    return Status::Unavailable(
      "GPU quantization pipeline requires prepared AQ support");
  }
  GpuAcStrategySearchProvider strategy_search(gpu);
  GpuAdaptiveQuantizationProvider adaptive_quantization(gpu);
  const Status status =
    quantization_pipeline_internal::RunQuantizationPipelineWithProviders(
      original_linear_rgb, opsin, strategy_search, adaptive_quantization,
      options, output);
  if (!status.ok()) {
    return status;
  }
  if (stats != nullptr) {
    *stats = strategy_search.stats();
  }
  return Status::Ok();
}

}  // namespace gjxl
