// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/ops/quantization_pipeline.h"

#include "codec/gaborish.h"
#include "codec/quantization_pipeline_internal.h"
#include "gpu/ops/adaptive_quantization.h"
#include "gpu/ops/aq_evaluation.h"
#include "gpu/ops/gaborish.h"

namespace gjxl {
namespace {

class GpuPipelineGaborishProvider final
    : public quantization_pipeline_internal::GaborishInverseProvider {
public:
  GpuPipelineGaborishProvider(GpuBackend& gpu, bool use_gpu)
    : gpu_(gpu), use_gpu_(use_gpu) {}

  Status Apply(
    ConstImage3FView input,
    std::array<float, 3> multipliers,
    Image3FView output) override {

    return use_gpu_
      ? ApplyGaborishInverseGpu(gpu_, input, multipliers, output)
      : ApplyGaborishInverse(input, multipliers, output);
  }

private:
  GpuBackend& gpu_;
  bool use_gpu_;
};

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
  GpuAdaptiveQuantizationProvider(
    GpuBackend& gpu,
    GpuAdaptiveQuantizationMode mode,
    adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
      prepared)
    : gpu_(gpu), mode_(mode), prepared_(prepared) {}

  Status Find(
    ConstImage3FView original_linear_rgb,
    ConstImage3FView opsin,
    const AcStrategyGrid& strategies,
    ConstPlaneF32View initial_quant_field,
    ConstPlaneU8View epf_sharpness,
    AdaptiveQuantizationOptions options,
    PreparedButteraugliReference*,
    AdaptiveQuantizationOutput output) override {

    if (prepared_ != nullptr) {
      return adaptive_quantization_gpu_internal::
        RunPreparedGpuAdaptiveQuantization(
          gpu_, original_linear_rgb, opsin, strategies,
          initial_quant_field, epf_sharpness, options, mode_, prepared_,
          output);
    }
    return RunGpuAdaptiveQuantization(
      gpu_, original_linear_rgb, opsin, strategies, initial_quant_field,
      epf_sharpness, options, mode_, output);
  }

private:
  GpuBackend& gpu_;
  GpuAdaptiveQuantizationMode mode_;
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
    prepared_ = nullptr;
};

}  // namespace

Status RunGpuQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output,
  AcStrategyGpuSearchStats* stats) {

  return RunGpuQuantizationPipeline(
    gpu, original_linear_rgb, opsin, options,
    GpuAdaptiveQuantizationMode::kExactCoefficients, output, stats);
}

Status RunGpuQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  GpuAdaptiveQuantizationMode aq_mode,
  CpuQuantizationPipelineOutput output,
  AcStrategyGpuSearchStats* stats) {

  switch (aq_mode) {
    case GpuAdaptiveQuantizationMode::kExactCoefficients:
    case GpuAdaptiveQuantizationMode::kFullyResident:
      break;
    default:
      return Status::InvalidArgument(
        "GPU quantization pipeline AQ mode is invalid");
  }
  if (QueryGpuAqEvaluation(gpu) == nullptr) {
    return Status::Unavailable(
      "GPU quantization pipeline requires prepared AQ support");
  }
  const bool fully_resident =
    aq_mode == GpuAdaptiveQuantizationMode::kFullyResident;
  quantization_pipeline_internal::PreparedQuantizationPipeline prepared;
  Status status = quantization_pipeline_internal::PrepareQuantizationPipeline(
    original_linear_rgb, opsin, options, &prepared, false,
    !fully_resident);
  if (!status.ok()) {
    return status;
  }
  return quantization_pipeline_internal::RunPreparedGpuQuantizationPipeline(
    gpu, original_linear_rgb, prepared, options, aq_mode, output, stats);
}

Status quantization_pipeline_internal::RunPreparedGpuQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared,
  CpuQuantizationPipelineOptions options,
  GpuAdaptiveQuantizationMode aq_mode,
  CpuQuantizationPipelineOutput output,
  AcStrategyGpuSearchStats* stats,
  adaptive_quantization_gpu_internal::PreparedAdaptiveQuantization*
    prepared_aq) {

  switch (aq_mode) {
    case GpuAdaptiveQuantizationMode::kExactCoefficients:
    case GpuAdaptiveQuantizationMode::kFullyResident:
      break;
    default:
      return Status::InvalidArgument(
        "GPU quantization pipeline AQ mode is invalid");
  }
  if (QueryGpuAqEvaluation(gpu) == nullptr) {
    return Status::Unavailable(
      "GPU quantization pipeline requires prepared AQ support");
  }
  const bool fully_resident =
    aq_mode == GpuAdaptiveQuantizationMode::kFullyResident;
  if (!prepared.preprocessing_ready ||
      prepared.fast_initial_color_correlation != fully_resident) {
    GpuPipelineGaborishProvider gaborish_inverse(gpu, fully_resident);
    Status status = PrepareQuantizationPreprocessing(
      prepared, gaborish_inverse, fully_resident);
    if (!status.ok()) {
      return status;
    }
  }
  GpuAcStrategySearchProvider strategy_search(gpu);
  GpuAdaptiveQuantizationProvider adaptive_quantization(
    gpu, aq_mode, prepared_aq);
  const Status status = RunPreparedQuantizationPipelineWithProviders(
    original_linear_rgb, prepared, strategy_search, adaptive_quantization,
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
