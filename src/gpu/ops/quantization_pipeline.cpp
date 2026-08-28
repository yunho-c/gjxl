// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/ops/quantization_pipeline.h"

#include <cmath>
#include <new>
#include <stdexcept>
#include <vector>

#include "codec/chroma_from_luma_internal.h"
#include "codec/gaborish.h"
#include "codec/quantization_pipeline_internal.h"
#include "core/block_grid.h"
#include "core/image_ops.h"
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

Status RunGpuFrameOnlyQuantizationPipeline(
  GpuBackend& gpu,
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  GpuFrameOnlyPipelineOutput output) {

  if (!original_linear_rgb.valid() || !opsin.valid() ||
      !BlockGrid::IsPaddedPixelExtent(opsin.extent()) ||
      !std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f ||
      !std::isfinite(options.initial_quant_rescale) ||
      options.initial_quant_rescale <= 0.0f ||
      !options.adaptive_quantization.profile.valid() ||
      output.frame == nullptr) {
    return Status::InvalidArgument(
      "GPU frame-only pipeline inputs or options are invalid");
  }
  const Extent2D block_extent =
    BlockGrid::FromPaddedPixelExtent(opsin.extent()).blocks;
  if (!output.initial_quantization.quant_field.valid() ||
      output.initial_quantization.quant_field.extent != block_extent ||
      !output.initial_quantization.strategy_mask.valid() ||
      output.initial_quantization.strategy_mask.extent != block_extent ||
      !output.initial_quantization.pixel_mask.valid() ||
      output.initial_quantization.pixel_mask.extent != opsin.extent() ||
      !output.quant_field.valid() || output.quant_field.extent != block_extent) {
    return Status::InvalidArgument(
      "GPU frame-only pipeline output geometry is invalid");
  }

  size_t block_count = 0;
  size_t pixel_count = 0;
  if (!block_extent.try_area(&block_count) ||
      !opsin.extent().try_area(&pixel_count)) {
    return Status::InvalidArgument(
      "GPU frame-only pipeline dimensions are too large");
  }
  try {
    std::vector<float> initial_quant(block_count);
    std::vector<float> strategy_mask(block_count);
    std::vector<float> pixel_mask(pixel_count);
    const float initial_quant_target =
      options.adaptive_quantization.profile.loop_filter.gaborish
        ? options.butteraugli_target
        : 0.62f * options.butteraugli_target;
    Status status = ComputeInitialQuantField(
      opsin,
      {
        .butteraugli_target = initial_quant_target,
        .rescale = options.initial_quant_rescale,
      },
      {
        .quant_field = {
          initial_quant.data(), block_extent, block_extent.width},
        .strategy_mask = {
          strategy_mask.data(), block_extent, block_extent.width},
        .pixel_mask = {
          pixel_mask.data(), opsin.extent(), opsin.width()},
      });
    if (!status.ok()) return status;

    AcStrategyGrid strategies;
    status = AcStrategyGrid::Create(block_extent, &strategies);
    if (!status.ok()) return status;
    strategies.fill_dct8();
    std::vector<uint8_t> sharpness(block_count);
    status = FillDefaultEpfSharpness(
      {sharpness.data(), block_extent, block_extent.width});
    if (!status.ok()) return status;
    std::vector<float> final_quant(block_count);
    VarDctEncoderFrame frame;
    AdaptiveQuantizationOptions adaptive_options =
      options.adaptive_quantization;
    adaptive_options.butteraugli_target = options.butteraugli_target;
    status = RunGpuFrameOnlyQuantizationResidentInitialCfl(
      gpu, original_linear_rgb, opsin, strategies,
      {initial_quant.data(), block_extent, block_extent.width},
      {sharpness.data(), block_extent, block_extent.width}, adaptive_options,
      {
        .quant_field = {
          final_quant.data(), block_extent, block_extent.width},
        .frame = &frame,
      });
    if (!status.ok()) return status;

    CopyContiguousPlane(
      initial_quant, output.initial_quantization.quant_field);
    CopyContiguousPlane(
      strategy_mask, output.initial_quantization.strategy_mask);
    CopyContiguousPlane(pixel_mask, output.initial_quantization.pixel_mask);
    CopyContiguousPlane(final_quant, output.quant_field);
    *output.frame = std::move(frame);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate GPU frame-only pipeline storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "GPU frame-only pipeline dimensions are too large");
  }
}

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
    case GpuAdaptiveQuantizationMode::kThroughput:
      break;
    default:
      return Status::InvalidArgument(
        "GPU quantization pipeline AQ mode is invalid");
  }
  if (QueryGpuAqEvaluation(gpu) == nullptr) {
    return Status::Unavailable(
      "GPU quantization pipeline requires prepared AQ support");
  }
  const bool resident =
    aq_mode != GpuAdaptiveQuantizationMode::kExactCoefficients;
  quantization_pipeline_internal::PreparedQuantizationPipeline prepared;
  Status status = quantization_pipeline_internal::PrepareQuantizationPipeline(
    original_linear_rgb, opsin, options, &prepared, false,
    !resident);
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
    case GpuAdaptiveQuantizationMode::kThroughput:
      break;
    default:
      return Status::InvalidArgument(
        "GPU quantization pipeline AQ mode is invalid");
  }
  if (QueryGpuAqEvaluation(gpu) == nullptr) {
    return Status::Unavailable(
      "GPU quantization pipeline requires prepared AQ support");
  }
  const bool resident =
    aq_mode != GpuAdaptiveQuantizationMode::kExactCoefficients;
  if (aq_mode == GpuAdaptiveQuantizationMode::kThroughput) {
    options.adaptive_quantization.iterations = 1;
  }
  if (!prepared.preprocessing_ready ||
      prepared.fast_initial_color_correlation != resident) {
    GpuPipelineGaborishProvider gaborish_inverse(gpu, resident);
    Status status = PrepareQuantizationPreprocessing(
      prepared, gaborish_inverse, resident);
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
