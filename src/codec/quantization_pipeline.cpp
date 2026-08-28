// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/quantization_pipeline.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "codec/ac_strategy.h"
#include "codec/adaptive_quantization_internal.h"
#include "codec/chroma_from_luma.h"
#include "codec/chroma_from_luma_internal.h"
#include "codec/gaborish.h"
#include "codec/quantization_pipeline_internal.h"
#include "core/block_grid.h"
#include "core/geometry.h"
#include "core/image_buffer.h"
#include "core/image_ops.h"

namespace gjxl {
namespace {

class CpuGaborishInverseProvider final
    : public quantization_pipeline_internal::GaborishInverseProvider {
public:
  Status Apply(
    ConstImage3FView input,
    std::array<float, 3> multipliers,
    Image3FView output) override {

    return ApplyGaborishInverse(input, multipliers, output);
  }
};

class CpuAcStrategySearchProvider final : public AcStrategySearchProvider {
public:
  Status Find(
    ConstImage3FView opsin,
    ConstPlaneF32View quant_field,
    ConstPlaneF32View pixel_mask,
    const ColorCorrelationMap& color_correlation,
    AcStrategySearchOptions options,
    AcStrategyGrid* out) override {

    return FindAcStrategyGrid(
      opsin,
      quant_field,
      pixel_mask,
      color_correlation,
      options,
      out);
  }
};

class CpuAdaptiveQuantizationProvider final
    : public quantization_pipeline_internal::AdaptiveQuantizationProvider {
public:
  Status Find(
    ConstImage3FView original_linear_rgb,
    ConstImage3FView opsin,
    const AcStrategyGrid& strategies,
    ConstPlaneF32View initial_quant_field,
    ConstPlaneU8View epf_sharpness,
    AdaptiveQuantizationOptions options,
    PreparedButteraugliReference* prepared_reference,
    AdaptiveQuantizationOutput output) override {

    return adaptive_quantization_internal::FindBestQuantizationPrepared(
      original_linear_rgb, opsin, strategies, initial_quant_field,
      epf_sharpness, options, prepared_reference, output);
  }
};
Status ValidatePipelineInputs(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  const CpuQuantizationPipelineOutput& output,
  Extent2D* block_extent) {

  if (!original_linear_rgb.valid() ||
      !opsin.valid() ||
      !BlockGrid::IsPaddedPixelExtent(opsin.extent()) ||
      !std::isfinite(options.initial_quant_rescale) ||
      options.initial_quant_rescale <= 0.0f ||
      block_extent == nullptr) {
    return Status::InvalidArgument(
      "Quantization pipeline inputs or options are invalid");
  }
  if (!options.adaptive_quantization.profile.valid()) {
    return Status::InvalidArgument(
      "Quantization pipeline profile is invalid");
  }
  switch (options.adaptive_quantization.control_mode) {
    case AdaptiveQuantizationControlMode::kButteraugli:
      if (!std::isfinite(options.butteraugli_target) ||
          options.butteraugli_target <= 0.0f) {
        return Status::InvalidArgument(
          "Quantization pipeline Butteraugli target is invalid");
      }
      break;
    case AdaptiveQuantizationControlMode::kMaximumError:
      break;
    default:
      return Status::InvalidArgument(
        "Quantization pipeline control mode is invalid");
  }

  *block_extent =
    BlockGrid::FromPaddedPixelExtent(opsin.extent()).blocks;
  if (!output.initial_quantization.quant_field.valid() ||
      !output.initial_quantization.strategy_mask.valid() ||
      !output.initial_quantization.pixel_mask.valid() ||
      !output.adaptive_quantization.quant_field.valid() ||
      !output.adaptive_quantization.block_distance_map.valid() ||
      !output.adaptive_quantization.reconstructed_linear_rgb.valid() ||
      output.adaptive_quantization.frame == nullptr ||
      output.adaptive_quantization.score_history == nullptr ||
      (options.adaptive_quantization.control_mode ==
         AdaptiveQuantizationControlMode::kMaximumError &&
       output.adaptive_quantization.maximum_error_result == nullptr) ||
      output.initial_quantization.quant_field.extent != *block_extent ||
      output.initial_quantization.strategy_mask.extent != *block_extent ||
      output.initial_quantization.pixel_mask.extent != opsin.extent() ||
      output.adaptive_quantization.quant_field.extent != *block_extent ||
      output.adaptive_quantization.block_distance_map.extent != *block_extent) {
    return Status::InvalidArgument(
      "Quantization pipeline outputs have invalid geometry");
  }

  // FindBestQuantization validates the original/padded extent relationship
  // once the selected strategy grid is available.
  return Status::Ok();
}

}  // namespace

Status quantization_pipeline_internal::PrepareQuantizationPreprocessing(
  PreparedQuantizationPipeline& prepared,
  GaborishInverseProvider& gaborish_inverse,
  bool fast_initial_color_correlation) {

  if (!prepared.coding_opsin.const_view().valid() ||
      !prepared.preprocessed_opsin.const_view().valid() ||
      prepared.coding_opsin.extent() != prepared.padded_extent ||
      prepared.preprocessed_opsin.extent() != prepared.padded_extent ||
      !prepared.profile.valid()) {
    return Status::InvalidArgument(
      "Prepared quantization preprocessing is invalid");
  }
  Status status = Status::Ok();
  if (prepared.profile.loop_filter.gaborish) {
    status = gaborish_inverse.Apply(
      prepared.coding_opsin.const_view(),
      prepared.profile.gaborish_inverse_multipliers,
      prepared.preprocessed_opsin.view());
  } else {
    CopyImage(
      prepared.coding_opsin.const_view(), prepared.preprocessed_opsin.view());
  }
  if (!status.ok()) {
    return status;
  }
  status = fast_initial_color_correlation
    ? chroma_from_luma_internal::ComputeInitialColorCorrelationMapFast(
        prepared.preprocessed_opsin.const_view(),
        &prepared.initial_color_correlation)
    : ComputeInitialColorCorrelationMap(
        prepared.preprocessed_opsin.const_view(),
        &prepared.initial_color_correlation);
  if (!status.ok()) {
    return status;
  }
  prepared.preprocessing_ready = true;
  prepared.fast_initial_color_correlation =
    fast_initial_color_correlation;
  return Status::Ok();
}

Status quantization_pipeline_internal::PrepareQuantizationPipeline(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  PreparedQuantizationPipeline* prepared,
  bool prepare_cpu_butteraugli,
  bool prepare_cpu_preprocessing) {

  if (prepared == nullptr || !original_linear_rgb.valid() || !opsin.valid() ||
      !BlockGrid::IsPaddedPixelExtent(opsin.extent()) ||
      !options.adaptive_quantization.profile.valid() ||
      !std::isfinite(options.initial_quant_rescale) ||
      options.initial_quant_rescale <= 0.0f ||
      original_linear_rgb.width() > opsin.width() ||
      original_linear_rgb.height() > opsin.height() ||
      original_linear_rgb.width() <= opsin.width() - kJxlBlockDimension ||
      original_linear_rgb.height() <= opsin.height() - kJxlBlockDimension) {
    return Status::InvalidArgument(
      "Quantization pipeline preparation is invalid");
  }

  try {
    PreparedQuantizationPipeline candidate;
    candidate.source_extent = original_linear_rgb.extent();
    candidate.padded_extent = opsin.extent();
    candidate.block_extent =
      BlockGrid::FromPaddedPixelExtent(opsin.extent()).blocks;
    candidate.initial_quant_rescale = options.initial_quant_rescale;
    candidate.profile = options.adaptive_quantization.profile;
    size_t block_count = 0;
    size_t pixel_count = 0;
    if (!candidate.block_extent.try_area(&block_count) ||
        !candidate.padded_extent.try_area(&pixel_count)) {
      return Status::InvalidArgument(
        "Quantization pipeline dimensions are too large");
    }
    candidate.coding_opsin.resize(candidate.padded_extent);
    CopyImage(opsin, candidate.coding_opsin.view());
    candidate.preprocessed_opsin.resize(candidate.padded_extent);
    Status status = Status::Ok();
    if (prepare_cpu_preprocessing) {
      CpuGaborishInverseProvider gaborish_inverse;
      status = PrepareQuantizationPreprocessing(
        candidate, gaborish_inverse, false);
      if (!status.ok()) {
        return status;
      }
    }
    candidate.epf_sharpness.resize(block_count);
    status = FillDefaultEpfSharpness({
      candidate.epf_sharpness.data(), candidate.block_extent,
      candidate.block_extent.width});
    if (!status.ok()) {
      return status;
    }
    candidate.initial_quant.resize(block_count);
    candidate.strategy_mask.resize(block_count);
    candidate.pixel_mask.resize(pixel_count);
    candidate.final_quant.resize(block_count);
    candidate.block_distance.resize(block_count);
    candidate.reconstructed_linear.resize(candidate.source_extent);
    candidate.butteraugli_options = options.adaptive_quantization.butteraugli;
    if (prepare_cpu_butteraugli &&
        options.adaptive_quantization.control_mode ==
          AdaptiveQuantizationControlMode::kButteraugli) {
      auto reference = std::make_unique<PreparedButteraugliReference>();
      status = reference->Prepare(
        original_linear_rgb, candidate.butteraugli_options);
      if (!status.ok()) {
        return status;
      }
      candidate.butteraugli_reference = std::move(reference);
    }
    *prepared = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate quantization pipeline preparation");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Quantization pipeline preparation is too large");
  }
  return Status::Ok();
}

Status
quantization_pipeline_internal::RunPreparedQuantizationPipelineWithProviders(
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared,
  AcStrategySearchProvider& strategy_search,
  AdaptiveQuantizationProvider& adaptive_quantization,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output,
  bool initial_quantization_ready) {

  Extent2D block_extent;
  Status status = ValidatePipelineInputs(
    original_linear_rgb, prepared.preprocessed_opsin.const_view(), options,
    output, &block_extent);
  if (!status.ok()) {
    return status;
  }
  if (!prepared.preprocessing_ready ||
      prepared.source_extent != original_linear_rgb.extent() ||
      prepared.padded_extent != prepared.preprocessed_opsin.extent() ||
      prepared.block_extent != block_extent ||
      prepared.profile != options.adaptive_quantization.profile ||
      prepared.initial_quant_rescale != options.initial_quant_rescale) {
    return Status::InvalidArgument(
      "Prepared quantization pipeline does not match this attempt");
  }

  constexpr float kMaximumErrorInitializationTarget = 1.0f;
  const float control_target =
    options.adaptive_quantization.control_mode ==
        AdaptiveQuantizationControlMode::kMaximumError
      ? kMaximumErrorInitializationTarget
      : options.butteraugli_target;
  const float initial_quant_target = prepared.profile.loop_filter.gaborish
    ? control_target
    : 0.62f * control_target;
  if (!initial_quantization_ready) {
    status = ComputeInitialQuantField(
      prepared.coding_opsin.const_view(),
      {
        .butteraugli_target = initial_quant_target,
        .rescale = options.initial_quant_rescale,
      },
      {
        .quant_field = {
          prepared.initial_quant.data(), block_extent, block_extent.width},
        .strategy_mask = {
          prepared.strategy_mask.data(), block_extent, block_extent.width},
        .pixel_mask = {
          prepared.pixel_mask.data(), prepared.padded_extent,
          prepared.padded_extent.width},
      });
    if (!status.ok()) {
      return status;
    }
  }

  status = strategy_search.Find(
    prepared.preprocessed_opsin.const_view(),
    {prepared.initial_quant.data(), block_extent, block_extent.width},
    {prepared.pixel_mask.data(), prepared.padded_extent,
     prepared.padded_extent.width},
    prepared.initial_color_correlation,
    {.butteraugli_target = control_target},
    &prepared.strategies);
  if (!status.ok()) {
    return status;
  }

  AdaptiveQuantizationOptions adaptive_options =
    options.adaptive_quantization;
  adaptive_options.butteraugli_target = control_target;
  status = adaptive_quantization.Find(
    original_linear_rgb,
    prepared.preprocessed_opsin.const_view(),
    prepared.strategies,
    {prepared.initial_quant.data(), block_extent, block_extent.width},
    {prepared.epf_sharpness.data(), block_extent, block_extent.width},
    adaptive_options,
    prepared.butteraugli_reference.get(),
    {
      .quant_field = {
        prepared.final_quant.data(), block_extent, block_extent.width},
      .block_distance_map = {
        prepared.block_distance.data(), block_extent, block_extent.width},
      .reconstructed_linear_rgb = prepared.reconstructed_linear.view(),
      .frame = &prepared.frame,
      .score_history = &prepared.score_history,
      .maximum_error_result = &prepared.maximum_error_result,
    });
  if (!status.ok()) {
    return status;
  }

  CopyContiguousPlane(
    prepared.initial_quant, output.initial_quantization.quant_field);
  CopyContiguousPlane(
    prepared.strategy_mask, output.initial_quantization.strategy_mask);
  CopyContiguousPlane(
    prepared.pixel_mask, output.initial_quantization.pixel_mask);
  CopyContiguousPlane(
    prepared.final_quant, output.adaptive_quantization.quant_field);
  CopyContiguousPlane(
    prepared.block_distance,
    output.adaptive_quantization.block_distance_map);
  CopyImage(
    prepared.reconstructed_linear.const_view(),
    output.adaptive_quantization.reconstructed_linear_rgb);
  *output.adaptive_quantization.frame = std::move(prepared.frame);
  *output.adaptive_quantization.score_history = prepared.score_history;
  if (output.adaptive_quantization.maximum_error_result != nullptr) {
    *output.adaptive_quantization.maximum_error_result =
      prepared.maximum_error_result;
  }
  return Status::Ok();
}

Status quantization_pipeline_internal::RunQuantizationPipelineWithProviders(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  AcStrategySearchProvider& strategy_search,
  AdaptiveQuantizationProvider& adaptive_quantization,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output) {

  PreparedQuantizationPipeline prepared;
  Status status = PrepareQuantizationPipeline(
    original_linear_rgb, opsin, options, &prepared);
  if (!status.ok()) {
    return status;
  }
  return RunPreparedQuantizationPipelineWithProviders(
    original_linear_rgb, prepared, strategy_search, adaptive_quantization,
    options, output);
}

Status quantization_pipeline_internal::RunPreparedCpuQuantizationPipeline(
  ConstImage3FView original_linear_rgb,
  PreparedQuantizationPipeline& prepared,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output) {

  if (!prepared.preprocessing_ready ||
      prepared.fast_initial_color_correlation) {
    CpuGaborishInverseProvider gaborish_inverse;
    Status status = PrepareQuantizationPreprocessing(
      prepared, gaborish_inverse, false);
    if (!status.ok()) {
      return status;
    }
  }
  if (options.adaptive_quantization.control_mode ==
      AdaptiveQuantizationControlMode::kButteraugli &&
      (prepared.butteraugli_reference == nullptr ||
       prepared.butteraugli_options !=
         options.adaptive_quantization.butteraugli)) {
    try {
      auto reference = std::make_unique<PreparedButteraugliReference>();
      Status status = reference->Prepare(
        original_linear_rgb, options.adaptive_quantization.butteraugli);
      if (!status.ok()) {
        return status;
      }
      prepared.butteraugli_options =
        options.adaptive_quantization.butteraugli;
      prepared.butteraugli_reference = std::move(reference);
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate the prepared CPU Butteraugli reference");
    }
  }
  CpuAcStrategySearchProvider strategy_search;
  CpuAdaptiveQuantizationProvider adaptive_quantization;
  return RunPreparedQuantizationPipelineWithProviders(
    original_linear_rgb, prepared, strategy_search, adaptive_quantization,
    options, output);
}

Status RunQuantizationPipeline(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  AcStrategySearchProvider& strategy_search,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output) {

  CpuAdaptiveQuantizationProvider adaptive_quantization;
  return quantization_pipeline_internal::RunQuantizationPipelineWithProviders(
    original_linear_rgb, opsin, strategy_search, adaptive_quantization,
    options, output);
}

Status RunCpuQuantizationPipeline(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output) {

  CpuAcStrategySearchProvider strategy_search;
  return RunQuantizationPipeline(
    original_linear_rgb,
    opsin,
    strategy_search,
    options,
    output);
}

}  // namespace gjxl
