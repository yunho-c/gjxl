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
#include "codec/chroma_from_luma.h"
#include "codec/gaborish.h"
#include "core/block_grid.h"
#include "core/geometry.h"
#include "core/image_buffer.h"
#include "core/image_ops.h"

namespace gjxl {
namespace {

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
Status ValidatePipelineInputs(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  const CpuQuantizationPipelineOutput& output,
  Extent2D* block_extent) {

  if (!original_linear_rgb.valid() ||
      !opsin.valid() ||
      !BlockGrid::IsPaddedPixelExtent(opsin.extent()) ||
      !std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f ||
      !std::isfinite(options.initial_quant_rescale) ||
      options.initial_quant_rescale <= 0.0f ||
      block_extent == nullptr) {
    return Status::InvalidArgument(
      "Quantization pipeline inputs or options are invalid");
  }
  for (float multiplier : options.gaborish_inverse_multipliers) {
    if (!std::isfinite(multiplier)) {
      return Status::InvalidArgument(
        "Quantization pipeline Gaborish multiplier is invalid");
    }
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

Status RunQuantizationPipeline(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  AcStrategySearchProvider& strategy_search,
  CpuQuantizationPipelineOptions options,
  CpuQuantizationPipelineOutput output) {

  Extent2D block_extent;
  Status status = ValidatePipelineInputs(
    original_linear_rgb,
    opsin,
    options,
    output,
    &block_extent);
  if (!status.ok()) {
    return status;
  }

  size_t block_count = 0;
  size_t pixel_count = 0;
  if (!block_extent.try_area(&block_count) ||
      !opsin.extent().try_area(&pixel_count)) {
    return Status::InvalidArgument(
      "Quantization pipeline dimensions are too large");
  }

  try {
    std::vector<float> initial_quant(block_count);
    std::vector<float> strategy_mask(block_count);
    std::vector<float> pixel_mask(pixel_count);
    const float initial_quant_target =
      options.adaptive_quantization.loop_filter.gaborish
        ? options.butteraugli_target
        : 0.62f * options.butteraugli_target;
    status = ComputeInitialQuantField(
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
    if (!status.ok()) {
      return status;
    }

    Image3FBuffer preprocessed_opsin(opsin.extent());
    if (options.adaptive_quantization.loop_filter.gaborish) {
      status = ApplyGaborishInverse(
        opsin,
        options.gaborish_inverse_multipliers,
        preprocessed_opsin.view());
      if (!status.ok()) {
        return status;
      }
    } else {
      CopyImage(opsin, preprocessed_opsin.view());
    }

    ColorCorrelationMap initial_color_correlation;
    status = ComputeInitialColorCorrelationMap(
      preprocessed_opsin.const_view(),
      &initial_color_correlation);
    if (!status.ok()) {
      return status;
    }

    AcStrategyGrid strategies;
    status = strategy_search.Find(
      preprocessed_opsin.const_view(),
      {initial_quant.data(), block_extent, block_extent.width},
      {pixel_mask.data(), opsin.extent(), opsin.width()},
      initial_color_correlation,
      {.butteraugli_target = options.butteraugli_target},
      &strategies);
    if (!status.ok()) {
      return status;
    }

    std::vector<uint8_t> sharpness(block_count);
    status = FillDefaultEpfSharpness(
      {sharpness.data(), block_extent, block_extent.width});
    if (!status.ok()) {
      return status;
    }

    std::vector<float> final_quant(block_count);
    std::vector<float> block_distance(block_count);
    Image3FBuffer reconstructed_linear(original_linear_rgb.extent());
    VarDctEncoderFrame frame;
    std::vector<double> score_history;
    AdaptiveQuantizationOptions adaptive_options =
      options.adaptive_quantization;
    adaptive_options.butteraugli_target = options.butteraugli_target;
    status = FindBestQuantization(
      original_linear_rgb,
      preprocessed_opsin.const_view(),
      strategies,
      {initial_quant.data(), block_extent, block_extent.width},
      {sharpness.data(), block_extent, block_extent.width},
      adaptive_options,
      {
        .quant_field = {
          final_quant.data(), block_extent, block_extent.width},
        .block_distance_map = {
          block_distance.data(), block_extent, block_extent.width},
        .reconstructed_linear_rgb = reconstructed_linear.view(),
        .frame = &frame,
        .score_history = &score_history,
      });
    if (!status.ok()) {
      return status;
    }

    CopyContiguousPlane(
      initial_quant, output.initial_quantization.quant_field);
    CopyContiguousPlane(
      strategy_mask, output.initial_quantization.strategy_mask);
    CopyContiguousPlane(
      pixel_mask, output.initial_quantization.pixel_mask);
    CopyContiguousPlane(
      final_quant, output.adaptive_quantization.quant_field);
    CopyContiguousPlane(
      block_distance,
      output.adaptive_quantization.block_distance_map);
    CopyImage(
      reconstructed_linear.const_view(),
      output.adaptive_quantization.reconstructed_linear_rgb);
    *output.adaptive_quantization.frame = std::move(frame);
    *output.adaptive_quantization.score_history = std::move(score_history);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate quantization pipeline storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Quantization pipeline dimensions are too large");
  }

  return Status::Ok();
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
