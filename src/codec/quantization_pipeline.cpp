// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/quantization_pipeline.h"

#include <algorithm>
#include <array>
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
#include "core/geometry.h"

namespace gjxl {
namespace {

struct OwnedImage3F {
  Extent2D extent;
  std::array<std::vector<float>, 3> plane;

  [[nodiscard]] Image3FView View() {
    return {{
      PlaneF32View{plane[0].data(), extent, extent.width},
      PlaneF32View{plane[1].data(), extent, extent.width},
      PlaneF32View{plane[2].data(), extent, extent.width},
    }};
  }

  [[nodiscard]] ConstImage3FView ConstView() const {
    return {{
      ConstPlaneF32View{plane[0].data(), extent, extent.width},
      ConstPlaneF32View{plane[1].data(), extent, extent.width},
      ConstPlaneF32View{plane[2].data(), extent, extent.width},
    }};
  }
};

Status AllocateImage(Extent2D extent, OwnedImage3F* image) {
  size_t count = 0;
  if (image == nullptr || extent.empty() || !extent.try_area(&count)) {
    return Status::InvalidArgument(
      "CPU quantization pipeline image extent is invalid");
  }
  image->extent = extent;
  for (std::vector<float>& plane : image->plane) {
    plane.resize(count);
  }
  return Status::Ok();
}

template <typename T>
void CopyPlane(
  const std::vector<T>& source,
  PlaneView<T> destination) {

  for (size_t y = 0; y < destination.extent.height; ++y) {
    std::copy_n(
      source.data() + y * destination.extent.width,
      destination.extent.width,
      destination.Row(y));
  }
}

void CopyImage(const OwnedImage3F& source, Image3FView destination) {
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t y = 0; y < destination.height(); ++y) {
      std::copy_n(
        source.plane[channel].data() + y * source.extent.width,
        destination.width(),
        destination.plane[channel].Row(y));
    }
  }
}

Status ValidatePipelineInputs(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
  CpuQuantizationPipelineOptions options,
  const CpuQuantizationPipelineOutput& output,
  Extent2D* block_extent) {

  if (!original_linear_rgb.valid() ||
      !opsin.valid() ||
      opsin.width() % kJxlBlockDimension != 0 ||
      opsin.height() % kJxlBlockDimension != 0 ||
      !std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f ||
      !std::isfinite(options.initial_quant_rescale) ||
      options.initial_quant_rescale <= 0.0f ||
      block_extent == nullptr ||
      output.strategies == nullptr) {
    return Status::InvalidArgument(
      "CPU quantization pipeline inputs or options are invalid");
  }
  for (float multiplier : options.gaborish_inverse_multipliers) {
    if (!std::isfinite(multiplier)) {
      return Status::InvalidArgument(
        "CPU quantization pipeline Gaborish multiplier is invalid");
    }
  }

  *block_extent = {
    opsin.width() / kJxlBlockDimension,
    opsin.height() / kJxlBlockDimension,
  };
  if (!output.initial_quantization.quant_field.valid() ||
      !output.initial_quantization.strategy_mask.valid() ||
      !output.initial_quantization.pixel_mask.valid() ||
      output.initial_quantization.quant_field.extent != *block_extent ||
      output.initial_quantization.strategy_mask.extent != *block_extent ||
      output.initial_quantization.pixel_mask.extent != opsin.extent() ||
      output.adaptive_quantization.quant_field.extent != *block_extent ||
      output.adaptive_quantization.raw_quant_field.extent != *block_extent ||
      output.adaptive_quantization.block_distance_map.extent != *block_extent) {
    return Status::InvalidArgument(
      "CPU quantization pipeline outputs have invalid geometry");
  }

  // Reuse the full AQ validator for the original/padded extent relationship
  // and the remaining nested output contract after strategies are available.
  return Status::Ok();
}

}  // namespace

Status RunCpuQuantizationPipeline(
  ConstImage3FView original_linear_rgb,
  ConstImage3FView opsin,
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
      "CPU quantization pipeline dimensions are too large");
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

    OwnedImage3F preprocessed_opsin;
    status = AllocateImage(opsin.extent(), &preprocessed_opsin);
    if (!status.ok()) {
      return status;
    }
    if (options.adaptive_quantization.loop_filter.gaborish) {
      status = ApplyGaborishInverse(
        opsin,
        options.gaborish_inverse_multipliers,
        preprocessed_opsin.View());
      if (!status.ok()) {
        return status;
      }
    } else {
      for (size_t channel = 0; channel < 3; ++channel) {
        for (size_t y = 0; y < opsin.height(); ++y) {
          std::copy_n(
            opsin.plane[channel].Row(y),
            opsin.width(),
            preprocessed_opsin.View().plane[channel].Row(y));
        }
      }
    }

    ColorCorrelationMap initial_color_correlation;
    status = ComputeInitialColorCorrelationMap(
      preprocessed_opsin.ConstView(),
      &initial_color_correlation);
    if (!status.ok()) {
      return status;
    }

    AcStrategyGrid strategies;
    status = FindAcStrategyGrid(
      preprocessed_opsin.ConstView(),
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
    std::vector<int32_t> raw_quant(block_count);
    std::vector<float> block_distance(block_count);
    OwnedImage3F reconstructed_linear;
    status = AllocateImage(
      original_linear_rgb.extent(),
      &reconstructed_linear);
    if (!status.ok()) {
      return status;
    }
    Quantizer quantizer;
    ColorCorrelationMap final_color_correlation;
    std::vector<double> score_history;
    AdaptiveQuantizationOptions adaptive_options =
      options.adaptive_quantization;
    adaptive_options.butteraugli_target = options.butteraugli_target;
    status = FindBestQuantization(
      original_linear_rgb,
      preprocessed_opsin.ConstView(),
      strategies,
      {initial_quant.data(), block_extent, block_extent.width},
      {sharpness.data(), block_extent, block_extent.width},
      adaptive_options,
      {
        .quant_field = {
          final_quant.data(), block_extent, block_extent.width},
        .raw_quant_field = {
          raw_quant.data(), block_extent, block_extent.width},
        .block_distance_map = {
          block_distance.data(), block_extent, block_extent.width},
        .reconstructed_linear_rgb = reconstructed_linear.View(),
        .quantizer = &quantizer,
        .color_correlation = &final_color_correlation,
        .score_history = &score_history,
      });
    if (!status.ok()) {
      return status;
    }

    CopyPlane(initial_quant, output.initial_quantization.quant_field);
    CopyPlane(strategy_mask, output.initial_quantization.strategy_mask);
    CopyPlane(pixel_mask, output.initial_quantization.pixel_mask);
    CopyPlane(final_quant, output.adaptive_quantization.quant_field);
    CopyPlane(raw_quant, output.adaptive_quantization.raw_quant_field);
    CopyPlane(
      block_distance,
      output.adaptive_quantization.block_distance_map);
    CopyImage(
      reconstructed_linear,
      output.adaptive_quantization.reconstructed_linear_rgb);
    *output.adaptive_quantization.quantizer = quantizer;
    *output.adaptive_quantization.color_correlation =
      std::move(final_color_correlation);
    *output.adaptive_quantization.score_history = std::move(score_history);
    *output.strategies = std::move(strategies);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate CPU quantization pipeline storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "CPU quantization pipeline dimensions are too large");
  }

  return Status::Ok();
}

}  // namespace gjxl
