// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/workflow.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "codec/color_transform.h"
#include "codec/quantization_pipeline.h"
#include "codec/vardct_frame.h"
#include "codestream/encoder.h"
#include "core/frame_geometry.h"
#include "core/image_buffer.h"

namespace gjxl {
namespace {

constexpr float kInitialProfileIntensityTarget = 255.0f;

struct PipelineStorage {
  PipelineStorage(Extent2D frame_extent, Extent2D padded_extent)
      : block_extent{
          padded_extent.width / kJxlBlockDimension,
          padded_extent.height / kJxlBlockDimension},
        padded_extent(padded_extent),
        initial_quant(BlockCount()),
        strategy_mask(BlockCount()),
        pixel_mask(PixelCount(padded_extent)),
        final_quant(BlockCount()),
        block_distance(BlockCount()),
        reconstructed(frame_extent) {}

  [[nodiscard]] CpuQuantizationPipelineOutput Output() {
    return {
      .initial_quantization = {
        .quant_field = {
          initial_quant.data(), block_extent, block_extent.width},
        .strategy_mask = {
          strategy_mask.data(), block_extent, block_extent.width},
        .pixel_mask = {
          pixel_mask.data(), padded_extent, padded_extent.width},
      },
      .adaptive_quantization = {
        .quant_field = {
          final_quant.data(), block_extent, block_extent.width},
        .block_distance_map = {
          block_distance.data(), block_extent, block_extent.width},
        .reconstructed_linear_rgb = reconstructed.view(),
        .frame = &frame,
        .score_history = &score_history,
      },
    };
  }

  [[nodiscard]] size_t BlockCount() const {
    size_t count = 0;
    if (!block_extent.try_area(&count)) {
      throw std::length_error("VarDCT block extent is too large");
    }
    return count;
  }

  [[nodiscard]] static size_t PixelCount(Extent2D extent) {
    size_t count = 0;
    if (!extent.try_area(&count)) {
      throw std::length_error("VarDCT padded extent is too large");
    }
    return count;
  }

  Extent2D block_extent;
  Extent2D padded_extent;
  std::vector<float> initial_quant;
  std::vector<float> strategy_mask;
  std::vector<float> pixel_mask;
  std::vector<float> final_quant;
  std::vector<float> block_distance;
  Image3FBuffer reconstructed;
  VarDctEncoderFrame frame;
  std::vector<double> score_history;
};

[[nodiscard]] Status EdgeExtend(
  ConstImage3FView source,
  Image3FView destination) {

  if (!source.valid() || !destination.valid() ||
      source.width() > destination.width() ||
      source.height() > destination.height()) {
    return Status::InvalidArgument(
      "Linear RGB source or padded destination is invalid");
  }
  for (size_t y = 0; y < destination.height(); ++y) {
    const size_t source_y = std::min(y, source.height() - 1);
    for (size_t x = 0; x < destination.width(); ++x) {
      const size_t source_x = std::min(x, source.width() - 1);
      for (size_t channel = 0; channel < 3; ++channel) {
        const float value = source.plane[channel].Row(source_y)[source_x];
        if (!std::isfinite(value)) {
          return Status::InvalidArgument(
            "Linear RGB input pixels must be finite");
        }
        destination.plane[channel].Row(y)[x] = value;
      }
    }
  }
  return Status::Ok();
}

}  // namespace

Status EncodeLinearRgbVarDctCodestream(
  ConstImage3FView linear_rgb,
  VarDctEncodingOptions options,
  std::vector<uint8_t>* codestream,
  VarDctEncodingSummary* summary) {

  if (codestream == nullptr || !linear_rgb.valid() ||
      !std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f) {
    return Status::InvalidArgument(
      "VarDCT encoding input, target, or output is invalid");
  }

  try {
    FrameGeometry geometry;
    Status status = FrameGeometry::Create(linear_rgb.extent(), &geometry);
    if (!status.ok()) {
      return status;
    }

    Image3FBuffer padded_linear(geometry.padded_frame());
    status = EdgeExtend(linear_rgb, padded_linear.view());
    if (!status.ok()) {
      return status;
    }
    Image3FBuffer opsin(geometry.padded_frame());
    status = LinearRgbToOpsin(
      padded_linear.const_view(),
      kInitialProfileIntensityTarget,
      opsin.view());
    if (!status.ok()) {
      return status;
    }

    PipelineStorage pipeline(linear_rgb.extent(), geometry.padded_frame());
    CpuQuantizationPipelineOptions pipeline_options;
    pipeline_options.butteraugli_target = options.butteraugli_target;
    status = RunCpuQuantizationPipeline(
      linear_rgb,
      opsin.const_view(),
      pipeline_options,
      pipeline.Output());
    if (!status.ok()) {
      return status;
    }

    std::vector<uint8_t> candidate;
    status = EncodeVarDctCodestream(pipeline.frame, &candidate);
    if (!status.ok()) {
      return status;
    }

    VarDctEncodingSummary candidate_summary;
    candidate_summary.extent = linear_rgb.extent();
    candidate_summary.encoded_bytes = candidate.size();
    candidate_summary.score_history = std::move(pipeline.score_history);
    status = pipeline.frame.strategies().ForEachAnchor(
      [&](size_t, size_t, AcStrategyType strategy) {
        const size_t index = static_cast<size_t>(strategy);
        if (index >= candidate_summary.strategy_counts.size()) {
          return Status::Internal(
            "Completed frame contains an unknown AC strategy");
        }
        ++candidate_summary.strategy_counts[index];
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }

    *codestream = std::move(candidate);
    if (summary != nullptr) {
      *summary = std::move(candidate_summary);
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate public VarDCT encoding storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Public VarDCT encoding dimensions are too large");
  }
  return Status::Ok();
}

}  // namespace gjxl
