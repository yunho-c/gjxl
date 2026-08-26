// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/reconstruction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "codec/dc_conversion.h"
#include "codec/dc_quantization.h"
#include "codec/dct.h"
#include "codec/quantization.h"
#include "core/geometry.h"

namespace gjxl {
namespace {

constexpr std::array<XybChannel, 3> kChannels = {
  XybChannel::kX,
  XybChannel::kY,
  XybChannel::kB,
};

bool ValidOptions(CoefficientCodingOptions options) {
  return std::isfinite(options.x_matrix_multiplier) &&
    options.x_matrix_multiplier > 0.0f &&
    std::isfinite(options.b_matrix_multiplier) &&
    options.b_matrix_multiplier > 0.0f;
}

float MatrixMultiplier(
  size_t channel,
  CoefficientCodingOptions options) {

  if (channel == 0) {
    return options.x_matrix_multiplier;
  }
  if (channel == 2) {
    return options.b_matrix_multiplier;
  }
  return 1.0f;
}

Status ValidateImageContract(
  ConstImage3FView opsin,
  VarDctFrameInput input,
  CoefficientCodingOptions options) {

  if (!opsin.valid() ||
      input.geometry.frame().empty() ||
      input.strategies == nullptr ||
      !input.strategies->complete() ||
      !input.raw_quant_field.valid() ||
      input.quantizer == nullptr ||
      !input.quantizer->valid() ||
      input.color_correlation == nullptr ||
      !input.color_correlation->valid() ||
      !input.epf_sharpness.valid() ||
      !ValidOptions(options) ||
      input.raw_quant_field.extent != input.strategies->extent() ||
      input.epf_sharpness.extent != input.strategies->extent() ||
      input.geometry.block_grid().blocks != input.strategies->extent()) {
    return Status::InvalidArgument(
      "Coefficient coding inputs are invalid or differently sized");
  }

  const Extent2D block_extent = input.strategies->extent();
  if (block_extent.width >
        std::numeric_limits<size_t>::max() / kJxlBlockDimension ||
      block_extent.height >
        std::numeric_limits<size_t>::max() / kJxlBlockDimension ||
      opsin.extent() != input.geometry.padded_frame()) {
    return Status::InvalidArgument(
      "Coefficient image does not match its block grid");
  }

  const Extent2D expected_color_tiles{
    (opsin.width() + kColorTileDimension - 1) / kColorTileDimension,
    (opsin.height() + kColorTileDimension - 1) / kColorTileDimension,
  };
  if (input.color_correlation->tile_extent() != expected_color_tiles) {
    return Status::InvalidArgument(
      "Color-correlation map does not match the coefficient image");
  }

  for (size_t y = 0; y < block_extent.height; ++y) {
    for (size_t x = 0; x < block_extent.width; ++x) {
      const int32_t raw_quant = input.raw_quant_field.Row(y)[x];
      if (raw_quant < 1 || raw_quant > kMaxRawQuant ||
          input.epf_sharpness.Row(y)[x] >= 8) {
        return Status::InvalidArgument(
          "Raw quantization or EPF sharpness is out of range");
      }
    }
  }

  return Status::Ok();
}

void CopyPixelsFromImage(
  ConstPlaneF32View plane,
  size_t block_x,
  size_t block_y,
  Extent2D pixel_extent,
  std::vector<float>* pixels) {

  for (size_t y = 0; y < pixel_extent.height; ++y) {
    const float* source = plane.Row(
      block_y * kJxlBlockDimension + y) +
      block_x * kJxlBlockDimension;
    std::copy_n(
      source,
      pixel_extent.width,
      pixels->data() + y * pixel_extent.width);
  }
}

void CopyPixelsToImage(
  const std::vector<float>& pixels,
  size_t block_x,
  size_t block_y,
  Extent2D pixel_extent,
  PlaneF32View plane) {

  for (size_t y = 0; y < pixel_extent.height; ++y) {
    float* destination = plane.Row(
      block_y * kJxlBlockDimension + y) +
      block_x * kJxlBlockDimension;
    std::copy_n(
      pixels.data() + y * pixel_extent.width,
      pixel_extent.width,
      destination);
  }
}

}  // namespace

Status ComputeQuantizedCoefficients(
  ConstImage3FView opsin,
  VarDctFrameInput input,
  CoefficientCodingOptions options,
  VarDctEncoderFrame* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Quantized coefficient output is null");
  }

  Status status = ValidateImageContract(
    opsin,
    input,
    options);
  if (!status.ok()) {
    return status;
  }

  try {
    const Extent2D block_extent = input.strategies->extent();
    size_t block_count = 0;
    if (!block_extent.try_area(&block_count)) {
      return Status::InvalidArgument(
        "Coefficient coding block grid is too large");
    }

    VarDctEncoderFrame result;
    result.geometry_ = input.geometry;
    result.strategies_ = *input.strategies;
    result.raw_quant_field_.resize(block_count);
    result.epf_sharpness_.resize(block_count);
    for (size_t y = 0; y < block_extent.height; ++y) {
      std::copy_n(
        input.raw_quant_field.Row(y),
        block_extent.width,
        result.raw_quant_field_.data() + y * block_extent.width);
      std::copy_n(
        input.epf_sharpness.Row(y),
        block_extent.width,
        result.epf_sharpness_.data() + y * block_extent.width);
    }
    result.quantizer_ = *input.quantizer;
    result.color_correlation_ = *input.color_correlation;
    result.coding_options_ = options;
    for (size_t channel = 0; channel < 3; ++channel) {
      result.quantized_dc_[channel].resize(block_count);
      result.dc_[channel].assign(block_count, 0.0f);
    }

    result.ac_group_extent_ = {
      (block_extent.width + kVarDctAcGroupBlockDimension - 1) /
        kVarDctAcGroupBlockDimension,
      (block_extent.height + kVarDctAcGroupBlockDimension - 1) /
        kVarDctAcGroupBlockDimension,
    };
    size_t group_count = 0;
    if (!result.ac_group_extent_.try_area(&group_count) ||
        group_count > std::numeric_limits<size_t>::max() / 3 ||
        group_count * 3 > std::numeric_limits<size_t>::max() /
          kVarDctAcGroupCoefficientCapacity) {
      return Status::InvalidArgument(
        "Coefficient coding group grid is too large");
    }
    result.group_used_coefficient_count_.assign(group_count, 0);
    result.ac_coefficients_.assign(
      group_count * 3 * kVarDctAcGroupCoefficientCapacity,
      0);

    status = input.strategies->ForEachAnchor(
      [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
        const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
        if (info == nullptr || !SupportsCpuDct(strategy)) {
          return Status::Unavailable(
            "Coefficient coding does not support an AC strategy");
        }

        const size_t group_x = block_x / kVarDctAcGroupBlockDimension;
        const size_t group_y = block_y / kVarDctAcGroupBlockDimension;
        if ((block_x + info->covered_blocks.width - 1) /
              kVarDctAcGroupBlockDimension != group_x ||
            (block_y + info->covered_blocks.height - 1) /
              kVarDctAcGroupBlockDimension != group_y) {
          return Status::InvalidArgument(
            "AC strategy crosses a VarDCT group boundary");
        }
        const size_t group_index =
          group_y * result.ac_group_extent_.width + group_x;
        const size_t coefficient_count = info->coefficient_count();
        std::array<std::vector<float>, 3> coefficients;
        std::array<std::vector<int32_t>, 3> quantized;
        std::vector<float> pixels(coefficient_count);
        for (size_t channel = 0; channel < coefficients.size(); ++channel) {
          coefficients[channel].resize(coefficient_count);
          quantized[channel].resize(coefficient_count);
          CopyPixelsFromImage(
            opsin.plane[channel],
            block_x,
            block_y,
            info->pixel_extent(),
            &pixels);
          Status transform_status = ForwardDctCpu(
            strategy,
            pixels,
            coefficients[channel]);
          if (!transform_status.ok()) {
            return transform_status;
          }
        }

        const int32_t raw_quant =
          result.raw_quant_field_[block_y * block_extent.width + block_x];

        // Y is round-trip dequantized before removing its prediction from
        // X/B, exactly as in libjxl's VarDCT coefficient path.
        std::vector<float> y_dc(
          info->covered_blocks.width * info->covered_blocks.height);
        const PlaneF32View y_dc_view{
          .data = y_dc.data(),
          .extent = info->covered_blocks,
          .stride = info->covered_blocks.width,
        };
        Status block_status = ConvertLowFrequenciesToDc(
          strategy,
          coefficients[1],
          y_dc_view);
        if (!block_status.ok()) {
          return block_status;
        }
        for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
          for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
            result.dc_[1][
              (block_y + dy) * block_extent.width + block_x + dx] =
                y_dc[dy * info->covered_blocks.width + dx];
          }
        }

        block_status = QuantizeAcBlock(
          strategy,
          result.quantizer_,
          raw_quant,
          {.channel = XybChannel::kY},
          coefficients[1],
          quantized[1]);
        if (!block_status.ok()) {
          return block_status;
        }
        block_status = DequantizeAcBlock(
          strategy,
          result.quantizer_,
          raw_quant,
          {.channel = XybChannel::kY},
          quantized[1],
          coefficients[1]);
        if (!block_status.ok()) {
          return block_status;
        }

        const std::array<float, 3> factors =
          result.color_correlation_.AcFactors(block_x / 8, block_y / 8);
        for (size_t index = 0; index < coefficient_count; ++index) {
          coefficients[0][index] -= factors[0] * coefficients[1][index];
          coefficients[2][index] -= factors[2] * coefficients[1][index];
        }

        for (size_t channel : {size_t{0}, size_t{2}}) {
          std::vector<float> dc(
            info->covered_blocks.width * info->covered_blocks.height);
          const PlaneF32View dc_view{
            .data = dc.data(),
            .extent = info->covered_blocks,
            .stride = info->covered_blocks.width,
          };
          block_status = ConvertLowFrequenciesToDc(
            strategy,
            coefficients[channel],
            dc_view);
          if (!block_status.ok()) {
            return block_status;
          }
          for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
            for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
              result.dc_[channel][
                (block_y + dy) * block_extent.width + block_x + dx] =
                  dc[dy * info->covered_blocks.width + dx];
            }
          }

          block_status = QuantizeAcBlock(
            strategy,
            result.quantizer_,
            raw_quant,
            {
              .channel = kChannels[channel],
              .matrix_multiplier = MatrixMultiplier(channel, options),
            },
            coefficients[channel],
            quantized[channel]);
          if (!block_status.ok()) {
            return block_status;
          }
        }

        const size_t group_offset =
          result.group_used_coefficient_count_[group_index];
        if (coefficient_count >
            kVarDctAcGroupCoefficientCapacity - group_offset) {
          return Status::Internal(
            "VarDCT AC group coefficient storage overflowed");
        }
        for (size_t channel = 0; channel < 3; ++channel) {
          std::copy(
            quantized[channel].begin(),
            quantized[channel].end(),
            result.ac_coefficients_.begin() +
              result.AcGroupChannelOffset(group_index, channel) +
              group_offset);
        }
        result.group_used_coefficient_count_[group_index] +=
          coefficient_count;
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }

    const Image3I32View quantized_dc{{
      PlaneI32View{
        result.quantized_dc_[0].data(), block_extent, block_extent.width},
      PlaneI32View{
        result.quantized_dc_[1].data(), block_extent, block_extent.width},
      PlaneI32View{
        result.quantized_dc_[2].data(), block_extent, block_extent.width},
    }};
    const Image3FView reconstructed_dc{{
      PlaneF32View{result.dc_[0].data(), block_extent, block_extent.width},
      PlaneF32View{result.dc_[1].data(), block_extent, block_extent.width},
      PlaneF32View{result.dc_[2].data(), block_extent, block_extent.width},
    }};
    status = QuantizeDcCoefficients(
      result.dc(),
      result.quantizer_,
      {.quantized = quantized_dc, .reconstructed = reconstructed_dc});
    if (!status.ok()) {
      return status;
    }
    if (!result.valid()) {
      return Status::Internal(
        "Coefficient coding did not produce a valid encoder frame");
    }

    *out = std::move(result);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate coefficient coding scratch storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Coefficient coding dimensions are too large");
  }

  return Status::Ok();
}

Status ReconstructQuantizedCoefficients(
  const VarDctEncoderFrame& frame,
  Image3FView output) {

  if (!frame.valid() || !output.valid()) {
    return Status::InvalidArgument(
      "Coefficient reconstruction inputs are invalid");
  }

  const Extent2D block_extent = frame.geometry_.block_grid().blocks;
  if (output.extent() != frame.geometry_.padded_frame()) {
    return Status::InvalidArgument(
      "Coefficient reconstruction output has the wrong extent");
  }

  try {
    size_t pixel_count = 0;
    if (!output.extent().try_area(&pixel_count)) {
      return Status::InvalidArgument(
        "Coefficient reconstruction dimensions are too large");
    }
    std::array<std::vector<float>, 3> result;
    for (std::vector<float>& plane : result) {
      plane.resize(pixel_count);
    }
    const Image3FView result_view{{
      PlaneF32View{result[0].data(), output.extent(), output.width()},
      PlaneF32View{result[1].data(), output.extent(), output.width()},
      PlaneF32View{result[2].data(), output.extent(), output.width()},
    }};

    std::vector<size_t> group_offsets(frame.ac_group_count(), 0);
    const ConstImage3FView frame_dc = frame.dc();
    const Status reconstruct_status = frame.strategies_.ForEachAnchor(
      [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
        const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
        if (info == nullptr || !SupportsCpuDct(strategy)) {
          return Status::InvalidArgument(
            "Stored coefficient strategy is unsupported");
        }

        const size_t group_x = block_x / kVarDctAcGroupBlockDimension;
        const size_t group_y = block_y / kVarDctAcGroupBlockDimension;
        const size_t group_index =
          group_y * frame.ac_group_extent_.width + group_x;
        const size_t group_offset = group_offsets[group_index];
        const size_t coefficient_count = info->coefficient_count();
        if (group_offset >
              frame.group_used_coefficient_count_[group_index] ||
            coefficient_count >
            frame.group_used_coefficient_count_[group_index] - group_offset) {
          return Status::Internal(
            "Stored VarDCT AC group ended inside a transform");
        }

        const int32_t raw_quant = frame.raw_quant_field_[
          block_y * block_extent.width + block_x];
        std::array<std::vector<float>, 3> coefficients;
        for (size_t channel = 0; channel < coefficients.size(); ++channel) {
          coefficients[channel].resize(coefficient_count);
          const size_t source =
            frame.AcGroupChannelOffset(group_index, channel) + group_offset;
          Status status = DequantizeAcBlock(
            strategy,
            frame.quantizer_,
            raw_quant,
            {
              .channel = kChannels[channel],
              .matrix_multiplier = MatrixMultiplier(
                channel,
                frame.coding_options_),
            },
            std::span<const int32_t>(
              frame.ac_coefficients_.data() + source,
              coefficient_count),
            coefficients[channel]);
          if (!status.ok()) {
            return status;
          }
        }

        const std::array<float, 3> factors =
          frame.color_correlation_.AcFactors(block_x / 8, block_y / 8);
        for (size_t index = 0; index < coefficient_count; ++index) {
          coefficients[0][index] += factors[0] * coefficients[1][index];
          coefficients[2][index] += factors[2] * coefficients[1][index];
        }

        for (size_t channel = 0; channel < coefficients.size(); ++channel) {
          std::vector<float> dc(
            info->covered_blocks.width * info->covered_blocks.height);
          for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
            for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
              dc[dy * info->covered_blocks.width + dx] =
                frame_dc.plane[channel].Row(block_y + dy)[block_x + dx];
            }
          }
          const ConstPlaneF32View dc_view{
            .data = dc.data(),
            .extent = info->covered_blocks,
            .stride = info->covered_blocks.width,
          };
          Status status = ConvertDcToLowFrequencies(
            strategy,
            dc_view,
            coefficients[channel]);
          if (!status.ok()) {
            return status;
          }

          std::vector<float> pixels(coefficient_count);
          status = InverseDctCpu(strategy, coefficients[channel], pixels);
          if (!status.ok()) {
            return status;
          }
          CopyPixelsToImage(
            pixels,
            block_x,
            block_y,
            info->pixel_extent(),
            result_view.plane[channel]);
        }

        group_offsets[group_index] += coefficient_count;
        return Status::Ok();
      });
    if (!reconstruct_status.ok()) {
      return reconstruct_status;
    }
    if (group_offsets != frame.group_used_coefficient_count_) {
      return Status::Internal(
        "Stored VarDCT AC groups contain unconsumed coefficients");
    }

    for (size_t channel = 0; channel < result.size(); ++channel) {
      for (size_t y = 0; y < output.height(); ++y) {
        std::copy_n(
          result_view.plane[channel].Row(y),
          output.width(),
          output.plane[channel].Row(y));
      }
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate coefficient reconstruction scratch storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Coefficient reconstruction dimensions are too large");
  }

  return Status::Ok();
}

}  // namespace gjxl
