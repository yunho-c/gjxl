// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/reconstruction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "codec/dc_conversion.h"
#include "codec/dct.h"
#include "codec/quantization.h"
#include "core/block_grid.h"
#include "core/geometry.h"
#include "core/image_buffer.h"
#include "core/image_ops.h"

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
  const AcStrategyGrid& strategies,
  ConstPlaneI32View raw_quant_field,
  const Quantizer& quantizer,
  const ColorCorrelationMap& color_correlation,
  CoefficientCodingOptions options) {

  if (!opsin.valid() ||
      !strategies.complete() ||
      !raw_quant_field.valid() ||
      !quantizer.valid() ||
      !color_correlation.valid() ||
      !ValidOptions(options) ||
      raw_quant_field.extent != strategies.extent()) {
    return Status::InvalidArgument(
      "Coefficient coding inputs are invalid or differently sized");
  }

  const Extent2D block_extent = strategies.extent();
  Extent2D padded_pixel_extent;
  if (!BlockGrid{block_extent}.try_padded_pixel_extent(
        &padded_pixel_extent) ||
      opsin.extent() != padded_pixel_extent) {
    return Status::InvalidArgument(
      "Coefficient image does not match its block grid");
  }

  const Extent2D expected_color_tiles = ColorTileExtent(opsin.extent());
  if (color_correlation.tile_extent() != expected_color_tiles) {
    return Status::InvalidArgument(
      "Color-correlation map does not match the coefficient image");
  }

  for (size_t y = 0; y < block_extent.height; ++y) {
    for (size_t x = 0; x < block_extent.width; ++x) {
      const int32_t raw_quant = raw_quant_field.Row(y)[x];
      if (raw_quant < 1 || raw_quant > kMaxRawQuant) {
        return Status::InvalidArgument(
          "Raw quantization field value is out of range");
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
  const AcStrategyGrid& strategies,
  ConstPlaneI32View raw_quant_field,
  const Quantizer& quantizer,
  const ColorCorrelationMap& color_correlation,
  CoefficientCodingOptions options,
  QuantizedCoefficientFrame* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Quantized coefficient output is null");
  }

  Status status = ValidateImageContract(
    opsin,
    strategies,
    raw_quant_field,
    quantizer,
    color_correlation,
    options);
  if (!status.ok()) {
    return status;
  }

  try {
    QuantizedCoefficientFrame result;
    status = QuantizedCoefficientFrame::Create(
      strategies.extent(),
      &result);
    if (!status.ok()) {
      return status;
    }

    status = strategies.ForEachAnchor(
      [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
        const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
        if (info == nullptr || !SupportsCpuDct(strategy)) {
          return Status::Unavailable(
            "Coefficient coding does not support an AC strategy");
        }

        const size_t coefficient_count = info->coefficient_count();
        std::array<std::vector<float>, 3> coefficients;
        std::vector<float> pixels(coefficient_count);
        for (size_t channel = 0; channel < coefficients.size(); ++channel) {
          coefficients[channel].resize(coefficient_count);
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

        const int32_t raw_quant = raw_quant_field.Row(block_y)[block_x];
        QuantizedTransform transform{
          .block_x = block_x,
          .block_y = block_y,
          .strategy = strategy,
          .raw_quant = raw_quant,
        };
        for (std::vector<int32_t>& ac : transform.ac) {
          ac.resize(coefficient_count);
        }

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
            block_status = result.SetDc(
              1,
              block_x + dx,
              block_y + dy,
              y_dc[dy * info->covered_blocks.width + dx]);
            if (!block_status.ok()) {
              return block_status;
            }
          }
        }

        block_status = QuantizeAcBlock(
          strategy,
          quantizer,
          raw_quant,
          {.channel = XybChannel::kY},
          coefficients[1],
          transform.ac[1]);
        if (!block_status.ok()) {
          return block_status;
        }
        block_status = DequantizeAcBlock(
          strategy,
          quantizer,
          raw_quant,
          {.channel = XybChannel::kY},
          transform.ac[1],
          coefficients[1]);
        if (!block_status.ok()) {
          return block_status;
        }

        const std::array<float, 3> factors =
          color_correlation.AcFactors(block_x / 8, block_y / 8);
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
              block_status = result.SetDc(
                channel,
                block_x + dx,
                block_y + dy,
                dc[dy * info->covered_blocks.width + dx]);
              if (!block_status.ok()) {
                return block_status;
              }
            }
          }

          block_status = QuantizeAcBlock(
            strategy,
            quantizer,
            raw_quant,
            {
              .channel = kChannels[channel],
              .matrix_multiplier = MatrixMultiplier(channel, options),
            },
            coefficients[channel],
            transform.ac[channel]);
          if (!block_status.ok()) {
            return block_status;
          }
        }

        return result.AddTransform(std::move(transform));
      });
    if (!status.ok()) {
      return status;
    }
    if (!result.complete()) {
      return Status::Internal(
        "Coefficient coding did not cover the complete image");
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
  const QuantizedCoefficientFrame& frame,
  const Quantizer& quantizer,
  const ColorCorrelationMap& color_correlation,
  CoefficientCodingOptions options,
  Image3FView output) {

  if (!frame.complete() ||
      !quantizer.valid() ||
      !color_correlation.valid() ||
      !ValidOptions(options) ||
      !output.valid()) {
    return Status::InvalidArgument(
      "Coefficient reconstruction inputs are invalid");
  }

  const Extent2D block_extent = frame.block_extent();
  Extent2D padded_pixel_extent;
  if (!BlockGrid{block_extent}.try_padded_pixel_extent(
        &padded_pixel_extent) ||
      output.extent() != padded_pixel_extent) {
    return Status::InvalidArgument(
      "Coefficient reconstruction output has the wrong extent");
  }

  const Extent2D expected_color_tiles = ColorTileExtent(output.extent());
  if (color_correlation.tile_extent() != expected_color_tiles) {
    return Status::InvalidArgument(
      "Color-correlation map does not match the reconstruction image");
  }

  try {
    Image3FBuffer result(output.extent());
    const Image3FView result_view = result.view();

    for (const QuantizedTransform& transform : frame.transforms()) {
      const AcStrategyInfo* info = GetAcStrategyInfo(transform.strategy);
      if (info == nullptr || !SupportsCpuDct(transform.strategy)) {
        return Status::InvalidArgument(
          "Stored coefficient strategy is unsupported");
      }

      const size_t coefficient_count = info->coefficient_count();
      std::array<std::vector<float>, 3> coefficients;
      for (size_t channel = 0; channel < coefficients.size(); ++channel) {
        coefficients[channel].resize(coefficient_count);
        Status status = DequantizeAcBlock(
          transform.strategy,
          quantizer,
          transform.raw_quant,
          {
            .channel = kChannels[channel],
            .matrix_multiplier = MatrixMultiplier(channel, options),
          },
          transform.ac[channel],
          coefficients[channel]);
        if (!status.ok()) {
          return status;
        }
      }

      const std::array<float, 3> factors = color_correlation.AcFactors(
        transform.block_x / 8,
        transform.block_y / 8);
      for (size_t index = 0; index < coefficient_count; ++index) {
        coefficients[0][index] += factors[0] * coefficients[1][index];
        coefficients[2][index] += factors[2] * coefficients[1][index];
      }

      for (size_t channel = 0; channel < coefficients.size(); ++channel) {
        std::vector<float> dc(
          info->covered_blocks.width * info->covered_blocks.height);
        for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
          for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
            dc[dy * info->covered_blocks.width + dx] = frame.dc(
              channel,
              transform.block_x + dx,
              transform.block_y + dy);
          }
        }
        const ConstPlaneF32View dc_view{
          .data = dc.data(),
          .extent = info->covered_blocks,
          .stride = info->covered_blocks.width,
        };
        Status status = ConvertDcToLowFrequencies(
          transform.strategy,
          dc_view,
          coefficients[channel]);
        if (!status.ok()) {
          return status;
        }

        std::vector<float> pixels(coefficient_count);
        status = InverseDctCpu(
          transform.strategy,
          coefficients[channel],
          pixels);
        if (!status.ok()) {
          return status;
        }
        CopyPixelsToImage(
          pixels,
          transform.block_x,
          transform.block_y,
          info->pixel_extent(),
          result_view.plane[channel]);
      }
    }

    CopyImage(result.const_view(), output);
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
