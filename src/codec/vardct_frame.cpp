// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/vardct_frame.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

#include "codec/chroma_from_luma_internal.h"
#include "codec/dct.h"
#include "codec/vardct_frame_internal.h"

namespace gjxl {
namespace {

bool ValidGeometry(const FrameGeometry& geometry) {
  if (geometry.frame().empty() || geometry.block_grid().blocks.empty()) {
    return false;
  }
  return geometry.padded_frame() == Extent2D{
    geometry.block_grid().blocks.width * kJxlBlockDimension,
    geometry.block_grid().blocks.height * kJxlBlockDimension,
  };
}

Extent2D ExpectedColorTileExtent(const FrameGeometry& geometry) {
  const Extent2D pixels = geometry.padded_frame();
  return {
    (pixels.width + kColorTileDimension - 1) / kColorTileDimension,
    (pixels.height + kColorTileDimension - 1) / kColorTileDimension,
  };
}

Extent2D GroupBlockExtent(
  Extent2D blocks,
  Extent2D groups,
  size_t group_index,
  size_t* block_x,
  size_t* block_y) {

  const size_t group_x = group_index % groups.width;
  const size_t group_y = group_index / groups.width;
  *block_x = group_x * kVarDctAcGroupBlockDimension;
  *block_y = group_y * kVarDctAcGroupBlockDimension;
  return {
    std::min(kVarDctAcGroupBlockDimension, blocks.width - *block_x),
    std::min(kVarDctAcGroupBlockDimension, blocks.height - *block_y),
  };
}

}  // namespace

size_t VarDctEncoderFrame::AcGroupChannelOffset(
  size_t group_index,
  size_t channel) const noexcept {

  return (group_index * 3 + channel) *
    kVarDctAcGroupCoefficientCapacity;
}

ConstPlaneI32View VarDctEncoderFrame::raw_quant_field() const noexcept {
  const Extent2D extent = geometry_.block_grid().blocks;
  return {raw_quant_field_.data(), extent, extent.width};
}

ConstPlaneU8View VarDctEncoderFrame::epf_sharpness() const noexcept {
  const Extent2D extent = geometry_.block_grid().blocks;
  return {epf_sharpness_.data(), extent, extent.width};
}

ConstImage3I32View VarDctEncoderFrame::quantized_dc() const noexcept {
  const Extent2D extent = geometry_.block_grid().blocks;
  return ConstImage3I32View{{
    ConstPlaneI32View{quantized_dc_[0].data(), extent, extent.width},
    ConstPlaneI32View{quantized_dc_[1].data(), extent, extent.width},
    ConstPlaneI32View{quantized_dc_[2].data(), extent, extent.width},
  }};
}

ConstImage3FView VarDctEncoderFrame::dc() const noexcept {
  const Extent2D extent = geometry_.block_grid().blocks;
  return ConstImage3FView{{
    ConstPlaneF32View{dc_[0].data(), extent, extent.width},
    ConstPlaneF32View{dc_[1].data(), extent, extent.width},
    ConstPlaneF32View{dc_[2].data(), extent, extent.width},
  }};
}

Status VarDctEncoderFrame::GetAcGroup(
  size_t group_index,
  VarDctAcGroupView* out) const {

  if (out == nullptr) {
    return Status::InvalidArgument("VarDCT AC-group output is null");
  }
  size_t group_count = 0;
  if (!ac_group_extent_.try_area(&group_count) ||
      group_count != group_used_coefficient_count_.size() ||
      group_index >= group_count ||
      group_count > std::numeric_limits<size_t>::max() / 3 ||
      group_count * 3 > std::numeric_limits<size_t>::max() /
        kVarDctAcGroupCoefficientCapacity ||
      ac_coefficients_.size() != group_count * 3 *
        kVarDctAcGroupCoefficientCapacity) {
    return Status::InvalidArgument("VarDCT AC-group index is invalid");
  }

  size_t block_x = 0;
  size_t block_y = 0;
  const Extent2D block_extent = GroupBlockExtent(
    geometry_.block_grid().blocks,
    ac_group_extent_,
    group_index,
    &block_x,
    &block_y);

  VarDctAcGroupView result{
    .block_x = block_x,
    .block_y = block_y,
    .block_extent = block_extent,
    .used_coefficient_count =
      group_used_coefficient_count_[group_index],
  };
  for (size_t channel = 0; channel < 3; ++channel) {
    result.coefficients[channel] = {
      ac_coefficients_.data() + AcGroupChannelOffset(group_index, channel),
      kVarDctAcGroupCoefficientCapacity,
    };
  }
  *out = result;
  return Status::Ok();
}

bool VarDctEncoderFrame::valid() const {
  if (!ValidGeometry(geometry_) ||
      !strategies_.complete() ||
      strategies_.extent() != geometry_.block_grid().blocks ||
      !quantizer_.valid() ||
      !color_correlation_.valid() ||
      color_correlation_.tile_extent() != ExpectedColorTileExtent(geometry_) ||
      !profile_.valid()) {
    return false;
  }

  size_t block_count = 0;
  size_t group_count = 0;
  const Extent2D blocks = geometry_.block_grid().blocks;
  const Extent2D expected_group_extent{
    (blocks.width + kVarDctAcGroupBlockDimension - 1) /
      kVarDctAcGroupBlockDimension,
    (blocks.height + kVarDctAcGroupBlockDimension - 1) /
      kVarDctAcGroupBlockDimension,
  };
  if (!geometry_.block_grid().blocks.try_area(&block_count) ||
      !ac_group_extent_.try_area(&group_count) ||
      ac_group_extent_ != expected_group_extent ||
      raw_quant_field_.size() != block_count ||
      epf_sharpness_.size() != block_count ||
      group_used_coefficient_count_.size() != group_count) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    if (quantized_dc_[channel].size() != block_count ||
        dc_[channel].size() != block_count ||
        !std::ranges::all_of(dc_[channel], [](float value) {
          return std::isfinite(value);
        })) {
      return false;
    }
  }
  const std::array<float, 3>& dc_steps = quantizer_.dc_steps();
  for (size_t index = 0; index < block_count; ++index) {
    const float reconstructed_y =
      static_cast<float>(quantized_dc_[1][index]) * dc_steps[1];
    if (dc_[0][index] !=
          static_cast<float>(quantized_dc_[0][index]) * dc_steps[0] ||
        dc_[1][index] != reconstructed_y ||
        dc_[2][index] !=
          static_cast<float>(quantized_dc_[2][index]) * dc_steps[2] +
            reconstructed_y) {
      return false;
    }
  }
  if (!std::ranges::all_of(raw_quant_field_, [](int32_t value) {
        return value >= 1 && value <= kMaxRawQuant;
      }) ||
      !std::ranges::all_of(
        epf_sharpness_,
        [](uint8_t value) { return value < 8; })) {
    return false;
  }

  if (group_count > std::numeric_limits<size_t>::max() / 3 ||
      group_count * 3 > std::numeric_limits<size_t>::max() /
        kVarDctAcGroupCoefficientCapacity ||
      ac_coefficients_.size() != group_count * 3 *
        kVarDctAcGroupCoefficientCapacity) {
    return false;
  }

  for (size_t group_index = 0; group_index < group_count; ++group_index) {
    size_t block_x = 0;
    size_t block_y = 0;
    const Extent2D group_blocks = GroupBlockExtent(
      blocks,
      ac_group_extent_,
      group_index,
      &block_x,
      &block_y);
    size_t covered_blocks = 0;
    if (!group_blocks.try_area(&covered_blocks) ||
        covered_blocks > kVarDctAcGroupCoefficientCapacity / kJxlBlockArea) {
      return false;
    }
    const size_t expected = covered_blocks * kJxlBlockArea;
    if (group_used_coefficient_count_[group_index] != expected) {
      return false;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      const size_t base = AcGroupChannelOffset(group_index, channel);
      if (!std::ranges::all_of(
            ac_coefficients_.begin() + base + expected,
            ac_coefficients_.begin() + base +
              kVarDctAcGroupCoefficientCapacity,
            [](int32_t value) { return value == 0; })) {
        return false;
      }
    }
  }

  const Status strategy_status = strategies_.ForEachAnchor(
    [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
      const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
      if (info == nullptr) {
        return Status::InvalidArgument("Unknown AC strategy");
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
      return Status::Ok();
    });
  return strategy_status.ok();
}

namespace vardct_frame_internal {
namespace {

Status ValidateAssemblyInput(
  const QuantizedFrameAssemblyInput& input,
  size_t* block_count,
  Extent2D* group_extent,
  size_t* group_count) {

  if (block_count == nullptr || group_extent == nullptr ||
      group_count == nullptr || input.strategies == nullptr ||
      !input.strategies->complete() || !input.raw_quant_field.valid() ||
      input.quantizer == nullptr || !input.quantizer->valid() ||
      !input.y_to_x.valid() || !input.y_to_b.valid() ||
      !input.epf_sharpness.valid() ||
      !input.quantized_dc.valid() || !input.profile.valid() ||
      input.geometry.frame().empty()) {
    return Status::InvalidArgument(
      "Quantized VarDCT frame assembly input is invalid");
  }

  const Extent2D blocks = input.geometry.block_grid().blocks;
  const Extent2D expected_tiles = ExpectedColorTileExtent(input.geometry);
  if (input.geometry.padded_frame() != Extent2D{
        blocks.width * kJxlBlockDimension,
        blocks.height * kJxlBlockDimension} ||
      input.strategies->extent() != blocks ||
      input.raw_quant_field.extent != blocks ||
      input.epf_sharpness.extent != blocks ||
      input.quantized_dc.extent() != blocks ||
      input.y_to_x.extent != expected_tiles ||
      input.y_to_b.extent != expected_tiles ||
      !blocks.try_area(block_count)) {
    return Status::InvalidArgument(
      "Quantized VarDCT frame assembly geometry does not match");
  }

  *group_extent = {
    (blocks.width + kVarDctAcGroupBlockDimension - 1) /
      kVarDctAcGroupBlockDimension,
    (blocks.height + kVarDctAcGroupBlockDimension - 1) /
      kVarDctAcGroupBlockDimension,
  };
  if (!group_extent->try_area(group_count) ||
      *group_count > std::numeric_limits<size_t>::max() / 3 ||
      *group_count * 3 > std::numeric_limits<size_t>::max() /
        kVarDctAcGroupCoefficientCapacity) {
    return Status::InvalidArgument(
      "Quantized VarDCT frame assembly group grid is too large");
  }

  size_t anchor_count = 0;
  Status status = input.strategies->ForEachAnchor(
    [&](size_t, size_t, AcStrategyType) {
      ++anchor_count;
      return Status::Ok();
    });
  if (!status.ok() || anchor_count != input.transforms.size()) {
    return Status::InvalidArgument(
      "Quantized VarDCT transform list does not cover the strategy grid");
  }
  return Status::Ok();
}

}  // namespace

Status AssembleVarDctEncoderFrame(
  QuantizedFrameAssemblyInput input,
  VarDctEncoderFrame* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Quantized VarDCT frame assembly output is null");
  }

  size_t block_count = 0;
  size_t group_count = 0;
  Extent2D group_extent;
  Status status = ValidateAssemblyInput(
    input, &block_count, &group_extent, &group_count);
  if (!status.ok()) {
    return status;
  }

  try {
    VarDctEncoderFrame result;
    result.geometry_ = input.geometry;
    result.strategies_ = *input.strategies;
    result.quantizer_ = *input.quantizer;
    status = chroma_from_luma_internal::CreateColorCorrelationMap(
      input.y_to_x, input.y_to_b, &result.color_correlation_);
    if (!status.ok()) {
      return status;
    }
    result.profile_ = input.profile;
    result.ac_group_extent_ = group_extent;
    result.raw_quant_field_.resize(block_count);
    result.epf_sharpness_.resize(block_count);
    result.group_used_coefficient_count_.assign(group_count, 0);
    result.ac_coefficients_.assign(
      group_count * 3 * kVarDctAcGroupCoefficientCapacity, 0);
    for (size_t channel = 0; channel < 3; ++channel) {
      result.quantized_dc_[channel].resize(block_count);
      result.dc_[channel].resize(block_count);
    }

    const Extent2D blocks = input.strategies->extent();
    for (size_t y = 0; y < blocks.height; ++y) {
      std::copy_n(
        input.raw_quant_field.Row(y), blocks.width,
        result.raw_quant_field_.data() + y * blocks.width);
      std::copy_n(
        input.epf_sharpness.Row(y), blocks.width,
        result.epf_sharpness_.data() + y * blocks.width);
      for (size_t channel = 0; channel < 3; ++channel) {
        std::copy_n(
          input.quantized_dc.plane[channel].Row(y), blocks.width,
          result.quantized_dc_[channel].data() + y * blocks.width);
      }
    }

    const std::array<float, 3>& dc_steps = result.quantizer_.dc_steps();
    for (size_t index = 0; index < block_count; ++index) {
      const float reconstructed_y =
        static_cast<float>(result.quantized_dc_[1][index]) * dc_steps[1];
      result.dc_[0][index] =
        static_cast<float>(result.quantized_dc_[0][index]) * dc_steps[0];
      result.dc_[1][index] = reconstructed_y;
      result.dc_[2][index] =
        static_cast<float>(result.quantized_dc_[2][index]) * dc_steps[2] +
        reconstructed_y;
      if (!std::isfinite(result.dc_[0][index]) ||
          !std::isfinite(result.dc_[1][index]) ||
          !std::isfinite(result.dc_[2][index])) {
        return Status::InvalidArgument(
          "Quantized VarDCT DC reconstruction is not finite");
      }
    }

    size_t transform_index = 0;
    status = input.strategies->ForEachAnchor(
      [&](size_t block_x, size_t block_y, AcStrategyType strategy) {
        if (transform_index >= input.transforms.size()) {
          return Status::Internal(
            "Quantized VarDCT transform list ended early");
        }
        const QuantizedAcTransformView& transform =
          input.transforms[transform_index++];
        const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
        if (info == nullptr || !SupportsCpuDct(strategy) ||
            transform.block_x != block_x ||
            transform.block_y != block_y ||
            transform.strategy != strategy) {
          return Status::InvalidArgument(
            "Quantized VarDCT transform metadata does not match strategies");
        }
        const size_t coefficient_count = info->coefficient_count();
        for (std::span<const int32_t> coefficients : transform.coefficients) {
          if (coefficients.size() != coefficient_count) {
            return Status::InvalidArgument(
              "Quantized VarDCT transform coefficient count is invalid");
          }
        }

        const size_t group_x = block_x / kVarDctAcGroupBlockDimension;
        const size_t group_y = block_y / kVarDctAcGroupBlockDimension;
        if ((block_x + info->covered_blocks.width - 1) /
              kVarDctAcGroupBlockDimension != group_x ||
            (block_y + info->covered_blocks.height - 1) /
              kVarDctAcGroupBlockDimension != group_y) {
          return Status::InvalidArgument(
            "Quantized VarDCT transform crosses an AC group boundary");
        }
        const size_t group_index =
          group_y * result.ac_group_extent_.width + group_x;
        const size_t group_offset =
          result.group_used_coefficient_count_[group_index];
        if (coefficient_count >
            kVarDctAcGroupCoefficientCapacity - group_offset) {
          return Status::InvalidArgument(
            "Quantized VarDCT AC group coefficient capacity overflowed");
        }
        for (size_t channel = 0; channel < 3; ++channel) {
          std::copy(
            transform.coefficients[channel].begin(),
            transform.coefficients[channel].end(),
            result.ac_coefficients_.begin() +
              result.AcGroupChannelOffset(group_index, channel) +
              group_offset);
        }
        result.group_used_coefficient_count_[group_index] += coefficient_count;
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }
    if (transform_index != input.transforms.size() || !result.valid()) {
      return Status::Internal(
        "Quantized data did not assemble a valid VarDCT encoder frame");
    }
    *out = std::move(result);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate quantized VarDCT frame assembly storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Quantized VarDCT frame assembly dimensions are too large");
  }
}

}  // namespace vardct_frame_internal

}  // namespace gjxl
