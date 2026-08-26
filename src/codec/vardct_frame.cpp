// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/vardct_frame.h"

#include <algorithm>
#include <limits>

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

}  // namespace gjxl
