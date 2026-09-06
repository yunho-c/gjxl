// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/vardct_frame.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

#include "core/managed_allocator.h"
#include "codec/chroma_from_luma_internal.h"
#include "codec/dct.h"
#include "codec/vardct_frame_internal.h"
#include "codec/vardct_frame_view_internal.h"

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
  return vardct_frame_internal::BorrowFrame(*this).GetAcGroup(group_index, out);
}

bool VarDctEncoderFrame::valid() const {
  return vardct_frame_internal::BorrowFrame(*this).valid();
}

vardct_frame_internal::VarDctFrameView vardct_frame_internal::BorrowFrame(
  const VarDctEncoderFrame& frame) noexcept {
  // Validate owned allocation sizes before exposing uncounted core plane views.
  // Value validation remains in the common view validator.
  size_t block_count = 0;
  if (!frame.geometry_.block_grid().blocks.try_area(&block_count) ||
      frame.raw_quant_field_.size() != block_count ||
      frame.epf_sharpness_.size() != block_count) {
    return {};
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    if (frame.quantized_dc_[channel].size() != block_count ||
        frame.dc_[channel].size() != block_count) {
      return {};
    }
  }
  return VarDctFrameView({
    .input = {
      .geometry = frame.geometry_,
      .strategies = &frame.strategies_,
      .raw_quant_field = frame.raw_quant_field(),
      .quantizer = &frame.quantizer_,
      .color_correlation = &frame.color_correlation_,
      .epf_sharpness = frame.epf_sharpness(),
    },
    .profile = frame.profile_,
    .quantized_dc = frame.quantized_dc(),
    .dc = frame.dc(),
    .ac_group_extent = frame.ac_group_extent_,
    .group_used_coefficient_count = frame.group_used_coefficient_count_,
    .ac_coefficients = frame.ac_coefficients_,
  });
}

Status vardct_frame_internal::VarDctFrameView::GetAcGroup(
  size_t group_index,
  VarDctAcGroupView* out) const {

  if (out == nullptr) {
    return Status::InvalidArgument("VarDCT AC-group output is null");
  }
  size_t group_count = 0;
  if (!ValidGeometry(data_.input.geometry) ||
      data_.ac_group_extent != data_.input.geometry.block_grid().blocks.ceil_div(
        kVarDctAcGroupBlockDimension) ||
      !data_.ac_group_extent.try_area(&group_count) ||
      group_count != data_.group_used_coefficient_count.size() ||
      group_index >= group_count ||
      group_count > std::numeric_limits<size_t>::max() / 3 ||
      group_count * 3 > std::numeric_limits<size_t>::max() /
        kVarDctAcGroupCoefficientCapacity ||
      data_.ac_coefficients.size() != group_count * 3 *
        kVarDctAcGroupCoefficientCapacity) {
    return Status::InvalidArgument("VarDCT AC-group index is invalid");
  }

  size_t block_x = 0;
  size_t block_y = 0;
  const Extent2D block_extent = GroupBlockExtent(
    data_.input.geometry.block_grid().blocks,
    data_.ac_group_extent,
    group_index,
    &block_x,
    &block_y);

  VarDctAcGroupView result{
    .block_x = block_x,
    .block_y = block_y,
    .block_extent = block_extent,
    .used_coefficient_count =
      data_.group_used_coefficient_count[group_index],
  };
  for (size_t channel = 0; channel < 3; ++channel) {
    const size_t offset = (group_index * 3 + channel) *
      kVarDctAcGroupCoefficientCapacity;
    result.coefficients[channel] = {
      data_.ac_coefficients.data() + offset,
      kVarDctAcGroupCoefficientCapacity,
    };
  }
  *out = result;
  return Status::Ok();
}

bool vardct_frame_internal::VarDctFrameView::valid() const {
  if (!ValidGeometry(geometry()) ||
      data_.input.strategies == nullptr ||
      !strategies().complete() ||
      strategies().extent() != geometry().block_grid().blocks ||
      data_.input.quantizer == nullptr || !quantizer().valid() ||
      data_.input.color_correlation == nullptr || !color_correlation().valid() ||
      color_correlation().tile_extent() != ExpectedColorTileExtent(geometry()) ||
      !profile().valid()) {
    return false;
  }

  size_t block_count = 0;
  size_t group_count = 0;
  const Extent2D blocks = geometry().block_grid().blocks;
  const auto valid_plane = [blocks](const auto& plane) {
    const size_t maximum_elements =
      std::numeric_limits<size_t>::max() / sizeof(*plane.data);
    return plane.valid() && plane.extent == blocks &&
      blocks.width <= maximum_elements &&
      (blocks.height == 1 ||
       plane.stride <= (maximum_elements - blocks.width) / (blocks.height - 1));
  };
  if (!blocks.try_area(&block_count) ||
      !ac_group_extent().try_area(&group_count) ||
      ac_group_extent() != blocks.ceil_div(kVarDctAcGroupBlockDimension) ||
      ac_group_count() != group_count ||
      !valid_plane(raw_quant_field()) || !valid_plane(epf_sharpness())) {
    return false;
  }
  for (size_t channel = 0; channel < 3; ++channel) {
    if (!valid_plane(quantized_dc().plane[channel]) ||
        !valid_plane(dc().plane[channel])) {
      return false;
    }
    for (size_t y = 0; y < blocks.height; ++y) {
      if (!std::ranges::all_of(
            std::span<const float>(dc().plane[channel].Row(y), blocks.width),
            [](float value) { return std::isfinite(value); })) {
        return false;
      }
    }
  }
  const std::array<float, 3>& dc_steps = quantizer().dc_steps();
  for (size_t y = 0; y < blocks.height; ++y) {
    for (size_t x = 0; x < blocks.width; ++x) {
      const float reconstructed_y =
        static_cast<float>(quantized_dc().plane[1].Row(y)[x]) * dc_steps[1];
      const float dc_x = dc().plane[0].Row(y)[x];
      const float dc_y = dc().plane[1].Row(y)[x];
      const float dc_b = dc().plane[2].Row(y)[x];
      if (dc_x != static_cast<float>(quantized_dc().plane[0].Row(y)[x]) *
            dc_steps[0] ||
          dc_y != reconstructed_y ||
          dc_b != static_cast<float>(quantized_dc().plane[2].Row(y)[x]) *
            dc_steps[2] + reconstructed_y) {
        return false;
      }
    }
  }
  for (size_t y = 0; y < blocks.height; ++y) {
    if (!std::ranges::all_of(
          std::span<const int32_t>(raw_quant_field().Row(y), blocks.width),
          [](int32_t value) { return value >= 1 && value <= kMaxRawQuant; }) ||
        !std::ranges::all_of(
          std::span<const uint8_t>(epf_sharpness().Row(y), blocks.width),
          [](uint8_t value) { return value < 8; })) {
      return false;
    }
  }

  if (group_count > std::numeric_limits<size_t>::max() / 3 ||
      group_count * 3 > std::numeric_limits<size_t>::max() /
        kVarDctAcGroupCoefficientCapacity ||
      data_.ac_coefficients.size() != group_count * 3 *
        kVarDctAcGroupCoefficientCapacity) {
    return false;
  }
  // The storage sizes are checked above. Scan fixed-capacity tails directly;
  // the integer reduction lets the compiler vectorize the exact zero check.
  for (size_t group_index = 0; group_index < group_count; ++group_index) {
    size_t block_x = 0;
    size_t block_y = 0;
    const Extent2D group_blocks = GroupBlockExtent(
      blocks, ac_group_extent(), group_index, &block_x, &block_y);
    size_t covered_blocks = 0;
    if (!group_blocks.try_area(&covered_blocks) ||
        covered_blocks > kVarDctAcGroupCoefficientCapacity / kJxlBlockArea) {
      return false;
    }
    const size_t expected = covered_blocks * kJxlBlockArea;
    if (data_.group_used_coefficient_count[group_index] != expected) {
      return false;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      const size_t base = (group_index * 3 + channel) *
        kVarDctAcGroupCoefficientCapacity;
      uint32_t nonzero = 0;
      for (size_t i = expected; i < kVarDctAcGroupCoefficientCapacity; ++i) {
        nonzero |= static_cast<uint32_t>(data_.ac_coefficients[base + i]);
      }
      if (nonzero != 0) {
        return false;
      }
    }
  }

  const Status strategy_status = strategies().ForEachAnchor(
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

bool CopyQuantizedCoefficients(
  std::span<const int32_t> source,
  bool reject_unwritten,
  int32_t* destination) {

  if (!reject_unwritten) {
    std::copy(source.begin(), source.end(), destination);
    return true;
  }
  uint32_t found_unwritten = 0;
  for (size_t index = 0; index < source.size(); ++index) {
    const int32_t value = source[index];
    destination[index] = value;
    found_unwritten |= static_cast<uint32_t>(
      value == kUnwrittenQuantizedCoefficient);
  }
  return found_unwritten == 0;
}

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
      !input.epf_sharpness.valid() || input.quantized_ac.empty() ||
      !input.quantized_dc.valid() || !input.profile.valid() ||
      !ValidGeometry(input.geometry)) {
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

    const resource_budget_internal::ResourceClassScope resource_class(
      resource_budget_internal::ResourceClass::kCompletedFrame);
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
      for (size_t x = 0; x < blocks.width; ++x) {
        const size_t index = y * blocks.width + x;
        const int32_t raw_quant = input.raw_quant_field.Row(y)[x];
        const uint8_t epf_sharpness = input.epf_sharpness.Row(y)[x];
        if (raw_quant < 1 || raw_quant > kMaxRawQuant ||
            epf_sharpness >= 8) {
          return Status::InvalidArgument(
            "Quantized VarDCT raw quantization or EPF sharpness is invalid");
        }
        result.raw_quant_field_[index] = raw_quant;
        result.epf_sharpness_[index] = epf_sharpness;
      }
      for (size_t channel = 0; channel < 3; ++channel) {
        if (!CopyQuantizedCoefficients(
              {input.quantized_dc.plane[channel].Row(y), blocks.width},
              input.reject_unwritten_coefficients,
              result.quantized_dc_[channel].data() + y * blocks.width)) {
          return Status::InvalidArgument(
            "Quantized VarDCT DC coefficients contain unwritten values");
        }
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
        const QuantizedAcTransformLayout& transform =
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
        if (transform.coefficient_count != coefficient_count) {
          return Status::InvalidArgument(
            "Quantized VarDCT transform coefficient count is invalid");
        }
        for (size_t offset : transform.coefficient_offsets) {
          if (offset > input.quantized_ac.size() ||
              coefficient_count > input.quantized_ac.size() - offset) {
            return Status::InvalidArgument(
              "Quantized VarDCT transform coefficient range is invalid");
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
          if (!CopyQuantizedCoefficients(
                input.quantized_ac.subspan(
                  transform.coefficient_offsets[channel], coefficient_count),
                input.reject_unwritten_coefficients,
                result.ac_coefficients_.data() +
                  result.AcGroupChannelOffset(group_index, channel) +
                  group_offset)) {
            return Status::InvalidArgument(
              "Quantized VarDCT AC coefficients contain unwritten values");
          }
        }
        result.group_used_coefficient_count_[group_index] += coefficient_count;
        return Status::Ok();
      });
    if (!status.ok()) {
      return status;
    }
    if (transform_index != input.transforms.size()) {
      return Status::Internal(
        "Quantized VarDCT transform list contains trailing entries");
    }
    for (size_t group_index = 0; group_index < group_count; ++group_index) {
      size_t block_x = 0;
      size_t block_y = 0;
      const Extent2D group_blocks = GroupBlockExtent(
        blocks, group_extent, group_index, &block_x, &block_y);
      size_t covered_blocks = 0;
      if (!group_blocks.try_area(&covered_blocks) ||
          covered_blocks >
            kVarDctAcGroupCoefficientCapacity / kJxlBlockArea ||
          result.group_used_coefficient_count_[group_index] !=
            covered_blocks * kJxlBlockArea) {
        return Status::Internal(
          "Quantized data did not completely fill its VarDCT AC groups");
      }
    }
    *out = std::move(result);
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
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
