// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's encoder/enc_frame.cc.

#include "codestream/dc_group.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "codec/chroma_from_luma.h"
#include "codec/codestream.h"
#include "codec/vardct_frame.h"

namespace gjxl {
namespace {

static_assert(kSimpleDcGroupDimension / kJxlBlockDimension
              == kSimpleDcGroupBlockDimension);
static_assert(kColorTileDimension % kJxlBlockDimension == 0);

struct GradientContextRun {
  uint16_t inclusive_end;
  uint8_t context;
};

// Run-length form of libjxl-tiny's pinned 1024-entry lookup table.
constexpr std::array<GradientContextRun, 34> kGradientContextRuns{{
  {12, 44},  {120, 43}, {257, 40}, {321, 39}, {385, 38},  {417, 37},  {449, 36},
  {465, 35}, {481, 34}, {489, 33}, {497, 32}, {501, 31},  {505, 30},  {508, 29},
  {509, 28}, {511, 27}, {512, 26}, {513, 42}, {515, 41},  {517, 25},  {519, 24},
  {523, 23}, {527, 22}, {535, 21}, {543, 20}, {559, 19},  {575, 18},  {607, 17},
  {639, 16}, {703, 15}, {767, 14}, {904, 13}, {1012, 12}, {1023, 11},
}};

struct StrategyAnchor {
  size_t x;
  size_t y;
  int32_t code;
};

Status AllocationFailure() {
  return Status::OutOfMemory("DC-group token allocation failed");
}

bool IsSimpleStrategy(AcStrategyType strategy) {
  switch (strategy) {
    case AcStrategyType::kDct8:
    case AcStrategyType::kDct16x16:
    case AcStrategyType::kDct32x32:
    case AcStrategyType::kDct16x8:
    case AcStrategyType::kDct8x16:
    case AcStrategyType::kDct32x16:
    case AcStrategyType::kDct16x32:
      return true;
    default:
      return false;
  }
}

bool IsValidGroupExtent(Extent2D extent, size_t* area) {
  return !extent.empty() && extent.width <= kSimpleDcGroupBlockDimension
         && extent.height <= kSimpleDcGroupBlockDimension
         && extent.try_area(area);
}

int32_t ClampedGradient(int32_t top, int32_t left, int32_t top_left) {
  const int32_t minimum = std::min(top, left);
  const int32_t maximum = std::max(top, left);
  if (top_left < minimum) {
    return maximum;
  }
  if (top_left > maximum) {
    return minimum;
  }
  return static_cast<int32_t>(static_cast<int64_t>(top) + left - top_left);
}

uint32_t GradientContext(int64_t gradient_property) {
  const uint16_t clamped =
    static_cast<uint16_t>(std::clamp<int64_t>(gradient_property, 0, 1023));
  for (const GradientContextRun run : kGradientContextRuns) {
    if (clamped <= run.inclusive_end) {
      return run.context;
    }
  }
  return kGradientContextRuns.back().context;
}

Status AppendResidual(uint32_t context, int32_t value, int32_t prediction,
                      std::vector<EntropyToken>* tokens) {
  const int64_t residual = static_cast<int64_t>(value) - prediction;
  if (residual < std::numeric_limits<int32_t>::min()
      || residual > std::numeric_limits<int32_t>::max()) {
    return Status::InvalidArgument(
      "Modular predictor residual exceeds int32_t");
  }
  tokens->push_back({
    context,
    PackSigned(static_cast<int32_t>(residual)),
  });
  return Status::Ok();
}

template <typename T>
PlaneView<T> SlicePlane(PlaneView<T> plane, size_t x, size_t y,
                        Extent2D extent) {
  return {
    plane.Row(y) + x,
    extent,
    plane.stride,
  };
}

Status ValidateAndCollectAnchors(const SimpleAcMetadataInput& input,
                                 std::vector<StrategyAnchor>* anchors) {
  const Extent2D extent = input.raw_quant_field.extent;
  size_t block_count = 0;
  if (input.strategies == nullptr || !input.strategies->valid()
      || !IsValidGroupExtent(extent, &block_count)
      || !input.raw_quant_field.valid() || !input.epf_sharpness.valid()
      || input.epf_sharpness.extent != extent
      || input.block_x > input.strategies->extent().width
      || input.block_y > input.strategies->extent().height
      || extent.width > input.strategies->extent().width - input.block_x
      || extent.height > input.strategies->extent().height - input.block_y) {
    return Status::InvalidArgument("AC-metadata block views are invalid");
  }

  const Extent2D expected_map_extent =
    extent.ceil_div(kColorTileDimension / kJxlBlockDimension);
  if (!input.y_to_x_map.valid() || !input.y_to_b_map.valid()
      || input.y_to_x_map.extent != expected_map_extent
      || input.y_to_b_map.extent != expected_map_extent) {
    return Status::InvalidArgument("AC-metadata CfL views are invalid");
  }

  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      const int32_t quant = input.raw_quant_field.Row(y)[x];
      if (quant < 1 || quant > 256 || input.epf_sharpness.Row(y)[x] >= 8) {
        return Status::InvalidArgument(
          "AC-metadata quant or sharpness value is invalid");
      }
    }
  }

  std::vector<uint8_t> covered(block_count, 0);
  anchors->clear();
  anchors->reserve(block_count);
  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      AcStrategyCell cell;
      Status status =
        input.strategies->Get(input.block_x + x, input.block_y + y, &cell);
      if (!status.ok()) {
        return Status::InvalidArgument(
          "AC-metadata strategy rectangle is incomplete");
      }
      if (!cell.is_anchor) {
        continue;
      }
      const AcStrategyInfo* info = GetAcStrategyInfo(cell.strategy);
      if (!IsSimpleStrategy(cell.strategy) || info == nullptr
          || info->covered_blocks.width > extent.width - x
          || info->covered_blocks.height > extent.height - y) {
        return Status::InvalidArgument(
          "AC-metadata strategy is unsupported or crosses its group");
      }

      for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
        for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
          AcStrategyCell covered_cell;
          status = input.strategies->Get(input.block_x + x + dx,
                                         input.block_y + y + dy, &covered_cell);
          const size_t index = (y + dy) * extent.width + x + dx;
          if (!status.ok() || covered[index] != 0
              || covered_cell.strategy != cell.strategy
              || covered_cell.is_anchor != (dx == 0 && dy == 0)) {
            return Status::InvalidArgument(
              "AC-metadata strategy coverage is inconsistent");
          }
          covered[index] = 1;
        }
      }
      anchors->push_back({
        x,
        y,
        static_cast<int32_t>(cell.strategy),
      });
    }
  }

  if (anchors->empty() || std::ranges::any_of(covered, [](uint8_t value) {
        return value == 0;
      })) {
    return Status::InvalidArgument(
      "AC-metadata rectangle is not covered by local anchors");
  }
  return Status::Ok();
}

Status AppendCflTokens(ConstPlaneI8View map, uint32_t context,
                       std::vector<EntropyToken>* tokens) {
  for (size_t y = 0; y < map.extent.height; ++y) {
    for (size_t x = 0; x < map.extent.width; ++x) {
      const int32_t left = x != 0   ? map.Row(y)[x - 1]
                           : y != 0 ? map.Row(y - 1)[x]
                                    : 0;
      const int32_t top = y != 0 ? map.Row(y - 1)[x] : left;
      const int32_t top_left = x != 0 && y != 0 ? map.Row(y - 1)[x - 1] : left;
      const Status status = AppendResidual(
        context, map.Row(y)[x], ClampedGradient(top, left, top_left), tokens);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return Status::Ok();
}

}  // namespace

Status TokenizeSimpleDcGroup(ConstImage3I32View quantized_dc,
                             std::vector<EntropyToken>* tokens) {
  if (tokens == nullptr) {
    return Status::InvalidArgument("DC token output is null");
  }
  size_t block_count = 0;
  if (!quantized_dc.valid()
      || !IsValidGroupExtent(quantized_dc.extent(), &block_count)
      || block_count > std::numeric_limits<size_t>::max() / 3) {
    return Status::InvalidArgument("Quantized DC group view is invalid");
  }

  try {
    std::vector<EntropyToken> candidate;
    candidate.reserve(block_count * 3);
    constexpr std::array<size_t, 3> kChannelOrder = {1, 0, 2};
    for (const size_t channel : kChannelOrder) {
      const ConstPlaneI32View plane = quantized_dc.plane[channel];
      for (size_t y = 0; y < plane.extent.height; ++y) {
        for (size_t x = 0; x < plane.extent.width; ++x) {
          const int32_t left = x != 0   ? plane.Row(y)[x - 1]
                               : y != 0 ? plane.Row(y - 1)[x]
                                        : 0;
          const int32_t top = y != 0 ? plane.Row(y - 1)[x] : left;
          const int32_t top_left =
            x != 0 && y != 0 ? plane.Row(y - 1)[x - 1] : left;
          const int32_t prediction = ClampedGradient(top, left, top_left);
          const uint32_t context =
            GradientContext(int64_t{512} + top + left - top_left);
          const Status status =
            AppendResidual(context, plane.Row(y)[x], prediction, &candidate);
          if (!status.ok()) {
            return status;
          }
        }
      }
    }
    *tokens = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status TokenizeSimpleAcMetadata(const SimpleAcMetadataInput& input,
                                std::vector<EntropyToken>* tokens,
                                size_t* transform_anchor_count) {
  if (tokens == nullptr || transform_anchor_count == nullptr) {
    return Status::InvalidArgument("AC-metadata output is null");
  }

  try {
    std::vector<StrategyAnchor> anchors;
    Status status = ValidateAndCollectAnchors(input, &anchors);
    if (!status.ok()) {
      return status;
    }

    size_t block_count = 0;
    size_t map_count = 0;
    if (!input.raw_quant_field.extent.try_area(&block_count)
        || !input.y_to_x_map.extent.try_area(&map_count)
        || map_count > (std::numeric_limits<size_t>::max() - block_count) / 2
        || anchors.size() > (std::numeric_limits<size_t>::max() - block_count
                             - 2 * map_count)
                              / 2) {
      return Status::OutOfMemory("AC-metadata token count overflow");
    }

    std::vector<EntropyToken> candidate;
    candidate.reserve(2 * map_count + 2 * anchors.size() + block_count);
    status = AppendCflTokens(input.y_to_x_map, 2, &candidate);
    if (!status.ok()) {
      return status;
    }
    status = AppendCflTokens(input.y_to_b_map, 1, &candidate);
    if (!status.ok()) {
      return status;
    }

    int32_t previous = 0;
    for (const StrategyAnchor& anchor : anchors) {
      const uint32_t context = previous > 11  ? 7
                               : previous > 5 ? 8
                               : previous > 3 ? 9
                                              : 10;
      candidate.push_back({context, PackSigned(anchor.code)});
      previous = anchor.code;
    }

    AcStrategyCell origin;
    status = input.strategies->Get(input.block_x, input.block_y, &origin);
    if (!status.ok()) {
      return Status::InvalidArgument("AC-metadata strategy origin is invalid");
    }
    previous = static_cast<int32_t>(origin.strategy);
    for (const StrategyAnchor& anchor : anchors) {
      const int32_t current = input.raw_quant_field.Row(anchor.y)[anchor.x] - 1;
      const uint32_t context = previous > 11  ? 3
                               : previous > 5 ? 4
                               : previous > 3 ? 5
                                              : 6;
      status = AppendResidual(context, current, previous, &candidate);
      if (!status.ok()) {
        return status;
      }
      previous = current;
    }

    for (size_t y = 0; y < input.epf_sharpness.extent.height; ++y) {
      for (size_t x = 0; x < input.epf_sharpness.extent.width; ++x) {
        candidate.push_back({
          0,
          PackSigned(input.epf_sharpness.Row(y)[x]),
        });
      }
    }

    *tokens = std::move(candidate);
    *transform_anchor_count = anchors.size();
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status TokenizeSimpleDcGroups(const VarDctEncoderFrame& frame,
                              std::vector<SimpleDcGroupTokenStreams>* groups) {
  if (groups == nullptr) {
    return Status::InvalidArgument("DC-group token output is null");
  }
  Status status = ValidateSimpleCodestreamFrame(frame);
  if (!status.ok()) {
    return status;
  }

  return codestream_internal::TokenizeSimpleDcGroupsForEncoder(
    frame, groups);
}

Status codestream_internal::TokenizeSimpleDcGroupsForEncoder(
  const VarDctEncoderFrame& frame,
  std::vector<SimpleDcGroupTokenStreams>* groups) {

  if (groups == nullptr) {
    return Status::InvalidArgument("DC-group token output is null");
  }

  const Extent2D blocks = frame.geometry().block_grid().blocks;
  const Extent2D group_extent = blocks.ceil_div(kSimpleDcGroupBlockDimension);
  size_t group_count = 0;
  if (!group_extent.try_area(&group_count)) {
    return Status::OutOfMemory("DC-group count overflow");
  }

  try {
    std::vector<SimpleDcGroupTokenStreams> candidate;
    candidate.reserve(group_count);
    const ConstImage3I32View dc = frame.quantized_dc();
    const ConstPlaneI32View quant = frame.raw_quant_field();
    const ConstPlaneU8View sharpness = frame.epf_sharpness();
    const ConstPlaneI8View y_to_x = frame.color_correlation().y_to_x_map();
    const ConstPlaneI8View y_to_b = frame.color_correlation().y_to_b_map();

    for (size_t group_y = 0; group_y < group_extent.height; ++group_y) {
      for (size_t group_x = 0; group_x < group_extent.width; ++group_x) {
        SimpleDcGroupTokenStreams stream;
        stream.block_x = group_x * kSimpleDcGroupBlockDimension;
        stream.block_y = group_y * kSimpleDcGroupBlockDimension;
        stream.block_extent = {
          std::min(kSimpleDcGroupBlockDimension, blocks.width - stream.block_x),
          std::min(kSimpleDcGroupBlockDimension,
                   blocks.height - stream.block_y),
        };

        ConstImage3I32View group_dc;
        for (size_t channel = 0; channel < 3; ++channel) {
          group_dc.plane[channel] =
            SlicePlane(dc.plane[channel], stream.block_x, stream.block_y,
                       stream.block_extent);
        }
        Status status = TokenizeSimpleDcGroup(group_dc, &stream.dc_tokens);
        if (!status.ok()) {
          return status;
        }

        const size_t tile_x =
          stream.block_x / (kColorTileDimension / kJxlBlockDimension);
        const size_t tile_y =
          stream.block_y / (kColorTileDimension / kJxlBlockDimension);
        const Extent2D tile_extent = stream.block_extent.ceil_div(
          kColorTileDimension / kJxlBlockDimension);
        const SimpleAcMetadataInput metadata{
          .y_to_x_map = SlicePlane(y_to_x, tile_x, tile_y, tile_extent),
          .y_to_b_map = SlicePlane(y_to_b, tile_x, tile_y, tile_extent),
          .strategies = &frame.strategies(),
          .block_x = stream.block_x,
          .block_y = stream.block_y,
          .raw_quant_field = SlicePlane(quant, stream.block_x, stream.block_y,
                                        stream.block_extent),
          .epf_sharpness = SlicePlane(sharpness, stream.block_x, stream.block_y,
                                      stream.block_extent),
        };
        status = TokenizeSimpleAcMetadata(metadata, &stream.ac_metadata_tokens,
                                          &stream.transform_anchor_count);
        if (!status.ok()) {
          return status;
        }
        candidate.push_back(std::move(stream));
      }
    }
    *groups = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status WriteSimpleDcGroupModularHeader(BitWriter* writer) {
  if (writer == nullptr) {
    return Status::InvalidArgument("DC modular-header writer is null");
  }
  return writer->WithMaxBits(6, [&]() {
    Status status = writer->WriteBits(2, 0);
    if (!status.ok()) {
      return status;
    }
    return writer->WriteBits(4, 3);
  });
}

Status WriteSimpleAcMetadataModularHeader(Extent2D block_extent,
                                          size_t transform_anchor_count,
                                          BitWriter* writer) {
  if (writer == nullptr) {
    return Status::InvalidArgument("AC-metadata header writer is null");
  }
  size_t block_count = 0;
  if (!IsValidGroupExtent(block_extent, &block_count)
      || transform_anchor_count == 0 || transform_anchor_count > block_count) {
    return Status::InvalidArgument(
      "AC-metadata transform-anchor count is invalid");
  }
  const size_t count_bits = std::bit_width(block_count - 1);
  return writer->WithMaxBits(count_bits + 4, [&]() {
    Status status = Status::Ok();
    if (count_bits != 0) {
      status = writer->WriteBits(count_bits, transform_anchor_count - 1);
      if (!status.ok()) {
        return status;
      }
    }
    return writer->WriteBits(4, 3);
  });
}

}  // namespace gjxl
