// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "codec/vardct_frame_internal.h"
#include "codestream/encoder.h"

// Deterministic completed frames shared by independent serializer fixtures.
namespace gjxl_test {

constexpr std::array kStrategies = {
  gjxl::AcStrategyType::kDct8, gjxl::AcStrategyType::kDct16x16,
  gjxl::AcStrategyType::kDct32x32, gjxl::AcStrategyType::kDct16x8,
  gjxl::AcStrategyType::kDct8x16, gjxl::AcStrategyType::kDct32x16,
  gjxl::AcStrategyType::kDct16x32,
};

inline void Check(gjxl::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

template <typename T>
gjxl::PlaneView<const T> View(const std::vector<T>& data, gjxl::Extent2D size) {
  return {data.data(), size, size.width};
}

inline gjxl::VarDctEncoderFrame MakeFrame(size_t strategy_index, size_t pattern,
                                 size_t block_side = 36) {
  gjxl::FrameGeometry geometry;
  Check(gjxl::FrameGeometry::Create(
    block_side * 8 - 3, block_side * 8 - 7, &geometry));
  const auto blocks = geometry.block_grid().blocks;
  gjxl::AcStrategyGrid grid;
  Check(gjxl::AcStrategyGrid::Create(blocks, &grid));
  // Four groups, including narrow right/bottom groups; mixed tiles retain
  // both orientations of the shared rectangular order families.
  for (size_t y = 0; y < blocks.height; y += 4) {
    for (size_t x = 0; x < blocks.width; x += 4) {
      const auto strategy = kStrategies[strategy_index < kStrategies.size()
        ? strategy_index : (x / 4 + 3 * (y / 4)) % kStrategies.size()];
      const auto covered = gjxl::GetAcStrategyInfo(strategy)->covered_blocks;
      for (size_t dy = 0; dy < 4; dy += covered.height) {
        for (size_t dx = 0; dx < 4; dx += covered.width) {
          Check(grid.Set(x + dx, y + dy, strategy));
        }
      }
    }
  }
  std::vector<gjxl::vardct_frame_internal::QuantizedAcTransformLayout> layouts;
  std::vector<int32_t> coefficients;
  uint32_t random = 0x7f4a7c15u;
  Check(grid.ForEachAnchor([&](size_t x, size_t y, gjxl::AcStrategyType type) {
    const size_t count = gjxl::GetAcStrategyInfo(type)->coefficient_count();
    gjxl::vardct_frame_internal::QuantizedAcTransformLayout layout{
      .block_x = x, .block_y = y, .strategy = type, .coefficient_count = count};
    for (size_t channel = 0; channel < 3; ++channel) {
      layout.coefficient_offsets[channel] = coefficients.size();
      for (size_t index = 0; index < count; ++index) {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        int32_t value = 0;
        switch (pattern) {
          case 0: break;  // Every coefficient zero: stable natural-order ties.
          case 1: value = (index & 1) ? -1 : 1; break;  // No zeros.
          case 2: value = (random & 7) == 0 ? -17 : 0; break;
          case 3: value = (random & 7) == 0 ? 0 : 29; break;
          case 4: value = ((index + channel + x + y) % 5) == 0 ? 0 : -3; break;
          default:
            value = (random & 3) == 0 ? 0 : (random & 1)
              ? std::numeric_limits<int32_t>::min()
              : std::numeric_limits<int32_t>::max();
            break;
        }
        coefficients.push_back(value);
      }
    }
    layouts.push_back(layout);
    return gjxl::Status::Ok();
  }));
  const size_t area = blocks.width * blocks.height;
  const std::vector<int32_t> quant(area, 29), dc(area, 0);
  const std::vector<uint8_t> sharpness(area, 4);
  const auto tiles = gjxl::ColorTileExtent(geometry.padded_frame());
  const std::vector<int8_t> correlation(tiles.width * tiles.height, 0);
  gjxl::Quantizer quantizer;
  Check(gjxl::Quantizer::Create({3541, 10}, &quantizer));
  gjxl::VarDctEncoderFrame frame;
  Check(gjxl::vardct_frame_internal::AssembleVarDctEncoderFrame({
    .geometry = geometry, .strategies = &grid,
    .raw_quant_field = View(quant, blocks), .quantizer = &quantizer,
    .y_to_x = View(correlation, tiles), .y_to_b = View(correlation, tiles),
    .epf_sharpness = View(sharpness, blocks),
    .quantized_dc = {{View(dc, blocks), View(dc, blocks), View(dc, blocks)}},
    .quantized_ac = coefficients, .transforms = layouts,
  }, &frame));
  Check(gjxl::ValidateSimpleCodestreamFrame(frame));
  return frame;
}

}  // namespace gjxl_test
