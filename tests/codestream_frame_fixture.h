// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <vector>

#include "codec/vardct_frame_view_internal.h"

namespace gjxl::codestream_test_internal {

inline bool FixtureOk(const Status &status) {
  if (!status.ok())
    std::cerr << status.message() << '\n';
  return status.ok();
}
inline bool FixtureCheck(bool condition, const char *message) {
  if (!condition)
    std::cerr << message << '\n';
  return condition;
}

// Caller-owned synthetic group-major backing. Construct outside the reservation
// under test; LLF slots remain zero for every transform and channel.
constexpr std::array<AcStrategyType, 7> kStrategies{
    AcStrategyType::kDct8,     AcStrategyType::kDct16x16,
    AcStrategyType::kDct32x32, AcStrategyType::kDct16x8,
    AcStrategyType::kDct8x16,  AcStrategyType::kDct32x16,
    AcStrategyType::kDct16x32};

struct FrameFixture {
  FrameGeometry geometry;
  AcStrategyGrid strategies;
  Quantizer quantizer;
  ColorCorrelationMap cfl;
  Extent2D blocks, groups;
  std::vector<int32_t> quant, zero_dc, coefficients;
  std::vector<float> zero_reconstruction;
  std::vector<uint8_t> sharpness;
  std::vector<size_t> used;

  bool Create(Extent2D extent, size_t family, int pattern) {
    blocks = extent;
    groups = blocks.ceil_div(32);
    if (!FixtureOk(FrameGeometry::Create({extent.width * 8, extent.height * 8},
                                         &geometry)) ||
        !FixtureOk(AcStrategyGrid::Create(extent, &strategies)) ||
        !FixtureOk(Quantizer::Create({3541, 10}, &quantizer)))
      return false;
    const size_t b = extent.width * extent.height;
    quant.resize(b);
    sharpness.resize(b);
    zero_dc.resize(b);
    zero_reconstruction.resize(b);
    for (size_t i = 0; i < b; ++i) {
      quant[i] = static_cast<int32_t>(1 + i % 256);
      sharpness[i] = static_cast<uint8_t>(i % 8);
    }
    const auto tiles = extent.ceil_div(8);
    std::vector<int8_t> map(tiles.width * tiles.height, 0);
    ConstPlaneI8View cfl_view{map.data(), tiles, tiles.width};
    if (!FixtureOk(chroma_from_luma_internal::CreateColorCorrelationMap(
            cfl_view, cfl_view, &cfl)))
      return false;
    used.resize(groups.width * groups.height);
    coefficients.resize(used.size() * 3 * 65536);
    size_t sequence = 0;
    for (size_t gy = 0; gy < groups.height; ++gy) {
      for (size_t gx = 0; gx < groups.width; ++gx) {
        const size_t g = gy * groups.width + gx;
        const Extent2D local{std::min(32ul, extent.width - gx * 32),
                             std::min(32ul, extent.height - gy * 32)};
        size_t offset = 0;
        for (size_t y = 0; y < local.height; ++y) {
          for (size_t x = 0; x < local.width; ++x) {
            if (strategies.occupied(gx * 32 + x, gy * 32 + y))
              continue;
            auto strategy = kStrategies[family < 7 ? family : sequence++ % 7];
            const auto *info = GetAcStrategyInfo(strategy);
            bool fits = info->covered_blocks.width <= local.width - x &&
                        info->covered_blocks.height <= local.height - y;
            for (size_t dy = 0; fits && dy < info->covered_blocks.height; ++dy)
              for (size_t dx = 0; fits && dx < info->covered_blocks.width; ++dx)
                fits = !strategies.occupied(gx * 32 + x + dx, gy * 32 + y + dy);
            if (!fits)
              strategy = AcStrategyType::kDct8;
            info = GetAcStrategyInfo(strategy);
            if (!FixtureOk(strategies.Set(gx * 32 + x, gy * 32 + y, strategy)))
              return false;
            for (size_t c = 0; c < 3; ++c) {
              auto *values = coefficients.data() + (g * 3 + c) * 65536 + offset;
              for (size_t i = 0; i < info->coefficient_count(); ++i) {
                const uint32_t seed = static_cast<uint32_t>(
                    (offset + i) * 1664525 + c * 1013904223);
                if (pattern == 1)
                  values[i] = static_cast<int32_t>(seed % 65535) - 32767;
                if (pattern == 2)
                  values[i] =
                      i % 17 == 0
                          ? (i % 2 == 0 ? std::numeric_limits<int32_t>::min()
                                        : std::numeric_limits<int32_t>::max())
                          : 0;
              }
              auto llf = info->low_frequency_extent();
              for (size_t iy = 0; iy < llf.height; ++iy)
                for (size_t ix = 0; ix < llf.width; ++ix)
                  values[iy * info->coefficient_extent().width + ix] = 0;
            }
            offset += info->coefficient_count();
          }
        }
        used[g] = offset;
      }
    }
    return FixtureCheck(view().valid(), "Synthetic tokenizer frame is invalid");
  }

  vardct_frame_internal::VarDctFrameView view() const {
    ConstImage3I32View dc;
    ConstImage3FView reconstruction;
    for (size_t c = 0; c < 3; ++c) {
      dc.plane[c] = {zero_dc.data(), blocks, blocks.width};
      reconstruction.plane[c] = {zero_reconstruction.data(), blocks,
                                 blocks.width};
    }
    return vardct_frame_internal::VarDctFrameView(
        {.input = {.geometry = geometry,
                   .strategies = &strategies,
                   .raw_quant_field = {quant.data(), blocks, blocks.width},
                   .quantizer = &quantizer,
                   .color_correlation = &cfl,
                   .epf_sharpness = {sharpness.data(), blocks, blocks.width}},
         .quantized_dc = dc,
         .dc = reconstruction,
         .ac_group_extent = groups,
         .group_used_coefficient_count = used,
         .ac_coefficients = coefficients});
  }
};

} // namespace gjxl::codestream_test_internal
