// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include "codec/vardct_frame_view_internal.h"
#include "codestream/coefficient_order.h"

namespace gjxl_test {

// Scalar CPU oracle over group-major coefficients, independent of GPU batches,
// destination mapping, threadgroup reductions, and sample-flag storage.
struct OrderPopulationFixture {
  std::array<uint32_t, 6144> counts{};
  uint16_t mask = 0;
  std::vector<size_t> group_used;

  explicit OrderPopulationFixture(gjxl::vardct_frame_internal::VarDctFrameView frame) {
    const auto checked = [](gjxl::Status s) {
      if (!s.ok()) throw std::runtime_error(std::string(s.message()));
    };
    checked(frame.strategies().ForEachAnchor([&](size_t, size_t, gjxl::AcStrategyType strategy) {
      mask |= uint16_t{1} << gjxl::codestream_internal::kSimpleStrategyOrder[static_cast<size_t>(strategy)];
      return gjxl::Status::Ok();
    }));
    uint64_t a = 0x94D049BB133111EBull, b = 0xBF58476D1CE4E5B9ull;
    for (size_t g = 0; g < frame.ac_group_count(); ++g) {
      gjxl::VarDctAcGroupView group;
      checked(frame.GetAcGroup(g, &group));
      group_used.push_back(group.used_coefficient_count);
      size_t offset = 0;
      for (size_t y = 0; y < group.block_extent.height; ++y) {
        for (size_t x = 0; x < group.block_extent.width; ++x) {
          gjxl::AcStrategyCell cell;
          checked(frame.strategies().Get(group.block_x + x, group.block_y + y, &cell));
          if (!cell.is_anchor) continue;
          const size_t n = gjxl::GetAcStrategyInfo(cell.strategy)->coefficient_count();
          const size_t base = n == 64 ? 0 : n == 256 ? 64 : n == 1024 ? 320 : n == 128 ? 1344 : 1472;
          bool sampled = false;
          if (mask == 1) {
            const uint64_t bits = a + b, old_b = b;
            a ^= a << 23;
            b = a ^ old_b ^ (a >> 18) ^ (old_b >> 5);
            a = old_b;
            sampled = (bits >> 32) <= 0x7FFFFFFFull;
          }
          for (size_t c = 0; c < 3; ++c) {
            for (size_t i = 0; i < n; ++i) {
              if (group.coefficients[c][offset + i] == 0) {
                ++counts[c * 1984 + base + i];
                if (sampled) ++counts[5952 + c * 64 + i];
              }
            }
          }
          offset += n;
        }
      }
      if (offset != group.used_coefficient_count) throw std::runtime_error("Oracle group length differs");
    }
  }

  // Both this fixture and the original frame must outlive the borrowed view.
  gjxl::vardct_frame_internal::VarDctFrameView WithPopulation(
      gjxl::vardct_frame_internal::VarDctFrameView frame,
      gjxl::vardct_frame_internal::CoefficientOrderPopulationView population) const {
    gjxl::VarDctAcGroupView first;
    if (!frame.GetAcGroup(0, &first).ok()) throw std::runtime_error("Missing first group");
    return gjxl::vardct_frame_internal::VarDctFrameView({
      .input = {
        .geometry = frame.geometry(), .strategies = &frame.strategies(),
        .raw_quant_field = frame.raw_quant_field(), .quantizer = &frame.quantizer(),
        .color_correlation = &frame.color_correlation(), .epf_sharpness = frame.epf_sharpness(),
      },
      .profile = frame.profile(), .quantized_dc = frame.quantized_dc(), .dc = frame.dc(),
      .ac_group_extent = frame.ac_group_extent(), .group_used_coefficient_count = group_used,
      .ac_coefficients = {first.coefficients[0].data(), frame.ac_group_count() * 3 * 65536},
      .coefficient_order_population = population,
    });
  }
};

}  // namespace gjxl_test
