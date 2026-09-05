// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

// Compare frame-wide zero counting with a coefficient-major scalar oracle.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "codec/vardct_frame_internal.h"
#include "codestream/ac_group.h"
#include "codestream/coefficient_order.h"
#include "codestream/encoder.h"

namespace {

constexpr std::array kStrategies = {
  gjxl::AcStrategyType::kDct8, gjxl::AcStrategyType::kDct16x16,
  gjxl::AcStrategyType::kDct32x32, gjxl::AcStrategyType::kDct16x8,
  gjxl::AcStrategyType::kDct8x16, gjxl::AcStrategyType::kDct32x16,
  gjxl::AcStrategyType::kDct16x32,
};

void Check(gjxl::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

template <typename T>
gjxl::PlaneView<const T> View(const std::vector<T>& data, gjxl::Extent2D size) {
  return {data.data(), size, size.width};
}

gjxl::VarDctEncoderFrame MakeFrame(size_t strategy_index, size_t pattern,
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

gjxl::SimpleCoefficientOrders ReferenceOrders(
  const gjxl::VarDctEncoderFrame& frame,
  gjxl::VarDctCoefficientOrderBehavior behavior) {
  gjxl::SimpleCoefficientOrders result;
  const auto blocks = frame.geometry().block_grid().blocks;
  if (blocks.width < 5 && blocks.height < 5) return result;
  uint16_t present = 0;
  Check(frame.strategies().ForEachAnchor(
    [&](size_t, size_t, gjxl::AcStrategyType strategy) {
      present |= uint16_t{1} << gjxl::codestream_internal::kSimpleStrategyOrder[
        static_cast<size_t>(strategy)];
      return gjxl::Status::Ok();
    }));
  const bool sample = present == 1 && behavior ==
    gjxl::VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled;
  uint64_t a = 0x94D049BB133111EBull, b = 0xBF58476D1CE4E5B9ull;
  std::array<std::array<std::vector<std::span<const int32_t>>, 3>,
             gjxl::codestream_internal::kSimpleCoefficientOrderCount> selected;
  std::array<gjxl::AcStrategyType,
             gjxl::codestream_internal::kSimpleCoefficientOrderCount> representatives{};
  for (size_t group_index = 0; group_index < frame.ac_group_count(); ++group_index) {
    gjxl::VarDctAcGroupView group;
    Check(frame.GetAcGroup(group_index, &group));
    size_t offset = 0;
    for (size_t y = 0; y < group.block_extent.height; ++y) {
      for (size_t x = 0; x < group.block_extent.width; ++x) {
        gjxl::AcStrategyCell cell;
        Check(frame.strategies().Get(group.block_x + x, group.block_y + y, &cell));
        if (!cell.is_anchor) continue;
        const size_t family = gjxl::codestream_internal::kSimpleStrategyOrder[
          static_cast<size_t>(cell.strategy)];
        representatives[family] = cell.strategy;
        const size_t count = gjxl::GetAcStrategyInfo(cell.strategy)->coefficient_count();
        bool include = true;
        if (sample) {
          const uint64_t bits = a + b;
          const uint64_t old_b = b;
          a ^= a << 23;
          b = a ^ old_b ^ (a >> 18) ^ (old_b >> 5);
          a = old_b;
          include = (bits >> 32) <=
            (std::numeric_limits<uint64_t>::max() >> 32) / 2;
        }
        if (include) {
          for (size_t channel = 0; channel < 3; ++channel) {
            selected[family][channel].push_back(
              group.coefficients[channel].subspan(offset, count));
          }
        }
        offset += count;
      }
    }
  }
  for (size_t family = 0; family < selected.size(); ++family) {
    if (!(present & (uint16_t{1} << family))) continue;
    const auto* info = gjxl::GetAcStrategyInfo(representatives[family]);
    std::vector<uint32_t> natural;
    Check(gjxl::ComputeSimpleNaturalCoefficientOrder(info->type, &natural));
    const size_t skip = info->covered_blocks.width * info->covered_blocks.height;
    const float scale = 1.0f / std::sqrt(static_cast<float>(natural.size()));
    bool custom = false;
    for (size_t channel = 0; channel < 3; ++channel) {
      // Transpose the production traversal: count one coefficient column
      // across all selected transforms, then sort precomputed integer keys.
      std::vector<uint64_t> keys(natural.size());
      for (size_t coefficient = 0; coefficient < natural.size(); ++coefficient) {
        const auto zeros = std::count_if(
          selected[family][channel].begin(), selected[family][channel].end(),
          [coefficient](auto transform) { return transform[coefficient] == 0; });
        keys[coefficient] = static_cast<uint64_t>(
          static_cast<float>(zeros) * scale + 0.1f);
      }
      auto& order = result.orders[family][channel];
      order = natural;
      std::stable_sort(order.begin() + static_cast<ptrdiff_t>(skip), order.end(),
        [&](uint32_t left, uint32_t right) { return keys[left] < keys[right]; });
      custom |= order != natural;
    }
    if (custom) result.used_order_mask |= uint16_t{1} << family;
    else for (auto& order : result.orders[family]) order.clear();
  }
  return result;
}

}  // namespace

int main() {
  try {
    size_t cases = 0;
    for (size_t strategy = 0; strategy <= kStrategies.size(); ++strategy) {
      for (size_t pattern = 0; pattern < 6; ++pattern) {
        const auto frame = MakeFrame(strategy, pattern);
        for (const auto behavior : {gjxl::VarDctCoefficientOrderBehavior::kFull,
               gjxl::VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled}) {
          gjxl::SimpleCoefficientOrders orders;
          Check(gjxl::codestream_internal::ComputeSimpleCoefficientOrdersForEncoder(
            frame, behavior, &orders));
          const auto reference = ReferenceOrders(frame, behavior);
          if (orders != reference) throw std::runtime_error("Coefficient orders differ");
          std::vector<gjxl::EntropyToken> tokens, reference_tokens;
          Check(gjxl::TokenizeSimpleCoefficientOrders(orders, &tokens));
          Check(gjxl::TokenizeSimpleCoefficientOrders(reference, &reference_tokens));
          if (tokens != reference_tokens) throw std::runtime_error("Order tokens differ");
          if (pattern < 2 && orders.used_order_mask != 0)
            throw std::runtime_error("Constant zero populations changed tie order");
          ++cases;
        }
      }
    }
    const auto small = MakeFrame(0, 5, 4);
    gjxl::SimpleCoefficientOrders orders;
    Check(gjxl::ComputeSimpleCoefficientOrders(small, &orders));
    if (orders.used_order_mask != 0)
      throw std::runtime_error("Small frame selected a custom order");
    const auto frame = MakeFrame(kStrategies.size(), 2);
    Check(gjxl::ComputeSimpleCoefficientOrders(frame, &orders));
    const auto sentinel = orders;
    if (gjxl::ComputeSimpleCoefficientOrders({}, &orders).ok() ||
        orders != sentinel ||
        gjxl::codestream_internal::ComputeSimpleCoefficientOrdersForEncoder(
          frame, static_cast<gjxl::VarDctCoefficientOrderBehavior>(255),
          &orders).ok() || orders != sentinel ||
        gjxl::ComputeSimpleCoefficientOrders(frame, nullptr).ok())
      throw std::runtime_error("Invalid coefficient-order input was not atomic");
    std::cout << "Verified " << cases << " frame coefficient-order cases, "
              << "small-frame cutoff, and invalid-input atomicity.\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
