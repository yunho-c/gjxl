// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include "codec/vardct_frame_internal.h"
#include "codestream/coefficient_order.h"
#include "codestream/encoder.h"

namespace gjxl_test {

inline void PopulationCheck(gjxl::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

// Independent scalar recount of public, completed AC groups. The family table
// comes from the serializer, not the GPU batch implementation.
inline gjxl::vardct_frame_internal::CoefficientOrderPopulation ReferencePopulation(
  const gjxl::VarDctEncoderFrame& frame) {
  using namespace gjxl::vardct_frame_internal;
  CoefficientOrderPopulation result;
  PopulationCheck(frame.strategies().ForEachAnchor(
    [&](size_t, size_t, gjxl::AcStrategyType strategy) {
      const size_t family = gjxl::codestream_internal::kSimpleStrategyOrder[static_cast<size_t>(strategy)];
      if (family != OrderPopulationFamily(gjxl::GetAcStrategyInfo(strategy)->coefficient_count()))
        throw std::runtime_error("Population/serializer family mapping differs");
      result.present_mask |= uint16_t{1} << family;
      return gjxl::Status::Ok();
    }));
  uint64_t a = 0x94D049BB133111EBull, b = 0xBF58476D1CE4E5B9ull;
  for (size_t group_index = 0; group_index < frame.ac_group_count(); ++group_index) {
    gjxl::VarDctAcGroupView group;
    PopulationCheck(frame.GetAcGroup(group_index, &group));
    size_t offset = 0;
    for (size_t y = 0; y < group.block_extent.height; ++y) {
      for (size_t x = 0; x < group.block_extent.width; ++x) {
        gjxl::AcStrategyCell cell;
        PopulationCheck(frame.strategies().Get(group.block_x + x, group.block_y + y, &cell));
        if (!cell.is_anchor) continue;
        const size_t family = gjxl::codestream_internal::kSimpleStrategyOrder[static_cast<size_t>(cell.strategy)];
        const size_t count = gjxl::GetAcStrategyInfo(cell.strategy)->coefficient_count();
        bool sampled = false;
        if (result.present_mask == 1) {
          const uint64_t bits = a + b, old_b = b;
          a ^= a << 23;
          b = a ^ old_b ^ (a >> 18) ^ (old_b >> 5);
          a = old_b;
          sampled = (bits >> 32) <= (UINT64_MAX >> 32) / 2;
        }
        for (size_t channel = 0; channel < 3; ++channel) {
          for (size_t i = 0; i < count; ++i) {
            const bool zero = group.coefficients[channel][offset + i] == 0;
            result.counts[channel * kOrderPopulationStride + kOrderPopulationOffsets[family] + i] += zero;
            if (sampled) result.counts[kOrderPopulationFullCount + channel * 64 + i] += zero;
          }
        }
        offset += count;
      }
    }
    if (offset != group.used_coefficient_count) throw std::runtime_error("Population fixture group consumption differs");
  }
  return result;
}

struct PopulationAssembly {
  const gjxl::VarDctEncoderFrame& source;
  gjxl::OverwriteArray<int32_t> coefficients;
  std::vector<gjxl::vardct_frame_internal::QuantizedAcTransformLayout> layouts;

  explicit PopulationAssembly(const gjxl::VarDctEncoderFrame& frame) : source(frame) {
    constexpr size_t capacity = gjxl::kVarDctAcGroupCoefficientCapacity;
    coefficients.ResetForOverwrite(frame.ac_group_count() * 3 * capacity);
    for (size_t group_index = 0; group_index < frame.ac_group_count(); ++group_index) {
      gjxl::VarDctAcGroupView group;
      PopulationCheck(frame.GetAcGroup(group_index, &group));
      for (size_t channel = 0; channel < 3; ++channel)
        std::copy(group.coefficients[channel].begin(), group.coefficients[channel].end(),
          coefficients.data() + (group_index * 3 + channel) * capacity);
    }
    std::vector<size_t> used(frame.ac_group_count(), 0);
    PopulationCheck(frame.strategies().ForEachAnchor([&](size_t x, size_t y, gjxl::AcStrategyType strategy) {
      const size_t group = (y / 32) * frame.ac_group_extent().width + x / 32;
      const size_t count = gjxl::GetAcStrategyInfo(strategy)->coefficient_count();
      gjxl::vardct_frame_internal::QuantizedAcTransformLayout layout{
        .block_x = x, .block_y = y, .strategy = strategy, .coefficient_count = count};
      for (size_t channel = 0; channel < 3; ++channel)
        layout.coefficient_offsets[channel] = (group * 3 + channel) * capacity + used[group];
      used[group] += count;
      layouts.push_back(layout);
      return gjxl::Status::Ok();
    }));
  }

  gjxl::vardct_frame_internal::QuantizedFrameAssemblyInput Input(
    const gjxl::vardct_frame_internal::CoefficientOrderPopulation* population,
    bool owned = true) {
    return {.geometry = source.geometry(), .strategies = &source.strategies(),
      .raw_quant_field = source.raw_quant_field(), .quantizer = &source.quantizer(),
      .y_to_x = source.color_correlation().y_to_x_map(),
      .y_to_b = source.color_correlation().y_to_b_map(),
      .epf_sharpness = source.epf_sharpness(), .profile = source.profile(),
      .quantized_dc = source.quantized_dc(), .quantized_ac = {coefficients.data(), coefficients.size()},
      .transforms = layouts, .ac_group_storage = owned ? &coefficients : nullptr,
      .coefficient_order_population = population};
  }
};

inline gjxl::VarDctEncoderFrame FrameWithPopulation(const gjxl::VarDctEncoderFrame& frame, bool owned = true) {
  using namespace gjxl::vardct_frame_internal;
  auto population = ReferencePopulation(frame);
  const auto expected = population;
  PopulationAssembly assembly(frame);
  gjxl::VarDctEncoderFrame result;
  PopulationCheck(AssembleVarDctEncoderFrame(assembly.Input(&population, owned), &result));
  if (owned && assembly.coefficients.size()) throw std::runtime_error("Owned population assembly did not consume AC");
  population.counts.fill(UINT32_MAX);
  population.present_mask = 0;
  const auto* stored = GetCoefficientOrderPopulation(result);
  if (!stored || stored->counts != expected.counts || stored->present_mask != expected.present_mask)
    throw std::runtime_error("Frame borrowed mutable population input");
  return result;
}

inline bool CheckResidentPopulation(const gjxl::VarDctEncoderFrame& frame) {
  try {
    using namespace gjxl::vardct_frame_internal;
    const auto* stored = GetCoefficientOrderPopulation(frame);
    const auto blocks = frame.geometry().block_grid().blocks;
    if (blocks.width < 5 && blocks.height < 5) return stored == nullptr;
    const auto expected = ReferencePopulation(frame);
    if (!stored || stored->counts != expected.counts || stored->present_mask != expected.present_mask) return false;
    // Strip the cache through ordinary checked assembly, then compare both
    // later caller policies and their encoded order tokens.
    PopulationAssembly assembly(frame);
    gjxl::VarDctEncoderFrame uncached;
    PopulationCheck(AssembleVarDctEncoderFrame(assembly.Input(nullptr), &uncached));
    for (auto behavior : {gjxl::VarDctCoefficientOrderBehavior::kFull,
                         gjxl::VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled}) {
      gjxl::SimpleCoefficientOrders actual, reference;
      PopulationCheck(gjxl::codestream_internal::ComputeSimpleCoefficientOrdersForEncoder(frame, behavior, &actual));
      PopulationCheck(gjxl::codestream_internal::ComputeSimpleCoefficientOrdersForEncoder(uncached, behavior, &reference));
      if (actual != reference) return false;
      std::vector<gjxl::EntropyToken> actual_tokens, reference_tokens;
      PopulationCheck(gjxl::TokenizeSimpleCoefficientOrders(actual, &actual_tokens));
      PopulationCheck(gjxl::TokenizeSimpleCoefficientOrders(reference, &reference_tokens));
      if (actual_tokens != reference_tokens) return false;
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace gjxl_test
