// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's encoder/enc_group.cc and pinned
// libjxl's lib/jxl/ac_strategy.cc.

#include "codestream/ac_group.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "codec/codestream.h"
#include "codec/vardct_frame.h"
#include "codestream/block_context_map.h"
#include "codestream/coefficient_order.h"
#include "codestream/simple_ac_context.h"

namespace gjxl {
namespace {

inline constexpr size_t kBlockContextCount = 4;

static_assert(kSimpleAcContextCount
              == kBlockContextCount
                   * (kSimpleNonzeroBucketCount +
                      kSimpleZeroDensityContextCount));

constexpr std::array<uint16_t, 64> kCoefficientFrequencyContext = {
  0xBAD, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
  15,    15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22,
  23,    23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26,
  27,    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
};

constexpr std::array<uint16_t, 64> kCoefficientNonzeroContext = {
  0xBAD, 0,   31,  62,  62,  93,  93,  93,  93,  123, 123, 123, 123,
  152,   152, 152, 152, 152, 152, 152, 152, 180, 180, 180, 180, 180,
  180,   180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
  206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
  206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
};

using StrategyAnchor = codestream_internal::SimpleAcStrategyAnchor;

Status AllocationFailure() {
  return Status::OutOfMemory("AC-group token allocation failed");
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
  return !extent.empty() && extent.width <= kVarDctAcGroupBlockDimension
         && extent.height <= kVarDctAcGroupBlockDimension
         && extent.try_area(area);
}

uint16_t NonzeroBucket(uint32_t prediction) {
  return static_cast<uint16_t>(
    prediction < 8 ? prediction : prediction >= 64 ? 36 : 4 + prediction / 2);
}

uint32_t ZeroDensityContext(int32_t nonzeros, size_t scan_index,
                            size_t covered_blocks, size_t log2_covered_blocks,
                            uint32_t previous_nonzero) {
  const size_t normalized_nonzeros =
    (static_cast<size_t>(nonzeros) + covered_blocks - 1) >> log2_covered_blocks;
  const size_t normalized_index = scan_index >> log2_covered_blocks;
  return static_cast<uint32_t>(
    (kCoefficientNonzeroContext[normalized_nonzeros]
     + kCoefficientFrequencyContext[normalized_index])
      * 2
    + previous_nonzero);
}

uint32_t PredictNonzeros(std::span<const uint8_t> map, Extent2D extent,
                         size_t x, size_t y) {
  if (x == 0) {
    return y == 0 ? 32 : map[(y - 1) * extent.width];
  }
  if (y == 0) {
    return map[x - 1];
  }
  return static_cast<uint32_t>(
    (map[(y - 1) * extent.width + x] + map[y * extent.width + x - 1] + 1) / 2);
}

uint32_t BlockContextValidated(const SimpleBlockContextMap& map,
                               const SimpleAcBlockContextKey& key) {

  size_t qf_segment = 0;
  while (qf_segment < map.qf_thresholds.size() &&
         static_cast<uint32_t>(key.raw_quant) > map.qf_thresholds[qf_segment]) {
    ++qf_segment;
  }
  return map.context_map[
    (static_cast<size_t>(key.channel_row) *
       codestream_internal::kSimpleCoefficientOrderCount +
     key.order_family) *
      (map.qf_thresholds.size() + 1) +
    qf_segment];
}

constexpr uint32_t FinalNonzeroContext(uint32_t num_contexts,
                                       uint32_t block_context,
                                       uint32_t local_context) {
  return local_context * num_contexts + block_context;
}

constexpr uint32_t FinalCoefficientContext(uint32_t num_contexts,
                                           uint32_t block_context,
                                           uint32_t local_context) {
  return num_contexts * kSimpleNonzeroBucketCount +
    kSimpleZeroDensityContextCount * block_context + local_context;
}

Status FinalAcContextFromBlockValidated(const SimpleBlockContextMap& map,
                                        uint32_t block_context,
                                        uint32_t local_context,
                                        bool is_coefficient,
                                        uint16_t* context) {
  if (block_context >= map.num_contexts) {
    return Status::InvalidArgument("AC block context is invalid");
  }
  uint32_t result = 0;
  if (is_coefficient) {
    if (local_context >= kSimpleZeroDensityContextCount) {
      return Status::InvalidArgument("AC coefficient context is invalid");
    }
    result = FinalCoefficientContext(
      map.num_contexts, block_context, local_context);
  } else {
    if (local_context >= kSimpleNonzeroBucketCount) {
      return Status::InvalidArgument("AC nonzero context is invalid");
    }
    result = FinalNonzeroContext(
      map.num_contexts, block_context, local_context);
  }
  if (result > std::numeric_limits<uint16_t>::max()) {
    return Status::Internal("AC-group context exceeds 16 bits");
  }
  *context = static_cast<uint16_t>(result);
  return Status::Ok();
}

Status ValidateAndCollectAnchors(const VarDctAcGroupView& group,
                                 const AcStrategyGrid& strategies,
                                 std::vector<StrategyAnchor>* anchors) {
  size_t block_count = 0;
  if (!strategies.valid()
      || !IsValidGroupExtent(group.block_extent, &block_count)
      || group.block_x > strategies.extent().width
      || group.block_y > strategies.extent().height
      || group.block_extent.width > strategies.extent().width - group.block_x
      || group.block_extent.height > strategies.extent().height - group.block_y
      || group.used_coefficient_count == 0) {
    return Status::InvalidArgument("AC-group view is invalid");
  }
  for (std::span<const int32_t> coefficients : group.coefficients) {
    if (coefficients.size() < group.used_coefficient_count) {
      return Status::InvalidArgument("AC-group coefficient span is too short");
    }
  }

  std::vector<uint8_t> covered(block_count, 0);
  anchors->clear();
  anchors->reserve(block_count);
  size_t expected_coefficients = 0;
  for (size_t y = 0; y < group.block_extent.height; ++y) {
    for (size_t x = 0; x < group.block_extent.width; ++x) {
      AcStrategyCell cell;
      Status status =
        strategies.Get(group.block_x + x, group.block_y + y, &cell);
      if (!status.ok()) {
        return Status::InvalidArgument(
          "AC-group strategy rectangle is incomplete");
      }
      if (!cell.is_anchor) {
        continue;
      }

      const AcStrategyInfo* info = GetAcStrategyInfo(cell.strategy);
      if (!IsSimpleStrategy(cell.strategy) || info == nullptr
          || info->covered_blocks.width > group.block_extent.width - x
          || info->covered_blocks.height > group.block_extent.height - y) {
        return Status::InvalidArgument(
          "AC strategy is unsupported or crosses its group");
      }
      if (info->coefficient_count()
          > std::numeric_limits<size_t>::max() - expected_coefficients) {
        return Status::InvalidArgument("AC-group coefficient count overflow");
      }

      for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
        for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
          AcStrategyCell covered_cell;
          status = strategies.Get(group.block_x + x + dx,
                                  group.block_y + y + dy, &covered_cell);
          const size_t index = (y + dy) * group.block_extent.width + x + dx;
          if (!status.ok() || covered[index] != 0
              || covered_cell.strategy != cell.strategy
              || covered_cell.is_anchor != (dx == 0 && dy == 0)) {
            return Status::InvalidArgument(
              "AC-group strategy coverage is inconsistent");
          }
          covered[index] = 1;
        }
      }

      anchors->push_back({
        x,
        y,
        cell.strategy,
        info->coefficient_count(),
      });
      expected_coefficients += info->coefficient_count();
    }
  }

  if (anchors->empty()
      || std::ranges::any_of(covered, [](uint8_t value) { return value == 0; })
      || expected_coefficients != group.used_coefficient_count) {
    return Status::InvalidArgument(
      "AC-group coefficient consumption is inconsistent");
  }
  return Status::Ok();
}

int32_t CountNonzerosExceptLlf(std::span<const int32_t> coefficients,
                               const AcStrategyInfo& info) {
  const Extent2D coefficient_extent = info.coefficient_extent();
  const Extent2D llf_extent = info.low_frequency_extent();
  int32_t nonzeros = 0;
  for (size_t y = 0; y < coefficient_extent.height; ++y) {
    for (size_t x = 0; x < coefficient_extent.width; ++x) {
      if (x < llf_extent.width && y < llf_extent.height) {
        continue;
      }
      nonzeros += coefficients[y * coefficient_extent.width + x] != 0;
    }
  }
  return nonzeros;
}

}  // namespace

Status ComputeSimpleNaturalCoefficientOrder(AcStrategyType strategy,
                                            std::vector<uint32_t>* order) {
  if (order == nullptr) {
    return Status::InvalidArgument("Natural coefficient-order output is null");
  }
  const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
  if (!IsSimpleStrategy(strategy) || info == nullptr) {
    return Status::InvalidArgument(
      "Natural coefficient order does not support this strategy");
  }

  try {
    size_t coefficient_blocks = info->covered_blocks.width;
    size_t coefficient_block_rows = info->covered_blocks.height;
    if (coefficient_blocks < coefficient_block_rows) {
      std::swap(coefficient_blocks, coefficient_block_rows);
    }
    if (coefficient_block_rows == 0
        || coefficient_blocks % coefficient_block_rows != 0) {
      return Status::Internal("Natural coefficient layout is invalid");
    }
    const size_t anisotropy = coefficient_blocks / coefficient_block_rows;
    if (!std::has_single_bit(anisotropy)) {
      return Status::Internal("Natural coefficient anisotropy is invalid");
    }
    const size_t anisotropy_mask = anisotropy - 1;
    const size_t anisotropy_shift = std::countr_zero(anisotropy);
    const size_t coefficient_width = coefficient_blocks * kJxlBlockDimension;
    const size_t coefficient_count = info->coefficient_count();
    const uint32_t sentinel = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> candidate(coefficient_count, sentinel);

    size_t next_index = coefficient_blocks * coefficient_block_rows;
    for (size_t diagonal = 0; diagonal < coefficient_width; ++diagonal) {
      for (size_t position = 0; position <= diagonal; ++position) {
        size_t x = position;
        size_t y = diagonal - position;
        if ((diagonal & 1u) != 0) {
          std::swap(x, y);
        }
        if ((y & anisotropy_mask) != 0) {
          continue;
        }
        y >>= anisotropy_shift;
        const size_t order_index =
          x < coefficient_blocks && y < coefficient_block_rows
            ? y * coefficient_blocks + x
            : next_index++;
        if (order_index >= candidate.size()) {
          return Status::Internal("Natural coefficient order overflowed");
        }
        candidate[order_index] =
          static_cast<uint32_t>(y * coefficient_width + x);
      }
    }

    for (size_t reverse = coefficient_width - 1; reverse > 0; --reverse) {
      const size_t diagonal = reverse - 1;
      for (size_t position = 0; position <= diagonal; ++position) {
        size_t x = coefficient_width - 1 - (diagonal - position);
        size_t y = coefficient_width - 1 - position;
        if ((diagonal & 1u) != 0) {
          std::swap(x, y);
        }
        if ((y & anisotropy_mask) != 0) {
          continue;
        }
        y >>= anisotropy_shift;
        if (next_index >= candidate.size()) {
          return Status::Internal("Natural coefficient order overflowed");
        }
        candidate[next_index++] =
          static_cast<uint32_t>(y * coefficient_width + x);
      }
    }
    if (next_index != coefficient_count) {
      return Status::Internal("Natural coefficient order is incomplete");
    }

    std::vector<uint8_t> seen(coefficient_count, 0);
    for (uint32_t index : candidate) {
      if (index >= coefficient_count || seen[index] != 0) {
        return Status::Internal(
          "Natural coefficient order is not a permutation");
      }
      seen[index] = 1;
    }
    for (size_t y = 0; y < coefficient_block_rows; ++y) {
      for (size_t x = 0; x < coefficient_blocks; ++x) {
        const size_t order_index = y * coefficient_blocks + x;
        const uint32_t expected =
          static_cast<uint32_t>(y * coefficient_width + x);
        if (candidate[order_index] != expected) {
          return Status::Internal(
            "Natural coefficient order has an invalid LLF prefix");
        }
      }
    }
    *order = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

namespace {

Status BuildSimpleAcGroupTokenTemplateValidated(
  const VarDctAcGroupView& group,
  const AcStrategyGrid& strategies,
  const SimpleCoefficientOrders& coefficient_orders,
  ConstPlaneI32View raw_quant_field,
  SimpleAcGroupTokenTemplate* token_template) {
  if (token_template == nullptr) {
    return Status::InvalidArgument("AC-group token-template output is null");
  }

  try {
    std::vector<StrategyAnchor> anchors;
    Status status = ValidateAndCollectAnchors(group, strategies, &anchors);
    if (!status.ok()) {
      return status;
    }

    if (group.used_coefficient_count
        > std::numeric_limits<size_t>::max() / 3 - anchors.size()) {
      return Status::OutOfMemory("AC-group token count overflow");
    }
    SimpleAcGroupTokenTemplate candidate{
      .block_x = group.block_x,
      .block_y = group.block_y,
      .block_extent = group.block_extent,
    };
    candidate.block_context_keys.reserve(3 * anchors.size());
    candidate.values.reserve(
      3 * (group.used_coefficient_count + anchors.size()));
    candidate.tokens.reserve(
      3 * (group.used_coefficient_count + anchors.size()));

    size_t block_count = 0;
    if (!group.block_extent.try_area(&block_count)) {
      return Status::InvalidArgument("AC-group block count overflow");
    }
    std::array<std::vector<uint8_t>, 3> nonzero_maps;
    for (std::vector<uint8_t>& map : nonzero_maps) {
      map.assign(block_count, 0);
    }
    std::array<std::vector<uint32_t>, kAcStrategyCount> natural_orders;
    std::array<bool, kAcStrategyCount> order_ready{};

    size_t source_offset = 0;
    constexpr std::array<size_t, 3> kChannelOrder = {1, 0, 2};
    for (const StrategyAnchor& anchor : anchors) {
      const AcStrategyInfo* info = GetAcStrategyInfo(anchor.strategy);
      if (info == nullptr || source_offset > group.used_coefficient_count
          || anchor.coefficient_count
               > group.used_coefficient_count - source_offset) {
        return Status::InvalidArgument("AC-group ended inside a transform");
      }

      const size_t strategy_index = static_cast<size_t>(anchor.strategy);
      const size_t order_family =
        codestream_internal::kSimpleStrategyOrder[strategy_index];
      const bool custom_order =
        (coefficient_orders.used_order_mask &
         (uint16_t{1} << order_family)) != 0;
      if (!custom_order && !order_ready[strategy_index]) {
        status = ComputeSimpleNaturalCoefficientOrder(anchor.strategy,
          &natural_orders[strategy_index]);
        if (!status.ok()) {
          return status;
        }
        order_ready[strategy_index] = true;
      }
      const size_t covered_blocks =
        info->covered_blocks.width * info->covered_blocks.height;
      if (!std::has_single_bit(covered_blocks)) {
        return Status::Internal("AC strategy block count is invalid");
      }
      const size_t log2_covered_blocks = std::countr_zero(covered_blocks);

      for (const size_t channel : kChannelOrder) {
        const std::vector<uint32_t>& order = custom_order
          ? coefficient_orders.orders[order_family][channel]
          : natural_orders[strategy_index];
        if (order.size() != anchor.coefficient_count) {
          return Status::InvalidArgument(
            "Coefficient order does not match its AC strategy");
        }
        const std::span<const int32_t> coefficients =
          group.coefficients[channel].subspan(source_offset,
                                              anchor.coefficient_count);
        int32_t nonzeros = CountNonzerosExceptLlf(coefficients, *info);
        const uint8_t scaled_nonzeros = static_cast<uint8_t>(
          (nonzeros + static_cast<int32_t>(covered_blocks) - 1)
          / static_cast<int32_t>(covered_blocks));
        std::vector<uint8_t>& map = nonzero_maps[channel];
        for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
          for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
            map[(anchor.y + dy) * group.block_extent.width + anchor.x + dx] =
              scaled_nonzeros;
          }
        }

        const uint32_t prediction =
          PredictNonzeros(map, group.block_extent, anchor.x, anchor.y);
        if (raw_quant_field.valid() &&
            (group.block_x + anchor.x >= raw_quant_field.extent.width ||
             group.block_y + anchor.y >= raw_quant_field.extent.height)) {
          return Status::InvalidArgument(
            "AC-group raw quantization field is incomplete");
        }
        const int32_t raw_quant = raw_quant_field.valid()
          ? raw_quant_field.Row(group.block_y + anchor.y)[
              group.block_x + anchor.x]
          : 1;
        if (raw_quant < 1 || raw_quant > 256) {
          return Status::InvalidArgument(
            "AC-group raw quantization is out of range");
        }
        if (candidate.block_context_keys.size() >=
            SimpleAcTokenTemplate::kCoefficientFlag) {
          return Status::Internal("AC-group block-context key overflow");
        }
        const uint16_t block_context_key = static_cast<uint16_t>(
          candidate.block_context_keys.size());
        candidate.block_context_keys.push_back({
          .raw_quant = static_cast<uint16_t>(raw_quant),
          .channel_row = static_cast<uint8_t>(channel < 2 ? channel ^ 1u : 2),
          .order_family = static_cast<uint8_t>(order_family),
        });
        candidate.values.push_back(static_cast<uint32_t>(nonzeros));
        candidate.tokens.push_back({
          .block_context_key = block_context_key,
          .local_context = NonzeroBucket(prediction),
        });

        int32_t remaining_nonzeros = nonzeros;
        uint32_t previous_nonzero =
          nonzeros > static_cast<int32_t>(anchor.coefficient_count / 16) ? 0
                                                                         : 1;
        for (size_t scan = covered_blocks;
             scan < anchor.coefficient_count && remaining_nonzeros != 0;
             ++scan) {
          const int32_t coefficient = coefficients[order[scan]];
          const uint32_t local_context = ZeroDensityContext(
            remaining_nonzeros, scan, covered_blocks, log2_covered_blocks,
            previous_nonzero);
          if (local_context >= SimpleAcTokenTemplate::kCoefficientFlag) {
            return Status::Internal("AC token-template context overflow");
          }
          candidate.values.push_back(PackSigned(coefficient));
          candidate.tokens.push_back({
            .block_context_key = block_context_key,
            .local_context = static_cast<uint16_t>(
              SimpleAcTokenTemplate::kCoefficientFlag | local_context),
          });
          previous_nonzero = coefficient != 0 ? 1 : 0;
          remaining_nonzeros -= static_cast<int32_t>(previous_nonzero);
        }
        if (remaining_nonzeros != 0) {
          return Status::Internal(
            "Coefficient scan missed a nonzero value");
        }
      }
      source_offset += anchor.coefficient_count;
    }

    if (source_offset != group.used_coefficient_count) {
      return Status::InvalidArgument(
        "AC-group coefficient consumption is incomplete");
    }
    *token_template = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status AppendDirectAcToken(
  uint32_t value,
  uint16_t context,
  bool collect_fixed_populations,
  codestream_internal::SimpleAcTokenizationScratch* scratch,
  codestream_internal::SimpleAcGroupTokenData* group) {

  group->values.push_back(value);
  group->contexts.push_back(context);
  if (!collect_fixed_populations) return Status::Ok();
  if (context >= scratch->population_slots.size()) {
    return Status::Internal("AC population context is out of range");
  }
  uint32_t symbol = value;
  uint8_t extra_bit_count = 0;
  if (value >= 16) {
    const uint32_t exponent =
      31u - static_cast<uint32_t>(std::countl_zero(value));
    const uint32_t mantissa = value - (uint32_t{1} << exponent);
    symbol = 16 + ((exponent - 4) << 2) +
      (mantissa >> (exponent - 2));
    extra_bit_count = static_cast<uint8_t>(exponent - 2);
  }
  if (symbol >= kPrefixAlphabetSize) {
    return Status::Internal("AC population symbol is out of range");
  }
  constexpr uint16_t kMissing = std::numeric_limits<uint16_t>::max();
  uint16_t& slot = scratch->population_slots[context];
  if (slot == kMissing) {
    if (scratch->populations.size() >= kMissing) {
      return Status::Internal("AC population slot overflow");
    }
    slot = static_cast<uint16_t>(scratch->populations.size());
    scratch->populations.push_back({.context = context});
  }
  auto& population = scratch->populations[slot];
  uint32_t& count = population.counts[symbol];
  if (count == std::numeric_limits<uint32_t>::max() ||
      population.token_count == std::numeric_limits<uint64_t>::max() ||
      population.extra_bits > std::numeric_limits<uint64_t>::max() -
        extra_bit_count) {
    return Status::InvalidArgument("AC population count overflow");
  }
  ++count;
  ++population.token_count;
  population.extra_bits += extra_bit_count;
  population.maximum_symbol = std::max(
    population.maximum_symbol, symbol);
  return Status::Ok();
}

Status TokenizeSimpleAcGroupDirectValidated(
  const VarDctAcGroupView& group,
  const AcStrategyGrid& strategies,
  const SimpleCoefficientOrders& coefficient_orders,
  const codestream_internal::SimpleAcNaturalOrders& natural_orders,
  const SimpleBlockContextMap& block_context_map,
  ConstPlaneI32View raw_quant_field,
  bool collect_fixed_populations,
  codestream_internal::SimpleAcTokenizationScratch* scratch,
  codestream_internal::SimpleAcGroupTokenData* output) {

  if (scratch == nullptr || output == nullptr) {
    return Status::InvalidArgument("AC direct-token output is null");
  }
  try {
    Status status = ValidateAndCollectAnchors(
      group, strategies, &scratch->anchors);
    if (!status.ok()) return status;
    if (group.used_coefficient_count >
        std::numeric_limits<size_t>::max() / 3 - scratch->anchors.size()) {
      return Status::OutOfMemory("AC-group token count overflow");
    }
    const size_t maximum_token_count =
      3 * (group.used_coefficient_count + scratch->anchors.size());
    codestream_internal::SimpleAcGroupTokenData candidate;
    candidate.values.reserve(maximum_token_count);
    candidate.contexts.reserve(maximum_token_count);

    size_t block_count = 0;
    if (!group.block_extent.try_area(&block_count)) {
      return Status::InvalidArgument("AC-group block count overflow");
    }
    for (std::vector<uint8_t>& map : scratch->nonzero_maps) {
      map.assign(block_count, 0);
    }
    if (collect_fixed_populations) {
      const size_t context_count = block_context_map.ac_context_count();
      if (context_count == 0 ||
          context_count >= std::numeric_limits<uint16_t>::max()) {
        return Status::Internal("AC context count is invalid");
      }
      scratch->population_slots.assign(
        context_count, std::numeric_limits<uint16_t>::max());
      scratch->populations.clear();
      scratch->populations.reserve(
        std::min(context_count, maximum_token_count));
    }

    size_t source_offset = 0;
    constexpr std::array<size_t, 3> kChannelOrder = {1, 0, 2};
    for (const StrategyAnchor& anchor : scratch->anchors) {
      const AcStrategyInfo* info = GetAcStrategyInfo(anchor.strategy);
      if (info == nullptr || source_offset > group.used_coefficient_count ||
          anchor.coefficient_count >
            group.used_coefficient_count - source_offset) {
        return Status::InvalidArgument("AC-group ended inside a transform");
      }
      const size_t strategy_index = static_cast<size_t>(anchor.strategy);
      const size_t order_family =
        codestream_internal::kSimpleStrategyOrder[strategy_index];
      const bool custom_order =
        (coefficient_orders.used_order_mask &
         (uint16_t{1} << order_family)) != 0;
      const size_t covered_blocks =
        info->covered_blocks.width * info->covered_blocks.height;
      if (!std::has_single_bit(covered_blocks)) {
        return Status::Internal("AC strategy block count is invalid");
      }
      const size_t log2_covered_blocks = std::countr_zero(covered_blocks);
      if (group.block_x + anchor.x >= raw_quant_field.extent.width ||
          group.block_y + anchor.y >= raw_quant_field.extent.height) {
        return Status::InvalidArgument(
          "AC-group raw quantization field is incomplete");
      }
      const int32_t raw_quant = raw_quant_field.Row(
        group.block_y + anchor.y)[group.block_x + anchor.x];
      if (raw_quant < 1 || raw_quant > 256) {
        return Status::InvalidArgument(
          "AC-group raw quantization is out of range");
      }

      for (const size_t channel : kChannelOrder) {
        const std::vector<uint32_t>& order = custom_order
          ? coefficient_orders.orders[order_family][channel]
          : natural_orders.orders[strategy_index];
        if (order.size() != anchor.coefficient_count) {
          return Status::InvalidArgument(
            "Coefficient order does not match its AC strategy");
        }
        const std::span<const int32_t> coefficients =
          group.coefficients[channel].subspan(
            source_offset, anchor.coefficient_count);
        int32_t nonzeros = CountNonzerosExceptLlf(coefficients, *info);
        const uint8_t scaled_nonzeros = static_cast<uint8_t>(
          (nonzeros + static_cast<int32_t>(covered_blocks) - 1) /
          static_cast<int32_t>(covered_blocks));
        std::vector<uint8_t>& map = scratch->nonzero_maps[channel];
        for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
          for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
            map[(anchor.y + dy) * group.block_extent.width + anchor.x + dx] =
              scaled_nonzeros;
          }
        }

        const uint32_t prediction =
          PredictNonzeros(map, group.block_extent, anchor.x, anchor.y);
        const SimpleAcBlockContextKey key{
          .raw_quant = static_cast<uint16_t>(raw_quant),
          .channel_row = static_cast<uint8_t>(channel < 2 ? channel ^ 1u : 2),
          .order_family = static_cast<uint8_t>(order_family),
        };
        const uint32_t block_context =
          BlockContextValidated(block_context_map, key);
        if (block_context >= block_context_map.num_contexts) {
          return Status::Internal("AC block context is invalid");
        }
        uint16_t context = static_cast<uint16_t>(FinalNonzeroContext(
          block_context_map.num_contexts, block_context,
          NonzeroBucket(prediction)));
        status = AppendDirectAcToken(
          static_cast<uint32_t>(nonzeros), context,
          collect_fixed_populations, scratch, &candidate);
        if (!status.ok()) return status;

        int32_t remaining_nonzeros = nonzeros;
        uint32_t previous_nonzero =
          nonzeros > static_cast<int32_t>(anchor.coefficient_count / 16) ? 0
                                                                         : 1;
        for (size_t scan = covered_blocks;
             scan < anchor.coefficient_count && remaining_nonzeros != 0;
             ++scan) {
          const int32_t coefficient = coefficients[order[scan]];
          const uint32_t local_context = ZeroDensityContext(
            remaining_nonzeros, scan, covered_blocks, log2_covered_blocks,
            previous_nonzero);
          context = static_cast<uint16_t>(FinalCoefficientContext(
            block_context_map.num_contexts, block_context, local_context));
          status = AppendDirectAcToken(
            PackSigned(coefficient), context, collect_fixed_populations,
            scratch, &candidate);
          if (!status.ok()) return status;
          previous_nonzero = coefficient != 0 ? 1 : 0;
          remaining_nonzeros -= static_cast<int32_t>(previous_nonzero);
        }
        if (remaining_nonzeros != 0) {
          return Status::Internal("Coefficient scan missed a nonzero value");
        }
      }
      source_offset += anchor.coefficient_count;
    }
    if (source_offset != group.used_coefficient_count ||
        candidate.values.size() != candidate.contexts.size()) {
      return Status::InvalidArgument(
        "AC-group coefficient consumption is incomplete");
    }

    if (collect_fixed_populations) {
      candidate.context_populations.reserve(scratch->populations.size());
      for (const auto& population : scratch->populations) {
        const size_t symbol_offset = candidate.symbol_populations.size();
        for (size_t symbol = 0; symbol <= population.maximum_symbol;
             ++symbol) {
          const uint32_t count = population.counts[symbol];
          if (count != 0) {
            candidate.symbol_populations.push_back({
              .symbol = static_cast<uint8_t>(symbol),
              .count = count,
            });
          }
        }
        const size_t symbol_count =
          candidate.symbol_populations.size() - symbol_offset;
        if (symbol_offset > std::numeric_limits<uint32_t>::max() ||
            symbol_count > std::numeric_limits<uint16_t>::max()) {
          return Status::Internal("AC sparse population overflow");
        }
        candidate.context_populations.push_back({
          .context = population.context,
          .symbol_offset = static_cast<uint32_t>(symbol_offset),
          .symbol_count = static_cast<uint16_t>(symbol_count),
          .token_count = population.token_count,
          .extra_bits = population.extra_bits,
          .maximum_symbol = population.maximum_symbol,
        });
      }
    }
    *output = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status MaterializeSimpleAcGroupContextsValidated(
  const SimpleAcGroupTokenTemplate& token_template,
  const SimpleBlockContextMap& block_context_map,
  std::vector<uint16_t>* contexts) {
  if (contexts == nullptr) {
    return Status::InvalidArgument("AC-group context output is null");
  }
  if (token_template.values.size() != token_template.tokens.size()) {
    return Status::InvalidArgument("AC-group token-template size is invalid");
  }

  try {
    std::vector<uint8_t> block_contexts;
    block_contexts.reserve(token_template.block_context_keys.size());
    for (const SimpleAcBlockContextKey& key :
         token_template.block_context_keys) {
      if (key.raw_quant < 1 || key.raw_quant > 256 || key.channel_row >= 3 ||
          key.order_family >=
            codestream_internal::kSimpleCoefficientOrderCount) {
        return Status::InvalidArgument(
          "AC-group token-template block-context key is invalid");
      }
      const uint32_t block_context =
        BlockContextValidated(block_context_map, key);
      if (block_context >= block_context_map.num_contexts) {
        return Status::InvalidArgument(
          "AC-group token-template block context is invalid");
      }
      block_contexts.push_back(static_cast<uint8_t>(block_context));
    }

    std::vector<uint16_t> candidate;
    candidate.reserve(token_template.tokens.size());
    for (const SimpleAcTokenTemplate& token : token_template.tokens) {
      if (token.block_context_key >= block_contexts.size()) {
        return Status::InvalidArgument(
          "AC-group token-template key index is invalid");
      }
      const uint32_t local_context = token.context_without_block();
      const uint32_t block_context = block_contexts[token.block_context_key];
      uint16_t resolved = 0;
      if (Status status = FinalAcContextFromBlockValidated(
            block_context_map, block_context, local_context,
            token.is_coefficient(), &resolved); !status.ok()) {
        return status;
      }
      candidate.push_back(resolved);
    }
    *contexts = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status MaterializeSimpleAcGroupTokenTemplateValidated(
  const SimpleAcGroupTokenTemplate& token_template,
  const SimpleBlockContextMap& block_context_map,
  std::vector<EntropyToken>* tokens) {
  if (tokens == nullptr) {
    return Status::InvalidArgument("AC-group token output is null");
  }
  std::vector<uint16_t> contexts;
  Status status = MaterializeSimpleAcGroupContextsValidated(
    token_template, block_context_map, &contexts);
  if (!status.ok()) {
    return status;
  }
  try {
    std::vector<EntropyToken> candidate;
    candidate.reserve(contexts.size());
    for (size_t index = 0; index < contexts.size(); ++index) {
      candidate.push_back({contexts[index], token_template.values[index]});
    }
    *tokens = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status TokenizeSimpleAcGroupValidated(
  const VarDctAcGroupView& group,
  const AcStrategyGrid& strategies,
  const SimpleCoefficientOrders& coefficient_orders,
  const SimpleBlockContextMap& block_context_map,
  ConstPlaneI32View raw_quant_field,
  std::vector<EntropyToken>* tokens) {
  SimpleAcGroupTokenTemplate token_template;
  const ConstPlaneI32View template_raw_quant =
    block_context_map.qf_thresholds.empty()
    ? ConstPlaneI32View{}
    : raw_quant_field;
  Status status = BuildSimpleAcGroupTokenTemplateValidated(
    group, strategies, coefficient_orders, template_raw_quant,
    &token_template);
  if (!status.ok()) {
    return status;
  }
  return MaterializeSimpleAcGroupTokenTemplateValidated(
    token_template, block_context_map, tokens);
}

}  // namespace

Status TokenizeSimpleAcGroup(const VarDctAcGroupView& group,
                             const AcStrategyGrid& strategies,
                             const SimpleCoefficientOrders& coefficient_orders,
                             std::vector<EntropyToken>* tokens) {
  Status status = ValidateSimpleCoefficientOrders(coefficient_orders);
  if (!status.ok()) {
    return status;
  }
  const SimpleBlockContextMap block_context_map =
    DefaultSimpleBlockContextMap();
  return TokenizeSimpleAcGroupValidated(
    group, strategies, coefficient_orders, block_context_map, {}, tokens);
}

Status TokenizeSimpleAcGroup(const VarDctAcGroupView& group,
                             const AcStrategyGrid& strategies,
                             std::vector<EntropyToken>* tokens) {
  return TokenizeSimpleAcGroup(
    group, strategies, SimpleCoefficientOrders{}, tokens);
}

Status BuildSimpleAcGroupTokenTemplates(
  const VarDctEncoderFrame& frame,
  const SimpleCoefficientOrders& orders,
  std::vector<SimpleAcGroupTokenTemplate>* groups) {
  if (groups == nullptr) {
    return Status::InvalidArgument("AC-group token-template output is null");
  }
  Status status = ValidateSimpleCodestreamFrame(frame);
  if (!status.ok()) {
    return status;
  }
  status = ValidateSimpleCoefficientOrders(orders);
  if (!status.ok()) {
    return status;
  }

  try {
    std::vector<SimpleAcGroupTokenTemplate> candidate;
    candidate.reserve(frame.ac_group_count());
    for (size_t group_index = 0; group_index < frame.ac_group_count();
         ++group_index) {
      VarDctAcGroupView group;
      status = frame.GetAcGroup(group_index, &group);
      if (!status.ok()) {
        return status;
      }
      SimpleAcGroupTokenTemplate token_template;
      status = BuildSimpleAcGroupTokenTemplateValidated(
        group, frame.strategies(), orders, frame.raw_quant_field(),
        &token_template);
      if (!status.ok()) {
        return status;
      }
      candidate.push_back(std::move(token_template));
    }
    *groups = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status MaterializeSimpleAcGroupTokenStreams(
  std::span<const SimpleAcGroupTokenTemplate> templates,
  const SimpleBlockContextMap& block_context_map,
  std::vector<SimpleAcGroupTokenStream>* groups) {
  if (groups == nullptr) {
    return Status::InvalidArgument("AC-group token output is null");
  }
  Status status = ValidateSimpleBlockContextMap(block_context_map);
  if (!status.ok()) {
    return status;
  }

  try {
    std::vector<SimpleAcGroupTokenStream> candidate;
    candidate.reserve(templates.size());
    for (const SimpleAcGroupTokenTemplate& token_template : templates) {
      size_t block_count = 0;
      if (!IsValidGroupExtent(token_template.block_extent, &block_count) ||
          token_template.block_context_keys.empty() ||
          token_template.values.empty() ||
          token_template.tokens.empty()) {
        return Status::InvalidArgument("AC-group token template is invalid");
      }
      SimpleAcGroupTokenStream stream{
        .block_x = token_template.block_x,
        .block_y = token_template.block_y,
        .block_extent = token_template.block_extent,
      };
      status = MaterializeSimpleAcGroupTokenTemplateValidated(
        token_template, block_context_map, &stream.tokens);
      if (!status.ok()) {
        return status;
      }
      candidate.push_back(std::move(stream));
    }
    *groups = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status MaterializeSimpleAcGroupContexts(
  std::span<const SimpleAcGroupTokenTemplate> templates,
  const SimpleBlockContextMap& block_context_map,
  std::vector<std::vector<uint16_t>>* contexts) {
  if (contexts == nullptr) {
    return Status::InvalidArgument("AC-group context output is null");
  }
  Status status = ValidateSimpleBlockContextMap(block_context_map);
  if (!status.ok()) {
    return status;
  }
  try {
    std::vector<std::vector<uint16_t>> candidate;
    candidate.reserve(templates.size());
    for (const SimpleAcGroupTokenTemplate& token_template : templates) {
      size_t block_count = 0;
      if (!IsValidGroupExtent(token_template.block_extent, &block_count) ||
          token_template.block_context_keys.empty() ||
          token_template.values.empty() || token_template.tokens.empty()) {
        return Status::InvalidArgument("AC-group token template is invalid");
      }
      std::vector<uint16_t> group_contexts;
      status = MaterializeSimpleAcGroupContextsValidated(
        token_template, block_context_map, &group_contexts);
      if (!status.ok()) {
        return status;
      }
      candidate.push_back(std::move(group_contexts));
    }
    *contexts = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status codestream_internal::PrepareSimpleAcNaturalOrders(
  SimpleAcNaturalOrders* orders) {

  if (orders == nullptr) {
    return Status::InvalidArgument("Natural-order table output is null");
  }
  try {
    SimpleAcNaturalOrders candidate;
    for (size_t strategy_index = 0; strategy_index < kAcStrategyCount;
         ++strategy_index) {
      const auto strategy = static_cast<AcStrategyType>(strategy_index);
      if (!IsSimpleStrategy(strategy)) continue;
      if (Status status = ComputeSimpleNaturalCoefficientOrder(
            strategy, &candidate.orders[strategy_index]); !status.ok()) {
        return status;
      }
    }
    *orders = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status codestream_internal::TokenizeSimpleAcGroupForEncoder(
  const VarDctEncoderFrame& frame,
  const SimpleCoefficientOrders& orders,
  const SimpleAcNaturalOrders& natural_orders,
  const SimpleBlockContextMap& block_context_map,
  size_t group_index,
  bool collect_fixed_populations,
  SimpleAcTokenizationScratch* scratch,
  SimpleAcGroupTokenData* group) {

  if (scratch == nullptr || group == nullptr) {
    return Status::InvalidArgument("AC direct-token output is null");
  }
  VarDctAcGroupView group_view;
  if (Status status = frame.GetAcGroup(group_index, &group_view);
      !status.ok()) {
    return status;
  }
  return TokenizeSimpleAcGroupDirectValidated(
    group_view, frame.strategies(), orders, natural_orders,
    block_context_map, frame.raw_quant_field(), collect_fixed_populations,
    scratch, group);
}

Status codestream_internal::BuildSimpleAcGroupTokenTemplateForEncoder(
  const VarDctEncoderFrame& frame,
  const SimpleCoefficientOrders& orders,
  size_t group_index,
  SimpleAcGroupTokenTemplate* group) {

  if (group == nullptr) {
    return Status::InvalidArgument("AC-group token-template output is null");
  }
  VarDctAcGroupView group_view;
  if (Status status = frame.GetAcGroup(group_index, &group_view);
      !status.ok()) {
    return status;
  }
  return BuildSimpleAcGroupTokenTemplateValidated(
    group_view, frame.strategies(), orders, frame.raw_quant_field(), group);
}

Status codestream_internal::MaterializeSimpleAcGroupContextsForEncoder(
  const SimpleAcGroupTokenTemplate& token_template,
  const SimpleBlockContextMap& block_context_map,
  std::vector<uint16_t>* contexts) {

  return MaterializeSimpleAcGroupContextsValidated(
    token_template, block_context_map, contexts);
}

Status TokenizeSimpleAcGroups(const VarDctEncoderFrame& frame,
                              const SimpleCoefficientOrders& orders,
                              const SimpleBlockContextMap& block_context_map,
                              std::vector<SimpleAcGroupTokenStream>* groups) {
  if (groups == nullptr) {
    return Status::InvalidArgument("AC-group token output is null");
  }
  std::vector<SimpleAcGroupTokenTemplate> templates;
  Status status = BuildSimpleAcGroupTokenTemplates(frame, orders, &templates);
  if (!status.ok()) {
    return status;
  }
  return MaterializeSimpleAcGroupTokenStreams(
    templates, block_context_map, groups);
}

Status TokenizeSimpleAcGroups(const VarDctEncoderFrame& frame,
                              const SimpleCoefficientOrders& orders,
                              std::vector<SimpleAcGroupTokenStream>* groups) {
  return TokenizeSimpleAcGroups(
    frame, orders, DefaultSimpleBlockContextMap(), groups);
}

Status TokenizeSimpleAcGroups(const VarDctEncoderFrame& frame,
                              std::vector<SimpleAcGroupTokenStream>* groups) {
  return TokenizeSimpleAcGroups(frame, SimpleCoefficientOrders{}, groups);
}

}  // namespace gjxl
