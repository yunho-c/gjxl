// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl's enc_coeff_order.cc and lehmer_code.h.

#include "codestream/coefficient_order.h"

#include <algorithm>
#include <array>
#include <cmath>
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
#include "codestream/ac_group.h"
#include "codestream/encoder.h"
#include "core/ac_strategy.h"

namespace gjxl {
namespace {

constexpr uint16_t kSupportedOrderMask =
  (uint16_t{1} << 0) | (uint16_t{1} << 2) | (uint16_t{1} << 3) |
  (uint16_t{1} << 4) | (uint16_t{1} << 6);

Status AllocationFailure() {
  return Status::OutOfMemory("Coefficient-order allocation failed");
}

const AcStrategyInfo* RepresentativeInfo(size_t family) {
  AcStrategyType strategy;
  switch (family) {
    case 0:
      strategy = AcStrategyType::kDct8;
      break;
    case 2:
      strategy = AcStrategyType::kDct16x16;
      break;
    case 3:
      strategy = AcStrategyType::kDct32x32;
      break;
    case 4:
      strategy = AcStrategyType::kDct16x8;
      break;
    case 6:
      strategy = AcStrategyType::kDct32x16;
      break;
    default:
      return nullptr;
  }
  return GetAcStrategyInfo(strategy);
}

AcStrategyType RepresentativeStrategy(size_t family) {
  switch (family) {
    case 0:
      return AcStrategyType::kDct8;
    case 2:
      return AcStrategyType::kDct16x16;
    case 3:
      return AcStrategyType::kDct32x32;
    case 4:
      return AcStrategyType::kDct16x8;
    case 6:
      return AcStrategyType::kDct32x16;
    default:
      return AcStrategyType::kCount;
  }
}

Status ValidateOrder(
  std::span<const uint32_t> order,
  const AcStrategyInfo& info) {

  const size_t size = info.coefficient_count();
  if (order.size() != size) {
    return Status::InvalidArgument("Coefficient-order size is invalid");
  }
  std::vector<uint8_t> seen(size, 0);
  for (uint32_t coefficient : order) {
    if (coefficient >= size || seen[coefficient] != 0) {
      return Status::InvalidArgument(
        "Coefficient order is not a permutation");
    }
    seen[coefficient] = 1;
  }

  const size_t llf_size =
    info.covered_blocks.width * info.covered_blocks.height;
  std::vector<uint32_t> natural;
  Status status =
    ComputeSimpleNaturalCoefficientOrder(info.type, &natural);
  if (!status.ok()) {
    return status;
  }
  if (!std::ranges::equal(
        order.first(llf_size),
        std::span<const uint32_t>(natural).first(llf_size))) {
    return Status::InvalidArgument(
      "Coefficient order does not preserve the LLF prefix");
  }
  return Status::Ok();
}

// Isolating the contiguous update lets the compiler vectorize narrow counts
// without architecture-specific intrinsics or aliasing assumptions.
template <typename Count>
void CountCoefficientZeros(
  const int32_t* coefficients, Count* counts, size_t size) {
  for (size_t coefficient = 0; coefficient < size; ++coefficient) {
    counts[coefficient] += coefficients[coefficient] == 0;
  }
}

constexpr bool Use32BitZeroCounts(Extent2D blocks) noexcept {
  size_t area = 0;
  return blocks.try_area(&area) &&
    area <= std::numeric_limits<uint32_t>::max();
}

static_assert(Use32BitZeroCounts({65535, 65537}));
static_assert(!Use32BitZeroCounts({65536, 65536}));
static_assert(!Use32BitZeroCounts({std::numeric_limits<size_t>::max(), 2}));

template <typename Count>
Status CountGroupZeros(
  const VarDctAcGroupView& group,
  const AcStrategyGrid& strategies,
  std::array<std::array<std::vector<Count>, 3>,
             codestream_internal::kSimpleCoefficientOrderCount>* zero_counts,
  bool sample_dct8,
  std::array<uint64_t, 2>* random_state,
  uint16_t* present_mask) {

  const auto use_sample = [&]() {
    uint64_t state_1 = (*random_state)[0];
    const uint64_t state_0 = (*random_state)[1];
    const uint64_t bits = state_1 + state_0;
    (*random_state)[0] = state_0;
    state_1 ^= state_1 << 23;
    state_1 ^= state_0 ^ (state_1 >> 18) ^ (state_0 >> 5);
    (*random_state)[1] = state_1;
    constexpr uint64_t kHalfThreshold =
      (std::numeric_limits<uint64_t>::max() >> 32) / 2;
    return (bits >> 32) <= kHalfThreshold;
  };

  size_t source_offset = 0;
  for (size_t y = 0; y < group.block_extent.height; ++y) {
    for (size_t x = 0; x < group.block_extent.width; ++x) {
      AcStrategyCell cell;
      Status status = strategies.Get(group.block_x + x, group.block_y + y,
                                     &cell);
      if (!status.ok()) {
        return Status::InvalidArgument(
          "Coefficient-order strategy rectangle is incomplete");
      }
      if (!cell.is_anchor) {
        continue;
      }
      const AcStrategyInfo* info = GetAcStrategyInfo(cell.strategy);
      if (info == nullptr || source_offset > group.used_coefficient_count ||
          info->coefficient_count() >
            group.used_coefficient_count - source_offset) {
        return Status::InvalidArgument(
          "Coefficient-order group ended inside a transform");
      }
      const size_t family = codestream_internal::kSimpleStrategyOrder[
        static_cast<size_t>(cell.strategy)];
      if (family >= codestream_internal::kSimpleCoefficientOrderCount) {
        return Status::Internal("Coefficient-order family is invalid");
      }
      if (family <= 6) {
        const uint16_t family_bit = uint16_t{1} << family;
        if ((kSupportedOrderMask & family_bit) == 0) {
          return Status::InvalidArgument(
            "Coefficient-order strategy is outside the simple profile");
        }
        *present_mask |= family_bit;
        for (size_t channel = 0; channel < 3; ++channel) {
          std::vector<Count>& counts = (*zero_counts)[family][channel];
          if (counts.empty()) {
            counts.assign(info->coefficient_count(), 0);
          } else if (counts.size() != info->coefficient_count()) {
            return Status::Internal(
              "Coefficient-order family dimensions disagree");
          }
        }
        const bool selected = !sample_dct8 || use_sample();
        if (selected) {
          for (size_t channel = 0; channel < 3; ++channel) {
            Count* const counts = (*zero_counts)[family][channel].data();
            const std::span<const int32_t> coefficients =
              group.coefficients[channel].subspan(
                source_offset, info->coefficient_count());
            // The validated frame has a size_t-representable block area.
            // Each zero-initialized counter is incremented at most once per
            // anchor, and anchors partition that area. The caller selects
            // uint32_t only when the area fits, and uint64_t otherwise.
            // Neither counter type can overflow for the selected frame.
            static_assert(std::numeric_limits<size_t>::digits <=
                          std::numeric_limits<uint64_t>::digits);
            CountCoefficientZeros(
              coefficients.data(), counts, coefficients.size());
          }
        }
      }
      source_offset += info->coefficient_count();
    }
  }
  if (source_offset != group.used_coefficient_count) {
    return Status::InvalidArgument(
      "Coefficient-order group consumption is incomplete");
  }
  return Status::Ok();
}

Status PresentOrderMask(const VarDctEncoderFrame& frame, uint16_t* mask) {
  uint16_t present = 0;
  for (size_t group_index = 0; group_index < frame.ac_group_count();
       ++group_index) {
    VarDctAcGroupView group;
    if (Status status = frame.GetAcGroup(group_index, &group); !status.ok()) {
      return status;
    }
    for (size_t y = 0; y < group.block_extent.height; ++y) {
      for (size_t x = 0; x < group.block_extent.width; ++x) {
        AcStrategyCell cell;
        if (Status status = frame.strategies().Get(
              group.block_x + x, group.block_y + y, &cell); !status.ok()) {
          return status;
        }
        if (!cell.is_anchor) continue;
        const size_t family = codestream_internal::kSimpleStrategyOrder[
          static_cast<size_t>(cell.strategy)];
        if (family >= codestream_internal::kSimpleCoefficientOrderCount ||
            (kSupportedOrderMask & (uint16_t{1} << family)) == 0) {
          return Status::InvalidArgument(
            "Coefficient-order strategy is outside the simple profile");
        }
        present |= uint16_t{1} << family;
      }
    }
  }
  *mask = present;
  return Status::Ok();
}

Status ComputeLehmerCode(
  std::span<const uint32_t> permutation,
  std::vector<uint32_t>* code) {

  const size_t size = permutation.size();
  std::vector<uint32_t> tree(size + 1, 0);
  std::vector<uint32_t> candidate(size, 0);
  for (size_t index = 0; index < size; ++index) {
    const uint32_t value = permutation[index];
    if (value >= size) {
      return Status::InvalidArgument("Coefficient permutation is invalid");
    }
    uint32_t penalty = 0;
    size_t cursor = static_cast<size_t>(value) + 1;
    while (cursor != 0) {
      penalty += tree[cursor];
      cursor &= cursor - 1;
    }
    if (value < penalty) {
      return Status::InvalidArgument("Coefficient permutation repeats a value");
    }
    candidate[index] = value - penalty;
    cursor = static_cast<size_t>(value) + 1;
    while (cursor <= size) {
      ++tree[cursor];
      cursor += cursor & (~cursor + 1);
    }
  }
  *code = std::move(candidate);
  return Status::Ok();
}

Status CoefficientOrderContext(uint32_t value, uint32_t* context) {
  if (context == nullptr) {
    return Status::InvalidArgument("Coefficient-order context output is null");
  }
  HybridUintToken token;
  Status status = EncodeHybridUint(value, {0, 0, 0}, &token);
  if (!status.ok()) {
    return status;
  }
  *context = std::min<uint32_t>(
    token.symbol, kSimplePermutationContextCount - 1);
  return Status::Ok();
}

Status TokenizePermutation(
  std::span<const uint32_t> order,
  const AcStrategyInfo& info,
  std::vector<EntropyToken>* tokens) {

  std::vector<uint32_t> natural;
  Status status =
    ComputeSimpleNaturalCoefficientOrder(info.type, &natural);
  if (!status.ok()) {
    return status;
  }
  std::vector<uint32_t> natural_lut(natural.size());
  for (size_t index = 0; index < natural.size(); ++index) {
    natural_lut[natural[index]] = static_cast<uint32_t>(index);
  }
  std::vector<uint32_t> permutation(order.size());
  for (size_t index = 0; index < order.size(); ++index) {
    permutation[index] = natural_lut[order[index]];
  }
  std::vector<uint32_t> lehmer;
  status = ComputeLehmerCode(permutation, &lehmer);
  if (!status.ok()) {
    return status;
  }

  const size_t skip = info.covered_blocks.width * info.covered_blocks.height;
  size_t end = lehmer.size();
  while (end > skip && lehmer[end - 1] == 0) {
    --end;
  }
  uint32_t context = 0;
  status = CoefficientOrderContext(
    static_cast<uint32_t>(lehmer.size()), &context);
  if (!status.ok()) {
    return status;
  }
  tokens->push_back({context, static_cast<uint32_t>(end - skip)});
  uint32_t last = 0;
  for (size_t index = skip; index < end; ++index) {
    status = CoefficientOrderContext(last, &context);
    if (!status.ok()) {
      return status;
    }
    tokens->push_back({context, lehmer[index]});
    last = lehmer[index];
  }
  return Status::Ok();
}

}  // namespace

Status ComputeSimpleCoefficientOrders(
  const VarDctEncoderFrame& frame,
  SimpleCoefficientOrders* orders) {

  if (orders == nullptr) {
    return Status::InvalidArgument("Coefficient-order output is null");
  }
  Status status = ValidateSimpleCodestreamFrame(frame);
  if (!status.ok()) {
    return status;
  }

  return codestream_internal::ComputeSimpleCoefficientOrdersForEncoder(
    frame, VarDctCoefficientOrderBehavior::kFull, orders);
}

namespace {

template <typename Count>
Status ComputeCoefficientOrdersWithCounts(
  const VarDctEncoderFrame& frame,
  VarDctCoefficientOrderBehavior behavior,
  SimpleCoefficientOrders* orders) {

  if (orders == nullptr) {
    return Status::InvalidArgument("Coefficient-order output is null");
  }
  switch (behavior) {
    case VarDctCoefficientOrderBehavior::kFull:
    case VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled:
      break;
    default:
      return Status::InvalidArgument(
        "Coefficient-order behavior is invalid");
  }

  try {
    SimpleCoefficientOrders candidate;
    const Extent2D blocks = frame.geometry().block_grid().blocks;
    if (blocks.width < 5 && blocks.height < 5) {
      *orders = std::move(candidate);
      return Status::Ok();
    }

    std::array<std::array<std::vector<Count>, 3>,
               codestream_internal::kSimpleCoefficientOrderCount> zero_counts;
    uint16_t present_mask = 0;
    Status status;
    if (behavior ==
        VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled) {
      status = PresentOrderMask(frame, &present_mask);
      if (!status.ok()) return status;
    }
    const bool sample_dct8 =
      behavior == VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled &&
      present_mask == 1;
    std::array<uint64_t, 2> random_state = {
      0x94D049BB133111EBull,
      0xBF58476D1CE4E5B9ull,
    };
    for (size_t group_index = 0; group_index < frame.ac_group_count();
         ++group_index) {
      VarDctAcGroupView group;
      status = frame.GetAcGroup(group_index, &group);
      if (!status.ok()) {
        return status;
      }
      status = CountGroupZeros(
        group, frame.strategies(), &zero_counts, sample_dct8, &random_state,
        &present_mask);
      if (!status.ok()) {
        return status;
      }
    }

    for (size_t family = 0;
         family < codestream_internal::kSimpleCoefficientOrderCount;
         ++family) {
      const uint16_t family_bit = uint16_t{1} << family;
      if ((present_mask & family_bit) == 0) {
        continue;
      }
      const AcStrategyInfo* info = RepresentativeInfo(family);
      if (info == nullptr) {
        return Status::Internal(
          "Present coefficient-order family has no representative");
      }
      std::vector<uint32_t> natural;
      status = ComputeSimpleNaturalCoefficientOrder(
        RepresentativeStrategy(family), &natural);
      if (!status.ok()) {
        return status;
      }
      const size_t llf_size =
        info->covered_blocks.width * info->covered_blocks.height;
      const float inverse_sqrt_size =
        1.0f / std::sqrt(static_cast<float>(natural.size()));
      bool nondefault = false;
      for (size_t channel = 0; channel < 3; ++channel) {
        const std::vector<Count>& counts = zero_counts[family][channel];
        if (counts.size() != natural.size()) {
          return Status::Internal(
            "Coefficient-order zero counts are incomplete");
        }
        std::vector<uint32_t> custom = natural;
        std::stable_sort(
          custom.begin() + static_cast<ptrdiff_t>(llf_size), custom.end(),
          [&](uint32_t left, uint32_t right) {
            const uint64_t left_count = static_cast<uint64_t>(
              static_cast<float>(counts[left]) * inverse_sqrt_size + 0.1f);
            const uint64_t right_count = static_cast<uint64_t>(
              static_cast<float>(counts[right]) * inverse_sqrt_size + 0.1f);
            return left_count < right_count;
          });
        nondefault |= custom != natural;
        candidate.orders[family][channel] = std::move(custom);
      }
      if (nondefault) {
        candidate.used_order_mask |= family_bit;
      } else {
        for (std::vector<uint32_t>& channel : candidate.orders[family]) {
          channel.clear();
        }
      }
    }
    status = ValidateSimpleCoefficientOrders(candidate);
    if (!status.ok()) {
      return status;
    }
    *orders = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

}  // namespace

Status codestream_internal::ComputeSimpleCoefficientOrdersForEncoder(
  const VarDctEncoderFrame& frame,
  VarDctCoefficientOrderBehavior behavior,
  SimpleCoefficientOrders* orders) {
  // Each anchor can add at most one to a coefficient's zero population.
  // Keep the original range on exceptionally large frames, while ordinary
  // frames use smaller counters and the vectorizable contiguous helper.
  return Use32BitZeroCounts(frame.geometry().block_grid().blocks)
    ? ComputeCoefficientOrdersWithCounts<uint32_t>(frame, behavior, orders)
    : ComputeCoefficientOrdersWithCounts<uint64_t>(frame, behavior, orders);
}

Status ValidateSimpleCoefficientOrders(const SimpleCoefficientOrders& orders) {
  if ((orders.used_order_mask & ~kSupportedOrderMask) != 0) {
    return Status::InvalidArgument(
      "Coefficient-order mask uses an unsupported family");
  }
  try {
    for (size_t family = 0;
         family < codestream_internal::kSimpleCoefficientOrderCount;
         ++family) {
      const bool used =
        (orders.used_order_mask & (uint16_t{1} << family)) != 0;
      const AcStrategyInfo* info = RepresentativeInfo(family);
      for (size_t channel = 0; channel < 3; ++channel) {
        const std::vector<uint32_t>& order = orders.orders[family][channel];
        if (!used) {
          if (!order.empty()) {
            return Status::InvalidArgument(
              "Unused coefficient-order family has retained state");
          }
          continue;
        }
        if (info == nullptr) {
          return Status::InvalidArgument(
            "Coefficient-order family has no supported strategy");
        }
        Status status = ValidateOrder(order, *info);
        if (!status.ok()) {
          return status;
        }
      }
    }
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status TokenizeSimpleCoefficientOrders(
  const SimpleCoefficientOrders& orders,
  std::vector<EntropyToken>* tokens) {

  if (tokens == nullptr) {
    return Status::InvalidArgument("Coefficient-order token output is null");
  }
  Status status = ValidateSimpleCoefficientOrders(orders);
  if (!status.ok()) {
    return status;
  }
  try {
    std::vector<EntropyToken> candidate;
    for (size_t family = 0;
         family < codestream_internal::kSimpleCoefficientOrderCount;
         ++family) {
      if ((orders.used_order_mask & (uint16_t{1} << family)) == 0) {
        continue;
      }
      const AcStrategyInfo* info = RepresentativeInfo(family);
      if (info == nullptr) {
        return Status::Internal(
          "Selected coefficient-order family has no representative");
      }
      for (size_t channel = 0; channel < 3; ++channel) {
        status = TokenizePermutation(
          orders.orders[family][channel], *info, &candidate);
        if (!status.ok()) {
          return status;
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

}  // namespace gjxl
