// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl's enc_coeff_order.cc and lehmer_code.h.

#include "codestream/coefficient_order.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "codec/codestream.h"
#include "codec/vardct_frame_view_internal.h"
#include "codestream/ac_group.h"
#include "codestream/encoder.h"
#include "core/ac_strategy.h"
#include "core/thread_budget.h"

namespace gjxl {
using vardct_frame_internal::VarDctFrameView;
namespace {

constexpr uint16_t kSupportedOrderMask =
  (uint16_t{1} << 0) | (uint16_t{1} << 2) | (uint16_t{1} << 3) |
  (uint16_t{1} << 4) | (uint16_t{1} << 6);

constexpr size_t kMaximumCoefficientOrderWorkers = 8;
constexpr size_t kMinimumParallelCoefficientCount = 256 * 256;

using ZeroCounts = std::array<
  std::array<std::vector<uint64_t>, 3>,
  codestream_internal::kSimpleCoefficientOrderCount>;

Status AllocationFailure() {
  return Status::OutOfMemory("Coefficient-order allocation failed");
}

size_t CoefficientOrderParticipantCount(
  size_t count,
  size_t coefficient_count) {

  if (count == 0) return 0;
  if (thread_budget_internal::InExplicitParallelScope()) return 1;
  const size_t hardware_workers = std::max<size_t>(
    std::thread::hardware_concurrency(), 1);
  const size_t automatic_worker_count =
    coefficient_count < kMinimumParallelCoefficientCount
      ? 1
      : std::min(
          count,
          std::min(kMaximumCoefficientOrderWorkers, hardware_workers));
  const size_t cpu_thread_count =
    thread_budget_internal::CpuThreadCount();
  return cpu_thread_count == 0
    ? automatic_worker_count
    : std::min(automatic_worker_count, cpu_thread_count);
}

template <typename Function>
Status RunParallelCoefficientGroups(
  size_t count,
  size_t coefficient_count,
  Function&& function) {

  const size_t participant_count =
    CoefficientOrderParticipantCount(count, coefficient_count);
  if (participant_count == 0) return Status::Ok();
  const size_t cpu_thread_count =
    thread_budget_internal::CpuThreadCount();
  auto* const participant_tracker =
    thread_budget_internal::ParticipantTracker();
  if (participant_count == 1) {
    thread_budget_internal::ParallelScope scope(
      cpu_thread_count, participant_tracker);
    for (size_t index = 0; index < count; ++index) {
      Status status = function(index, 0);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  }

  std::vector<Status> statuses(count);
  std::atomic<size_t> next_index{0};
  std::vector<std::thread> workers;
  const size_t spawned_worker_count = cpu_thread_count == 0
    ? participant_count
    : participant_count - 1;
  workers.reserve(spawned_worker_count);
  const auto run_worker = [&](size_t worker_index) {
    thread_budget_internal::ParallelScope scope(
      cpu_thread_count, participant_tracker);
    while (true) {
      const size_t index =
        next_index.fetch_add(1, std::memory_order_relaxed);
      if (index >= count) break;
      try {
        statuses[index] = function(index, worker_index);
      } catch (const std::bad_alloc&) {
        statuses[index] = AllocationFailure();
      } catch (const std::length_error&) {
        statuses[index] = AllocationFailure();
      } catch (...) {
        statuses[index] = Status::Internal(
          "Coefficient-order worker failed unexpectedly");
      }
    }
  };
  try {
    for (size_t worker = 0; worker < spawned_worker_count; ++worker) {
      workers.emplace_back(run_worker, worker);
    }
  } catch (const std::system_error&) {
    next_index.store(count, std::memory_order_relaxed);
    for (std::thread& worker : workers) worker.join();
    return Status::Internal("Unable to start coefficient-order workers");
  }
  if (cpu_thread_count != 0) run_worker(spawned_worker_count);
  for (std::thread& worker : workers) worker.join();
  for (const Status& status : statuses) {
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

bool UseCoefficientOrderSample(std::array<uint64_t, 2>* random_state) {
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

Status CountGroupZeros(
  const VarDctAcGroupView& group,
  const AcStrategyGrid& strategies,
  ZeroCounts* zero_counts,
  bool sample_dct8,
  std::span<const uint8_t> sample_decisions,
  uint16_t* present_mask) {

  size_t source_offset = 0;
  size_t sample_index = 0;
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
          std::vector<uint64_t>& counts = (*zero_counts)[family][channel];
          if (counts.empty()) {
            counts.assign(info->coefficient_count(), 0);
          } else if (counts.size() != info->coefficient_count()) {
            return Status::Internal(
              "Coefficient-order family dimensions disagree");
          }
        }
        if (sample_dct8 && sample_index >= sample_decisions.size()) {
          return Status::Internal(
            "Coefficient-order sample decisions are incomplete");
        }
        const bool selected =
          !sample_dct8 || sample_decisions[sample_index++] != 0;
        if (selected) {
          for (size_t channel = 0; channel < 3; ++channel) {
            std::vector<uint64_t>& counts = (*zero_counts)[family][channel];
            const std::span<const int32_t> coefficients =
              group.coefficients[channel].subspan(
                source_offset, info->coefficient_count());
            for (size_t coefficient = 0; coefficient < coefficients.size();
                 ++coefficient) {
              if (coefficients[coefficient] == 0) {
                if (counts[coefficient] ==
                    std::numeric_limits<uint64_t>::max()) {
                  return Status::InvalidArgument(
                    "Coefficient zero count overflow");
                }
                ++counts[coefficient];
              }
            }
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
  if (sample_dct8 && sample_index != sample_decisions.size()) {
    return Status::Internal(
      "Coefficient-order sample decisions were not consumed");
  }
  return Status::Ok();
}

Status PresentOrderMask(
  std::span<const VarDctAcGroupView> groups,
  const AcStrategyGrid& strategies,
  uint16_t* mask) {

  uint16_t present = 0;
  for (const VarDctAcGroupView& group : groups) {
    for (size_t y = 0; y < group.block_extent.height; ++y) {
      for (size_t x = 0; x < group.block_extent.width; ++x) {
        AcStrategyCell cell;
        if (Status status = strategies.Get(
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
    vardct_frame_internal::BorrowFrame(frame),
    VarDctCoefficientOrderBehavior::kFull, orders);
}

Status codestream_internal::ComputeSimpleCoefficientOrdersForEncoder(
  const VarDctFrameView& frame,
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

    std::vector<VarDctAcGroupView> groups(frame.ac_group_count());
    size_t coefficient_count = 0;
    for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
      Status status = frame.GetAcGroup(group_index, &groups[group_index]);
      if (!status.ok()) {
        return status;
      }
      if (groups[group_index].used_coefficient_count >
          std::numeric_limits<size_t>::max() - coefficient_count) {
        return Status::InvalidArgument(
          "Coefficient-order value count overflows");
      }
      coefficient_count += groups[group_index].used_coefficient_count;
    }

    uint16_t present_mask = 0;
    Status status;
    if (behavior ==
        VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled) {
      status = PresentOrderMask(
        groups, frame.strategies(), &present_mask);
      if (!status.ok()) return status;
    }
    const bool sample_dct8 =
      behavior == VarDctCoefficientOrderBehavior::kEffort7Dct8Sampled &&
      present_mask == 1;

    std::vector<std::vector<uint8_t>> sample_decisions;
    if (sample_dct8) {
      sample_decisions.resize(groups.size());
      std::array<uint64_t, 2> random_state = {
        0x94D049BB133111EBull,
        0xBF58476D1CE4E5B9ull,
      };
      for (size_t group_index = 0; group_index < groups.size();
           ++group_index) {
        size_t anchor_count = 0;
        if (!groups[group_index].block_extent.try_area(&anchor_count)) {
          return Status::InvalidArgument(
            "Coefficient-order sample count overflows");
        }
        std::vector<uint8_t>& decisions = sample_decisions[group_index];
        decisions.resize(anchor_count);
        for (uint8_t& selected : decisions) {
          selected = static_cast<uint8_t>(
            UseCoefficientOrderSample(&random_state));
        }
      }
    }

    ZeroCounts zero_counts;
    const size_t participant_count =
      CoefficientOrderParticipantCount(groups.size(), coefficient_count);
    if (participant_count == 1) {
      for (size_t group_index = 0; group_index < groups.size();
           ++group_index) {
        status = CountGroupZeros(
          groups[group_index], frame.strategies(), &zero_counts, sample_dct8,
          sample_dct8
            ? std::span<const uint8_t>(sample_decisions[group_index])
            : std::span<const uint8_t>{},
          &present_mask);
        if (!status.ok()) return status;
      }
    } else {
      std::array<ZeroCounts, kMaximumCoefficientOrderWorkers> worker_counts;
      std::array<uint16_t, kMaximumCoefficientOrderWorkers> worker_masks{};
      status = RunParallelCoefficientGroups(
        groups.size(), coefficient_count,
        [&](size_t group_index, size_t worker_index) {
          return CountGroupZeros(
            groups[group_index], frame.strategies(),
            &worker_counts[worker_index], sample_dct8,
            sample_dct8
              ? std::span<const uint8_t>(sample_decisions[group_index])
              : std::span<const uint8_t>{},
            &worker_masks[worker_index]);
        });
      if (!status.ok()) {
        return status;
      }

      for (size_t worker = 0; worker < worker_counts.size(); ++worker) {
        present_mask |= worker_masks[worker];
        for (size_t family = 0; family < zero_counts.size(); ++family) {
          for (size_t channel = 0; channel < 3; ++channel) {
            const std::vector<uint64_t>& source =
              worker_counts[worker][family][channel];
            if (source.empty()) {
              continue;
            }
            std::vector<uint64_t>& destination =
              zero_counts[family][channel];
            if (destination.empty()) {
              destination.assign(source.size(), 0);
            } else if (destination.size() != source.size()) {
              return Status::Internal(
                "Coefficient-order worker dimensions disagree");
            }
            for (size_t coefficient = 0; coefficient < source.size();
                 ++coefficient) {
              if (source[coefficient] >
                  std::numeric_limits<uint64_t>::max() -
                    destination[coefficient]) {
                return Status::InvalidArgument(
                  "Coefficient zero count overflow");
              }
              destination[coefficient] += source[coefficient];
            }
          }
        }
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
        const std::vector<uint64_t>& counts = zero_counts[family][channel];
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
