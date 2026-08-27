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

namespace gjxl {
namespace {

inline constexpr size_t kNonZeroBucketCount = 37;
inline constexpr size_t kZeroDensityContextCount = 458;
inline constexpr size_t kBlockContextCount = 4;

static_assert(kSimpleAcContextCount
              == kBlockContextCount
                   * (kNonZeroBucketCount + kZeroDensityContextCount));

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

// X, Y, and B rows for all 27 raw strategy codes. This is the tiny initial-
// profile map, not libjxl's more general default block-context map.
constexpr std::array<uint8_t, 3 * kAcStrategyCount> kBlockContextMap = {
  2, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 3, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

struct StrategyAnchor {
  size_t x;
  size_t y;
  AcStrategyType strategy;
  size_t coefficient_count;
};

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

uint32_t NonzeroContext(uint32_t prediction, uint32_t block_context) {
  const uint32_t bucket = prediction < 8     ? prediction
                          : prediction >= 64 ? 36
                                             : 4 + prediction / 2;
  return bucket * kBlockContextCount + block_context;
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

Status TokenizeSimpleAcGroup(const VarDctAcGroupView& group,
                             const AcStrategyGrid& strategies,
                             std::vector<EntropyToken>* tokens) {
  if (tokens == nullptr) {
    return Status::InvalidArgument("AC-group token output is null");
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
    std::vector<EntropyToken> candidate;
    candidate.reserve(3 * (group.used_coefficient_count + anchors.size()));

    size_t block_count = 0;
    if (!group.block_extent.try_area(&block_count)) {
      return Status::InvalidArgument("AC-group block count overflow");
    }
    std::array<std::vector<uint8_t>, 3> nonzero_maps;
    for (std::vector<uint8_t>& map : nonzero_maps) {
      map.assign(block_count, 0);
    }
    std::array<std::vector<uint32_t>, kAcStrategyCount> orders;
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
      if (!order_ready[strategy_index]) {
        status = ComputeSimpleNaturalCoefficientOrder(anchor.strategy,
                                                      &orders[strategy_index]);
        if (!status.ok()) {
          return status;
        }
        order_ready[strategy_index] = true;
      }
      const std::vector<uint32_t>& order = orders[strategy_index];
      const size_t covered_blocks =
        info->covered_blocks.width * info->covered_blocks.height;
      if (!std::has_single_bit(covered_blocks)) {
        return Status::Internal("AC strategy block count is invalid");
      }
      const size_t log2_covered_blocks = std::countr_zero(covered_blocks);

      for (const size_t channel : kChannelOrder) {
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
        const uint32_t block_context =
          kBlockContextMap[channel * kAcStrategyCount + strategy_index];
        candidate.push_back({
          NonzeroContext(prediction, block_context),
          static_cast<uint32_t>(nonzeros),
        });

        int32_t remaining_nonzeros = nonzeros;
        uint32_t previous_nonzero =
          nonzeros > static_cast<int32_t>(anchor.coefficient_count / 16) ? 0
                                                                         : 1;
        const uint32_t histogram_offset =
          static_cast<uint32_t>(kBlockContextCount * kNonZeroBucketCount
                                + kZeroDensityContextCount * block_context);
        for (size_t scan = covered_blocks;
             scan < anchor.coefficient_count && remaining_nonzeros != 0;
             ++scan) {
          const int32_t coefficient = coefficients[order[scan]];
          const uint32_t context =
            histogram_offset
            + ZeroDensityContext(remaining_nonzeros, scan, covered_blocks,
                                 log2_covered_blocks, previous_nonzero);
          candidate.push_back({context, PackSigned(coefficient)});
          previous_nonzero = coefficient != 0 ? 1 : 0;
          remaining_nonzeros -= static_cast<int32_t>(previous_nonzero);
        }
        if (remaining_nonzeros != 0) {
          return Status::Internal(
            "Natural coefficient scan missed a nonzero value");
        }
      }
      source_offset += anchor.coefficient_count;
    }

    if (source_offset != group.used_coefficient_count) {
      return Status::InvalidArgument(
        "AC-group coefficient consumption is incomplete");
    }
    *tokens = std::move(candidate);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

Status TokenizeSimpleAcGroups(const VarDctEncoderFrame& frame,
                              std::vector<SimpleAcGroupTokenStream>* groups) {
  if (groups == nullptr) {
    return Status::InvalidArgument("AC-group token output is null");
  }
  Status status = ValidateSimpleCodestreamFrame(frame);
  if (!status.ok()) {
    return status;
  }

  try {
    std::vector<SimpleAcGroupTokenStream> candidate;
    candidate.reserve(frame.ac_group_count());
    for (size_t group_index = 0; group_index < frame.ac_group_count();
         ++group_index) {
      VarDctAcGroupView group;
      status = frame.GetAcGroup(group_index, &group);
      if (!status.ok()) {
        return status;
      }
      SimpleAcGroupTokenStream stream{
        .block_x = group.block_x,
        .block_y = group.block_y,
        .block_extent = group.block_extent,
      };
      status = TokenizeSimpleAcGroup(group, frame.strategies(), &stream.tokens);
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

}  // namespace gjxl
