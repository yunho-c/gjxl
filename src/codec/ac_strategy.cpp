// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/ac_strategy.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

#include "codec/dct.h"
#include "codec/quantization.h"
#include "codec/ac_strategy_search_policy.h"
#include "core/block_grid.h"
#include "util/fast_math.h"

namespace gjxl {
namespace {

constexpr size_t kMaxCoefficientCount = 32 * 32;

size_t CeilLog2Nonzero(size_t value) {
  return value <= 1 ? 0 : std::bit_width(value - 1);
}

Status ValidateCostInputs(
  AcStrategyType strategy,
  size_t block_x,
  size_t block_y,
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  AcStrategyCostOptions options,
  float* cost,
  const AcStrategyInfo** info) {

  if (cost == nullptr || info == nullptr) {
    return Status::InvalidArgument(
      "AC-strategy cost output is null");
  }

  if (!opsin.valid() ||
      !BlockGrid::IsPaddedPixelExtent(opsin.extent())) {
    return Status::InvalidArgument(
      "AC-strategy cost requires a padded opsin image");
  }

  const Extent2D block_extent =
    BlockGrid::FromPaddedPixelExtent(opsin.extent()).blocks;
  if (!quant_field.valid() ||
      quant_field.extent != block_extent ||
      !pixel_mask.valid() ||
      pixel_mask.extent != opsin.extent()) {
    return Status::InvalidArgument(
      "AC-strategy cost fields have invalid geometry");
  }

  const AcStrategyInfo* candidate = GetAcStrategyInfo(strategy);
  if (candidate == nullptr) {
    return Status::InvalidArgument(
      "Unknown AC strategy");
  }
  if (!SupportsCpuDct(strategy)) {
    return Status::Unavailable(
      "AC-strategy cost is unavailable for this strategy");
  }

  const Extent2D covered = candidate->covered_blocks;
  if (block_x >= block_extent.width ||
      block_y >= block_extent.height ||
      covered.width > block_extent.width - block_x ||
      covered.height > block_extent.height - block_y) {
    return Status::InvalidArgument(
      "AC-strategy candidate does not fit in the image");
  }

  if (!std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f ||
      !std::isfinite(options.entropy_multiplier) ||
      options.entropy_multiplier <= 0.0f ||
      options.cfl_factors[1] != 0.0f) {
    return Status::InvalidArgument(
      "AC-strategy cost options are invalid");
  }
  for (float factor : options.cfl_factors) {
    if (!std::isfinite(factor)) {
      return Status::InvalidArgument(
        "AC-strategy CfL factors must be finite");
    }
  }

  *info = candidate;
  return Status::Ok();
}

float QuantNorm(
  ConstPlaneF32View quant_field,
  size_t block_x,
  size_t block_y,
  Extent2D covered) {

  const size_t block_count = covered.width * covered.height;
  if (block_count == 1) {
    return quant_field.Row(block_y)[block_x];
  }

  if (block_count == 2) {
    if (covered.height == 2) {
      return std::max(
        quant_field.Row(block_y)[block_x],
        quant_field.Row(block_y + 1)[block_x]);
    }
    return std::max(
      quant_field.Row(block_y)[block_x],
      quant_field.Row(block_y)[block_x + 1]);
  }

  float sum = 0.0f;
  for (size_t dy = 0; dy < covered.height; ++dy) {
    for (size_t dx = 0; dx < covered.width; ++dx) {
      float value = quant_field.Row(block_y + dy)[block_x + dx];
      value *= value;
      value *= value;
      value *= value;
      sum += value * value;
    }
  }
  sum /= static_cast<float>(block_count);
  return fast_math::FastPow(sum, 1.0f / 16.0f);
}

class SearchGrid {
public:
  explicit SearchGrid(Extent2D extent)
    : extent_(extent), cells_(extent.width * extent.height, Encode(
        AcStrategyType::kDct8, true)) {}

  [[nodiscard]] AcStrategyCell Get(size_t x, size_t y) const {
    const uint8_t encoded = cells_[y * extent_.width + x];
    return {
      .strategy = static_cast<AcStrategyType>(encoded >> 1),
      .is_anchor = (encoded & 1u) != 0,
    };
  }

  void Set(size_t x, size_t y, AcStrategyType strategy) {
    const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
    for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
      for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
        cells_[(y + dy) * extent_.width + x + dx] = Encode(
          strategy,
          dx == 0 && dy == 0);
      }
    }
  }

  [[nodiscard]] bool CrossesHorizontalBoundary(
    size_t start_x,
    size_t y,
    size_t end_x) const {

    if (start_x >= extent_.width || y >= extent_.height || y % 8 == 0) {
      return false;
    }
    end_x = std::min(end_x, extent_.width);
    const size_t start_limit = start_x & ~size_t{7};
    while (start_x != start_limit && !Get(start_x, y).is_anchor) {
      --start_x;
    }
    for (size_t x = start_x; x < end_x;) {
      const AcStrategyCell cell = Get(x, y);
      if (!cell.is_anchor) {
        return true;
      }
      x += GetAcStrategyInfo(cell.strategy)->covered_blocks.width;
    }
    return false;
  }

  [[nodiscard]] bool CrossesVerticalBoundary(
    size_t x,
    size_t start_y,
    size_t end_y) const {

    if (x >= extent_.width || start_y >= extent_.height || x % 8 == 0) {
      return false;
    }
    end_y = std::min(end_y, extent_.height);
    const size_t start_limit = start_y & ~size_t{7};
    while (start_y != start_limit && !Get(x, start_y).is_anchor) {
      --start_y;
    }
    for (size_t y = start_y; y < end_y;) {
      const AcStrategyCell cell = Get(x, y);
      if (!cell.is_anchor) {
        return true;
      }
      y += GetAcStrategyInfo(cell.strategy)->covered_blocks.height;
    }
    return false;
  }

  [[nodiscard]] Status Export(AcStrategyGrid* out) const {
    AcStrategyGrid result;
    Status status = AcStrategyGrid::Create(extent_, &result);
    if (!status.ok()) {
      return status;
    }

    for (size_t y = 0; y < extent_.height; ++y) {
      for (size_t x = 0; x < extent_.width; ++x) {
        const AcStrategyCell cell = Get(x, y);
        if (!cell.is_anchor) {
          continue;
        }
        const AcStrategyInfo* info = GetAcStrategyInfo(cell.strategy);
        if (info == nullptr ||
            info->covered_blocks.width > extent_.width - x ||
            info->covered_blocks.height > extent_.height - y) {
          return Status::Internal(
            "AC-strategy search produced an invalid anchor");
        }
        for (size_t dy = 0; dy < info->covered_blocks.height; ++dy) {
          for (size_t dx = 0; dx < info->covered_blocks.width; ++dx) {
            const AcStrategyCell covered = Get(x + dx, y + dy);
            if (covered.strategy != cell.strategy ||
                covered.is_anchor != (dx == 0 && dy == 0)) {
              return Status::Internal(
                "AC-strategy search produced inconsistent coverage");
            }
          }
        }
        status = result.Set(x, y, cell.strategy);
        if (!status.ok()) {
          return Status::Internal(
            "AC-strategy search produced overlapping transforms");
        }
      }
    }

    if (!result.complete()) {
      return Status::Internal(
        "AC-strategy search produced an incomplete grid");
    }
    *out = std::move(result);
    return Status::Ok();
  }

private:
  [[nodiscard]] static uint8_t Encode(
    AcStrategyType strategy,
    bool anchor) {
    return static_cast<uint8_t>(
      (static_cast<uint8_t>(strategy) << 1) |
      static_cast<uint8_t>(anchor));
  }

  Extent2D extent_;
  std::vector<uint8_t> cells_;
};

struct SearchContext {
  ConstImage3FView opsin;
  ConstPlaneF32View quant_field;
  ConstPlaneF32View pixel_mask;
  float butteraugli_target;
  std::array<float, 3> cfl_factors;
  size_t tile_block_x;
  size_t tile_block_y;
  Extent2D tile_block_extent;
  SearchGrid* grid;
  std::array<float, 64> costs{};
  std::array<uint8_t, 64> priorities{};
};

Status CandidateCost(
  const SearchContext& context,
  AcStrategyType strategy,
  size_t local_x,
  size_t local_y,
  float entropy_multiplier,
  float* cost) {

  return EstimateAcStrategyCost(
    strategy,
    context.tile_block_x + local_x,
    context.tile_block_y + local_y,
    context.opsin,
    context.quant_field,
    context.pixel_mask,
    {
      .butteraugli_target = context.butteraugli_target,
      .entropy_multiplier = entropy_multiplier,
      .cfl_factors = context.cfl_factors,
    },
    cost);
}

void SetCostForTransform(
  SearchContext* context,
  size_t local_x,
  size_t local_y,
  AcStrategyType strategy,
  float cost) {

  const Extent2D covered = GetAcStrategyInfo(strategy)->covered_blocks;
  for (size_t dy = 0; dy < covered.height; ++dy) {
    for (size_t dx = 0; dx < covered.width; ++dx) {
      context->costs[(local_y + dy) * 8 + local_x + dx] = 0.0f;
    }
  }
  context->costs[local_y * 8 + local_x] = cost;
}

Status TryMerge(
  SearchContext* context,
  AcStrategyType strategy,
  size_t local_x,
  size_t local_y,
  float entropy_multiplier,
  uint8_t candidate_priority) {

  const Extent2D covered = GetAcStrategyInfo(strategy)->covered_blocks;
  float current_cost = 0.0f;
  for (size_t dy = 0; dy < covered.height; ++dy) {
    for (size_t dx = 0; dx < covered.width; ++dx) {
      const size_t index = (local_y + dy) * 8 + local_x + dx;
      if (context->priorities[index] >= candidate_priority) {
        return Status::Ok();
      }
      current_cost += context->costs[index];
    }
  }

  float candidate_cost = 0.0f;
  Status status = CandidateCost(
    *context,
    strategy,
    local_x,
    local_y,
    entropy_multiplier,
    &candidate_cost);
  if (!status.ok() || candidate_cost >= current_cost) {
    return status;
  }

  for (size_t dy = 0; dy < covered.height; ++dy) {
    for (size_t dx = 0; dx < covered.width; ++dx) {
      const size_t index = (local_y + dy) * 8 + local_x + dx;
      context->costs[index] = 0.0f;
      context->priorities[index] = candidate_priority;
    }
  }
  context->grid->Set(
    context->tile_block_x + local_x,
    context->tile_block_y + local_y,
    strategy);
  context->costs[local_y * 8 + local_x] = candidate_cost;
  return Status::Ok();
}

AcStrategyType SquareStrategy(size_t blocks) {
  return blocks == 2
    ? AcStrategyType::kDct16x16
    : AcStrategyType::kDct32x32;
}

AcStrategyType VerticalSplitStrategy(size_t blocks) {
  return blocks == 2
    ? AcStrategyType::kDct16x8
    : AcStrategyType::kDct32x16;
}

AcStrategyType HorizontalSplitStrategy(size_t blocks) {
  return blocks == 2
    ? AcStrategyType::kDct8x16
    : AcStrategyType::kDct16x32;
}

Status FindBestFirstLevelDivision(
  SearchContext* context,
  size_t blocks,
  size_t local_x,
  size_t local_y,
  float rectangle_entropy_multiplier,
  float square_entropy_multiplier) {

  const size_t half = blocks / 2;
  const size_t block_x = context->tile_block_x + local_x;
  const size_t block_y = context->tile_block_y + local_y;
  SearchGrid& grid = *context->grid;

  if (grid.CrossesHorizontalBoundary(
        block_x, block_y, block_x + blocks) ||
      grid.CrossesHorizontalBoundary(
        block_x, block_y + blocks, block_x + blocks) ||
      grid.CrossesVerticalBoundary(
        block_x, block_y, block_y + blocks) ||
      grid.CrossesVerticalBoundary(
        block_x + blocks, block_y, block_y + blocks)) {
    return Status::Ok();
  }

  const bool allow_vertical = !grid.CrossesVerticalBoundary(
    block_x + half,
    block_y,
    block_y + blocks);
  const bool allow_horizontal = !grid.CrossesHorizontalBoundary(
    block_x,
    block_y + half,
    block_x + blocks);

  std::array<std::array<float, 2>, 2> quadrant_costs{};
  for (size_t dy = 0; dy < blocks; ++dy) {
    for (size_t dx = 0; dx < blocks; ++dx) {
      quadrant_costs[dy / half][dx / half] +=
        context->costs[(local_y + dy) * 8 + local_x + dx];
    }
  }

  constexpr float kUnavailable = std::numeric_limits<float>::max();
  float vertical_left = kUnavailable;
  float vertical_right = kUnavailable;
  float horizontal_top = kUnavailable;
  float horizontal_bottom = kUnavailable;
  float square = kUnavailable;
  const AcStrategyType vertical_strategy = VerticalSplitStrategy(blocks);
  const AcStrategyType horizontal_strategy = HorizontalSplitStrategy(blocks);
  const AcStrategyType square_strategy = SquareStrategy(blocks);
  Status status;

  if (allow_vertical) {
    if (grid.Get(block_x, block_y).strategy != vertical_strategy) {
      status = CandidateCost(
        *context,
        vertical_strategy,
        local_x,
        local_y,
        rectangle_entropy_multiplier,
        &vertical_left);
      if (!status.ok()) {
        return status;
      }
    }
    if (grid.Get(block_x + half, block_y).strategy != vertical_strategy) {
      status = CandidateCost(
        *context,
        vertical_strategy,
        local_x + half,
        local_y,
        rectangle_entropy_multiplier,
        &vertical_right);
      if (!status.ok()) {
        return status;
      }
    }
  }
  if (allow_horizontal) {
    if (grid.Get(block_x, block_y).strategy != horizontal_strategy) {
      status = CandidateCost(
        *context,
        horizontal_strategy,
        local_x,
        local_y,
        rectangle_entropy_multiplier,
        &horizontal_top);
      if (!status.ok()) {
        return status;
      }
    }
    if (grid.Get(block_x, block_y + half).strategy != horizontal_strategy) {
      status = CandidateCost(
        *context,
        horizontal_strategy,
        local_x,
        local_y + half,
        rectangle_entropy_multiplier,
        &horizontal_bottom);
      if (!status.ok()) {
        return status;
      }
    }
  }
  status = CandidateCost(
    *context,
    square_strategy,
    local_x,
    local_y,
    square_entropy_multiplier,
    &square);
  if (!status.ok()) {
    return status;
  }

  const float vertical_cost =
    std::min(
      vertical_left,
      quadrant_costs[0][0] + quadrant_costs[1][0]) +
    std::min(
      vertical_right,
      quadrant_costs[0][1] + quadrant_costs[1][1]);
  const float horizontal_cost =
    std::min(
      horizontal_top,
      quadrant_costs[0][0] + quadrant_costs[0][1]) +
    std::min(
      horizontal_bottom,
      quadrant_costs[1][0] + quadrant_costs[1][1]);

  const ac_strategy_internal::FirstLevelDivision division =
    ac_strategy_internal::ChooseFirstLevelDivision(
      square,
      vertical_cost,
      horizontal_cost);
  if (division == ac_strategy_internal::FirstLevelDivision::kSquare) {
    grid.Set(block_x, block_y, square_strategy);
    SetCostForTransform(
      context,
      local_x,
      local_y,
      square_strategy,
      square);
  } else if (division ==
             ac_strategy_internal::FirstLevelDivision::kVertical) {
    if (vertical_left < quadrant_costs[0][0] + quadrant_costs[1][0]) {
      grid.Set(block_x, block_y, vertical_strategy);
      SetCostForTransform(
        context,
        local_x,
        local_y,
        vertical_strategy,
        vertical_left);
    }
    if (vertical_right < quadrant_costs[0][1] + quadrant_costs[1][1]) {
      grid.Set(block_x + half, block_y, vertical_strategy);
      SetCostForTransform(
        context,
        local_x + half,
        local_y,
        vertical_strategy,
        vertical_right);
    }
  } else {
    if (horizontal_top < quadrant_costs[0][0] + quadrant_costs[0][1]) {
      grid.Set(block_x, block_y, horizontal_strategy);
      SetCostForTransform(
        context,
        local_x,
        local_y,
        horizontal_strategy,
        horizontal_top);
    }
    if (horizontal_bottom < quadrant_costs[1][0] + quadrant_costs[1][1]) {
      grid.Set(block_x, block_y + half, horizontal_strategy);
      SetCostForTransform(
        context,
        local_x,
        local_y + half,
        horizontal_strategy,
        horizontal_bottom);
    }
  }
  return Status::Ok();
}

Status SearchTile(SearchContext* context) {
  constexpr float kDct8MultiplierSlope = -0.4f;
  constexpr float kDct8MultiplierOffset = 1.0f;
  constexpr float kDct8MultiplierBase = 1.4f;
  const float baseline_multiplier = kDct8MultiplierOffset +
    kDct8MultiplierSlope /
      (context->butteraugli_target + kDct8MultiplierBase);

  for (size_t y = 0; y < context->tile_block_extent.height; ++y) {
    for (size_t x = 0; x < context->tile_block_extent.width; ++x) {
      float cost = 0.0f;
      Status status = CandidateCost(
        *context,
        AcStrategyType::kDct8,
        x,
        y,
        1.0f,
        &cost);
      if (!status.ok()) {
        return status;
      }
      context->costs[y * 8 + x] = cost * baseline_multiplier;
    }
  }

  struct MergeTry {
    AcStrategyType strategy;
    uint8_t priority;
    float entropy_multiplier;
  };
  constexpr std::array kMergeTries = {
    MergeTry{AcStrategyType::kDct16x8, 2, 1.21f},
    MergeTry{AcStrategyType::kDct8x16, 2, 1.21f},
    MergeTry{AcStrategyType::kDct16x32, 4, 1.49f},
    MergeTry{AcStrategyType::kDct32x16, 4, 1.49f},
  };

  for (const MergeTry merge : kMergeTries) {
    const Extent2D covered = GetAcStrategyInfo(merge.strategy)->covered_blocks;
    for (size_t y = 0;
         y + covered.height <= context->tile_block_extent.height;
         y += covered.height) {
      for (size_t x = 0;
           x + covered.width <= context->tile_block_extent.width;
           x += covered.width) {
        if (y + 3 < context->tile_block_extent.height &&
            x + 3 < context->tile_block_extent.width) {
          if (merge.strategy == AcStrategyType::kDct16x32) {
            if ((y | x) % 4 == 0) {
              Status status = FindBestFirstLevelDivision(
                context, 4, x, y, 1.49f, 1.48f);
              if (!status.ok()) {
                return status;
              }
            }
            continue;
          }
          if (merge.strategy == AcStrategyType::kDct32x16) {
            continue;
          }
        }
        if ((merge.strategy == AcStrategyType::kDct16x32 && y % 4 != 0) ||
            (merge.strategy == AcStrategyType::kDct32x16 && x % 4 != 0)) {
          continue;
        }

        if (y + 1 < context->tile_block_extent.height &&
            x + 1 < context->tile_block_extent.width) {
          if (merge.strategy == AcStrategyType::kDct8x16) {
            if ((y | x) % 2 == 0) {
              Status status = FindBestFirstLevelDivision(
                context, 2, x, y, 1.21f, 1.34f);
              if (!status.ok()) {
                return status;
              }
            }
            continue;
          }
          if (merge.strategy == AcStrategyType::kDct16x8) {
            continue;
          }
        }
        if ((merge.strategy == AcStrategyType::kDct8x16 && y % 2 == 1) ||
            (merge.strategy == AcStrategyType::kDct16x8 && x % 2 == 1)) {
          continue;
        }

        Status status = TryMerge(
          context,
          merge.strategy,
          x,
          y,
          merge.entropy_multiplier,
          merge.priority);
        if (!status.ok()) {
          return status;
        }
      }
    }
  }

  for (size_t y = 0; y + 1 < context->tile_block_extent.height; ++y) {
    for (size_t x = 0; x + 1 < context->tile_block_extent.width; ++x) {
      if ((y | x) % 2 != 0) {
        Status status = FindBestFirstLevelDivision(
          context, 2, x, y, 1.21f, 1.34f);
        if (!status.ok()) {
          return status;
        }
      }
    }
  }

  constexpr size_t kDct32SearchStep = 2;
  for (size_t y = 0;
       y + 3 < context->tile_block_extent.height;
       y += kDct32SearchStep) {
    for (size_t x = 0;
         x + 3 < context->tile_block_extent.width;
         x += kDct32SearchStep) {
      if ((y | x) % 4 == 0) {
        continue;
      }
      Status status = FindBestFirstLevelDivision(
        context, 4, x, y, 1.49f, 1.48f);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return Status::Ok();
}

}  // namespace

Status EstimateAcStrategyCost(
  AcStrategyType strategy,
  size_t block_x,
  size_t block_y,
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  AcStrategyCostOptions options,
  float* cost) {

  const AcStrategyInfo* info = nullptr;
  Status status = ValidateCostInputs(
    strategy,
    block_x,
    block_y,
    opsin,
    quant_field,
    pixel_mask,
    options,
    cost,
    &info);
  if (!status.ok()) {
    return status;
  }

  const Extent2D extent = info->pixel_extent();
  const Extent2D covered = info->covered_blocks;
  const size_t coefficient_count = info->coefficient_count();
  const size_t pixel_x = block_x * kJxlBlockDimension;
  const size_t pixel_y = block_y * kJxlBlockDimension;

  std::array<std::array<float, kMaxCoefficientCount>, 3> coefficients{};
  std::array<float, kMaxCoefficientCount> pixels{};
  for (size_t channel = 0; channel < coefficients.size(); ++channel) {
    for (size_t y = 0; y < extent.height; ++y) {
      const float* source = opsin.plane[channel].Row(pixel_y + y) + pixel_x;
      for (size_t x = 0; x < extent.width; ++x) {
        if (!std::isfinite(source[x])) {
          return Status::InvalidArgument(
            "AC-strategy source samples must be finite");
        }
        pixels[y * extent.width + x] = source[x];
      }
    }

    status = ForwardDctCpu(
      strategy,
      std::span<const float>(pixels).first(coefficient_count),
      std::span<float>(coefficients[channel]).first(coefficient_count));
    if (!status.ok()) {
      return status;
    }
  }

  const float quant_norm = QuantNorm(
    quant_field,
    block_x,
    block_y,
    covered);
  if (!std::isfinite(quant_norm) || quant_norm <= 0.0f) {
    return Status::InvalidArgument(
      "AC-strategy quant field must be finite and positive");
  }

  constexpr float kBias = 0.13731742964354549f;
  const float ratio =
    (options.butteraugli_target + kBias) / (1.0f + kBias);
  const float info_loss_multiplier = 1.2f *
    std::pow(ratio, 0.33677806662454718f);
  const float zeros_multiplier = 9.3089059022677905f *
    std::pow(ratio, 0.50990926717963703f);
  const float cost_delta = 10.833273317067883f *
    std::pow(ratio, 0.36702940662370243f);

  constexpr std::array<float, 3> kMaskOffset = {12.0f, 0.0f, 4.0f};
  constexpr std::array<float, 3> kChannelMultiplier = {
    2.0441408586549744e7f,
    1.0f,
    1.266770081387616f,
  };

  float entropy = 0.0f;
  float loss = 0.0f;
  std::array<float, kMaxCoefficientCount> residual_coefficients{};
  std::array<float, kMaxCoefficientCount> residual_pixels{};

  for (size_t channel = 0; channel < coefficients.size(); ++channel) {
    QuantizationMatrixView matrix;
    status = GetDefaultQuantizationMatrix(
      strategy,
      static_cast<XybChannel>(channel),
      &matrix);
    if (!status.ok()) {
      return status;
    }

    float magnitude_cost = 0.0f;
    size_t nonzero_count = 0;
    for (size_t i = 0; i < coefficient_count; ++i) {
      const float decorrelated = coefficients[channel][i] -
        coefficients[1][i] * options.cfl_factors[channel];
      const float scaled = decorrelated *
        matrix.inverse_dequant[i] * quant_norm;
      const float rounded = std::round(scaled);
      const float difference = scaled - rounded;
      residual_coefficients[i] = matrix.dequant[i] * difference;
      magnitude_cost += std::sqrt(std::abs(rounded));
      nonzero_count += rounded != 0.0f ? 1 : 0;
    }

    status = InverseDctCpu(
      strategy,
      std::span<const float>(residual_coefficients).first(coefficient_count),
      std::span<float>(residual_pixels).first(coefficient_count));
    if (!status.ok()) {
      return status;
    }

    float channel_loss = 0.0f;
    for (size_t y = 0; y < extent.height; ++y) {
      const float* mask_row = pixel_mask.Row(pixel_y + y) + pixel_x;
      for (size_t x = 0; x < extent.width; ++x) {
        if (!std::isfinite(mask_row[x]) || mask_row[x] <= 0.0f) {
          return Status::InvalidArgument(
            "AC-strategy pixel mask must be finite and positive");
        }
        float weighted =
          (mask_row[x] + kMaskOffset[channel]) *
          residual_pixels[y * extent.width + x];
        weighted *= weighted;
        weighted *= weighted;
        weighted *= weighted;
        channel_loss += weighted;
      }
    }
    loss += channel_loss * kChannelMultiplier[channel];

    entropy += cost_delta * magnitude_cost;
    const size_t nonzero_bits = CeilLog2Nonzero(nonzero_count + 1) + 1;
    entropy += zeros_multiplier * static_cast<float>(
      CeilLog2Nonzero(nonzero_bits + 17) + nonzero_bits);

    const size_t block_count = covered.width * covered.height;
    if (channel == 0 && block_count >= 2) {
      const float weight = 1.0f + std::min(
        3.0f,
        static_cast<float>(block_count) / 8.0f);
      entropy *= weight;
      loss *= weight;
    }
  }

  const float normalized_loss =
    loss / static_cast<float>(coefficient_count);
  const float loss_cost =
    std::pow(normalized_loss, 1.0f / 8.0f) *
    static_cast<float>(coefficient_count) /
    quant_norm;
  const float result =
    entropy * options.entropy_multiplier +
    info_loss_multiplier * loss_cost;
  if (!std::isfinite(result) || result < 0.0f) {
    return Status::InvalidArgument(
      "AC-strategy cost produced a non-finite result");
  }

  *cost = result;
  return Status::Ok();
}

Status FindAcStrategyGrid(
  ConstImage3FView opsin,
  ConstPlaneF32View quant_field,
  ConstPlaneF32View pixel_mask,
  const ColorCorrelationMap& color_correlation,
  AcStrategySearchOptions options,
  AcStrategyGrid* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "AC-strategy grid output is null");
  }
  if (!opsin.valid() ||
      !BlockGrid::IsPaddedPixelExtent(opsin.extent())) {
    return Status::InvalidArgument(
      "AC-strategy search requires a padded opsin image");
  }
  const Extent2D block_extent =
    BlockGrid::FromPaddedPixelExtent(opsin.extent()).blocks;
  size_t block_count = 0;
  if (!block_extent.try_area(&block_count)) {
    return Status::InvalidArgument(
      "AC-strategy search dimensions are too large");
  }
  if (!quant_field.valid() || quant_field.extent != block_extent ||
      !pixel_mask.valid() || pixel_mask.extent != opsin.extent() ||
      !color_correlation.valid()) {
    return Status::InvalidArgument(
      "AC-strategy search fields have invalid geometry");
  }
  const Extent2D expected_tile_extent = ColorTileExtent(opsin.extent());
  if (color_correlation.tile_extent() != expected_tile_extent ||
      !std::isfinite(options.butteraugli_target) ||
      options.butteraugli_target <= 0.0f) {
    return Status::InvalidArgument(
      "AC-strategy search options or color map are invalid");
  }

  try {
    SearchGrid search_grid(block_extent);
    for (size_t tile_y = 0; tile_y < expected_tile_extent.height; ++tile_y) {
      const size_t block_y = tile_y * 8;
      const size_t tile_height = std::min<size_t>(
        8,
        block_extent.height - block_y);
      for (size_t tile_x = 0; tile_x < expected_tile_extent.width; ++tile_x) {
        const size_t block_x = tile_x * 8;
        const size_t tile_width = std::min<size_t>(
          8,
          block_extent.width - block_x);
        SearchContext context{
          .opsin = opsin,
          .quant_field = quant_field,
          .pixel_mask = pixel_mask,
          .butteraugli_target = options.butteraugli_target,
          .cfl_factors = color_correlation.AcFactors(tile_x, tile_y),
          .tile_block_x = block_x,
          .tile_block_y = block_y,
          .tile_block_extent = {tile_width, tile_height},
          .grid = &search_grid,
        };
        Status status = SearchTile(&context);
        if (!status.ok()) {
          return status;
        }
      }
    }
    return search_grid.Export(out);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate AC-strategy search state");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "AC-strategy search dimensions are too large");
  }
}

}  // namespace gjxl
