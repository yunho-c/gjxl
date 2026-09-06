// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/chroma_from_luma.h"

#include "core/managed_allocator.h"
#include "codec/chroma_from_luma_internal.h"
#include "codec/prepared_coefficients_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <vector>

#include "codec/dct.h"
#include "codec/quantization.h"
#include "core/ac_strategy.h"

namespace gjxl {
using resource_budget_internal::ManagedVector;

namespace {

constexpr float kBaseCorrelationX = 0.0f;
constexpr float kBaseCorrelationB = 1.0f;
constexpr float kDistanceMultiplierAc = 1.0e-9f;

struct Derivatives {
  float center = 0.0f;
  float plus_epsilon = 0.0f;
  float minus_epsilon = 0.0f;
};

Derivatives CflDerivatives(
  std::span<const float> luma,
  std::span<const float> chroma,
  float base,
  float x,
  float epsilon) {

  const float sample_count = static_cast<float>(luma.size());
  Derivatives result{
    .center = 2.0f * kDistanceMultiplierAc * sample_count * x,
    .plus_epsilon = 2.0f * kDistanceMultiplierAc *
      sample_count * (x + epsilon),
    .minus_epsilon = 2.0f * kDistanceMultiplierAc *
      sample_count * (x - epsilon),
  };

  const auto derivative = [](float a, float residual) {
    float value = (2.0f / 3.0f) * a * (std::abs(residual) + 1.0f);
    if (residual < 0.0f) {
      value = -value;
    }
    return value;
  };

  // The pinned arm64 libjxl path accumulates four SIMD lanes independently
  // before its horizontal sum. Preserve that order so the deliberately noisy
  // Newton approximation selects the same integer factor at its thresholds.
  std::array<std::array<float, 4>, 3> lane_sums{};
  for (size_t i = 0; i < luma.size(); ++i) {
    const float a = luma[i] / static_cast<float>(kDefaultColorFactor);
    const float b = base * luma[i] - chroma[i];
    const size_t lane = i % 4;
    const float center_residual = std::fma(a, x, b);
    // libjxl gates all three finite-difference samples with the center
    // residual's threshold mask.
    if (std::abs(center_residual) >= 100.0f) {
      continue;
    }
    lane_sums[0][lane] += derivative(a, center_residual);
    lane_sums[1][lane] += derivative(
      a,
      std::fma(a, x + epsilon, b));
    lane_sums[2][lane] += derivative(
      a,
      std::fma(a, x - epsilon, b));
  }

  const auto horizontal_sum = [](const std::array<float, 4>& lanes) {
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
  };
  result.center += horizontal_sum(lane_sums[0]);
  result.plus_epsilon += horizontal_sum(lane_sums[1]);
  result.minus_epsilon += horizontal_sum(lane_sums[2]);
  return result;
}

int8_t QuantizeMultiplier(float x) {
  constexpr float kTowardsZero = 2.6f;
  if (x >= kTowardsZero) {
    x -= kTowardsZero;
  } else if (x <= -kTowardsZero) {
    x += kTowardsZero;
  } else {
    x = 0.0f;
  }

  return static_cast<int8_t>(std::clamp(
    std::round(x),
    -128.0f,
    127.0f));
}

int8_t FindBestMultiplier(
  std::span<const float> luma,
  std::span<const float> chroma,
  float base,
  bool fast) {

  if (luma.empty()) {
    return 0;
  }

  float x = 0.0f;
  if (fast) {
    std::array<float, 4> quadratic{};
    std::array<float, 4> linear{};
    for (size_t i = 0; i < luma.size(); ++i) {
      const float a = luma[i] / static_cast<float>(kDefaultColorFactor);
      const float b = base * luma[i] - chroma[i];
      const size_t lane = i % 4;
      quadratic[lane] = std::fma(a, a, quadratic[lane]);
      linear[lane] = std::fma(a, b, linear[lane]);
    }
    const auto horizontal_sum = [](const std::array<float, 4>& lanes) {
      return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    };
    x = -horizontal_sum(linear) /
      (horizontal_sum(quadratic) +
       static_cast<float>(luma.size()) *
         kDistanceMultiplierAc * 0.5f);
  } else {
    constexpr float kEpsilon = 100.0f;
    constexpr float kStepClamp = 20.0f;
    constexpr float kStabilizer = 0.85f;
    for (size_t iteration = 0; iteration < 20; ++iteration) {
      const Derivatives derivatives = CflDerivatives(
        luma,
        chroma,
        base,
        x,
        kEpsilon);
      const float second_derivative =
        (derivatives.plus_epsilon - derivatives.minus_epsilon) /
        (2.0f * kEpsilon);
      const float step = derivatives.center /
        (second_derivative + kStabilizer);
      x -= std::clamp(step, -kStepClamp, kStepClamp);
      if (std::abs(step) < 3.0e-3f) {
        break;
      }
    }
  }

  return QuantizeMultiplier(x);
}

Status ComputeInitialPixelColorCorrelationMap(
  ConstImage3FView opsin,
  ColorCorrelationMap* out) {

  const Extent2D tile_extent = ColorTileExtent(opsin.extent());
  size_t tile_count = 0;
  if (!tile_extent.try_area(&tile_count)) {
    return Status::InvalidArgument(
      "Chroma-from-luma map dimensions are too large");
  }
  try {
    ManagedVector<int8_t> y_to_x(tile_count);
    ManagedVector<int8_t> y_to_b(tile_count);
    for (size_t tile_y = 0; tile_y < tile_extent.height; ++tile_y) {
      const size_t y_begin = tile_y * kColorTileDimension;
      const size_t y_end = std::min(
        y_begin + kColorTileDimension, opsin.height());
      for (size_t tile_x = 0; tile_x < tile_extent.width; ++tile_x) {
        const size_t x_begin = tile_x * kColorTileDimension;
        const size_t x_end = std::min(
          x_begin + kColorTileDimension, opsin.width());
        std::array<float, 4> sum_y{};
        std::array<float, 4> sum_x{};
        std::array<float, 4> sum_b{};
        size_t sample_index = 0;
        for (size_t y = y_begin; y < y_end; ++y) {
          for (size_t x = x_begin; x < x_end; ++x, ++sample_index) {
            const size_t lane = sample_index & 3u;
            const float value_y = opsin.plane[1].Row(y)[x];
            const float value_x = opsin.plane[0].Row(y)[x];
            const float value_b = opsin.plane[2].Row(y)[x];
            if (!std::isfinite(value_y) || !std::isfinite(value_x) ||
                !std::isfinite(value_b)) {
              return Status::InvalidArgument(
                "Chroma-from-luma input must contain only finite values");
            }
            sum_y[lane] += value_y;
            sum_x[lane] += value_x;
            sum_b[lane] += value_b;
          }
        }
        const auto horizontal_sum = [](const std::array<float, 4>& lanes) {
          return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        };
        const float sample_count = static_cast<float>(sample_index);
        const float mean_y = horizontal_sum(sum_y) / sample_count;
        const float mean_x = horizontal_sum(sum_x) / sample_count;
        const float mean_b = horizontal_sum(sum_b) / sample_count;
        std::array<float, 4> quadratic{};
        std::array<float, 4> linear_x{};
        std::array<float, 4> linear_b{};
        sample_index = 0;
        for (size_t y = y_begin; y < y_end; ++y) {
          for (size_t x = x_begin; x < x_end; ++x, ++sample_index) {
            const size_t lane = sample_index & 3u;
            const float centered_y = opsin.plane[1].Row(y)[x] - mean_y;
            const float a = centered_y /
              static_cast<float>(kDefaultColorFactor);
            quadratic[lane] = std::fma(a, a, quadratic[lane]);
            linear_x[lane] = std::fma(
              a, -(opsin.plane[0].Row(y)[x] - mean_x), linear_x[lane]);
            linear_b[lane] = std::fma(
              a,
              centered_y - (opsin.plane[2].Row(y)[x] - mean_b),
              linear_b[lane]);
          }
        }
        const float denominator = horizontal_sum(quadratic) +
          sample_count * kDistanceMultiplierAc * 0.5f;
        const size_t tile_index = tile_y * tile_extent.width + tile_x;
        y_to_x[tile_index] = QuantizeMultiplier(
          -horizontal_sum(linear_x) / denominator);
        y_to_b[tile_index] = QuantizeMultiplier(
          -horizontal_sum(linear_b) / denominator);
      }
    }
    return chroma_from_luma_internal::CreateColorCorrelationMap(
      {y_to_x.data(), tile_extent, tile_extent.width},
      {y_to_b.data(), tile_extent, tile_extent.width},
      out);
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate pixel-domain chroma-from-luma storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Chroma-from-luma map dimensions are too large");
  }
}

Status AppendStrategyCoefficients(
  ConstImage3FView opsin,
  size_t block_x,
  size_t block_y,
  AcStrategyType strategy,
  float quant_scale,
  std::array<ManagedVector<float>, 4>* values) {

  constexpr size_t kMaxCoefficientCount = 32 * 32;
  const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
  if (info == nullptr || !SupportsCpuDct(strategy) ||
      !std::isfinite(quant_scale) || quant_scale <= 0.0f) {
    return Status::InvalidArgument(
      "Chroma-from-luma strategy or quantization scale is invalid");
  }
  const Extent2D pixel_extent = info->pixel_extent();
  const size_t coefficient_count = info->coefficient_count();
  std::array<std::array<float, kMaxCoefficientCount>, 3> pixels{};
  std::array<std::array<float, kMaxCoefficientCount>, 3> coefficients{};
  for (size_t channel = 0; channel < pixels.size(); ++channel) {
    for (size_t y = 0; y < pixel_extent.height; ++y) {
      const float* row = opsin.plane[channel].Row(
        block_y * kJxlBlockDimension + y) +
        block_x * kJxlBlockDimension;
      for (size_t x = 0; x < pixel_extent.width; ++x) {
        if (!std::isfinite(row[x])) {
          return Status::InvalidArgument(
            "Chroma-from-luma input must contain only finite values");
        }
        pixels[channel][y * pixel_extent.width + x] = row[x];
      }
    }

    Status status = ForwardDctCpu(
      strategy,
      std::span<const float>(pixels[channel]).first(coefficient_count),
      std::span<float>(coefficients[channel]).first(coefficient_count));
    if (!status.ok()) {
      return status;
    }
  }

  const Extent2D coefficient_extent = info->coefficient_extent();
  const Extent2D low_frequency_extent = info->low_frequency_extent();
  for (size_t y = 0; y < low_frequency_extent.height; ++y) {
    for (size_t x = 0; x < low_frequency_extent.width; ++x) {
      const size_t index = y * coefficient_extent.width + x;
      for (auto& channel_coefficients : coefficients) {
        channel_coefficients[index] = 0.0f;
      }
    }
  }

  QuantizationMatrixView matrix_x;
  QuantizationMatrixView matrix_b;
  Status status = GetDefaultQuantizationMatrix(
    strategy, XybChannel::kX, &matrix_x);
  if (!status.ok()) {
    return status;
  }
  status = GetDefaultQuantizationMatrix(
    strategy, XybChannel::kB, &matrix_b);
  if (!status.ok()) {
    return status;
  }

  for (size_t i = 0; i < coefficient_count; ++i) {
    (*values)[0].push_back(
      coefficients[1][i] * matrix_x.inverse_dequant[i] * quant_scale);
    (*values)[1].push_back(
      coefficients[0][i] * matrix_x.inverse_dequant[i] * quant_scale);
    (*values)[2].push_back(
      coefficients[1][i] * matrix_b.inverse_dequant[i] * quant_scale);
    (*values)[3].push_back(
      coefficients[2][i] * matrix_b.inverse_dequant[i] * quant_scale);
  }
  return Status::Ok();
}

}  // namespace

namespace chroma_from_luma_internal {

Status CreateColorCorrelationMap(
  ConstPlaneI8View y_to_x,
  ConstPlaneI8View y_to_b,
  ColorCorrelationMap* out) {

  if (out == nullptr || !y_to_x.valid() || !y_to_b.valid() ||
      y_to_x.extent != y_to_b.extent) {
    return Status::InvalidArgument(
      "Color-correlation map input or output is invalid");
  }
  size_t tile_count = 0;
  if (!y_to_x.extent.try_area(&tile_count)) {
    return Status::InvalidArgument(
      "Color-correlation map dimensions are too large");
  }
  try {
    ColorCorrelationMap result;
    result.tile_extent_ = y_to_x.extent;
    result.y_to_x_.resize(tile_count);
    result.y_to_b_.resize(tile_count);
    for (size_t y = 0; y < y_to_x.extent.height; ++y) {
      std::copy_n(
        y_to_x.Row(y), y_to_x.extent.width,
        result.y_to_x_.data() + y * y_to_x.extent.width);
      std::copy_n(
        y_to_b.Row(y), y_to_b.extent.width,
        result.y_to_b_.data() + y * y_to_b.extent.width);
    }
    *out = std::move(result);
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate color-correlation map storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Color-correlation map dimensions are too large");
  }
}

}  // namespace chroma_from_luma_internal

bool ColorCorrelationMap::valid() const noexcept {
  size_t tile_count = 0;
  return !tile_extent_.empty() &&
    tile_extent_.try_area(&tile_count) &&
    y_to_x_.size() == tile_count &&
    y_to_b_.size() == tile_count;
}

Extent2D ColorCorrelationMap::tile_extent() const noexcept {
  return tile_extent_;
}

ConstPlaneI8View ColorCorrelationMap::y_to_x_map() const noexcept {
  return {
    .data = y_to_x_.data(),
    .extent = tile_extent_,
    .stride = tile_extent_.width,
  };
}

ConstPlaneI8View ColorCorrelationMap::y_to_b_map() const noexcept {
  return {
    .data = y_to_b_.data(),
    .extent = tile_extent_,
    .stride = tile_extent_.width,
  };
}

std::array<float, 3> ColorCorrelationMap::AcFactors(
  size_t tile_x,
  size_t tile_y) const noexcept {

  if (!valid() ||
      tile_x >= tile_extent_.width ||
      tile_y >= tile_extent_.height) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    return {nan, nan, nan};
  }

  const size_t index = tile_y * tile_extent_.width + tile_x;
  return {
    static_cast<float>(y_to_x_[index]) /
      static_cast<float>(kDefaultColorFactor),
    0.0f,
    kBaseCorrelationB +
      static_cast<float>(y_to_b_[index]) /
      static_cast<float>(kDefaultColorFactor),
  };
}

Status ComputeInitialColorCorrelationMap(
  ConstImage3FView opsin,
  ColorCorrelationMap* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Chroma-from-luma map output is null");
  }
  if (!opsin.valid() ||
      !BlockGrid::IsPaddedPixelExtent(opsin.extent())) {
    return Status::InvalidArgument(
      "Chroma-from-luma input must have valid 8x8-aligned geometry");
  }

  const Extent2D tile_extent = ColorTileExtent(opsin.extent());
  size_t tile_count = 0;
  if (!tile_extent.try_area(&tile_count)) {
    return Status::InvalidArgument(
      "Chroma-from-luma map dimensions are too large");
  }

  Status status;

  try {
    ColorCorrelationMap result;
    result.tile_extent_ = tile_extent;
    result.y_to_x_.resize(tile_count);
    result.y_to_b_.resize(tile_count);

    for (size_t tile_y = 0; tile_y < tile_extent.height; ++tile_y) {
      const size_t block_y_begin = tile_y *
        (kColorTileDimension / kJxlBlockDimension);
      const size_t block_y_end = std::min(
        block_y_begin + kColorTileDimension / kJxlBlockDimension,
        opsin.height() / kJxlBlockDimension);
      for (size_t tile_x = 0; tile_x < tile_extent.width; ++tile_x) {
        const size_t block_x_begin = tile_x *
          (kColorTileDimension / kJxlBlockDimension);
        const size_t block_x_end = std::min(
          block_x_begin + kColorTileDimension / kJxlBlockDimension,
          opsin.width() / kJxlBlockDimension);

        std::array<ManagedVector<float>, 4> values;
        const size_t coefficient_count =
          (block_x_end - block_x_begin) *
          (block_y_end - block_y_begin) * 64;
        for (ManagedVector<float>& channel_values : values) {
          channel_values.reserve(coefficient_count);
        }

        for (size_t block_y = block_y_begin;
             block_y < block_y_end;
             ++block_y) {
          for (size_t block_x = block_x_begin;
               block_x < block_x_end;
               ++block_x) {
            status = AppendStrategyCoefficients(
              opsin,
              block_x,
              block_y,
              AcStrategyType::kDct8,
              1.0f,
              &values);
            if (!status.ok()) {
              return status;
            }
          }
        }

        const size_t tile_index = tile_y * tile_extent.width + tile_x;
        result.y_to_x_[tile_index] = FindBestMultiplier(
          values[0],
          values[1],
          kBaseCorrelationX,
          false);
        result.y_to_b_[tile_index] = FindBestMultiplier(
          values[2],
          values[3],
          kBaseCorrelationB,
          false);
      }
    }

    *out = std::move(result);
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate chroma-from-luma working storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Chroma-from-luma map dimensions are too large");
  }

  return Status::Ok();
}

Status chroma_from_luma_internal::ComputeInitialColorCorrelationMapFast(
  ConstImage3FView opsin,
  ColorCorrelationMap* out) {

  if (out == nullptr || !opsin.valid() ||
      !BlockGrid::IsPaddedPixelExtent(opsin.extent())) {
    return Status::InvalidArgument(
      "Fast chroma-from-luma input or output is invalid");
  }
  return ComputeInitialPixelColorCorrelationMap(opsin, out);
}

Status ComputeFinalColorCorrelationMap(
  ConstImage3FView opsin,
  const AcStrategyGrid& strategies,
  ConstPlaneI32View raw_quant_field,
  const Quantizer& quantizer,
  bool fast,
  ColorCorrelationMap* out) {

  if (out == nullptr) {
    return Status::InvalidArgument(
      "Final chroma-from-luma map output is null");
  }
  if (!opsin.valid() ||
      !BlockGrid::IsPaddedPixelExtent(opsin.extent())) {
    return Status::InvalidArgument(
      "Final chroma-from-luma input must have valid block geometry");
  }
  const Extent2D block_extent =
    BlockGrid::FromPaddedPixelExtent(opsin.extent()).blocks;
  if (!strategies.complete() || strategies.extent() != block_extent ||
      !raw_quant_field.valid() || raw_quant_field.extent != block_extent ||
      !quantizer.valid()) {
    return Status::InvalidArgument(
      "Final chroma-from-luma strategy or quantization state is invalid");
  }

  const Extent2D tile_extent = ColorTileExtent(opsin.extent());
  size_t tile_count = 0;
  if (!tile_extent.try_area(&tile_count)) {
    return Status::InvalidArgument(
      "Final chroma-from-luma map dimensions are too large");
  }

  try {
    for (size_t y = 0; y < block_extent.height; ++y) {
      for (size_t x = 0; x < block_extent.width; ++x) {
        const int32_t raw_quant = raw_quant_field.Row(y)[x];
        if (raw_quant < 1 || raw_quant > kMaxRawQuant) {
          return Status::InvalidArgument(
            "Final chroma-from-luma raw quantization is out of range");
        }
      }
    }

    ColorCorrelationMap result;
    result.tile_extent_ = tile_extent;
    result.y_to_x_.resize(tile_count);
    result.y_to_b_.resize(tile_count);
    for (size_t tile_y = 0; tile_y < tile_extent.height; ++tile_y) {
      const size_t block_y_begin = tile_y * 8;
      const size_t block_y_end = std::min(
        block_y_begin + 8,
        block_extent.height);
      for (size_t tile_x = 0; tile_x < tile_extent.width; ++tile_x) {
        const size_t block_x_begin = tile_x * 8;
        const size_t block_x_end = std::min(
          block_x_begin + 8,
          block_extent.width);
        std::array<ManagedVector<float>, 4> values;
        const size_t coefficient_count =
          (block_x_end - block_x_begin) *
          (block_y_end - block_y_begin) * 64;
        for (ManagedVector<float>& channel_values : values) {
          channel_values.reserve(coefficient_count);
        }

        for (size_t block_y = block_y_begin;
             block_y < block_y_end;
             ++block_y) {
          for (size_t block_x = block_x_begin;
               block_x < block_x_end;
               ++block_x) {
            AcStrategyCell cell;
            Status status = strategies.Get(block_x, block_y, &cell);
            if (!status.ok()) {
              return status;
            }
            if (!cell.is_anchor) {
              continue;
            }
            const AcStrategyInfo* info = GetAcStrategyInfo(cell.strategy);
            if (info == nullptr ||
                block_x + info->covered_blocks.width > block_x_end ||
                block_y + info->covered_blocks.height > block_y_end) {
              return Status::InvalidArgument(
                "Final chroma-from-luma strategy crosses a color tile");
            }
            const float quant_scale =
              quantizer.scale() * 128.0f *
              static_cast<float>(raw_quant_field.Row(block_y)[block_x]);
            status = AppendStrategyCoefficients(
              opsin,
              block_x,
              block_y,
              cell.strategy,
              quant_scale,
              &values);
            if (!status.ok()) {
              return status;
            }
          }
        }

        const size_t tile_index = tile_y * tile_extent.width + tile_x;
        result.y_to_x_[tile_index] = FindBestMultiplier(
          values[0], values[1], kBaseCorrelationX, fast);
        result.y_to_b_[tile_index] = FindBestMultiplier(
          values[2], values[3], kBaseCorrelationB, fast);
      }
    }
    *out = std::move(result);
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate final chroma-from-luma working storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Final chroma-from-luma map dimensions are too large");
  }
  return Status::Ok();
}

Status chroma_from_luma_internal::ComputeFinalColorCorrelationMapPrepared(
  const prepared_coefficients_internal::PreparedForwardDctCoefficients&
    prepared,
  ConstPlaneI32View raw_quant_field,
  const Quantizer& quantizer,
  bool fast,
  ColorCorrelationMap* out) {

  if (out == nullptr || !prepared.valid() || !raw_quant_field.valid() ||
      raw_quant_field.extent != prepared.block_extent || !quantizer.valid()) {
    return Status::InvalidArgument(
      "Prepared final chroma-from-luma inputs are invalid");
  }
  for (size_t y = 0; y < raw_quant_field.extent.height; ++y) {
    for (size_t x = 0; x < raw_quant_field.extent.width; ++x) {
      const int32_t raw_quant = raw_quant_field.Row(y)[x];
      if (raw_quant < 1 || raw_quant > kMaxRawQuant) {
        return Status::InvalidArgument(
          "Prepared final chroma-from-luma quantization is out of range");
      }
    }
  }

  try {
    size_t tile_count = 0;
    if (!prepared.color_tile_extent.try_area(&tile_count)) {
      return Status::InvalidArgument(
        "Prepared final chroma-from-luma dimensions are too large");
    }
    ColorCorrelationMap result;
    result.tile_extent_ = prepared.color_tile_extent;
    result.y_to_x_.resize(tile_count);
    result.y_to_b_.resize(tile_count);
    for (size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
      const size_t transform_begin = prepared.color_tile_offsets[tile_index];
      const size_t transform_end =
        prepared.color_tile_offsets[tile_index + 1];
      size_t coefficient_count = 0;
      for (size_t position = transform_begin; position < transform_end;
           ++position) {
        const size_t transform_index =
          prepared.color_tile_transform_indices[position];
        coefficient_count +=
          prepared.transforms[transform_index].coefficient_count;
      }
      std::array<ManagedVector<float>, 4> values;
      for (auto& channel_values : values) {
        channel_values.reserve(coefficient_count);
      }

      for (size_t position = transform_begin; position < transform_end;
           ++position) {
        const size_t transform_index =
          prepared.color_tile_transform_indices[position];
        const prepared_coefficients_internal::PreparedTransform& transform =
          prepared.transforms[transform_index];
        const AcStrategyInfo* info = GetAcStrategyInfo(transform.strategy);
        if (info == nullptr ||
            info->coefficient_count() != transform.coefficient_count) {
          return Status::Internal(
            "Prepared final chroma-from-luma strategy disappeared");
        }
        QuantizationMatrixView matrix_x;
        QuantizationMatrixView matrix_b;
        Status status = GetDefaultQuantizationMatrix(
          transform.strategy, XybChannel::kX, &matrix_x);
        if (status.ok()) {
          status = GetDefaultQuantizationMatrix(
            transform.strategy, XybChannel::kB, &matrix_b);
        }
        if (!status.ok()) {
          return status;
        }
        const float quant_scale = quantizer.scale() * 128.0f *
          static_cast<float>(
            raw_quant_field.Row(transform.block_y)[transform.block_x]);
        const Extent2D coefficient_extent = info->coefficient_extent();
        const Extent2D low_frequency_extent = info->low_frequency_extent();
        for (size_t index = 0; index < transform.coefficient_count; ++index) {
          const size_t x = index % coefficient_extent.width;
          const size_t y = index / coefficient_extent.width;
          const bool low_frequency =
            x < low_frequency_extent.width && y < low_frequency_extent.height;
          const float coefficient_y = low_frequency
            ? 0.0f
            : prepared.coefficients[1][transform.coefficient_offset + index];
          const float coefficient_x = low_frequency
            ? 0.0f
            : prepared.coefficients[0][transform.coefficient_offset + index];
          const float coefficient_b = low_frequency
            ? 0.0f
            : prepared.coefficients[2][transform.coefficient_offset + index];
          values[0].push_back(
            coefficient_y * matrix_x.inverse_dequant[index] * quant_scale);
          values[1].push_back(
            coefficient_x * matrix_x.inverse_dequant[index] * quant_scale);
          values[2].push_back(
            coefficient_y * matrix_b.inverse_dequant[index] * quant_scale);
          values[3].push_back(
            coefficient_b * matrix_b.inverse_dequant[index] * quant_scale);
        }
      }
      result.y_to_x_[tile_index] = FindBestMultiplier(
        values[0], values[1], kBaseCorrelationX, fast);
      result.y_to_b_[tile_index] = FindBestMultiplier(
        values[2], values[3], kBaseCorrelationB, fast);
    }
    *out = std::move(result);
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate prepared final chroma-from-luma storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Prepared final chroma-from-luma dimensions are too large");
  }
}

}  // namespace gjxl
