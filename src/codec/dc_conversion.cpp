// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/dc_conversion.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace gjxl {
namespace {

constexpr size_t kMaxSupportedBlockDimension = 4;
constexpr size_t kMaxSupportedBlockCount =
  kMaxSupportedBlockDimension * kMaxSupportedBlockDimension;

bool IsSupportedStrategy(AcStrategyType strategy) {
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

Status ValidateOperation(
  AcStrategyType strategy,
  size_t coefficient_count,
  Extent2D dc_extent,
  const AcStrategyInfo** strategy_info) {

  if (strategy_info == nullptr) {
    return Status::InvalidArgument(
      "AC strategy output is null");
  }

  const AcStrategyInfo* info = GetAcStrategyInfo(strategy);
  if (info == nullptr) {
    return Status::InvalidArgument(
      "Unknown AC strategy");
  }

  if (!IsSupportedStrategy(strategy)) {
    return Status::InvalidArgument(
      "DC conversion is not implemented for this AC strategy");
  }

  if (coefficient_count != info->coefficient_count()) {
    return Status::InvalidArgument(
      "Coefficient span has the wrong size for its strategy");
  }

  if (dc_extent != info->covered_blocks) {
    return Status::InvalidArgument(
      "DC view has the wrong extent for its strategy");
  }

  *strategy_info = info;
  return Status::Ok();
}

double ForwardBasis(
  size_t length,
  size_t frequency,
  size_t sample) {

  const double alpha = frequency == 0
    ? std::numbers::sqrt2_v<double> / 2.0
    : 1.0;
  const double angle =
    (static_cast<double>(sample) + 0.5) *
    static_cast<double>(frequency) *
    std::numbers::pi_v<double> /
    static_cast<double>(length);

  return std::numbers::sqrt2_v<double> * alpha *
    std::cos(angle) / static_cast<double>(length);
}

double InverseBasis(
  size_t length,
  size_t frequency,
  size_t sample) {

  const double alpha = frequency == 0
    ? std::numbers::sqrt2_v<double> / 2.0
    : 1.0;
  const double angle =
    (static_cast<double>(sample) + 0.5) *
    static_cast<double>(frequency) *
    std::numbers::pi_v<double> /
    static_cast<double>(length);

  return std::numbers::sqrt2_v<double> * alpha * std::cos(angle);
}

float DownsampleScale(size_t length, size_t frequency) {
  // libjxl's DCTTotalResampleScale corrections for an 8x spatial reduction.
  constexpr std::array<float, 1> kDct8To1 = {
    1.0f,
  };
  constexpr std::array<float, 2> kDct16To2 = {
    1.0f,
    0.901764195028874394f,
  };
  constexpr std::array<float, 4> kDct32To4 = {
    1.0f,
    0.974886821136879522f,
    0.901764195028874394f,
    0.787054918159101335f,
  };

  switch (length) {
    case 1:
      return kDct8To1[frequency];
    case 2:
      return kDct16To2[frequency];
    case 4:
      return kDct32To4[frequency];
    default:
      return 0.0f;
  }
}

float UpsampleScale(size_t length, size_t frequency) {
  // Exact table counterparts used while expanding the DC grid back to LLF.
  constexpr std::array<float, 1> kDct1To8 = {
    1.0f,
  };
  constexpr std::array<float, 2> kDct2To16 = {
    1.0f,
    1.108937353592731823f,
  };
  constexpr std::array<float, 4> kDct4To32 = {
    1.0f,
    1.025760096781116015f,
    1.108937353592731823f,
    1.270559368765487251f,
  };

  switch (length) {
    case 1:
      return kDct1To8[frequency];
    case 2:
      return kDct2To16[frequency];
    case 4:
      return kDct4To32[frequency];
    default:
      return 0.0f;
  }
}

size_t SmallCoefficientIndex(
  Extent2D extent,
  size_t vertical_frequency,
  size_t horizontal_frequency) {

  if (extent.height < extent.width) {
    return vertical_frequency * extent.width + horizontal_frequency;
  }

  return horizontal_frequency * extent.height + vertical_frequency;
}

}  // namespace

Status ConvertLowFrequenciesToDc(
  AcStrategyType strategy,
  std::span<const float> coefficients,
  PlaneF32View dc) {

  if (!dc.valid()) {
    return Status::InvalidArgument(
      "DC output view is invalid");
  }

  const AcStrategyInfo* info = nullptr;
  Status status = ValidateOperation(
    strategy,
    coefficients.size(),
    dc.extent,
    &info);

  if (!status.ok()) {
    return status;
  }

  const Extent2D block_extent = info->covered_blocks;
  std::array<double, kMaxSupportedBlockCount> scaled_low_frequencies{};
  std::array<float, kMaxSupportedBlockCount> output{};

  for (size_t v = 0; v < block_extent.height; ++v) {
    for (size_t u = 0; u < block_extent.width; ++u) {
      const float coefficient =
        coefficients[info->coefficient_index(v, u)];

      if (!std::isfinite(coefficient)) {
        return Status::InvalidArgument(
          "Low-frequency coefficients must be finite");
      }

      scaled_low_frequencies[
        SmallCoefficientIndex(block_extent, v, u)] =
          static_cast<double>(coefficient) *
          static_cast<double>(
            DownsampleScale(block_extent.height, v)) *
          static_cast<double>(
            DownsampleScale(block_extent.width, u));
    }
  }

  for (size_t y = 0; y < block_extent.height; ++y) {
    for (size_t x = 0; x < block_extent.width; ++x) {
      double value = 0.0;

      for (size_t v = 0; v < block_extent.height; ++v) {
        for (size_t u = 0; u < block_extent.width; ++u) {
          value +=
            scaled_low_frequencies[
              SmallCoefficientIndex(block_extent, v, u)] *
            InverseBasis(block_extent.height, v, y) *
            InverseBasis(block_extent.width, u, x);
        }
      }

      const float converted = static_cast<float>(value);
      if (!std::isfinite(converted)) {
        return Status::InvalidArgument(
          "Extracted DC value is not finite");
      }
      output[y * block_extent.width + x] = converted;
    }
  }

  for (size_t y = 0; y < block_extent.height; ++y) {
    for (size_t x = 0; x < block_extent.width; ++x) {
      dc.Row(y)[x] = output[y * block_extent.width + x];
    }
  }

  return Status::Ok();
}

Status ConvertDcToLowFrequencies(
  AcStrategyType strategy,
  ConstPlaneF32View dc,
  std::span<float> coefficients) {

  if (!dc.valid()) {
    return Status::InvalidArgument(
      "DC input view is invalid");
  }

  const AcStrategyInfo* info = nullptr;
  Status status = ValidateOperation(
    strategy,
    coefficients.size(),
    dc.extent,
    &info);

  if (!status.ok()) {
    return status;
  }

  const Extent2D block_extent = info->covered_blocks;
  for (size_t y = 0; y < block_extent.height; ++y) {
    for (size_t x = 0; x < block_extent.width; ++x) {
      if (!std::isfinite(dc.Row(y)[x])) {
        return Status::InvalidArgument(
          "DC values must be finite");
      }
    }
  }

  std::array<float, kMaxSupportedBlockCount> low_frequencies{};

  for (size_t v = 0; v < block_extent.height; ++v) {
    for (size_t u = 0; u < block_extent.width; ++u) {
      double value = 0.0;

      for (size_t y = 0; y < block_extent.height; ++y) {
        for (size_t x = 0; x < block_extent.width; ++x) {
          value +=
            static_cast<double>(dc.Row(y)[x]) *
            ForwardBasis(block_extent.height, v, y) *
            ForwardBasis(block_extent.width, u, x);
        }
      }

      const float converted = static_cast<float>(
        value *
        static_cast<double>(UpsampleScale(block_extent.height, v)) *
        static_cast<double>(UpsampleScale(block_extent.width, u)));
      if (!std::isfinite(converted)) {
        return Status::InvalidArgument(
          "Restored low-frequency coefficient is not finite");
      }
      low_frequencies[SmallCoefficientIndex(block_extent, v, u)] = converted;
    }
  }

  for (size_t v = 0; v < block_extent.height; ++v) {
    for (size_t u = 0; u < block_extent.width; ++u) {
      coefficients[info->coefficient_index(v, u)] =
        low_frequencies[SmallCoefficientIndex(block_extent, v, u)];
    }
  }

  return Status::Ok();
}

}  // namespace gjxl
