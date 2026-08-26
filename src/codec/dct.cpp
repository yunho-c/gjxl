// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codec/dct.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>

namespace gjxl {
namespace {

enum class BasisDirection {
  kForward,
  kInverse,
};

template <size_t N>
std::array<double, N * N> MakeBasis(BasisDirection direction) {
  std::array<double, N * N> basis{};
  const double sqrt_two = std::numbers::sqrt2_v<double>;
  const double scale = direction == BasisDirection::kForward
    ? sqrt_two / static_cast<double>(N)
    : sqrt_two;

  for (size_t frequency = 0; frequency < N; ++frequency) {
    const double alpha = frequency == 0 ? 1.0 / sqrt_two : 1.0;
    for (size_t sample = 0; sample < N; ++sample) {
      const double angle =
        (static_cast<double>(sample) + 0.5) *
        static_cast<double>(frequency) *
        std::numbers::pi_v<double> /
        static_cast<double>(N);
      basis[frequency * N + sample] =
        scale * alpha * std::cos(angle);
    }
  }

  return basis;
}

std::span<const double> Basis(
  size_t length,
  BasisDirection direction) {

  static const auto kForward8 = MakeBasis<8>(BasisDirection::kForward);
  static const auto kForward16 = MakeBasis<16>(BasisDirection::kForward);
  static const auto kForward32 = MakeBasis<32>(BasisDirection::kForward);
  static const auto kInverse8 = MakeBasis<8>(BasisDirection::kInverse);
  static const auto kInverse16 = MakeBasis<16>(BasisDirection::kInverse);
  static const auto kInverse32 = MakeBasis<32>(BasisDirection::kInverse);

  if (direction == BasisDirection::kForward) {
    switch (length) {
      case 8:
        return kForward8;
      case 16:
        return kForward16;
      case 32:
        return kForward32;
      default:
        return {};
    }
  }

  switch (length) {
    case 8:
      return kInverse8;
    case 16:
      return kInverse16;
    case 32:
      return kInverse32;
    default:
      return {};
  }
}

Status ValidateDct(
  AcStrategyType strategy,
  size_t input_size,
  size_t output_size,
  const AcStrategyInfo** info) {

  if (info == nullptr) {
    return Status::Internal(
      "CPU DCT strategy output is null");
  }

  const AcStrategyInfo* candidate = GetAcStrategyInfo(strategy);
  if (candidate == nullptr) {
    return Status::InvalidArgument(
      "Unknown AC strategy");
  }

  if (!SupportsCpuDct(strategy)) {
    return Status::Unavailable(
      "Scalar CPU DCT is unavailable for this strategy");
  }

  if (input_size != candidate->coefficient_count() ||
      output_size != candidate->coefficient_count()) {
    return Status::InvalidArgument(
      "CPU DCT input or output has the wrong size");
  }

  *info = candidate;
  return Status::Ok();
}

}  // namespace

bool SupportsCpuDct(AcStrategyType strategy) noexcept {
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

Status ForwardDctCpu(
  AcStrategyType strategy,
  std::span<const float> pixels,
  std::span<float> coefficients) {

  const AcStrategyInfo* info = nullptr;
  Status status = ValidateDct(
    strategy,
    pixels.size(),
    coefficients.size(),
    &info);
  if (!status.ok()) {
    return status;
  }

  constexpr size_t kMaxCoefficientCount = 32 * 32;
  std::array<double, kMaxCoefficientCount> horizontal{};
  const Extent2D extent = info->pixel_extent();
  const std::span<const double> horizontal_basis =
    Basis(extent.width, BasisDirection::kForward);
  const std::span<const double> vertical_basis =
    Basis(extent.height, BasisDirection::kForward);

  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t u = 0; u < extent.width; ++u) {
      double sum = 0.0;
      for (size_t x = 0; x < extent.width; ++x) {
        sum += static_cast<double>(pixels[y * extent.width + x]) *
          horizontal_basis[u * extent.width + x];
      }
      horizontal[y * extent.width + u] = sum;
    }
  }

  for (size_t v = 0; v < extent.height; ++v) {
    for (size_t u = 0; u < extent.width; ++u) {
      double sum = 0.0;
      for (size_t y = 0; y < extent.height; ++y) {
        sum += vertical_basis[v * extent.height + y] *
          horizontal[y * extent.width + u];
      }
      coefficients[info->coefficient_index(v, u)] =
        static_cast<float>(sum);
    }
  }

  return Status::Ok();
}

Status InverseDctCpu(
  AcStrategyType strategy,
  std::span<const float> coefficients,
  std::span<float> pixels) {

  const AcStrategyInfo* info = nullptr;
  Status status = ValidateDct(
    strategy,
    coefficients.size(),
    pixels.size(),
    &info);
  if (!status.ok()) {
    return status;
  }

  constexpr size_t kMaxCoefficientCount = 32 * 32;
  std::array<double, kMaxCoefficientCount> horizontal{};
  const Extent2D extent = info->pixel_extent();
  const std::span<const double> horizontal_basis =
    Basis(extent.width, BasisDirection::kInverse);
  const std::span<const double> vertical_basis =
    Basis(extent.height, BasisDirection::kInverse);

  for (size_t v = 0; v < extent.height; ++v) {
    for (size_t x = 0; x < extent.width; ++x) {
      double sum = 0.0;
      for (size_t u = 0; u < extent.width; ++u) {
        sum += static_cast<double>(
          coefficients[info->coefficient_index(v, u)]) *
          horizontal_basis[u * extent.width + x];
      }
      horizontal[v * extent.width + x] = sum;
    }
  }

  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      double sum = 0.0;
      for (size_t v = 0; v < extent.height; ++v) {
        sum += vertical_basis[v * extent.height + y] *
          horizontal[v * extent.width + x];
      }
      pixels[y * extent.width + x] = static_cast<float>(sum);
    }
  }

  return Status::Ok();
}

}  // namespace gjxl
