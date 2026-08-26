// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "codec/ac_strategy.h"

namespace {

constexpr gjxl::Extent2D kPixelExtent{32, 32};
constexpr gjxl::Extent2D kBlockExtent{4, 4};

struct Fixture {
  std::array<std::vector<float>, 3> plane;
  std::vector<float> quant_field = std::vector<float>(16);
  std::vector<float> pixel_mask = std::vector<float>(32 * 32);

  Fixture() {
    for (size_t channel = 0; channel < plane.size(); ++channel) {
      plane[channel].resize(32 * 32);
      for (size_t y = 0; y < 32; ++y) {
        for (size_t x = 0; x < 32; ++x) {
          const float base =
            0.11f +
            0.017f * std::sin(static_cast<float>(x + 1) * 0.21f) +
            0.013f * std::cos(static_cast<float>(y + 2) * 0.29f);
          plane[channel][y * 32 + x] = base +
            (channel == 0
              ? 0.012f * std::sin(
                  static_cast<float>(x + y + 1) * 0.17f)
              : channel == 1
                ? 0.002f * static_cast<float>((x * 5 + y * 3) % 11)
                : 0.019f * std::cos(
                    static_cast<float>(x * 2 + y + 3) * 0.13f));
        }
      }
    }

    for (size_t i = 0; i < quant_field.size(); ++i) {
      quant_field[i] = 0.35f + 0.017f * static_cast<float>(i);
    }
    for (size_t y = 0; y < 32; ++y) {
      for (size_t x = 0; x < 32; ++x) {
        pixel_mask[y * 32 + x] =
          42.0f +
          0.3f * static_cast<float>((x * 7 + y * 11) % 19);
      }
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView Image() const {
    return {{
      gjxl::ConstPlaneF32View{
        .data = plane[0].data(),
        .extent = kPixelExtent,
        .stride = 32,
      },
      gjxl::ConstPlaneF32View{
        .data = plane[1].data(),
        .extent = kPixelExtent,
        .stride = 32,
      },
      gjxl::ConstPlaneF32View{
        .data = plane[2].data(),
        .extent = kPixelExtent,
        .stride = 32,
      },
    }};
  }

  [[nodiscard]] gjxl::ConstPlaneF32View QuantField() const {
    return {
      .data = quant_field.data(),
      .extent = kBlockExtent,
      .stride = 4,
    };
  }

  [[nodiscard]] gjxl::ConstPlaneF32View PixelMask() const {
    return {
      .data = pixel_mask.data(),
      .extent = kPixelExtent,
      .stride = 32,
    };
  }
};

bool CheckDeterministicCosts() {
  struct Golden {
    gjxl::AcStrategyType strategy;
    float cost;
  };
  constexpr std::array kGoldens = {
    Golden{gjxl::AcStrategyType::kDct8, 466.2448730f},
    Golden{gjxl::AcStrategyType::kDct16x8, 672.1040039f},
    Golden{gjxl::AcStrategyType::kDct8x16, 715.2138672f},
    Golden{gjxl::AcStrategyType::kDct16x16, 1153.744751f},
    Golden{gjxl::AcStrategyType::kDct32x16, 1882.065186f},
    Golden{gjxl::AcStrategyType::kDct16x32, 1866.822510f},
    Golden{gjxl::AcStrategyType::kDct32x32, 3253.645508f},
  };

  const Fixture fixture;
  for (const Golden& golden : kGoldens) {
    float cost = 0.0f;
    const gjxl::Status status = gjxl::EstimateAcStrategyCost(
      golden.strategy,
      0,
      0,
      fixture.Image(),
      fixture.QuantField(),
      fixture.PixelMask(),
      {
        .butteraugli_target = 1.3f,
        .entropy_multiplier = 1.0f,
        .cfl_factors = {0.18f, 0.0f, -0.11f},
      },
      &cost);
    if (!status.ok() || std::abs(cost - golden.cost) > 3.0e-3f) {
      std::cerr << "AC-strategy candidate cost changed unexpectedly\n";
      return false;
    }
  }

  return true;
}

bool CheckInputsAffectCost() {
  const Fixture fixture;
  float base_cost = 0.0f;
  float cfl_cost = 0.0f;
  float target_cost = 0.0f;
  if (!gjxl::EstimateAcStrategyCost(
        gjxl::AcStrategyType::kDct16x16,
        1,
        1,
        fixture.Image(),
        fixture.QuantField(),
        fixture.PixelMask(),
        {},
        &base_cost).ok() ||
      !gjxl::EstimateAcStrategyCost(
        gjxl::AcStrategyType::kDct16x16,
        1,
        1,
        fixture.Image(),
        fixture.QuantField(),
        fixture.PixelMask(),
        {.cfl_factors = {0.25f, 0.0f, -0.15f}},
        &cfl_cost).ok() ||
      !gjxl::EstimateAcStrategyCost(
        gjxl::AcStrategyType::kDct16x16,
        1,
        1,
        fixture.Image(),
        fixture.QuantField(),
        fixture.PixelMask(),
        {.butteraugli_target = 2.0f},
        &target_cost).ok() ||
      base_cost == cfl_cost ||
      base_cost == target_cost) {
    std::cerr << "AC-strategy cost ignored a required input\n";
    return false;
  }

  return true;
}

bool CheckQuantNorm() {
  Fixture fixture;
  float quant_norm = -777.0f;
  if (!gjxl::ComputeAcStrategyQuantNorm(
        gjxl::AcStrategyType::kDct16x8,
        2,
        1,
        fixture.QuantField(),
        &quant_norm).ok() ||
      quant_norm != std::max(
        fixture.quant_field[1 * 4 + 2],
        fixture.quant_field[2 * 4 + 2])) {
    std::cerr << "AC-strategy quant norm aggregation is incorrect\n";
    return false;
  }

  quant_norm = -777.0f;
  fixture.quant_field[2 * 4 + 2] =
    std::numeric_limits<float>::quiet_NaN();
  if (gjxl::ComputeAcStrategyQuantNorm(
        gjxl::AcStrategyType::kDct16x8,
        2,
        1,
        fixture.QuantField(),
        &quant_norm).ok() ||
      gjxl::ComputeAcStrategyQuantNorm(
        gjxl::AcStrategyType::kDct32x32,
        1,
        1,
        fixture.QuantField(),
        &quant_norm).ok() ||
      gjxl::ComputeAcStrategyQuantNorm(
        gjxl::AcStrategyType::kDct8,
        0,
        0,
        fixture.QuantField(),
        nullptr).ok() ||
      quant_norm != -777.0f) {
    std::cerr << "Invalid AC-strategy quant norm request was accepted\n";
    return false;
  }

  return true;
}

bool CheckInvalidInputs() {
  Fixture fixture;
  float cost = -777.0f;
  if (gjxl::EstimateAcStrategyCost(
        gjxl::AcStrategyType::kDct32x32,
        1,
        0,
        fixture.Image(),
        fixture.QuantField(),
        fixture.PixelMask(),
        {},
        &cost).ok() ||
      gjxl::EstimateAcStrategyCost(
        gjxl::AcStrategyType::kAfv0,
        0,
        0,
        fixture.Image(),
        fixture.QuantField(),
        fixture.PixelMask(),
        {},
        &cost).ok() ||
      gjxl::EstimateAcStrategyCost(
        gjxl::AcStrategyType::kDct8,
        0,
        0,
        fixture.Image(),
        fixture.QuantField(),
        fixture.PixelMask(),
        {.cfl_factors = {0.0f, 1.0f, 0.0f}},
        &cost).ok() ||
      gjxl::EstimateAcStrategyCost(
        gjxl::AcStrategyType::kDct8,
        0,
        0,
        fixture.Image(),
        fixture.QuantField(),
        fixture.PixelMask(),
        {},
        nullptr).ok() ||
      cost != -777.0f) {
    std::cerr << "Invalid AC-strategy cost request was accepted\n";
    return false;
  }

  fixture.quant_field[0] = std::numeric_limits<float>::quiet_NaN();
  if (gjxl::EstimateAcStrategyCost(
        gjxl::AcStrategyType::kDct8,
        0,
        0,
        fixture.Image(),
        fixture.QuantField(),
        fixture.PixelMask(),
        {},
        &cost).ok()) {
    std::cerr << "Non-finite AC-strategy field was accepted\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!CheckDeterministicCosts() ||
      !CheckInputsAffectCost() ||
      !CheckQuantNorm() ||
      !CheckInvalidInputs()) {
    return EXIT_FAILURE;
  }

  std::cout << "All AC-strategy cost tests passed.\n";
  return EXIT_SUCCESS;
}
