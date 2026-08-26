// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates CPU AC-strategy selection and its placement invariants.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "codec/ac_strategy.h"
#include "codec/ac_strategy_search_policy.h"
#include "codec/chroma_from_luma.h"
#include "codec/dct.h"

namespace {

struct SearchFixture {
  explicit SearchFixture(gjxl::Extent2D pixel_extent)
    : pixel_extent(pixel_extent),
      block_extent{
        pixel_extent.width / gjxl::kJxlBlockDimension,
        pixel_extent.height / gjxl::kJxlBlockDimension,
      },
      pixel_stride(pixel_extent.width + 3),
      block_stride(block_extent.width + 2),
      quant_field(block_stride * block_extent.height, -777.0f),
      pixel_mask(pixel_stride * pixel_extent.height, -777.0f) {

    for (std::vector<float>& values : opsin) {
      values.assign(pixel_stride * pixel_extent.height, -777.0f);
    }
    for (size_t y = 0; y < pixel_extent.height; ++y) {
      for (size_t x = 0; x < pixel_extent.width; ++x) {
        const float luma =
          0.14f + 0.0013f * static_cast<float>(x) +
          0.0007f * static_cast<float>(y) +
          0.012f * std::sin(0.11f * static_cast<float>(x + y));
        opsin[1][y * pixel_stride + x] = luma;
        opsin[0][y * pixel_stride + x] = 0.35f * luma;
        opsin[2][y * pixel_stride + x] = 1.18f * luma;
        pixel_mask[y * pixel_stride + x] =
          48.0f + 0.1f * static_cast<float>((x + 3 * y) % 7);
      }
    }
    for (size_t y = 0; y < block_extent.height; ++y) {
      for (size_t x = 0; x < block_extent.width; ++x) {
        quant_field[y * block_stride + x] =
          0.48f + 0.01f * static_cast<float>(x + y);
      }
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView OpsinView() const {
    return {{
      gjxl::ConstPlaneF32View{
        opsin[0].data(), pixel_extent, pixel_stride},
      gjxl::ConstPlaneF32View{
        opsin[1].data(), pixel_extent, pixel_stride},
      gjxl::ConstPlaneF32View{
        opsin[2].data(), pixel_extent, pixel_stride},
    }};
  }

  [[nodiscard]] gjxl::ConstPlaneF32View QuantView() const {
    return {quant_field.data(), block_extent, block_stride};
  }

  [[nodiscard]] gjxl::ConstPlaneF32View PixelMaskView() const {
    return {pixel_mask.data(), pixel_extent, pixel_stride};
  }

  gjxl::Extent2D pixel_extent;
  gjxl::Extent2D block_extent;
  size_t pixel_stride;
  size_t block_stride;
  std::array<std::vector<float>, 3> opsin;
  std::vector<float> quant_field;
  std::vector<float> pixel_mask;
};

bool CheckPinnedSmoothMap() {
  const SearchFixture fixture({32, 32});
  gjxl::ColorCorrelationMap color_map;
  if (!gjxl::ComputeInitialColorCorrelationMap(
        fixture.OpsinView(), &color_map).ok()) {
    return false;
  }
  if (color_map.y_to_x_map().Row(0)[0] != 26 ||
      color_map.y_to_b_map().Row(0)[0] != 2) {
    std::cerr << "AC-search fixture CfL differs from pinned libjxl\n";
    return false;
  }

  struct CostGolden {
    gjxl::AcStrategyType strategy;
    float entropy_multiplier;
    float cost;
  };
  constexpr std::array kCostGoldens = {
    CostGolden{gjxl::AcStrategyType::kDct8, 1.0f, 240.882950f},
    CostGolden{gjxl::AcStrategyType::kDct16x8, 1.21f, 334.474823f},
    CostGolden{gjxl::AcStrategyType::kDct8x16, 1.21f, 326.605408f},
    CostGolden{gjxl::AcStrategyType::kDct16x16, 1.34f, 386.819519f},
    CostGolden{gjxl::AcStrategyType::kDct32x16, 1.49f, 558.955811f},
    CostGolden{gjxl::AcStrategyType::kDct16x32, 1.49f, 544.746704f},
    CostGolden{gjxl::AcStrategyType::kDct32x32, 1.48f, 706.478333f},
  };
  const std::array<float, 3> cfl_factors = color_map.AcFactors(0, 0);
  for (const CostGolden& golden : kCostGoldens) {
    float cost = 0.0f;
    const gjxl::Status cost_status = gjxl::EstimateAcStrategyCost(
      golden.strategy,
      0,
      0,
      fixture.OpsinView(),
      fixture.QuantView(),
      fixture.PixelMaskView(),
      {
        .butteraugli_target = 1.2f,
        .entropy_multiplier = golden.entropy_multiplier,
        .cfl_factors = cfl_factors,
      },
      &cost);
    if (!cost_status.ok() || std::abs(cost - golden.cost) > 2.0e-2f) {
      std::cerr
        << "Candidate cost differs from pinned libjxl for strategy "
        << static_cast<unsigned>(golden.strategy) << ": "
        << cost << " versus " << golden.cost << '\n';
      return false;
    }
  }

  gjxl::AcStrategyGrid grid;
  const gjxl::Status status = gjxl::FindAcStrategyGrid(
    fixture.OpsinView(),
    fixture.QuantView(),
    fixture.PixelMaskView(),
    color_map,
    {.butteraugli_target = 1.2f},
    &grid);
  if (!status.ok()) {
    std::cerr << "AC-strategy search failed: " << status.message() << '\n';
    return false;
  }

  for (size_t y = 0; y < 4; ++y) {
    for (size_t x = 0; x < 4; ++x) {
      gjxl::AcStrategyCell cell;
      if (!grid.Get(x, y, &cell).ok() ||
          cell.strategy != gjxl::AcStrategyType::kDct32x32 ||
          cell.is_anchor != (x == 0 && y == 0)) {
        std::cerr << "Smooth strategy map differs from pinned libjxl\n";
        return false;
      }
    }
  }
  return true;
}

bool CheckCompleteNonOverlappingGrid(gjxl::Extent2D pixel_extent) {
  const SearchFixture fixture(pixel_extent);
  gjxl::ColorCorrelationMap color_map;
  gjxl::AcStrategyGrid grid;
  if (!gjxl::ComputeInitialColorCorrelationMap(
        fixture.OpsinView(), &color_map).ok() ||
      !gjxl::FindAcStrategyGrid(
        fixture.OpsinView(),
        fixture.QuantView(),
        fixture.PixelMaskView(),
        color_map,
        {.butteraugli_target = 1.2f},
        &grid).ok() ||
      !grid.complete() ||
      grid.extent() != fixture.block_extent) {
    std::cerr << "Odd strategy grid search failed\n";
    return false;
  }

  size_t covered_count = 0;
  const gjxl::Status visit_status = grid.ForEachAnchor(
    [&](size_t x, size_t y, gjxl::AcStrategyType strategy) {
      const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(strategy);
      if (info == nullptr ||
          x + info->covered_blocks.width > fixture.block_extent.width ||
          y + info->covered_blocks.height > fixture.block_extent.height ||
          !gjxl::SupportsCpuDct(strategy)) {
        return gjxl::Status::Internal("Invalid selected strategy");
      }
      covered_count +=
        info->covered_blocks.width * info->covered_blocks.height;
      return gjxl::Status::Ok();
    });
  if (!visit_status.ok() ||
      covered_count != fixture.block_extent.width *
        fixture.block_extent.height) {
    std::cerr << "Strategy anchors do not cover the grid exactly once\n";
    return false;
  }

  // No transform may cross a 64x64 color-correlation boundary.
  bool boundary_ok = true;
  if (!grid.ForEachAnchor(
        [&](size_t x, size_t y, gjxl::AcStrategyType strategy) {
          const gjxl::Extent2D covered =
            gjxl::GetAcStrategyInfo(strategy)->covered_blocks;
          if (x / 8 != (x + covered.width - 1) / 8 ||
              y / 8 != (y + covered.height - 1) / 8) {
            boundary_ok = false;
          }
          return gjxl::Status::Ok();
        }).ok() ||
      !boundary_ok) {
    std::cerr << "Strategy crossed a color-tile boundary\n";
    return false;
  }
  return true;
}

bool CheckValidationAndAtomicCommit() {
  SearchFixture fixture({32, 32});
  gjxl::ColorCorrelationMap color_map;
  gjxl::AcStrategyGrid grid;
  if (!gjxl::ComputeInitialColorCorrelationMap(
        fixture.OpsinView(), &color_map).ok() ||
      !gjxl::FindAcStrategyGrid(
        fixture.OpsinView(), fixture.QuantView(), fixture.PixelMaskView(),
        color_map, {.butteraugli_target = 1.2f}, &grid).ok()) {
    return false;
  }

  gjxl::AcStrategyCell original;
  if (!grid.Get(0, 0, &original).ok()) {
    return false;
  }
  gjxl::ConstPlaneF32View bad_quant = fixture.QuantView();
  bad_quant.extent.width -= 1;
  if (gjxl::FindAcStrategyGrid(
        fixture.OpsinView(), bad_quant, fixture.PixelMaskView(), color_map,
        {.butteraugli_target = 1.2f}, &grid).ok() ||
      gjxl::FindAcStrategyGrid(
        fixture.OpsinView(), fixture.QuantView(), fixture.PixelMaskView(),
        color_map, {.butteraugli_target = 0.0f}, &grid).ok() ||
      gjxl::FindAcStrategyGrid(
        fixture.OpsinView(), fixture.QuantView(), fixture.PixelMaskView(),
        color_map, {.butteraugli_target = 1.2f}, nullptr).ok()) {
    std::cerr << "Invalid AC-strategy search request was accepted\n";
    return false;
  }
  gjxl::AcStrategyCell after_failure;
  if (!grid.Get(0, 0, &after_failure).ok() ||
      after_failure.strategy != original.strategy ||
      after_failure.is_anchor != original.is_anchor) {
    std::cerr << "Failed AC search changed the prior output\n";
    return false;
  }
  return true;
}

bool CheckTiePolicy() {
  using gjxl::ac_strategy_internal::ChooseFirstLevelDivision;
  using gjxl::ac_strategy_internal::FirstLevelDivision;
  if (ChooseFirstLevelDivision(1.0f, 2.0f, 3.0f) !=
        FirstLevelDivision::kSquare ||
      ChooseFirstLevelDivision(1.0f, 1.0f, 2.0f) !=
        FirstLevelDivision::kVertical ||
      ChooseFirstLevelDivision(1.0f, 1.0f, 1.0f) !=
        FirstLevelDivision::kHorizontal ||
      ChooseFirstLevelDivision(2.0f, 1.0f, 1.0f) !=
        FirstLevelDivision::kHorizontal) {
    std::cerr << "AC-strategy tie policy differs from libjxl\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckPinnedSmoothMap() ||
      !CheckCompleteNonOverlappingGrid({40, 24}) ||
      !CheckCompleteNonOverlappingGrid({136, 72}) ||
      !CheckValidationAndAtomicCommit() ||
      !CheckTiePolicy()) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
