// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates staged GPU candidate evaluation inside the CPU AC search.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "codec/ac_strategy.h"
#include "codec/chroma_from_luma.h"
#include "core/ac_strategy.h"
#include "core/image.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/ac_strategy_search.h"

#ifndef GJXL_METALLIB_PATH
#error "GJXL_METALLIB_PATH must point to the test metallib"
#endif

namespace {

struct SearchFixture {
  explicit SearchFixture(gjxl::Extent2D pixel_extent, float phase = 0.0f)
      : pixel_extent(pixel_extent),
        block_extent{
          pixel_extent.width / gjxl::kJxlBlockDimension,
          pixel_extent.height / gjxl::kJxlBlockDimension,
        },
        pixel_stride(pixel_extent.width + 5),
        block_stride(block_extent.width + 3),
        quant_field(block_stride * block_extent.height, -777.0f),
        pixel_mask(pixel_stride * pixel_extent.height, -777.0f) {
    for (std::vector<float>& plane : opsin) {
      plane.assign(pixel_stride * pixel_extent.height, -777.0f);
    }
    for (size_t y = 0; y < pixel_extent.height; ++y) {
      for (size_t x = 0; x < pixel_extent.width; ++x) {
        const float fx = static_cast<float>(x);
        const float fy = static_cast<float>(y);
        const float luma = 0.18f + 0.0011f * fx + 0.0009f * fy +
                           0.021f * std::sin(0.13f * fx + 0.07f * fy + phase) +
                           0.009f * std::cos(0.05f * fx - 0.17f * fy - phase);
        opsin[0][y * pixel_stride + x] =
          0.31f * luma + 0.003f * std::sin(0.19f * fy);
        opsin[1][y * pixel_stride + x] = luma;
        opsin[2][y * pixel_stride + x] =
          1.12f * luma - 0.004f * std::cos(0.11f * fx);
        pixel_mask[y * pixel_stride + x] =
          46.0f + 0.15f * static_cast<float>((x + 3 * y) % 11);
      }
    }
    for (size_t y = 0; y < block_extent.height; ++y) {
      for (size_t x = 0; x < block_extent.width; ++x) {
        quant_field[y * block_stride + x] =
          0.44f + 0.013f * static_cast<float>((2 * x + y) % 9);
      }
    }
  }

  [[nodiscard]] gjxl::ConstImage3FView Opsin() const {
    return {{
      gjxl::ConstPlaneF32View{opsin[0].data(), pixel_extent, pixel_stride},
      gjxl::ConstPlaneF32View{opsin[1].data(), pixel_extent, pixel_stride},
      gjxl::ConstPlaneF32View{opsin[2].data(), pixel_extent, pixel_stride},
    }};
  }

  [[nodiscard]] gjxl::ConstPlaneF32View QuantField() const {
    return {quant_field.data(), block_extent, block_stride};
  }

  [[nodiscard]] gjxl::ConstPlaneF32View PixelMask() const {
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

gjxl::MetalBackendOptions SimdgroupOptions() {
  constexpr auto implementation =
    gjxl::MetalDctImplementation::kSimdgroupMatmul;
  return {
    .forward_dct8 = implementation,
    .inverse_dct8 = implementation,
    .forward_dct16x16 = implementation,
    .inverse_dct16x16 = implementation,
    .forward_dct32x32 = implementation,
    .inverse_dct32x32 = implementation,
    .forward_dct16x8 = implementation,
    .inverse_dct16x8 = implementation,
    .forward_dct8x16 = implementation,
    .inverse_dct8x16 = implementation,
    .forward_dct32x16 = implementation,
    .inverse_dct32x16 = implementation,
    .forward_dct16x32 = implementation,
    .inverse_dct16x32 = implementation,
  };
}

bool GridsEqual(
  const gjxl::AcStrategyGrid& expected, const gjxl::AcStrategyGrid& actual) {
  if (expected.extent() != actual.extent()) {
    return false;
  }
  for (size_t y = 0; y < expected.extent().height; ++y) {
    for (size_t x = 0; x < expected.extent().width; ++x) {
      gjxl::AcStrategyCell expected_cell;
      gjxl::AcStrategyCell actual_cell;
      if (!expected.Get(x, y, &expected_cell).ok() ||
          !actual.Get(x, y, &actual_cell).ok() ||
          expected_cell.strategy != actual_cell.strategy ||
          expected_cell.is_anchor != actual_cell.is_anchor) {
        std::cerr << "CPU/GPU search differs at block " << x << ',' << y
                  << '\n';
        return false;
      }
    }
  }
  return true;
}

bool CheckSearchParity(gjxl::GpuBackend& gpu,
  gjxl::Extent2D pixel_extent,
  float butteraugli_target,
  float phase,
  bool check_full_tile_counts) {
  const SearchFixture fixture(pixel_extent, phase);
  gjxl::ColorCorrelationMap color_map;
  if (!gjxl::ComputeInitialColorCorrelationMap(fixture.Opsin(), &color_map)
        .ok()) {
    return false;
  }
  gjxl::AcStrategyGrid cpu_grid;
  gjxl::AcStrategyGrid gpu_grid;
  const gjxl::Status cpu_status = gjxl::FindAcStrategyGrid(fixture.Opsin(),
    fixture.QuantField(),
    fixture.PixelMask(),
    color_map,
    {.butteraugli_target = butteraugli_target},
    &cpu_grid);
  gjxl::AcStrategyGpuSearchStats stats;
  const gjxl::Status gpu_status = gjxl::FindAcStrategyGridGpu(gpu,
    fixture.Opsin(),
    fixture.QuantField(),
    fixture.PixelMask(),
    color_map,
    {.butteraugli_target = butteraugli_target},
    &gpu_grid,
    &stats);
  if (!cpu_status.ok() || !gpu_status.ok()) {
    std::cerr << "AC search failed: CPU=" << cpu_status.message()
              << ", GPU=" << gpu_status.message() << '\n';
    return false;
  }
  if (!gpu_grid.complete() || !GridsEqual(cpu_grid, gpu_grid)) {
    return false;
  }

  if (check_full_tile_counts) {
    const auto Count = [&](gjxl::AcStrategyType strategy) {
      return stats.candidate_counts[static_cast<size_t>(strategy)];
    };
    if (Count(gjxl::AcStrategyType::kDct8) != 64 ||
        Count(gjxl::AcStrategyType::kDct16x8) != 56 ||
        Count(gjxl::AcStrategyType::kDct8x16) != 56 ||
        Count(gjxl::AcStrategyType::kDct16x16) != 49 ||
        Count(gjxl::AcStrategyType::kDct32x16) != 12 ||
        Count(gjxl::AcStrategyType::kDct16x32) != 12 ||
        Count(gjxl::AcStrategyType::kDct32x32) != 9 ||
        stats.total_candidate_count != 258) {
      std::cerr << "Staged candidate counts are not dependency-minimal\n";
      return false;
    }
  }
  return true;
}

bool CheckValidationAndAtomicCommit(gjxl::GpuBackend& gpu) {
  SearchFixture fixture({64, 64});
  gjxl::ColorCorrelationMap color_map;
  gjxl::AcStrategyGrid grid;
  if (!gjxl::ComputeInitialColorCorrelationMap(fixture.Opsin(), &color_map)
        .ok() ||
      !gjxl::FindAcStrategyGridGpu(gpu,
        fixture.Opsin(),
        fixture.QuantField(),
        fixture.PixelMask(),
        color_map,
        {.butteraugli_target = 1.2f},
        &grid)
        .ok()) {
    return false;
  }
  gjxl::AcStrategyCell before;
  if (!grid.Get(0, 0, &before).ok()) {
    return false;
  }
  gjxl::ConstPlaneF32View invalid_quant = fixture.QuantField();
  invalid_quant.extent.width -= 1;
  gjxl::AcStrategyGpuSearchStats stats;
  stats.total_candidate_count = 777;
  if (gjxl::FindAcStrategyGridGpu(gpu,
        fixture.Opsin(),
        invalid_quant,
        fixture.PixelMask(),
        color_map,
        {.butteraugli_target = 1.2f},
        &grid,
        &stats)
        .ok()) {
    std::cerr << "Invalid GPU search request was accepted\n";
    return false;
  }
  fixture.opsin[0][0] = std::numeric_limits<float>::quiet_NaN();
  if (gjxl::FindAcStrategyGridGpu(gpu,
        fixture.Opsin(),
        fixture.QuantField(),
        fixture.PixelMask(),
        color_map,
        {.butteraugli_target = 1.2f},
        &grid,
        &stats)
        .ok()) {
    std::cerr << "Non-finite GPU search input was accepted\n";
    return false;
  }
  gjxl::AcStrategyCell after;
  if (!grid.Get(0, 0, &after).ok() || before.strategy != after.strategy ||
      before.is_anchor != after.is_anchor ||
      stats.total_candidate_count != 777) {
    std::cerr << "Failed GPU search changed an output\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  std::unique_ptr<gjxl::GpuBackend> gpu;
  const gjxl::Status create_status =
    gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, SimdgroupOptions(), &gpu);
  if (!create_status.ok()) {
    std::cerr << "Unable to create Metal backend: " << create_status.message()
              << '\n';
    return EXIT_FAILURE;
  }
  if (!CheckSearchParity(*gpu, {8, 8}, 1.2f, 0.0f, false) ||
      !CheckSearchParity(*gpu, {32, 32}, 0.8f, 0.2f, false) ||
      !CheckSearchParity(*gpu, {64, 64}, 1.2f, 0.0f, true) ||
      !CheckSearchParity(*gpu, {80, 72}, 0.8f, 0.4f, false) ||
      !CheckSearchParity(*gpu, {80, 72}, 1.2f, 0.8f, false) ||
      !CheckSearchParity(*gpu, {80, 72}, 2.0f, 1.2f, false) ||
      !CheckSearchParity(*gpu, {128, 96}, 1.2f, 1.6f, false) ||
      !CheckValidationAndAtomicCommit(*gpu)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<gjxl::GpuBackend> scalar_gpu;
  const gjxl::Status scalar_status =
    gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &scalar_gpu);
  if (!scalar_status.ok() ||
      !CheckSearchParity(*scalar_gpu, {64, 64}, 1.2f, 0.6f, true)) {
    std::cerr << "Scalar staged search failed: " << scalar_status.message()
              << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
