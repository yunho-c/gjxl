// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Compares complete CPU and staged-GPU AC-strategy searches.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "codec/ac_strategy.h"
#include "codec/chroma_from_luma.h"
#include "core/ac_strategy.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/ac_strategy_search.h"

#ifndef GJXL_METALLIB_PATH
#error "GJXL_METALLIB_PATH must point to the benchmark metallib"
#endif

namespace {

volatile size_t g_checksum = 0;

void Require(gjxl::Status status, std::string_view operation) {
  if (!status.ok()) {
    throw std::runtime_error(
      std::string(operation) + ": " + std::string(status.message()));
  }
}

struct SearchFixture {
  explicit SearchFixture(gjxl::Extent2D pixel_extent)
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
        const float luma = 0.16f + 0.0008f * fx + 0.0006f * fy +
                           0.024f * std::sin(0.09f * fx + 0.13f * fy) +
                           0.011f * std::cos(0.17f * fx - 0.04f * fy);
        opsin[0][y * pixel_stride + x] =
          0.33f * luma + 0.004f * std::sin(0.07f * fy);
        opsin[1][y * pixel_stride + x] = luma;
        opsin[2][y * pixel_stride + x] =
          1.15f * luma - 0.005f * std::cos(0.08f * fx);
        pixel_mask[y * pixel_stride + x] =
          45.0f + 0.18f * static_cast<float>((x + 5 * y) % 13);
      }
    }
    for (size_t y = 0; y < block_extent.height; ++y) {
      for (size_t x = 0; x < block_extent.width; ++x) {
        quant_field[y * block_stride + x] =
          0.43f + 0.012f * static_cast<float>((3 * x + 2 * y) % 11);
      }
    }
    Require(gjxl::ComputeInitialColorCorrelationMap(Opsin(), &color_map),
      "ComputeInitialColorCorrelationMap");
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
  gjxl::ColorCorrelationMap color_map;
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

void Consume(const gjxl::AcStrategyGrid& grid) {
  gjxl::AcStrategyCell first;
  gjxl::AcStrategyCell last;
  Require(grid.Get(0, 0, &first), "Read first strategy cell");
  Require(grid.Get(grid.extent().width - 1, grid.extent().height - 1, &last),
    "Read last strategy cell");
  g_checksum = g_checksum + static_cast<size_t>(first.strategy) +
               static_cast<size_t>(last.strategy) + grid.extent().width;
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
        return false;
      }
    }
  }
  return true;
}

template <class Function>
double TimeMilliseconds(Function&& function) {
  const auto start = std::chrono::steady_clock::now();
  function();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

struct Distribution {
  double median = 0.0;
  double p10 = 0.0;
  double p90 = 0.0;
};

Distribution Summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const auto Percentile = [&](double fraction) {
    const double position = fraction * static_cast<double>(samples.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, samples.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return samples[lower] * (1.0 - weight) + samples[upper] * weight;
  };
  return {
    .median = Percentile(0.5),
    .p10 = Percentile(0.1),
    .p90 = Percentile(0.9),
  };
}

void RunCase(
  gjxl::GpuBackend& gpu, gjxl::Extent2D pixel_extent, size_t sample_count) {
  SearchFixture fixture(pixel_extent);
  const auto CpuSearch = [&] {
    gjxl::AcStrategyGrid grid;
    Require(gjxl::FindAcStrategyGrid(fixture.Opsin(),
              fixture.QuantField(),
              fixture.PixelMask(),
              fixture.color_map,
              {.butteraugli_target = 1.2f},
              &grid),
      "CPU AC-strategy search");
    Consume(grid);
  };
  const auto GpuSearch = [&] {
    gjxl::AcStrategyGrid grid;
    Require(gjxl::FindAcStrategyGridGpu(gpu,
              fixture.Opsin(),
              fixture.QuantField(),
              fixture.PixelMask(),
              fixture.color_map,
              {.butteraugli_target = 1.2f},
              &grid),
      "GPU AC-strategy search");
    Consume(grid);
  };

  gjxl::AcStrategyGrid cpu_grid;
  gjxl::AcStrategyGrid gpu_grid;
  gjxl::AcStrategyGpuSearchStats stats;
  Require(gjxl::FindAcStrategyGrid(fixture.Opsin(),
            fixture.QuantField(),
            fixture.PixelMask(),
            fixture.color_map,
            {.butteraugli_target = 1.2f},
            &cpu_grid),
    "CPU validation search");
  Require(gjxl::FindAcStrategyGridGpu(gpu,
            fixture.Opsin(),
            fixture.QuantField(),
            fixture.PixelMask(),
            fixture.color_map,
            {.butteraugli_target = 1.2f},
            &gpu_grid,
            &stats),
    "GPU validation search");
  if (!GridsEqual(cpu_grid, gpu_grid)) {
    throw std::runtime_error("CPU and GPU strategy grids differ");
  }

  CpuSearch();
  GpuSearch();
  std::vector<double> cpu_samples;
  std::vector<double> gpu_samples;
  cpu_samples.reserve(sample_count);
  gpu_samples.reserve(sample_count);
  for (size_t sample = 0; sample < sample_count; ++sample) {
    if (sample % 2 == 0) {
      cpu_samples.push_back(TimeMilliseconds(CpuSearch));
      gpu_samples.push_back(TimeMilliseconds(GpuSearch));
    } else {
      gpu_samples.push_back(TimeMilliseconds(GpuSearch));
      cpu_samples.push_back(TimeMilliseconds(CpuSearch));
    }
  }

  const Distribution cpu = Summarize(cpu_samples);
  const Distribution gpu_distribution = Summarize(gpu_samples);
  std::vector<double> speedups(sample_count);
  for (size_t i = 0; i < sample_count; ++i) {
    speedups[i] = cpu_samples[i] / gpu_samples[i];
  }
  const Distribution speedup = Summarize(speedups);
  std::cout << std::setw(4) << pixel_extent.width << 'x' << std::left
            << std::setw(5) << pixel_extent.height << std::right
            << std::setw(11) << stats.total_candidate_count << std::setw(13)
            << std::fixed << std::setprecision(3) << cpu.median << " ["
            << cpu.p10 << ", " << cpu.p90 << ']' << std::setw(13)
            << gpu_distribution.median << " [" << gpu_distribution.p10 << ", "
            << gpu_distribution.p90 << ']' << std::setw(11) << speedup.median
            << "x\n";
}

size_t ParseSampleCount(int argc, char** argv) {
  if (argc <= 1) {
    return 12;
  }
  const std::string value(argv[1]);
  size_t consumed = 0;
  const unsigned long parsed = std::stoul(value, &consumed);
  if (consumed != value.size() || parsed < 3) {
    throw std::invalid_argument("sample count must be at least 3");
  }
  return static_cast<size_t>(parsed);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const size_t sample_count = ParseSampleCount(argc, argv);
    std::unique_ptr<gjxl::GpuBackend> gpu;
    Require(
      gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, SimdgroupOptions(), &gpu),
      "CreateMetalBackend");

    std::cout << "Complete AC-strategy search (ms, median [p10, p90], "
                 "alternating order)\n"
              << "Pixels  Candidates   CPU                 GPU E2E             "
                 "Speedup\n";
    RunCase(*gpu, {64, 64}, sample_count);
    RunCase(*gpu, {128, 96}, sample_count);
    RunCase(*gpu, {256, 192}, sample_count);
    RunCase(*gpu, {512, 384}, sample_count);
    std::cout << "checksum: " << g_checksum << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
