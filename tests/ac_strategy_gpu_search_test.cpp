// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

/// @file
/// Validates staged GPU candidate evaluation inside the CPU AC search.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

#include "codec/ac_strategy.h"
#include "codec/chroma_from_luma.h"
#include "core/ac_strategy.h"
#include "core/image.h"
#ifdef GJXL_TEST_CUDA
#include "gpu/cuda/cuda_backend.h"
#else
#include "gpu/metal/metal_backend.h"
#endif
#include "gpu/ops/ac_strategy_search.h"

#if !defined(GJXL_TEST_CUDA) && !defined(GJXL_METALLIB_PATH)
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

#ifndef GJXL_TEST_CUDA
gjxl::MetalBackendOptions OptionsFor(
  gjxl::MetalDctImplementation implementation) {

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

#endif

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

bool CheckPreparedResidentReuse(gjxl::GpuBackend& gpu,
  gjxl::Extent2D extent = {128, 96}) {
  const SearchFixture fixture(extent, 0.35f);
  size_t pixel_count = 0;
  size_t block_count = 0;
  if (!fixture.pixel_extent.try_area(&pixel_count) ||
      !fixture.block_extent.try_area(&block_count)) {
    return false;
  }
  std::array<std::vector<float>, 3> packed_opsin;
  for (size_t channel = 0; channel < 3; ++channel) {
    packed_opsin[channel].resize(pixel_count);
    for (size_t y = 0; y < fixture.pixel_extent.height; ++y) {
      std::copy_n(
        fixture.Opsin().plane[channel].Row(y), fixture.pixel_extent.width,
        packed_opsin[channel].data() + y * fixture.pixel_extent.width);
    }
  }
  std::vector<float> packed_quant(block_count);
  for (size_t y = 0; y < fixture.block_extent.height; ++y) {
    std::copy_n(
      fixture.QuantField().Row(y), fixture.block_extent.width,
      packed_quant.data() + y * fixture.block_extent.width);
  }
  std::vector<float> packed_mask(pixel_count);
  for (size_t y = 0; y < fixture.pixel_extent.height; ++y) {
    std::copy_n(
      fixture.PixelMask().Row(y), fixture.pixel_extent.width,
      packed_mask.data() + y * fixture.pixel_extent.width);
  }

  std::array<std::unique_ptr<gjxl::DeviceBuffer>, 3> device_opsin;
  std::unique_ptr<gjxl::DeviceBuffer> device_quant;
  std::unique_ptr<gjxl::DeviceBuffer> device_mask;
  for (size_t channel = 0; channel < 3; ++channel) {
    if (!gpu.Allocate(pixel_count * sizeof(float), &device_opsin[channel])
          .ok() ||
        !gpu.CopyHostToDevice(
          *device_opsin[channel], packed_opsin[channel].data(),
          pixel_count * sizeof(float)).ok()) {
      return false;
    }
  }
  if (!gpu.Allocate(block_count * sizeof(float), &device_quant).ok() ||
      !gpu.CopyHostToDevice(
        *device_quant, packed_quant.data(),
        block_count * sizeof(float)).ok() ||
      !gpu.Allocate(pixel_count * sizeof(float), &device_mask).ok() ||
      !gpu.CopyHostToDevice(
        *device_mask, packed_mask.data(),
        pixel_count * sizeof(float)).ok()) {
    return false;
  }
  const gjxl::ResidentAcStrategySearchInputs resident{
    .opsin = {{{
      {device_opsin[0].get(), 0, gjxl::DeviceElementType::kF32,
       fixture.pixel_extent, fixture.pixel_extent.width},
      {device_opsin[1].get(), 0, gjxl::DeviceElementType::kF32,
       fixture.pixel_extent, fixture.pixel_extent.width},
      {device_opsin[2].get(), 0, gjxl::DeviceElementType::kF32,
       fixture.pixel_extent, fixture.pixel_extent.width},
    }}},
    .quant_field = {
      device_quant.get(), 0, gjxl::DeviceElementType::kF32,
      fixture.block_extent, fixture.block_extent.width},
    .pixel_mask = {
      device_mask.get(), 0, gjxl::DeviceElementType::kF32,
      fixture.pixel_extent, fixture.pixel_extent.width},
  };
  gjxl::ColorCorrelationMap color_map;
  if (!gjxl::ComputeInitialColorCorrelationMap(fixture.Opsin(), &color_map)
        .ok()) {
    return false;
  }
  gjxl::PreparedAcStrategySearch prepared;
  gjxl::AcStrategyGrid first;
  gjxl::AcStrategyGrid second;
  gjxl::AcStrategyGpuSearchStats stats;
  if (!gjxl::FindAcStrategyGridGpuResident(
        gpu, fixture.Opsin(), fixture.QuantField(), fixture.PixelMask(),
        color_map, resident, {.butteraugli_target = 1.2f}, &first, &stats,
        &prepared).ok()) {
    return false;
  }
  const gjxl::GpuBackendStats after_first = gpu.stats();
  // Independently reconstruct the exact owning arena layout from the staged
  // counts. This catches a planner that queries compact sizes but allocates
  // or separates the ranges using the obsolete full-buffer formula.
  size_t expected_capacity = 0;
  gjxl::AcStrategyScratchRequirements expected_scratch;
  size_t conservative_packed = 0;
  const auto Append = [&](size_t bytes) {
    expected_capacity = (expected_capacity + 255) / 256 * 256 + bytes;
  };
  for (size_t i = 0; i < stats.candidate_counts.size(); ++i) {
    const size_t count = stats.candidate_counts[i];
    if (count == 0) continue;
    const auto strategy = static_cast<gjxl::AcStrategyType>(i);
    const size_t coefficients = gjxl::GetAcStrategyInfo(strategy)->coefficient_count();
    Append(count * sizeof(gjxl::AcStrategyCandidate));
    Append(6 * coefficients * sizeof(float));
    Append(count * sizeof(float));
    gjxl::AcStrategyScratchRequirements scratch;
    if (!gjxl::GetAcStrategyScratchRequirements(gpu, strategy, count,
          &scratch).ok()) return false;
    expected_scratch.scratch_a_bytes =
      std::max(expected_scratch.scratch_a_bytes, scratch.scratch_a_bytes);
    expected_scratch.scratch_b_bytes =
      std::max(expected_scratch.scratch_b_bytes, scratch.scratch_b_bytes);
    expected_scratch.rate_scratch_bytes =
      std::max(expected_scratch.rate_scratch_bytes, scratch.rate_scratch_bytes);
    conservative_packed = std::max(conservative_packed,
      count * 3 * coefficients * sizeof(float));
  }
  Append(expected_scratch.scratch_a_bytes);
  Append(expected_scratch.scratch_b_bytes);
  Append(expected_scratch.rate_scratch_bytes);
  if (stats.resource_capacity_bytes != expected_capacity ||
      stats.scratch.scratch_a_bytes != expected_scratch.scratch_a_bytes ||
      stats.scratch.scratch_b_bytes != expected_scratch.scratch_b_bytes ||
      stats.scratch.rate_scratch_bytes != expected_scratch.rate_scratch_bytes) {
    std::cerr << "Prepared AC search allocated the wrong scratch layout\n";
    return false;
  }
  const size_t first_capacity = stats.resource_capacity_bytes;
  if (!gjxl::FindAcStrategyGridGpuResident(
        gpu, fixture.Opsin(), fixture.QuantField(), fixture.PixelMask(),
        color_map, resident, {.butteraugli_target = 0.9f}, &second, &stats,
        &prepared).ok()) {
    return false;
  }
  const gjxl::GpuBackendStats after_second = gpu.stats();
  if (after_second.successful_allocations !=
        after_first.successful_allocations ||
      after_second.committed_submissions !=
        after_first.committed_submissions + 1 ||
      stats.resource_capacity_bytes != first_capacity ||
      !first.complete() || !second.complete()) {
    std::cerr << "Prepared resident AC search did not reuse allocations\n";
    return false;
  }
  std::cout << "Prepared AC search " << extent.width << 'x' << extent.height
            << " arena_bytes=" << stats.resource_capacity_bytes
            << " scratch_a_bytes=" << stats.scratch.scratch_a_bytes
            << " scratch_b_bytes=" << stats.scratch.scratch_b_bytes
            << " rate_bytes=" << stats.scratch.rate_scratch_bytes
            << " conservative_a_bytes=" << conservative_packed << '\n';
  // Shrink the logical views without changing their owning input buffers or
  // row strides, then restore the original layout. The scratch arena must
  // retain capacity, reset all offsets, and require no new allocation.
  auto small_resident = resident;
  for (auto& plane : small_resident.opsin.plane) plane.extent = {64, 64};
  small_resident.pixel_mask.extent = {64, 64};
  small_resident.quant_field.extent = {8, 8};
  auto small_opsin = fixture.Opsin();
  for (auto& plane : small_opsin.plane) plane.extent = {64, 64};
  auto small_quant = fixture.QuantField();
  small_quant.extent = {8, 8};
  auto small_mask = fixture.PixelMask();
  small_mask.extent = {64, 64};
  gjxl::ColorCorrelationMap small_color;
  gjxl::AcStrategyGrid small_grid;
  gjxl::AcStrategyGrid restored;
  if (!gjxl::ComputeInitialColorCorrelationMap(small_opsin, &small_color).ok() ||
      !gjxl::FindAcStrategyGridGpuResident(gpu, small_opsin, small_quant,
        small_mask, small_color, small_resident, {.butteraugli_target = 1.2f},
        &small_grid, &stats, &prepared).ok() ||
      small_grid.extent() != gjxl::Extent2D{8, 8} || !small_grid.complete() ||
      stats.resource_capacity_bytes != first_capacity ||
      !gjxl::FindAcStrategyGridGpuResident(gpu, fixture.Opsin(),
        fixture.QuantField(), fixture.PixelMask(), color_map, resident,
        {.butteraugli_target = 1.2f}, &restored, &stats, &prepared).ok() ||
      stats.resource_capacity_bytes != first_capacity ||
      gpu.stats().successful_allocations != after_first.successful_allocations ||
      !GridsEqual(first, restored)) {
    std::cerr << "Prepared AC search failed shrink/restore reuse\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::unique_ptr<gjxl::GpuBackend> gpu;
#ifdef GJXL_TEST_CUDA
  const gjxl::Status create_status = gjxl::CreateCudaBackend(&gpu);
  if (!create_status.ok()) {
    std::cerr << "CUDA backend unavailable: " << create_status.message() << '\n';
    return 77;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--memory-4k") {
    return CheckPreparedResidentReuse(*gpu, {3840, 2160})
      ? EXIT_SUCCESS : EXIT_FAILURE;
  }
#else
  (void)argc;
  (void)argv;
  const gjxl::Status create_status =
    gjxl::CreateMetalBackend(
      GJXL_METALLIB_PATH,
      OptionsFor(gjxl::MetalDctImplementation::kSimdgroupMatmul),
      &gpu);
  if (!create_status.ok()) {
    std::cerr << "Unable to create Metal backend: " << create_status.message()
              << '\n';
    return EXIT_FAILURE;
  }
#endif
  if (!CheckSearchParity(*gpu, {8, 8}, 1.2f, 0.0f, false) ||
      !CheckSearchParity(*gpu, {32, 32}, 0.8f, 0.2f, false) ||
      !CheckSearchParity(*gpu, {64, 64}, 1.2f, 0.0f, true) ||
      !CheckSearchParity(*gpu, {80, 72}, 0.8f, 0.4f, false) ||
      !CheckSearchParity(*gpu, {80, 72}, 1.2f, 0.8f, false) ||
      !CheckSearchParity(*gpu, {80, 72}, 2.0f, 1.2f, false) ||
      !CheckSearchParity(*gpu, {128, 96}, 1.2f, 1.6f, false) ||
      !CheckValidationAndAtomicCommit(*gpu) ||
      !CheckPreparedResidentReuse(*gpu)) {
    return EXIT_FAILURE;
  }

#ifndef GJXL_TEST_CUDA
  std::unique_ptr<gjxl::GpuBackend> scalar_gpu;
  const gjxl::Status scalar_status =
    gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &scalar_gpu);
  if (!scalar_status.ok() ||
      !CheckSearchParity(*scalar_gpu, {64, 64}, 1.2f, 0.6f, true)) {
    std::cerr << "Scalar staged search failed: " << scalar_status.message()
              << '\n';
    return EXIT_FAILURE;
  }

  std::unique_ptr<gjxl::GpuBackend> factored_gpu;
  const gjxl::Status factored_status =
    gjxl::CreateMetalBackend(
      GJXL_METALLIB_PATH,
      OptionsFor(gjxl::MetalDctImplementation::kFactoredRadix2),
      &factored_gpu);
  if (!factored_status.ok() ||
      !CheckSearchParity(*factored_gpu, {64, 64}, 1.2f, 0.9f, true)) {
    std::cerr << "Factored staged search failed: "
              << factored_status.message() << '\n';
    return EXIT_FAILURE;
  }
#endif
  return EXIT_SUCCESS;
}
