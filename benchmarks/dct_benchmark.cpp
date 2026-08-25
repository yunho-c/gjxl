// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "core/block_grid.h"
#include "gpu/backend.h"
#include "gpu/metal/metal_backend.h"

namespace {

constexpr size_t kWarmupIterations = 20;

struct ImplementationCase {
  gjxl::MetalDctImplementation implementation;
  std::string_view name;
};

constexpr std::array<ImplementationCase, 2>
kImplementations{{
  {
    .implementation =
      gjxl::MetalDctImplementation::kScalarMatmul,
    .name = "scalar matmul",
  },
  {
    .implementation =
      gjxl::MetalDctImplementation::kSimdgroupMatmul,
    .name = "simdgroup matmul",
  },
}};

void Require(
  const gjxl::Status& status,
  std::string_view operation) {

  if (status.ok()) {
    return;
  }

  std::cerr
    << operation
    << " failed: "
    << status.message()
    << '\n';

  std::exit(EXIT_FAILURE);
}

size_t ComparableTransformCount(
  gjxl::AcStrategyType strategy,
  size_t dct8_transform_count) {

  const gjxl::AcStrategyInfo* strategy_info =
    gjxl::GetAcStrategyInfo(strategy);

  if (strategy_info == nullptr) {
    std::cerr << "ComparableTransformCount received invalid strategy\n";
    std::exit(EXIT_FAILURE);
  }

  const size_t coefficient_count =
    strategy_info->coefficient_count();

  if (coefficient_count == 0 ||
      coefficient_count % gjxl::kJxlBlockArea != 0) {

    std::cerr
      << strategy_info->name
      << " does not cover a whole number of JPEG XL base blocks\n";
    std::exit(EXIT_FAILURE);
  }

  const size_t base_blocks_per_transform =
    coefficient_count / gjxl::kJxlBlockArea;

  return std::max<size_t>(
    1,
    dct8_transform_count / base_blocks_per_transform);
}

template <typename Submit>
void BenchmarkDct(
  gjxl::GpuBackend& gpu,
  std::string_view operation,
  const Submit& submit,
  const gjxl::TransformBatch& batch,
  size_t iterations) {

  const gjxl::AcStrategyInfo* strategy_info =
    gjxl::GetAcStrategyInfo(batch.strategy);

  if (strategy_info == nullptr) {
    std::cerr << "BenchmarkDct received invalid strategy\n";
    std::exit(EXIT_FAILURE);
  }

  const std::string warmup_operation =
    std::string("Warmup ") + std::string(operation);
  const std::string timed_operation =
    std::string("Timed ") + std::string(operation);

  for (size_t i = 0; i < kWarmupIterations; ++i) {
    Require(
      submit(batch),
      warmup_operation);
  }

  Require(
    gpu.Synchronize(),
    "Warmup synchronization");

  const auto begin = std::chrono::steady_clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    Require(
      submit(batch),
      timed_operation);
  }

  Require(
    gpu.Synchronize(),
    "Timed synchronization");

  const auto end = std::chrono::steady_clock::now();

  const double seconds =
    std::chrono::duration<double>(
      end - begin)
      .count();

  const double transforms =
    static_cast<double>(batch.transform_count) *
    static_cast<double>(iterations);

  const double pixels =
    transforms *
    static_cast<double>(strategy_info->coefficient_count());

  const gjxl::Extent2D extent = strategy_info->pixel_extent();

  // A dense separable transform evaluates one horizontal and one vertical
  // dot product per output element.
  const double multiply_accumulates =
    pixels *
    static_cast<double>(extent.width + extent.height);

  std::cout
    << "\nOperation:        " << operation << '\n'
    << "Transforms/launch: " << batch.transform_count << '\n'
    << "Iterations:       " << iterations << '\n'
    << "Elapsed:          " << seconds << '\n'
    << "Transforms/s:     " << transforms / seconds << '\n'
    << "MPixels/s:        " << pixels / seconds / 1.0e6 << '\n'
    << "Dense GMAC/s:     "
    << multiply_accumulates / seconds / 1.0e9
    << '\n';
}

void BenchmarkDctPair(
  gjxl::GpuBackend& gpu,
  std::string_view implementation_name,
  gjxl::AcStrategyType strategy,
  size_t transform_count,
  size_t iterations) {

  const gjxl::AcStrategyInfo* strategy_info =
    gjxl::GetAcStrategyInfo(strategy);

  if (strategy_info == nullptr) {
    std::cerr << "BenchmarkDctPair received invalid strategy\n";
    std::exit(EXIT_FAILURE);
  }

  const size_t element_count =
    transform_count * strategy_info->coefficient_count();
  const size_t bytes = element_count * sizeof(float);
  std::vector<float> input(element_count);

  std::mt19937 rng(
    12345u ^
    static_cast<unsigned int>(strategy_info->pixel_extent().width << 8) ^
    static_cast<unsigned int>(strategy_info->pixel_extent().height));
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  for (float& value : input) {
    value = distribution(rng);
  }

  std::unique_ptr<gjxl::DeviceBuffer> device_input;
  std::unique_ptr<gjxl::DeviceBuffer> device_output;

  Require(
    gpu.Allocate(bytes, &device_input),
    "Allocate input");

  Require(
    gpu.Allocate(bytes, &device_output),
    "Allocate output");

  Require(
    gpu.CopyHostToDevice(
      *device_input,
      input.data(),
      bytes),
    "Upload input");

  const gjxl::TransformBatch batch{
    .strategy = strategy,
    .input = device_input.get(),
    .output = device_output.get(),
    .transform_count = transform_count,
  };

  BenchmarkDct(
    gpu,
    std::string(implementation_name) + " Forward" +
      std::string(strategy_info->name),
    [&gpu](const gjxl::TransformBatch& transform_batch) {
      return gpu.ForwardTransform(transform_batch);
    },
    batch,
    iterations);

  BenchmarkDct(
    gpu,
    std::string(implementation_name) + " Inverse" +
      std::string(strategy_info->name),
    [&gpu](const gjxl::TransformBatch& transform_batch) {
      return gpu.InverseTransform(transform_batch);
    },
    batch,
    iterations);
}

}  // namespace

int main(int argc, char** argv) {
  size_t dct8_transform_count = 65536;
  size_t iterations = 200;

  if (argc >= 2) {
    dct8_transform_count =
      static_cast<size_t>(
        std::strtoull(argv[1], nullptr, 10));
  }

  if (argc >= 3) {
    iterations =
      static_cast<size_t>(
        std::strtoull(argv[2], nullptr, 10));
  }

  // Keep the number of transformed pixels per launch comparable to DCT8.
  const size_t dct16_transform_count =
    ComparableTransformCount(
      gjxl::AcStrategyType::kDct16x16,
      dct8_transform_count);
  const size_t dct16x8_transform_count =
    ComparableTransformCount(
      gjxl::AcStrategyType::kDct16x8,
      dct8_transform_count);
  const size_t dct8x16_transform_count =
    ComparableTransformCount(
      gjxl::AcStrategyType::kDct8x16,
      dct8_transform_count);
  const size_t dct32x16_transform_count =
    ComparableTransformCount(
      gjxl::AcStrategyType::kDct32x16,
      dct8_transform_count);
  const size_t dct16x32_transform_count =
    ComparableTransformCount(
      gjxl::AcStrategyType::kDct16x32,
      dct8_transform_count);
  const size_t dct32_transform_count =
    ComparableTransformCount(
      gjxl::AcStrategyType::kDct32x32,
      dct8_transform_count);

  for (const ImplementationCase& implementation :
       kImplementations) {

    const gjxl::MetalBackendOptions options{
      .forward_dct8 = implementation.implementation,
      .inverse_dct8 = implementation.implementation,
      .forward_dct16x16 = implementation.implementation,
      .inverse_dct16x16 = implementation.implementation,
      .forward_dct32x32 = implementation.implementation,
      .inverse_dct32x32 = implementation.implementation,
      .forward_dct16x8 = implementation.implementation,
      .inverse_dct16x8 = implementation.implementation,
      .forward_dct8x16 = implementation.implementation,
      .inverse_dct8x16 = implementation.implementation,
      .forward_dct32x16 = implementation.implementation,
      .inverse_dct32x16 = implementation.implementation,
      .forward_dct16x32 = implementation.implementation,
      .inverse_dct16x32 = implementation.implementation,
    };

    std::unique_ptr<gjxl::GpuBackend> gpu;

    Require(
      gjxl::CreateMetalBackend(
        GJXL_METALLIB_PATH,
        options,
        &gpu),
      std::string("CreateMetalBackend for ") +
        std::string(implementation.name));

    std::cout
      << "\nBackend: "
      << gpu->name()
      << " ["
      << implementation.name
      << "]\n";

    BenchmarkDctPair(
      *gpu,
      implementation.name,
      gjxl::AcStrategyType::kDct8,
      dct8_transform_count,
      iterations);

    BenchmarkDctPair(
      *gpu,
      implementation.name,
      gjxl::AcStrategyType::kDct16x16,
      dct16_transform_count,
      iterations);

    BenchmarkDctPair(
      *gpu,
      implementation.name,
      gjxl::AcStrategyType::kDct32x32,
      dct32_transform_count,
      iterations);

    BenchmarkDctPair(
      *gpu,
      implementation.name,
      gjxl::AcStrategyType::kDct16x8,
      dct16x8_transform_count,
      iterations);

    BenchmarkDctPair(
      *gpu,
      implementation.name,
      gjxl::AcStrategyType::kDct8x16,
      dct8x16_transform_count,
      iterations);

    BenchmarkDctPair(
      *gpu,
      implementation.name,
      gjxl::AcStrategyType::kDct32x16,
      dct32x16_transform_count,
      iterations);

    BenchmarkDctPair(
      *gpu,
      implementation.name,
      gjxl::AcStrategyType::kDct16x32,
      dct16x32_transform_count,
      iterations);
  }

  return EXIT_SUCCESS;
}
