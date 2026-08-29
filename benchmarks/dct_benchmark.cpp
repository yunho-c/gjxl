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
  std::string_view cli_name;
};

constexpr std::array<ImplementationCase, 3>
kImplementations{{
  {
    .implementation =
      gjxl::MetalDctImplementation::kScalarMatmul,
    .name = "scalar matmul",
    .cli_name = "scalar",
  },
  {
    .implementation =
      gjxl::MetalDctImplementation::kSimdgroupMatmul,
    .name = "simdgroup matmul",
    .cli_name = "simd",
  },
  {
    .implementation =
      gjxl::MetalDctImplementation::kFactoredRadix2,
    .name = "factored radix-2",
    .cli_name = "factored",
  },
}};

constexpr std::array<gjxl::AcStrategyType, 9>
kBenchmarkedStrategies{{
  gjxl::AcStrategyType::kDct8,
  gjxl::AcStrategyType::kDct16x16,
  gjxl::AcStrategyType::kDct32x32,
  gjxl::AcStrategyType::kDct16x8,
  gjxl::AcStrategyType::kDct8x16,
  gjxl::AcStrategyType::kDct32x16,
  gjxl::AcStrategyType::kDct16x32,
  gjxl::AcStrategyType::kDct64x32,
  gjxl::AcStrategyType::kDct32x64,
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

void RequireSubmission(
  const gjxl::Status& status,
  const std::unique_ptr<gjxl::GpuSubmission>& submission,
  std::string_view operation) {

  Require(status, operation);
  if (submission == nullptr) {
    std::cerr << operation << " returned no submission handle\n";
    std::exit(EXIT_FAILURE);
  }
}

gjxl::Status WaitAll(
  const std::vector<std::unique_ptr<gjxl::GpuSubmission>>& submissions) {

  gjxl::Status first_error;
  for (const auto& submission : submissions) {
    const gjxl::Status status = submission->Wait();
    if (first_error.ok() && !status.ok()) {
      first_error = status;
    }
  }
  return first_error;
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

  std::vector<std::unique_ptr<gjxl::GpuSubmission>> warmup_submissions;
  warmup_submissions.reserve(kWarmupIterations);
  for (size_t i = 0; i < kWarmupIterations; ++i) {
    std::unique_ptr<gjxl::GpuSubmission> submission;
    const gjxl::Status status = submit(batch, &submission);
    RequireSubmission(status, submission, warmup_operation);
    warmup_submissions.push_back(std::move(submission));
  }

  Require(
    WaitAll(warmup_submissions),
    "Warmup completion");

  std::vector<std::unique_ptr<gjxl::GpuSubmission>> timed_submissions;
  timed_submissions.reserve(iterations);

  const auto begin = std::chrono::steady_clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    std::unique_ptr<gjxl::GpuSubmission> submission;
    const gjxl::Status status = submit(batch, &submission);
    RequireSubmission(status, submission, timed_operation);
    timed_submissions.push_back(std::move(submission));
  }

  Require(
    WaitAll(timed_submissions),
    "Timed completion");

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
  const double dense_equivalent_multiply_accumulates =
    pixels *
    static_cast<double>(extent.width + extent.height);

  std::cout
    << "\nOperation:        " << operation << '\n'
    << "Transforms/launch: " << batch.transform_count << '\n'
    << "Iterations:       " << iterations << '\n'
    << "Elapsed:          " << seconds << '\n'
    << "Transforms/s:     " << transforms / seconds << '\n'
    << "MPixels/s:        " << pixels / seconds / 1.0e6 << '\n'
    << "Dense-equivalent GMAC/s: "
    << dense_equivalent_multiply_accumulates / seconds / 1.0e9
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
    std::string(implementation_name) + " Forward" +
      std::string(strategy_info->name),
    [&gpu](
      const gjxl::TransformBatch& transform_batch,
      std::unique_ptr<gjxl::GpuSubmission>* submission) {
      return gpu.ForwardTransform(transform_batch, submission);
    },
    batch,
    iterations);

  BenchmarkDct(
    std::string(implementation_name) + " Inverse" +
      std::string(strategy_info->name),
    [&gpu](
      const gjxl::TransformBatch& transform_batch,
      std::unique_ptr<gjxl::GpuSubmission>* submission) {
      return gpu.InverseTransform(transform_batch, submission);
    },
    batch,
    iterations);
}

}  // namespace

int main(int argc, char** argv) {
  size_t dct8_transform_count = 65536;
  size_t iterations = 200;
  std::string_view implementation_filter;
  std::string_view strategy_filter;

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

  if (argc >= 4) {
    implementation_filter = argv[3];
    const bool known_implementation =
      std::any_of(
        kImplementations.begin(),
        kImplementations.end(),
        [implementation_filter](const ImplementationCase& implementation) {
          return implementation.cli_name == implementation_filter;
        });
    if (!known_implementation) {
      std::cerr << "Implementation must be scalar, simd, or factored\n";
      return EXIT_FAILURE;
    }
  }

  if (argc >= 5) {
    strategy_filter = argv[4];
    const bool known_strategy =
      std::any_of(
        kBenchmarkedStrategies.begin(),
        kBenchmarkedStrategies.end(),
        [strategy_filter](gjxl::AcStrategyType strategy) {
          const gjxl::AcStrategyInfo* info =
            gjxl::GetAcStrategyInfo(strategy);
          return info != nullptr && info->name == strategy_filter;
        });
    if (!known_strategy) {
      std::cerr << "Unknown AC strategy filter: " << strategy_filter << '\n';
      return EXIT_FAILURE;
    }
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
  const size_t dct64x32_transform_count =
    ComparableTransformCount(
      gjxl::AcStrategyType::kDct64x32,
      dct8_transform_count);
  const size_t dct32x64_transform_count =
    ComparableTransformCount(
      gjxl::AcStrategyType::kDct32x64,
      dct8_transform_count);

  for (const ImplementationCase& implementation :
       kImplementations) {

    if (!implementation_filter.empty() &&
        implementation.cli_name != implementation_filter) {
      continue;
    }

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
      .forward_dct64x32 = implementation.implementation,
      .inverse_dct64x32 = implementation.implementation,
      .forward_dct32x64 = implementation.implementation,
      .inverse_dct32x64 = implementation.implementation,
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

    const auto benchmark_pair =
      [&](gjxl::AcStrategyType strategy, size_t transform_count) {
        const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(strategy);
        if (info != nullptr &&
            (strategy_filter.empty() || info->name == strategy_filter)) {
          BenchmarkDctPair(
            *gpu,
            implementation.name,
            strategy,
            transform_count,
            iterations);
        }
      };

    benchmark_pair(
      gjxl::AcStrategyType::kDct8,
      dct8_transform_count);

    benchmark_pair(
      gjxl::AcStrategyType::kDct16x16,
      dct16_transform_count);

    benchmark_pair(
      gjxl::AcStrategyType::kDct32x32,
      dct32_transform_count);

    benchmark_pair(
      gjxl::AcStrategyType::kDct16x8,
      dct16x8_transform_count);

    benchmark_pair(
      gjxl::AcStrategyType::kDct8x16,
      dct8x16_transform_count);

    benchmark_pair(
      gjxl::AcStrategyType::kDct32x16,
      dct32x16_transform_count);

    benchmark_pair(
      gjxl::AcStrategyType::kDct16x32,
      dct16x32_transform_count);

    benchmark_pair(
      gjxl::AcStrategyType::kDct64x32,
      dct64x32_transform_count);

    benchmark_pair(
      gjxl::AcStrategyType::kDct32x64,
      dct32x64_transform_count);
  }

  return EXIT_SUCCESS;
}
