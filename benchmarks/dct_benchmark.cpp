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

#include "gpu/backend.h"
#include "gpu/metal/metal_backend.h"

namespace {

constexpr size_t kWarmupIterations = 20;

struct Dct8ImplementationCase {
  gjxl::MetalDct8Implementation implementation;
  std::string_view name;
};

constexpr std::array<Dct8ImplementationCase, 2>
kDct8Implementations{{
  {
    .implementation =
      gjxl::MetalDct8Implementation::kScalarMatmul,
    .name = "scalar matmul",
  },
  {
    .implementation =
      gjxl::MetalDct8Implementation::kSimdgroupMatmul,
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

template <typename Batch, typename Submit>
void BenchmarkDct(
  gjxl::GpuBackend& gpu,
  std::string_view operation,
  const Submit& submit,
  const Batch& batch,
  size_t iterations) {

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
    static_cast<double>(batch.block_count) *
    static_cast<double>(iterations);

  const double pixels =
    transforms *
    static_cast<double>(Batch::kElementsPerBlock);

  std::cout
    << "\nOperation:        " << operation << '\n'
    << "Blocks/launch:    " << batch.block_count << '\n'
    << "Iterations:       " << iterations << '\n'
    << "Elapsed:          " << seconds << '\n'
    << "Transforms/s:     " << transforms / seconds << '\n'
    << "MPixels/s:        " << pixels / seconds / 1.0e6 << '\n';
}

template <typename Batch>
using DctOperation = gjxl::Status (gjxl::GpuBackend::*)(const Batch&);

template <typename Batch>
void BenchmarkDctPair(
  gjxl::GpuBackend& gpu,
  std::string_view implementation_name,
  std::string_view transform_name,
  DctOperation<Batch> forward,
  DctOperation<Batch> inverse,
  size_t block_count,
  size_t iterations) {

  const size_t element_count =
    block_count * Batch::kElementsPerBlock;
  const size_t bytes = element_count * sizeof(float);
  std::vector<float> input(element_count);

  std::mt19937 rng(
    12345u ^
    static_cast<unsigned int>(Batch::kDimension << 8));
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

  const Batch batch{
    .input = device_input.get(),
    .output = device_output.get(),
    .block_count = block_count,
  };

  BenchmarkDct(
    gpu,
    std::string(implementation_name) + " Forward" +
      std::string(transform_name),
    [&gpu, forward](const Batch& dct_batch) {
      return (gpu.*forward)(dct_batch);
    },
    batch,
    iterations);

  BenchmarkDct(
    gpu,
    std::string(implementation_name) + " Inverse" +
      std::string(transform_name),
    [&gpu, inverse](const Batch& dct_batch) {
      return (gpu.*inverse)(dct_batch);
    },
    batch,
    iterations);
}

}  // namespace

int main(int argc, char** argv) {
  size_t block_count = 65536;
  size_t iterations = 200;

  if (argc >= 2) {
    block_count =
      static_cast<size_t>(
        std::strtoull(argv[1], nullptr, 10));
  }

  if (argc >= 3) {
    iterations =
      static_cast<size_t>(
        std::strtoull(argv[2], nullptr, 10));
  }

  for (const Dct8ImplementationCase& implementation :
       kDct8Implementations) {

    const gjxl::MetalBackendOptions options{
      .forward_dct8 = implementation.implementation,
      .inverse_dct8 = implementation.implementation,
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

    BenchmarkDctPair<gjxl::Dct8Batch>(
      *gpu,
      implementation.name,
      "Dct8",
      &gjxl::GpuBackend::ForwardDct8,
      &gjxl::GpuBackend::InverseDct8,
      block_count,
      iterations);

    if (implementation.implementation ==
        gjxl::MetalDct8Implementation::kScalarMatmul) {

      // Keep the number of transformed pixels per launch comparable to DCT8.
      const size_t dct16_block_count =
        std::max<size_t>(1, block_count / 4);
      const size_t dct32_block_count =
        std::max<size_t>(1, block_count / 16);

      BenchmarkDctPair<gjxl::Dct16Batch>(
        *gpu,
        "scalar matmul",
        "Dct16",
        &gjxl::GpuBackend::ForwardDct16,
        &gjxl::GpuBackend::InverseDct16,
        dct16_block_count,
        iterations);

      BenchmarkDctPair<gjxl::Dct32Batch>(
        *gpu,
        "scalar matmul",
        "Dct32",
        &gjxl::GpuBackend::ForwardDct32,
        &gjxl::GpuBackend::InverseDct32,
        dct32_block_count,
        iterations);
    }
  }

  return EXIT_SUCCESS;
}
