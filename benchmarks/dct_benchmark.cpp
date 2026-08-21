// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

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

constexpr size_t kDctSize = 64;
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

template <typename Submit>
void BenchmarkDct8(
  gjxl::GpuBackend& gpu,
  std::string_view operation,
  const Submit& submit,
  const gjxl::Dct8Batch& batch,
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

  const double pixels = transforms * 64.0;

  std::cout
    << "\nOperation:        " << operation << '\n'
    << "Blocks/launch:    " << batch.block_count << '\n'
    << "Iterations:       " << iterations << '\n'
    << "Elapsed:          " << seconds << '\n'
    << "Transforms/s:     " << transforms / seconds << '\n'
    << "MPixels/s:        " << pixels / seconds / 1.0e6 << '\n';
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

  const size_t element_count = block_count * kDctSize;
  const size_t bytes = element_count * sizeof(float);

  std::vector<float> input(element_count);

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  for (float& value : input) {
    value = distribution(rng);
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

    std::unique_ptr<gjxl::DeviceBuffer> device_input;
    std::unique_ptr<gjxl::DeviceBuffer> device_output;

    Require(
      gpu->Allocate(bytes, &device_input),
      "Allocate input");

    Require(
      gpu->Allocate(bytes, &device_output),
      "Allocate output");

    Require(
      gpu->CopyHostToDevice(
        *device_input,
        input.data(),
        bytes),
      "Upload input");

    const gjxl::Dct8Batch batch{
      .input = device_input.get(),
      .output = device_output.get(),
      .block_count = block_count,
    };

    std::cout
      << "\nBackend: "
      << gpu->name()
      << " ["
      << implementation.name
      << "]\n";

    BenchmarkDct8(
      *gpu,
      std::string(implementation.name) + " ForwardDct8",
      [&gpu](const gjxl::Dct8Batch& dct_batch) {
        return gpu->ForwardDct8(dct_batch);
      },
      batch,
      iterations);

    BenchmarkDct8(
      *gpu,
      std::string(implementation.name) + " InverseDct8",
      [&gpu](const gjxl::Dct8Batch& dct_batch) {
        return gpu->InverseDct8(dct_batch);
      },
      batch,
      iterations);
  }

  return EXIT_SUCCESS;
}
