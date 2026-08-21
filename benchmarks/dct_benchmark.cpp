// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "gpu/backend.h"
#include "gpu/metal/metal_backend.h"

namespace {

constexpr size_t kDctSize = 64;

void Require(
  const gjxl::Status& status,
  const char* operation) {

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

}  // namespace

int main(int argc, char** argv) {
  size_t block_count = 65536;
  size_t iterations  = 200;

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

  std::unique_ptr<gjxl::GpuBackend> gpu;

  Require(
    gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
    "CreaeMetalBackend");

  const size_t element_count = block_count * kDctSize;

  const size_t bytes = element_count * sizeof(float);

  std::vector<float> input(element_count);

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  for (float& value : input) {
    value = distribution(rng);
  }

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
    "Upload");

  const gjxl::Dct8Batch batch{
    .input = device_input.get(),
    .output = device_output.get(),
    .block_count = block_count
  };

  // Warm-up
  for (int i = 0; i < 20; ++i) {
    Require(
      gpu->ForwardDct8(batch),
      "Warmup ForwardDct8");
  }

  Require(
    gpu->Synchronize(),
    "Warmup Synchronize");

  const auto begin = std::chrono::steady_clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    Require(
      gpu->ForwardDct8(batch),
      "ForwardDct8");
  }

  Require(
    gpu->Synchronize(),
    "Synchronize");

  const auto end = std::chrono::steady_clock::now();

  const double seconds =
    std::chrono::duration<double>(
      end - begin)
      .count();

  const double transforms =
    static_cast<double>(block_count) *
    static_cast<double>(iterations);

  const double pixels = transforms * 64.0;

  std::cout
    << "Backend:          " << gpu->name() << '\n'
    << "Blocks/launch:    " << block_count << '\n'
    << "Iterations:       " << iterations << '\n'
    << "Elapsed:          " << seconds << '\n'
    << "Transforms/s:     " << transforms / seconds << '\n'
    << "MPixels/s:        " << pixels / seconds / 1.0e6 << '\n';

  return EXIT_SUCCESS;
}
