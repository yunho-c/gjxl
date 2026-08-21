// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "gpu/backend.h"
#include "gpu/metal/metal_backend.h"

namespace {

constexpr size_t kDctSize = 64;

bool CheckStatus(
  const gjxl::Status& status,
  const char* operation) {

    if (status.ok()) {
      return true;
    }

    std::cerr
      << operation
      << " failed: "
      << status.message()
      << '\n';

    return false;
}

bool TestConstantBlock(
  gjxl::GpuBackend& gpu) {

  constexpr size_t kBlocks = 64;

  std::vector<float> input(kBlocks * kDctSize);

  for (size_t block = 0; block < kBlocks; ++block) {
    const float value = 0.01f * static_cast<float>(block + 1);

    std::fill_n(
      input.data() + block * kDctSize,
      kDctSize,
      value);
  }

  std::vector<float> output(input.size());

  std::unique_ptr<gjxl::DeviceBuffer> device_input;
  std::unique_ptr<gjxl::DeviceBuffer> device_output;

  const size_t bytes = input.size() * sizeof(float);

  if (!CheckStatus(
      gpu.Allocate(bytes, &device_input),
      "Allocate input")) {
    return false;
  }

  if (!CheckStatus(
      gpu.Allocate(bytes, &device_output),
      "Allocate output")) {
    return false;
  }

  if (!CheckStatus(
      gpu.CopyHostToDevice(
        *device_input,
        input.data(),
        bytes),
      "Upload")) {
    return false;
  }

  gjxl::Dct8Batch batch{
    .input = device_input.get(),
    .output = device_output.get(),
    .block_count = kBlocks,
  };

  if (!CheckStatus(
      gpu.ForwardDct8(batch),
      "ForwardDct8")) {
    return false;
  }

  if (!CheckStatus(
      gpu.Synchronize(),
      "Synchronize")) {
    return false;
  }

  if (!CheckStatus(
      gpu.CopyDeviceToHost(
        *device_output,
        output.data(),
        bytes),
      "Download")) {
    return false;
  }

  // A constant block must have zero AC coefficients,
  // independent of the precise DC normalization convention.
  constexpr float kTolerance = 1e-5f;

  for (size_t block = 0; block < kBlocks; ++block) {
    const float* coeff = output.data() + block * kDctSize;

    for (size_t i = 1; i < kDctSize; ++i) {
      if (std::abs(coeff[i]) > kTolerance) {
        std::cerr
          << "Constant-block test failed at block "
          << block
          << ", coefficient "
          << i
          << ": "
          << coeff[i]
          << '\n';

        return false;
      }
    }
  }

  return true;
}

bool TestRoundTrip(
  gjxl::GpuBackend& gpu) {

  constexpr size_t kBlocks = 1024;

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  std::vector<float> input(kBlocks * kDctSize);

  for (float& x : input) {
    x = distribution(rng);
  }

  std::vector<float> reconstructed(input.size());

  const size_t bytes = input.size() * sizeof(float);

  std::unique_ptr<gjxl::DeviceBuffer> pixels;
  std::unique_ptr<gjxl::DeviceBuffer> coefficients;
  std::unique_ptr<gjxl::DeviceBuffer> output;

  if (!CheckStatus(
      gpu.Allocate(bytes, &pixels),
      "Allocate pixels")) {
    return false;
  }

  if (!CheckStatus(
      gpu.Allocate(bytes, &coefficients),
      "Allocate coefficients")) {
    return false;
  }

  if (!CheckStatus(
      gpu.Allocate(bytes, &output),
      "Allocate reconstruction")) {
    return false;
  }

  if (!CheckStatus(
      gpu.CopyHostToDevice(
        *pixels,
        input.data(),
        bytes),
      "Upload pixels")) {
    return false;
  }

  if (!CheckStatus(
      gpu.ForwardDct8({
        .input = pixels.get(),
        .output = coefficients.get(),
        .block_count = kBlocks,
      }),
      "ForwardDct8")) {
    return false;
  }

  if (!CheckStatus(
      gpu.InverseDct8({
        .input = coefficients.get(),
        .output = output.get(),
        .block_count = kBlocks,
      }),
      "InverseDct8")) {
    return false;
  }

  if (!CheckStatus(
      gpu.Synchronize(),
      "Synchronize")) {
    return false;
  }

  if (!CheckStatus(
      gpu.CopyDeviceToHost(
        *output,
        reconstructed.data(),
        bytes),
      "Download reconstruction")) {
    return false;
  }

  float max_error = 0.0f;

  for (size_t i = 0; i < input.size(); ++i) {
    max_error = std::max(
      max_error,
      std::abs(input[i] - reconstructed[i]));
  }

  std::cout
    << "Round-trip max error: "
    << max_error
    << '\n';

  constexpr float kTolerance = 5e-4f;

  if (max_error > kTolerance) {
    std::cerr
      << "Round-trip error exceeds tolerance\n";

    return false;
  }

  return true;
}

}  // namespace

int main() {
  std::unique_ptr<gjxl::GpuBackend> gpu;

  gjxl::Status status = gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu);

  if (!CheckStatus(status, "CreateMetalBackend")) {
    return EXIT_FAILURE;
  }

  std::cout
    << "Testing backend: "
    << gpu->name()
    << '\n';

  if (!TestConstantBlock(*gpu)) {
    return EXIT_FAILURE;
  }

  if (!TestRoundTrip(*gpu)) {
    return EXIT_FAILURE;
  }

  std::cout << "All DCT tests passed.\n";
  return EXIT_SUCCESS;
}
