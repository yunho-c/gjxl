// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/backend.h"
#include "gpu/metal/metal_backend.h"
#include "dct_reference.h"

namespace {

constexpr size_t kDctRows = 8;
constexpr size_t kDctCols = 8;
constexpr size_t kDctSize = kDctRows * kDctCols;
constexpr gjxl::test::DctShape kDctShape{
  .rows = kDctRows,
  .cols = kDctCols,
};

bool CheckStatus(
  const gjxl::Status& status,
  std::string_view operation) {

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

bool CheckReferenceResults(
  std::string_view operation,
  const std::vector<float>& actual,
  const std::vector<double>& expected,
  double absolute_tolerance,
  double relative_tolerance) {

  if (actual.size() != expected.size()) {
    std::cerr
      << operation
      << " result size mismatch\n";

    return false;
  }

  double max_error = 0.0;
  double worst_ratio = 0.0;
  size_t worst_index = 0;

  for (size_t i = 0; i < actual.size(); ++i) {
    const double actual_value = static_cast<double>(actual[i]);
    const double expected_value = expected[i];
    const double error = std::abs(actual_value - expected_value);
    const double allowed_error =
      absolute_tolerance +
      relative_tolerance * std::abs(expected_value);

    max_error = std::max(max_error, error);

    const double ratio = std::isfinite(actual_value)
      ? error / allowed_error
      : std::numeric_limits<double>::infinity();

    if (ratio > worst_ratio) {
      worst_ratio = ratio;
      worst_index = i;
    }
  }

  std::cout
    << operation
    << " max absolute error: "
    << max_error
    << '\n';

  if (worst_ratio <= 1.0) {
    return true;
  }

  const size_t block = worst_index / kDctSize;
  const size_t element = worst_index % kDctSize;
  const double expected_value = expected[worst_index];
  const double actual_value = static_cast<double>(actual[worst_index]);
  const double allowed_error =
    absolute_tolerance +
    relative_tolerance * std::abs(expected_value);

  std::cerr
    << operation
    << " reference comparison failed at block "
    << block
    << ", element "
    << element
    << ": expected "
    << expected_value
    << ", got "
    << actual_value
    << ", error "
    << std::abs(actual_value - expected_value)
    << ", tolerance "
    << allowed_error
    << '\n';

  return false;
}

std::vector<float> MakeReferenceInput() {
  constexpr size_t kImpulseBlocks = kDctSize;
  constexpr size_t kRandomBlocks = 16;
  constexpr size_t kBlockCount = kImpulseBlocks + kRandomBlocks;

  std::vector<float> input(kBlockCount * kDctSize, 0.0f);

  for (size_t block = 0; block < kImpulseBlocks; ++block) {
    input[block * kDctSize + block] = 1.0f;
  }

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  for (size_t block = kImpulseBlocks; block < kBlockCount; ++block) {
    for (size_t i = 0; i < kDctSize; ++i) {
      input[block * kDctSize + i] = distribution(rng);
    }
  }

  return input;
}

using Dct8Operation = gjxl::Status (gjxl::GpuBackend::*)(const gjxl::Dct8Batch&);

bool CheckDct8Operation(
  gjxl::GpuBackend& gpu,
  std::string_view operation,
  Dct8Operation transform,
  const gjxl::Dct8Batch& batch,
  std::vector<float>* actual,
  const std::vector<double>& expected,
  double absolute_tolerance,
  double relative_tolerance) {

  if (actual == nullptr || batch.output == nullptr) {
    std::cerr << operation << " test received invalid output\n";
    return false;
  }

  const size_t bytes = actual->size() * sizeof(float);
  std::fill(
    actual->begin(),
    actual->end(),
    std::numeric_limits<float>::quiet_NaN());

  if (!CheckStatus(
      gpu.CopyHostToDevice(
        *batch.output,
        actual->data(),
        bytes),
      std::string(operation) + " output initialization")) {
    return false;
  }

  if (!CheckStatus(
      (gpu.*transform)(batch),
      std::string(operation) + " submission")) {
    return false;
  }

  if (!CheckStatus(
      gpu.Synchronize(),
      std::string(operation) + " synchronization")) {
    return false;
  }

  if (!CheckStatus(
      gpu.CopyDeviceToHost(
        *batch.output,
        actual->data(),
        bytes),
      std::string(operation) + " download")) {
    return false;
  }

  return CheckReferenceResults(
    operation,
    *actual,
    expected,
    absolute_tolerance,
    relative_tolerance);
}

bool TestDefaultDct8Kernels(
  gjxl::GpuBackend& gpu) {

  const std::vector<float> input = MakeReferenceInput();
  std::vector<float> output(input.size());

  std::unique_ptr<gjxl::DeviceBuffer> device_input;
  std::unique_ptr<gjxl::DeviceBuffer> device_output;

  const size_t bytes = input.size() * sizeof(float);

  if (!CheckStatus(
      gpu.Allocate(bytes, &device_input),
      "Allocate reference input")) {
    return false;
  }

  if (!CheckStatus(
      gpu.Allocate(bytes, &device_output),
      "Allocate reference output")) {
    return false;
  }

  if (!CheckStatus(
      gpu.CopyHostToDevice(
        *device_input,
        input.data(),
        bytes),
      "Upload reference input")) {
    return false;
  }

  const gjxl::Dct8Batch batch{
    .input = device_input.get(),
    .output = device_output.get(),
    .block_count = input.size() / kDctSize,
  };

  std::vector<double> expected(input.size());
  constexpr double kRelativeTolerance = 2e-5;

  gjxl::test::ReferenceForwardDct(
    kDctShape,
    input.data(),
    expected.data(),
    input.size() / kDctSize);

  if (!CheckDct8Operation(
      gpu,
      "Default ForwardDct8",
      &gjxl::GpuBackend::ForwardDct8,
      batch,
      &output,
      expected,
      1e-5,
      kRelativeTolerance)) {
    return false;
  }

  gjxl::test::ReferenceInverseDct(
    kDctShape,
    input.data(),
    expected.data(),
    input.size() / kDctSize);

  return CheckDct8Operation(
    gpu,
    "Default InverseDct8",
    &gjxl::GpuBackend::InverseDct8,
    batch,
    &output,
    expected,
    5e-5,
    kRelativeTolerance);
}

bool TestDefaultRoundTrip(
  gjxl::GpuBackend& gpu) {

  constexpr size_t kBlocks = 257;

  std::mt19937 rng(67890);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  std::vector<float> input(kBlocks * kDctSize);

  for (float& value : input) {
    value = distribution(rng);
  }

  std::vector<float> reconstructed(input.size());

  const size_t bytes = input.size() * sizeof(float);

  std::unique_ptr<gjxl::DeviceBuffer> pixels;
  std::unique_ptr<gjxl::DeviceBuffer> coefficients;
  std::unique_ptr<gjxl::DeviceBuffer> output;

  if (!CheckStatus(
      gpu.Allocate(bytes, &pixels),
      "Allocate round-trip pixels")) {
    return false;
  }

  if (!CheckStatus(
      gpu.Allocate(bytes, &coefficients),
      "Allocate round-trip coefficients")) {
    return false;
  }

  if (!CheckStatus(
      gpu.Allocate(bytes, &output),
      "Allocate round-trip output")) {
    return false;
  }

  if (!CheckStatus(
      gpu.CopyHostToDevice(
        *pixels,
        input.data(),
        bytes),
      "Upload round-trip pixels")) {
    return false;
  }

  if (!CheckStatus(
      gpu.ForwardDct8({
        .input = pixels.get(),
        .output = coefficients.get(),
        .block_count = kBlocks,
      }),
      "Default ForwardDct8")) {
    return false;
  }

  if (!CheckStatus(
      gpu.InverseDct8({
        .input = coefficients.get(),
        .output = output.get(),
        .block_count = kBlocks,
      }),
      "Default InverseDct8")) {
    return false;
  }

  if (!CheckStatus(
      gpu.Synchronize(),
      "Default round-trip synchronization")) {
    return false;
  }

  if (!CheckStatus(
      gpu.CopyDeviceToHost(
        *output,
        reconstructed.data(),
        bytes),
      "Download round-trip output")) {
    return false;
  }

  float max_error = 0.0f;

  for (size_t i = 0; i < input.size(); ++i) {
    max_error = std::max(
      max_error,
      std::abs(input[i] - reconstructed[i]));
  }

  std::cout
    << "Default round-trip max error: "
    << max_error
    << '\n';

  constexpr float kTolerance = 5e-4f;

  if (max_error > kTolerance) {
    std::cerr
      << "Default round-trip error exceeds tolerance\n";

    return false;
  }

  return true;
}

}  // namespace

int main() {
  std::unique_ptr<gjxl::GpuBackend> gpu;

  const gjxl::Status status =
    gjxl::CreateMetalBackend(
      GJXL_METALLIB_PATH,
      &gpu);

  if (!CheckStatus(status, "CreateMetalBackend")) {
    return EXIT_FAILURE;
  }

  std::cout
    << "Testing backend: "
    << gpu->name()
    << '\n';

  if (!TestDefaultDct8Kernels(*gpu)) {
    return EXIT_FAILURE;
  }

  if (!TestDefaultRoundTrip(*gpu)) {
    return EXIT_FAILURE;
  }

  std::cout << "All DCT tests passed.\n";
  return EXIT_SUCCESS;
}
