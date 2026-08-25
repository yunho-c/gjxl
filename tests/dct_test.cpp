// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
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
  size_t elements_per_block,
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

  const size_t block = worst_index / elements_per_block;
  const size_t element = worst_index % elements_per_block;
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

template <typename Batch>
std::vector<float> MakeReferenceInput() {
  constexpr size_t kElementsPerBlock = Batch::kElementsPerBlock;
  constexpr size_t kImpulseBlocks = kElementsPerBlock;
  constexpr size_t kRandomBlocks = 16;
  constexpr size_t kBlockCount = kImpulseBlocks + kRandomBlocks;

  std::vector<float> input(kBlockCount * kElementsPerBlock, 0.0f);

  for (size_t block = 0; block < kImpulseBlocks; ++block) {
    input[block * kElementsPerBlock + block] = 1.0f;
  }

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  for (size_t block = kImpulseBlocks; block < kBlockCount; ++block) {
    for (size_t i = 0; i < kElementsPerBlock; ++i) {
      input[block * kElementsPerBlock + i] = distribution(rng);
    }
  }

  return input;
}

template <typename Batch>
using DctOperation = gjxl::Status (gjxl::GpuBackend::*)(const Batch&);

template <typename Batch>
bool CheckDctOperation(
  gjxl::GpuBackend& gpu,
  std::string_view operation,
  DctOperation<Batch> transform,
  const Batch& batch,
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
    Batch::kElementsPerBlock,
    absolute_tolerance,
    relative_tolerance);
}

template <typename Batch>
bool TestDctKernels(
  gjxl::GpuBackend& gpu,
  std::string_view implementation_name,
  std::string_view transform_name,
  DctOperation<Batch> forward,
  DctOperation<Batch> inverse,
  double forward_absolute_tolerance,
  double inverse_absolute_tolerance) {

  const std::vector<float> input =
    MakeReferenceInput<Batch>();
  std::vector<float> output(input.size());
  std::vector<double> expected(input.size());

  std::unique_ptr<gjxl::DeviceBuffer> device_input;
  std::unique_ptr<gjxl::DeviceBuffer> device_output;
  const size_t bytes = input.size() * sizeof(float);

  if (!CheckStatus(
      gpu.Allocate(bytes, &device_input),
      std::string("Allocate ") + std::string(transform_name) +
        " reference input")) {
    return false;
  }

  if (!CheckStatus(
      gpu.Allocate(bytes, &device_output),
      std::string("Allocate ") + std::string(transform_name) +
        " reference output")) {
    return false;
  }

  if (!CheckStatus(
      gpu.CopyHostToDevice(
        *device_input,
        input.data(),
        bytes),
      std::string("Upload ") + std::string(transform_name) +
        " reference input")) {
    return false;
  }

  const Batch batch{
    .input = device_input.get(),
    .output = device_output.get(),
    .block_count = input.size() / Batch::kElementsPerBlock,
  };
  const gjxl::test::DctShape shape{
    .rows = Batch::kDimension,
    .cols = Batch::kDimension,
  };
  const size_t block_count = input.size() / Batch::kElementsPerBlock;
  const std::string forward_operation =
    std::string(implementation_name) + " Forward" +
      std::string(transform_name);
  const std::string inverse_operation =
    std::string(implementation_name) + " Inverse" +
      std::string(transform_name);

  gjxl::test::ReferenceForwardDct(
    shape,
    input.data(),
    expected.data(),
    block_count);

  if (!CheckDctOperation(
      gpu,
      forward_operation,
      forward,
      batch,
      &output,
      expected,
      forward_absolute_tolerance,
      5e-5)) {
    return false;
  }

  gjxl::test::ReferenceInverseDct(
    shape,
    input.data(),
    expected.data(),
    block_count);

  return CheckDctOperation(
    gpu,
    inverse_operation,
    inverse,
    batch,
    &output,
    expected,
    inverse_absolute_tolerance,
    5e-5);
}

template <typename Batch>
bool TestRoundTrip(
  gjxl::GpuBackend& gpu,
  std::string_view implementation_name,
  std::string_view transform_name,
  DctOperation<Batch> forward,
  DctOperation<Batch> inverse) {

  constexpr size_t kBlocks = 257;

  std::mt19937 rng(
    67890u ^
    static_cast<unsigned int>(Batch::kDimension << 8));
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  std::vector<float> input(
    kBlocks * Batch::kElementsPerBlock);

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

  const Batch forward_batch{
    .input = pixels.get(),
    .output = coefficients.get(),
    .block_count = kBlocks,
  };
  const Batch inverse_batch{
    .input = coefficients.get(),
    .output = output.get(),
    .block_count = kBlocks,
  };

  if (!CheckStatus(
      (gpu.*forward)(forward_batch),
      std::string(implementation_name) + " Forward" +
        std::string(transform_name))) {
    return false;
  }

  if (!CheckStatus(
      (gpu.*inverse)(inverse_batch),
      std::string(implementation_name) + " Inverse" +
        std::string(transform_name))) {
    return false;
  }

  if (!CheckStatus(
      gpu.Synchronize(),
      std::string(implementation_name) +
        " round-trip synchronization")) {
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
    << implementation_name
    << ' '
    << transform_name
    << " round-trip max error: "
    << max_error
    << '\n';

  constexpr float kTolerance = 5e-4f;

  if (max_error > kTolerance) {
    std::cerr
      << implementation_name
      << ' '
      << transform_name
      << " round-trip error exceeds tolerance\n";

    return false;
  }

  return true;
}

}  // namespace

int main() {
  for (const Dct8ImplementationCase& implementation :
       kDct8Implementations) {

    const gjxl::MetalBackendOptions options{
      .forward_dct8 = implementation.implementation,
      .inverse_dct8 = implementation.implementation,
    };

    std::unique_ptr<gjxl::GpuBackend> gpu;

    const gjxl::Status status =
      gjxl::CreateMetalBackend(
        GJXL_METALLIB_PATH,
        options,
        &gpu);

    if (!CheckStatus(
        status,
        std::string("CreateMetalBackend for ") +
          std::string(implementation.name))) {
      return EXIT_FAILURE;
    }

    std::cout
      << "Testing backend: "
      << gpu->name()
      << " ["
      << implementation.name
      << "]\n";

    if (!TestDctKernels<gjxl::Dct8Batch>(
        *gpu,
        implementation.name,
        "Dct8",
        &gjxl::GpuBackend::ForwardDct8,
        &gjxl::GpuBackend::InverseDct8,
        1e-5,
        5e-5)) {
      return EXIT_FAILURE;
    }

    if (implementation.implementation ==
        gjxl::MetalDct8Implementation::kScalarMatmul) {

      if (!TestDctKernels<gjxl::Dct16Batch>(
          *gpu,
          "scalar matmul",
          "Dct16",
          &gjxl::GpuBackend::ForwardDct16,
          &gjxl::GpuBackend::InverseDct16,
          2e-5,
          2e-4)) {
        return EXIT_FAILURE;
      }

      if (!TestRoundTrip<gjxl::Dct16Batch>(
          *gpu,
          "scalar matmul",
          "Dct16",
          &gjxl::GpuBackend::ForwardDct16,
          &gjxl::GpuBackend::InverseDct16)) {
        return EXIT_FAILURE;
      }

      if (!TestDctKernels<gjxl::Dct32Batch>(
          *gpu,
          "scalar matmul",
          "Dct32",
          &gjxl::GpuBackend::ForwardDct32,
          &gjxl::GpuBackend::InverseDct32,
          2e-5,
          2e-4)) {
        return EXIT_FAILURE;
      }

      if (!TestRoundTrip<gjxl::Dct32Batch>(
          *gpu,
          "scalar matmul",
          "Dct32",
          &gjxl::GpuBackend::ForwardDct32,
          &gjxl::GpuBackend::InverseDct32)) {
        return EXIT_FAILURE;
      }
    }

    if (!TestRoundTrip<gjxl::Dct8Batch>(
        *gpu,
        implementation.name,
        "Dct8",
        &gjxl::GpuBackend::ForwardDct8,
        &gjxl::GpuBackend::InverseDct8)) {
      return EXIT_FAILURE;
    }
  }

  std::cout << "All DCT tests passed.\n";
  return EXIT_SUCCESS;
}
