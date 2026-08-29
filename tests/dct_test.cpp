// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

struct ImplementationCase {
  gjxl::MetalDctImplementation implementation;
  std::string_view name;
};

constexpr std::array<ImplementationCase, 3>
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
  {
    .implementation =
      gjxl::MetalDctImplementation::kFactoredRadix2,
    .name = "factored radix-2",
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
  size_t elements_per_transform,
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

  const size_t transform = worst_index / elements_per_transform;
  const size_t element = worst_index % elements_per_transform;
  const double expected_value = expected[worst_index];
  const double actual_value = static_cast<double>(actual[worst_index]);
  const double allowed_error =
    absolute_tolerance +
    relative_tolerance * std::abs(expected_value);

  std::cerr
    << operation
    << " reference comparison failed at transform "
    << transform
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

std::vector<float> MakeReferenceInput(size_t elements_per_transform) {
  const size_t impulse_transforms = elements_per_transform;
  // Keep the total DCT8 count non-divisible by its packing factor so direct
  // oracle checks exercise the guarded final threadgroup.
  constexpr size_t kRandomTransforms = 17;
  const size_t transform_count =
    impulse_transforms + kRandomTransforms;

  std::vector<float> input(
    transform_count * elements_per_transform,
    0.0f);

  for (size_t transform = 0;
       transform < impulse_transforms;
       ++transform) {

    input[transform * elements_per_transform + transform] = 1.0f;
  }

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  for (size_t transform = impulse_transforms;
       transform < transform_count;
       ++transform) {

    for (size_t i = 0; i < elements_per_transform; ++i) {
      input[transform * elements_per_transform + i] = distribution(rng);
    }
  }

  return input;
}

using TransformOperation =
  gjxl::Status (gjxl::GpuBackend::*)(
    const gjxl::TransformBatch&,
    std::unique_ptr<gjxl::GpuSubmission>*);

using ReferenceTransform = void (*)(
  gjxl::Extent2D,
  const float*,
  double*,
  size_t);

bool CheckDctOperation(
  gjxl::GpuBackend& gpu,
  std::string_view operation,
  TransformOperation transform,
  const gjxl::TransformBatch& batch,
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

  std::unique_ptr<gjxl::GpuSubmission> submission;
  if (!CheckStatus(
      (gpu.*transform)(batch, &submission),
      std::string(operation) + " submission") ||
      submission == nullptr) {
    std::cerr << operation << " did not return a submission handle\n";
    return false;
  }

  if (!CheckStatus(
      submission->Wait(),
      std::string(operation) + " completion")) {
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

  const gjxl::AcStrategyInfo* strategy_info =
    gjxl::GetAcStrategyInfo(batch.strategy);

  if (strategy_info == nullptr) {
    std::cerr << operation << " test received invalid strategy\n";
    return false;
  }

  return CheckReferenceResults(
    operation,
    *actual,
    expected,
    strategy_info->coefficient_count(),
    absolute_tolerance,
    relative_tolerance);
}

bool TestDctKernel(
  gjxl::GpuBackend& gpu,
  std::string_view implementation_name,
  gjxl::AcStrategyType strategy,
  std::string_view direction_name,
  TransformOperation transform,
  ReferenceTransform reference_transform,
  double absolute_tolerance) {

  const gjxl::AcStrategyInfo* strategy_info =
    gjxl::GetAcStrategyInfo(strategy);

  if (strategy_info == nullptr) {
    std::cerr << "TestDctKernel received invalid strategy\n";
    return false;
  }

  const size_t elements_per_transform =
    strategy_info->coefficient_count();
  const std::vector<float> input =
    MakeReferenceInput(elements_per_transform);
  std::vector<float> output(input.size());
  std::vector<double> expected(input.size());

  std::unique_ptr<gjxl::DeviceBuffer> device_input;
  std::unique_ptr<gjxl::DeviceBuffer> device_output;
  const size_t bytes = input.size() * sizeof(float);

  if (!CheckStatus(
      gpu.Allocate(bytes, &device_input),
      std::string("Allocate ") + std::string(strategy_info->name) +
        " reference input")) {
    return false;
  }

  if (!CheckStatus(
      gpu.Allocate(bytes, &device_output),
      std::string("Allocate ") + std::string(strategy_info->name) +
        " reference output")) {
    return false;
  }

  if (!CheckStatus(
      gpu.CopyHostToDevice(
        *device_input,
        input.data(),
        bytes),
      std::string("Upload ") + std::string(strategy_info->name) +
        " reference input")) {
    return false;
  }

  const gjxl::TransformBatch batch{
    .strategy = strategy,
    .input = device_input.get(),
    .output = device_output.get(),
    .transform_count = input.size() / elements_per_transform,
  };
  const gjxl::Extent2D extent = strategy_info->pixel_extent();
  const size_t transform_count =
    input.size() / elements_per_transform;
  const std::string operation =
    std::string(implementation_name) + ' ' +
      std::string(direction_name) +
      std::string(strategy_info->name);

  reference_transform(
    extent,
    input.data(),
    expected.data(),
    transform_count);

  return CheckDctOperation(
    gpu,
    operation,
    transform,
    batch,
    &output,
    expected,
    absolute_tolerance,
    5e-5);
}

bool TestDctKernels(
  gjxl::GpuBackend& gpu,
  std::string_view implementation_name,
  gjxl::AcStrategyType strategy,
  double forward_absolute_tolerance,
  double inverse_absolute_tolerance) {

  return
    TestDctKernel(
      gpu,
      implementation_name,
      strategy,
      "Forward",
      &gjxl::GpuBackend::ForwardTransform,
      &gjxl::test::ReferenceForwardDct,
      forward_absolute_tolerance) &&
    TestDctKernel(
      gpu,
      implementation_name,
      strategy,
      "Inverse",
      &gjxl::GpuBackend::InverseTransform,
      &gjxl::test::ReferenceInverseDct,
      inverse_absolute_tolerance);
}

bool TestRoundTrip(
  gjxl::GpuBackend& gpu,
  std::string_view implementation_name,
  gjxl::AcStrategyType strategy) {

  const gjxl::AcStrategyInfo* strategy_info =
    gjxl::GetAcStrategyInfo(strategy);

  if (strategy_info == nullptr) {
    std::cerr << "TestRoundTrip received invalid strategy\n";
    return false;
  }

  constexpr size_t kTransformCount = 257;
  const gjxl::Extent2D extent = strategy_info->pixel_extent();
  const size_t elements_per_transform =
    strategy_info->coefficient_count();

  std::mt19937 rng(
    67890u ^
    static_cast<unsigned int>(extent.width << 8) ^
    static_cast<unsigned int>(extent.height));
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  std::vector<float> input(
    kTransformCount * elements_per_transform);

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

  const gjxl::TransformBatch forward_batch{
    .strategy = strategy,
    .input = pixels.get(),
    .output = coefficients.get(),
    .transform_count = kTransformCount,
  };
  const gjxl::TransformBatch inverse_batch{
    .strategy = strategy,
    .input = coefficients.get(),
    .output = output.get(),
    .transform_count = kTransformCount,
  };

  std::unique_ptr<gjxl::GpuSubmission> forward_submission;
  std::unique_ptr<gjxl::GpuSubmission> inverse_submission;
  if (!CheckStatus(
      gpu.ForwardTransform(forward_batch, &forward_submission),
      std::string(implementation_name) + " Forward" +
        std::string(strategy_info->name)) ||
      forward_submission == nullptr) {
    return false;
  }

  if (!CheckStatus(
      gpu.InverseTransform(inverse_batch, &inverse_submission),
      std::string(implementation_name) + " Inverse" +
        std::string(strategy_info->name)) ||
      inverse_submission == nullptr) {
    return false;
  }

  const gjxl::Status forward_completion = forward_submission->Wait();
  const gjxl::Status inverse_completion = inverse_submission->Wait();
  if (!CheckStatus(
        forward_completion,
        std::string(implementation_name) + " forward completion") ||
      !CheckStatus(
        inverse_completion,
        std::string(implementation_name) + " inverse completion")) {
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
    << strategy_info->name
    << " round-trip max error: "
    << max_error
    << '\n';

  constexpr float kTolerance = 5e-4f;

  if (max_error > kTolerance) {
    std::cerr
      << implementation_name
      << ' '
      << strategy_info->name
      << " round-trip error exceeds tolerance\n";

    return false;
  }

  return true;
}

bool TestTransformValidation(gjxl::GpuBackend& gpu) {
  constexpr size_t kDct8Bytes = 64 * sizeof(float);
  std::unique_ptr<gjxl::DeviceBuffer> input;
  std::unique_ptr<gjxl::DeviceBuffer> output;

  if (!CheckStatus(
      gpu.Allocate(kDct8Bytes, &input),
      "Allocate validation input") ||
      !CheckStatus(
        gpu.Allocate(kDct8Bytes, &output),
        "Allocate validation output")) {

    return false;
  }

  const gjxl::TransformBatch valid_batch{
    .strategy = gjxl::AcStrategyType::kDct8,
    .input = input.get(),
    .output = output.get(),
    .transform_count = 1,
  };
  const uint64_t before = gpu.stats().committed_submissions;
  if (gpu.ForwardTransform(valid_batch, nullptr).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      gpu.stats().committed_submissions != before) {
    std::cerr << "Null transform submission output committed work\n";
    return false;
  }

  std::unique_ptr<gjxl::GpuSubmission> submission;
  if (!gpu.ForwardTransform(valid_batch, &submission).ok() ||
      submission == nullptr || !submission->Wait().ok()) {
    std::cerr << "Valid transform did not return a usable submission\n";
    return false;
  }
  const gjxl::Status unsupported =
    gpu.ForwardTransform({
      .strategy = gjxl::AcStrategyType::kIdentity,
      .input = input.get(),
      .output = output.get(),
      .transform_count = 1,
    }, &submission);

  if (unsupported.code() != gjxl::StatusCode::kUnavailable ||
      submission != nullptr) {
    std::cerr
      << "Unsupported strategy did not return unavailable: "
      << unsupported.message()
      << '\n';
    return false;
  }

  const gjxl::Status undersized =
    gpu.ForwardTransform({
      .strategy = gjxl::AcStrategyType::kDct16x16,
      .input = input.get(),
      .output = output.get(),
      .transform_count = 1,
    }, &submission);

  if (undersized.code() != gjxl::StatusCode::kInvalidArgument ||
      submission != nullptr) {
    std::cerr
      << "Undersized transform did not return invalid argument: "
      << undersized.message()
      << '\n';
    return false;
  }

  const gjxl::Status invalid =
    gpu.ForwardTransform({
      .strategy = static_cast<gjxl::AcStrategyType>(255),
      .input = input.get(),
      .output = output.get(),
      .transform_count = 1,
    }, &submission);

  if (invalid.code() != gjxl::StatusCode::kInvalidArgument ||
      submission != nullptr) {
    std::cerr
      << "Invalid strategy did not return invalid argument: "
      << invalid.message()
      << '\n';
    return false;
  }

  const gjxl::Status empty = gpu.ForwardTransform({
      .strategy = gjxl::AcStrategyType::kDct8,
      .transform_count = 0,
    }, &submission);
  return CheckStatus(empty, "Empty transform batch") &&
         submission == nullptr;
}

bool TestTransformFailureStatuses() {
  constexpr size_t kDct8Bytes = 64 * sizeof(float);

  gjxl::MetalBackendOptions submit_options;
  submit_options.test_fail_submission = true;
  std::unique_ptr<gjxl::GpuBackend> submit_gpu;
  if (!gjxl::CreateMetalBackend(
        GJXL_METALLIB_PATH, submit_options, &submit_gpu).ok()) {
    return false;
  }
  std::unique_ptr<gjxl::DeviceBuffer> submit_input;
  std::unique_ptr<gjxl::DeviceBuffer> submit_output;
  if (!submit_gpu->Allocate(kDct8Bytes, &submit_input).ok() ||
      !submit_gpu->Allocate(kDct8Bytes, &submit_output).ok()) {
    return false;
  }
  const gjxl::TransformBatch submit_batch{
    .strategy = gjxl::AcStrategyType::kDct8,
    .input = submit_input.get(),
    .output = submit_output.get(),
    .transform_count = 1,
  };
  std::unique_ptr<gjxl::GpuSubmission> submission;
  if (submit_gpu->ForwardTransform(submit_batch, &submission).code() !=
        gjxl::StatusCode::kSubmissionFailed ||
      submission != nullptr ||
      submit_gpu->stats().committed_submissions != 0) {
    std::cerr << "Injected transform submission failure was not isolated\n";
    return false;
  }

  gjxl::MetalBackendOptions completion_options;
  completion_options.test_fail_completion = true;
  std::unique_ptr<gjxl::GpuBackend> completion_gpu;
  if (!gjxl::CreateMetalBackend(
        GJXL_METALLIB_PATH, completion_options, &completion_gpu).ok()) {
    return false;
  }
  std::unique_ptr<gjxl::DeviceBuffer> completion_input;
  std::unique_ptr<gjxl::DeviceBuffer> completion_output_a;
  std::unique_ptr<gjxl::DeviceBuffer> completion_output_b;
  if (!completion_gpu->Allocate(kDct8Bytes, &completion_input).ok() ||
      !completion_gpu->Allocate(kDct8Bytes, &completion_output_a).ok() ||
      !completion_gpu->Allocate(kDct8Bytes, &completion_output_b).ok()) {
    return false;
  }
  std::unique_ptr<gjxl::GpuSubmission> first;
  std::unique_ptr<gjxl::GpuSubmission> second;
  const gjxl::TransformBatch first_batch{
    .strategy = gjxl::AcStrategyType::kDct8,
    .input = completion_input.get(),
    .output = completion_output_a.get(),
    .transform_count = 1,
  };
  gjxl::TransformBatch second_batch = first_batch;
  second_batch.output = completion_output_b.get();
  if (!completion_gpu->ForwardTransform(first_batch, &first).ok() ||
      !completion_gpu->ForwardTransform(second_batch, &second).ok() ||
      first == nullptr || second == nullptr) {
    std::cerr << "Outstanding transform submissions were not returned\n";
    return false;
  }
  const gjxl::Status first_status = first->Wait();
  const gjxl::Status second_status = second->Wait();
  const gjxl::Status first_again = first->Wait();
  return first_status.code() == gjxl::StatusCode::kDeviceError &&
         second_status.code() == gjxl::StatusCode::kDeviceError &&
         first_again.code() == first_status.code() &&
         first_again.message() == first_status.message() &&
         completion_gpu->stats().committed_submissions == 2;
}

}  // namespace

int main() {
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
      .forward_dct64x32 = implementation.implementation,
      .inverse_dct64x32 = implementation.implementation,
      .forward_dct32x64 = implementation.implementation,
      .inverse_dct32x64 = implementation.implementation,
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

    if (!TestDctKernels(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct8,
        1e-5,
        5e-5)) {
      return EXIT_FAILURE;
    }

    if (!TestRoundTrip(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct8)) {
      return EXIT_FAILURE;
    }

    if (!TestDctKernels(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct16x16,
        2e-5,
        2e-4)) {
      return EXIT_FAILURE;
    }

    if (!TestRoundTrip(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct16x16)) {
      return EXIT_FAILURE;
    }

    if (!TestDctKernels(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct32x32,
        2e-5,
        2e-4)) {
      return EXIT_FAILURE;
    }

    if (!TestRoundTrip(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct32x32)) {
      return EXIT_FAILURE;
    }

    if (implementation.implementation ==
        gjxl::MetalDctImplementation::kScalarMatmul) {

      if (!TestTransformValidation(*gpu)) {
        return EXIT_FAILURE;
      }
    }

    if (!TestDctKernels(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct32x16,
        2e-5,
        2e-4)) {
      return EXIT_FAILURE;
    }

    if (!TestRoundTrip(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct32x16)) {
      return EXIT_FAILURE;
    }

    if (!TestDctKernels(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct16x32,
        2e-5,
        2e-4)) {
      return EXIT_FAILURE;
    }

    if (!TestRoundTrip(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct16x32)) {
      return EXIT_FAILURE;
    }

    if (!TestDctKernels(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct64x32,
        2e-5,
        2e-4)) {
      return EXIT_FAILURE;
    }

    if (!TestRoundTrip(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct64x32)) {
      return EXIT_FAILURE;
    }

    if (!TestDctKernels(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct32x64,
        2e-5,
        2e-4)) {
      return EXIT_FAILURE;
    }

    if (!TestRoundTrip(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct32x64)) {
      return EXIT_FAILURE;
    }

    if (!TestDctKernels(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct16x8,
        2e-5,
        2e-4)) {
      return EXIT_FAILURE;
    }

    if (!TestRoundTrip(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct16x8)) {
      return EXIT_FAILURE;
    }

    if (!TestDctKernels(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct8x16,
        2e-5,
        2e-4)) {
      return EXIT_FAILURE;
    }

    if (!TestRoundTrip(
        *gpu,
        implementation.name,
        gjxl::AcStrategyType::kDct8x16)) {
      return EXIT_FAILURE;
    }
  }

  if (!TestTransformFailureStatuses()) {
    return EXIT_FAILURE;
  }

  std::cout << "All DCT tests passed.\n";
  return EXIT_SUCCESS;
}
