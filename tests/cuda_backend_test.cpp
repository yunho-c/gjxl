// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/ac_strategy.h"
#include "core/status.h"
#include "dct_reference.h"
#include "gpu/backend.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu/cuda/cuda_kernels.h"

namespace {

constexpr std::array kStrategies = {
  gjxl::AcStrategyType::kDct8,
  gjxl::AcStrategyType::kDct16x16,
  gjxl::AcStrategyType::kDct32x32,
  gjxl::AcStrategyType::kDct16x8,
  gjxl::AcStrategyType::kDct8x16,
  gjxl::AcStrategyType::kDct32x16,
  gjxl::AcStrategyType::kDct16x32,
  gjxl::AcStrategyType::kDct64x32,
  gjxl::AcStrategyType::kDct32x64,
};

bool CheckStatus(const gjxl::Status& status, const char* operation) {
  if (status.ok()) {
    return true;
  }
  std::cerr << operation << " failed: " << status.message() << '\n';
  return false;
}

bool CloseEnough(
  const std::vector<float>& actual,
  const std::vector<double>& expected,
  double absolute_tolerance,
  double relative_tolerance,
  const char* operation) {
  if (actual.size() != expected.size()) {
    std::cerr << operation << " result size differs\n";
    return false;
  }
  double worst_ratio = 0.0;
  size_t worst_index = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    const double error = std::abs(
      static_cast<double>(actual[index]) - expected[index]);
    const double tolerance = absolute_tolerance +
      relative_tolerance * std::abs(expected[index]);
    const double ratio = std::isfinite(actual[index])
      ? error / tolerance
      : std::numeric_limits<double>::infinity();
    if (ratio > worst_ratio) {
      worst_ratio = ratio;
      worst_index = index;
    }
  }
  if (worst_ratio <= 1.0) {
    return true;
  }
  std::cerr << operation << " differs at " << worst_index
            << ": actual=" << actual[worst_index]
            << " expected=" << expected[worst_index]
            << " ratio=" << worst_ratio << '\n';
  return false;
}

bool CheckFactoryAndBuffers(gjxl::GpuBackend& backend) {
  if (backend.kind() != gjxl::BackendKind::kCuda ||
      !backend.name().starts_with("CUDA")) {
    std::cerr << "CUDA backend identity is invalid\n";
    return false;
  }
  std::unique_ptr<gjxl::DeviceBuffer> buffer;
  if (backend.Allocate(0, &buffer).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      backend.Allocate(32, nullptr).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      !CheckStatus(backend.Allocate(32, &buffer), "Allocate CUDA buffer") ||
      buffer == nullptr || buffer->backend() != gjxl::BackendKind::kCuda ||
      buffer->size_bytes() != 32 || !backend.owns(*buffer)) {
    return false;
  }

  constexpr std::array<uint32_t, 4> source = {
    0x12345678u, 0x9abcdef0u, 0x13579bdfu, 0x2468ace0u};
  std::array<uint32_t, 4> destination{};
  if (!CheckStatus(
        backend.CopyHostToDevice(
          *buffer, source.data(), sizeof(source), sizeof(uint32_t)),
        "Offset host-to-device copy") ||
      !CheckStatus(
        backend.CopyDeviceToHost(
          *buffer, destination.data(), sizeof(destination), sizeof(uint32_t)),
        "Offset device-to-host copy") ||
      source != destination) {
    return false;
  }
  if (backend.CopyHostToDevice(
        *buffer, nullptr, 1).code() != gjxl::StatusCode::kInvalidArgument ||
      backend.CopyDeviceToHost(
        *buffer, nullptr, 1).code() != gjxl::StatusCode::kInvalidArgument ||
      backend.CopyHostToDevice(
        *buffer, source.data(), sizeof(source), 20).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      backend.CopyDeviceToHost(
        *buffer, destination.data(), sizeof(destination), 20).code() !=
        gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "CUDA copy validation did not reject an invalid request\n";
    return false;
  }
  return backend.stats().successful_allocations == 1;
}

bool CheckTwoDimensionalCopy(
  gjxl::GpuBackend& backend,
  gjxl::GpuBackend& other) {
  auto* cuda = dynamic_cast<gjxl::cuda_internal::CudaBackend*>(&backend);
  if (cuda == nullptr) {
    std::cerr << "CUDA backend implementation is unavailable\n";
    return false;
  }
  constexpr uint32_t kSentinel = 0xdeadbeefu;
  constexpr std::array<uint32_t, 8> source = {
    11, 12, 13, 101, 21, 22, 23, 102};
  std::array<uint32_t, 12> destination;
  destination.fill(kSentinel);
  std::unique_ptr<gjxl::DeviceBuffer> buffer;
  std::unique_ptr<gjxl::DeviceBuffer> foreign;
  if (!CheckStatus(
        backend.Allocate(sizeof(destination), &buffer),
        "Allocate CUDA 2D destination") ||
      !CheckStatus(
        other.Allocate(sizeof(destination), &foreign),
        "Allocate foreign CUDA 2D destination") ||
      !CheckStatus(
        backend.CopyHostToDevice(
          *buffer, destination.data(), sizeof(destination)),
        "Initialize CUDA 2D destination") ||
      !CheckStatus(
        cuda->CopyHostToDevice2D(
          *buffer, source.data(), 4 * sizeof(uint32_t),
          3 * sizeof(uint32_t), 2, 5 * sizeof(uint32_t),
          sizeof(uint32_t)),
        "Strided CUDA 2D copy") ||
      !CheckStatus(
        backend.CopyDeviceToHost(
          *buffer, destination.data(), sizeof(destination)),
        "Download CUDA 2D destination")) {
    return false;
  }
  constexpr std::array<uint32_t, 12> expected = {
    kSentinel, 11, 12, 13, kSentinel, kSentinel,
    21, 22, 23, kSentinel, kSentinel, kSentinel};
  if (destination != expected) {
    std::cerr << "Strided CUDA 2D copy changed padding or copied wrong rows\n";
    return false;
  }
  if (cuda->CopyHostToDevice2D(
        *buffer, nullptr, sizeof(uint32_t), sizeof(uint32_t), 1,
        sizeof(uint32_t)).code() != gjxl::StatusCode::kInvalidArgument ||
      cuda->CopyHostToDevice2D(
        *buffer, source.data(), 2 * sizeof(uint32_t),
        3 * sizeof(uint32_t), 2, 3 * sizeof(uint32_t)).code() !=
          gjxl::StatusCode::kInvalidArgument ||
      cuda->CopyHostToDevice2D(
        *buffer, source.data(), 4 * sizeof(uint32_t),
        3 * sizeof(uint32_t), 2, 2 * sizeof(uint32_t)).code() !=
          gjxl::StatusCode::kInvalidArgument ||
      cuda->CopyHostToDevice2D(
        *buffer, source.data(), 4 * sizeof(uint32_t),
        3 * sizeof(uint32_t), 2, 5 * sizeof(uint32_t),
        8 * sizeof(uint32_t)).code() !=
          gjxl::StatusCode::kInvalidArgument ||
      cuda->CopyHostToDevice2D(
        *buffer, source.data(), sizeof(uint32_t), sizeof(uint32_t),
        std::numeric_limits<size_t>::max(), sizeof(uint32_t)).code() !=
          gjxl::StatusCode::kInvalidArgument ||
      cuda->CopyHostToDevice2D(
        *foreign, source.data(), sizeof(uint32_t), sizeof(uint32_t), 1,
        sizeof(uint32_t)).code() != gjxl::StatusCode::kInvalidArgument ||
      !cuda->CopyHostToDevice2D(
        *buffer, nullptr, 0, 0, 0, 0, buffer->size_bytes()).ok()) {
    std::cerr << "CUDA 2D copy validation accepted an invalid request\n";
    return false;
  }
  return true;
}

bool CheckKernelLaunchErrorConsumption() {
  float* device_value = nullptr;
  cudaError_t status = cudaMalloc(
    reinterpret_cast<void**>(&device_value), sizeof(*device_value));
  if (status != cudaSuccess) {
    std::cerr << "CUDA launch-error test allocation failed: "
              << cudaGetErrorString(status) << '\n';
    return false;
  }
  status = cudaMemset(device_value, 0, sizeof(*device_value));
  if (status != cudaSuccess) {
    (void)cudaFree(device_value);
    std::cerr << "CUDA launch-error test initialization failed: "
              << cudaGetErrorString(status) << '\n';
    return false;
  }

  // Start with a clean per-thread launch-error slot, then deliberately create
  // an invalid zero-grid launch. Its wrapper must report and consume the
  // error so that the following valid launch is judged independently.
  (void)cudaGetLastError();
  const cudaError_t invalid_launch =
    gjxl::cuda_internal::LaunchCudaPointwiseAffine(
      nullptr, nullptr, 0, 0, 0, 0, 1.0f, 0.0f, nullptr);
  const cudaError_t valid_launch =
    gjxl::cuda_internal::LaunchCudaPointwiseAffine(
      device_value, device_value, 1, 1, 1, 1, 1.0f, 0.0f, nullptr);
  const cudaError_t completion = cudaDeviceSynchronize();
  const cudaError_t release = cudaFree(device_value);
  if (invalid_launch != cudaErrorInvalidConfiguration ||
      valid_launch != cudaSuccess || completion != cudaSuccess ||
      release != cudaSuccess) {
    std::cerr << "CUDA kernel wrapper preserved a stale launch error: invalid="
              << cudaGetErrorString(invalid_launch)
              << " valid=" << cudaGetErrorString(valid_launch)
              << " completion=" << cudaGetErrorString(completion)
              << " release=" << cudaGetErrorString(release) << '\n';
    return false;
  }
  return true;
}

bool CheckTransform(
  gjxl::GpuBackend& backend,
  gjxl::AcStrategyType strategy,
  std::mt19937* random) {
  const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(strategy);
  constexpr size_t kTransformCount = 3;
  const size_t element_count = info->coefficient_count();
  const size_t total_elements = kTransformCount * element_count;
  const size_t bytes = total_elements * sizeof(float);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> pixels(total_elements);
  std::generate(pixels.begin(), pixels.end(), [&] { return distribution(*random); });

  std::vector<double> expected_forward(total_elements);
  gjxl::test::ReferenceForwardDct(
    info->pixel_extent(), pixels.data(), expected_forward.data(),
    kTransformCount);

  std::unique_ptr<gjxl::DeviceBuffer> input;
  std::unique_ptr<gjxl::DeviceBuffer> coefficients;
  std::unique_ptr<gjxl::DeviceBuffer> reconstructed;
  if (!CheckStatus(backend.Allocate(bytes, &input), "Allocate DCT input") ||
      !CheckStatus(
        backend.Allocate(bytes, &coefficients), "Allocate DCT coefficients") ||
      !CheckStatus(
        backend.Allocate(bytes, &reconstructed), "Allocate DCT output") ||
      !CheckStatus(
        backend.CopyHostToDevice(*input, pixels.data(), bytes),
        "Upload DCT input")) {
    return false;
  }

  const gjxl::TransformBatch forward{
    .strategy = strategy,
    .input = input.get(),
    .output = coefficients.get(),
    .transform_count = kTransformCount,
  };
  const gjxl::TransformBatch inverse{
    .strategy = strategy,
    .input = coefficients.get(),
    .output = reconstructed.get(),
    .transform_count = kTransformCount,
  };
  std::unique_ptr<gjxl::GpuSubmission> forward_submission;
  std::unique_ptr<gjxl::GpuSubmission> inverse_submission;
  if (!CheckStatus(
        backend.ForwardTransform(forward, &forward_submission),
        "Submit forward CUDA DCT") ||
      !CheckStatus(
        backend.InverseTransform(inverse, &inverse_submission),
        "Submit inverse CUDA DCT") ||
      forward_submission == nullptr || inverse_submission == nullptr) {
    return false;
  }

  std::array<gjxl::Status, 4> wait_statuses;
  std::array<std::thread, 4> waiters;
  for (size_t index = 0; index < waiters.size(); ++index) {
    waiters[index] = std::thread([&, index] {
      wait_statuses[index] = inverse_submission->Wait();
    });
  }
  for (std::thread& waiter : waiters) {
    waiter.join();
  }
  for (const gjxl::Status& status : wait_statuses) {
    if (!CheckStatus(status, "Concurrent CUDA submission wait")) {
      return false;
    }
  }
  if (!CheckStatus(forward_submission->Wait(), "Wait for forward CUDA DCT") ||
      !CheckStatus(inverse_submission->Wait(), "Repeat CUDA submission wait")) {
    return false;
  }

  std::vector<float> actual_forward(total_elements);
  std::vector<float> actual_inverse(total_elements);
  if (!CheckStatus(
        backend.CopyDeviceToHost(
          *coefficients, actual_forward.data(), bytes),
        "Download CUDA DCT coefficients") ||
      !CheckStatus(
        backend.CopyDeviceToHost(
          *reconstructed, actual_inverse.data(), bytes),
        "Download inverse CUDA DCT")) {
    return false;
  }
  if (!CloseEnough(
        actual_forward, expected_forward, 3.0e-5, 3.0e-4,
        info->name.data())) {
    return false;
  }
  std::vector<double> expected_pixels(
    pixels.begin(), pixels.end());
  return CloseEnough(
    actual_inverse, expected_pixels, 5.0e-4, 5.0e-4,
    "CUDA DCT round trip");
}

bool CheckValidationAndOwnership(
  gjxl::GpuBackend& backend,
  gjxl::GpuBackend& other) {
  constexpr size_t kBytes = 64 * sizeof(float);
  std::unique_ptr<gjxl::DeviceBuffer> input;
  std::unique_ptr<gjxl::DeviceBuffer> output;
  std::unique_ptr<gjxl::DeviceBuffer> foreign;
  if (!backend.Allocate(kBytes, &input).ok() ||
      !backend.Allocate(kBytes, &output).ok() ||
      !other.Allocate(kBytes, &foreign).ok()) {
    return false;
  }
  const gjxl::TransformBatch valid{
    .input = input.get(), .output = output.get(), .transform_count = 1};
  std::unique_ptr<gjxl::GpuSubmission> submission;
  if (backend.ForwardTransform(valid, nullptr).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      backend.ForwardTransform(
        {.strategy = gjxl::AcStrategyType::kIdentity,
         .input = input.get(), .output = output.get(), .transform_count = 1},
        &submission).code() != gjxl::StatusCode::kUnavailable ||
      backend.ForwardTransform(
        {.input = foreign.get(), .output = output.get(), .transform_count = 1},
        &submission).code() != gjxl::StatusCode::kInvalidArgument ||
      backend.ForwardTransform(
        {.input = input.get(), .output = input.get(), .transform_count = 1},
        &submission).code() != gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "CUDA transform validation failed\n";
    return false;
  }
  if (!backend.ForwardTransform(
        {.strategy = gjxl::AcStrategyType::kDct8, .transform_count = 0},
        &submission).ok() || submission != nullptr) {
    std::cerr << "Empty CUDA transform committed work\n";
    return false;
  }

  if (!gjxl::ArmNextCudaSubmissionFailureForTest(
        backend, true, false).ok() ||
      backend.ForwardTransform(valid, &submission).code() !=
        gjxl::StatusCode::kSubmissionFailed || submission != nullptr) {
    std::cerr << "CUDA submission failure injection failed\n";
    return false;
  }
  if (!gjxl::ArmNextCudaSubmissionFailureForTest(
        backend, false, true).ok() ||
      !backend.ForwardTransform(valid, &submission).ok() ||
      submission == nullptr ||
      submission->Wait().code() != gjxl::StatusCode::kDeviceError) {
    std::cerr << "CUDA completion failure injection failed\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (gjxl::CreateCudaBackend(
        static_cast<std::unique_ptr<gjxl::GpuBackend>*>(nullptr)).code() !=
        gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "CUDA factory accepted a null output\n";
    return EXIT_FAILURE;
  }
  std::unique_ptr<gjxl::GpuBackend> invalid;
  if (gjxl::CreateCudaBackend({.device_ordinal = -1}, &invalid).code() !=
      gjxl::StatusCode::kInvalidArgument) {
    std::cerr << "CUDA factory accepted a negative device ordinal\n";
    return EXIT_FAILURE;
  }

  std::unique_ptr<gjxl::GpuBackend> backend;
  const gjxl::Status create = gjxl::CreateCudaBackend(&backend);
  if (create.code() == gjxl::StatusCode::kUnavailable) {
    std::cout << "CUDA backend unavailable: " << create.message() << '\n';
    return 77;
  }
  if (!CheckStatus(create, "Create CUDA backend") || backend == nullptr) {
    return EXIT_FAILURE;
  }
  std::unique_ptr<gjxl::GpuBackend> other;
  if (!CheckStatus(gjxl::CreateCudaBackend(&other), "Create second CUDA backend") ||
      other == nullptr ||
      !CheckFactoryAndBuffers(*backend) ||
      !CheckTwoDimensionalCopy(*backend, *other) ||
      !CheckKernelLaunchErrorConsumption() ||
      !CheckValidationAndOwnership(*backend, *other)) {
    return EXIT_FAILURE;
  }

  std::mt19937 random(12345);
  for (gjxl::AcStrategyType strategy : kStrategies) {
    if (!CheckTransform(*backend, strategy, &random)) {
      return EXIT_FAILURE;
    }
  }
  std::cout << "All CUDA backend tests passed on " << backend->name() << ".\n";
  return EXIT_SUCCESS;
}
