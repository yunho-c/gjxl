// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <bit>
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
  std::mt19937* random,
  size_t transform_count) {
  const gjxl::AcStrategyInfo* info = gjxl::GetAcStrategyInfo(strategy);
  const size_t element_count = info->coefficient_count();
  const size_t total_elements = transform_count * element_count;
  const size_t bytes = total_elements * sizeof(float);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> pixels(total_elements);
  std::generate(pixels.begin(), pixels.end(), [&] { return distribution(*random); });
  // Exercise coefficient layout and shared-tile boundaries with impulses,
  // a constant, and unequal horizontal/vertical structure as well as noise.
  if (transform_count >= 6) {
    std::fill_n(pixels.begin(), 3 * element_count, 0.0f);
    pixels[0] = 1.0f;
    pixels[element_count + element_count / 2 + info->pixel_extent().width - 1] = -0.5f;
    pixels[3 * element_count - 1] = 0.75f;
    std::fill_n(pixels.begin() + 3 * element_count, element_count, 0.375f);
    for (size_t index = 0; index < element_count; ++index) {
      const size_t x = index % info->pixel_extent().width;
      const size_t y = index / info->pixel_extent().width;
      pixels[4 * element_count + index] = (x + y) % 2 == 0 ? 0.75f : -0.75f;
      pixels[5 * element_count + index] =
        0.5f * static_cast<float>(x) / info->pixel_extent().width -
        0.25f * static_cast<float>(y) / info->pixel_extent().height;
    }
  }

  std::vector<double> expected_forward(total_elements);
  gjxl::test::ReferenceForwardDct(
    info->pixel_extent(), pixels.data(), expected_forward.data(),
    transform_count);

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
    .transform_count = transform_count,
  };
  const gjxl::TransformBatch inverse{
    .strategy = strategy,
    .input = coefficients.get(),
    .output = reconstructed.get(),
    .transform_count = transform_count,
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
  if (!CloseEnough(
        actual_inverse, expected_pixels, 5.0e-4, 5.0e-4,
        "CUDA DCT round trip")) {
    return false;
  }

  // Also test inverse transforms independently: a round trip alone can hide
  // compensating layout mistakes in the forward and inverse implementations.
  gjxl::test::ReferenceInverseDct(
    info->pixel_extent(), pixels.data(), expected_pixels.data(),
    transform_count);
  if (!CheckStatus(
        backend.CopyHostToDevice(*coefficients, pixels.data(), bytes),
        "Upload independent inverse CUDA DCT input") ||
      !CheckStatus(
        backend.InverseTransform(inverse, &inverse_submission),
        "Submit independent inverse CUDA DCT") ||
      inverse_submission == nullptr ||
      !CheckStatus(inverse_submission->Wait(), "Wait for independent inverse DCT") ||
      !CheckStatus(
        backend.CopyDeviceToHost(*reconstructed, actual_inverse.data(), bytes),
        "Download independent inverse CUDA DCT")) {
    return false;
  }
  return CloseEnough(actual_inverse, expected_pixels, 5.0e-4, 5.0e-4,
    "CUDA independent inverse DCT");
}

bool CheckAcStrategyInverseLoss(
  gjxl::GpuBackend& backend, gjxl::AcStrategyType strategy,
  std::mt19937* random, uint32_t candidate_count) {
  using gjxl::cuda_internal::CudaBuffer;
  using gjxl::cuda_internal::CudaRuntimeStatus;
  const auto extent = gjxl::GetAcStrategyInfo(strategy)->pixel_extent();
  const size_t count = static_cast<size_t>(extent.width) * extent.height;
  const size_t transform_count = static_cast<size_t>(candidate_count) * 3;
  constexpr uint32_t kWidth = 72;
  constexpr uint32_t kHeight = 64;
  constexpr uint32_t kStride = kWidth + 11;
  constexpr size_t kMaskOffset = 13;
  constexpr size_t kLossOffset = 7;
  constexpr float kGuard = -12345.0f;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> mask(kMaskOffset + kStride * kHeight + 17, nan);
  for (size_t y = 0; y < kHeight; ++y) {
    for (size_t x = 0; x < kWidth; ++x) {
      mask[kMaskOffset + y * kStride + x] =
        0.125f + static_cast<float>((x * 13 + y * 19) % 127) / 64.0f;
    }
  }
  const auto valid_mask = mask;
  std::vector<gjxl::AcStrategyCandidate> candidates(candidate_count);
  for (size_t i = 0; i < candidate_count; ++i) {
    // Include all image corners, overlapping rectangles, and bad footprints.
    candidates[i].block_x = static_cast<uint32_t>(i % 2) *
      ((kWidth - extent.width) / 8);
    candidates[i].block_y = static_cast<uint32_t>((i / 2) % 2) *
      ((kHeight - extent.height) / 8);
    if (i % 11 == 9) candidates[i].block_x = kWidth / 8;
    if (i % 11 == 10) {
      candidates[i].block_y = std::numeric_limits<uint32_t>::max();
    }
  }
  std::uniform_real_distribution<float> distribution(-0.005f, 0.005f);
  std::vector<float> coefficients(transform_count * count);
  std::generate(coefficients.begin(), coefficients.end(),
    [&] { return distribution(*random); });
  // Zero and differently positioned impulses accompany the random channels.
  for (size_t t = 0; t < transform_count; ++t) {
    if (t % 6 < 3) {
      std::fill_n(coefficients.begin() + t * count, count, 0.0f);
      if (t % 6 != 0) {
        coefficients[t * count + (t % 6 == 1 ? 0 : count - 1)] = 0.002f;
      }
    }
  }
  std::unique_ptr<gjxl::DeviceBuffer> input, pixels, device_mask;
  std::unique_ptr<gjxl::DeviceBuffer> descriptors, losses;
  const auto upload = [&](const void* data, size_t bytes,
                          std::unique_ptr<gjxl::DeviceBuffer>* buffer) {
    return CheckStatus(backend.Allocate(bytes, buffer), "Allocate loss input") &&
      CheckStatus(backend.CopyHostToDevice(**buffer, data, bytes),
        "Upload loss input");
  };
  std::vector<float> actual(kLossOffset + transform_count + 11, kGuard);
  if (!upload(coefficients.data(), coefficients.size() * sizeof(float), &input) ||
      !upload(mask.data(), mask.size() * sizeof(float), &device_mask) ||
      !upload(candidates.data(), candidates.size() * sizeof(candidates[0]),
        &descriptors) ||
      !upload(actual.data(), actual.size() * sizeof(float), &losses) ||
      !CheckStatus(backend.Allocate(coefficients.size() * sizeof(float), &pixels),
        "Allocate independent inverse pixels")) {
    return false;
  }
  const auto inverse = gjxl::TransformBatch{
    .strategy = strategy, .input = input.get(), .output = pixels.get(),
    .transform_count = transform_count};
  std::unique_ptr<gjxl::GpuSubmission> submission;
  std::vector<float> reconstructed(coefficients.size());
  if (!CheckStatus(backend.InverseTransform(inverse, &submission),
        "Submit independent inverse for loss") || submission == nullptr ||
      !CheckStatus(submission->Wait(), "Wait for loss reference inverse") ||
      !CheckStatus(backend.CopyDeviceToHost(*pixels, reconstructed.data(),
        reconstructed.size() * sizeof(float)), "Download loss reference pixels")) {
    return false;
  }
  auto* cuda_input = dynamic_cast<CudaBuffer*>(input.get());
  auto* cuda_mask = dynamic_cast<CudaBuffer*>(device_mask.get());
  auto* cuda_descriptors = dynamic_cast<CudaBuffer*>(descriptors.get());
  auto* cuda_losses = dynamic_cast<CudaBuffer*>(losses.get());
  if (!cuda_input || !cuda_mask || !cuda_descriptors || !cuda_losses) return false;
  gjxl::cuda_internal::ScopedCudaDevice device(cuda_input->state()->ordinal);
  if (!CheckStatus(CudaRuntimeStatus(device.status(), "Select loss-test device"),
        "Select loss-test device")) return false;
  const gjxl::cuda_internal::CudaAcStrategyBatchParams params{
    .pixel_width = kWidth, .pixel_height = kHeight,
    .pixel_mask_row_stride = kStride, .candidate_count = candidate_count,
    .coefficient_count = static_cast<uint32_t>(count),
    .transform_width = static_cast<uint32_t>(extent.width),
    .transform_height = static_cast<uint32_t>(extent.height)};
  const auto stream = cuda_input->state()->stream;
  constexpr float kOffsets[3] = {12.0f, 0.0f, 4.0f};
  // One valid batch and four invalid-mask batches. The bad values occupy
  // different reduction lanes, including the final pixel of corner tiles.
  const std::array bad_masks = {0.0f, -1.0f, nan,
    std::numeric_limits<float>::infinity()};
  for (size_t mode = 0; mode <= bad_masks.size(); ++mode) {
    mask = valid_mask;
    if (mode != 0) {
      const size_t x = mode == 1 ? 0 : mode == 2 ? kWidth - 1 :
        mode == 3 ? extent.width / 2 : extent.width - 1;
      const size_t y = mode == 1 || mode == 4 ? 0 : mode == 2 ? kHeight - 1 :
        extent.height / 2;
      mask[kMaskOffset + y * kStride + x] = bad_masks[mode - 1];
    }
    std::fill(actual.begin(), actual.end(), kGuard);
    if (!CheckStatus(backend.CopyHostToDevice(*device_mask, mask.data(),
          mask.size() * sizeof(float)), "Upload loss mask variant") ||
        !CheckStatus(backend.CopyHostToDevice(*losses, actual.data(),
          actual.size() * sizeof(float)), "Reset loss output guards") ||
        !CheckStatus(CudaRuntimeStatus(
          gjxl::cuda_internal::LaunchCudaAcStrategyInverseLoss(
            static_cast<const float*>(cuda_input->pointer()),
            static_cast<const float*>(cuda_mask->pointer()) + kMaskOffset,
            cuda_descriptors->pointer(),
            static_cast<float*>(cuda_losses->pointer()) + kLossOffset,
            params, stream), "Launch fused inverse loss"),
          "Launch fused inverse loss") ||
        !CheckStatus(CudaRuntimeStatus(cudaStreamSynchronize(stream),
          "Wait for fused inverse loss"), "Wait for fused inverse loss") ||
        !CheckStatus(backend.CopyDeviceToHost(*losses, actual.data(),
          actual.size() * sizeof(float)), "Download fused inverse loss")) {
      return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
      if ((i < kLossOffset || i >= kLossOffset + transform_count) &&
          actual[i] != kGuard) {
        std::cerr << "Fused inverse loss overwrote an output guard\n";
        return false;
      }
    }
    for (size_t t = 0; t < transform_count; ++t) {
      const auto candidate = candidates[t / 3];
      const bool fits = candidate.block_x <= (kWidth - extent.width) / 8 &&
        candidate.block_y <= (kHeight - extent.height) / 8;
      std::vector<float> reference(count, nan);
      if (fits) {
        for (size_t i = 0; i < count; ++i) {
          const float value = mask[kMaskOffset +
            (candidate.block_y * 8 + i / extent.width) * kStride +
            candidate.block_x * 8 + i % extent.width];
          float weighted = (value + kOffsets[t % 3]) * reconstructed[t * count + i];
          weighted *= weighted;
          weighted *= weighted;
          weighted *= weighted;
          reference[i] = std::isfinite(value) && value > 0.0f ? weighted : nan;
        }
      }
      // Independent FP32 halving tree over materialized inverse pixels.
      // Require exact sums; a final cost/quality tolerance could hide errors.
      for (size_t stride = count / 2; stride != 0; stride /= 2) {
        for (size_t i = 0; i < stride; ++i) reference[i] += reference[i + stride];
      }
      const float result = actual[kLossOffset + t];
      if (!(std::isnan(reference[0]) && std::isnan(result)) &&
          std::bit_cast<uint32_t>(reference[0]) != std::bit_cast<uint32_t>(result)) {
        std::cerr << "Fused inverse loss differs: " << extent.width << 'x'
                  << extent.height << " candidates=" << candidate_count
                  << " mask=" << mode << " transform=" << t
                  << " actual=" << result << " expected=" << reference[0] << '\n';
        return false;
      }
    }
  }
  return true;
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
  // Injecting a completion error still launches a real transform. Its input
  // must be initialized even though this test discards the numerical result.
  constexpr std::array<float, 64> pixels{};
  if (!CheckStatus(backend.CopyHostToDevice(*input, pixels.data(), kBytes),
        "Initialize CUDA failure-injection input")) {
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
    // Include full and partial blocks around every packed-transform size.
    for (const size_t count :
         {1u, 2u, 3u, 4u, 7u, 8u, 9u, 15u, 16u, 17u, 19u, 31u, 32u, 33u}) {
      if (!CheckTransform(*backend, strategy, &random, count)) {
        std::cerr << "CUDA DCT batch size: " << count << '\n';
        return EXIT_FAILURE;
      }
    }
  }
  for (gjxl::AcStrategyType strategy : kStrategies) {
    const auto extent = gjxl::GetAcStrategyInfo(strategy)->pixel_extent();
    if (extent.width > 32 || extent.height > 32) continue;
    for (const uint32_t count : {1u, 2u, 3u, 5u, 7u, 8u, 9u, 10u, 11u, 16u, 17u, 33u}) {
      if (!CheckAcStrategyInverseLoss(*backend, strategy, &random, count)) {
        return EXIT_FAILURE;
      }
    }
  }
  std::cout << "All CUDA backend tests passed on " << backend->name() << ".\n";
  return EXIT_SUCCESS;
}
