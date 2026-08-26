// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "gpu/backend.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/primitives.h"
#include "gpu/scratch.h"
#include "gpu_test_utils.h"

namespace {

bool CheckStatus(gjxl::Status status, std::string_view operation) {
  if (status.ok()) return true;
  std::cerr << operation << " failed: " << status.message() << '\n';
  return false;
}

std::vector<float> MakeInput(gjxl::Extent2D extent, uint32_t seed) {
  std::vector<float> input(extent.width * extent.height);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> distribution(-3.0f, 2.0f);
  for (float& value : input) value = distribution(rng);
  return input;
}

std::vector<float> ReferenceAffine(
  std::span<const float> input,
  float scale,
  float bias) {

  std::vector<float> output(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    output[index] = static_cast<float>(
      static_cast<double>(input[index]) * scale + bias);
  }
  return output;
}

std::vector<float> ReferenceConvolution(
  std::span<const float> input,
  gjxl::Extent2D extent,
  std::span<const float> kernel) {

  const int radius = static_cast<int>(kernel.size() / 2);
  std::vector<float> horizontal(input.size());
  std::vector<float> output(input.size());
  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      double sum = 0.0;
      double weight_sum = 0.0;
      for (int delta = -radius; delta <= radius; ++delta) {
        const int source_x = static_cast<int>(x) + delta;
        if (source_x < 0 || source_x >= static_cast<int>(extent.width)) {
          continue;
        }
        const double weight = kernel[delta + radius];
        sum += input[y * extent.width + static_cast<size_t>(source_x)] *
               weight;
        weight_sum += weight;
      }
      horizontal[y * extent.width + x] =
        static_cast<float>(sum / weight_sum);
    }
  }
  for (size_t y = 0; y < extent.height; ++y) {
    for (size_t x = 0; x < extent.width; ++x) {
      double sum = 0.0;
      double weight_sum = 0.0;
      for (int delta = -radius; delta <= radius; ++delta) {
        const int source_y = static_cast<int>(y) + delta;
        if (source_y < 0 || source_y >= static_cast<int>(extent.height)) {
          continue;
        }
        const double weight = kernel[delta + radius];
        sum += horizontal[static_cast<size_t>(source_y) * extent.width + x] *
               weight;
        weight_sum += weight;
      }
      output[y * extent.width + x] =
        static_cast<float>(sum / weight_sum);
    }
  }
  return output;
}

bool Compare(
  std::string_view label,
  std::span<const float> actual,
  std::span<const float> expected,
  float absolute_tolerance,
  float relative_tolerance) {

  if (actual.size() != expected.size()) return false;
  float maximum_error = 0.0f;
  for (size_t index = 0; index < actual.size(); ++index) {
    const float error = std::abs(actual[index] - expected[index]);
    maximum_error = std::max(maximum_error, error);
    const float allowed = absolute_tolerance +
      relative_tolerance * std::abs(expected[index]);
    if (!std::isfinite(actual[index]) || error > allowed) {
      std::cerr << label << " mismatch at " << index << ": expected "
                << expected[index] << ", got " << actual[index]
                << ", tolerance " << allowed << '\n';
      return false;
    }
  }
  std::cout << label << " max error: " << maximum_error << '\n';
  return true;
}

bool PreparePlane(
  gjxl::GpuBackend& gpu,
  gjxl::Extent2D extent,
  size_t stride,
  gjxl::test::GuardedDevicePlane* plane) {

  return CheckStatus(
    plane->Prepare(gpu, extent, stride), "guarded plane allocation");
}

bool SubmitAndWait(
  gjxl::GpuImagePrimitives& primitives,
  std::span<const gjxl::ImagePrimitiveCommand> commands,
  std::string_view operation) {

  std::unique_ptr<gjxl::GpuSubmission> submission;
  if (!CheckStatus(
        primitives.SubmitImagePrimitiveSequence(commands, &submission),
        std::string(operation) + " submission") ||
      submission == nullptr) {
    std::cerr << operation << " did not return a submission handle\n";
    return false;
  }
  return CheckStatus(
    submission->Wait(), std::string(operation) + " completion");
}

bool RunConvolutionCase(
  gjxl::GpuBackend& gpu,
  gjxl::GpuImagePrimitives& primitives,
  size_t kernel_size) {

  constexpr gjxl::Extent2D kExtent{17, 11};
  const std::vector<float> input_values = MakeInput(kExtent, 100 + kernel_size);
  std::vector<float> kernel(kernel_size);
  const int radius = static_cast<int>(kernel_size / 2);
  for (size_t index = 0; index < kernel.size(); ++index) {
    kernel[index] = 1.0f /
      static_cast<float>(1 + std::abs(static_cast<int>(index) - radius));
  }

  gjxl::test::GuardedDevicePlane input;
  gjxl::test::GuardedDevicePlane weights;
  gjxl::test::GuardedDevicePlane intermediate;
  gjxl::test::GuardedDevicePlane output;
  if (!PreparePlane(gpu, kExtent, 23, &input) ||
      !PreparePlane(gpu, {kernel_size, 1}, kernel_size + 3, &weights) ||
      !PreparePlane(gpu, kExtent, 21, &intermediate) ||
      !PreparePlane(gpu, kExtent, 25, &output)) {
    return false;
  }
  input.SetLogical(input_values);
  weights.SetLogical(kernel);
  intermediate.PoisonLogical();
  output.PoisonLogical();
  if (!CheckStatus(input.Upload(), "convolution input upload") ||
      !CheckStatus(weights.Upload(), "convolution kernel upload") ||
      !CheckStatus(intermediate.Upload(), "convolution scratch poison") ||
      !CheckStatus(output.Upload(), "convolution output poison")) {
    return false;
  }
  const gjxl::ImagePrimitiveCommand command =
    gjxl::SeparableConvolutionCommand{
      input.ConstView(), weights.ConstView(), intermediate.View(),
      output.View()};
  if (!SubmitAndWait(
        primitives,
        std::span<const gjxl::ImagePrimitiveCommand>(&command, 1),
        "convolution") ||
      !CheckStatus(output.Download(), "convolution output download") ||
      !CheckStatus(intermediate.Download(), "convolution scratch download")) {
    return false;
  }
  const std::vector<float> expected =
    ReferenceConvolution(input_values, kExtent, kernel);
  return Compare(
           "convolution", output.Logical(), expected, 2e-5f, 2e-5f) &&
         output.GuardsIntact() && intermediate.GuardsIntact();
}

bool CheckPrimitiveChain(
  gjxl::GpuBackend& gpu,
  gjxl::GpuImagePrimitives& primitives) {
  constexpr gjxl::Extent2D kExtent{17, 11};
  constexpr float kScale = -0.75f;
  constexpr float kBias = 0.2f;
  const std::vector<float> input_values = MakeInput(kExtent, 12345);
  const std::array<float, 5> kernel{1.0f, 4.0f, 6.0f, 4.0f, 1.0f};

  gjxl::test::GuardedDevicePlane input;
  gjxl::test::GuardedDevicePlane affine;
  gjxl::test::GuardedDevicePlane weights;
  gjxl::test::GuardedDevicePlane convolved;
  gjxl::test::GuardedDevicePlane maximum;
  if (!PreparePlane(gpu, kExtent, 23, &input) ||
      !PreparePlane(gpu, kExtent, 21, &affine) ||
      !PreparePlane(gpu, {kernel.size(), 1}, 7, &weights) ||
      !PreparePlane(gpu, kExtent, 25, &convolved) ||
      !PreparePlane(gpu, {1, 1}, 3, &maximum)) {
    return false;
  }
  input.SetLogical(input_values);
  weights.SetLogical(kernel);
  affine.PoisonLogical();
  convolved.PoisonLogical();
  maximum.PoisonLogical();
  if (!CheckStatus(input.Upload(), "chain input upload") ||
      !CheckStatus(weights.Upload(), "chain kernel upload") ||
      !CheckStatus(affine.Upload(), "chain affine poison") ||
      !CheckStatus(convolved.Upload(), "chain convolution poison") ||
      !CheckStatus(maximum.Upload(), "chain maximum poison")) {
    return false;
  }

  gjxl::DeviceScratchArena scratch;
  if (!CheckStatus(scratch.Prepare(gpu, 4096), "chain scratch preparation")) {
    return false;
  }
  gjxl::DevicePlaneView convolution_scratch;
  gjxl::DevicePlaneView reduction_a;
  gjxl::DevicePlaneView reduction_b;
  if (!CheckStatus(scratch.AllocatePlane(
        gjxl::DeviceElementType::kF32, kExtent, 23, 256,
        &convolution_scratch), "convolution scratch layout") ||
      !CheckStatus(scratch.AllocatePlane(
        gjxl::DeviceElementType::kF32, {1, 1}, 1, 256,
        &reduction_a), "reduction A layout") ||
      !CheckStatus(scratch.AllocatePlane(
        gjxl::DeviceElementType::kF32, {1, 1}, 1, 256,
        &reduction_b), "reduction B layout")) {
    return false;
  }

  const std::array<gjxl::ImagePrimitiveCommand, 3> commands{{
    gjxl::PointwiseAffineCommand{
      input.ConstView(), affine.View(), kScale, kBias},
    gjxl::SeparableConvolutionCommand{
      affine.ConstView(), weights.ConstView(), convolution_scratch,
      convolved.View()},
    gjxl::MaximumReductionCommand{
      convolved.ConstView(), reduction_a, reduction_b, maximum.View()},
  }};

  if (!SubmitAndWait(primitives, commands, "chain warmup")) {
    return false;
  }
  const gjxl::GpuBackendStats before = gpu.stats();
  for (size_t repeat = 0; repeat < 3; ++repeat) {
    affine.PoisonLogical();
    convolved.PoisonLogical();
    maximum.PoisonLogical();
    if (!CheckStatus(affine.Upload(), "measured affine poison") ||
        !CheckStatus(convolved.Upload(), "measured convolution poison") ||
        !CheckStatus(maximum.Upload(), "measured maximum poison") ||
        !SubmitAndWait(primitives, commands, "measured chain")) {
      return false;
    }
    const gjxl::GpuBackendStats after = gpu.stats();
    if (after.successful_allocations != before.successful_allocations ||
        after.committed_submissions != before.committed_submissions +
          repeat + 1) {
      std::cerr << "Primitive chain allocated or used multiple submissions\n";
      return false;
    }
  }
  if (!CheckStatus(affine.Download(), "affine intermediate readback") ||
      !CheckStatus(convolved.Download(), "convolution output readback") ||
      !CheckStatus(maximum.Download(), "maximum readback")) {
    return false;
  }
  const std::vector<float> expected_affine =
    ReferenceAffine(input_values, kScale, kBias);
  const std::vector<float> expected_convolution =
    ReferenceConvolution(expected_affine, kExtent, kernel);
  const std::vector<float> actual_affine = affine.Logical();
  const std::vector<float> actual_convolution = convolved.Logical();
  const std::vector<float> actual_maximum = maximum.Logical();
  const float expected_actual_maximum =
    *std::max_element(actual_convolution.begin(), actual_convolution.end());
  if (!Compare("affine", actual_affine, expected_affine, 2e-6f, 2e-6f) ||
      !Compare("chain convolution", actual_convolution,
               expected_convolution, 2e-5f, 2e-5f) ||
      actual_maximum.size() != 1 ||
      std::bit_cast<uint32_t>(actual_maximum[0]) !=
        std::bit_cast<uint32_t>(expected_actual_maximum) ||
      !affine.GuardsIntact() || !convolved.GuardsIntact() ||
      !maximum.GuardsIntact()) {
    std::cerr << "Primitive chain output or guards are invalid\n";
    return false;
  }
  std::cout << "primitive chain scratch capacity=" << scratch.capacity_bytes()
            << " peak_layout=" << scratch.peak_layout_bytes() << '\n';
  return true;
}

bool CheckLargeMaximum(
  gjxl::GpuBackend& gpu,
  gjxl::GpuImagePrimitives& primitives) {
  constexpr gjxl::Extent2D kExtent{37, 29};
  std::vector<float> input_values = MakeInput(kExtent, 991);
  for (float& value : input_values) {
    value = -std::abs(value) - 0.5f;
  }
  input_values.front() = -500.0f;
  input_values.back() = -0.125f;
  input_values[input_values.size() / 2] = -0.25f;
  const float expected =
    *std::max_element(input_values.begin(), input_values.end());
  gjxl::test::GuardedDevicePlane input;
  gjxl::test::GuardedDevicePlane scratch_a;
  gjxl::test::GuardedDevicePlane scratch_b;
  gjxl::test::GuardedDevicePlane output;
  constexpr size_t kPartials =
    (kExtent.width * kExtent.height + 255) / 256;
  if (!PreparePlane(gpu, kExtent, 41, &input) ||
      !PreparePlane(gpu, {kPartials, 1}, kPartials + 2, &scratch_a) ||
      !PreparePlane(gpu, {kPartials, 1}, kPartials + 3, &scratch_b) ||
      !PreparePlane(gpu, {1, 1}, 2, &output)) {
    return false;
  }
  input.SetLogical(input_values);
  scratch_a.PoisonLogical();
  scratch_b.PoisonLogical();
  output.PoisonLogical();
  if (!CheckStatus(input.Upload(), "large maximum input upload") ||
      !CheckStatus(scratch_a.Upload(), "large maximum scratch A poison") ||
      !CheckStatus(scratch_b.Upload(), "large maximum scratch B poison") ||
      !CheckStatus(output.Upload(), "large maximum output poison")) {
    return false;
  }
  const gjxl::ImagePrimitiveCommand command = gjxl::MaximumReductionCommand{
    input.ConstView(), scratch_a.View(), scratch_b.View(), output.View()};
  if (!SubmitAndWait(
        primitives,
        std::span<const gjxl::ImagePrimitiveCommand>(&command, 1),
        "large maximum") ||
      !CheckStatus(output.Download(), "large maximum output download") ||
      !CheckStatus(scratch_a.Download(), "large maximum scratch A download") ||
      !CheckStatus(scratch_b.Download(), "large maximum scratch B download")) {
    return false;
  }
  const std::vector<float> actual = output.Logical();
  return actual.size() == 1 &&
         std::bit_cast<uint32_t>(actual[0]) == std::bit_cast<uint32_t>(expected) &&
         output.GuardsIntact() && scratch_a.GuardsIntact() &&
         scratch_b.GuardsIntact();
}

bool CheckInPlaceAndConstant(
  gjxl::GpuBackend& gpu,
  gjxl::GpuImagePrimitives& primitives) {
  constexpr gjxl::Extent2D kExtent{17, 11};
  constexpr float kScale = 0.5f;
  constexpr float kBias = 0.25f;
  std::vector<float> input_values(kExtent.width * kExtent.height, 2.5f);
  std::array<float, 33> kernel{};
  for (size_t index = 0; index < kernel.size(); ++index) {
    kernel[index] = 1.0f /
      static_cast<float>(1 + std::abs(static_cast<int>(index) - 16));
  }
  gjxl::test::GuardedDevicePlane in_place;
  gjxl::test::GuardedDevicePlane weights;
  gjxl::test::GuardedDevicePlane intermediate;
  if (!PreparePlane(gpu, kExtent, 23, &in_place) ||
      !PreparePlane(gpu, {kernel.size(), 1}, 35, &weights) ||
      !PreparePlane(gpu, kExtent, 19, &intermediate)) {
    return false;
  }
  in_place.SetLogical(input_values);
  weights.SetLogical(kernel);
  intermediate.PoisonLogical();
  if (!CheckStatus(in_place.Upload(), "in-place input upload") ||
      !CheckStatus(weights.Upload(), "in-place kernel upload") ||
      !CheckStatus(intermediate.Upload(), "in-place scratch poison")) {
    return false;
  }
  const std::array<gjxl::ImagePrimitiveCommand, 2> commands{{
    gjxl::PointwiseAffineCommand{
      in_place.ConstView(), in_place.View(), kScale, kBias},
    gjxl::SeparableConvolutionCommand{
      in_place.ConstView(), weights.ConstView(), intermediate.View(),
      in_place.View()},
  }};
  if (!SubmitAndWait(primitives, commands, "in-place primitive") ||
      !CheckStatus(in_place.Download(), "in-place output download") ||
      !CheckStatus(intermediate.Download(), "in-place scratch download")) {
    return false;
  }
  const std::vector<float> affine =
    ReferenceAffine(input_values, kScale, kBias);
  const std::vector<float> expected =
    ReferenceConvolution(affine, kExtent, kernel);
  return Compare(
           "in-place constant convolution", in_place.Logical(), expected,
           2e-5f, 2e-5f) &&
         in_place.GuardsIntact() && intermediate.GuardsIntact();
}

bool CheckValidation(
  gjxl::GpuBackend& gpu,
  gjxl::GpuImagePrimitives& primitives) {
  constexpr gjxl::Extent2D kExtent{17, 11};
  gjxl::test::GuardedDevicePlane input;
  gjxl::test::GuardedDevicePlane output;
  if (!PreparePlane(gpu, kExtent, 23, &input) ||
      !PreparePlane(gpu, kExtent, 23, &output)) {
    return false;
  }

  const gjxl::ImagePrimitiveCommand valid = gjxl::PointwiseAffineCommand{
    input.ConstView(), output.View(), 1.0f, 0.0f};
  std::unique_ptr<gjxl::GpuSubmission> submission;
  if (!primitives.SubmitImagePrimitiveSequence(
        std::span<const gjxl::ImagePrimitiveCommand>(&valid, 1),
        &submission).ok() ||
      submission == nullptr || !submission->Wait().ok()) {
    std::cerr << "Valid primitive did not return a usable submission\n";
    return false;
  }
  const uint64_t submissions = gpu.stats().committed_submissions;
  gjxl::PointwiseAffineCommand invalid_stride{
    input.ConstView(), output.View(), 1.0f, 0.0f};
  invalid_stride.output.row_stride = 16;
  const gjxl::ImagePrimitiveCommand first = invalid_stride;
  if (primitives.SubmitImagePrimitiveSequence(
        std::span<const gjxl::ImagePrimitiveCommand>(&first, 1),
        &submission).code() != gjxl::StatusCode::kInvalidArgument ||
      submission != nullptr) {
    std::cerr << "Invalid primitive stride was accepted\n";
    return false;
  }

  if (primitives.SubmitImagePrimitiveSequence(
        std::span<const gjxl::ImagePrimitiveCommand>(&valid, 1),
        nullptr).code() != gjxl::StatusCode::kInvalidArgument ||
      gpu.stats().committed_submissions != submissions) {
    std::cerr << "Null primitive submission output committed work\n";
    return false;
  }

  gjxl::DevicePlaneView partial = input.View();
  partial.offset_bytes += sizeof(float);
  const gjxl::ImagePrimitiveCommand second = gjxl::PointwiseAffineCommand{
    input.ConstView(), partial, 1.0f, 0.0f};
  if (primitives.SubmitImagePrimitiveSequence(
        std::span<const gjxl::ImagePrimitiveCommand>(&second, 1),
        &submission).code() != gjxl::StatusCode::kInvalidArgument ||
      submission != nullptr) {
    std::cerr << "Partially overlapping primitive planes were accepted\n";
    return false;
  }

  gjxl::test::GuardedDevicePlane large_input;
  gjxl::test::GuardedDevicePlane short_scratch_a;
  gjxl::test::GuardedDevicePlane short_scratch_b;
  gjxl::test::GuardedDevicePlane scalar_output;
  if (!PreparePlane(gpu, {37, 29}, 41, &large_input) ||
      !PreparePlane(gpu, {1, 1}, 1, &short_scratch_a) ||
      !PreparePlane(gpu, {1, 1}, 1, &short_scratch_b) ||
      !PreparePlane(gpu, {1, 1}, 1, &scalar_output)) {
    return false;
  }
  const gjxl::ImagePrimitiveCommand short_scratch =
    gjxl::MaximumReductionCommand{
      large_input.ConstView(), short_scratch_a.View(), short_scratch_b.View(),
      scalar_output.View()};
  if (primitives.SubmitImagePrimitiveSequence(
        std::span<const gjxl::ImagePrimitiveCommand>(&short_scratch, 1),
        &submission).code() != gjxl::StatusCode::kInvalidArgument ||
      submission != nullptr ||
      primitives.SubmitImagePrimitiveSequence({}, &submission).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      submission != nullptr) {
    std::cerr << "Insufficient reduction scratch or empty sequence failed\n";
    return false;
  }

  std::unique_ptr<gjxl::GpuBackend> other;
  if (!CheckStatus(gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &other),
                   "second backend creation")) {
    return false;
  }
  gjxl::test::GuardedDevicePlane foreign;
  if (!PreparePlane(*other, kExtent, 23, &foreign)) return false;
  const gjxl::ImagePrimitiveCommand third = gjxl::PointwiseAffineCommand{
    foreign.ConstView(), output.View(), 1.0f, 0.0f};
  if (primitives.SubmitImagePrimitiveSequence(
        std::span<const gjxl::ImagePrimitiveCommand>(&third, 1),
        &submission).code() != gjxl::StatusCode::kInvalidArgument ||
      submission != nullptr ||
      gpu.stats().committed_submissions != submissions) {
    std::cerr << "Foreign primitive buffer submitted work\n";
    return false;
  }
  return true;
}

bool CheckFailureStatuses() {
  const auto check = [](gjxl::MetalBackendOptions options,
                        gjxl::StatusCode expected_submit,
                        gjxl::StatusCode expected_wait) {
    std::unique_ptr<gjxl::GpuBackend> gpu;
    if (!gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, options, &gpu).ok()) {
      return false;
    }
    gjxl::GpuImagePrimitives* primitives =
      gjxl::QueryGpuImagePrimitives(*gpu);
    if (primitives == nullptr) return false;
    gjxl::test::GuardedDevicePlane input;
    gjxl::test::GuardedDevicePlane output;
    if (!input.Prepare(*gpu, {3, 2}, 5).ok() ||
        !output.Prepare(*gpu, {3, 2}, 5).ok()) {
      return false;
    }
    const gjxl::ImagePrimitiveCommand command = gjxl::PointwiseAffineCommand{
      input.ConstView(), output.View(), 1.0f, 0.0f};
    std::unique_ptr<gjxl::GpuSubmission> submission;
    const gjxl::Status submitted = primitives->SubmitImagePrimitiveSequence(
      std::span<const gjxl::ImagePrimitiveCommand>(&command, 1),
      &submission);
    if (submitted.code() != expected_submit) return false;
    if (!submitted.ok()) {
      return submission == nullptr &&
             gpu->stats().committed_submissions == 0;
    }
    if (submission == nullptr) return false;
    std::unique_ptr<gjxl::GpuSubmission> second_submission;
    if (!primitives->SubmitImagePrimitiveSequence(
          std::span<const gjxl::ImagePrimitiveCommand>(&command, 1),
          &second_submission).ok() ||
        second_submission == nullptr) {
      return false;
    }

    constexpr size_t kWaiterCount = 4;
    std::array<gjxl::Status, kWaiterCount> concurrent_results;
    std::array<std::thread, kWaiterCount> waiters;
    for (size_t index = 0; index < kWaiterCount; ++index) {
      waiters[index] = std::thread(
        [&submission, &concurrent_results, index] {
          concurrent_results[index] = submission->Wait();
        });
    }
    for (std::thread& waiter : waiters) waiter.join();

    const gjxl::Status second_first = second_submission->Wait();
    const gjxl::Status second_again = second_submission->Wait();
    for (const gjxl::Status& result : concurrent_results) {
      if (result.code() != expected_wait ||
          result.message() != concurrent_results.front().message()) {
        return false;
      }
    }
    return second_first.code() == expected_wait &&
           second_again.code() == second_first.code() &&
           second_again.message() == second_first.message() &&
           gpu->stats().committed_submissions == 2;
  };

  gjxl::MetalBackendOptions submit_failure;
  submit_failure.test_fail_submission = true;
  gjxl::MetalBackendOptions completion_failure;
  completion_failure.test_fail_completion = true;
  return check(
           submit_failure,
           gjxl::StatusCode::kSubmissionFailed,
           gjxl::StatusCode::kOk) &&
         check(
           completion_failure,
           gjxl::StatusCode::kOk,
           gjxl::StatusCode::kDeviceError);
}

bool CheckSubmissionOutlivesBackend() {
  std::unique_ptr<gjxl::GpuSubmission> submission;
  {
    std::unique_ptr<gjxl::GpuBackend> gpu;
    if (!gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu).ok()) {
      return false;
    }
    gjxl::GpuImagePrimitives* primitives =
      gjxl::QueryGpuImagePrimitives(*gpu);
    if (primitives == nullptr) return false;
    gjxl::test::GuardedDevicePlane input;
    gjxl::test::GuardedDevicePlane output;
    if (!input.Prepare(*gpu, {3, 2}, 5).ok() ||
        !output.Prepare(*gpu, {3, 2}, 5).ok()) {
      return false;
    }
    const gjxl::ImagePrimitiveCommand command =
      gjxl::PointwiseAffineCommand{
        input.ConstView(), output.View(), 1.0f, 0.0f};
    if (!primitives->SubmitImagePrimitiveSequence(
          std::span<const gjxl::ImagePrimitiveCommand>(&command, 1),
          &submission).ok() ||
        submission == nullptr) {
      return false;
    }
    gpu.reset();
  }
  return submission->Wait().ok() && submission->Wait().ok();
}

bool CheckConcurrentSubmissions(
  gjxl::GpuBackend& gpu,
  gjxl::GpuImagePrimitives& primitives) {

  constexpr size_t kSubmissionCount = 4;
  constexpr gjxl::Extent2D kExtent{17, 11};
  std::array<gjxl::test::GuardedDevicePlane, kSubmissionCount> inputs;
  std::array<gjxl::test::GuardedDevicePlane, kSubmissionCount> outputs;
  std::array<gjxl::ImagePrimitiveCommand, kSubmissionCount> commands;
  for (size_t index = 0; index < kSubmissionCount; ++index) {
    if (!inputs[index].Prepare(gpu, kExtent, 23).ok() ||
        !outputs[index].Prepare(gpu, kExtent, 25).ok()) {
      return false;
    }
    inputs[index].SetLogical(MakeInput(kExtent, 900 +
      static_cast<uint32_t>(index)));
    outputs[index].PoisonLogical();
    if (!inputs[index].Upload().ok() || !outputs[index].Upload().ok()) {
      return false;
    }
    commands[index] = gjxl::PointwiseAffineCommand{
      inputs[index].ConstView(), outputs[index].View(), 0.5f, 0.25f};
  }

  const uint64_t before = gpu.stats().committed_submissions;
  std::array<std::unique_ptr<gjxl::GpuSubmission>, kSubmissionCount>
    submissions;
  std::array<gjxl::Status, kSubmissionCount> statuses;
  std::array<std::thread, kSubmissionCount> threads;
  for (size_t index = 0; index < kSubmissionCount; ++index) {
    threads[index] = std::thread([&, index] {
      statuses[index] = primitives.SubmitImagePrimitiveSequence(
        std::span<const gjxl::ImagePrimitiveCommand>(&commands[index], 1),
        &submissions[index]);
    });
  }
  for (std::thread& thread : threads) thread.join();
  for (size_t index = 0; index < kSubmissionCount; ++index) {
    if (!statuses[index].ok() || submissions[index] == nullptr ||
        !submissions[index]->Wait().ok() ||
        !outputs[index].Download().ok() || !outputs[index].GuardsIntact()) {
      return false;
    }
    const std::vector<float> expected = ReferenceAffine(
      inputs[index].Logical(), 0.5f, 0.25f);
    if (!Compare(
          "concurrent affine", outputs[index].Logical(), expected,
          2e-6f, 2e-6f)) {
      return false;
    }
  }
  return gpu.stats().committed_submissions == before + kSubmissionCount;
}

}  // namespace

int main() {
  std::unique_ptr<gjxl::GpuBackend> gpu;
  if (!CheckStatus(
        gjxl::CreateMetalBackend(GJXL_METALLIB_PATH, &gpu),
        "Metal backend creation")) {
    return EXIT_FAILURE;
  }
  gjxl::GpuImagePrimitives* primitives =
    gjxl::QueryGpuImagePrimitives(*gpu);
  if (primitives == nullptr ||
      !CheckPrimitiveChain(*gpu, *primitives) ||
      !RunConvolutionCase(*gpu, *primitives, 1) ||
      !RunConvolutionCase(*gpu, *primitives, 33) ||
      !CheckLargeMaximum(*gpu, *primitives) ||
      !CheckInPlaceAndConstant(*gpu, *primitives) ||
      !CheckValidation(*gpu, *primitives) ||
      !CheckConcurrentSubmissions(*gpu, *primitives) ||
      !CheckFailureStatuses() ||
      !CheckSubmissionOutlivesBackend()) {
    return EXIT_FAILURE;
  }
  std::cout << "All Metal primitive infrastructure tests passed.\n";
  return EXIT_SUCCESS;
}
