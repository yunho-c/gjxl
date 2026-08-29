// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/ops/gaborish.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "codec/gaborish_internal.h"
#include "gpu/ops/gaborish_profile_internal.h"
#include "gpu/ops/primitives.h"

namespace gjxl {

static Status ApplyGaborishInverseGpuImpl(
  GpuBackend& gpu,
  ConstImage3FView input,
  std::array<float, 3> multipliers,
  Image3FView output,
  gpu_profile_internal::GpuProfilingSession* profiling_session) {

  if (!input.valid() || !output.valid() ||
      input.extent() != output.extent()) {
    return Status::InvalidArgument(
      "GPU Gaborish images are invalid or differently sized");
  }
  for (float multiplier : multipliers) {
    if (!std::isfinite(multiplier)) {
      return Status::InvalidArgument(
        "GPU Gaborish multipliers must be finite");
    }
  }
  GpuImagePrimitives* primitives = QueryGpuImagePrimitives(gpu);
  if (primitives == nullptr) {
    return Status::Unavailable(
      "GPU Gaborish requires image primitive support");
  }
  auto* primitive_profiler = profiling_session == nullptr
    ? nullptr
    : dynamic_cast<gpu_profile_internal::GpuImagePrimitivesProfiler*>(&gpu);
  auto* submission_profiler = profiling_session == nullptr
    ? nullptr
    : dynamic_cast<gpu_profile_internal::GpuSubmissionProfiler*>(&gpu);
  if (profiling_session != nullptr &&
      (primitive_profiler == nullptr || submission_profiler == nullptr)) {
    return Status::Unavailable(
      "GPU Gaborish profiling is unavailable");
  }

  size_t pixel_count = 0;
  if (!input.extent().try_area(&pixel_count) ||
      pixel_count > std::numeric_limits<size_t>::max() / 6 ||
      pixel_count * 6 > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return Status::InvalidArgument(
      "GPU Gaborish image dimensions are too large");
  }
  const size_t plane_bytes = pixel_count * sizeof(float);
  const size_t image_values = 3 * pixel_count;
  const size_t allocation_bytes = 6 * plane_bytes;

  try {
    const auto preparation_begin = profiling_session == nullptr
      ? gpu_profile_internal::GpuProfilingSession::TimePoint{}
      : gpu_profile_internal::GpuProfilingSession::BeginWallStage();
    std::vector<float> packed_input(image_values);
    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t y = 0; y < input.height(); ++y) {
        const float* source = input.plane[channel].Row(y);
        float* destination = packed_input.data() +
          channel * pixel_count + y * input.width();
        std::copy_n(source, input.width(), destination);
        if (!std::all_of(
              destination,
              destination + input.width(),
              [](float value) { return std::isfinite(value); })) {
          return Status::InvalidArgument(
            "GPU Gaborish input must be finite");
        }
      }
    }

    std::unique_ptr<DeviceBuffer> storage;
    Status status = gpu.Allocate(allocation_bytes, &storage);
    if (!status.ok()) return status;
    if (storage == nullptr) {
      return Status::Internal(
        "GPU Gaborish allocation returned no buffer");
    }
    status = gpu.CopyHostToDevice(
      *storage, packed_input.data(), 3 * plane_bytes);
    if (!status.ok()) return status;

    std::array<ImagePrimitiveCommand, 3> commands;
    for (size_t channel = 0; channel < 3; ++channel) {
      const Symmetric5Weights weights =
        gaborish_internal::GaborishInverseWeights(multipliers[channel]);
      commands[channel] = Symmetric5ConvolutionCommand{
        .input = {
          storage.get(), channel * plane_bytes, DeviceElementType::kF32,
          input.extent(), input.width()},
        .output = {
          storage.get(), (3 + channel) * plane_bytes,
          DeviceElementType::kF32, input.extent(), input.width()},
        .weights = {
          weights.distance0,
          weights.distance1,
          weights.distance2,
          weights.distance4,
          weights.distance8,
          weights.distance5,
        },
      };
    }
    std::unique_ptr<GpuSubmission> submission;
    status = profiling_session == nullptr
      ? primitives->SubmitImagePrimitiveSequence(commands, &submission)
      : primitive_profiler->SubmitImagePrimitiveSequenceProfiled(
          commands, "frontend.preprocessing.gaborish",
          profiling_session->mode(), &submission);
    if (!status.ok()) return status;
    if (submission == nullptr) {
      return Status::Internal(
        "GPU Gaborish submission returned no handle");
    }
    if (profiling_session != nullptr) {
      status = profiling_session->EndWallStage(
        "frontend.preprocessing.gaborish.prepare",
        gpu_profile_internal::GpuWallStageKind::kPreparation,
        preparation_begin);
      if (!status.ok()) return status;
    }
    const auto wait_begin = profiling_session == nullptr
      ? gpu_profile_internal::GpuProfilingSession::TimePoint{}
      : gpu_profile_internal::GpuProfilingSession::BeginWallStage();
    status = submission->Wait();
    if (!status.ok()) return status;
    if (profiling_session != nullptr) {
      status = profiling_session->EndWallStage(
        "frontend.preprocessing.gaborish.wait",
        gpu_profile_internal::GpuWallStageKind::kWait, wait_begin);
      if (!status.ok()) return status;
      gpu_profile_internal::GpuExecutionProfile child_profile;
      status = submission_profiler->ResolveGpuSubmissionProfile(
        *submission, "frontend.preprocessing.gaborish",
        profiling_session->mode(), &child_profile);
      if (status.ok()) {
        status = profiling_session->Append(std::move(child_profile));
      }
      if (!status.ok()) return status;
    }

    const auto readback_begin = profiling_session == nullptr
      ? gpu_profile_internal::GpuProfilingSession::TimePoint{}
      : gpu_profile_internal::GpuProfilingSession::BeginWallStage();
    std::vector<float> filtered(image_values);
    status = gpu.CopyDeviceToHost(
      *storage, filtered.data(), 3 * plane_bytes, 3 * plane_bytes);
    if (!status.ok()) return status;
    if (!std::ranges::all_of(
          filtered,
          [](float value) { return std::isfinite(value); })) {
      return Status::InvalidArgument(
        "GPU Gaborish filtering produced a non-finite result");
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t y = 0; y < output.height(); ++y) {
        std::copy_n(
          filtered.data() + channel * pixel_count + y * output.width(),
          output.width(),
          output.plane[channel].Row(y));
      }
    }
    if (profiling_session != nullptr) {
      status = profiling_session->EndWallStage(
        "frontend.preprocessing.gaborish.readback",
        gpu_profile_internal::GpuWallStageKind::kReadback,
        readback_begin);
      if (!status.ok()) return status;
    }
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate GPU Gaborish storage");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "GPU Gaborish image dimensions are too large");
  }

  return Status::Ok();
}

Status ApplyGaborishInverseGpu(
  GpuBackend& gpu,
  ConstImage3FView input,
  std::array<float, 3> multipliers,
  Image3FView output) {

  return ApplyGaborishInverseGpuImpl(
    gpu, input, multipliers, output, nullptr);
}

Status gpu_profile_internal::ApplyGaborishInverseGpuProfiled(
  GpuBackend& gpu,
  ConstImage3FView input,
  std::array<float, 3> multipliers,
  Image3FView output,
  GpuProfilingSession* profiling_session) {

  if (profiling_session == nullptr) {
    return Status::InvalidArgument(
      "GPU Gaborish profiling session is null");
  }
  return ApplyGaborishInverseGpuImpl(
    gpu, input, multipliers, output, profiling_session);
}

}  // namespace gjxl
