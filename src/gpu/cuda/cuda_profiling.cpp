// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/cuda/cuda_backend_internal.h"

#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace gjxl::cuda_internal {
namespace {
using namespace gpu_profile_internal;
constexpr GpuProfilingCapabilities kCapabilities{
  .timestamp_counter = true, .stage_boundary = true,
  .dispatch_boundary = false,
};
}  // namespace

thread_local CudaProfileCapture* CudaProfileCapture::current_ = nullptr;

CudaProfileCapture::CudaProfileCapture(CudaBackend& backend, std::string_view operation)
  : backend_(backend), operation_(operation),
    session_(GpuProfilingMode::kStage, kCapabilities), previous_(current_) {
  current_ = this;
}

CudaProfileCapture::~CudaProfileCapture() { current_ = previous_; }

CudaProfileCapture* CudaProfileCapture::Current(const CudaBackend& backend) noexcept {
  for (auto* capture = current_; capture != nullptr; capture = capture->previous_) {
    if (&capture->backend_ == &backend) return capture;
  }
  return nullptr;
}

Status CudaProfileCapture::Append(CudaSubmission& submission) {
  GpuExecutionProfile child;
  Status status = backend_.ResolveGpuSubmissionProfile(
    submission, operation_, GpuProfilingMode::kStage, &child);
  if (!status.ok()) return status;
  return session_.Append(std::move(child));
}

Status ValidateCudaProfileRequest(GpuProfilingMode mode,
                                 const GpuExecutionProfile* profile) {
  if (profile == nullptr) return Status::InvalidArgument("CUDA profile output is null");
  if (mode == GpuProfilingMode::kDispatch)
    return Status::Unavailable("CUDA dispatch profiling is not implemented");
  if (mode != GpuProfilingMode::kStage)
    return Status::InvalidArgument("CUDA operation requires stage profiling");
  return Status::Ok();
}

Status CudaBackend::PrepareAqEvaluationProfiled(
  const AqEvaluationPreparation& preparation, GpuProfilingMode mode,
  std::unique_ptr<PreparedAqEvaluation>* prepared, GpuExecutionProfile* profile) {
  Status status = ValidateCudaProfileRequest(mode, profile);
  if (!status.ok()) return status;
  if (prepared == nullptr)
    return Status::InvalidArgument("CUDA prepared AQ output pointer is null");
  if (preparation.frame_only || !preparation.resident_quantization)
    return Status::Unavailable("CUDA AQ profiling requires resident quantization");
  CudaProfileCapture capture(*this, "aq.prepare_reference");
  std::unique_ptr<PreparedAqEvaluation> candidate;
  status = PrepareAqEvaluation(preparation, &candidate);
  if (!status.ok()) return status;
  *prepared = std::move(candidate);
  *profile = std::move(capture).Finish();
  return Status::Ok();
}

gpu_profile_internal::GpuProfilingCapabilities
CudaBackend::QueryGpuProfilingCapabilities() const {
  return kCapabilities;
}

Status CudaBackend::ResolveGpuSubmissionProfile(
  GpuSubmission& submission, std::string_view submission_id,
  GpuProfilingMode mode, GpuExecutionProfile* profile) {
  if (mode != GpuProfilingMode::kStage || profile == nullptr ||
      submission_id.empty()) {
    return Status::InvalidArgument("CUDA submission profile request is invalid");
  }
  auto* cuda = dynamic_cast<CudaSubmission*>(&submission);
  if (cuda == nullptr) {
    return Status::InvalidArgument("Submission is not a CUDA submission");
  }
  return cuda->ResolveProfile(state_.get(), submission_id, profile);
}

Status CudaSubmission::ResolveProfile(
  const CudaDeviceState* state, std::string_view submission_id,
  GpuExecutionProfile* profile) {
  if (state != state_.get() || profile_begin_ == nullptr ||
      profile_.stages.size() != 1) {
    return Status::InvalidArgument(
      "CUDA submission is unprofiled or belongs to another backend");
  }
  Status status = Wait();
  if (!status.ok()) return status;
  ScopedCudaDevice device(state_->ordinal);
  if (device.status() != cudaSuccess) {
    return CudaRuntimeStatus(device.status(), "Select CUDA profile device");
  }
  float milliseconds = 0.0f;
  const auto error = cudaEventElapsedTime(&milliseconds, profile_begin_, event_);
  if (error != cudaSuccess) {
    return CudaRuntimeStatus(error, "Resolve CUDA stage timestamps");
  }
  const double nanoseconds = static_cast<double>(milliseconds) * 1000000.0;
  if (!std::isfinite(nanoseconds) || nanoseconds < 0.0 ||
      nanoseconds >= static_cast<double>(std::numeric_limits<uint64_t>::max())) {
    return Status::DeviceError("CUDA stage elapsed time is invalid");
  }
  try {
    GpuExecutionProfile candidate;
    candidate.mode = GpuProfilingMode::kStage;
    candidate.capabilities = kCapabilities;
    candidate.submissions.reserve(1);
    candidate.submissions.push_back(profile_);
    auto& resolved = candidate.submissions.front();
    resolved.submission_id = ProfileString(submission_id);
    resolved.command_buffer_gpu_nanoseconds = static_cast<uint64_t>(nanoseconds);
    // CUDA events expose elapsed device time rather than an absolute clock.
    // Each submission has its own zero origin; timestamps across submissions
    // must not be compared or treated as a host/device clock correlation.
    auto& stage = resolved.stages.front();
    stage.begin_timestamp = 0;
    stage.end_timestamp = resolved.command_buffer_gpu_nanoseconds;
    stage.gpu_nanoseconds = resolved.command_buffer_gpu_nanoseconds;
    *profile = std::move(candidate);
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Allocate CUDA profile snapshot");
  } catch (const std::length_error&) {
    return Status::InvalidArgument("CUDA profile snapshot is too large");
  }
  return Status::Ok();
}

}  // namespace gjxl::cuda_internal
