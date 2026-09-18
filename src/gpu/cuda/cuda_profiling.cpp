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
  .dispatch_boundary = true,
};
}  // namespace

thread_local CudaProfileCapture* CudaProfileCapture::current_ = nullptr;
thread_local const CudaKernelProfileHooks* cuda_kernel_profile_hooks = nullptr;

CudaProfileCapture::CudaProfileCapture(CudaBackend& backend, std::string_view operation,
                                     GpuProfilingMode mode)
  : backend_(backend), operation_(operation),
    session_(mode, kCapabilities), previous_(current_) {
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
    submission, operation_, mode(), &child);
  if (!status.ok()) return status;
  return session_.Append(std::move(child));
}

bool CudaSubmission::BeginKernelProfile(const char* id, dim3 grid, dim3 block,
                                        cudaStream_t stream) noexcept {
  if (!kernel_profile_status_.ok() || kernel_event_error_ != cudaSuccess) return false;
  if (stream != state_->stream || profile_.stages.size() != 1) {
    kernel_profile_status_ = Status::InvalidArgument("Profile stream");
    return false;
  }
  try {
    auto& dispatches = profile_.stages.front().dispatches;
    if (dispatches.size() >= std::numeric_limits<uint32_t>::max()) {
      kernel_profile_status_ = Status::InvalidArgument("Profile count");
      return false;
    }
    uint32_t invocation = 0;
    for (const auto& previous : dispatches)
      if (previous.kernel_id == std::string_view(id)) ++invocation;
    dispatches.push_back({.kernel_id = ProfileString(id),
      .kind = GpuDispatchKind::kThreadgroups,
      .grid = {grid.x, grid.y, grid.z},
      .threads_per_threadgroup = {block.x, block.y, block.z},
      .invocation = invocation});
    if (profile_mode_ == GpuProfilingMode::kDispatch) {
      auto& events = kernel_events_.emplace_back();
      kernel_event_error_ = cudaEventCreate(&events.begin);
      if (kernel_event_error_ == cudaSuccess)
        kernel_event_error_ = cudaEventCreate(&events.end);
      if (kernel_event_error_ == cudaSuccess)
        kernel_event_error_ = cudaEventRecord(events.begin, stream);
    }
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    // These short messages fit the audited libraries' inline string storage;
    // failure latching itself must not allocate inside the noexcept hook.
    kernel_profile_status_ = failure.status().resource_plan_exceeded()
      ? Status::ResourcePlanExceeded("Profile bound") : Status::OutOfMemory("Profile OOM");
  } catch (const std::bad_alloc&) {
    kernel_profile_status_ = Status::OutOfMemory("Profile OOM");
  } catch (const std::length_error&) {
    kernel_profile_status_ = Status::InvalidArgument("Profile length");
  }
  return kernel_profile_status_.ok() && kernel_event_error_ == cudaSuccess;
}

void CudaSubmission::EndKernelProfile(cudaStream_t stream) noexcept {
  if (profile_mode_ == GpuProfilingMode::kDispatch &&
      kernel_event_error_ == cudaSuccess && kernel_profile_status_.ok())
    kernel_event_error_ = cudaEventRecord(kernel_events_.back().end, stream);
}

Status CudaSubmission::KernelProfileStatus() const {
  if (!kernel_profile_status_.ok()) return kernel_profile_status_;
  return CudaRuntimeStatus(kernel_event_error_, "Record CUDA dispatch timestamp");
}

Status ValidateCudaProfileRequest(GpuProfilingMode mode,
                                 const GpuExecutionProfile* profile) {
  if (profile == nullptr) return Status::InvalidArgument("CUDA profile output is null");
  if (mode != GpuProfilingMode::kStage && mode != GpuProfilingMode::kDispatch)
    return Status::InvalidArgument("CUDA operation requires stage or dispatch profiling");
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
  CudaProfileCapture capture(*this, "aq.prepare_reference", mode);
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
  if ((mode != GpuProfilingMode::kStage && mode != GpuProfilingMode::kDispatch) || profile == nullptr ||
      submission_id.empty()) {
    return Status::InvalidArgument("CUDA submission profile request is invalid");
  }
  auto* cuda = dynamic_cast<CudaSubmission*>(&submission);
  if (cuda == nullptr) {
    return Status::InvalidArgument("Submission is not a CUDA submission");
  }
  return cuda->ResolveProfile(state_.get(), submission_id, mode, profile);
}

Status CudaSubmission::ResolveProfile(
  const CudaDeviceState* state, std::string_view submission_id, GpuProfilingMode mode,
  GpuExecutionProfile* profile) {
  if (state != state_.get() || profile_begin_ == nullptr ||
      profile_.stages.size() != 1 || mode != profile_mode_) {
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
    candidate.mode = mode;
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
    if (mode == GpuProfilingMode::kDispatch) {
      if (kernel_events_.size() != stage.dispatches.size())
        return Status::Internal("CUDA dispatch event count differs from metadata");
      for (size_t i = 0; i < kernel_events_.size(); ++i) {
        float start = 0.0f, end = 0.0f;
        auto event_status = cudaEventElapsedTime(&start, profile_begin_, kernel_events_[i].begin);
        if (event_status == cudaSuccess)
          event_status = cudaEventElapsedTime(&end, profile_begin_, kernel_events_[i].end);
        if (event_status != cudaSuccess)
          return CudaRuntimeStatus(event_status, "Resolve CUDA dispatch timestamps");
        if (!std::isfinite(start) || !std::isfinite(end) || start < 0 || end < start ||
            end > milliseconds)
          return Status::DeviceError("CUDA dispatch timestamps are invalid");
        auto& dispatch = stage.dispatches[i];
        dispatch.begin_timestamp = static_cast<uint64_t>(static_cast<double>(start) * 1000000.0);
        dispatch.end_timestamp = static_cast<uint64_t>(static_cast<double>(end) * 1000000.0);
        dispatch.gpu_nanoseconds = dispatch.end_timestamp - dispatch.begin_timestamp;
      }
    }
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
