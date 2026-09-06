// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "gpu/metal/metal_backend_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/metal/metal_status.h"

namespace gjxl::metal_internal {
namespace {

using gpu_profile_internal::GpuProfilingCapabilities;
using gpu_profile_internal::GpuDispatchKind;
using gpu_profile_internal::GpuDispatchProfile;
using gpu_profile_internal::GpuProfilingMode;
using gpu_profile_internal::GpuSubmissionProfile;
using gpu_profile_internal::ProfileString;

constexpr size_t kMaximumDispatchTimestampSamples = 4096;
constexpr size_t kMaximumRegisteredPipelines = 512;
constexpr size_t kMaximumKernelIdBytes = 128;

struct PipelineRegistryEntry {
  MTL::ComputePipelineState* pipeline = nullptr;
  std::array<char, kMaximumKernelIdBytes> kernel_id{};
  size_t kernel_id_size = 0;
};

struct ActiveDispatchProfile {
  GpuProfilingMode mode = GpuProfilingMode::kDisabled;
  MTL::CounterSampleBuffer* sample_buffer = nullptr;
  gpu_profile_internal::GpuStageProfile* stage = nullptr;
  size_t next_sample = 0;
  size_t maximum_samples = 0;
  bool overflow = false;
  bool allocation_failed = false;
  Status allocation_failure;
  std::array<char, kMaximumKernelIdBytes> current_kernel_id{};
  size_t current_kernel_id_size = 0;
};

thread_local ActiveDispatchProfile* g_active_dispatch_profile = nullptr;

struct ScopedActiveDispatchProfile {
  explicit ScopedActiveDispatchProfile(ActiveDispatchProfile* active)
    : previous(std::exchange(g_active_dispatch_profile, active)) {}
  ~ScopedActiveDispatchProfile() { g_active_dispatch_profile = previous; }
  ScopedActiveDispatchProfile(const ScopedActiveDispatchProfile&) = delete;
  ScopedActiveDispatchProfile& operator=(const ScopedActiveDispatchProfile&) = delete;
  ActiveDispatchProfile* previous;
};

struct ScopedComputeEncoding {
  explicit ScopedComputeEncoding(MTL::ComputeCommandEncoder* encoder) : encoder(encoder) {}
  ~ScopedComputeEncoding() { encoder->endEncoding(); }
  ScopedComputeEncoding(const ScopedComputeEncoding&) = delete;
  ScopedComputeEncoding& operator=(const ScopedComputeEncoding&) = delete;
  MTL::ComputeCommandEncoder* encoder;
};

std::mutex g_pipeline_registry_mutex;
std::array<PipelineRegistryEntry, kMaximumRegisteredPipelines>
  g_pipeline_registry;
size_t g_next_pipeline_registry_victim = 0;

void EncodeProfiledDispatch(
    MTL::ComputeCommandEncoder* encoder, GpuDispatchKind kind,
    MTL::Size grid, MTL::Size threads_per_threadgroup) {
  ActiveDispatchProfile* active = g_active_dispatch_profile;
  if (active == nullptr || active->stage == nullptr) {
    if (kind == GpuDispatchKind::kThreads) {
      encoder->dispatchThreads(grid, threads_per_threadgroup);
    } else {
      encoder->dispatchThreadgroups(grid, threads_per_threadgroup);
    }
    return;
  }

  const size_t invocation = active->stage->dispatches.size();
  if (active->mode == GpuProfilingMode::kDispatch) {
    if (active->next_sample + 2 > active->maximum_samples) {
      active->overflow = true;
    } else {
      encoder->sampleCountersInBuffer(
        active->sample_buffer, active->next_sample++, true);
    }
  }
  try {
    active->stage->dispatches.push_back({
      .kernel_id = active->current_kernel_id_size == 0
        ? active->stage->stage_id + ".dispatch_" +
            ProfileString(std::to_string(invocation))
        : ProfileString(
            active->current_kernel_id.data(),
            active->current_kernel_id_size),
      .kind = kind,
      .grid = {grid.width, grid.height, grid.depth},
      .threads_per_threadgroup = {
        threads_per_threadgroup.width,
        threads_per_threadgroup.height,
        threads_per_threadgroup.depth,
      },
      .invocation = static_cast<uint32_t>(invocation),
    });
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    active->allocation_failed = true;
    if (active->allocation_failure.ok()) active->allocation_failure = failure.status();
  } catch (const std::bad_alloc&) {
    active->allocation_failed = true;
  } catch (const std::length_error&) {
    active->allocation_failed = true;
  }
  if (kind == GpuDispatchKind::kThreads) {
    encoder->dispatchThreads(grid, threads_per_threadgroup);
  } else {
    encoder->dispatchThreadgroups(grid, threads_per_threadgroup);
  }
  if (active->mode == GpuProfilingMode::kDispatch && !active->overflow) {
    encoder->sampleCountersInBuffer(
      active->sample_buffer, active->next_sample++, true);
  }
}

MTL::CounterSet* FindTimestampCounterSet(MTL::Device* device) {
  if (device == nullptr) return nullptr;
  NS::Array* counter_sets = device->counterSets();
  if (counter_sets == nullptr) return nullptr;
  for (NS::UInteger index = 0; index < counter_sets->count(); ++index) {
    MTL::CounterSet* counter_set = counter_sets->object<MTL::CounterSet>(index);
    if (counter_set != nullptr && counter_set->name() != nullptr &&
        counter_set->name()->isEqualToString(MTL::CommonCounterSetTimestamp)) {
      return counter_set;
    }
  }
  return nullptr;
}

class MetalSubmission final : public GpuSubmission {
public:
  MetalSubmission(
    NS::SharedPtr<MTL::CommandBuffer> command_buffer,
    NS::SharedPtr<MTL::CommandQueue> command_queue,
    NS::SharedPtr<MTL::Device> device,
    bool test_fail_completion,
    NS::SharedPtr<MTL::CounterSampleBuffer> counter_sample_buffer = {},
    GpuSubmissionProfile profile = {},
    GpuProfilingMode profiling_mode = GpuProfilingMode::kDisabled,
    size_t resolved_sample_count = 0)
    : command_buffer_(std::move(command_buffer)),
      command_queue_(std::move(command_queue)),
      device_(std::move(device)),
      test_fail_completion_(test_fail_completion),
      counter_sample_buffer_(std::move(counter_sample_buffer)),
      profile_(std::move(profile)), profiling_mode_(profiling_mode),
      resolved_sample_count_(resolved_sample_count) {}

  ~MetalSubmission() override = default;

  Status Wait() override {
    std::call_once(wait_once_, [this] {
      auto pool = NS::TransferPtr(
        NS::AutoreleasePool::alloc()->init());
      command_buffer_->waitUntilCompleted();
      if (test_fail_completion_) {
        completion_status_ = Status::DeviceError(
          "Injected Metal command-buffer completion failure");
        return;
      }
      if (command_buffer_->status() == MTL::CommandBufferStatusError) {
        completion_status_ = metal::ErrorToDeviceStatus(
          command_buffer_->error(), "Metal command buffer");
      }
    });
    return completion_status_;
  }

  Status GpuDuration(uint64_t* nanoseconds) const {
    if (nanoseconds == nullptr) {
      return Status::InvalidArgument(
        "Metal GPU duration output pointer is null");
    }
    if (!completion_status_.ok() ||
        command_buffer_->status() != MTL::CommandBufferStatusCompleted) {
      return Status::FailedPrecondition(
        "Metal GPU duration requires successful completion");
    }
    const double begin = command_buffer_->GPUStartTime();
    const double end = command_buffer_->GPUEndTime();
    const double duration = (end - begin) * 1.0e9;
    if (!std::isfinite(duration) || duration < 0.0 ||
        duration > static_cast<double>(std::numeric_limits<uint64_t>::max())) {
      return Status::DeviceError(
        "Metal command buffer returned invalid GPU timestamps");
    }
    *nanoseconds = static_cast<uint64_t>(duration);
    return Status::Ok();
  }

  Status GpuProfile(GpuSubmissionProfile* profile) {
    if (profile == nullptr) {
      return Status::InvalidArgument(
        "Metal GPU profile output pointer is null");
    }
    if (!completion_status_.ok() ||
        command_buffer_->status() != MTL::CommandBufferStatusCompleted) {
      return Status::FailedPrecondition(
        "Metal GPU profile requires successful completion");
    }
    if (counter_sample_buffer_.get() == nullptr || profile_.stages.empty()) {
      return Status::FailedPrecondition(
        "Metal submission does not contain stage timestamps");
    }

    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    const size_t sample_count = resolved_sample_count_;
    NS::Data* resolved = counter_sample_buffer_->resolveCounterRange(
      NS::Range::Make(0, static_cast<NS::UInteger>(sample_count)));
    const size_t required_bytes =
      sample_count * sizeof(MTL::CounterResultTimestamp);
    if (resolved == nullptr || resolved->bytes() == nullptr ||
        resolved->length() < required_bytes) {
      return Status::DeviceError(
        "Metal timestamp counter resolution returned incomplete data");
    }

    try {
      GpuSubmissionProfile candidate = profile_;
      const auto* bytes = static_cast<const uint8_t*>(resolved->bytes());
      if (profiling_mode_ == GpuProfilingMode::kStage) {
        for (size_t index = 0; index < candidate.stages.size(); ++index) {
          MTL::CounterResultTimestamp begin{};
          MTL::CounterResultTimestamp end{};
          std::memcpy(
            &begin, bytes + (2 * index) * sizeof(begin), sizeof(begin));
          std::memcpy(
            &end, bytes + (2 * index + 1) * sizeof(end), sizeof(end));
          if (begin.timestamp == MTL::CounterErrorValue ||
              end.timestamp == MTL::CounterErrorValue ||
              end.timestamp < begin.timestamp) {
            return Status::DeviceError(
              "Metal timestamp counter returned an invalid stage interval");
          }
          candidate.stages[index].begin_timestamp = begin.timestamp;
          candidate.stages[index].end_timestamp = end.timestamp;
          candidate.stages[index].gpu_nanoseconds =
            end.timestamp - begin.timestamp;
        }
      } else {
        size_t sample_index = 0;
        for (auto& stage : candidate.stages) {
          for (auto& dispatch : stage.dispatches) {
            MTL::CounterResultTimestamp begin{};
            MTL::CounterResultTimestamp end{};
            std::memcpy(
              &begin, bytes + sample_index++ * sizeof(begin), sizeof(begin));
            std::memcpy(
              &end, bytes + sample_index++ * sizeof(end), sizeof(end));
            if (begin.timestamp == MTL::CounterErrorValue ||
                end.timestamp == MTL::CounterErrorValue ||
                end.timestamp < begin.timestamp) {
              return Status::DeviceError(
                "Metal timestamp counter returned an invalid dispatch interval");
            }
            dispatch.begin_timestamp = begin.timestamp;
            dispatch.end_timestamp = end.timestamp;
            dispatch.gpu_nanoseconds = end.timestamp - begin.timestamp;
          }
          if (!stage.dispatches.empty()) {
            stage.begin_timestamp = stage.dispatches.front().begin_timestamp;
            stage.end_timestamp = stage.dispatches.back().end_timestamp;
            stage.gpu_nanoseconds =
              stage.end_timestamp - stage.begin_timestamp;
          }
        }
      }
      Status status = GpuDuration(&candidate.command_buffer_gpu_nanoseconds);
      if (!status.ok()) return status;
      *profile = std::move(candidate);
      return Status::Ok();
    } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
      return failure.status();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("Unable to allocate Metal GPU profile snapshot");
    } catch (const std::length_error&) {
      return Status::InvalidArgument("Metal GPU profile snapshot is too large");
    }
  }

private:
  NS::SharedPtr<MTL::CommandBuffer> command_buffer_;
  NS::SharedPtr<MTL::CommandQueue> command_queue_;
  NS::SharedPtr<MTL::Device> device_;
  bool test_fail_completion_ = false;
  NS::SharedPtr<MTL::CounterSampleBuffer> counter_sample_buffer_;
  GpuSubmissionProfile profile_;
  GpuProfilingMode profiling_mode_ = GpuProfilingMode::kDisabled;
  size_t resolved_sample_count_ = 0;
  std::once_flag wait_once_;
  Status completion_status_;
};

}  // namespace

void DispatchMetalThreads(
    MTL::ComputeCommandEncoder* encoder, MTL::Size threads_per_grid,
    MTL::Size threads_per_threadgroup) {
  EncodeProfiledDispatch(
    encoder, GpuDispatchKind::kThreads, threads_per_grid,
    threads_per_threadgroup);
}

void DispatchMetalThreadgroups(
    MTL::ComputeCommandEncoder* encoder, MTL::Size threadgroups_per_grid,
    MTL::Size threads_per_threadgroup) {
  EncodeProfiledDispatch(
    encoder, GpuDispatchKind::kThreadgroups, threadgroups_per_grid,
    threads_per_threadgroup);
}

void RegisterMetalComputePipeline(
    MTL::ComputePipelineState* pipeline, std::string_view kernel_id) {
  if (pipeline == nullptr || kernel_id.empty() ||
      kernel_id.size() >= kMaximumKernelIdBytes) {
    return;
  }
  std::lock_guard lock(g_pipeline_registry_mutex);
  PipelineRegistryEntry* destination = nullptr;
  for (PipelineRegistryEntry& entry : g_pipeline_registry) {
    if (entry.pipeline == pipeline) {
      destination = &entry;
      break;
    }
    if (destination == nullptr && entry.pipeline == nullptr) {
      destination = &entry;
    }
  }
  if (destination == nullptr) {
    destination = &g_pipeline_registry[
      g_next_pipeline_registry_victim++ % g_pipeline_registry.size()];
  }
  destination->pipeline = pipeline;
  destination->kernel_id_size = kernel_id.size();
  kernel_id.copy(destination->kernel_id.data(), kernel_id.size());
  destination->kernel_id[kernel_id.size()] = '\0';
}

void RecordMetalComputePipelineState(MTL::ComputePipelineState* pipeline) {
  ActiveDispatchProfile* active = g_active_dispatch_profile;
  if (active == nullptr) return;
  std::lock_guard lock(g_pipeline_registry_mutex);
  active->current_kernel_id_size = 0;
  for (const PipelineRegistryEntry& entry : g_pipeline_registry) {
    if (entry.pipeline != pipeline) continue;
    active->current_kernel_id_size = entry.kernel_id_size;
    std::copy_n(
      entry.kernel_id.data(), entry.kernel_id_size,
      active->current_kernel_id.data());
    break;
  }
}

Status GetMetalSubmissionGpuDuration(
  GpuSubmission& submission,
  uint64_t* nanoseconds) {

  auto* metal = dynamic_cast<MetalSubmission*>(&submission);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "GPU duration requires a Metal submission");
  }
  return metal->GpuDuration(nanoseconds);
}

Status GetMetalSubmissionGpuProfile(
  GpuSubmission& submission,
  GpuSubmissionProfile* profile) {

  auto* metal = dynamic_cast<MetalSubmission*>(&submission);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "GPU profile requires a Metal submission");
  }
  return metal->GpuProfile(profile);
}

GpuProfilingCapabilities MetalBackend::QueryGpuProfilingCapabilities() const {
  return ProfilingCapabilities();
}

Status MetalBackend::ResolveGpuSubmissionProfile(
  GpuSubmission& submission,
  std::string_view submission_id,
  GpuProfilingMode mode,
  gpu_profile_internal::GpuExecutionProfile* profile) {

  if (profile == nullptr || submission_id.empty() ||
      mode == GpuProfilingMode::kDisabled) {
    return Status::InvalidArgument(
      "Metal GPU profile output is invalid");
  }
  GpuSubmissionProfile submission_profile;
  Status status = GetMetalSubmissionGpuProfile(
    submission, &submission_profile);
  if (!status.ok()) return status;
  try {
    submission_profile.submission_id = submission_id;
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate GPU submission profile ID");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "GPU submission profile ID is too large");
  }
  gpu_profile_internal::GpuExecutionProfile candidate;
  candidate.mode = mode;
  candidate.capabilities = ProfilingCapabilities();
  try {
    candidate.submissions.push_back(std::move(submission_profile));
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate GPU submission profile metadata");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "GPU submission profile metadata is too large");
  }
  *profile = std::move(candidate);
  return Status::Ok();
}

GpuProfilingCapabilities MetalBackend::ProfilingCapabilities() const {
  return {
    .timestamp_counter = FindTimestampCounterSet(device_.get()) != nullptr,
    .stage_boundary = device_.get() != nullptr && device_->supportsCounterSampling(
      MTL::CounterSamplingPointAtStageBoundary),
    .dispatch_boundary = device_.get() != nullptr && device_->supportsCounterSampling(
      MTL::CounterSamplingPointAtDispatchBoundary),
  };
}

Status MetalBackend::SubmitCompute(
  const char* label,
  ComputeEncodeCallback encode,
  const void* context,
  std::unique_ptr<GpuSubmission>* submission) {

  if (submission == nullptr) {
    return Status::InvalidArgument(
      "GPU submission output pointer is null");
  }
  submission->reset();
  if (label == nullptr || encode == nullptr) {
    return Status::Internal(
      "Metal compute submission callback is invalid");
  }
  const bool fail_submission =
    test_fail_submission_ || test_fail_next_submission_.exchange(false);
  const bool fail_completion =
    test_fail_completion_ || test_fail_next_completion_.exchange(false);
  if (fail_submission) {
    return Status::SubmissionFailed(
      "Injected Metal submission failure");
  }

  auto pool = NS::TransferPtr(
    NS::AutoreleasePool::alloc()->init());
  MTL::CommandBuffer* raw_command_buffer = command_queue_->commandBuffer();
  if (raw_command_buffer == nullptr) {
    return Status::SubmissionFailed(
      "Failed to create Metal command buffer");
  }
  auto command_buffer = NS::RetainPtr(raw_command_buffer);
  raw_command_buffer->setLabel(NS::String::string(
    label, NS::UTF8StringEncoding));
  MTL::ComputeCommandEncoder* encoder =
    raw_command_buffer->computeCommandEncoder();
  if (encoder == nullptr) {
    return Status::SubmissionFailed(
      "Failed to create Metal compute encoder");
  }

  encode(*this, encoder, context);
  encoder->endEncoding();
  std::unique_ptr<GpuSubmission> pending(
    new MetalSubmission(
      command_buffer,
      command_queue_,
      device_,
      fail_completion));
  raw_command_buffer->commit();
  RecordCommittedSubmission();
  *submission = std::move(pending);
  return Status::Ok();
}

Status MetalBackend::SubmitComputeProfiled(
  const char* label,
  std::span<const MetalProfiledComputeStage> stages,
  GpuProfilingMode mode,
  std::unique_ptr<GpuSubmission>* submission) {

  if (submission == nullptr) {
    return Status::InvalidArgument(
      "GPU submission output pointer is null");
  }
  submission->reset();
  if (label == nullptr || stages.empty() ||
      mode == GpuProfilingMode::kDisabled) {
    return Status::InvalidArgument(
      "Profiled Metal compute submission is invalid");
  }
  for (const MetalProfiledComputeStage& stage : stages) {
    if (stage.stage_id == nullptr || stage.encode == nullptr) {
      return Status::InvalidArgument(
        "Profiled Metal compute stage is invalid");
    }
  }

  const GpuProfilingCapabilities capabilities = ProfilingCapabilities();
  if (!capabilities.timestamp_counter) {
    return Status::Unavailable(
      "Metal timestamp counter sampling is unavailable");
  }
  if (mode == GpuProfilingMode::kStage && !capabilities.stage_boundary) {
    return Status::Unavailable(
      "Metal stage-boundary timestamp sampling is unavailable");
  }
  if (mode == GpuProfilingMode::kDispatch) {
    if (!capabilities.dispatch_boundary) {
      return Status::Unavailable(
        "Metal dispatch-boundary timestamp sampling is unavailable");
    }
  }
  if (stages.size() >
      std::numeric_limits<NS::UInteger>::max() / size_t{2}) {
    return Status::InvalidArgument(
      "Metal stage timestamp sample count overflows");
  }

  const bool fail_submission =
    test_fail_submission_ || test_fail_next_submission_.exchange(false);
  const bool fail_completion =
    test_fail_completion_ || test_fail_next_completion_.exchange(false);
  if (fail_submission) {
    return Status::SubmissionFailed(
      "Injected Metal submission failure");
  }

  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  MTL::CounterSet* timestamp_set = FindTimestampCounterSet(device_.get());
  if (timestamp_set == nullptr) {
    return Status::Unavailable(
      "Metal timestamp counter set is unavailable");
  }
  auto sample_descriptor = NS::TransferPtr(
    MTL::CounterSampleBufferDescriptor::alloc()->init());
  sample_descriptor->setCounterSet(timestamp_set);
  sample_descriptor->setLabel(NS::String::string(
    "gjxl stage timestamps", NS::UTF8StringEncoding));
  const size_t allocated_sample_count = mode == GpuProfilingMode::kStage
    ? stages.size() * 2
    : kMaximumDispatchTimestampSamples;
  sample_descriptor->setSampleCount(
    static_cast<NS::UInteger>(allocated_sample_count));
  sample_descriptor->setStorageMode(MTL::StorageModeShared);
  NS::Error* sample_error = nullptr;
  MTL::CounterSampleBuffer* raw_sample_buffer =
    device_->newCounterSampleBuffer(sample_descriptor.get(), &sample_error);
  if (raw_sample_buffer == nullptr) {
    return metal::ErrorToDeviceStatus(
      sample_error, "Metal timestamp sample-buffer allocation");
  }
  auto sample_buffer = NS::TransferPtr(raw_sample_buffer);

  MTL::CommandBuffer* raw_command_buffer = command_queue_->commandBuffer();
  if (raw_command_buffer == nullptr) {
    return Status::SubmissionFailed(
      "Failed to create profiled Metal command buffer");
  }
  auto command_buffer = NS::RetainPtr(raw_command_buffer);
  raw_command_buffer->setLabel(NS::String::string(
    label, NS::UTF8StringEncoding));

  GpuSubmissionProfile profile;
  try {
    profile.stages.reserve(stages.size());
  } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
    return failure.status();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory(
      "Unable to allocate Metal stage profile metadata");
  } catch (const std::length_error&) {
    return Status::InvalidArgument(
      "Metal stage profile metadata is too large");
  }
  ActiveDispatchProfile active_dispatch{
    .mode = mode,
    .sample_buffer = sample_buffer.get(),
    .maximum_samples = allocated_sample_count,
  };
  for (size_t index = 0; index < stages.size(); ++index) {
    const MetalProfiledComputeStage& stage = stages[index];
    MTL::ComputeCommandEncoder* encoder = nullptr;
    if (mode == GpuProfilingMode::kStage) {
      MTL::ComputePassDescriptor* pass =
        MTL::ComputePassDescriptor::computePassDescriptor();
      MTL::ComputePassSampleBufferAttachmentDescriptor* attachment =
        pass == nullptr ? nullptr : pass->sampleBufferAttachments()->object(0);
      if (attachment == nullptr) {
        return Status::SubmissionFailed(
          "Failed to create Metal timestamp attachment");
      }
      attachment->setSampleBuffer(sample_buffer.get());
      attachment->setStartOfEncoderSampleIndex(
        static_cast<NS::UInteger>(index * 2));
      attachment->setEndOfEncoderSampleIndex(
        static_cast<NS::UInteger>(index * 2 + 1));
      encoder = raw_command_buffer->computeCommandEncoder(pass);
    } else {
      encoder = raw_command_buffer->computeCommandEncoder();
    }
    if (encoder == nullptr) {
      return Status::SubmissionFailed(
        "Failed to create profiled Metal compute encoder");
    }
    const ScopedComputeEncoding encoding_scope(encoder);
    try {
      profile.stages.push_back({
        .stage_id = stage.stage_id,
        .group_id = stage.group_id == nullptr ? stage.stage_id : stage.group_id,
        .iteration = stage.iteration,
        .invocation = stage.invocation,
      });
    } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
      return failure.status();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory(
        "Unable to allocate Metal stage profile metadata");
    } catch (const std::length_error&) {
      return Status::InvalidArgument(
        "Metal stage profile metadata is too large");
    }
    active_dispatch.stage = &profile.stages.back();
    encoder->setLabel(NS::String::string(
      stage.stage_id, NS::UTF8StringEncoding));
    {
      const ScopedActiveDispatchProfile active_scope(&active_dispatch);
      try {
        stage.encode(*this, encoder, stage.context);
      } catch (const resource_budget_internal::ManagedAllocationFailure& failure) {
        return failure.status();
      } catch (const std::bad_alloc&) {
        return Status::OutOfMemory("Unable to allocate profiled compute encoding metadata");
      } catch (const std::length_error&) {
        return Status::InvalidArgument("Profiled compute encoding metadata is too large");
      }
    }
    if (active_dispatch.overflow) {
      return Status::InvalidArgument(
        "Metal dispatch profile exceeds the timestamp sample capacity");
    }
    if (active_dispatch.allocation_failed) {
      if (!active_dispatch.allocation_failure.ok()) return active_dispatch.allocation_failure;
      return Status::OutOfMemory(
        "Unable to allocate Metal dispatch profile metadata");
    }
  }

  const size_t resolved_sample_count = mode == GpuProfilingMode::kStage
    ? stages.size() * 2
    : active_dispatch.next_sample;
  if (resolved_sample_count == 0) {
    return Status::Internal(
      "Profiled Metal submission encoded no timestamp samples");
  }
  std::unique_ptr<GpuSubmission> pending(new MetalSubmission(
    command_buffer, command_queue_, device_, fail_completion,
    sample_buffer, std::move(profile), mode, resolved_sample_count));
  raw_command_buffer->commit();
  RecordCommittedSubmission();
  *submission = std::move(pending);
  return Status::Ok();
}

void MetalBackend::ArmNextSubmissionFailureForTest(
  bool fail_submission,
  bool fail_completion) noexcept {

  test_fail_next_submission_.store(fail_submission);
  test_fail_next_completion_.store(fail_completion);
}

}  // namespace gjxl::metal_internal

namespace gjxl {

Status ArmNextMetalSubmissionFailureForTest(
  GpuBackend& backend,
  bool fail_submission,
  bool fail_completion) {

  auto* metal = dynamic_cast<metal_internal::MetalBackend*>(&backend);
  if (metal == nullptr) {
    return Status::InvalidArgument(
      "Submission failure injection requires a Metal backend");
  }
  metal->ArmNextSubmissionFailureForTest(
    fail_submission, fail_completion);
  return Status::Ok();
}

}  // namespace gjxl
