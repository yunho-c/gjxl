// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/ops/profile_storage_plan.h"

namespace {
using namespace gjxl;
using namespace gjxl::gpu_profile_internal;
using namespace gjxl::resource_budget_internal;
void Check(Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
void Require(bool good, const char* message) {
  if (!good) throw std::runtime_error(message);
}
void Empty(const ResourceBudget& budget) {
  const auto s = budget.snapshot();
  Require(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
    s.total.pending_count == 0 && s.open_reservations == 0,
    "Profile backing survived destruction");
}

void Run(cuda_internal::CudaBackend& gpu) {
  constexpr size_t count = 4096;
  std::vector<float> input(count), output(count);
  for (size_t i = 0; i < count; ++i) input[i] = static_cast<float>(i);
  std::unique_ptr<DeviceBuffer> buffer;
  Check(gpu.Allocate(count * sizeof(float), &buffer));
  Check(gpu.CopyHostToDevice(*buffer, input.data(), count * sizeof(float), 0));
  const DevicePlaneView plane{
    buffer.get(), 0, DeviceElementType::kF32, {64, 64}, 64};
  const std::array<ImagePrimitiveCommand, 2> commands{
    PointwiseAffineCommand{plane, plane, 2.0f, 1.0f},
    PointwiseAffineCommand{plane, plane, 0.5f, -0.5f},
  };
  const std::string stage_id(129, 's'), submission_id(135, 'q');
  const auto capabilities = gpu.QueryGpuProfilingCapabilities();
  Require(capabilities.timestamp_counter && capabilities.stage_boundary &&
    !capabilities.dispatch_boundary, "CUDA profiling capabilities are incorrect");
  SubmissionProfileStoragePlan plan;
  Check(ComputeSubmissionProfileStoragePlan({
    .stages = 1, .maximum_stage_id_length = stage_id.size(),
    .maximum_submission_id_length = submission_id.size()}, &plan));
  HostStorageBound bound = plan.resolution;
  Require(bound.Add(plan.resolved_output), "Profile bound overflow");
  ResourceBudget budget(bound.peak_bytes);
  ResourceReservation reservation;
  Check(budget.TryReserve(bound.peak_bytes, &reservation));
  GpuExecutionProfile profile;
  std::unique_ptr<GpuSubmission> submission;
  {
    ResourceContextScope scope({&reservation, ResourceClass::kDiagnostics});
    Check(gpu.SubmitImagePrimitiveSequenceProfiled(
      commands, stage_id, GpuProfilingMode::kStage, &submission));
    Require(submission != nullptr, "No profiled submission");
    Check(gpu.ResolveGpuSubmissionProfile(
      *submission, submission_id, GpuProfilingMode::kStage, &profile));
    Require(profile.mode == GpuProfilingMode::kStage &&
      profile.capabilities == capabilities && profile.submissions.size() == 1,
      "Invalid profile graph");
    const auto& s = profile.submissions.front();
    Require(s.submission_id == std::string_view(submission_id) && s.stages.size() == 1 &&
      s.stages.front().stage_id == std::string_view(stage_id) && s.stages.front().dispatches.empty() &&
      s.stages.front().begin_timestamp == 0 &&
      s.stages.front().end_timestamp == s.stages.front().gpu_nanoseconds &&
      s.command_buffer_gpu_nanoseconds == s.stages.front().gpu_nanoseconds &&
      s.command_buffer_gpu_nanoseconds > 0,
      "Invalid CUDA device elapsed time or stage identity");
    GpuExecutionProfile repeated;
    Check(gpu.ResolveGpuSubmissionProfile(
      *submission, submission_id, GpuProfilingMode::kStage, &repeated));
    Require(repeated == profile, "Repeated profile resolution changed timestamps");
  }
  Check(gpu.CopyDeviceToHost(*buffer, output.data(), count * sizeof(float), 0));
  Require(input == output, "Profiling changed the dependent primitive output");
  Require(budget.snapshot().peak_backing_bytes <= bound.peak_bytes &&
    budget.snapshot().classes[static_cast<size_t>(ResourceClass::kDiagnostics)]
      .live_capacity_bytes > 0, "Profile backing escaped diagnostics accounting");
  const GpuExecutionProfile sentinel = profile;
  const auto before = gpu.stats().committed_submissions;
  std::unique_ptr<GpuSubmission> failed;
  Require(gpu.SubmitImagePrimitiveSequenceProfiled(
    commands, stage_id, GpuProfilingMode::kDispatch, &failed).code() ==
      StatusCode::kUnavailable && !failed &&
    gpu.stats().committed_submissions == before,
    "Unsupported dispatch profiling submitted work");
  Require(!gpu.SubmitImagePrimitiveSequenceProfiled(
    commands, {}, GpuProfilingMode::kStage, &failed).ok() && !failed,
    "Empty stage ID accepted");
  ResourceBudget tiny(1);
  ResourceReservation tiny_job;
  Check(tiny.TryReserve(1, &tiny_job));
  Check(tiny_job.ReduceCapacity(0));
  {
    ResourceContextScope scope({&tiny_job, ResourceClass::kDiagnostics});
    Require(gpu.SubmitImagePrimitiveSequenceProfiled(
      commands, stage_id, GpuProfilingMode::kStage, &failed)
        .resource_plan_exceeded() && !failed &&
      gpu.stats().committed_submissions == before,
      "Under-planned profile submission was not rejected atomically");
    Require(gpu.ResolveGpuSubmissionProfile(
      *submission, submission_id, GpuProfilingMode::kStage, &profile)
        .resource_plan_exceeded() && profile == sentinel,
      "Failed snapshot allocation changed the published profile");
  }
  tiny_job.Reset(); Empty(tiny);
  std::unique_ptr<GpuBackend> other;
  Check(CreateCudaBackend(&other));
  auto& other_profiler = dynamic_cast<GpuSubmissionProfiler&>(*other);
  Require(!other_profiler.ResolveGpuSubmissionProfile(
    *submission, submission_id, GpuProfilingMode::kStage, &profile).ok() &&
    profile == sentinel, "Foreign submission accepted or output changed");
  gpu.ArmNextSubmissionFailureForTest(false, true);
  Check(gpu.SubmitImagePrimitiveSequenceProfiled(
    commands, stage_id, GpuProfilingMode::kStage, &failed));
  Require(gpu.ResolveGpuSubmissionProfile(
    *failed, submission_id, GpuProfilingMode::kStage, &profile).code() ==
      StatusCode::kDeviceError && profile == sentinel,
    "Failed completion published a profile");
  failed.reset();
  Check(gpu.SubmitImagePrimitiveSequence(commands, &failed));
  Check(failed->Wait());
  Require(!gpu.ResolveGpuSubmissionProfile(
    *failed, submission_id, GpuProfilingMode::kStage, &profile).ok() &&
    profile == sentinel, "Unprofiled submission supplied timestamps");
  reservation.Reset();
  Require(budget.snapshot().committed_bytes() > 0,
    "Submission/profile dropped live charges on reservation closure");
  submission.reset(); profile = {};
  Empty(budget);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 2) throw std::runtime_error("Expected at most one completion-report path");
    std::unique_ptr<GpuBackend> gpu;
    const Status status = CreateCudaBackend(&gpu);
    if (status.code() == StatusCode::kUnavailable) return 77;
    Check(status);
    Run(dynamic_cast<cuda_internal::CudaBackend&>(*gpu));
    std::cout << "CUDA stage timing, output, accounting and failure checks passed\n";
    // Sanitizer console redirection can hide child output on Windows. This
    // marker proves the instrumented child reached the end of every check.
    if (argc == 2) {
      std::ofstream completion(argv[1]);
      completion << "PASS CUDA stage profiling checks\n";
      completion.close();
      if (!completion) throw std::runtime_error("Cannot write completion report");
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
