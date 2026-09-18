// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <barrier>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gpu/cuda/cuda_backend_internal.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/cuda/cuda_profile_storage_plan.h"
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

void RunCaptures(cuda_internal::CudaBackend& gpu, GpuProfilingMode mode) {
  constexpr size_t submissions = 8;
  constexpr std::array<std::string_view, 2> labels{"capture.left", "capture.right"};
  constexpr size_t label_length = labels[1].size();
  const auto prepare = [](cuda_internal::CudaBackend& backend,
                          std::unique_ptr<DeviceBuffer>* buffer) {
    const float initial = 1.0f;
    Check(backend.Allocate(sizeof(float), buffer));
    Check(backend.CopyHostToDevice(**buffer, &initial, sizeof(float), 0));
    const DevicePlaneView plane{buffer->get(), 0, DeviceElementType::kF32, {1, 1}, 1};
    return std::array<ImagePrimitiveCommand, 1>{
      PointwiseAffineCommand{plane, plane, 1.0f, 0.0f}};
  };
  std::unique_ptr<DeviceBuffer> buffer;
  const auto commands = prepare(gpu, &buffer);
  HostStorageBound bound;
  Check(ComputeProfileStorageBound({.submissions = submissions,
    .stages = submissions, .dispatches = submissions,
    .maximum_id_length = cuda_internal::kCudaKernelProfileIdLength}, &bound));
  SubmissionProfileStoragePlan child;
  Check(cuda_internal::ComputeCudaSubmissionProfileStoragePlan({.stages = 1, .dispatches = 1,
    .maximum_stage_id_length = label_length,
    .maximum_kernel_id_length = cuda_internal::kCudaKernelProfileIdLength,
    .maximum_submission_id_length = label_length}, &child));
  Require(bound.Add(child.resolution), "Capture bound overflow");
  std::barrier rendezvous(2);
  std::array<std::exception_ptr, 2> errors;
  std::array<std::thread, 2> workers;
  for (size_t side = 0; side < workers.size(); ++side) {
    workers[side] = std::thread([&, side] {
      // Capture construction allocates no backing. Both scopes overlap before
      // any submission, so a process-global capture would mix these graphs.
      cuda_internal::CudaProfileCapture capture(gpu, labels[side], mode);
      rendezvous.arrive_and_wait();
      try {
        ResourceBudget budget(bound.peak_bytes);
        ResourceReservation reservation;
        Check(budget.TryReserve(bound.peak_bytes, &reservation));
        GpuExecutionProfile profile;
        {
          ResourceContextScope scope({&reservation, ResourceClass::kDiagnostics});
          for (size_t i = 0; i < submissions; ++i) {
            std::unique_ptr<GpuSubmission> submission;
            Check(gpu.SubmitImagePrimitiveSequence(commands, &submission));
            Check(submission->Wait());
          }
          profile = std::move(capture).Finish();
        }
        Require(profile.submissions.size() == submissions,
          "Concurrent capture mixed submission counts");
        for (size_t i = 0; i < submissions; ++i) {
          const auto& s = profile.submissions[i];
          Require(s.submission_id == labels[side] && s.invocation == i &&
            s.stages.size() == 1 && s.stages[0].stage_id == labels[side],
            "Concurrent capture mixed operation identities");
        }
        reservation.Reset();
        Require(budget.snapshot().committed_bytes() > 0 &&
          budget.snapshot().peak_backing_bytes <= bound.peak_bytes,
          "Capture graph escaped its finite diagnostic domain");
        profile = {};
        Empty(budget);
      } catch (...) { errors[side] = std::current_exception(); }
    });
  }
  for (auto& worker : workers) worker.join();
  for (auto& error : errors) if (error) std::rethrow_exception(error);

  std::unique_ptr<GpuBackend> other;
  Check(CreateCudaBackend(&other));
  auto& foreign = dynamic_cast<cuda_internal::CudaBackend&>(*other);
  std::unique_ptr<DeviceBuffer> foreign_buffer;
  const auto foreign_commands = prepare(foreign, &foreign_buffer);
  GpuExecutionProfile outer, inner, foreign_profile;
  const auto submit = [&](cuda_internal::CudaBackend& backend) {
    std::unique_ptr<GpuSubmission> submission;
    Check(backend.SubmitImagePrimitiveSequence(
      &backend == &gpu ? commands : foreign_commands, &submission));
    Check(submission->Wait());
  };
  {
    cuda_internal::CudaProfileCapture capture(gpu, "outer");
    submit(gpu);
    {
      cuda_internal::CudaProfileCapture other_capture(foreign, "foreign");
      submit(gpu); // Finds the matching outer scope beneath another backend.
      submit(foreign);
      {
        cuda_internal::CudaProfileCapture nested(gpu, "inner");
        submit(gpu);
        inner = std::move(nested).Finish();
      }
      foreign_profile = std::move(other_capture).Finish();
    }
    submit(gpu);
    outer = std::move(capture).Finish();
  }
  Require(outer.submissions.size() == 3 && inner.submissions.size() == 1 &&
    inner.submissions[0].submission_id == "inner" &&
    foreign_profile.submissions.size() == 1 &&
    foreign_profile.submissions[0].submission_id == "foreign" &&
    cuda_internal::CudaProfileCapture::Current(gpu) == nullptr &&
    cuda_internal::CudaProfileCapture::Current(foreign) == nullptr,
    "Nested capture failed to restore its backend-specific scope");
  std::unique_ptr<GpuSubmission> ordinary;
  Check(gpu.SubmitImagePrimitiveSequence(commands, &ordinary));
  Check(ordinary->Wait());
  Require(!gpu.ResolveGpuSubmissionProfile(*ordinary, "ordinary",
    GpuProfilingMode::kStage, &outer).ok(),
    "Capture leaked into a later ordinary submission");
}

void Run(cuda_internal::CudaBackend& gpu, GpuProfilingMode mode) {
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
    capabilities.dispatch_boundary, "CUDA profiling capabilities are incorrect");
  SubmissionProfileStoragePlan plan;
  Check(cuda_internal::ComputeCudaSubmissionProfileStoragePlan({
    .stages = 1, .dispatches = commands.size(), .maximum_stage_id_length = stage_id.size(),
    .maximum_kernel_id_length = cuda_internal::kCudaKernelProfileIdLength,
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
      commands, stage_id, mode, &submission));
    Require(submission != nullptr, "No profiled submission");
    Check(gpu.ResolveGpuSubmissionProfile(
      *submission, submission_id, mode, &profile));
    Require(profile.mode == mode &&
      profile.capabilities == capabilities && profile.submissions.size() == 1,
      "Invalid profile graph");
    const auto& s = profile.submissions.front();
    Require(s.submission_id == std::string_view(submission_id) && s.stages.size() == 1 &&
      s.stages.front().stage_id == std::string_view(stage_id) &&
      s.stages.front().dispatches.size() == commands.size() &&
      s.stages.front().begin_timestamp == 0 &&
      s.stages.front().end_timestamp == s.stages.front().gpu_nanoseconds &&
      s.command_buffer_gpu_nanoseconds == s.stages.front().gpu_nanoseconds &&
      s.command_buffer_gpu_nanoseconds > 0,
      "Invalid CUDA device elapsed time or stage identity");
    uint64_t previous_end = 0;
    for (size_t i = 0; i < commands.size(); ++i) {
      const auto& dispatch = s.stages.front().dispatches[i];
      Require(dispatch.kernel_id == "PointwiseAffineKernel" && dispatch.invocation == i &&
        dispatch.kind == GpuDispatchKind::kThreadgroups &&
        dispatch.grid == GpuExtent3D{16, 1, 1} &&
        dispatch.threads_per_threadgroup == GpuExtent3D{256, 1, 1},
        "CUDA launch metadata differs from the actual primitive launch");
      Require(dispatch.begin_timestamp >= previous_end &&
        dispatch.end_timestamp >= dispatch.begin_timestamp &&
        dispatch.gpu_nanoseconds == dispatch.end_timestamp - dispatch.begin_timestamp &&
        dispatch.end_timestamp <= s.stages.front().end_timestamp,
        "CUDA dispatch timestamps are not ordered within their submission");
      Require(mode == GpuProfilingMode::kDispatch ? dispatch.gpu_nanoseconds > 0 :
        dispatch.begin_timestamp == 0 && dispatch.end_timestamp == 0,
        "CUDA dispatch timestamp mode differs from the requested mode");
      previous_end = dispatch.end_timestamp;
    }
    GpuExecutionProfile repeated;
    Check(gpu.ResolveGpuSubmissionProfile(
      *submission, submission_id, mode, &repeated));
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
    commands, stage_id, static_cast<GpuProfilingMode>(255), &failed).code() ==
      StatusCode::kInvalidArgument && !failed &&
    gpu.stats().committed_submissions == before,
    "Invalid profiling mode submitted work");
  Require(!gpu.ResolveGpuSubmissionProfile(*submission, submission_id,
      mode == GpuProfilingMode::kStage ? GpuProfilingMode::kDispatch : GpuProfilingMode::kStage,
      &profile).ok() && profile == sentinel, "A snapshot changed its recorded profiling mode");
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
      commands, stage_id, mode, &failed)
        .resource_plan_exceeded() && !failed &&
      gpu.stats().committed_submissions == before,
      "Under-planned profile submission was not rejected atomically");
    Require(gpu.ResolveGpuSubmissionProfile(
      *submission, submission_id, mode, &profile)
        .resource_plan_exceeded() && profile == sentinel,
      "Failed snapshot allocation changed the published profile");
  }
  tiny_job.Reset(); Empty(tiny);
  std::unique_ptr<GpuBackend> other;
  Check(CreateCudaBackend(&other));
  auto& other_profiler = dynamic_cast<GpuSubmissionProfiler&>(*other);
  Require(!other_profiler.ResolveGpuSubmissionProfile(
    *submission, submission_id, mode, &profile).ok() &&
    profile == sentinel, "Foreign submission accepted or output changed");
  gpu.ArmNextSubmissionFailureForTest(false, true);
  Check(gpu.SubmitImagePrimitiveSequenceProfiled(
    commands, stage_id, mode, &failed));
  Require(gpu.ResolveGpuSubmissionProfile(
    *failed, submission_id, mode, &profile).code() ==
      StatusCode::kDeviceError && profile == sentinel,
    "Failed completion published a profile");
  failed.reset();
  Check(gpu.SubmitImagePrimitiveSequence(commands, &failed));
  Check(failed->Wait());
  Require(!gpu.ResolveGpuSubmissionProfile(
    *failed, submission_id, mode, &profile).ok() &&
    profile == sentinel, "Unprofiled submission supplied timestamps");
  reservation.Reset();
  Require(budget.snapshot().committed_bytes() > 0,
    "Submission/profile dropped live charges on reservation closure");
  submission.reset(); profile = {};
  Empty(budget);
}

void RunAllocationFailures(cuda_internal::CudaBackend& gpu, GpuProfilingMode mode) {
  const float initial = 3.0f;
  std::unique_ptr<DeviceBuffer> buffer;
  Check(gpu.Allocate(sizeof(float), &buffer));
  const DevicePlaneView plane{buffer.get(), 0, DeviceElementType::kF32, {1, 1}, 1};
  const std::array<ImagePrimitiveCommand, 2> commands{
    PointwiseAffineCommand{plane, plane, 2.0f, 1.0f},
    PointwiseAffineCommand{plane, plane, 0.5f, -0.5f}};
  SubmissionProfileStoragePlan plan;
  Check(cuda_internal::ComputeCudaSubmissionProfileStoragePlan({
    .stages = 1, .dispatches = commands.size(), .maximum_stage_id_length = 7,
    .maximum_kernel_id_length = cuda_internal::kCudaKernelProfileIdLength,
    .maximum_submission_id_length = 7}, &plan));
  bool complete = false, partial_launch = false, snapshot_failure = false;
  for (size_t skip = 0; skip < 64; ++skip) {
    Check(gpu.CopyHostToDevice(*buffer, &initial, sizeof(initial), 0));
    ResourceBudget budget(plan.resolution.peak_bytes);
    ResourceReservation reservation;
    Check(budget.TryReserve(plan.resolution.peak_bytes, &reservation));
    GpuExecutionProfile profile;
    profile.mode = static_cast<GpuProfilingMode>(255);
    std::unique_ptr<GpuSubmission> submission;
    Status status;
    {
      ResourceContextScope scope({&reservation, ResourceClass::kDiagnostics});
      ArmManagedHostClassAllocationFailureAfterForTest(ResourceClass::kDiagnostics, skip);
      status = gpu.SubmitImagePrimitiveSequenceProfiled(commands, "failure", mode, &submission);
      if (status.ok())
        status = gpu.ResolveGpuSubmissionProfile(*submission, "failure", mode, &profile);
      complete = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
    }
    if (complete) Check(status);
    else {
      Require(status.code() == StatusCode::kOutOfMemory &&
        profile.mode == static_cast<GpuProfilingMode>(255) && profile.submissions.empty(),
        "Injected diagnostic allocation failure published a partial snapshot");
      snapshot_failure |= submission != nullptr;
      float actual = 0;
      Check(gpu.CopyDeviceToHost(*buffer, &actual, sizeof(actual), 0));
      partial_launch |= actual == 7.0f && submission == nullptr;
    }
    submission.reset(); profile = {};
    reservation.Reset(); Empty(budget);
    // The same backend remains usable after failures before/between launches
    // and after completion while deep-copying a snapshot.
    Check(gpu.CopyHostToDevice(*buffer, &initial, sizeof(initial), 0));
    Check(gpu.SubmitImagePrimitiveSequence(commands, &submission));
    Check(submission->Wait());
    float recovered = 0;
    Check(gpu.CopyDeviceToHost(*buffer, &recovered, sizeof(recovered), 0));
    Require(recovered == initial, "CUDA backend did not recover after profile failure");
    if (complete) break;
  }
  Require(complete && partial_launch && snapshot_failure,
    "CUDA diagnostic failure sweep missed launch or snapshot boundaries");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 2) throw std::runtime_error("Expected at most one completion-report path");
    std::unique_ptr<GpuBackend> gpu;
    const Status status = CreateCudaBackend(&gpu);
    if (status.code() == StatusCode::kUnavailable) return 77;
    Check(status);
    for (auto mode : {GpuProfilingMode::kStage, GpuProfilingMode::kDispatch}) {
      Run(dynamic_cast<cuda_internal::CudaBackend&>(*gpu), mode);
      RunCaptures(dynamic_cast<cuda_internal::CudaBackend&>(*gpu), mode);
      RunAllocationFailures(dynamic_cast<cuda_internal::CudaBackend&>(*gpu), mode);
    }
    std::cout << "CUDA stage/dispatch timing, output, accounting and failure checks passed\n";
    // Sanitizer console redirection can hide child output on Windows. This
    // marker proves the instrumented child reached the end of every check.
    if (argc == 2) {
      std::ofstream completion(argv[1]);
      completion << "PASS CUDA stage/dispatch profiling checks\n";
      completion.close();
      if (!completion) throw std::runtime_error("Cannot write completion report");
    }
  } catch (const std::exception& e) {
    DisarmManagedHostAllocationFailureForTest();
    std::cerr << e.what() << '\n'; return 1;
  }
}
