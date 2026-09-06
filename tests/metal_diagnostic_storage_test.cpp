// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "codestream/workflow_internal.h"
#include "core/image_buffer.h"
#include "gpu/metal/metal_backend.h"

namespace {
using namespace gjxl;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::gpu_profile_internal;
using namespace gjxl::codestream_internal;

bool Check(bool good, const char* message) {
  if (!good) std::cerr << message << '\n';
  return good;
}
bool Ok(const Status& status) {
  if (!status.ok()) std::cerr << status.message() << '\n';
  return status.ok();
}
bool Empty(const ResourceBudget& budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
    s.total.pending_count == 0 && s.open_reservations == 0 && s.waiting_requests == 0,
    "Metal diagnostic resource leak");
}

bool CheckPrimitiveFailures(GpuBackend& gpu, GpuProfilingMode mode) {
  auto* primitives = QueryGpuImagePrimitives(gpu);
  auto* profiler = dynamic_cast<GpuImagePrimitivesProfiler*>(&gpu);
  auto* resolver = dynamic_cast<GpuSubmissionProfiler*>(&gpu);
  if (!Check(primitives && profiler && resolver, "Missing Metal profiling interface")) return false;
  // Caller buffers have a separate explicit domain; the profile must never
  // borrow its allowance or escape to the process-default domain.
  ResourceBudget input_budget(128);
  ResourceReservation inputs;
  std::unique_ptr<DeviceBuffer> input, output;
  if (!Ok(input_budget.TryReserve(128, &inputs))) return false;
  {
    ResourceContextScope scope({&inputs, ResourceClass::kInput});
    if (!Ok(gpu.Allocate(64, &input)) || !Ok(gpu.Allocate(64, &output))) return false;
  }
  std::array<float, 16> values, expected, sentinel, actual;
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i);
    expected[i] = 2 * values[i] + 1;
    sentinel[i] = -99;
  }
  if (!Ok(gpu.CopyHostToDevice(*input, values.data(), sizeof(values))) ||
      !Ok(gpu.CopyHostToDevice(*output, sentinel.data(), sizeof(sentinel)))) return false;
  const DevicePlaneView in{input.get(), 0, DeviceElementType::kF32, {4, 4}, 4};
  const DevicePlaneView out{output.get(), 0, DeviceElementType::kF32, {4, 4}, 4};
  const std::array<ImagePrimitiveCommand, 1> commands{
    PointwiseAffineCommand{in, out, 2, 1}};
  const auto read_output = [&](const auto& oracle) {
    return Ok(gpu.CopyDeviceToHost(*output, actual.data(), sizeof(actual))) &&
      Check(actual == oracle, "Profiled affine output changed");
  };

  // The stage array fits exactly. Short stage/group IDs use inline storage.
  // The next allocation is inside the void dispatch encoder (kernel label or
  // dispatch array). It must preserve the terminal under-plan reason, not merely
  // return generic OOM, and must end encoding without committing GPU work.
  {
    ResourceBudget budget(sizeof(GpuStageProfile));
    ResourceReservation job;
    if (!Ok(budget.TryReserve(sizeof(GpuStageProfile), &job))) return false;
    const auto before = gpu.stats().committed_submissions;
    std::unique_ptr<GpuSubmission> submission;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      const Status status = profiler->SubmitImagePrimitiveSequenceProfiled(
        commands, "affine", mode, &submission);
      if (!Check(status.resource_plan_exceeded() && !submission &&
          gpu.stats().committed_submissions == before &&
          budget.snapshot().peak_backing_bytes == sizeof(GpuStageProfile),
          "Dispatch metadata overrun lost its reason or submitted work")) return false;
    }
    job.Reset();
    if (!Empty(budget) || !read_output(sentinel)) return false;
  }
  // Exercise the unprofiled path immediately: a dangling active-dispatch TLS
  // pointer would be used here. Reuse the same backend and buffers.
  {
    std::unique_ptr<GpuSubmission> submission;
    if (!Ok(primitives->SubmitImagePrimitiveSequence(commands, &submission)) ||
        !Check(submission != nullptr, "Missing recovery submission") ||
        !Ok(submission->Wait()) || !read_output(expected)) return false;
  }

  bool covered_snapshot = false;
  size_t fail_at = 0;
  for (; fail_at < 128; ++fail_at) {
    ResourceBudget budget(65536);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(65536, &job))) return false;
    bool injected;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      std::unique_ptr<GpuSubmission> submission;
      GpuExecutionProfile result;
      result.mode = GpuProfilingMode::kDisabled;
      ArmManagedHostClassAllocationFailureAfterForTest(ResourceClass::kDiagnostics, fail_at);
      Status status = profiler->SubmitImagePrimitiveSequenceProfiled(
        commands, "a deliberately long primitive stage identifier", mode, &submission);
      if (status.ok()) {
        if (!Check(submission != nullptr, "Missing profiled submission") ||
            !Ok(submission->Wait())) return false;
        status = resolver->ResolveGpuSubmissionProfile(*submission,
          "a deliberately long completed submission identifier", mode, &result);
        covered_snapshot |= !status.ok();
      }
      injected = !ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (injected) {
        if (!Check(status.code() == StatusCode::kOutOfMemory &&
            result.mode == GpuProfilingMode::kDisabled && result.submissions.empty(),
            "Metal diagnostic failure changed caller output")) return false;
      } else if (!Ok(status) || !Check(result.submissions.size() == 1 &&
          result.submissions[0].stages.size() == 1 &&
          result.submissions[0].stages[0].dispatches.size() == 1,
          "Metal diagnostic fixture lost dispatch metadata") || !read_output(expected)) return false;
      submission.reset();
      result = {};
      // Successful reuse and snapshot resolution after every allocation failure.
      if (!Ok(profiler->SubmitImagePrimitiveSequenceProfiled(commands, "affine", mode, &submission)) ||
          !Ok(submission->Wait()) || !Ok(resolver->ResolveGpuSubmissionProfile(
            *submission, "recovery", mode, &result)) || !read_output(expected)) return false;
    }
    job.Reset();
    if (!Empty(budget)) return false;
    if (!injected) break;
  }
  if (!Check(fail_at > 5 && fail_at < 128 && covered_snapshot,
      "Metal diagnostic failure enumeration missed snapshot storage")) return false;
  std::cout << "Metal primitive diagnostic failure positions (" << static_cast<int>(mode)
            << "): " << fail_at << '\n';
  input.reset(); output.reset(); inputs.Reset();
  return Empty(input_budget);
}

bool CheckWorkflowFailures(GpuBackend& gpu, GpuProfilingMode mode) {
  Image3FBuffer image({17, 9});
  for (size_t c = 0; c < 3; ++c)
    for (size_t i = 0; i < image.plane(c).size(); ++i)
      image.plane(c)[i] = 0.05f + 0.7f * ((i * (c + 3)) % 127) / 127.0f;
  VarDctEncodingOptions options;
  options.backend = VarDctBackendPreference::kMetal;
  options.butteraugli_target = 1.2f;
  options.effort = 1;
  options.cpu_thread_count = 1;
  std::vector<uint8_t> oracle;
  VarDctEncodingSummary oracle_summary;
  // Ample manual test envelope, not a production estimator.
  constexpr size_t kEnvelope = 64 * 1024 * 1024;
  {
    ResourceBudget budget(kEnvelope);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(kEnvelope, &job))) return false;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      if (!Ok(EncodeLinearRgbVarDctCodestreamWithBackendForTesting(image.const_view(),
          options, &gpu, true, &oracle, &oracle_summary))) return false;
    }
    if (!Ok(gpu.TrimPreparationCache())) return false;
    job.Reset();
    if (!Empty(budget)) return false;
  }
  const auto encode = [&](auto* bytes, auto* summary, auto* profile, auto* gpu_profile) {
    return EncodeLinearRgbVarDctCodestreamGpuProfiledWithBackendForTesting(
      image.const_view(), options, &gpu, true, mode, bytes, summary, profile, gpu_profile);
  };
  size_t fail_at = 0;
  for (; fail_at < 1024; ++fail_at) {
    ResourceBudget budget(kEnvelope);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(kEnvelope, &job))) return false;
    bool injected;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      std::vector<uint8_t> output{1, 2, 3};
      VarDctEncodingSummary summary;
      summary.score_history = {42};
      const auto old_summary = summary;
      VarDctEncodingProfile profile;
      profile.total_nanoseconds = 123;
      const auto old_profile = profile;
      GpuExecutionProfile gpu_profile;
      ArmManagedHostClassAllocationFailureAfterForTest(ResourceClass::kDiagnostics, fail_at);
      const Status status = encode(&output, &summary, &profile, &gpu_profile);
      injected = !ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (injected) {
        if (!Check(status.code() == StatusCode::kOutOfMemory &&
            output == std::vector<uint8_t>({1, 2, 3}) && summary == old_summary &&
            profile == old_profile && gpu_profile == GpuExecutionProfile{},
            "Profiled workflow allocation failure was not atomic")) {
          std::cerr << "Failure position " << fail_at << ": " << status.message() << '\n';
          return false;
        }
        if (!Ok(encode(&output, &summary, &profile, &gpu_profile))) return false;
      } else if (!Ok(status)) return false;
      if (!Check(output == oracle && summary == oracle_summary &&
          gpu_profile.mode == mode && !gpu_profile.submissions.empty() &&
          budget.snapshot().classes[static_cast<size_t>(ResourceClass::kDiagnostics)].live_capacity_bytes == 0,
          "Profiled workflow changed bytes/summary or retained published charges")) return false;
      // Public diagnostic objects remain alive through cache trim/job close.
      if (!Ok(gpu.TrimPreparationCache())) return false;
      job.Reset();
      if (!Empty(budget)) return false;
    }
    if (!injected) break;
  }
  std::cout << "Resident workflow diagnostic failure positions (" << static_cast<int>(mode)
            << "): " << fail_at << '\n';
  return Check(fail_at > 20 && fail_at < 1024, "Profiled workflow failure enumeration did not finish");
}
}  // namespace

int main() {
  std::unique_ptr<GpuBackend> gpu;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu))) return EXIT_FAILURE;
  auto* profiler = dynamic_cast<GpuSubmissionProfiler*>(gpu.get());
  if (!Check(profiler != nullptr, "Missing Metal submission profiler")) return EXIT_FAILURE;
  const auto capabilities = profiler->QueryGpuProfilingCapabilities();
  for (auto mode : {GpuProfilingMode::kStage, GpuProfilingMode::kDispatch}) {
    if (!capabilities.timestamp_counter ||
        (mode == GpuProfilingMode::kStage ? !capabilities.stage_boundary : !capabilities.dispatch_boundary)) {
      std::cout << "Skipping unavailable Metal profiling mode " << static_cast<int>(mode) << '\n';
      continue;
    }
    if (!CheckPrimitiveFailures(*gpu, mode) || !CheckWorkflowFailures(*gpu, mode)) return EXIT_FAILURE;
  }
  gpu.reset();
  return Empty(DefaultResourceBudget()) &&
    Check(DefaultResourceBudget().snapshot().peak_backing_bytes == 0,
          "Metal diagnostics escaped their explicit resource domains") ? EXIT_SUCCESS : EXIT_FAILURE;
}
