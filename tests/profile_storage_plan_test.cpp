// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "gpu/ops/gpu_execution_profile_internal.h"
#include "gpu/ops/profile_storage_plan.h"

#ifdef GJXL_PROFILE_PLAN_METAL
#include "gpu/metal/metal_backend.h"
#endif

namespace {
using namespace gjxl;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::gpu_profile_internal;

bool Check(bool good, const char *message) {
  if (!good)
    std::cerr << message << '\n';
  return good;
}
bool Ok(const Status &status) {
  if (!status.ok())
    std::cerr << status.message() << '\n';
  return status.ok();
}
bool Empty(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
                   s.total.pending_count == 0 && s.open_reservations == 0 &&
                   s.waiting_requests == 0,
               "Profile plan test leaked resources");
}
bool Reserve(ResourceBudget &budget, size_t bytes, ResourceReservation *job) {
  return Ok(budget.TryReserve(std::max(size_t{1}, bytes), job)) &&
         (bytes != 0 || Ok(job->ReduceCapacity(0)));
}

#ifndef GJXL_PROFILE_PLAN_METAL
bool StringBounds() {
  size_t cases = 0;
  for (size_t length = 0; length <= 4096; length += length < 256 ? 1 : 128) {
    const ProfileString source(length, 's'); // Borrowed, uncharged source.
    for (auto policy :
         {StringCapacityPolicy::kFresh, StringCapacityPolicy::kGrowing}) {
      HostStorageBound bound;
      if (!Check(bound.AddString(length, policy),
                 "Valid string bound rejected"))
        return false;
      ResourceBudget budget(std::max(size_t{1}, bound.peak_bytes));
      ResourceReservation job;
      ProfileString value;
      if (!Reserve(budget, bound.peak_bytes, &job))
        return false;
      {
        ResourceContextScope scope({&job, ResourceClass::kDiagnostics});
        if (policy == StringCapacityPolicy::kFresh) {
          value = ProfileString(source);
        } else {
          for (size_t i = 0; i < length; ++i)
            value.push_back('a');
          value.clear(); // Cleared capacity remains charged and bounded.
          value.resize(length / 2, 'b');
          value.shrink_to_fit();
          value.reserve(length);
          value.assign(length, 'c');
          if (length > 2)
            value.replace(0, length / 2, length / 2, 'd');
        }
      }
      const auto snapshot = budget.snapshot();
      if (!Check(value.size() == length &&
                     snapshot.total.live_capacity_bytes <=
                         bound.retained_bytes &&
                     snapshot.peak_backing_bytes <= bound.peak_bytes,
                 "String capacity or replacement exceeded its bound"))
        return false;
      job.Reset();
      const auto before = value.data();
      ReleaseManagedBackingAfterPublication(value);
      if (!Check(value.data() == before, "String publication copied backing") ||
          !Empty(budget))
        return false;
      ++cases;
    }
  }
  HostStorageBound sentinel{17, 19};
  const auto before = sentinel;
  if (!Check(!sentinel.AddString(std::numeric_limits<size_t>::max(),
                                 StringCapacityPolicy::kGrowing) &&
                 sentinel == before &&
                 !sentinel.AddString(64, StringCapacityPolicy(255)) &&
                 sentinel == before &&
                 !sentinel.AddString(128, StringCapacityPolicy::kFresh,
                                     std::numeric_limits<size_t>::max()) &&
                 sentinel == before,
             "Invalid/overflowing string bound mutated output"))
    return false;
  std::cout << "String growth/copy/publication cases: " << cases << '\n';
  return true;
}

bool PlanFailures() {
  HostStorageBound graph{17, 19};
  const auto old_graph = graph;
  SubmissionProfileStoragePlan submission;
  submission.recorded = graph;
  const auto old_submission = submission;
  const size_t huge = std::numeric_limits<size_t>::max();
  for (const auto shape : {ProfileStorageShape{huge, 0, 0, 0, 32},
                           {0, huge, 0, 0, 32},
                           {0, 0, huge, 0, 32},
                           {0, 0, 0, huge, 32},
                           {1, 1, 1, 1, huge}})
    if (!Check(!ComputeProfileStorageBound(shape, &graph).ok() &&
                   graph == old_graph,
               "Overflowing profile bound changed output"))
      return false;
  for (const auto options : {SubmissionProfileStorageOptions{},
                             {huge, 1, 32, 32, 32, 32},
                             {1, huge, 32, 32, 32, 32},
                             {1, 1, huge, 32, 32, 32},
                             {1, 1, 32, huge, 32, 32},
                             {1, 1, 32, 32, huge, 32},
                             {1, 1, 32, 32, 32, huge}})
    if (!Check(
            !ComputeSubmissionProfileStoragePlan(options, &submission).ok() &&
                submission == old_submission,
            "Invalid submission profile plan changed output"))
      return false;
  if (!Check(!ComputeProfileStorageBound({}, nullptr).ok() &&
                 !ComputeSubmissionProfileStoragePlan({1}, nullptr).ok(),
             "Null profile plan accepted"))
    return false;
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!Reserve(budget, 0, &job))
    return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    ArmNextManagedHostAllocationFailureForTest();
    const bool ok = ComputeProfileStorageBound(
                        {1000000, 1000000, 1000000, 1000000, 1024}, &graph)
                        .ok() &&
                    ComputeSubmissionProfileStoragePlan(
                        {1000000, 1000000, 256, 256, 286, 256}, &submission)
                        .ok();
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!Check(ok && pending && budget.snapshot().peak_backing_bytes == 0,
               "Profile planning allocated backing"))
      return false;
  }
  job.Reset();
  return Empty(budget);
}

GpuSubmissionProfile Recorded(const SubmissionProfileStorageOptions &o) {
  GpuSubmissionProfile result;
  result.stages.reserve(o.stages);
  for (size_t s = 0; s < o.stages; ++s) {
    result.stages.push_back(
        {.stage_id = ProfileString(o.maximum_stage_id_length, 's'),
         .group_id = ProfileString(o.maximum_group_id_length, 'g')});
    auto &stage = result.stages.back();
    for (size_t d = s; d < o.dispatches; d += o.stages) {
      // The fallback recorder expression can grow the first concatenation's
      // owner before moving it into the newly appended dispatch record.
      ProfileString name =
          d % 2 == 0
              ? stage.stage_id + ".dispatch_" + ProfileString(std::to_string(d))
              : ProfileString(o.maximum_kernel_id_length, 'k');
      stage.dispatches.push_back({.kernel_id = std::move(name),
                                  .invocation = static_cast<uint32_t>(d)});
    }
  }
  return result;
}
Status Resolve(const GpuSubmissionProfile &recorded, const std::string &id,
               GpuExecutionProfile *out) {
  try {
    auto snapshot = recorded;
    snapshot.submission_id.assign(id.data(), id.size());
    GpuExecutionProfile candidate;
    candidate.mode = GpuProfilingMode::kStage;
    candidate.capabilities = {true, true, true};
    candidate.submissions.push_back(std::move(snapshot));
    *out = std::move(candidate);
    return Status::Ok();
  } catch (const ManagedAllocationFailure &failure) {
    return failure.status();
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory("Injected profile snapshot failure");
  }
}

bool SubmissionCases() {
  size_t cases = 0, failures = 0;
  for (size_t stages : {size_t{1}, size_t{3}, size_t{7}})
    for (size_t dispatches : {size_t{0}, size_t{1}, size_t{17}, size_t{257}})
      for (size_t length : {size_t{0}, size_t{22}, size_t{23}, size_t{128}}) {
        SubmissionProfileStorageOptions o{stages, dispatches,  length,
                                          length, length + 30, length};
        SubmissionProfileStoragePlan plan;
        if (!Ok(ComputeSubmissionProfileStoragePlan(o, &plan)))
          return false;
        const size_t envelope =
            std::max(plan.recorded.peak_bytes, plan.resolution.peak_bytes);
        ResourceBudget budget(envelope);
        ResourceReservation job;
        GpuExecutionProfile output;
        const std::string id(length, 'i');
        if (!Reserve(budget, envelope, &job))
          return false;
        {
          ResourceContextScope scope({&job, ResourceClass::kPreparation});
          auto recorded = Recorded(o);
          if (!Check(budget.snapshot().peak_backing_bytes <=
                             plan.recorded.peak_bytes &&
                         budget.snapshot().total.live_capacity_bytes <=
                             plan.recorded.retained_bytes,
                     "Recording exceeded its profile bound") ||
              !Ok(Resolve(recorded, id, &output)) ||
              !Check(output.submissions[0].stages == recorded.stages,
                     "Profile snapshot changed stages"))
            return false;
        }
        if (!Check(budget.snapshot().total.live_capacity_bytes <=
                           plan.resolved_output.retained_bytes &&
                       budget.snapshot().peak_backing_bytes <= envelope,
                   "Resolved profile exceeded its bound"))
          return false;
        job.Reset();
        const auto *pointer = output.submissions.data();
        output.ReleaseResourceChargesAfterPublication();
        if (!Check(output.submissions.data() == pointer,
                   "Profile publication copied output") ||
            !Empty(budget))
          return false;
        ++cases;
      }
  const SubmissionProfileStorageOptions o{3, 11, 64, 64, 94, 96};
  SubmissionProfileStoragePlan plan;
  if (!Ok(ComputeSubmissionProfileStoragePlan(o, &plan)))
    return false;
  const size_t envelope =
      std::max(plan.recorded.peak_bytes, plan.resolution.peak_bytes);
  const std::string id(96, 'i');
  bool reached_success = false;
  for (size_t position = 0; position < 128; ++position) {
    ResourceBudget budget(envelope);
    ResourceReservation job;
    if (!Reserve(budget, envelope, &job))
      return false;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      auto recorded = Recorded(o);
      GpuExecutionProfile output;
      ArmManagedHostAllocationFailureAfterForTest(position);
      const auto status = Resolve(recorded, id, &output);
      const bool pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (status.ok()) {
        reached_success = pending;
      } else {
        if (!Check(status.code() == StatusCode::kOutOfMemory &&
                       !status.resource_plan_exceeded() && !pending &&
                       output == GpuExecutionProfile{},
                   "Snapshot physical failure was not atomic") ||
            !Ok(Resolve(recorded, id, &output)))
          return false;
        ++failures;
      }
    }
    job.Reset();
    if (!Empty(budget))
      return false;
    if (reached_success)
      break;
  }
  std::cout << "Submission profile shapes: " << cases
            << "; snapshot failure/recovery positions: " << failures << '\n';
  if (!Check(reached_success, "Snapshot failure sweep did not finish"))
    return false;
  ResourceBudget budget(envelope);
  ResourceReservation job;
  if (!Reserve(budget, envelope, &job))
    return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    auto recorded = Recorded(o);
    const size_t retained = budget.snapshot().total.live_capacity_bytes;
    if (!Ok(job.ReduceCapacity(retained)))
      return false;
    GpuExecutionProfile output;
    ArmNextManagedHostAllocationFailureForTest();
    const auto status = Resolve(recorded, id, &output);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!Check(status.resource_plan_exceeded() && pending &&
                   output == GpuExecutionProfile{} &&
                   budget.snapshot().total.live_capacity_bytes == retained,
               "Snapshot underplan escaped domain or corrupted original"))
      return false;
  }
  job.Reset();
  return Empty(budget);
}

bool SessionCases() {
  constexpr ProfileStorageShape shape{12, 4, 12, 44, 96};
  HostStorageBound bound;
  if (!Ok(ComputeProfileStorageBound(shape, &bound)))
    return false;
  // One complete child is live during Append's parent reserve/insert. Use a
  // separate child envelope rather than treating its arrays as parent aliases.
  const size_t envelope = 2 * bound.peak_bytes;
  ResourceBudget budget(envelope);
  ResourceReservation job;
  GpuExecutionProfile output;
  if (!Reserve(budget, envelope, &job))
    return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    GpuProfilingSession session(GpuProfilingMode::kStage, {true, true, true});
    for (size_t i = 0; i < 4; ++i) {
      GpuExecutionProfile child;
      auto record = Recorded({3, 11, 64, 64, 94, 96});
      // Build the child by moving, as production does after resolution; the
      // original recorded submission/snapshot overlap is tested separately.
      record.submission_id = ProfileString(96, 'i');
      child.mode = GpuProfilingMode::kStage;
      child.capabilities = {true, true, true};
      child.submissions.push_back(std::move(record));
      for (size_t w = 0; w < 3; ++w)
        child.wall_stages.push_back({.stage_id = ProfileString(96, 'w')});
      if (!Ok(session.Append(std::move(child))))
        return false;
    }
    output = std::move(session).Finish();
  }
  if (!Check(
          output.submissions.size() == 4 && output.wall_stages.size() == 12 &&
              output.submissions[3].invocation == 3 &&
              budget.snapshot().total.live_capacity_bytes <=
                  bound.retained_bytes,
          "Profile session aggregation exceeded bound or changed invocation"))
    return false;
  // Cleared inner labels/vectors preserve capacities but no longer contribute
  // logical lengths. Historical shape must still cover them.
  output.submissions[0].submission_id.clear();
  output.submissions[0].stages[0].dispatches.clear();
  job.Reset();
  output = {};
  return Empty(budget);
}
#else
bool MetalCase(GpuBackend &gpu, GpuProfilingMode mode) {
  auto *profiler = dynamic_cast<GpuImagePrimitivesProfiler *>(&gpu);
  auto *resolver = dynamic_cast<GpuSubmissionProfiler *>(&gpu);
  if (!Check(profiler && resolver, "Missing Metal profiler"))
    return false;
  ResourceBudget input_budget(128);
  ResourceReservation inputs;
  std::unique_ptr<DeviceBuffer> input, output;
  if (!Reserve(input_budget, 128, &inputs))
    return false;
  {
    ResourceContextScope scope({&inputs, ResourceClass::kInput});
    if (!Ok(gpu.Allocate(64, &input)) || !Ok(gpu.Allocate(64, &output)))
      return false;
  }
  std::array<float, 16> values{}, actual{};
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>(i);
  if (!Ok(gpu.CopyHostToDevice(*input, values.data(), sizeof(values))))
    return false;
  const DevicePlaneView in{input.get(), 0, DeviceElementType::kF32, {4, 4}, 4};
  const DevicePlaneView out{
      output.get(), 0, DeviceElementType::kF32, {4, 4}, 4};
  std::array<ImagePrimitiveCommand, 7> commands;
  for (auto &command : commands)
    command = PointwiseAffineCommand{in, out, 2, 1};
  for (size_t count : {size_t{1}, size_t{3}, size_t{7}})
    for (size_t length : {size_t{1}, size_t{22}, size_t{23}, size_t{128}}) {
      const std::string stage_id(length, 's'), submission_id(length, 'i');
      SubmissionProfileStoragePlan plan;
      if (!Ok(ComputeSubmissionProfileStoragePlan(
              {1, count, length, length, std::max(size_t{128}, length + 30),
               length},
              &plan)))
        return false;
      // The public primitive adapter owns one copied stage ID while recording.
      // Its stack context/command span and driver allocations are separate.
      auto recording = plan.recorded;
      if (!Check(recording.AddString(length, StringCapacityPolicy::kFresh),
                 "Primitive adapter label bound failed"))
        return false;
      const size_t envelope =
          std::max(recording.peak_bytes, plan.resolution.peak_bytes);
      ResourceBudget budget(envelope);
      ResourceReservation job;
      GpuExecutionProfile result;
      if (!Reserve(budget, envelope, &job))
        return false;
      {
        ResourceContextScope scope({&job, ResourceClass::kPreparation});
        std::unique_ptr<GpuSubmission> submission;
        if (!Ok(profiler->SubmitImagePrimitiveSequenceProfiled(
                std::span(commands).first(count), stage_id, mode,
                &submission)) ||
            !Check(submission != nullptr, "Missing profiled submission") ||
            !Ok(submission->Wait()) ||
            !Ok(resolver->ResolveGpuSubmissionProfile(
                *submission, submission_id, mode, &result)))
          return false;
      }
      if (!Check(result.submissions.size() == 1 &&
                     result.submissions[0].stages.size() == 1 &&
                     result.submissions[0].stages[0].dispatches.size() ==
                         count &&
                     budget.snapshot().total.live_capacity_bytes <=
                         plan.resolved_output.retained_bytes &&
                     budget.snapshot().peak_backing_bytes <= envelope,
                 "Metal profile shape or capacity exceeded plan") ||
          !Ok(gpu.CopyDeviceToHost(*output, actual.data(), sizeof(actual))))
        return false;
      for (size_t i = 0; i < actual.size(); ++i)
        if (!Check(actual[i] == 2 * values[i] + 1,
                   "Profile planning fixture changed affine output"))
          return false;
      job.Reset();
      result.ReleaseResourceChargesAfterPublication();
      if (!Empty(budget))
        return false;
    }
  input.reset();
  output.reset();
  inputs.Reset();
  return Empty(input_budget);
}
#endif
} // namespace

int main() {
#ifndef GJXL_PROFILE_PLAN_METAL
  if (!StringBounds() || !PlanFailures() || !SubmissionCases() ||
      !SessionCases())
    return EXIT_FAILURE;
#else
  std::unique_ptr<GpuBackend> gpu;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu)))
    return EXIT_FAILURE;
  const auto *resolver = dynamic_cast<GpuSubmissionProfiler *>(gpu.get());
  if (!Check(resolver != nullptr, "No Metal profile capability"))
    return EXIT_FAILURE;
  const auto capabilities = resolver->QueryGpuProfilingCapabilities();
  for (auto mode : {GpuProfilingMode::kStage, GpuProfilingMode::kDispatch}) {
    const bool supported =
        capabilities.timestamp_counter &&
        (mode == GpuProfilingMode::kStage ? capabilities.stage_boundary
                                          : capabilities.dispatch_boundary);
    if (!supported) {
      std::cout << "Unsupported Metal profiling mode: " << int(mode) << '\n';
      continue;
    }
    if (!MetalCase(*gpu, mode))
      return EXIT_FAILURE;
    std::cout << "Metal profile bound cases: 12; mode=" << int(mode) << '\n';
  }
#endif
  return Check(DefaultResourceBudget().snapshot().peak_backing_bytes == 0,
               "Profile bound test escaped to default domain")
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
