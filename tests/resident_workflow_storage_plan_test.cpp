// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "codestream/rate_control_internal.h"
#include "codestream/resident_workflow_storage_plan.h"
#include "codestream/workflow_internal.h"
#include "core/image_buffer.h"
#include "gpu/metal/metal_backend.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::gpu_profile_internal;
using namespace gjxl::resource_budget_internal;

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
               "Whole workflow leaked backing or admission");
}

bool CheckPlans() {
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(1, &job)) || !Ok(job.ReduceCapacity(0)))
    return false;
  ResourceContextScope context({&job, ResourceClass::kPreparation});
  size_t cases = 0;
  for (const Extent2D source : {Extent2D{1, 1},
                                {14, 15},
                                {15, 15},
                                {65, 63},
                                {257, 9},
                                {257, 257},
                                {3839, 2159}}) {
    for (int effort = 1; effort <= 10; ++effort) {
      for (size_t flags = 0; flags < 32; ++flags) {
        ResidentWorkflowStorageOptions o;
        o.encoding.backend = VarDctBackendPreference::kMetal;
        o.encoding.effort = effort;
        o.encoding.cpu_thread_count = flags & 1 ? 1 : 0;
        o.encoding.collect_final_butteraugli_score = bool(flags & 2);
        o.encoding.compression_mode =
            flags & 4 ? VarDctCompressionMode::kMaximumCompression
                      : VarDctCompressionMode::kAutomatic;
        o.collect_profile = bool(flags & 8);
        o.collect_gpu_profile = bool(flags & 16);
        o.collect_timing = true;
        ResidentWorkflowStoragePlan p;
        ArmNextManagedHostAllocationFailureForTest();
        const auto status = ComputeResidentWorkflowStoragePlan(source, o, &p);
        const bool pending = ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        const size_t iterations = AdaptiveQuantizationIterations(o.encoding);
        if (!Ok(status) ||
            !Check(pending && p.maximum_attempts == 1 &&
                       p.score_count == iterations + size_t(iterations == 0 ||
                                                            bool(flags & 2)) &&
                       p.working.peak_bytes >=
                           p.device_bytes + p.frontend.peak_bytes +
                               p.serializer.working.peak_bytes +
                               p.diagnostics.peak_bytes &&
                       p.output.peak_bytes <= p.working.peak_bytes &&
                       p.profile_shape.submissions ==
                           (o.collect_gpu_profile ? 5 : 0) &&
                       p.profile_shape.wall_stages ==
                           (o.collect_gpu_profile ? 13 : 0),
                   "Workflow plan formula or allocation-free contract failed"))
          return false;
        ++cases;
      }
    }
  }
  ResidentWorkflowStorageOptions o;
  o.encoding.backend = VarDctBackendPreference::kMetal;
  o.encoding.rate_control_mode = VarDctRateControlMode::kTargetBytes;
  o.collect_timing = true;
  ResidentWorkflowStoragePlan one;
  o.encoding.target_size_maximum_attempts = 1;
  if (!Ok(ComputeResidentWorkflowStoragePlan({89, 57}, o, &one)))
    return false;
  for (size_t attempts = 1; attempts <= 64; ++attempts) {
    o.encoding.target_size_maximum_attempts = attempts;
    ResidentWorkflowStoragePlan p;
    if (!Ok(ComputeResidentWorkflowStoragePlan({89, 57}, o, &p)) ||
        !Check(p.maximum_attempts == attempts &&
                   p.device_bytes == one.device_bytes &&
                   p.frontend == one.frontend &&
                   p.retained_best.retained_bytes ==
                       (attempts > 1 ? p.serializer.output.retained_bytes +
                                           sizeof(double) * p.score_count
                                     : 0) &&
                   p.search_control.peak_bytes ==
                       3 * 2 * sizeof(float) *
                           std::max(size_t{1}, attempts - 1),
               "Search retained-owner or interval bound failed"))
      return false;
  }
  ResidentWorkflowStoragePlan sentinel;
  sentinel.device_bytes = 42;
  sentinel.working = {43, 44};
  auto p = sentinel;
  for (size_t bad = 0; bad < 10; ++bad) {
    auto invalid = o;
    switch (bad) {
    case 0:
      invalid.encoding.backend = VarDctBackendPreference::kCpu;
      break;
    case 1:
      invalid.encoding.backend = VarDctBackendPreference::kAutomatic;
      break;
    case 2:
      invalid.encoding.metal_aq_mode =
          GpuAdaptiveQuantizationMode::kExactCoefficients;
      break;
    case 3:
      invalid.encoding.rate_control_mode = VarDctRateControlMode::kMaximumError;
      break;
    case 4:
      invalid.collect_gpu_profile = true;
      break;
    case 5:
      invalid.encoding.effort = 0;
      break;
    case 6:
      invalid.encoding.target_size_maximum_attempts = 0;
      break;
    case 7:
      invalid.encoding.target_size_maximum_attempts = 65;
      break;
    case 8:
      invalid.encoding.cpu_thread_count = 257;
      break;
    case 9:
      invalid.encoding.metal_aq_mode =
          GpuAdaptiveQuantizationMode::kMaximumThroughput;
      break;
    }
    if (!Check(
            !ComputeResidentWorkflowStoragePlan({89, 57}, invalid, &p).ok() &&
                p == sentinel,
            "Unsupported workflow changed its plan"))
      return false;
  }
  for (auto bad :
       {Extent2D{}, {std::numeric_limits<size_t>::max(), 8}, {65536, 65536}})
    if (!Check(!ComputeResidentWorkflowStoragePlan(bad, o, &p).ok() &&
                   p == sentinel,
               "Invalid workflow geometry changed its plan"))
      return false;
  if (!Check(!ComputeResidentWorkflowStoragePlan({8, 8}, o, nullptr).ok() &&
                 !ComputeTargetSizeControlStorageBound(1, nullptr).ok(),
             "Null plan accepted"))
    return false;
  job.Reset();
  std::cout << "Whole workflow plan shapes: " << cases
            << "; search bounds: 64\n";
  return Empty(budget);
}

Image3FBuffer MakeImage(Extent2D extent) {
  Image3FBuffer image(extent);
  for (size_t c = 0; c < 3; ++c)
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        image.plane(c)[y * extent.width + x] =
            0.03f + 0.8f * ((x * 11 + y * 7 + c * 23 + x * y) % 251) / 251.0f;
  return image;
}

struct Result {
  std::vector<uint8_t> bytes;
  VarDctEncodingSummary summary;
  VarDctEncodingTiming timing;
  VarDctEncodingProfile profile;
  GpuExecutionProfile gpu_profile;
};

Status Encode(GpuBackend &gpu, ConstImage3FView image,
              const ResidentWorkflowStorageOptions &o, Result *result) {
  if (o.collect_gpu_profile)
    return EncodeLinearRgbVarDctCodestreamGpuProfiledWithBackendForTesting(
        image, o.encoding, &gpu, true, GpuProfilingMode::kStage, &result->bytes,
        &result->summary, &result->profile, &result->gpu_profile);
  if (o.collect_timing)
    return EncodeLinearRgbVarDctCodestreamProfiled(
        image, o.encoding, &result->bytes, &result->summary, &result->timing);
  if (o.collect_profile)
    return EncodeLinearRgbVarDctCodestreamProfiledWithBackendForTesting(
        image, o.encoding, &gpu, true, &result->bytes, &result->summary,
        &result->profile);
  return EncodeLinearRgbVarDctCodestreamWithBackendForTesting(
      image, o.encoding, &gpu, true, &result->bytes, &result->summary);
}

bool CheckProfile(const GpuExecutionProfile &profile,
                  const ResidentWorkflowStoragePlan &plan) {
  size_t stages = 0, dispatches = 0;
  const auto id = [&](const auto &s) {
    return s.size() <= plan.profile_shape.maximum_id_length;
  };
  for (const auto &wall : profile.wall_stages)
    if (!Check(id(wall.stage_id), "Wall ID exceeds bound"))
      return false;
  for (const auto &submission : profile.submissions) {
    if (!Check(id(submission.submission_id), "Submission ID exceeds bound"))
      return false;
    stages += submission.stages.size();
    for (const auto &stage : submission.stages) {
      if (!Check(id(stage.stage_id) && id(stage.group_id),
                 "Stage ID exceeds bound"))
        return false;
      dispatches += stage.dispatches.size();
      for (const auto &dispatch : stage.dispatches)
        if (!Check(id(dispatch.kernel_id), "Kernel ID exceeds bound"))
          return false;
    }
  }
  return Check(profile.wall_stages.size() <= plan.profile_shape.wall_stages &&
                   profile.submissions.size() ==
                       plan.profile_shape.submissions &&
                   stages <= plan.profile_shape.stages &&
                   dispatches <= plan.profile_shape.dispatches,
               "Observed workflow profile exceeds preflight count");
}

bool Trim(GpuBackend &gpu) {
  return Ok(gpu.TrimPreparationCache()) && Ok(TrimVarDctPreparationCache());
}

bool RunCase(GpuBackend &gpu, ConstImage3FView image,
             const ResidentWorkflowStorageOptions &o) {
  ResidentWorkflowStoragePlan plan;
  if (!Ok(ComputeResidentWorkflowStoragePlan(image.extent(), o, &plan)))
    return false;
  Result oracle;
  {
    // Reference uses an unlimited domain and ample manually reserved credit;
    // the candidate's bound is never derived from an observed peak.
    ResourceBudget budget(0);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(
            std::max(plan.working.peak_bytes, size_t{64} << 20), &job)))
      return false;
    ResourceContextScope context({&job, ResourceClass::kPreparation});
    auto reference_options = o;
    reference_options.collect_gpu_profile = reference_options.collect_profile =
        reference_options.collect_timing = false;
    if (!Ok(Encode(gpu, image, reference_options, &oracle)) || !Trim(gpu))
      return false;
    job.Reset();
    if (!Empty(budget))
      return false;
  }
  ResourceBudget budget(plan.working.peak_bytes);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(plan.working.peak_bytes, &job)))
    return false;
  Result measured;
  for (size_t pass = 0; pass < 2; ++pass) {
    ResourceContextScope context({&job, ResourceClass::kPreparation});
    const Status status = Encode(gpu, image, o, &measured);
    if (!Ok(status)) {
      std::cerr << "Shape " << image.width() << 'x' << image.height()
                << " effort " << o.encoding.effort << " plan "
                << plan.working.peak_bytes << " peak "
                << budget.snapshot().peak_backing_bytes << '\n';
      return false;
    }
    if (!Check(measured.bytes == oracle.bytes &&
                   measured.summary == oracle.summary &&
                   budget.snapshot().peak_backing_bytes <=
                       plan.working.peak_bytes &&
                   budget.snapshot().committed_bytes() ==
                       plan.working.peak_bytes,
               "Bounded workflow changed bytes, decisions or reservation"))
      return false;
    if (o.collect_gpu_profile && !CheckProfile(measured.gpu_profile, plan))
      return false;
    if (o.collect_timing &&
        !Check(measured.timing.attempts.size() ==
                       measured.summary.encode_attempt_count &&
                   measured.timing.attempts.size() <= plan.maximum_attempts,
               "Timing attempt count exceeds plan"))
      return false;
    // Leave caches and old published output alive for the second call.
  }
  if (!Trim(gpu))
    return false;
  job.Reset();
  return Empty(budget);
}

bool CheckRuntime(GpuBackend &gpu) {
  const auto caps = dynamic_cast<GpuSubmissionProfiler &>(gpu)
                        .QueryGpuProfilingCapabilities();
  const bool profile_available = caps.timestamp_counter && caps.stage_boundary;
  size_t cases = 0;
  for (const Extent2D extent : {Extent2D{1, 1},
                                {14, 15},
                                {15, 15},
                                {65, 63},
                                {257, 9},
                                {89, 57},
                                {257, 257}}) {
    auto image = MakeImage(extent);
    for (int effort : {1, 4, 7, 9, 10}) {
      for (bool final : {false, true}) {
        ResidentWorkflowStorageOptions o;
        o.encoding.backend = VarDctBackendPreference::kMetal;
        o.encoding.effort = effort;
        o.encoding.butteraugli_target = 1.2f;
        o.encoding.cpu_thread_count = effort == 9 ? 0 : effort == 10 ? 4 : 1;
        o.encoding.collect_final_butteraugli_score = final;
        o.collect_profile = true;
        o.collect_gpu_profile = profile_available;
        if (!RunCase(gpu, image.const_view(), o))
          return false;
        ++cases;
      }
    }
  }
  auto image = MakeImage({89, 57});
  for (size_t flags = 0; flags < 4; ++flags) {
    ResidentWorkflowStorageOptions o;
    o.encoding.backend = VarDctBackendPreference::kMetal;
    o.encoding.cpu_thread_count = 1;
    o.collect_profile = bool(flags & 1);
    o.collect_gpu_profile = bool(flags & 2) && profile_available;
    if (!RunCase(gpu, image.const_view(), o))
      return false;
    ++cases;
  }
  for (size_t flags = 0; flags < 16; ++flags) {
    ResidentWorkflowStorageOptions o;
    o.encoding.backend = VarDctBackendPreference::kMetal;
    o.encoding.rate_control_mode =
        flags & 1 ? VarDctRateControlMode::kTargetBytes
                  : VarDctRateControlMode::kTargetBitsPerPixel;
    o.encoding.target_bytes = 650;
    o.encoding.target_bits_per_pixel = 1.2;
    o.encoding.target_size_maximum_attempts = 4;
    o.encoding.target_size_tolerance = 0;
    o.encoding.target_size_selection =
        flags & 2 ? TargetSizeSelectionPolicy::kClosestAbsolute
                  : TargetSizeSelectionPolicy::kLargestAtOrBelow;
    o.encoding.effort = flags & 4 ? 4 : 7;
    o.encoding.cpu_thread_count = 1;
    o.collect_timing = bool(flags & 8);
    o.collect_profile = !o.collect_timing;
    if (!RunCase(gpu, image.const_view(), o))
      return false;
    ++cases;
  }
  for (size_t flags = 0; flags < 4; ++flags) {
    ResidentWorkflowStorageOptions o;
    o.encoding.backend = VarDctBackendPreference::kMetal;
    o.encoding.effort = 1;
    o.encoding.cpu_thread_count = 1;
    o.encoding.compression_mode = VarDctCompressionMode::kMaximumCompression;
    if (flags & 1)
      o.encoding.density_mode = VarDctDensityMode::kHighDensity;
    else
      o.encoding.metal_aq_mode = GpuAdaptiveQuantizationMode::kThroughput;
    o.collect_timing = bool(flags & 2);
    o.collect_profile = !o.collect_timing;
    if (!RunCase(gpu, image.const_view(), o))
      return false;
    ++cases;
  }
  {
    auto large = MakeImage({3839, 2159});
    ResidentWorkflowStorageOptions o;
    o.encoding.backend = VarDctBackendPreference::kMetal;
    o.encoding.cpu_thread_count = 4;
    o.collect_profile = true;
    o.collect_gpu_profile = profile_available;
    if (!RunCase(gpu, large.const_view(), o))
      return false;
    ++cases;
  }
  std::cout << "Bounded complete-workflow cases (cold/warm): " << cases << '\n';
  if (!profile_available)
    std::cout << "Stage-boundary profiling unavailable; not qualified\n";
  return true;
}

bool CheckAdmissionFailure(GpuBackend &gpu) {
  auto image = MakeImage({17, 9});
  for (bool search : {false, true}) {
    ResidentWorkflowStorageOptions o;
    o.encoding.backend = VarDctBackendPreference::kMetal;
    o.encoding.effort = 1;
    o.encoding.cpu_thread_count = 1;
    if (search) {
      o.encoding.rate_control_mode = VarDctRateControlMode::kTargetBytes;
      o.encoding.target_bytes = 200;
      o.encoding.target_size_maximum_attempts = 4;
    }
    ResidentWorkflowStoragePlan plan;
    if (!Ok(ComputeResidentWorkflowStoragePlan(image.extent(), o, &plan)) ||
        !Trim(gpu))
      return false;
    const auto before = gpu.stats();
    ResourceBudget insufficient(plan.working.peak_bytes - 1);
    ResourceReservation rejected;
    if (!Check(
            !insufficient.TryReserve(plan.working.peak_bytes, &rejected).ok() &&
                !rejected.valid(),
            "Oversized whole job was admitted") ||
        !Empty(insufficient))
      return false;
    Result result;
    result.bytes = {1, 2, 3};
    result.summary.score_history = {42};
    const auto old_summary = result.summary;
    {
      ResourceBudget budget(1);
      ResourceReservation job;
      if (!Ok(budget.TryReserve(1, &job)))
        return false;
      ResourceContextScope context({&job, ResourceClass::kPreparation});
      const Status status = Encode(gpu, image.const_view(), o, &result);
      if (!Check(status.resource_plan_exceeded() &&
                     result.bytes == std::vector<uint8_t>({1, 2, 3}) &&
                     result.summary == old_summary &&
                     gpu.stats().successful_allocations ==
                         before.successful_allocations &&
                     gpu.stats().committed_submissions ==
                         before.committed_submissions,
                 "Underplanned workflow escaped, retried or changed output"))
        return false;
      job.Reset();
      if (!Empty(budget))
        return false;
    }
    // A physical failure is distinct from an underplan. In a size search it
    // may be candidate-local, while the prior typed error remains terminal.
    ResourceBudget budget(plan.working.peak_bytes);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(plan.working.peak_bytes, &job)) ||
        !Ok(ArmNextMetalAllocationFailureForTest(gpu)))
      return false;
    {
      ResourceContextScope context({&job, ResourceClass::kPreparation});
      const Status status = Encode(gpu, image.const_view(), o, &result);
      // The first resident-input allocation fails before search attempts start.
      if (!Check(status.code() == StatusCode::kOutOfMemory &&
                     !status.resource_plan_exceeded() &&
                     result.bytes == std::vector<uint8_t>({1, 2, 3}) &&
                     result.summary == old_summary,
                 "Physical preparation failure was not atomic"))
        return false;
      if (!Ok(Encode(gpu, image.const_view(), o, &result)) ||
          !Check(!result.bytes.empty() &&
                     result.bytes != std::vector<uint8_t>({1, 2, 3}),
                 "Workflow did not recover"))
        return false;
    }
    if (!Trim(gpu))
      return false;
    job.Reset();
    if (!Empty(budget))
      return false;
  }
  return true;
}

bool CheckRetainedOutput() {
  auto image = MakeImage({89, 57});
  ResidentWorkflowStorageOptions o;
  o.encoding.backend = VarDctBackendPreference::kMetal;
  o.encoding.cpu_thread_count = 1;
  o.encoding.rate_control_mode = VarDctRateControlMode::kTargetBytes;
  o.encoding.target_bytes = 650;
  o.encoding.target_size_maximum_attempts = 4;
  o.collect_timing = true;
  ResidentWorkflowStoragePlan plan;
  if (!Ok(ComputeResidentWorkflowStoragePlan(image.extent(), o, &plan)))
    return false;
  ResourceBudget budget(plan.working.peak_bytes);
  ResourceReservation job;
  if (!Ok(budget.TryReserve(plan.working.peak_bytes, &job)))
    return false;
  OwnedEncodingResult result;
  {
    ResourceContextScope context({&job, ResourceClass::kPreparation});
    if (!Ok(EncodeLinearRgbVarDctCodestreamOwned(
            image.const_view(), o.encoding, &result.codestream, &result.summary,
            &result.timing)))
      return false;
  }
  if (!Ok(TrimVarDctPreparationCache()))
    return false;
  job.Reset();
  const auto s = budget.snapshot();
  if (!Check(s.open_reservations == 0 && s.total.live_capacity_bytes > 0 &&
                 s.total.live_capacity_bytes <= plan.output.retained_bytes &&
                 s.total.idle_capacity_bytes == 0 && s.total.pending_count == 0,
             "Closed job lost retained-output charges or exceeded the output "
             "bound"))
    return false;
  std::vector<uint8_t> published;
  VarDctEncodingSummary summary;
  VarDctEncodingTiming timing;
  result.codestream.PublishTo(&published);
  result.summary.PublishTo(&summary);
  result.timing.PublishTo(&timing);
  return Check(!published.empty() &&
                   summary.encoded_bytes == published.size() &&
                   summary.encode_attempt_count == timing.attempts.size(),
               "Retained publication changed output") &&
         Empty(budget);
}

bool CheckSearchIntervals() {
  for (size_t attempts = 1; attempts <= 64; ++attempts) {
    HostStorageBound plan;
    if (!Ok(ComputeTargetSizeControlStorageBound(attempts, &plan)))
      return false;
    ResourceBudget budget(plan.peak_bytes);
    ResourceReservation job;
    if (!Ok(budget.TryReserve(plan.peak_bytes, &job)))
      return false;
    {
      ResourceContextScope context({&job, ResourceClass::kSerializer});
      TargetSizeSearchResult result;
      // Evaluator-owned default-allocator output is outside this interval-only
      // fixture. Every candidate misses tolerance, exercising all split steps.
      const TargetSizeEvaluator evaluator = [](float target,
                                               std::vector<uint8_t> *bytes,
                                               VarDctEncodingSummary *summary) {
        bytes->assign(1, 42);
        summary->encoded_bytes = 1;
        summary->selected_butteraugli_target = target;
        return Status::Ok();
      };
      if (!Ok(SearchTargetSize(
              {.target_bytes = 2, .maximum_attempts = attempts}, evaluator,
              &result)) ||
          !Check(result.attempt_count == attempts && result.search_exhausted &&
                     budget.snapshot().peak_backing_bytes <= plan.peak_bytes,
                 "Actual interval growth exceeded the preflight bound"))
        return false;
    }
    job.Reset();
    if (!Empty(budget))
      return false;
  }
  return true;
}
} // namespace

int main() {
  if (!CheckPlans() || !CheckSearchIntervals())
    return EXIT_FAILURE;
  std::unique_ptr<GpuBackend> gpu;
  if (!Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu)) ||
      !Ok(EnsureProductionMetalBackendAvailable()) || !CheckRuntime(*gpu) ||
      !CheckAdmissionFailure(*gpu) || !CheckRetainedOutput())
    return EXIT_FAILURE;
  gpu.reset();
  return Empty(DefaultResourceBudget()) &&
                 Check(DefaultResourceBudget().snapshot().peak_backing_bytes ==
                           0,
                       "Workflow allocation escaped the explicit domain")
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
