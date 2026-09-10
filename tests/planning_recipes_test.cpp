// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>

#include "codec/color_transform.h"
#include "codec/frontend_dispatch_internal.h"
#include "codec/frontend_storage_plan.h"
#include "codestream/workflow_publication_storage_plan.h"
#include "core/image_buffer.h"
#include "core/thread_budget.h"

namespace {
using namespace gjxl;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::frontend_storage_internal;
using namespace gjxl::codestream_internal;
namespace dispatch = gjxl::frontend_dispatch_internal;

bool Check(bool good, const char* message) {
  if (!good) std::cerr << message << '\n';
  return good;
}
bool Ok(const Status& status) {
  return Check(status.ok(), status.message().data());
}

bool DispatchPolicy() {
  for (auto policy : {dispatch::kColor, dispatch::kInitialQuant, dispatch::kForwardTransform}) {
    // Fixed expectations preserve the existing scheduling policy independently
    // of the helper implementation. Zero hardware availability means one CPU.
    if (!Check(policy.minimum_parallel_work == 65536 &&
          policy.Participants(0, 65536, 0, 16) == 0 &&
          policy.Participants(64, 65535, 0, 16) == 1 &&
          policy.Participants(64, 65536, 0, 0) == 1 &&
          policy.Participants(64, 65537, 0, 1) == 1 &&
          policy.Participants(64, 65536, 1, 16) == 1 &&
          policy.Participants(64, 65536, 2, 16) == 2 &&
          policy.Participants(64, 65536, 8, 16) == 8 &&
          policy.Participants(3, 65536, 8, 16) == 3 &&
          policy.Participants(64, 65536, 8, 4) == 4,
          "Dispatch threshold, task bound or CPU cap changed")) return false;
    for (size_t tasks : {0ul, 1ul, 2ul, 63ul})
      for (size_t work : {65535ul, 65536ul, 65537ul})
        for (size_t limit : {0ul, 1ul, 2ul, 8ul})
          for (size_t hardware : {0ul, 1ul, 2ul, 16ul}) {
            const auto actual = policy.Participants(tasks, work, limit, hardware);
            const auto planned = policy.MaximumParticipants(tasks, work, limit);
            for (bool managed : {false, true})
              if (!Check(actual <= planned &&
                    dispatch::SpawnedWorkers(actual, limit != 0 || managed) <=
                      dispatch::SpawnedWorkers(planned, limit != 0),
                    "Runtime dispatcher exceeds its planned thread backing")) return false;
          }
  }
  return Check(dispatch::kColor.MaximumParticipants(64, 65536, 0) == 12 &&
      dispatch::kInitialQuant.MaximumParticipants(64, 65536, 0) == 12 &&
      dispatch::kForwardTransform.MaximumParticipants(64, 65536, 0) == 8 &&
      dispatch::SpawnedWorkers(0, true) == 0 && dispatch::SpawnedWorkers(1, false) == 0 &&
      dispatch::SpawnedWorkers(8, false) == 8 && dispatch::SpawnedWorkers(8, true) == 7,
      "Stage caps or caller participation changed");
}

bool PublicationRecipe() {
  const HostStorageBound bytes{13, 17};
  for (size_t attempts : {1ul, 2ul, 64ul})
    for (bool search : {false, true})
      for (bool timing : {false, true})
        for (size_t scores : {0ul, 3ul}) {
          if (!search && attempts != 1) continue;
          WorkflowPublicationStoragePlan p;
          ArmNextManagedHostAllocationFailureForTest();
          const auto status = ComputeWorkflowPublicationStoragePlan(bytes, scores, attempts, search, timing, &p);
          const bool pending = ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
          const size_t score_bytes = scores * sizeof(double);
          const size_t timing_bytes = timing ? attempts * sizeof(VarDctEncodingAttemptTiming) : 0;
          const HostStorageBound best = search && attempts > 1
            ? HostStorageBound{13 + score_bytes, 17 + score_bytes} : HostStorageBound{};
          if (!Ok(status) || !Check(pending &&
                p.scores == HostStorageBound{score_bytes, score_bytes} &&
                p.timing == HostStorageBound{timing_bytes, timing_bytes} && p.retained_best == best &&
                p.output == HostStorageBound{13 + score_bytes + timing_bytes, 17 + score_bytes + timing_bytes} &&
                (p.search_control.peak_bytes != 0) == search,
                "Publication owners or one-attempt search control changed")) return false;
        }
  const WorkflowPublicationStoragePlan sentinel{.scores={5,7}, .output={11,13}};
  auto p = sentinel;
  for (size_t attempts : {0ul, 65ul, std::numeric_limits<size_t>::max()})
    if (!Check(!ComputeWorkflowPublicationStoragePlan(bytes, 3, attempts, true, true, &p).ok() && p == sentinel,
               "Invalid publication attempts changed output")) return false;
  return Check(
      ComputeWorkflowPublicationStoragePlan(bytes, SIZE_MAX, 1, false, false, &p).code() == StatusCode::kOutOfMemory &&
      p == sentinel &&
      ComputeWorkflowPublicationStoragePlan({SIZE_MAX, SIZE_MAX}, 3, 1, false, false, &p).code() == StatusCode::kOutOfMemory &&
      p == sentinel &&
      !ComputeWorkflowPublicationStoragePlan(bytes, 3, 2, false, false, &p).ok() && p == sentinel &&
      !ComputeWorkflowPublicationStoragePlan(bytes, 3, 1, false, false, nullptr).ok(),
      "Publication overflow/validation failed to preserve output");
}

bool TightColorReservation() {
  const Extent2D extent{256, 256};
  Image3FBuffer input(extent), reference(extent), output(extent);
  for (size_t c = 0; c < 3; ++c)
    std::fill(input.plane(c).begin(), input.plane(c).end(), 0.1f * (c + 1));
  {
    thread_budget_internal::EncodeScope serial(1);
    if (!Ok(LinearRgbToOpsin(input.const_view(), 255.0f, reference.view()))) return false;
  }
  for (size_t limit : {0ul, 1ul, 2ul, 8ul}) {
    ColorTransformStoragePlan plan;
    if (!Ok(ComputeColorTransformStoragePlan(extent, extent, false, limit, &plan))) return false;
    if (limit == 2 && !Check(plan.working.peak_bytes == 3 * 256 * 256 * sizeof(float) +
          256 * sizeof(Status) + sizeof(std::thread), "Explicit two-participant plan includes the caller as a thread"))
      return false;
    // Admission must reject the declared plan before any physical allocation.
    ResourceBudget too_small(plan.working.peak_bytes - 1);
    ResourceReservation denied;
    const auto status = too_small.Reserve(plan.working.peak_bytes, &denied);
    if (!Check(status.code() == StatusCode::kOutOfMemory && !status.resource_plan_exceeded() &&
          too_small.snapshot().peak_backing_bytes == 0, "Insufficient admission was not rejected cleanly")) return false;
    for (bool nested : {false, true}) {
      ResourceBudget budget(plan.working.peak_bytes);
      ResourceReservation job;
      if (!Ok(budget.Reserve(plan.working.peak_bytes, &job))) return false;
      {
        ResourceContextScope resources({&job, ResourceClass::kPreparation});
        thread_budget_internal::EncodeScope cpu(limit);
        if (nested) {
          thread_budget_internal::ParallelScope parent(limit, nullptr, CurrentResourceContext());
          if (!Ok(LinearRgbToOpsin(input.const_view(), 255.0f, output.view()))) return false;
        } else if (!Ok(LinearRgbToOpsin(input.const_view(), 255.0f, output.view()))) return false;
      }
      if (!Check(budget.snapshot().peak_backing_bytes <= plan.working.peak_bytes &&
            budget.snapshot().total.backing_count == 0, "Color execution exceeded or leaked its reservation")) return false;
      for (size_t c = 0; c < 3; ++c)
        if (!Check(std::ranges::equal(output.plane(c), reference.plane(c)), "Dispatcher changed color output")) return false;
      job.Reset();
      if (!Check(budget.snapshot().committed_bytes() == 0, "Color reservation leaked")) return false;
    }
  }
  return true;
}
} // namespace

int main() {
  return DispatchPolicy() && PublicationRecipe() && TightColorReservation() ? EXIT_SUCCESS : EXIT_FAILURE;
}
