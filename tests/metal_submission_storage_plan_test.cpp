// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cstdlib>
#include <iostream>
#include <limits>

#include "gpu/metal/metal_backend_internal.h"

namespace {
using namespace gjxl;
using namespace gjxl::metal_internal;
using namespace gjxl::resource_budget_internal;

bool Check(bool good, const char *message) {
  if (!good)
    std::cerr << message << '\n';
  return good;
}
bool Plans() {
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!budget.TryReserve(1, &job).ok() || !job.ReduceCapacity(0).ok())
    return false;
  size_t ac_cases = 0, aq_cases = 0;
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    ArmNextManagedHostAllocationFailureForTest();
    for (size_t batches = 0; batches <= 33; ++batches) {
      for (size_t nonempty = 0; nonempty <= batches; ++nonempty) {
        for (bool profiling : {false, true}) {
          AcSubmissionStoragePlan p;
          if (!Check(
                  ComputeAcSubmissionStoragePlan(
                      {batches, nonempty, profiling, 1024}, &p)
                          .ok() &&
                      p.batch_capacity == batches &&
                      p.stage_capacity == (profiling ? nonempty : 0) &&
                      p.maximum_dispatches == 5 * p.stage_capacity &&
                      p.input.retained_bytes == p.input.peak_bytes &&
                      p.working.peak_bytes >=
                          p.input.peak_bytes + p.profile.recorded.peak_bytes &&
                      p.working.peak_bytes >= p.profile.resolution.peak_bytes &&
                      (p.stage_capacity != 0 ||
                       p.profile == gpu_profile_internal::
                                        SubmissionProfileStoragePlan{}),
                  "AC submission formula failed"))
            return false;
          ++ac_cases;
        }
      }
    }
    for (size_t iterations = 0; iterations <= 4; ++iterations)
      for (bool final : {false, true})
        for (bool sinks : {false, true})
          for (bool gaborish : {false, true})
            for (size_t epf = 0; epf <= 3; ++epf) {
              ResidentAqProfileInputStoragePlan p;
              if (!Check(ComputeResidentAqProfileInputStoragePlan(
                             {iterations, final, sinks, gaborish, epf}, &p)
                                 .ok() &&
                             p.score_count == iterations + size_t(final) &&
                             p.stage_capacity ==
                                 p.score_count * (47 + 11 * size_t(sinks) +
                                                  size_t(gaborish) + epf) +
                                     8 * size_t(!final) + 1 +
                                     9 * size_t(p.score_count == 0) &&
                             p.input.peak_bytes == p.input.retained_bytes &&
                             p.input.peak_bytes >=
                                 p.stage_capacity *
                                     sizeof(MetalProfiledComputeStage),
                         "Resident AQ reserve formula changed"))
                return false;
              ++aq_cases;
            }
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!Check(pending && budget.snapshot().peak_backing_bytes == 0,
               "Submission planning allocated managed backing"))
      return false;
  }
  AcSubmissionStoragePlan ac;
  ResidentAqProfileInputStoragePlan aq;
  if (!ComputeAcSubmissionStoragePlan({7, 7, true}, &ac).ok() ||
      !ComputeResidentAqProfileInputStoragePlan({4}, &aq).ok())
    return false;
  const auto old_ac = ac;
  const auto old_aq = aq;
  const size_t huge = std::numeric_limits<size_t>::max();
  for (const auto bad : {AcSubmissionStorageOptions{0, 1, true},
                         {huge, 0, false},
                         {huge, huge, true},
                         {1, 1, true, huge}})
    if (!Check(!ComputeAcSubmissionStoragePlan(bad, &ac).ok() && ac == old_ac,
               "Invalid AC plan changed output"))
      return false;
  for (const auto bad : {ResidentAqProfileInputOptions{5},
                         {huge},
                         {1, true, false, false, 4},
                         {1, true, false, false, huge}})
    if (!Check(!ComputeResidentAqProfileInputStoragePlan(bad, &aq).ok() &&
                   aq == old_aq,
               "Invalid AQ plan changed output"))
      return false;
  if (!Check(!ComputeAcSubmissionStoragePlan({}, nullptr).ok() &&
                 !ComputeResidentAqProfileInputStoragePlan({}, nullptr).ok(),
             "Null plan output accepted"))
    return false;
  job.Reset();
  std::cout << "AC submission shapes: " << ac_cases
            << "; resident AQ reserve shapes: " << aq_cases << '\n';
  return budget.snapshot().committed_bytes() == 0;
}

bool AppendGuard() {
  struct Context {
    size_t value;
  };
  size_t cases = 0;
  for (size_t context_capacity : {size_t{0}, size_t{1}, size_t{7}})
    for (size_t stage_capacity : {size_t{0}, size_t{1}, size_t{7}}) {
      ResourceBudget budget(4096);
      ResourceReservation job;
      if (!budget.TryReserve(4096, &job).ok())
        return false;
      {
        ResourceContextScope scope({&job, ResourceClass::kPreparation});
        ManagedVector<Context> contexts;
        ManagedVector<MetalProfiledComputeStage> stages;
        contexts.reserve(context_capacity);
        stages.reserve(stage_capacity);
        const size_t count = std::min(context_capacity, stage_capacity);
        for (size_t i = 0; i < count; ++i)
          AppendMetalProfileStage(contexts, stages, Context{i}, {});
        const auto *contexts_before = contexts.data();
        const auto *stages_before = stages.data();
        const auto bytes_before = budget.snapshot().peak_backing_bytes;
        bool typed = false;
        ArmNextManagedHostAllocationFailureForTest();
        try {
          AppendMetalProfileStage(contexts, stages, Context{99}, {});
        } catch (const ManagedAllocationFailure &failure) {
          typed = failure.status().resource_plan_exceeded();
        }
        const bool pending = ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        if (!Check(typed && pending && contexts.data() == contexts_before &&
                       stages.data() == stages_before &&
                       contexts.size() == count && stages.size() == count &&
                       budget.snapshot().peak_backing_bytes == bytes_before,
                   "Capacity exhaustion grew or corrupted callback arrays"))
          return false;
        for (size_t i = 0; i < count; ++i)
          if (!Check(
                  stages[i].context == &contexts[i] &&
                      static_cast<const Context *>(stages[i].context)->value ==
                          i,
                  "An earlier stage context pointer changed"))
            return false;
        if (count != 0) {
          stages.pop_back(); // Reject unequal logical arrays even with spare
                             // capacity.
          try {
            AppendMetalProfileStage(contexts, stages, Context{99}, {});
            return Check(false, "Mismatched callback arrays accepted");
          } catch (const ManagedAllocationFailure &failure) {
            if (!failure.status().resource_plan_exceeded())
              return false;
          }
        }
      }
      job.Reset();
      if (!Check(budget.snapshot().committed_bytes() == 0,
                 "Callback array backing leaked"))
        return false;
      ++cases;
    }
  std::cout << "Stable-context capacity guard cases: " << cases << '\n';
  return true;
}
} // namespace

int main() { return Plans() && AppendGuard() ? EXIT_SUCCESS : EXIT_FAILURE; }
