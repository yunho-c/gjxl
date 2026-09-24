// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/serializer_storage_plan.h"
#include "codestream/storage.h"
#include "core/parallel_work_internal.h"
#include <algorithm>
#include <thread>
#include <type_traits>
namespace gjxl::codestream_internal {
template <typename Function>
Status
RunParallelSections(size_t count, Function &&function,
                    size_t maximum_workers = kSerializerMaximumSectionWorkers) {
  const auto invoke = [&](size_t index, size_t worker_index) -> Status {
    if constexpr (std::is_invocable_r_v<Status, Function &, size_t, size_t>) {
      return function(index, worker_index);
    } else {
      return function(index);
    }
  };
  if (count == 0)
    return Status::Ok();
  if (thread_budget_internal::InExplicitParallelScope()) {
    for (size_t index = 0; index < count; ++index) {
      Status status = invoke(index, 0);
      if (!status.ok())
        return status;
    }
    return Status::Ok();
  }
  const size_t hardware_workers =
      std::max<size_t>(std::thread::hardware_concurrency(), 1);
  const size_t automatic_worker_count =
      std::min(count, std::min(maximum_workers, hardware_workers));
  const size_t cpu_thread_count = thread_budget_internal::CpuThreadCount();
  auto *const participant_tracker =
      thread_budget_internal::ParticipantTracker();
  const auto resource_context =
      resource_budget_internal::CurrentResourceContext();
  thread_budget_internal::CpuWorkerGroup cpu_workers(
      cpu_thread_count == 0
          ? automatic_worker_count
          : std::min(automatic_worker_count, cpu_thread_count));
  const size_t participant_count = cpu_workers.participants();
  if (participant_count == 1) {
    thread_budget_internal::ParallelScope scope(
        cpu_thread_count, participant_tracker, resource_context, &cpu_workers);
    for (size_t index = 0; index < count; ++index) {
      Status status = invoke(index, 0);
      if (!status.ok())
        return status;
    }
    return Status::Ok();
  }

  const size_t spawned_worker_count =
      cpu_thread_count == 0 && !cpu_workers.enabled() ? participant_count
                                                      : participant_count - 1;
  constexpr thread_budget_internal::ParallelWorkErrors errors{
      .allocation = "Codestream assembly allocation failed",
      .unexpected = "Codestream section worker failed unexpectedly",
      .length_code = StatusCode::kOutOfMemory,
      .length = "Codestream assembly allocation failed",
      .launch_allocation = "Codestream assembly allocation failed",
      .launch_action =
          thread_budget_internal::LaunchFailureAction::kReturnError,
      .launch = "Unable to start codestream section workers",
  };
  return thread_budget_internal::RunParallelWork<Storage>(
      count, cpu_workers, spawned_worker_count,
      thread_budget_internal::WorkerLaunchSite::kSerializerSections, errors,
      invoke);
}
} // namespace gjxl::codestream_internal
