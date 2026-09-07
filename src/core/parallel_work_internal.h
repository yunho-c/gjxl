// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <thread>

#include "core/managed_allocator.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"

namespace gjxl::thread_budget_internal {

enum class LaunchFailureAction { kReturnError, kRetrySerial };

// Literal descriptions keep policy setup allocation-free. Status strings are
// constructed only on the corresponding error path.
struct ParallelWorkErrors {
  const char* allocation;
  const char* unexpected;
  StatusCode length_code;
  const char* length;
  const char* launch_allocation;
  LaunchFailureAction launch_action;
  const char* launch = nullptr;
};

// Only the parallel branch lives here. Callers retain participant selection,
// admission and serial fast paths, including their exception boundaries. The
// admitted group must outlive this call; Storage retains each stage's allocator
// owner. Workers use [0, spawned_workers), and a participating caller uses the
// next index, just as the per-worker scratch planners expect.
template <template <typename> class Storage, typename Function>
Status RunParallelWork(
    size_t count, const CpuWorkerGroup& group, size_t spawned_workers,
    WorkerLaunchSite site, const ParallelWorkErrors& errors,
    Function&& function) {
  assert(count != 0 && group.participants() > 1);
  assert(spawned_workers == group.participants() ||
         spawned_workers + 1 == group.participants());
  assert(errors.launch_action == LaunchFailureAction::kRetrySerial ||
         errors.launch != nullptr);
  const size_t cpu_thread_count = CpuThreadCount();
  auto* const tracker = ParticipantTracker();
  const auto resources = resource_budget_internal::CurrentResourceContext();

  // Keep setup outside the launch exception boundary: allocation failures here
  // are translated by the existing component-level handlers.
  Storage<Status> statuses(count);
  std::atomic<size_t> next_index{0};
  Storage<std::thread> workers;
  workers.reserve(spawned_workers);
  const auto run_worker = [&](size_t worker_index) {
    ParallelScope scope(cpu_thread_count, tracker, resources, &group);
    while (true) {
      const size_t index = next_index.fetch_add(1, std::memory_order_relaxed);
      if (index >= count) break;
      try {
        statuses[index] = function(index, worker_index);
      } catch (const resource_budget_internal::ManagedAllocationFailure& error) {
        statuses[index] = error.status();
      } catch (const std::bad_alloc&) {
        statuses[index] = Status::OutOfMemory(errors.allocation);
      } catch (const std::length_error&) {
        statuses[index] = Status(errors.length_code, errors.length);
      } catch (...) {
        statuses[index] = Status::Internal(errors.unexpected);
      }
    }
  };
  const auto stop_and_join = [&] {
    next_index.store(count, std::memory_order_relaxed);
    JoinCpuWorkers(workers);
  };
  try {
    for (size_t worker = 0; worker < spawned_workers; ++worker) {
      LaunchWorker(workers, site, worker, run_worker, worker);
    }
  } catch (const resource_budget_internal::ManagedAllocationFailure& error) {
    stop_and_join();
    return error.status();
  } catch (const std::bad_alloc&) {
    stop_and_join();
    return Status::OutOfMemory(errors.launch_allocation);
  } catch (const std::system_error&) {
    stop_and_join();
    if (errors.launch_action == LaunchFailureAction::kReturnError) {
      return Status::Internal(errors.launch);
    }
    // JoinCpuWorkers resumes caller participation before fallback. Deliberately
    // retry every task, including ones completed by a partially launched group,
    // with the original serial exception behavior (no worker adapter/scope).
    for (size_t index = 0; index < count; ++index) {
      Status status = function(index, 0);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  } catch (...) {
    stop_and_join();
    throw;
  }
  try {
    if (spawned_workers < group.participants()) run_worker(spawned_workers);
  } catch (...) {
    // Even an exception while constructing a caller-side error Status must not
    // unwind through joinable threads.
    stop_and_join();
    throw;
  }
  JoinCpuWorkers(workers);
  for (const Status& status : statuses) {
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

}  // namespace gjxl::thread_budget_internal
