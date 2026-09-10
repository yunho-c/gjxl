// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <new>
#include <system_error>
#include <utility>

namespace gjxl::thread_budget_internal {

enum class WorkerLaunchSite { kColorRows, kInitialQuantization, kForwardTransforms,
                              kCoefficientOrders, kSerializerSections, kBatchDriver };
enum class WorkerLaunchFailureKind { kSystemError, kBadAlloc };

// Deterministic fault at a real thread-construction boundary. The earlier
// workers in that loop are genuinely constructed; no OS resource exhaustion
// is attempted. State is caller-thread-local and is not inherited by workers.
struct WorkerLaunchFaultForTesting {
  WorkerLaunchSite site;
  size_t fail_before_worker = 0;
  WorkerLaunchFailureKind kind = WorkerLaunchFailureKind::kSystemError;
  size_t launched_in_group = 0;
  bool triggered = false;
  void* context = nullptr;
  void (*before_failure)(void*) noexcept = nullptr;
};
inline thread_local WorkerLaunchFaultForTesting* worker_launch_fault_for_testing = nullptr;

inline WorkerLaunchFaultForTesting* SetWorkerLaunchFaultForTesting(WorkerLaunchFaultForTesting* fault) noexcept {
  return std::exchange(worker_launch_fault_for_testing, fault);
}
class WorkerLaunchFaultScopeForTesting {
public:
  explicit WorkerLaunchFaultScopeForTesting(WorkerLaunchFaultForTesting* fault) noexcept
    : previous_(SetWorkerLaunchFaultForTesting(fault)) {}
  ~WorkerLaunchFaultScopeForTesting() { (void)SetWorkerLaunchFaultForTesting(previous_); }
  WorkerLaunchFaultScopeForTesting(const WorkerLaunchFaultScopeForTesting&) = delete;
  WorkerLaunchFaultScopeForTesting& operator=(const WorkerLaunchFaultScopeForTesting&) = delete;
private:
  WorkerLaunchFaultForTesting* previous_;
};

template<class Workers, class... Args>
void LaunchWorker(Workers& workers, WorkerLaunchSite site, size_t index, Args&&... args) {
  auto* fault = worker_launch_fault_for_testing;
  const bool selected = fault != nullptr && !fault->triggered && fault->site == site;
  if (selected) {
    if (index == 0) fault->launched_in_group = 0;
    if (index == fault->fail_before_worker) {
      fault->triggered = true;
      if (fault->before_failure != nullptr) fault->before_failure(fault->context);
      if (fault->kind == WorkerLaunchFailureKind::kBadAlloc) throw std::bad_alloc();
      throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
    }
  }
  workers.emplace_back(std::forward<Args>(args)...);
  if (selected) ++fault->launched_in_group;
}

} // namespace gjxl::thread_budget_internal
