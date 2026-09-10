// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <chrono>
#include <optional>

#include "core/execution_domain.h"

namespace gjxl::thread_budget_internal {

struct CpuExecutionTiming {
  uint64_t initial_queue_nanoseconds = 0;
  uint64_t resume_queue_nanoseconds = 0;
  uint64_t blocked_nanoseconds = 0;
};
struct CpuParticipationObserver {
  void* context = nullptr;
  void (*enter)(void*) noexcept = nullptr;
  void (*leave)(void*) noexcept = nullptr;
};
struct CpuExecutionContext {
  cpu_budget_internal::CpuBudget domain;
  // Per-image capacity includes dormant caller slots, so resuming a caller
  // never waits for its own descendants. Only the global permit is suspended.
  cpu_budget_internal::CpuBudget job;
  cpu_budget_internal::CpuPermit* permit = nullptr;
  CpuExecutionTiming* timing = nullptr;  // Calling-thread wall spans, not worker sums.
  CpuParticipationObserver observer;
};
inline thread_local CpuExecutionContext current_cpu_execution;

inline bool HasCpuParticipation() noexcept {
  return current_cpu_execution.permit != nullptr && current_cpu_execution.permit->active();
}

/// Enter after complete memory admission. C adapters can wrap converted input
/// and final copy; nested same-domain codec calls inherit this participation.
class CpuExecutionScope {
public:
  CpuExecutionScope() = default;
  CpuExecutionScope(const CpuExecutionScope&) = delete;
  CpuExecutionScope& operator=(const CpuExecutionScope&) = delete;
  ~CpuExecutionScope() {
    if (owns_) {
      if (current_cpu_execution.observer.leave)
        current_cpu_execution.observer.leave(current_cpu_execution.observer.context);
      current_cpu_execution = previous_;
    }
  }
  [[nodiscard]] Status Start(std::shared_ptr<const ExecutionDomain> domain = {},
                            size_t per_image_limit = 0, bool collect_timing = false) try {
    if (started_) return Status::InvalidArgument("CPU execution scope was already started");
    const ExecutionDomain* selected = domain ? domain.get() :
      resource_budget_internal::CurrentResourceContext().domain;
    if (selected == nullptr) { domain = ExecutionDomain::Default(); selected = domain.get(); }
    const auto budget = selected->cpu_budget_;
    if (current_cpu_execution.permit != nullptr) {
      if (!HasCpuParticipation() || !budget.Shares(*current_cpu_execution.permit))
        return Status::InvalidArgument("Nested CPU execution uses a different or suspended domain");
      if (collect_timing) admitted_at_ = std::chrono::steady_clock::now();
      started_ = true;
      return Status::Ok();
    }
    const size_t job_limit = per_image_limit == 0 ? budget.limit() :
      std::min(per_image_limit, budget.limit());
    cpu_budget_internal::CpuBudget job(job_limit);
    Status status = job.Acquire(&job_permit_);
    if (!status.ok()) return status;
    status = budget.Acquire(&permit_);
    if (!status.ok()) { job_permit_.Reset(); return status; }
    if (collect_timing) admitted_at_ = std::chrono::steady_clock::now();
    timing_.initial_queue_nanoseconds = permit_.wait_nanoseconds();
    previous_ = current_cpu_execution;
    current_cpu_execution = {budget, std::move(job), &permit_, collect_timing ? &timing_ : nullptr, {}};
    started_ = true;
    owns_ = true;
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Unable to allocate CPU execution state");
  }
  [[nodiscard]] CpuExecutionTiming timing() const noexcept { return timing_; }
  /// Calling-thread observation of successful initial admission, only when
  /// timing was requested. A nested borrowing scope observes its own Start.
  [[nodiscard]] std::optional<std::chrono::steady_clock::time_point> admitted_at() const noexcept {
    return admitted_at_;
  }

private:
  CpuExecutionContext previous_;
  cpu_budget_internal::CpuPermit permit_;
  cpu_budget_internal::CpuPermit job_permit_;
  CpuExecutionTiming timing_;
  std::optional<std::chrono::steady_clock::time_point> admitted_at_;
  bool started_ = false;
  bool owns_ = false;
};

/// Suspend for the entire blocking boundary, including any call_once wait.
/// Image callers yield their global slot and requeue; already-created workers
/// retain protected capacity while inactive so nested joins cannot multiply
/// dormant worker threads. The immutable budget is retained. Public execution never cancels a resume;
/// synchronization failure is not recoverable by running without a permit.
class CpuSuspension {
public:
  CpuSuspension() : context_(current_cpu_execution), suspended_(HasCpuParticipation()) {
    if (!suspended_) return;
    if (context_.observer.leave) context_.observer.leave(context_.observer.context);
    if (context_.timing) begin_ = std::chrono::steady_clock::now();
    worker_ = context_.permit->is_worker();
    if (worker_) context_.permit->SuspendWorker();
    else context_.permit->Reset();
  }
  ~CpuSuspension() {
    if (!suspended_) return;
    if (context_.timing) context_.timing->blocked_nanoseconds += static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin_).count());
    if (worker_) {
      // This thread still owns protected capacity; resumption takes no new slot
      // and cannot wait behind a caller that depends on this joined worker.
      context_.permit->ResumeWorker();
    } else {
      const Status status = context_.domain.Acquire(context_.permit);
      if (!status.ok()) std::terminate();
      if (context_.timing) context_.timing->resume_queue_nanoseconds += context_.permit->wait_nanoseconds();
    }
    if (context_.observer.enter) context_.observer.enter(context_.observer.context);
  }
  CpuSuspension(const CpuSuspension&) = delete;
  CpuSuspension& operator=(const CpuSuspension&) = delete;

private:
  CpuExecutionContext context_;
  std::chrono::steady_clock::time_point begin_;
  bool suspended_;
  bool worker_ = false;
};

/// Keep the suspension outside the entire join boundary, including launch-
/// failure cleanup. Resumption happens before any serial fallback or result scan.
template <typename Workers>
void JoinCpuWorkers(Workers& workers) {
  if (workers.empty()) return;
  CpuSuspension suspension;
  for (auto& worker : workers) worker.join();
}

/// Reserve before worker construction, including at nested automatic loops.
/// Both per-image and aggregate limits are nonblocking for additional workers;
/// the caller always retains capacity to execute the remaining tasks itself.
class CpuWorkerGroup {
public:
  explicit CpuWorkerGroup(size_t desired_participants)
    : context_(current_cpu_execution), participants_(desired_participants) {
    if (!HasCpuParticipation() || desired_participants == 0) return;
    enabled_ = true;
    job_slots_ = context_.job.TryReserveWorkers(desired_participants - 1);
    domain_slots_ = context_.domain.TryReserveWorkers(job_slots_.count());
    job_slots_.KeepAtMost(domain_slots_.count());
    participants_ = 1 + domain_slots_.count();
  }
  [[nodiscard]] size_t participants() const noexcept { return participants_; }
  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

private:
  friend class CpuWorkerScope;
  CpuExecutionContext context_;
  cpu_budget_internal::CpuWorkerReservation domain_slots_;
  cpu_budget_internal::CpuWorkerReservation job_slots_;
  size_t participants_;
  bool enabled_ = false;
};

/// Installs a pre-reserved worker. The already participating caller does not
/// take a second slot when it executes the same loop body.
class CpuWorkerScope {
public:
  explicit CpuWorkerScope(const CpuWorkerGroup* group) {
    if (group == nullptr || !group->enabled_) return;
    if (HasCpuParticipation() && group->context_.domain.Shares(*current_cpu_execution.permit)) return;
    domain_permit_ = group->domain_slots_.Take();
    job_permit_ = group->job_slots_.Take();
    // Only the pre-granted number of threads may be launched. Do not execute
    // unaccounted work if an internal caller violates that invariant.
    if (!domain_permit_.active() || !job_permit_.active()) std::terminate();
    previous_ = current_cpu_execution;
    current_cpu_execution = group->context_;
    current_cpu_execution.permit = &domain_permit_;
    current_cpu_execution.timing = nullptr;
    if (current_cpu_execution.observer.enter)
      current_cpu_execution.observer.enter(current_cpu_execution.observer.context);
    owns_ = true;
  }
  ~CpuWorkerScope() {
    if (!owns_) return;
    if (current_cpu_execution.observer.leave)
      current_cpu_execution.observer.leave(current_cpu_execution.observer.context);
    current_cpu_execution = previous_;
  }
  CpuWorkerScope(const CpuWorkerScope&) = delete;
  CpuWorkerScope& operator=(const CpuWorkerScope&) = delete;

private:
  CpuExecutionContext previous_;
  cpu_budget_internal::CpuPermit domain_permit_;
  cpu_budget_internal::CpuPermit job_permit_;
  bool owns_ = false;
};

}  // namespace gjxl::thread_budget_internal
