// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <memory>
#include <thread>

#include "core/cpu_budget.h"
#include "core/resource_context.h"

namespace gjxl {
namespace codestream_internal {
class WorkflowAdmission;
}
namespace thread_budget_internal { class CpuExecutionScope; }

inline constexpr size_t kMaximumDomainCpuParticipants = 256;

struct ExecutionDomainOptions {
  /// Maximum managed backing/reservation capacity. Zero is unlimited but
  /// accounted. This is not an RSS limit; caller-owned input/published output,
  /// immutable backend setup, driver internals, stacks and small controls are
  /// excluded. CPU participation is limited separately below.
  size_t managed_memory_bytes = 0;
  /// Aggregate executing callers and workers in this domain. Zero resolves to
  /// hardware concurrency (at least one, at most 256); positive values <= 256
  /// set an explicit limit. Per-image CPU options remain separate upper bounds.
  size_t cpu_participant_limit = 0;
};

struct ExecutionDomainSnapshot {
  size_t live_requested_bytes = 0;
  size_t live_capacity_bytes = 0;
  size_t idle_capacity_bytes = 0;
  size_t reserved_unbacked_bytes = 0;
  size_t peak_backing_bytes = 0;
  size_t peak_committed_bytes = 0;
  size_t active_reservations = 0;
  size_t waiting_requests = 0;
  size_t effective_cpu_participant_limit = 0;
  size_t active_cpu_participants = 0;
  size_t reserved_cpu_workers = 0;
  size_t suspended_cpu_workers = 0;
  size_t waiting_cpu_callers = 0;
  size_t peak_cpu_participants = 0;
  size_t peak_cpu_protected_slots = 0;
};

/// Immutable shared memory-admission and CPU-participation domain. Copies of a
/// shared handle use the same allowances across C/C++, backends and batch drivers.
/// Allocations may outlive their producing call; tickets retain the accounting.
class ExecutionDomain final {
public:
  ExecutionDomain(const ExecutionDomain &) = delete;
  ExecutionDomain &operator=(const ExecutionDomain &) = delete;

  [[nodiscard]] static Status Create(ExecutionDomainOptions options,
                                     std::shared_ptr<const ExecutionDomain> *out) {
    if (out == nullptr)
      return Status::InvalidArgument("Execution domain output is null");
    if (options.cpu_participant_limit > kMaximumDomainCpuParticipants)
      return Status::InvalidArgument("Execution domain CPU limit must be zero or at most 256");
    try {
      auto candidate = std::shared_ptr<const ExecutionDomain>(new ExecutionDomain(
          options, resource_budget_internal::ResourceBudget(options.managed_memory_bytes)));
      *out = std::move(candidate);
      return Status::Ok();
    } catch (const std::bad_alloc &) {
      return Status::OutOfMemory("Unable to allocate execution domain");
    }
  }

  /// The same default is used by every null workflow/context domain. It shares
  /// the preexisting fallback ledger, including legacy explicit backend caches.
  [[nodiscard]] static std::shared_ptr<const ExecutionDomain> Default() {
    static const std::shared_ptr<const ExecutionDomain> domain(
        new ExecutionDomain({}, resource_budget_internal::DefaultResourceBudget()));
    return domain;
  }

  [[nodiscard]] ExecutionDomainOptions options() const noexcept { return options_; }
  /// Memory and CPU counters are each internally consistent, but the two sets
  /// are sampled separately: this is not an atomic cross-resource snapshot.
  [[nodiscard]] ExecutionDomainSnapshot snapshot() const {
    const auto s = budget_.snapshot();
    const auto c = cpu_budget_.snapshot();
    return {s.total.live_requested_bytes, s.total.live_capacity_bytes, s.total.idle_capacity_bytes,
            s.reserved_unbacked_bytes,    s.peak_backing_bytes,        s.peak_committed_bytes,
            s.open_reservations,          s.waiting_requests,
            cpu_budget_.limit(), c.active_participants, c.reserved_workers, c.suspended_workers,
            c.waiting_callers, c.peak_active_participants, c.peak_protected_slots};
  }

private:
  friend class codestream_internal::WorkflowAdmission;
  friend class thread_budget_internal::CpuExecutionScope;
  ExecutionDomain(ExecutionDomainOptions options, resource_budget_internal::ResourceBudget budget)
      : options_(options), budget_(std::move(budget)), cpu_budget_(
          options.cpu_participant_limit != 0 ? options.cpu_participant_limit :
          std::clamp<size_t>(std::thread::hardware_concurrency(), 1, kMaximumDomainCpuParticipants)) {}
  const ExecutionDomainOptions options_;
  const resource_budget_internal::ResourceBudget budget_;
  const cpu_budget_internal::CpuBudget cpu_budget_;
};
} // namespace gjxl
