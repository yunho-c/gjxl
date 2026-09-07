// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <memory>

#include "core/resource_context.h"

namespace gjxl {
namespace codestream_internal {
class WorkflowAdmission;
}

struct ExecutionDomainOptions {
  /// Maximum managed backing/reservation capacity. Zero is unlimited but
  /// accounted. This is not an RSS limit; caller-owned input/published output,
  /// immutable backend setup, driver internals, stacks and small controls are
  /// excluded. CPU scheduling is configured separately by encoding options.
  size_t managed_memory_bytes = 0;
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
};

/// Immutable shared memory-admission domain. Copies of a shared handle use one
/// allowance across C/C++, backends and batch drivers. Allocations may outlive
/// their producing call; their tickets keep the domain's accounting alive.
class ExecutionDomain final {
public:
  ExecutionDomain(const ExecutionDomain &) = delete;
  ExecutionDomain &operator=(const ExecutionDomain &) = delete;

  [[nodiscard]] static Status Create(ExecutionDomainOptions options,
                                     std::shared_ptr<const ExecutionDomain> *out) {
    if (out == nullptr)
      return Status::InvalidArgument("Execution domain output is null");
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
  [[nodiscard]] ExecutionDomainSnapshot snapshot() const {
    const auto s = budget_.snapshot();
    return {s.total.live_requested_bytes, s.total.live_capacity_bytes, s.total.idle_capacity_bytes,
            s.reserved_unbacked_bytes,    s.peak_backing_bytes,        s.peak_committed_bytes,
            s.open_reservations,          s.waiting_requests};
  }

private:
  friend class codestream_internal::WorkflowAdmission;
  ExecutionDomain(ExecutionDomainOptions options, resource_budget_internal::ResourceBudget budget)
      : options_(options), budget_(std::move(budget)) {}
  const ExecutionDomainOptions options_;
  const resource_budget_internal::ResourceBudget budget_;
};
} // namespace gjxl
