// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "core/resource_budget.h"

namespace gjxl::resource_budget_internal {

/// Shared accounting for calls without an explicit admitted context. Unlimited
/// by design; never interpret this fallback as whole-workflow admission.
[[nodiscard]] inline ResourceBudget& DefaultResourceBudget() {
  static ResourceBudget budget;
  return budget;
}

struct ResourceContext {
  // Borrowed until the synchronous work and all propagated workers finish.
  ResourceReservation* reservation = nullptr;
  ResourceClass resource_class = ResourceClass::kUnclassified;
  // Opt in at encoder entry points, not when caller-owned inputs are created.
  // An explicit reservation also enables managed host allocation.
  bool track_host_allocations = false;
};

inline thread_local ResourceContext current_resource_context;

[[nodiscard]] inline ResourceContext CurrentResourceContext() noexcept {
  return current_resource_context;
}

/// Explicit context propagation for synchronous entry points and joined workers.
/// Installing this scope does not admit a job or extend reservation lifetime.
class ResourceContextScope {
public:
  explicit ResourceContextScope(ResourceContext context) noexcept
    : previous_(current_resource_context) {
    current_resource_context = context;
  }
  ~ResourceContextScope() { current_resource_context = previous_; }
  ResourceContextScope(const ResourceContextScope&) = delete;
  ResourceContextScope& operator=(const ResourceContextScope&) = delete;
private:
  ResourceContext previous_;
};

class ResourceClassScope {
public:
  explicit ResourceClassScope(ResourceClass resource_class) noexcept
    : previous_(current_resource_context.resource_class) {
    current_resource_context.resource_class = resource_class;
  }
  ~ResourceClassScope() { current_resource_context.resource_class = previous_; }
  ResourceClassScope(const ResourceClassScope&) = delete;
  ResourceClassScope& operator=(const ResourceClassScope&) = delete;
private:
  ResourceClass previous_;
};

/// Enables managed host storage for one encoder call, preserving its domain.
/// Only containers using the managed allocator are covered by this scope.
class ManagedHostScope {
public:
  explicit ManagedHostScope(ResourceClass resource_class) noexcept
    : scope_({current_resource_context.reservation, resource_class, true}) {}
private:
  ResourceContextScope scope_;
};

/// A backing owner calls this before allocation, then commits after success.
/// Outside an admitted context each backing has its own unbounded-domain charge;
/// within one, allocation cannot silently escape an insufficient plan.
[[nodiscard]] inline Status PrepareResourceAllocation(
  size_t requested_bytes, size_t capacity_bytes, ResourceAllocation* allocation) {
  const auto context = CurrentResourceContext();
  if (context.reservation != nullptr) {
    return context.reservation->PrepareAllocation(
      context.resource_class, requested_bytes, capacity_bytes, allocation);
  }
  try {
    ResourceReservation standalone;
    Status status = DefaultResourceBudget().TryReserve(capacity_bytes, &standalone);
    if (!status.ok()) return status;
    return standalone.PrepareAllocation(
      context.resource_class, requested_bytes, capacity_bytes, allocation);
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("Resource accounting metadata allocation failed");
  }
}

/// Exclusive idle-cache acquisition. A failed transfer is a cache miss: the
/// caller must release that backing before attempting a correctly sized fresh
/// allocation. Cross-domain backings are never silently lent to another domain.
[[nodiscard]] inline Status ActivateCachedResource(
  ResourceAllocation& allocation, size_t requested_bytes) {
  if (!allocation.idle() || requested_bytes == 0 ||
      requested_bytes > allocation.capacity_bytes())
    return Status::InvalidArgument("Invalid cached resource activation");
  const auto context = CurrentResourceContext();
  if (context.reservation != nullptr) {
    Status status = allocation.TransferTo(*context.reservation);
    if (!status.ok()) return status;
  } else if (!allocation.SharesDomain(DefaultResourceBudget())) {
    return Status::InvalidArgument("Cached resource belongs to another domain");
  }
  return allocation.MakeLive(requested_bytes);
}

}  // namespace gjxl::resource_budget_internal
