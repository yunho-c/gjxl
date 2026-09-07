// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <optional>
#include <utility>

#include "codestream/workflow_admission_test.h"
#include "codestream/workflow_storage_plan.h"
#include "core/execution_domain.h"
#include "gpu/backend.h"
#include "gpu/metal/metal_backend.h"

namespace gjxl::codestream_internal {

/// Normal option/geometry validation followed by existing backend qualification
/// and complete planning. No managed backing is allocated. Automatic exact
/// searches may initialize Metal earlier, but never change candidate selection.
[[nodiscard]] Status PlanWorkflowAdmission(Extent2D source, const WorkflowStorageOptions &options,
                                           GpuBackend *supplied_backend,
                                           bool supplied_backend_is_qualified,
                                           bool resolve_production_backend,
                                           WorkflowStoragePlan *out);

class WorkflowAdmission {
public:
  [[nodiscard]] Status Start(size_t bytes, std::shared_ptr<const ExecutionDomain> domain = {}) {
    const auto current = resource_budget_internal::CurrentResourceContext();
    if (current.reservation != nullptr) {
      // Internal completed-output and C/batch adapters share the outer plan.
      // Explicit public domains must never bypass their own hard limit through
      // an unrelated injected/outer reservation.
      if (domain && !domain->budget_.SharesDomain(*current.reservation))
        return Status::InvalidArgument("Nested workflow uses a different execution domain");
      return Status::Ok();
    }
    try {
      domain_ = domain ? std::move(domain) : ExecutionDomain::Default();
      if (bytes == 0)
        return Status::Ok(); // Empty batch has no backing.
      if (next_admission_capacity_for_testing)
        bytes = *std::exchange(next_admission_capacity_for_testing, {});
      const Status status = domain_->budget_.Reserve(
          bytes, &reservation_, {},
          +[](void *opaque) {
            return metal_internal::TrimMetalPreparationCachesForDomain(
                static_cast<WorkflowAdmission *>(opaque)->domain_->budget_);
          },
          this);
      if (!status.ok())
        return status;
      auto context = current;
      context.reservation = &reservation_;
      context.domain = domain_.get();
      context_.emplace(context);
      return Status::Ok();
    } catch (const std::bad_alloc &) {
      return Status::OutOfMemory("Unable to admit workflow");
    }
  }

  [[nodiscard]] static Status TrimIdle(const ExecutionDomain &domain) {
    return metal_internal::TrimMetalPreparationCachesForDomain(domain.budget_);
  }

private:
  std::shared_ptr<const ExecutionDomain> domain_;
  resource_budget_internal::ResourceReservation reservation_;
  std::optional<resource_budget_internal::ResourceContextScope> context_;
};
} // namespace gjxl::codestream_internal
