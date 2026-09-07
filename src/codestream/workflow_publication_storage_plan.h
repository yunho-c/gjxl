// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include "codestream/rate_control_internal.h"

namespace gjxl::codestream_internal {

struct WorkflowPublicationStoragePlan {
  resource_budget_internal::HostStorageBound scores;
  resource_budget_internal::HostStorageBound timing;
  resource_budget_internal::HostStorageBound search_control;
  resource_budget_internal::HostStorageBound retained_best;
  resource_budget_internal::HostStorageBound output;
  bool operator==(const WorkflowPublicationStoragePlan&) const = default;
};

// Common owners, NOT a complete working envelope. The route decides where
// these owners coexist with evaluator/serializer storage; both may already
// include some current outputs. GPU-profile output is a separate route owner.
// Search retains at most one best result, regardless of the attempt cap.
// All inputs are shape bounds; no allocation or search occurs on success.
[[nodiscard]] inline Status ComputeWorkflowPublicationStoragePlan(
    resource_budget_internal::HostStorageBound codestream, size_t score_count,
    size_t maximum_attempts, bool search, bool collect_timing,
    WorkflowPublicationStoragePlan* out,
    const char* overflow_message = "Workflow publication storage bound overflows") {
  if (out == nullptr || maximum_attempts == 0 || (!search && maximum_attempts != 1))
    return Status::InvalidArgument("Workflow publication storage shape is invalid");
  using enum resource_budget_internal::VectorCapacityPolicy;
  WorkflowPublicationStoragePlan p;
  if (!p.scores.AddVector<double>(score_count, kFreshExact) ||
      (collect_timing && !p.timing.AddVector<VarDctEncodingAttemptTiming>(
                            maximum_attempts, kFreshExact)))
    return Status::OutOfMemory(overflow_message);
  if (search) {
    const Status status = ComputeTargetSizeControlStorageBound(maximum_attempts, &p.search_control);
    if (!status.ok()) return status;
    if (maximum_attempts > 1 &&
        (!p.retained_best.Add(codestream) || !p.retained_best.Add(p.scores)))
      return Status::OutOfMemory(overflow_message);
  }
  p.output = codestream;
  if (!p.output.Add(p.scores) || !p.output.Add(p.timing))
    return Status::OutOfMemory(overflow_message);
  *out = p;
  return Status::Ok();
}

} // namespace gjxl::codestream_internal
