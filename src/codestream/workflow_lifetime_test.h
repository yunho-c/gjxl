// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

namespace gjxl::codestream_internal {

/// Calling-thread-only observation point after successful frontend work and
/// optional last-use release, immediately before CPU serialization. The
/// callback may inspect counters but must not throw or mutate encoder state.
struct WorkflowLifetimeObserverForTest {
  void (*before_serialization)(bool completed_frame, bool may_retry,
                               void *context) noexcept = nullptr;
  void *context = nullptr;
};

/// Returns the previous observer, for scoped/nested restoration. No global
/// synchronization or cross-thread propagation is implied by this test hook.
[[nodiscard]] WorkflowLifetimeObserverForTest
SetWorkflowLifetimeObserverForTest(
    WorkflowLifetimeObserverForTest observer) noexcept;

} // namespace gjxl::codestream_internal
