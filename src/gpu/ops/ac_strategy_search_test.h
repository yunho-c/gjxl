// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

namespace gjxl::ac_strategy_search_internal {

// Keep GPU candidate scoring but select on the CPU in this thread. This gives
// tests an independent selector oracle on devices without timestamp counters.
// Nested scopes restore the preceding policy; production defaults are unchanged.
class ScopedCpuSelectionForTesting {
public:
  ScopedCpuSelectionForTesting() noexcept;
  ~ScopedCpuSelectionForTesting();
  ScopedCpuSelectionForTesting(const ScopedCpuSelectionForTesting&) = delete;
  ScopedCpuSelectionForTesting& operator=(const ScopedCpuSelectionForTesting&) = delete;

private:
  bool previous_;
};

}  // namespace gjxl::ac_strategy_search_internal
