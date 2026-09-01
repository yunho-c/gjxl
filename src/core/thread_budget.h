// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>

namespace gjxl::thread_budget_internal {

struct CpuThreadBudgetState {
  size_t thread_count = 0;
  size_t parallel_depth = 0;
};

inline thread_local CpuThreadBudgetState cpu_thread_budget_state;

/// Installs one encode's CPU thread budget on the calling thread.
class EncodeScope {
public:
  explicit EncodeScope(size_t thread_count)
      : previous_(cpu_thread_budget_state) {
    cpu_thread_budget_state = {
      .thread_count = thread_count,
      .parallel_depth = 0,
    };
  }

  ~EncodeScope() {
    cpu_thread_budget_state = previous_;
  }

  EncodeScope(const EncodeScope&) = delete;
  EncodeScope& operator=(const EncodeScope&) = delete;

private:
  CpuThreadBudgetState previous_;
};

/// Propagates an explicit budget into a participant and marks nested work.
class ParallelScope {
public:
  explicit ParallelScope(size_t thread_count)
      : previous_(cpu_thread_budget_state) {
    cpu_thread_budget_state = {
      .thread_count = thread_count,
      .parallel_depth = previous_.parallel_depth + 1,
    };
  }

  ~ParallelScope() {
    cpu_thread_budget_state = previous_;
  }

  ParallelScope(const ParallelScope&) = delete;
  ParallelScope& operator=(const ParallelScope&) = delete;

private:
  CpuThreadBudgetState previous_;
};

[[nodiscard]] inline size_t CpuThreadCount() noexcept {
  return cpu_thread_budget_state.thread_count;
}

[[nodiscard]] inline bool InExplicitParallelScope() noexcept {
  return cpu_thread_budget_state.thread_count != 0 &&
    cpu_thread_budget_state.parallel_depth != 0;
}

}  // namespace gjxl::thread_budget_internal
