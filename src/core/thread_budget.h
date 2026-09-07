// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <atomic>
#include <cstddef>

#include "core/resource_context.h"
#include "core/cpu_execution.h"

namespace gjxl::thread_budget_internal {

class CpuParticipantTracker {
public:
  void Enter() noexcept {
    const size_t active =
      active_.fetch_add(1, std::memory_order_relaxed) + 1;
    size_t peak = peak_.load(std::memory_order_relaxed);
    while (peak < active &&
           !peak_.compare_exchange_weak(
             peak, active, std::memory_order_relaxed,
             std::memory_order_relaxed)) {}
  }

  void Leave() noexcept {
    active_.fetch_sub(1, std::memory_order_relaxed);
  }

  [[nodiscard]] size_t peak() const noexcept {
    return peak_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] size_t active() const noexcept {
    return active_.load(std::memory_order_relaxed);
  }

private:
  std::atomic<size_t> active_{0};
  std::atomic<size_t> peak_{0};
};

struct CpuThreadBudgetState {
  size_t thread_count = 0;
  size_t parallel_depth = 0;
  CpuParticipantTracker* tracker = nullptr;
};

inline thread_local CpuThreadBudgetState cpu_thread_budget_state;

/// Installs one encode's CPU thread budget on the calling thread.
class EncodeScope {
public:
  explicit EncodeScope(
    size_t thread_count, CpuParticipantTracker* tracker = nullptr)
      : previous_(cpu_thread_budget_state) {
    cpu_thread_budget_state = {
      .thread_count = thread_count,
      .parallel_depth = 0,
      .tracker = tracker,
    };
    if (tracker != nullptr && HasCpuParticipation() &&
        current_cpu_execution.observer.context != tracker) {
      previous_observer_ = current_cpu_execution.observer;
      if (previous_observer_.leave) previous_observer_.leave(previous_observer_.context);
      current_cpu_execution.observer = {
        tracker,
        [](void* p) noexcept { static_cast<CpuParticipantTracker*>(p)->Enter(); },
        [](void* p) noexcept { static_cast<CpuParticipantTracker*>(p)->Leave(); },
      };
      tracker->Enter();
      owns_observer_ = true;
    }
  }

  ~EncodeScope() {
    if (owns_observer_) {
      current_cpu_execution.observer.leave(current_cpu_execution.observer.context);
      current_cpu_execution.observer = previous_observer_;
      if (previous_observer_.enter) previous_observer_.enter(previous_observer_.context);
    }
    cpu_thread_budget_state = previous_;
  }

  EncodeScope(const EncodeScope&) = delete;
  EncodeScope& operator=(const EncodeScope&) = delete;

private:
  CpuThreadBudgetState previous_;
  CpuParticipationObserver previous_observer_;
  bool owns_observer_ = false;
};

/// Propagates an explicit budget into a participant and marks nested work.
class ParallelScope {
public:
  explicit ParallelScope(
    size_t thread_count, CpuParticipantTracker* tracker,
    resource_budget_internal::ResourceContext resources,
    const CpuWorkerGroup* group = nullptr)
      : resources_(resources), worker_(group), previous_(cpu_thread_budget_state),
        tracker_(HasCpuParticipation() ? nullptr : tracker) {
    cpu_thread_budget_state = {
      .thread_count = thread_count,
      .parallel_depth = previous_.parallel_depth + 1,
      .tracker = tracker,
    };
    if (tracker_ != nullptr) tracker_->Enter();
  }

  ~ParallelScope() {
    if (tracker_ != nullptr) tracker_->Leave();
    cpu_thread_budget_state = previous_;
  }

  ParallelScope(const ParallelScope&) = delete;
  ParallelScope& operator=(const ParallelScope&) = delete;

private:
  resource_budget_internal::ResourceContextScope resources_;
  CpuWorkerScope worker_;
  CpuThreadBudgetState previous_;
  CpuParticipantTracker* tracker_ = nullptr;
};

[[nodiscard]] inline size_t CpuThreadCount() noexcept {
  return cpu_thread_budget_state.thread_count;
}

[[nodiscard]] inline CpuParticipantTracker* ParticipantTracker() noexcept {
  return cpu_thread_budget_state.tracker;
}

[[nodiscard]] inline bool InExplicitParallelScope() noexcept {
  return cpu_thread_budget_state.thread_count != 0 &&
    cpu_thread_budget_state.parallel_depth != 0;
}

}  // namespace gjxl::thread_budget_internal
