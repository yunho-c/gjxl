// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "core/managed_allocator.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"
#include <condition_variable>
#include <mutex>
#include <thread>

namespace gjxl::modular_internal::histogram_parallel_internal {

// One team per fit. Property tasks allocate nothing and only read sample rows;
// the caller waits for every task before selecting a split or moving any row.
// Sleeping workers retain protected CPU capacity, but are not active CPU users.
class PropertyTeam {
public:
  explicit PropertyTeam(size_t participants) : group_(participants) {}
  ~PropertyTeam() { Stop(); }
  PropertyTeam(const PropertyTeam &) = delete;
  PropertyTeam &operator=(const PropertyTeam &) = delete;
  size_t participants() const { return group_.participants(); }

  Status Start() {
    if (participants() == 1)
      return Status::Ok();
    // Reserve before launch; a partial launch is always stopped and joined.
    workers_.reserve(participants() - 1);
    const size_t cpu_count = thread_budget_internal::CpuThreadCount();
    auto *tracker = thread_budget_internal::ParticipantTracker();
    const auto resources = resource_budget_internal::CurrentResourceContext();
    try {
      for (size_t i = 0; i + 1 < participants(); ++i)
        thread_budget_internal::LaunchWorker(
            workers_,
            thread_budget_internal::WorkerLaunchSite::
                kModularHistogramProperties,
            i, [this, i, cpu_count, tracker, resources] {
              thread_budget_internal::ParallelScope scope(cpu_count, tracker,
                                                          resources, &group_);
              size_t seen = 0;
              for (;;) {
                {
                  // Destroy the lock before resuming CPU participation.
                  thread_budget_internal::CpuSuspension suspension;
                  std::unique_lock lock(mutex_);
                  ready_.wait(lock,
                              [&] { return stopping_ || generation_ != seen; });
                  if (stopping_)
                    return;
                  seen = generation_;
                }
                Work(i);
                std::lock_guard lock(mutex_);
                if (--remaining_ == 0)
                  done_.notify_one();
              }
            });
    } catch (const std::bad_alloc &) {
      Stop();
      return Status::OutOfMemory(
          "Histogram property worker launch allocation failed");
    } catch (const std::system_error &) {
      Stop();
      return Status::Internal("Unable to start histogram property workers");
    }
    return Status::Ok();
  }

  template <class Function> void Run(size_t count, const Function &function) {
    {
      std::lock_guard lock(mutex_);
      context_ = &function;
      function_ = [](const void *context, size_t index,
                     size_t worker) noexcept {
        (*static_cast<const Function *>(context))(index, worker);
      };
      count_ = count;
      next_.store(0, std::memory_order_relaxed);
      remaining_ = workers_.size();
      ++generation_;
    }
    ready_.notify_all();
    Work(workers_.size()); // The admitted caller is also a participant.
    {
      thread_budget_internal::CpuSuspension suspension;
      std::unique_lock lock(mutex_);
      done_.wait(lock, [&] { return remaining_ == 0; });
    }
  }

private:
  void Work(size_t worker) noexcept {
    for (;;) {
      const size_t index = next_.fetch_add(1, std::memory_order_relaxed);
      if (index >= count_)
        return;
      function_(context_, index, worker);
    }
  }
  void Stop() {
    if (workers_.empty())
      return;
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    thread_budget_internal::JoinCpuWorkers(workers_);
    workers_.clear();
  }
  thread_budget_internal::CpuWorkerGroup group_;
  resource_budget_internal::ManagedVector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable ready_, done_;
  std::atomic<size_t> next_{0};
  size_t generation_ = 0, remaining_ = 0, count_ = 0;
  bool stopping_ = false;
  const void *context_ = nullptr;
  void (*function_)(const void *, size_t, size_t) noexcept = nullptr;
};
} // namespace gjxl::modular_internal::histogram_parallel_internal
