// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "core/managed_allocator.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"
#include <array>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace gjxl::modular_internal::reference_workflow_internal {
// Incremental CPU jobs with caller-owned payloads. Submit publishes the payload;
// Take publishes completion. The caller may prepare/tokenize independent owners
// between these operations. At most kCapacity jobs may be outstanding, including
// completed jobs not yet taken. Stop joins before any payload can be destroyed.
class LocalLearningQueue {
public:
  static constexpr size_t kCapacity = 256;
  explicit LocalLearningQueue(size_t participants) : group_(participants) {}
  ~LocalLearningQueue() { Stop(); }
  LocalLearningQueue(const LocalLearningQueue &) = delete;
  LocalLearningQueue &operator=(const LocalLearningQueue &) = delete;
  size_t participants() const { return group_.participants(); }
  template <class Function> Status Start(const Function &function) {
    if (function_)
      return Status::InvalidArgument("Local learning queue cannot be restarted");
    context_ = &function;
    function_ = [](const void *p, size_t job, size_t worker) noexcept {
      (*static_cast<const Function *>(p))(job, worker);
    };
    const size_t cpus = thread_budget_internal::CpuThreadCount();
    auto *tracker = thread_budget_internal::ParticipantTracker();
    const auto resources = resource_budget_internal::CurrentResourceContext();
    try {
      workers_.reserve(participants() - 1);
      for (size_t i = 1; i < participants(); ++i)
        thread_budget_internal::LaunchWorker(workers_,
            thread_budget_internal::WorkerLaunchSite::kModularTreeLearning, i - 1,
            [this, i, cpus, tracker, resources] {
              thread_budget_internal::ParallelScope scope(cpus, tracker, resources, &group_);
              for (;;) {
                size_t job;
                {
                  thread_budget_internal::CpuSuspension suspension;
                  std::unique_lock lock(mutex_);
                  changed_.wait(lock, [&] { return stopping_ || queued_; });
                  if (stopping_) return;
                  job = Pop();
                }
                Execute(job, i);
              }
            });
    } catch (const resource_budget_internal::ManagedAllocationFailure &e) {
      Stop();
      return e.status();
    } catch (const std::bad_alloc &) {
      Stop();
      return Status::OutOfMemory("Local learning queue allocation failed");
    } catch (const std::system_error &) {
      Stop();
      return Status::Internal("Unable to launch local learning queue workers");
    }
    return Status::Ok();
  }
  Status Submit(size_t job) {
    {
      std::lock_guard lock(mutex_);
      if (stopping_ || !function_ || outstanding_ == kCapacity)
        return Status::Internal("Local learning queue admission overflow");
      ready_[(head_ + queued_) % kCapacity] = job;
      ++queued_;
      ++outstanding_;
    }
    changed_.notify_all();
    return Status::Ok();
  }
  bool Take(size_t *job) {
    std::lock_guard lock(mutex_);
    if (!completed_) return false;
    *job = done_[done_head_];
    done_head_ = (done_head_ + 1) % kCapacity;
    --completed_;
    --outstanding_;
    return true;
  }
  // Make progress without occupying a CPU permit while waiting. The caller
  // contributes one worker when no coordinator operation is ready, including
  // the single-participant case. No thread is launched for the caller.
  void Progress() {
    size_t job;
    {
      thread_budget_internal::CpuSuspension suspension;
      std::unique_lock lock(mutex_);
      if (completed_ || !outstanding_ || stopping_) return;
      if (!queued_) {
        changed_.wait(lock, [&] { return completed_ || stopping_; });
        return;
      }
      job = Pop();
    }
    Execute(job, 0);
  }
  void Stop() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    changed_.notify_all();
    thread_budget_internal::JoinCpuWorkers(workers_);
    workers_.clear();
  }
private:
  size_t Pop() {
    const size_t job = ready_[head_];
    head_ = (head_ + 1) % kCapacity;
    --queued_;
    return job;
  }
  void Execute(size_t job, size_t worker) {
    function_(context_, job, worker);
    {
      std::lock_guard lock(mutex_);
      done_[(done_head_ + completed_) % kCapacity] = job;
      ++completed_;
    }
    changed_.notify_all();
  }
  thread_budget_internal::CpuWorkerGroup group_;
  resource_budget_internal::ManagedVector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::array<size_t, kCapacity> ready_{}, done_{};
  size_t head_ = 0, queued_ = 0, done_head_ = 0, completed_ = 0, outstanding_ = 0;
  bool stopping_ = false;
  const void *context_ = nullptr;
  void (*function_)(const void *, size_t, size_t) noexcept = nullptr;
};
} // namespace gjxl::modular_internal::reference_workflow_internal
