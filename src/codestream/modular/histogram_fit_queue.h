// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "core/managed_allocator.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <span>
#include <thread>

namespace gjxl::modular_internal::reference_workflow_internal {
// One owner of a fit at a time: a CPU worker, the GPU coordinator, or a queue.
// GPU-ready membership is stable while independent CPU jobs keep progressing.
// Fixed inline queues are bounded by the operation's 256-fit cohort contract.
class HistogramFitQueue {
public:
  enum class Result { kGpu, kDone, kFailed };
  explicit HistogramFitQueue(size_t participants) : group_(participants) {}
  ~HistogramFitQueue() { Stop(); }
  HistogramFitQueue(const HistogramFitQueue &) = delete;
  HistogramFitQueue &operator=(const HistogramFitQueue &) = delete;
  size_t participants() const { return group_.participants(); }
  template <class Function>
  Status Start(size_t count, const Function &function) {
    if (count > states_.size() || function_)
      return Status::InvalidArgument(
          "Invalid histogram fit queue size or reuse");
    count_ = count;
    context_ = &function;
    function_ = [](const void *p, size_t i) noexcept {
      return (*static_cast<const Function *>(p))(i);
    };
    for (size_t i = 0; i < count; ++i)
      Push(i);
    workers_.reserve(participants() - 1);
    const size_t cpus = thread_budget_internal::CpuThreadCount();
    auto *tracker = thread_budget_internal::ParticipantTracker();
    const auto resources = resource_budget_internal::CurrentResourceContext();
    try {
      for (size_t i = 0; i + 1 < participants(); ++i)
        thread_budget_internal::LaunchWorker(
            workers_,
            thread_budget_internal::WorkerLaunchSite::kModularTreeLearning, i,
            [this, cpus, tracker, resources] {
              thread_budget_internal::ParallelScope scope(cpus, tracker,
                                                          resources, &group_);
              for (;;) {
                size_t job;
                {
                  thread_budget_internal::CpuSuspension suspension;
                  std::unique_lock lock(mutex_);
                  changed_.wait(
                      lock, [&] { return stopping_ || failed_ || queued_; });
                  if (stopping_ || failed_)
                    return;
                  job = Pop();
                }
                const auto result = function_(context_, job);
                {
                  std::lock_guard lock(mutex_);
                  Complete(job, result);
                }
                changed_.notify_all();
              }
            });
    } catch (const resource_budget_internal::ManagedAllocationFailure &e) {
      Stop();
      return e.status();
    } catch (const std::bad_alloc &) {
      Stop();
      return Status::OutOfMemory(
          "Histogram fit queue worker allocation failed");
    } catch (const std::system_error &) {
      Stop();
      return Status::Internal("Unable to launch histogram fit queue workers");
    }
    return Status::Ok();
  }
  // False means a CPU task failed; callers join before inspecting job errors.
  // A true result with count=0 means all fits completed. Otherwise this caller
  // exclusively owns the returned jobs until Resume publishes completed data.
  bool Next(std::span<size_t> jobs, size_t *count) {
    for (;;) {
      size_t cpu_job = SIZE_MAX;
      {
        thread_budget_internal::CpuSuspension suspension;
        std::unique_lock lock(mutex_);
        if (failed_)
          return false;
        if (completed_ == count_) {
          *count = 0;
          return true;
        }
        const auto deadline = first_ready_ + std::chrono::microseconds(500);
        if (ready_ && (ready_ >= std::min(size_t{32}, count_ - completed_) ||
                       (!queued_ && !running_) || Clock::now() >= deadline)) {
          *count = 0;
          for (size_t i = 0; i < count_; ++i)
            if (states_[i] == State::kGpuReady) {
              jobs[(*count)++] = i;
              states_[i] = State::kGpuRunning;
            }
          ready_ = 0;
          return true;
        }
        if (participants() == 1 && queued_) {
          cpu_job = Pop();
        } else if (ready_) {
          changed_.wait_until(lock, deadline);
        } else {
          changed_.wait(lock);
        }
      }
      if (cpu_job != SIZE_MAX) {
        const auto result = function_(context_, cpu_job);
        std::lock_guard lock(mutex_);
        Complete(cpu_job, result);
      }
    }
  }
  void Resume(std::span<const size_t> jobs) {
    {
      std::lock_guard lock(mutex_);
      for (size_t i : jobs)
        Push(i);
    }
    changed_.notify_all();
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
  enum class State { kCpuReady, kCpuRunning, kGpuReady, kGpuRunning, kDone };
  using Clock = std::chrono::steady_clock;
  void Push(size_t i) {
    states_[i] = State::kCpuReady;
    queue_[(head_ + queued_) % queue_.size()] = i;
    ++queued_;
  }
  size_t Pop() {
    const size_t i = queue_[head_];
    head_ = (head_ + 1) % queue_.size();
    --queued_;
    ++running_;
    states_[i] = State::kCpuRunning;
    return i;
  }
  void Complete(size_t i, Result result) {
    --running_;
    if (result == Result::kGpu) {
      if (!ready_)
        first_ready_ = Clock::now();
      ++ready_;
      states_[i] = State::kGpuReady;
    } else {
      states_[i] = State::kDone;
      ++completed_;
      failed_ |= result == Result::kFailed;
    }
  }
  thread_budget_internal::CpuWorkerGroup group_;
  resource_budget_internal::ManagedVector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::array<State, 256> states_{};
  std::array<size_t, 256> queue_{};
  size_t count_ = 0, head_ = 0, queued_ = 0, running_ = 0, ready_ = 0,
         completed_ = 0;
  bool stopping_ = false, failed_ = false;
  Clock::time_point first_ready_{};
  const void *context_ = nullptr;
  Result (*function_)(const void *, size_t) noexcept = nullptr;
};
} // namespace gjxl::modular_internal::reference_workflow_internal
