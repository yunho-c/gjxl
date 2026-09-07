// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stop_token>
#include <utility>

#include "core/status.h"

namespace gjxl::cpu_budget_internal {

struct CpuBudgetSnapshot {
  size_t active_participants = 0;
  size_t reserved_workers = 0;
  size_t suspended_workers = 0;
  size_t waiting_callers = 0;
  size_t peak_active_participants = 0;
  size_t peak_protected_slots = 0;
};

namespace detail {
struct Waiter {
  Waiter* previous = nullptr;
  Waiter* next = nullptr;
};

struct State {
  explicit State(size_t count) : limit(count) {}

  size_t Protected() const noexcept {
    return counters.active_participants + counters.reserved_workers + counters.suspended_workers;
  }
  void UpdatePeaks() noexcept {
    assert(Protected() <= limit);
    counters.peak_active_participants = std::max(counters.peak_active_participants,
      counters.active_participants);
    counters.peak_protected_slots = std::max(counters.peak_protected_slots, Protected());
  }
  void Enqueue(Waiter* waiter) noexcept {
    waiter->previous = last;
    if (last != nullptr) last->next = waiter;
    else first = waiter;
    last = waiter;
    ++counters.waiting_callers;
  }
  void Remove(Waiter* waiter) noexcept {
    if (waiter->previous != nullptr) waiter->previous->next = waiter->next;
    else first = waiter->next;
    if (waiter->next != nullptr) waiter->next->previous = waiter->previous;
    else last = waiter->previous;
    --counters.waiting_callers;
  }
  const size_t limit;
  mutable std::mutex mutex;
  std::condition_variable_any changed;
  CpuBudgetSnapshot counters;
  Waiter* first = nullptr;
  Waiter* last = nullptr;
};
}  // namespace detail

class CpuBudget;
class CpuWorkerReservation;

/// One participation ticket. A suspended worker retains protected capacity but
/// is not active. The ticket keeps the immutable domain alive until release.
class CpuPermit {
public:
  CpuPermit() = default;
  ~CpuPermit() { Reset(); }
  CpuPermit(const CpuPermit&) = delete;
  CpuPermit& operator=(const CpuPermit&) = delete;
  CpuPermit(CpuPermit&& other) noexcept
    : state_(std::move(other.state_)), wait_ns_(std::exchange(other.wait_ns_, 0)),
      worker_(std::exchange(other.worker_, false)), suspended_(std::exchange(other.suspended_, false)) {}
  CpuPermit& operator=(CpuPermit&& other) noexcept {
    if (this != &other) {
      Reset();
      state_ = std::move(other.state_);
      wait_ns_ = std::exchange(other.wait_ns_, 0);
      worker_ = std::exchange(other.worker_, false);
      suspended_ = std::exchange(other.suspended_, false);
    }
    return *this;
  }
  [[nodiscard]] bool has_ticket() const noexcept { return state_ != nullptr; }
  [[nodiscard]] bool active() const noexcept { return has_ticket() && !suspended_; }
  [[nodiscard]] bool is_worker() const noexcept { return worker_; }
  [[nodiscard]] uint64_t wait_nanoseconds() const noexcept { return wait_ns_; }
  /// A created worker retains its protected capacity while blocked. Otherwise
  /// nested joins could turn one domain's limit into many dormant OS threads.
  void SuspendWorker() noexcept {
    assert(active() && worker_);
    std::lock_guard lock(state_->mutex);
    --state_->counters.active_participants;
    ++state_->counters.suspended_workers;
    suspended_ = true;
  }
  void ResumeWorker() noexcept {
    assert(has_ticket() && worker_ && suspended_);
    std::lock_guard lock(state_->mutex);
    --state_->counters.suspended_workers;
    ++state_->counters.active_participants;
    suspended_ = false;
    state_->UpdatePeaks();
  }
  void Reset() noexcept {
    if (!state_) return;
    {
      std::lock_guard lock(state_->mutex);
      auto& count = suspended_ ? state_->counters.suspended_workers : state_->counters.active_participants;
      assert(count != 0);
      --count;
    }
    state_->changed.notify_all();
    state_.reset();
    wait_ns_ = 0;
    worker_ = false;
    suspended_ = false;
  }

private:
  friend class CpuBudget;
  friend class CpuWorkerReservation;
  explicit CpuPermit(std::shared_ptr<detail::State> state, uint64_t wait_ns = 0, bool worker = false)
    : state_(std::move(state)), wait_ns_(wait_ns), worker_(worker) {}
  std::shared_ptr<detail::State> state_;
  uint64_t wait_ns_ = 0;
  bool worker_ = false;
  bool suspended_ = false;
};

/// Pre-created worker slots, not active threads. Call Take() once when each
/// worker starts. No worker waits while its caller holds a slot. Unclaimed
/// slots are released on launch failure; claimed permits outlive this owner.
/// Destruction/move must not race Take(); ordinary use joins launched workers.
class CpuWorkerReservation {
public:
  CpuWorkerReservation() = default;
  ~CpuWorkerReservation() { Reset(); }
  CpuWorkerReservation(const CpuWorkerReservation&) = delete;
  CpuWorkerReservation& operator=(const CpuWorkerReservation&) = delete;
  CpuWorkerReservation(CpuWorkerReservation&& other) noexcept
    : state_(std::move(other.state_)), count_(std::exchange(other.count_, 0)),
      taken_(other.taken_.exchange(0)) {}
  CpuWorkerReservation& operator=(CpuWorkerReservation&& other) noexcept {
    if (this != &other) {
      Reset();
      state_ = std::move(other.state_);
      count_ = std::exchange(other.count_, 0);
      taken_.store(other.taken_.exchange(0));
    }
    return *this;
  }
  [[nodiscard]] size_t count() const noexcept { return count_; }
  /// Return excess pending slots before any worker can call Take().
  void KeepAtMost(size_t maximum) noexcept {
    assert(taken_.load(std::memory_order_relaxed) == 0);
    if (!state_ || maximum >= count_) return;
    {
      std::lock_guard lock(state_->mutex);
      state_->counters.reserved_workers -= count_ - maximum;
    }
    count_ = maximum;
    state_->changed.notify_all();
    if (count_ == 0) state_.reset();
  }
  [[nodiscard]] CpuPermit Take() const noexcept {
    size_t taken = taken_.load(std::memory_order_relaxed);
    do {
      if (taken == count_) return {};
    } while (!taken_.compare_exchange_weak(taken, taken + 1, std::memory_order_relaxed));
    {
      std::lock_guard lock(state_->mutex);
      assert(state_->counters.reserved_workers != 0);
      --state_->counters.reserved_workers;
      ++state_->counters.active_participants;
      state_->UpdatePeaks();
    }
    return CpuPermit(state_, 0, true);
  }
  void Reset() noexcept {
    if (!state_) return;
    const size_t unused = count_ - taken_.load(std::memory_order_relaxed);
    {
      std::lock_guard lock(state_->mutex);
      assert(state_->counters.reserved_workers >= unused);
      state_->counters.reserved_workers -= unused;
    }
    state_->changed.notify_all();
    state_.reset();
    count_ = 0;
    taken_ = 0;
  }

private:
  friend class CpuBudget;
  CpuWorkerReservation(std::shared_ptr<detail::State> state, size_t count)
    : state_(std::move(state)), count_(count) {}
  std::shared_ptr<detail::State> state_;
  size_t count_ = 0;
  mutable std::atomic<size_t> taken_{0};
};

/// CPU execution capacity, separate from managed-memory admission. Copies
/// share one allowance. Callers and resumed callers queue FIFO for one slot;
/// additional workers reserve only currently free slots, never bypass waiters.
class CpuBudget {
public:
  CpuBudget() = default;
  explicit CpuBudget(size_t participants)
    : state_(participants == 0 ? nullptr : std::make_shared<detail::State>(participants)) {}
  [[nodiscard]] size_t limit() const noexcept { return state_ ? state_->limit : 0; }
  [[nodiscard]] bool Shares(const CpuPermit& permit) const noexcept {
    return state_ && state_ == permit.state_;
  }
  [[nodiscard]] CpuBudgetSnapshot snapshot() const {
    if (!state_) return {};
    std::lock_guard lock(state_->mutex);
    return state_->counters;
  }
  [[nodiscard]] Status Acquire(CpuPermit* output, std::stop_token stop = {}) const {
    if (!state_ || output == nullptr || output->has_ticket())
      return Status::InvalidArgument("CPU admission requires a domain and empty permit output");
    if (stop.stop_requested()) return Status::Unavailable("CPU admission was cancelled");
    std::unique_lock lock(state_->mutex);
    detail::Waiter waiter;
    const bool waits = state_->first != nullptr || state_->Protected() == state_->limit;
    const auto begin = waits ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    state_->Enqueue(&waiter);
    bool ready = false;
    try {
      ready = state_->changed.wait(lock, stop, [&] {
        return state_->first == &waiter && state_->Protected() < state_->limit;
      });
    } catch (...) {
      state_->Remove(&waiter);
      lock.unlock();
      state_->changed.notify_all();
      throw;
    }
    state_->Remove(&waiter);
    if (!ready || stop.stop_requested()) {
      lock.unlock();
      state_->changed.notify_all();
      return Status::Unavailable("CPU admission was cancelled");
    }
    ++state_->counters.active_participants;
    state_->UpdatePeaks();
    const uint64_t wait_ns = waits ? static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count()) : 0;
    *output = CpuPermit(state_, wait_ns);
    lock.unlock();
    state_->changed.notify_all();
    return Status::Ok();
  }
  [[nodiscard]] CpuWorkerReservation TryReserveWorkers(size_t maximum) const noexcept {
    if (!state_ || maximum == 0) return {};
    std::lock_guard lock(state_->mutex);
    if (state_->first != nullptr) return {};
    const size_t count = std::min(maximum, state_->limit - state_->Protected());
    if (count == 0) return {};
    state_->counters.reserved_workers += count;
    state_->UpdatePeaks();
    return CpuWorkerReservation(state_, count);
  }

private:
  std::shared_ptr<detail::State> state_;
};
}  // namespace gjxl::cpu_budget_internal
