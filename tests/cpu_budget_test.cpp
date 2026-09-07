// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "core/cpu_budget.h"

namespace {
using namespace gjxl;
using namespace gjxl::cpu_budget_internal;
bool Check(bool good, const char* text) {
  if (!good) std::cerr << text << '\n';
  return good;
}
bool Empty(const CpuBudget& budget) {
  const auto s = budget.snapshot();
  return Check(s.active_participants == 0 && s.reserved_workers == 0 && s.suspended_workers == 0 &&
    s.waiting_callers == 0, "CPU budget leaked a participant, reservation or waiter");
}
template<class Predicate> bool Until(Predicate predicate) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= end) return false;
    std::this_thread::yield();
  }
  return true;
}
struct Gate {
  std::mutex mutex;
  std::condition_variable_any changed;
  bool open = false;
  void Wait(std::stop_token stop) {
    std::unique_lock lock(mutex);
    changed.wait(lock, stop, [&] { return open; });
  }
  void Release() {
    { std::lock_guard lock(mutex); open = true; }
    changed.notify_all();
  }
};

bool Lifetime() {
  CpuBudget budget(3), same = budget, other(3);
  CpuPermit caller;
  if (!Check(budget.Acquire(&caller).ok() && caller.active() && caller.wait_nanoseconds() == 0 &&
      same.Shares(caller) && !other.Shares(caller), "CPU caller admission/domain identity failed")) return false;
  if (!Check(budget.Acquire(&caller).code() == StatusCode::kInvalidArgument && caller.active(),
      "Invalid CPU output lost its existing participant")) return false;
  auto workers = budget.TryReserveWorkers(9);
  auto s = budget.snapshot();
  if (!Check(workers.count() == 2 && s.active_participants == 1 && s.reserved_workers == 2 &&
      s.peak_protected_slots == 3, "Worker slots were not reserved before launch")) return false;
  auto worker = workers.Take();
  s = budget.snapshot();
  if (!Check(worker.active() && s.active_participants == 2 && s.reserved_workers == 1,
      "Worker activation did not transfer its pending slot")) return false;
  auto moved = std::move(workers);
  budget = {};
  moved.Reset();  // Models failure before the other worker was launched.
  if (!Check(same.snapshot().reserved_workers == 0 && same.snapshot().active_participants == 2,
      "Reservation teardown reclaimed a live worker or leaked an unused slot")) return false;
  CpuPermit retained = std::move(caller);
  worker.Reset();
  retained.Reset();
  return Empty(same) && Empty(other) && Check(!caller.active(), "Moved CPU permit remained active");
}

bool SuspendedWorkerLifetime() {
  CpuBudget budget(2);
  CpuPermit caller;
  if (!budget.Acquire(&caller).ok()) return false;
  auto reservation = budget.TryReserveWorkers(10);
  auto worker = reservation.Take();
  worker.SuspendWorker();
  const auto s = budget.snapshot();
  if (!Check(worker.has_ticket() && !worker.active() && worker.is_worker() &&
      s.active_participants == 1 && s.suspended_workers == 1 &&
      budget.TryReserveWorkers(1).count() == 0 &&
      budget.Acquire(&worker).code() == StatusCode::kInvalidArgument,
      "Suspended worker lost protected capacity or accepted a second ticket")) return false;
  CpuPermit moved = std::move(worker);
  moved.ResumeWorker();
  if (!Check(moved.active() && budget.snapshot().active_participants == 2 &&
      budget.snapshot().suspended_workers == 0, "Moved suspended worker failed to resume")) return false;
  moved.SuspendWorker();
  moved.Reset(); // Destruction may release a suspended ticket directly.
  reservation.Reset(); caller.Reset();
  auto pending = budget.TryReserveWorkers(2);
  pending.KeepAtMost(1);
  if (!Check(pending.count() == 1 && budget.snapshot().reserved_workers == 1,
      "Shrinking a pre-launch reservation leaked capacity")) return false;
  pending.KeepAtMost(0);
  return Check(!pending.Take().active(), "Zero reservation activated a worker") && Empty(budget);
}

bool WorkerActivation() {
  CpuBudget budget(4);
  CpuPermit caller;
  if (!budget.Acquire(&caller).ok()) return false;
  auto reservation = budget.TryReserveWorkers(3);
  Gate gate;
  std::atomic<size_t> entered{0};
  std::atomic<bool> good{true};
  std::vector<std::jthread> workers;
  for (size_t i = 0; i < 3; ++i) workers.emplace_back([&](std::stop_token stop) {
    auto permit = reservation.Take();
    if (!permit.active()) { good = false; return; }
    ++entered;
    gate.Wait(stop);
  });
  if (!Check(Until([&] { return entered == 3; }), "Reserved workers failed to activate")) return false;
  const auto s = budget.snapshot();
  if (!Check(s.active_participants == 4 && s.reserved_workers == 0 &&
      !reservation.Take().active(), "CPU workers escaped their reservation")) return false;
  gate.Release();
  workers.clear();
  caller.Reset();
  reservation.Reset();
  return Check(good && budget.snapshot().peak_active_participants == 4,
    "Actual CPU participation was not observed") && Empty(budget);
}

bool FifoAndCancellation() {
  CpuBudget budget(1);
  CpuPermit blocker;
  if (!budget.Acquire(&blocker).ok()) return false;
  Gate head_gate;
  std::atomic<size_t> order{0};
  std::atomic<bool> first_entered{false}, second_entered{false}, good{true};
  std::jthread first([&](std::stop_token stop) {
    CpuPermit permit;
    if (!budget.Acquire(&permit, stop).ok()) return;
    if (++order != 1 || permit.wait_nanoseconds() == 0) good = false;
    first_entered = true;
    head_gate.Wait(stop);
  });
  if (!Check(Until([&] { return budget.snapshot().waiting_callers == 1; }), "First CPU caller did not queue")) return false;
  std::jthread second([&](std::stop_token stop) {
    CpuPermit permit;
    if (!budget.Acquire(&permit, stop).ok()) return;
    if (++order != 2) good = false;
    second_entered = true;
  });
  if (!Check(Until([&] { return budget.snapshot().waiting_callers == 2; }) &&
      budget.TryReserveWorkers(1).count() == 0, "Additional workers bypassed CPU admission")) return false;
  blocker.Reset();
  if (!Check(Until([&] { return first_entered.load(); }) && !second_entered,
      "CPU caller FIFO order changed")) return false;
  head_gate.Release();
  first.join(); second.join();
  if (!Check(good && second_entered, "CPU FIFO failed to drain") || !Empty(budget)) return false;

  if (!budget.Acquire(&blocker).ok()) return false;
  std::atomic<bool> cancelled{false};
  std::jthread waiter([&](std::stop_token stop) {
    CpuPermit permit;
    const auto status = budget.Acquire(&permit, stop);
    cancelled = status.code() == StatusCode::kUnavailable && !permit.active();
  });
  if (!Check(Until([&] { return budget.snapshot().waiting_callers == 1; }), "Cancellation target did not queue")) return false;
  waiter.request_stop(); waiter.join();
  blocker.Reset();
  if (!Check(cancelled, "CPU cancellation did not preserve the empty output") || !Empty(budget)) return false;
  std::stop_source stopped;
  stopped.request_stop();
  return Check(budget.Acquire(&blocker, stopped.get_token()).code() == StatusCode::kUnavailable &&
    CpuBudget().Acquire(&blocker).code() == StatusCode::kInvalidArgument,
    "Cancelled/invalid CPU admission was accepted") && Empty(budget);
}

bool Contention() {
  CpuBudget budget(3);
  std::atomic<size_t> active{0}, peak{0}, completed{0};
  std::atomic<bool> good{true};
  std::barrier start(13);
  std::vector<std::jthread> callers;
  for (size_t i = 0; i < 12; ++i) callers.emplace_back([&, shared = budget] {
    start.arrive_and_wait();
    for (size_t attempt = 0; attempt < 100; ++attempt) {
      CpuPermit permit;
      if (!shared.Acquire(&permit).ok()) { good = false; return; }
      const size_t now = ++active;
      size_t previous = peak.load();
      while (previous < now && !peak.compare_exchange_weak(previous, now)) {}
      if (now > 3) good = false;
      for (size_t work = 0; work < 20; ++work) std::this_thread::yield();
      --active;
      ++completed;
    }
  });
  start.arrive_and_wait();
  callers.clear();
  return Check(good && completed == 1200 && active == 0 && peak <= 3 && peak > 1 &&
    budget.snapshot().peak_protected_slots <= 3,
    "Concurrent CPU callers overspent capacity or failed to make progress") && Empty(budget);
}
}  // namespace

int main() {
  return Lifetime() && SuspendedWorkerLifetime() && WorkerActivation() && FifoAndCancellation() && Contention()
    ? EXIT_SUCCESS : EXIT_FAILURE;
}
