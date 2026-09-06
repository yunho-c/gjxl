// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "core/resource_budget.h"

namespace {
using namespace gjxl::resource_budget_internal;
using gjxl::Status;
using gjxl::StatusCode;

bool Check(bool condition, const char* message) {
  if (!condition) std::cerr << message << '\n';
  return condition;
}

bool Ok(const Status& status) {
  if (!status.ok()) std::cerr << status.message() << '\n';
  return status.ok();
}

bool Code(const Status& status, StatusCode expected) {
  return Check(status.code() == expected, "Unexpected resource status");
}

bool Consistent(const ResourceBudgetSnapshot& snapshot, size_t limit) {
  ResourceUsage sum;
  for (const auto& usage : snapshot.classes) {
    sum.live_requested_bytes += usage.live_requested_bytes;
    sum.live_capacity_bytes += usage.live_capacity_bytes;
    sum.idle_capacity_bytes += usage.idle_capacity_bytes;
    sum.pending_capacity_bytes += usage.pending_capacity_bytes;
    sum.backing_count += usage.backing_count;
    sum.pending_count += usage.pending_count;
    if (usage.live_requested_bytes > usage.live_capacity_bytes) return false;
  }
  const auto& total = snapshot.total;
  return sum.live_requested_bytes == total.live_requested_bytes &&
    sum.live_capacity_bytes == total.live_capacity_bytes &&
    sum.idle_capacity_bytes == total.idle_capacity_bytes &&
    sum.pending_capacity_bytes == total.pending_capacity_bytes &&
    sum.backing_count == total.backing_count &&
    sum.pending_count == total.pending_count &&
    total.pending_capacity_bytes <= snapshot.reserved_unbacked_bytes &&
    snapshot.committed_bytes() <= limit &&
    snapshot.peak_committed_bytes <= limit;
}

bool Empty(const ResourceBudget& budget) {
  const auto s = budget.snapshot();
  return Check(Consistent(s, std::numeric_limits<size_t>::max()) &&
    s.committed_bytes() == 0 && s.total.backing_count == 0 &&
    s.total.pending_count == 0 && s.open_reservations == 0 &&
    s.waiting_requests == 0, "Leaked resource charge or waiter");
}

template <typename Predicate>
bool Until(Predicate&& predicate) {
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::seconds(5);
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::yield();
  }
  return true;
}

bool CheckLifecycle() {
  ResourceBudget budget(100);
  ResourceReservation job;
  if (!Ok(budget.Reserve(80, &job))) return false;
  auto s = budget.snapshot();
  if (!Check(s.reserved_unbacked_bytes == 80 && s.open_reservations == 1 &&
      s.total.backing_count == 0, "Reservation pretends to own backing"))
    return false;

  ResourceAllocation allocation;
  if (!Ok(job.PrepareAllocation(ResourceClass::kAqScratch, 48, 64, &allocation)))
    return false;
  s = budget.snapshot();
  if (!Check(s.reserved_unbacked_bytes == 80 && s.total.pending_count == 1 &&
      s.total.pending_capacity_bytes == 64 && s.total.backing_count == 0,
      "Pending backing charged twice")) return false;
  if (!Ok(allocation.Commit())) return false;
  s = budget.snapshot();
  if (!Check(s.total.live_requested_bytes == 48 &&
      s.total.live_capacity_bytes == 64 && s.reserved_unbacked_bytes == 16 &&
      s.total.backing_count == 1 && s.total.pending_count == 0 &&
      s.peak_backing_bytes == 64 && s.peak_committed_bytes == 80,
      "Backing capacity accounting differs")) return false;
  if (!Code(job.ReduceCapacity(81), StatusCode::kInvalidArgument) ||
      !Code(job.ReduceCapacity(63), StatusCode::kFailedPrecondition) ||
      !Ok(job.ReduceCapacity(64)) ||
      !Check(budget.snapshot().reserved_unbacked_bytes == 0,
             "Stage boundary retained unused reservation capacity")) return false;

  // Many borrowed views would all refer to this one ticket, without charging it.
  const auto* alias1 = &allocation;
  const auto* alias2 = alias1;
  if (!Check(alias1 == alias2 && budget.snapshot().total.backing_count == 1,
      "Borrowing changed allocation ownership")) return false;
  if (!Ok(allocation.MakeIdle())) return false;
  job.Reset();
  s = budget.snapshot();
  if (!Check(s.committed_bytes() == 64 && s.total.idle_capacity_bytes == 64 &&
      s.total.live_requested_bytes == 0 && s.reserved_unbacked_bytes == 0,
      "Idle backing lost its producer-independent charge")) return false;
  if (!Code(budget.TryReserve(37, &job), StatusCode::kUnavailable) ||
      !Ok(budget.TryReserve(36, &job)) ||
      !Code(allocation.TransferTo(job), StatusCode::kOutOfMemory)) return false;
  allocation.Reset();
  job.Reset();
  return Empty(budget);
}

bool CheckTransfersAndMoves() {
  ResourceBudget budget(256);
  ResourceBudget same_domain = budget;
  ResourceBudget independent(256);
  ResourceReservation producer, consumer, other_domain;
  ResourceAllocation cached;
  if (!Ok(budget.Reserve(128, &producer)) ||
      !Ok(producer.PrepareAllocation(ResourceClass::kButteraugli, 64, 80, &cached)) ||
      !Ok(cached.Commit()) || !Ok(cached.MakeIdle())) return false;
  producer.Reset();
  if (!Ok(same_domain.Reserve(128, &consumer)) ||
      !Ok(independent.Reserve(128, &other_domain)) ||
      !Code(cached.TransferTo(other_domain), StatusCode::kInvalidArgument))
    return false;
  if (!Check(budget.snapshot().committed_bytes() == 208,
      "Detached cache not included before reuse") ||
      !Ok(cached.TransferTo(consumer)) || !Ok(cached.MakeLive(24))) return false;
  auto s = budget.snapshot();
  if (!Check(s.committed_bytes() == 128 && s.reserved_unbacked_bytes == 48 &&
      s.total.live_requested_bytes == 24 && s.total.live_capacity_bytes == 80,
      "Reused cache double-counted its reservation")) return false;

  // Transferring between two open jobs returns credit to the source, without
  // changing the total protection or the one underlying allocation.
  if (!Ok(budget.Reserve(128, &producer)) ||
      !Ok(cached.TransferTo(producer)) || !Ok(cached.TransferTo(producer)))
    return false;
  if (!Check(budget.snapshot().committed_bytes() == 256,
      "Open-job transfer changed total protection")) return false;
  ResourceAllocation second;
  if (!Ok(consumer.PrepareAllocation(ResourceClass::kCompletedFrame, 128, 128,
                                    &second)) || !Ok(second.Commit())) return false;
  // Move assignment must release the previous owner's charge exactly once.
  second = std::move(cached);
  if (!Check(!cached.valid() && cached.capacity_bytes() == 0 &&
      budget.snapshot().total.backing_count == 1,
      "Allocation move leaked or duplicated a charge")) return false;
  ResourceAllocation moved(std::move(second));
  consumer = std::move(producer);
  if (!Check(!producer.valid() && budget.snapshot().open_reservations == 1,
      "Reservation move failed to close its previous envelope")) return false;
  consumer.Reset();
  if (!Check(budget.snapshot().committed_bytes() == 80,
      "Retained output kept unused reservation capacity")) return false;
  moved.Reset();
  other_domain.Reset();
  return Empty(budget) && Empty(independent);
}

bool CheckPendingAndFailure() {
  ResourceBudget budget(100);
  ResourceReservation job;
  ResourceAllocation pending;
  if (!Code(budget.Reserve(101, &job), StatusCode::kOutOfMemory) ||
      !Code(budget.Reserve(0, &job), StatusCode::kInvalidArgument) ||
      !Code(budget.Reserve(1, nullptr), StatusCode::kInvalidArgument) ||
      !Ok(budget.Reserve(100, &job)) ||
      !Code(budget.Reserve(1, &job), StatusCode::kInvalidArgument)) return false;
  if (!Code(job.PrepareAllocation(ResourceClass::kCount, 1, 1, &pending),
            StatusCode::kInvalidArgument) ||
      !Code(job.PrepareAllocation(ResourceClass::kSerializer, 2, 1, &pending),
            StatusCode::kInvalidArgument) ||
      !Code(job.PrepareAllocation(ResourceClass::kSerializer, 101, 101, &pending),
            StatusCode::kOutOfMemory) ||
      !Ok(job.PrepareAllocation(ResourceClass::kSerializer, 70, 80, &pending)))
    return false;
  if (!Code(job.PrepareAllocation(ResourceClass::kInput, 1, 1, &pending),
            StatusCode::kInvalidArgument) ||
      !Code(pending.MakeIdle(), StatusCode::kFailedPrecondition)) return false;
  // A failed real allocation never becomes backing and returns its plan credit.
  pending.Reset();
  if (!Check(budget.snapshot().reserved_unbacked_bytes == 100 &&
      budget.snapshot().total.pending_count == 0,
      "Failed allocation lost reservation credit")) return false;
  if (!Ok(job.PrepareAllocation(ResourceClass::kInput, 10, 20, &pending)))
    return false;
  job.Reset();
  if (!Check(budget.snapshot().reserved_unbacked_bytes == 20,
      "Closing a producer lost pending allocation authorization")) return false;
  if (!Ok(pending.Commit()) ||
      !Code(pending.Commit(), StatusCode::kFailedPrecondition) ||
      !Ok(pending.MakeIdle()) ||
      !Code(pending.MakeIdle(), StatusCode::kFailedPrecondition) ||
      !Code(pending.MakeLive(21), StatusCode::kInvalidArgument) ||
      !Code(pending.MakeLive(0), StatusCode::kInvalidArgument) ||
      !Ok(pending.MakeLive(15))) return false;
  pending.Reset();
  try {
    ResourceReservation failed_job;
    ResourceAllocation failed_allocation;
    if (!Ok(budget.Reserve(100, &failed_job)) ||
        !Ok(failed_job.PrepareAllocation(ResourceClass::kSerializer, 100, 100,
                                        &failed_allocation))) return false;
    throw std::runtime_error("Simulated backing allocation failure");
  } catch (const std::runtime_error&) {}
  // Pending tickets can also be abandoned after their reservation closes.
  if (!Ok(budget.Reserve(100, &job)) ||
      !Ok(job.PrepareAllocation(ResourceClass::kInput, 70, 80, &pending)))
    return false;
  job.Reset();
  pending.Reset();
  return Empty(budget);
}

bool CheckOverflowAndLifetime() {
  ResourceBudget unlimited;
  const size_t maximum = std::numeric_limits<size_t>::max();
  ResourceReservation first, second, failed;
  ResourceAllocation pending;
  if (!Ok(unlimited.Reserve(maximum - 8, &first)) ||
      !Ok(unlimited.Reserve(8, &second)) ||
      !Code(unlimited.TryReserve(1, &failed), StatusCode::kUnavailable) ||
      !Ok(first.PrepareAllocation(ResourceClass::kInput, 8, 8, &pending)) ||
      !Ok(pending.TransferTo(second)) || !Ok(pending.Commit())) return false;
  if (!Check(unlimited.snapshot().committed_bytes() == maximum,
      "Unlimited accounting overflowed")) return false;
  first.Reset();
  second.Reset();
  pending.Reset();
  if (!Empty(unlimited)) return false;

  // Neither the domain wrapper nor its producer needs to outlive the backing.
  ResourceAllocation retained;
  {
    ResourceBudget transient(64);
    ResourceReservation producer;
    if (!Ok(transient.Reserve(64, &producer)) ||
        !Ok(producer.PrepareAllocation(ResourceClass::kCompletedFrame, 48, 64,
                                      &retained)) || !Ok(retained.Commit()))
      return false;
  }
  if (!Ok(retained.MakeIdle()) || !Ok(retained.MakeLive(64))) return false;
  retained.Reset();
  return true;
}

bool CheckFifoAndCancellation() {
  ResourceBudget budget(100);
  ResourceReservation blocker;
  if (!Ok(budget.Reserve(60, &blocker))) return false;
  std::atomic<bool> large_acquired{false}, small_acquired{false};
  std::atomic<bool> large_done{false}, small_done{false}, good{true};
  std::atomic<bool> release_large{false};
  std::jthread large([&](std::stop_token stop) {
    ResourceReservation job;
    const Status status = budget.Reserve(80, &job, stop);
    if (status.ok()) {
      large_acquired = true;
      while (!release_large && !stop.stop_requested()) std::this_thread::yield();
    } else if (status.code() != StatusCode::kUnavailable) good = false;
    large_done = true;
  });
  if (!Check(Until([&] { return budget.snapshot().waiting_requests == 1; }),
      "Large request did not queue")) return false;
  std::jthread small([&](std::stop_token stop) {
    ResourceReservation job;
    if (budget.Reserve(50, &job, stop).ok()) small_acquired = true;
    else if (!stop.stop_requested()) good = false;
    small_done = true;
  });
  if (!Check(Until([&] { return budget.snapshot().waiting_requests == 2; }),
      "Second request did not queue")) return false;
  ResourceReservation bypass;
  if (!Code(budget.TryReserve(1, &bypass), StatusCode::kUnavailable)) return false;
  blocker.Reset();
  if (!Check(Until([&] { return large_acquired.load(); }) && !small_acquired,
      "FIFO admission bypassed a large request")) return false;
  release_large = true;
  if (!Check(Until([&] { return large_done && small_done; }) && small_acquired,
      "Queued request did not make progress")) return false;
  large.join();
  small.join();
  if (!good || !Empty(budget)) return false;

  // Cancel the head while a following smaller request already fits. It must
  // wake immediately, without waiting for any unrelated allocation to finish.
  if (!Ok(budget.Reserve(60, &blocker))) return false;
  std::atomic<bool> cancelled{false};
  small_acquired = false;
  std::jthread head([&](std::stop_token stop) {
    ResourceReservation job;
    cancelled = budget.Reserve(80, &job, stop).code() == StatusCode::kUnavailable &&
      !job.valid();
  });
  if (!Until([&] { return budget.snapshot().waiting_requests == 1; })) return false;
  std::jthread follower([&](std::stop_token stop) {
    ResourceReservation job;
    small_acquired = budget.Reserve(20, &job, stop).ok();
  });
  if (!Until([&] { return budget.snapshot().waiting_requests == 2; })) return false;
  if (!Check(!small_acquired, "Small request bypassed the queued head")) return false;
  head.request_stop();
  if (!Check(Until([&] { return cancelled && small_acquired; }),
      "Head cancellation stranded its follower")) return false;
  head.join();
  follower.join();
  blocker.Reset();
  std::stop_source stopped;
  stopped.request_stop();
  if (!Code(budget.Reserve(10, &blocker, stopped.get_token()),
            StatusCode::kUnavailable) || blocker.valid()) return false;
  return Empty(budget);
}

bool CheckConcurrentAccounting() {
  ResourceBudget budget(256);
  std::atomic<bool> good{true};
  std::vector<std::jthread> workers;
  for (size_t worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&, shared = budget, worker](std::stop_token stop) {
      for (size_t iteration = 0; iteration < 250; ++iteration) {
        ResourceReservation job;
        ResourceAllocation first, second;
        const size_t capacity = 32 + (worker + iteration) % 33;
        if (!shared.Reserve(capacity, &job, stop).ok() ||
            !job.PrepareAllocation(ResourceClass::kAqScratch, 8, 16, &first).ok() ||
            !job.PrepareAllocation(ResourceClass::kCompletedFrame,
                                   capacity - 16, capacity - 16, &second).ok() ||
            !first.Commit().ok() || !second.Commit().ok() ||
            !first.MakeIdle().ok()) {
          good = false;
          return;
        }
        if (iteration % 2 == 0) job.Reset();
        if (!Consistent(shared.snapshot(), 256)) good = false;
        if (!first.MakeLive(4).ok()) good = false;
        // Destruction alternates between tickets-first and producer-first.
      }
    });
  }
  for (auto& worker : workers) worker.join();
  return Check(good, "Concurrent accounting diverged") && Empty(budget);
}

bool CheckRetainedBackingAndQueueRemoval() {
  ResourceBudget budget(100);
  ResourceReservation producer;
  ResourceAllocation retained;
  if (!Ok(budget.Reserve(100, &producer)) ||
      !Ok(producer.PrepareAllocation(ResourceClass::kCompletedFrame, 90, 100,
                                    &retained)) || !Ok(retained.Commit()))
    return false;
  producer.Reset();
  std::array<std::atomic<bool>, 3> acquired{};
  std::array<std::atomic<bool>, 3> cancelled{};
  std::vector<std::jthread> workers;
  for (size_t i = 0; i < 3; ++i) {
    workers.emplace_back([&, i](std::stop_token stop) {
      ResourceReservation job;
      const Status status = budget.Reserve(100, &job, stop);
      acquired[i] = status.ok();
      cancelled[i] = status.code() == StatusCode::kUnavailable && !job.valid();
    });
    if (!Check(Until([&] { return budget.snapshot().waiting_requests == i + 1; }),
        "Retained backing did not protect its capacity")) return false;
  }
  // Remove both an interior node and the tail while the head remains blocked.
  workers[1].request_stop();
  if (!Check(Until([&] { return cancelled[1].load(); }),
      "Interior cancellation failed")) return false;
  workers[2].request_stop();
  if (!Check(Until([&] { return cancelled[2].load(); }) &&
      budget.snapshot().waiting_requests == 1 && !acquired[0],
      "Tail cancellation damaged the queue")) return false;
  retained.Reset();
  if (!Check(Until([&] { return acquired[0].load(); }),
      "Releasing retained output did not wake admission")) return false;
  for (auto& worker : workers) worker.join();
  return Empty(budget);
}

bool CheckStateModel() {
  // This oracle recomputes protection from open jobs and surviving backings;
  // it does not update counters using the implementation's transition formulas.
  struct Job { bool open = false; size_t id = 0; size_t capacity = 0; };
  struct Backing {
    bool valid = false;
    size_t owner = 0;
    size_t capacity = 0;
    size_t requested = 0;
    size_t resource_class = 0;
    enum { kPending, kLive, kIdle } phase = kPending;
  };
  ResourceBudget budget(256);
  std::array<ResourceReservation, 4> jobs;
  std::array<Job, 4> model_jobs{};
  std::array<ResourceAllocation, 16> allocations;
  std::array<Backing, 16> model_allocations{};
  size_t next_id = 1;
  uint32_t random_state = 0x91ef2a37;
  const auto random = [&] {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
  };
  const auto assigned = [&](size_t id) {
    size_t bytes = 0;
    for (const auto& a : model_allocations)
      if (a.valid && a.owner == id) bytes += a.capacity;
    return bytes;
  };
  const auto expected = [&] {
    ResourceBudgetSnapshot result;
    size_t protected_bytes = 0;
    for (const auto& job : model_jobs) {
      if (!job.open) continue;
      ++result.open_reservations;
      protected_bytes += job.capacity;
    }
    for (const auto& a : model_allocations) {
      if (!a.valid) continue;
      bool producer_open = false;
      for (const auto& job : model_jobs)
        producer_open |= job.open && job.id == a.owner;
      if (!producer_open) protected_bytes += a.capacity;
      for (auto* usage : {&result.total, &result.classes[a.resource_class]}) {
        if (a.phase == Backing::kPending) {
          ++usage->pending_count;
          usage->pending_capacity_bytes += a.capacity;
        } else {
          ++usage->backing_count;
          if (a.phase == Backing::kLive) {
            usage->live_capacity_bytes += a.capacity;
            usage->live_requested_bytes += a.requested;
          } else usage->idle_capacity_bytes += a.capacity;
        }
      }
    }
    result.reserved_unbacked_bytes = protected_bytes -
      result.total.live_capacity_bytes - result.total.idle_capacity_bytes;
    return result;
  };
  const auto equal_usage = [](const ResourceUsage& a, const ResourceUsage& b) {
    return a.live_requested_bytes == b.live_requested_bytes &&
      a.live_capacity_bytes == b.live_capacity_bytes &&
      a.idle_capacity_bytes == b.idle_capacity_bytes &&
      a.pending_capacity_bytes == b.pending_capacity_bytes &&
      a.backing_count == b.backing_count && a.pending_count == b.pending_count;
  };
  for (size_t step = 0; step < 20000; ++step) {
    const size_t action = random() % 11;
    const size_t j = random() % jobs.size();
    const size_t a = random() % allocations.size();
    auto& job = model_jobs[j];
    auto& backing = model_allocations[a];
    if (action == 0 && !job.open) {
      const size_t bytes = 1 + random() % 300;
      const auto code = bytes > 256 ? StatusCode::kOutOfMemory :
        bytes > 256 - expected().committed_bytes() ? StatusCode::kUnavailable :
        StatusCode::kOk;
      if (!Code(budget.TryReserve(bytes, &jobs[j]), code)) return false;
      if (code == StatusCode::kOk) job = {true, next_id++, bytes};
    } else if (action == 1 && job.open) {
      jobs[j].Reset();
      job.open = false;
    } else if (action == 2 && job.open && !backing.valid) {
      const size_t capacity = 1 + random() % 120;
      const size_t requested = 1 + random() % capacity;
      const size_t kind = random() % static_cast<size_t>(ResourceClass::kCount);
      const auto code = capacity > job.capacity - assigned(job.id)
        ? StatusCode::kOutOfMemory : StatusCode::kOk;
      if (!Code(jobs[j].PrepareAllocation(static_cast<ResourceClass>(kind),
          requested, capacity, &allocations[a]), code)) return false;
      if (code == StatusCode::kOk)
        backing = {true, job.id, capacity, requested, kind, Backing::kPending};
    } else if (action == 3 && backing.valid && backing.phase == Backing::kPending) {
      if (!Ok(allocations[a].Commit())) return false;
      backing.phase = Backing::kLive;
    } else if (action == 4 && backing.valid && backing.phase == Backing::kLive) {
      if (!Ok(allocations[a].MakeIdle())) return false;
      backing.phase = Backing::kIdle;
    } else if (action == 5 && backing.valid && backing.phase == Backing::kIdle) {
      const size_t requested = 1 + random() % backing.capacity;
      if (!Ok(allocations[a].MakeLive(requested))) return false;
      backing.requested = requested;
      backing.phase = Backing::kLive;
    } else if (action == 6 && backing.valid && job.open) {
      const auto code = backing.owner != job.id &&
        backing.capacity > job.capacity - assigned(job.id)
        ? StatusCode::kOutOfMemory : StatusCode::kOk;
      if (!Code(allocations[a].TransferTo(jobs[j]), code)) return false;
      if (code == StatusCode::kOk) backing.owner = job.id;
    } else if (action == 7 && backing.valid) {
      allocations[a].Reset();
      backing.valid = false;
    } else if (action == 8) {
      const size_t destination = random() % allocations.size();
      if (destination != a) {
        allocations[destination] = std::move(allocations[a]);
        model_allocations[destination] = backing;
        backing.valid = false;
      }
    } else if (action == 9) {
      const size_t destination = random() % jobs.size();
      if (destination != j) {
        jobs[destination] = std::move(jobs[j]);
        model_jobs[destination] = job;
        job.open = false;
      }
    } else if (action == 10 && job.open) {
      const size_t capacity = random() % (job.capacity + 17);
      const auto code = capacity > job.capacity ? StatusCode::kInvalidArgument :
        capacity < assigned(job.id) ? StatusCode::kFailedPrecondition :
        StatusCode::kOk;
      if (!Code(jobs[j].ReduceCapacity(capacity), code)) return false;
      if (code == StatusCode::kOk) job.capacity = capacity;
    }
    const auto actual = budget.snapshot();
    const auto oracle = expected();
    if (!Check(Consistent(actual, 256) &&
        actual.open_reservations == oracle.open_reservations &&
        actual.reserved_unbacked_bytes == oracle.reserved_unbacked_bytes &&
        equal_usage(actual.total, oracle.total), "Resource state model diverged"))
      return false;
    for (size_t i = 0; i < actual.classes.size(); ++i)
      if (!Check(equal_usage(actual.classes[i], oracle.classes[i]),
          "Per-class resource model diverged")) return false;
  }
  for (auto& job : jobs) job.Reset();
  for (auto& allocation : allocations) allocation.Reset();
  return Empty(budget);
}
}  // namespace

int main() {
  return CheckLifecycle() && CheckTransfersAndMoves() && CheckPendingAndFailure() &&
    CheckOverflowAndLifetime() && CheckFifoAndCancellation() &&
    CheckConcurrentAccounting() && CheckRetainedBackingAndQueueRemoval() &&
    CheckStateModel()
      ? EXIT_SUCCESS : EXIT_FAILURE;
}
