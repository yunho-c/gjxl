// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stop_token>
#include <utility>

#include "core/status.h"

namespace gjxl::resource_budget_internal {

// Classify the backing owner, not each view/consumer of the same allocation.
enum class ResourceClass : size_t {
  kInput,
  kPreparation,
  kAcSearch,
  kAqScratch,
  kButteraugli,
  kCompletedFrame,
  kSerializer,
  kRetainedResult,
  kCount,
};

struct ResourceUsage {
  size_t live_requested_bytes = 0;
  size_t live_capacity_bytes = 0;
  size_t idle_capacity_bytes = 0;
  size_t pending_capacity_bytes = 0;
  size_t backing_count = 0;
  size_t pending_count = 0;
};

struct ResourceBudgetSnapshot {
  ResourceUsage total;
  std::array<ResourceUsage, static_cast<size_t>(ResourceClass::kCount)> classes{};
  // Unused admission capacity plus pending allocation tickets. Pending bytes
  // are a subset of this number, not an additional charge.
  size_t reserved_unbacked_bytes = 0;
  size_t open_reservations = 0;
  size_t waiting_requests = 0;
  size_t peak_backing_bytes = 0;
  size_t peak_committed_bytes = 0;

  [[nodiscard]] size_t committed_bytes() const noexcept {
    return reserved_unbacked_bytes + total.live_capacity_bytes +
      total.idle_capacity_bytes;
  }
};

namespace detail {
struct Waiter {
  Waiter* previous = nullptr;
  Waiter* next = nullptr;
};

struct BudgetState {
  explicit BudgetState(size_t limit)
    : limit_bytes(limit == 0 ? std::numeric_limits<size_t>::max() : limit) {}

  void UpdatePeaks() noexcept {
    snapshot.peak_backing_bytes = std::max(snapshot.peak_backing_bytes,
      snapshot.total.live_capacity_bytes + snapshot.total.idle_capacity_bytes);
    snapshot.peak_committed_bytes = std::max(
      snapshot.peak_committed_bytes, snapshot.committed_bytes());
    assert(snapshot.committed_bytes() <= limit_bytes);
  }

  void Enqueue(Waiter* waiter) noexcept {
    waiter->previous = last;
    if (last != nullptr) last->next = waiter;
    else first = waiter;
    last = waiter;
    ++snapshot.waiting_requests;
  }

  void Remove(Waiter* waiter) noexcept {
    if (waiter->previous != nullptr) waiter->previous->next = waiter->next;
    else first = waiter->next;
    if (waiter->next != nullptr) waiter->next->previous = waiter->previous;
    else last = waiter->previous;
    --snapshot.waiting_requests;
  }

  const size_t limit_bytes;
  std::mutex mutex;
  std::condition_variable_any changed;
  ResourceBudgetSnapshot snapshot;
  Waiter* first = nullptr;
  Waiter* last = nullptr;
};

struct ReservationState {
  std::shared_ptr<BudgetState> budget;
  size_t capacity_bytes = 0;
  size_t charged_bytes = 0;
  bool open = false;
};
}  // namespace detail

class ResourceBudget;
class ResourceAllocation;

/// A complete-work reservation. Allocations never wait or grow this envelope.
/// Closing it releases unused capacity, but not any surviving backing tickets.
/// Separate handles can be used concurrently; each owning handle needs exclusive
/// access while moving/resetting it or changing its allocation's state.
class ResourceReservation {
public:
  ResourceReservation() = default;
  ~ResourceReservation() { Reset(); }
  ResourceReservation(const ResourceReservation&) = delete;
  ResourceReservation& operator=(const ResourceReservation&) = delete;
  ResourceReservation(ResourceReservation&& other) noexcept = default;
  ResourceReservation& operator=(ResourceReservation&& other) noexcept {
    if (this != &other) {
      Reset();
      state_ = std::move(other.state_);
    }
    return *this;
  }

  [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }

  /// Releases no-longer-needed plan capacity at a proven stage boundary. The
  /// envelope may only shrink and cannot release existing allocation tickets.
  [[nodiscard]] Status ReduceCapacity(size_t capacity_bytes) {
    if (!valid()) return Status::InvalidArgument("Invalid resource reservation");
    auto& budget = *state_->budget;
    {
      std::lock_guard lock(budget.mutex);
      if (capacity_bytes > state_->capacity_bytes)
        return Status::InvalidArgument("Resource reservations cannot grow");
      if (capacity_bytes < state_->charged_bytes)
        return Status::FailedPrecondition("Reservation still has allocation tickets");
      budget.snapshot.reserved_unbacked_bytes -=
        state_->capacity_bytes - capacity_bytes;
      state_->capacity_bytes = capacity_bytes;
    }
    budget.changed.notify_all();
    return Status::Ok();
  }

  void Reset() noexcept {
    auto state = std::move(state_);
    if (state == nullptr) return;
    auto& budget = *state->budget;
    {
      std::lock_guard lock(budget.mutex);
      assert(state->open && state->charged_bytes <= state->capacity_bytes);
      budget.snapshot.reserved_unbacked_bytes -=
        state->capacity_bytes - state->charged_bytes;
      --budget.snapshot.open_reservations;
      state->open = false;
    }
    budget.changed.notify_all();
  }

  /// Authorizes one backing allocation before allocating. Call Commit only
  /// after successful allocation; failure unwinds the pending ticket instead.
  [[nodiscard]] Status PrepareAllocation(
    ResourceClass resource_class, size_t requested_bytes, size_t capacity_bytes,
    ResourceAllocation* allocation);

private:
  friend class ResourceBudget;
  friend class ResourceAllocation;
  std::shared_ptr<detail::ReservationState> state_;
};

/// Move-only charge attached to one backing owner. Non-owning slices have no
/// ticket. The caller must free the backing before resetting its ticket.
class ResourceAllocation {
public:
  ResourceAllocation() = default;
  ~ResourceAllocation() { Reset(); }
  ResourceAllocation(const ResourceAllocation&) = delete;
  ResourceAllocation& operator=(const ResourceAllocation&) = delete;
  ResourceAllocation(ResourceAllocation&& other) noexcept { MoveFrom(other); }
  ResourceAllocation& operator=(ResourceAllocation&& other) noexcept {
    if (this != &other) {
      Reset();
      MoveFrom(other);
    }
    return *this;
  }

  [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
  [[nodiscard]] size_t capacity_bytes() const noexcept { return capacity_; }

  [[nodiscard]] Status Commit() {
    if (!valid() || phase_ != Phase::kPending)
      return Status::FailedPrecondition("Allocation is not pending");
    auto& budget = *state_->budget;
    std::lock_guard lock(budget.mutex);
    ForUsage([&](ResourceUsage& usage) {
      usage.pending_capacity_bytes -= capacity_;
      --usage.pending_count;
      usage.live_capacity_bytes += capacity_;
      usage.live_requested_bytes += requested_;
      ++usage.backing_count;
    });
    budget.snapshot.reserved_unbacked_bytes -= capacity_;
    phase_ = Phase::kLive;
    budget.UpdatePeaks();
    return Status::Ok();
  }

  /// Call only after all users complete. Idle capacity remains charged even
  /// when its OS pages are volatile or reclaimed; this is not a RAM counter.
  [[nodiscard]] Status MakeIdle() {
    if (!valid() || phase_ != Phase::kLive)
      return Status::FailedPrecondition("Allocation is not live");
    auto& budget = *state_->budget;
    std::lock_guard lock(budget.mutex);
    ForUsage([&](ResourceUsage& usage) {
      usage.live_requested_bytes -= requested_;
      usage.live_capacity_bytes -= capacity_;
      usage.idle_capacity_bytes += capacity_;
    });
    phase_ = Phase::kIdle;
    return Status::Ok();
  }

  /// Call after acquiring exclusive ownership and restoring nonvolatile state.
  [[nodiscard]] Status MakeLive(size_t requested_bytes) {
    if (!valid() || phase_ != Phase::kIdle)
      return Status::FailedPrecondition("Allocation is not idle");
    if (requested_bytes == 0 || requested_bytes > capacity_)
      return Status::InvalidArgument("Invalid reused allocation size");
    auto& budget = *state_->budget;
    std::lock_guard lock(budget.mutex);
    ForUsage([&](ResourceUsage& usage) {
      usage.idle_capacity_bytes -= capacity_;
      usage.live_capacity_bytes += capacity_;
      usage.live_requested_bytes += requested_bytes;
    });
    requested_ = requested_bytes;
    phase_ = Phase::kLive;
    return Status::Ok();
  }

  /// Transfers a backing's charge into another admitted job in the same domain.
  /// No backing is copied or counted twice. Cross-domain caches must shed their
  /// allocation instead of silently borrowing another domain's allowance.
  [[nodiscard]] Status TransferTo(ResourceReservation& reservation) {
    if (!valid() || !reservation.valid())
      return Status::InvalidArgument("Invalid allocation transfer");
    auto next = reservation.state_;
    if (next->budget != state_->budget)
      return Status::InvalidArgument("Allocation domains differ");
    auto& budget = *state_->budget;
    {
      std::lock_guard lock(budget.mutex);
      if (next == state_) return Status::Ok();
      if (capacity_ > next->capacity_bytes - next->charged_bytes)
        return Status::OutOfMemory("Allocation exceeds destination reservation");
      state_->charged_bytes -= capacity_;
      // An open source's newly unused capacity exactly replaces the target's
      // consumed capacity. Apply the net change to avoid transient overflow.
      if (!state_->open) budget.snapshot.reserved_unbacked_bytes -= capacity_;
      next->charged_bytes += capacity_;
      state_ = std::move(next);
    }
    budget.changed.notify_all();
    return Status::Ok();
  }

  void Reset() noexcept {
    if (!valid()) return;
    auto state = state_;
    auto& budget = *state->budget;
    {
      std::lock_guard lock(budget.mutex);
      ForUsage([&](ResourceUsage& usage) {
        if (phase_ == Phase::kPending) {
          usage.pending_capacity_bytes -= capacity_;
          --usage.pending_count;
        } else {
          --usage.backing_count;
          if (phase_ == Phase::kLive) {
            usage.live_capacity_bytes -= capacity_;
            usage.live_requested_bytes -= requested_;
          } else usage.idle_capacity_bytes -= capacity_;
        }
      });
      state->charged_bytes -= capacity_;
      if (phase_ == Phase::kPending) {
        if (!state->open) budget.snapshot.reserved_unbacked_bytes -= capacity_;
      } else if (state->open) {
        budget.snapshot.reserved_unbacked_bytes += capacity_;
      }
      state_.reset();
      capacity_ = requested_ = 0;
    }
    budget.changed.notify_all();
  }

private:
  friend class ResourceReservation;
  enum class Phase { kPending, kLive, kIdle };

  template <typename Function>
  void ForUsage(Function&& function) {
    auto& snapshot = state_->budget->snapshot;
    function(snapshot.total);
    function(snapshot.classes[static_cast<size_t>(resource_class_)]);
  }

  void MoveFrom(ResourceAllocation& other) noexcept {
    state_ = std::move(other.state_);
    capacity_ = std::exchange(other.capacity_, 0);
    requested_ = std::exchange(other.requested_, 0);
    resource_class_ = other.resource_class_;
    phase_ = other.phase_;
  }

  std::shared_ptr<detail::ReservationState> state_;
  size_t capacity_ = 0;
  size_t requested_ = 0;
  ResourceClass resource_class_ = ResourceClass::kInput;
  Phase phase_ = Phase::kPending;
};

inline Status ResourceReservation::PrepareAllocation(
  ResourceClass resource_class, size_t requested_bytes, size_t capacity_bytes,
  ResourceAllocation* allocation) {
  if (!valid() || allocation == nullptr || allocation->valid() ||
      requested_bytes == 0 || requested_bytes > capacity_bytes ||
      static_cast<size_t>(resource_class) >=
        static_cast<size_t>(ResourceClass::kCount))
    return Status::InvalidArgument("Invalid allocation reservation");
  auto& budget = *state_->budget;
  std::lock_guard lock(budget.mutex);
  if (capacity_bytes > state_->capacity_bytes - state_->charged_bytes)
    return Status::OutOfMemory("Allocation exceeds admitted resource plan");
  state_->charged_bytes += capacity_bytes;
  allocation->state_ = state_;
  allocation->capacity_ = capacity_bytes;
  allocation->requested_ = requested_bytes;
  allocation->resource_class_ = resource_class;
  allocation->phase_ = ResourceAllocation::Phase::kPending;
  allocation->ForUsage([&](ResourceUsage& usage) {
    usage.pending_capacity_bytes += capacity_bytes;
    ++usage.pending_count;
  });
  return Status::Ok();
}

/// One immutable managed-capacity domain. Copies share the same allowance.
/// Zero means unlimited capacity (still checked for size_t overflow), not an
/// automatic fraction of RAM. This primitive does not estimate workload sizes,
/// evict caches, intercept allocation, or impose a process-footprint limit.
class ResourceBudget {
public:
  explicit ResourceBudget(size_t limit_bytes = 0)
    : state_(std::make_shared<detail::BudgetState>(limit_bytes)) {}

  [[nodiscard]] ResourceBudgetSnapshot snapshot() const {
    std::lock_guard lock(state_->mutex);
    return state_->snapshot;
  }

  [[nodiscard]] Status TryReserve(
    size_t capacity_bytes, ResourceReservation* reservation) const {
    Status status = Validate(capacity_bytes, reservation);
    if (!status.ok()) return status;
    std::shared_ptr<detail::ReservationState> candidate;
    status = MakeCandidate(capacity_bytes, &candidate);
    if (!status.ok()) return status;
    std::lock_guard lock(state_->mutex);
    if (state_->first != nullptr || !Fits(capacity_bytes))
      return Status::Unavailable("Managed resource capacity is busy");
    Publish(std::move(candidate), reservation);
    return Status::Ok();
  }

  /// FIFO admission, including unequal request sizes. Only callers holding no
  /// resources needed by another waiting request may wait here. Cache owners
  /// must arrange eviction before waiting; this class owns no physical storage.
  [[nodiscard]] Status Reserve(
    size_t capacity_bytes, ResourceReservation* reservation,
    std::stop_token stop = {}) const {
    Status status = Validate(capacity_bytes, reservation);
    if (!status.ok()) return status;
    std::shared_ptr<detail::ReservationState> candidate;
    status = MakeCandidate(capacity_bytes, &candidate);
    if (!status.ok()) return status;
    std::unique_lock lock(state_->mutex);
    detail::Waiter waiter;
    state_->Enqueue(&waiter);
    bool ready = false;
    try {
      ready = state_->changed.wait(lock, stop, [&] {
        return state_->first == &waiter && Fits(capacity_bytes);
      });
    } catch (...) {
      state_->Remove(&waiter);
      state_->changed.notify_all();
      throw;
    }
    state_->Remove(&waiter);
    if (ready && !stop.stop_requested()) Publish(std::move(candidate), reservation);
    else ready = false;
    lock.unlock();
    state_->changed.notify_all();
    return ready ? Status::Ok() : Status::Unavailable("Resource admission cancelled");
  }

private:
  [[nodiscard]] Status Validate(
    size_t capacity, const ResourceReservation* reservation) const {
    if (capacity == 0 || reservation == nullptr || reservation->valid())
      return Status::InvalidArgument("Invalid resource reservation");
    if (capacity > state_->limit_bytes)
      return Status::OutOfMemory("Resource plan exceeds managed memory limit");
    return Status::Ok();
  }

  [[nodiscard]] Status MakeCandidate(
    size_t capacity, std::shared_ptr<detail::ReservationState>* candidate) const {
    try {
      *candidate = std::make_shared<detail::ReservationState>();
      (*candidate)->budget = state_;
      (*candidate)->capacity_bytes = capacity;
      return Status::Ok();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("Resource reservation metadata allocation failed");
    }
  }

  [[nodiscard]] bool Fits(size_t capacity) const noexcept {
    return capacity <= state_->limit_bytes - state_->snapshot.committed_bytes();
  }

  void Publish(std::shared_ptr<detail::ReservationState> candidate,
               ResourceReservation* reservation) const noexcept {
    candidate->open = true;
    state_->snapshot.reserved_unbacked_bytes += candidate->capacity_bytes;
    ++state_->snapshot.open_reservations;
    state_->UpdatePeaks();
    reservation->state_ = std::move(candidate);
  }

  std::shared_ptr<detail::BudgetState> state_;
};

}  // namespace gjxl::resource_budget_internal
