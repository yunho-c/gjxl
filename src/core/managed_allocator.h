// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

#include "core/resource_context.h"

namespace gjxl::resource_budget_internal {

/// Retains the precise admission error across standard-container allocation.
/// Existing bad_alloc boundaries still recognize it as allocation failure.
class ManagedAllocationFailure final : public std::bad_alloc {
public:
  explicit ManagedAllocationFailure(Status status) noexcept
    : status_(std::move(status)) {}
  [[nodiscard]] const char* what() const noexcept override {
    return "Managed host allocation failed";
  }
  [[nodiscard]] const Status& status() const noexcept { return status_; }
private:
  Status status_;
};

namespace detail {
inline thread_local size_t managed_host_backings_before_failure =
  std::numeric_limits<size_t>::max();
}

inline void ArmNextManagedHostAllocationFailureForTest() noexcept {
  detail::managed_host_backings_before_failure = 0;
}

inline void ArmManagedHostAllocationFailureAfterForTest(size_t count) noexcept {
  detail::managed_host_backings_before_failure = count;
}

[[nodiscard]] inline bool ManagedHostAllocationFailurePendingForTest() noexcept {
  return detail::managed_host_backings_before_failure !=
    std::numeric_limits<size_t>::max();
}

inline void DisarmManagedHostAllocationFailureForTest() noexcept {
  detail::managed_host_backings_before_failure = std::numeric_limits<size_t>::max();
}

/// Stateless allocation-time domain selection, with an allocation-owned ticket.
/// Moves/swaps do not change a backing's domain; copies/new capacity use the
/// current scope. Deallocation needs neither that scope nor its producer thread.
/// Charge is n * sizeof(T), the container's backing request, not its logical size.
/// The aligned ticket header and allocator implementation overhead are excluded.
// kCount means inherit the current owner class; fixed owners avoid relabelling
// hot operations that do not allocate (for example individual bit writes).
template <typename T, ResourceClass Owner = ResourceClass::kCount>
class ManagedAllocator {
  static constexpr size_t kAlignment = std::max(
    alignof(T), std::max(alignof(ResourceAllocation), alignof(std::max_align_t)));
  struct alignas(kAlignment) Header {
    explicit Header(ResourceAllocation ticket) noexcept
      : allocation(std::move(ticket)) {}
    ResourceAllocation allocation;
  };

public:
  using value_type = T;
  using is_always_equal = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  template <typename U> struct rebind { using other = ManagedAllocator<U, Owner>; };

  ManagedAllocator() = default;
  template <typename U>
  ManagedAllocator(const ManagedAllocator<U, Owner>&) noexcept {}

  [[nodiscard]] size_t max_size() const noexcept {
    return std::min(
      (std::numeric_limits<size_t>::max() - sizeof(Header)) / sizeof(T),
      static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(T));
  }

  [[nodiscard]] T* allocate(size_t count) {
    if (count > max_size()) throw std::bad_array_new_length();
    if (count == 0) return nullptr;
    const size_t bytes = count * sizeof(T);
    ResourceAllocation allocation;
    const auto context = CurrentResourceContext();
    if (context.reservation != nullptr || context.track_host_allocations) {
      const ResourceClassScope resource_class(
        Owner == ResourceClass::kCount ? context.resource_class : Owner);
      Status status = PrepareResourceAllocation(bytes, bytes, &allocation);
      if (!status.ok()) throw ManagedAllocationFailure(std::move(status));
    }
    if (ManagedHostAllocationFailurePendingForTest()) {
      if (detail::managed_host_backings_before_failure == 0) {
        DisarmManagedHostAllocationFailureForTest();
        throw std::bad_alloc();
      }
      --detail::managed_host_backings_before_failure;
    }
    auto* backing = static_cast<std::byte*>(::operator new(
      sizeof(Header) + bytes, std::align_val_t{kAlignment}));
    if (allocation.valid()) {
      Status status = allocation.Commit();
      if (!status.ok()) {
        ::operator delete(backing, std::align_val_t{kAlignment});
        throw ManagedAllocationFailure(std::move(status));
      }
    }
    std::construct_at(reinterpret_cast<Header*>(backing), std::move(allocation));
    return reinterpret_cast<T*>(backing + sizeof(Header));
  }

  void deallocate(T* pointer, size_t /*count*/) noexcept {
    if (pointer == nullptr) return;
    auto* backing = reinterpret_cast<std::byte*>(pointer) - sizeof(Header);
    auto* header = reinterpret_cast<Header*>(backing);
    // Keep the ticket alive on this stack until AFTER freeing the backing.
    ResourceAllocation allocation = std::move(header->allocation);
    std::destroy_at(header);
    ::operator delete(backing, std::align_val_t{kAlignment});
  }

  template <typename U>
  [[nodiscard]] bool operator==(const ManagedAllocator<U, Owner>&) const noexcept {
    return true;
  }
};

template <typename T, ResourceClass Owner = ResourceClass::kCount>
using ManagedVector = std::vector<T, ManagedAllocator<T, Owner>>;

}  // namespace gjxl::resource_budget_internal
