// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <limits>

#include "core/managed_allocator.h"
#include "core/stdlib_storage_compat.h"

namespace gjxl::resource_budget_internal {

enum class VectorCapacityPolicy {
  kFreshExact,  // One allocation; subsequent writes stay within that capacity.
  kReusedExact, // Explicit reserve/assign; no implicit growth beyond a reserve.
  kGrowing,     // May also push/resize/insert forward ranges/shrink_to_fit.
};

enum class StringCapacityPolicy {
  kFresh,   // Count/pointer/forward-range construction or copy construction.
  kGrowing, // May assign, append, insert, resize, reserve or shrink_to_fit.
};

/// Conservative sum of vector backing capacities and replacement peaks. Counts
/// bound ALL sizes and reserve requests since each owner was initially empty,
/// not just its current logical size. Moves/swaps from larger owners,
/// single-pass input ranges, vector<bool>, and allocations inside elements need
/// separate accounting. Headers, allocator overhead and inline objects are
/// excluded. Summing peaks is safe even when their individual high-water marks
/// differ.
struct HostStorageBound {
  size_t retained_bytes = 0;
  size_t peak_bytes = 0;

  [[nodiscard]] bool Add(const HostStorageBound &other, size_t count = 1) {
    constexpr size_t max = std::numeric_limits<size_t>::max();
    if (retained_bytes > peak_bytes || other.retained_bytes > other.peak_bytes ||
        (count != 0 && other.peak_bytes > (max - peak_bytes) / count) ||
        (count != 0 && other.retained_bytes > (max - retained_bytes) / count))
      return false;
    retained_bytes += other.retained_bytes * count;
    peak_bytes += other.peak_bytes * count;
    return true;
  }

  template <typename T>
  [[nodiscard]] bool AddVector(size_t maximum_count,
                               VectorCapacityPolicy policy,
                               size_t instances = 1) {
    static_assert(!std::is_same_v<T, bool>);
    if (maximum_count > ManagedAllocator<T>{}.max_size())
      return false;
    const size_t bytes = maximum_count * sizeof(T);
    size_t retained_factor = 1, peak_factor = 1;
    switch (policy) {
    case VectorCapacityPolicy::kFreshExact:
      break;
    case VectorCapacityPolicy::kReusedExact:
      peak_factor = stdlib_storage_internal::kVectorReplacementPeakFactor;
      break;
    case VectorCapacityPolicy::kGrowing:
      retained_factor = stdlib_storage_internal::kVectorGrowingRetainedFactor;
      peak_factor = stdlib_storage_internal::kVectorGrowingPeakFactor;
      break;
    default:
      return false;
    }
    if (bytes > std::numeric_limits<size_t>::max() / peak_factor)
      return false;
    return Add({bytes * retained_factor, bytes * peak_factor}, instances);
  }

  /// Character backing for ManagedString, including its terminator.
  /// Counts bound every size/reserve request since the owner was empty, also
  /// when it is now cleared. Moves from larger owners, nontrivial/input ranges,
  /// and other character types need separate bounds. Inline storage is already
  /// counted in the containing record, not a second backing allocation.
  [[nodiscard]] bool AddString(size_t maximum_length,
                                StringCapacityPolicy policy,
                                size_t instances = 1) {
    const ManagedString<> empty;
    if (maximum_length > empty.max_size() ||
        (policy != StringCapacityPolicy::kFresh &&
         policy != StringCapacityPolicy::kGrowing))
      return false;
    if (maximum_length <= empty.capacity())
      return Add({}, instances);
    // Backing assumptions, including rounding and replacement overlap, live
    // in the compatibility boundary rather than in individual storage plans.
    using namespace stdlib_storage_internal;
    constexpr size_t max = std::numeric_limits<size_t>::max();
    if (maximum_length > max - kStringFreshSlack)
      return false;
    const size_t fresh = maximum_length + kStringFreshSlack;
    if (policy == StringCapacityPolicy::kFresh)
      return Add({fresh, fresh}, instances);
    if (fresh > max / kStringGrowingPeakFactor)
      return false;
    return Add({kStringGrowingRetainedFactor * fresh,
                kStringGrowingPeakFactor * fresh}, instances);
  }

  bool operator==(const HostStorageBound &) const = default;
};

} // namespace gjxl::resource_budget_internal
