// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <cstdlib>
#include <iostream>

#include "core/host_storage_bound.h"
#include "core/managed_publication.h"

namespace {
using namespace gjxl::resource_budget_internal;
namespace compat = gjxl::stdlib_storage_internal;

bool Check(bool good, const char* message) {
  if (!good) std::cerr << message << '\n';
  return good;
}

template <typename T>
bool ExactAllocatorResults() {
#if __cplusplus >= 202302L
  // Exercise both C++23 dispatch paths directly: the standard allocator's
  // member and allocator_traits' fallback for GJXL's custom allocator.
  static_assert(!requires(ManagedAllocator<T> allocator) { allocator.allocate_at_least(1); });
  for (size_t n : {1ul, 7ul, 16ul, 31ul, 256ul, 4097ul}) {
    std::allocator<T> standard;
    auto result = std::allocator_traits<decltype(standard)>::allocate_at_least(standard, n);
    const bool exact = result.count == n;
    standard.deallocate(result.ptr, result.count);
    if (!Check(exact, "Standard allocate_at_least returned excess capacity")) return false;
    ManagedHostScope scope(ResourceClass::kSerializer);
    ManagedAllocator<T> managed;
    auto backing = std::allocator_traits<decltype(managed)>::allocate_at_least(managed, n);
    const bool accounted = backing.count == n &&
      DefaultResourceBudget().snapshot().total.live_capacity_bytes == n * sizeof(T);
    managed.deallocate(backing.ptr, backing.count);
    if (!Check(accounted && DefaultResourceBudget().snapshot().committed_bytes() == 0,
               "Managed allocate_at_least fallback changed capacity or accounting")) return false;
  }
#endif
  return true;
}

template <typename T>
bool ExactVectors() {
  for (size_t n : {0ul, 1ul, 7ul, 16ul, 31ul, 256ul, 4097ul}) {
    auto counted = compat::MakeExactVector<T>(n);
    auto ranged = compat::MakeExactVector<T>(std::span<const T>(counted));
    if (!Check(counted.capacity() == n && ranged.capacity() == n,
               "Fresh publication backing is not exact")) return false;
    ManagedVector<T> reserved;
    reserved.reserve(n);
    if (!Check(reserved.capacity() == n, "Vector reserve is not exact")) return false;
    // The adaptive block-context map copies cells, then resizes to 3*cells.
    ManagedVector<T> remap(n);
    auto map = remap;
    map.resize(3 * n);
    if (!Check(map.capacity() == 3 * n, "Tripled map backing is not exact")) return false;
  }
  return true;
}

bool GrowingVectors() {
  for (size_t n : {1ul, 17ul, 255ul, 1025ul}) {
    HostStorageBound bound;
    if (!bound.AddVector<int>(n, VectorCapacityPolicy::kGrowing)) return false;
    ResourceBudget budget;
    ResourceReservation job;
    if (!budget.Reserve(bound.peak_bytes, &job).ok()) return false;
    {
      ResourceContextScope scope({&job, ResourceClass::kSerializer});
      ManagedVector<int> values;
      for (size_t i = 0; i < n; ++i) values.push_back(static_cast<int>(i));
      values.clear();
      values.resize(n / 2);
      values.shrink_to_fit();
      std::array<int, 1> one{1};
      values.insert(values.end(), one.begin(), one.end());
      values.reserve(n);
      values.assign(n, 7);
      const auto s = budget.snapshot();
      if (!Check(s.total.live_capacity_bytes <= bound.retained_bytes &&
                 s.peak_backing_bytes <= bound.peak_bytes,
                 "Observed vector growth/replacement exceeds its bound")) return false;
    }
    job.Reset();
    if (!Check(budget.snapshot().committed_bytes() == 0, "Vector charge leaked")) return false;
  }
  return true;
}

bool StringPointers() {
  auto& budget = DefaultResourceBudget();
  for (size_t n = 0; n <= 128; ++n) {
    ManagedHostScope scope(ResourceClass::kDiagnostics);
    ManagedString<> original(n, 'x');
    ManagedString<> value(std::move(original));
    const bool allocated = budget.snapshot().total.backing_count != 0;
    if (!Check((compat::StringHeapData(value) != nullptr) == allocated,
               "String heap detection disagrees with actual allocation")) return false;
    value.clear(); // Heap backing remains even when the string is empty.
    if (!Check((compat::StringHeapData(value) != nullptr) == allocated,
               "Cleared string lost its backing classification")) return false;
    ReleaseManagedBackingAfterPublication(value);
    if (!Check(budget.snapshot().committed_bytes() == 0,
               "String publication failed to release its backing charge")) return false;
  }
  return true;
}

// Observe actual rebound allocations without using the private node type or
// copying a layout into the test. Pointer arrays are buckets; other requests
// are nodes. Both sides of bucket replacement contribute to the physical peak.
struct AllocationLog {
  size_t nodes = 0, buckets = 0, peak = 0;
  void Add(size_t bytes, bool pointer) {
    (pointer ? buckets : nodes) += bytes;
    peak = std::max(peak, nodes + buckets);
  }
};
template <typename T>
struct ObservedAllocator {
  using value_type = T;
  AllocationLog* log;
  explicit ObservedAllocator(AllocationLog* value) : log(value) {}
  template <typename U>
  ObservedAllocator(const ObservedAllocator<U>& other) : log(other.log) {}
  T* allocate(size_t n) {
    auto* data = std::allocator<T>{}.allocate(n);
    log->Add(n * sizeof(T), std::is_pointer_v<T>);
    return data;
  }
  void deallocate(T* data, size_t n) {
    (std::is_pointer_v<T> ? log->buckets : log->nodes) -= n * sizeof(T);
    std::allocator<T>{}.deallocate(data, n);
  }
  template <typename U>
  bool operator==(const ObservedAllocator<U>& other) const { return log == other.log; }
};

bool HashBacking() {
  using Backing = compat::UnorderedMapBacking<uint32_t, uint64_t>;
  using Entry = std::pair<const uint32_t, uint64_t>;
  using Map = std::unordered_map<uint32_t, uint64_t, std::hash<uint32_t>,
                                 std::equal_to<uint32_t>, ObservedAllocator<Entry>>;
  AllocationLog log;
  {
    Map values{ObservedAllocator<Entry>(&log)};
    if (!Check(values.max_load_factor() == 1 && log.nodes + log.buckets == 0,
               "Empty hash-map contract changed")) return false;
    for (uint32_t i = 0; i < 4096; ++i) {
      ++values[i];
      ++values[i]; // Duplicate insertion must not allocate another node.
      const size_t n = values.size();
      if (!Check(log.nodes == n * Backing::kNodeBytes &&
                 log.buckets == values.bucket_count() * sizeof(void*) &&
                 log.buckets <= n * Backing::kRetainedBucketBytesPerEntry &&
                 log.peak <= n * (Backing::kNodeBytes + Backing::kPeakBucketBytesPerEntry),
                 "Observed hash node/bucket allocation exceeds its contract")) return false;
    }
  }
  return Check(log.nodes + log.buckets == 0, "Hash backing leaked");
}
}  // namespace

int main() {
  struct alignas(256) Aligned { int value = 0; };
  return ExactAllocatorResults<uint8_t>() && ExactAllocatorResults<double>() &&
         ExactAllocatorResults<Aligned>() &&
         ExactVectors<uint8_t>() && ExactVectors<double>() && ExactVectors<Aligned>() &&
         GrowingVectors() && StringPointers() && HashBacking() ? EXIT_SUCCESS : EXIT_FAILURE;
}
