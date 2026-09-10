// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

// The single compatibility boundary for implementation-defined backing sizes
// and layouts used by managed storage. Keep the source audit, configuration
// probe, and allocation tests in sync; see docs/storage-toolchain.md.
// Checking capacity after allocation cannot establish a hard allocation bound.
#if !defined(__APPLE__) || !defined(__apple_build_version__) || __apple_build_version__ != 17000604
#error "GJXL storage contract requires audited Apple Clang build 17000604; see docs/storage-toolchain.md"
#endif
#if !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION != 200100
#error "GJXL storage contract requires audited libc++ 200100; see docs/storage-toolchain.md"
#endif
#if !defined(_LIBCPP_STD_VER) || (_LIBCPP_STD_VER != 20 && _LIBCPP_STD_VER != 23)
#error "GJXL storage contract requires C++20 or C++23; see docs/storage-toolchain.md"
#endif
#if !defined(_LIBCPP_ABI_VERSION) || _LIBCPP_ABI_VERSION != 1 || defined(_LIBCPP_ABI_UNSTABLE)
#error "GJXL storage contract requires stable libc++ ABI version 1; see docs/storage-toolchain.md"
#endif

namespace gjxl::stdlib_storage_internal {

// libc++ __vector/vector.h: fresh count/forward-range construction and reserve
// allocate exactly the requested count. __memory/allocate_at_least.h returns
// {allocator.allocate(n), n} in C++20. In C++23 it uses allocator_traits:
// std::allocator::allocate_at_least still returns exactly {allocate(n), n},
// and ManagedAllocator has no allocate_at_least, so it takes the same fallback.
// These are audited allocator paths, not a promise for arbitrary allocators.
// Growth recommends max(2*capacity, size), saturating at max_size. Replacing a
// backing keeps the old allocation alive in both language modes.
inline constexpr size_t kVectorGrowthFactor = 2;
inline constexpr size_t kVectorReplacementPeakFactor = 2;
inline constexpr size_t kVectorGrowingRetainedFactor = kVectorGrowthFactor;
inline constexpr size_t kVectorGrowingPeakFactor = kVectorGrowthFactor + 1;

template <typename T>
std::vector<T> MakeExactVector(size_t count) {
  return std::vector<T>(count);
}
template <typename T>
std::vector<T> MakeExactVector(std::span<const T> source) {
  return std::vector<T>(source.begin(), source.end());
}

// libc++ string: char-string __recommend rounds length+1 to an 8-byte boundary,
// with an inline/long boundary adjustment. 32 bytes covers both. Growth doubles
// the old capacity then rounds; max_size saturation also fits this envelope.
// shrink_to_fit can retain the old doubled backing beside a fresh replacement.
inline constexpr size_t kStringFreshSlack = 32;
inline constexpr size_t kStringGrowingRetainedFactor = 2;
inline constexpr size_t kStringGrowingPeakFactor = 3;

// __get_short_pointer points inside the string object; __get_long_pointer is
// the allocator's original pointer outside it. Never read an allocation ticket
// before inline characters. This works with the audited alternate string layout.
template <typename Traits, typename Allocator>
char* StringHeapData(std::basic_string<char, Traits, Allocator>& value) noexcept {
  const auto object = reinterpret_cast<uintptr_t>(&value);
  const auto data = reinterpret_cast<uintptr_t>(value.data());
  return data < object || data - object >= sizeof(value) ? value.data() : nullptr;
}

// libc++ __hash_table: empty unordered_map, raw allocator pointers, unique-key
// operator[] insertion, default max_load_factor==1, no reserve/rehash/erase.
// Rehash requests <=2U buckets for U entries; next-prime rounding is <2x
// (the first allocation is 2). Current buckets <=4U; old+replacement <=5U.
// Node size is the concrete rebound allocation, excluding allocator overhead.
template <typename Key, typename Value>
struct UnorderedMapBacking {
  static constexpr size_t kNodeBytes =
    sizeof(std::__hash_node<std::__hash_value_type<Key, Value>, void*>);
  static constexpr size_t kRetainedBucketBytesPerEntry = 4 * sizeof(void*);
  static constexpr size_t kPeakBucketBytesPerEntry = 5 * sizeof(void*);
};

}  // namespace gjxl::stdlib_storage_internal
