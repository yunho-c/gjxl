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
#if defined(_MSVC_STL_VERSION)
// VS 2022 17.7's STL: vector/list/xhash/xstring/yvals_core.h. Debug iterator
// proxies allocate additional backing and are outside this release contract.
#if !defined(_MSC_VER) || _MSC_VER != 1937 || _MSVC_STL_VERSION != 143 || _MSVC_STL_UPDATE != 202305L
#error "GJXL storage contract requires audited MSVC 19.37/STL 143 update 202305; see docs/storage-toolchain.md"
#endif
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL != 0
#error "GJXL storage contract requires MSVC iterator debug level 0; see docs/storage-toolchain.md"
#endif
// This compiler reports 202004 for its /std:c++latest (CMake C++23) mode.
#if !defined(_MSVC_LANG) || (_MSVC_LANG != 202002L && _MSVC_LANG != 202004L)
#error "GJXL storage contract requires C++20 or C++23; see docs/storage-toolchain.md"
#endif
#elif defined(_LIBCPP_VERSION)
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
#elif defined(__GLIBCXX__)
// Ubuntu 24.04 GCC 13.3.0-6ubuntu2~24.04.1, release headers and the C++11
// string ABI. Debug/parallel library modes are separate, unaudited adapters.
#if !defined(__GNUC__) || defined(__clang__) || __GNUC__ != 13 || __GNUC_MINOR__ != 3 || __GNUC_PATCHLEVEL__ != 0 || _GLIBCXX_RELEASE != 13 || __GLIBCXX__ != 20240904
#error "GJXL storage contract requires audited GCC 13.3.0/libstdc++ headers 20240904; see docs/storage-toolchain.md"
#endif
#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI != 1 || defined(_GLIBCXX_DEBUG) || defined(_GLIBCXX_PARALLEL)
#error "GJXL storage contract requires release libstdc++ with C++11 ABI; see docs/storage-toolchain.md"
#endif
#if __cplusplus != 202002L && __cplusplus != 202100L
#error "GJXL storage contract requires C++20 or C++23; see docs/storage-toolchain.md"
#endif
#else
#error "GJXL storage contract requires an audited standard library; see docs/storage-toolchain.md"
#endif

namespace gjxl::stdlib_storage_internal {

// libc++ __vector/vector.h: fresh count/forward-range construction and reserve
// allocate exactly the requested count. __memory/allocate_at_least.h returns
// {allocator.allocate(n), n} in C++20. In C++23 it uses allocator_traits:
// std::allocator::allocate_at_least still returns exactly {allocate(n), n},
// and ManagedAllocator has no allocate_at_least, so it takes the same fallback.
// These are audited allocator paths, not a promise for arbitrary allocators.
// Growth recommends max(2*capacity, size), saturating at max_size. MSVC's
// _Calculate_growth uses max(capacity + capacity/2, size), bounded by the same
// factors; its fresh constructors and reserve also allocate exact counts.
// libstdc++ 13's _M_check_len grows to size + max(size, added), capped at
// max_size; count/forward-range constructors and reserve allocate exact counts.
// Replacing backing keeps the old allocation alive in both language modes.
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
// MSVC char strings round with (_Requested | 15) and grow by 1.5x. Fresh slack
// 32 and the same growth/replacement envelopes also bound that implementation.
// libstdc++'s C++11 string _M_create doubles capacity when growing and allocates
// capacity + 1 characters. Its inline buffer is part of the string object.
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
#if defined(_MSVC_STL_VERSION)
  // xhash owns a list sentinel even when empty and two pointer-sized unchecked
  // iterators per bucket. It starts with 8 buckets, grows 8x below 512 buckets,
  // then rounds the required count to a power of two. At default load factor 1,
  // bucket count <= 8*entries; old+new <= 9*entries during replacement. Include
  // the empty backing separately so dense-only input is also fully charged.
  static constexpr size_t kNodeBytes =
    sizeof(std::_List_node<std::pair<const Key, Value>, void*>);
  static_assert(sizeof(decltype(std::declval<std::unordered_map<Key, Value>&>()
                                 ._Unchecked_begin())) == sizeof(void*));
  static constexpr size_t kEmptyNodeCount = 1;
  static constexpr size_t kEmptyBucketBytes = 16 * sizeof(void*);
  static constexpr size_t kPointersPerBucket = 2;
  static constexpr size_t kRetainedBucketBytesPerEntry = 16 * sizeof(void*);
  static constexpr size_t kPeakBucketBytesPerEntry = 18 * sizeof(void*);
#elif defined(__GLIBCXX__)
  static constexpr size_t kNodeBytes = sizeof(std::__detail::_Hash_node<
      std::pair<const Key, Value>, std::__cache_default<Key, std::hash<Key>>::value>);
  // Empty tables use an embedded single bucket. Default prime rehash policy
  // first allocates 13 buckets. Later growth requests at most 2U and adjacent
  // entries of the audited prime table are less than 2x apart. Thus 13U bounds
  // retained buckets and 14U bounds old + replacement, including the first grow.
  static constexpr size_t kEmptyNodeCount = 0;
  static constexpr size_t kEmptyBucketBytes = 0;
  static constexpr size_t kPointersPerBucket = 1;
  static constexpr size_t kRetainedBucketBytesPerEntry = 13 * sizeof(void*);
  static constexpr size_t kPeakBucketBytesPerEntry = 14 * sizeof(void*);
#else
  static constexpr size_t kNodeBytes =
    sizeof(std::__hash_node<std::__hash_value_type<Key, Value>, void*>);
  static constexpr size_t kEmptyNodeCount = 0;
  static constexpr size_t kEmptyBucketBytes = 0;
  static constexpr size_t kPointersPerBucket = 1;
  static constexpr size_t kRetainedBucketBytesPerEntry = 4 * sizeof(void*);
  static constexpr size_t kPeakBucketBytesPerEntry = 5 * sizeof(void*);
#endif
  static constexpr size_t kEmptyBytes =
    kEmptyNodeCount * kNodeBytes + kEmptyBucketBytes;
};

}  // namespace gjxl::stdlib_storage_internal
