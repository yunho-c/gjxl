// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "core/managed_allocator.h"

namespace gjxl {

/// Fixed-size owning storage for trivial values, with explicit overwrite-only
/// allocation. Every element must be written before reading or copying it.
/// Copies are deep; moves transfer storage and leave the source empty.
template <typename T>
class OverwriteArray {
  static_assert(std::is_trivially_copyable_v<T> &&
                std::is_trivially_default_constructible_v<T> &&
                std::is_trivially_destructible_v<T>);

public:
  OverwriteArray() = default;

  OverwriteArray(const OverwriteArray& other) {
    ResetForOverwrite(other.size());
    if (size_ != 0) std::copy_n(other.data(), size_, data());
  }

  OverwriteArray& operator=(const OverwriteArray& other) {
    if (this != &other) {
      OverwriteArray candidate(other);
      *this = std::move(candidate);
    }
    return *this;
  }

  OverwriteArray(OverwriteArray&& other) noexcept
      : values_(std::move(other.values_)),
        size_(std::exchange(other.size_, 0)) {}

  OverwriteArray& operator=(OverwriteArray&& other) noexcept {
    if (this != &other) {
      values_ = std::move(other.values_);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  /// Replaces all storage, without initialization or preservation of values.
  /// Allocation failure leaves the original array unchanged.
  void ResetForOverwrite(size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::length_error("OverwriteArray size overflows");
    }
    Owner candidate;
    if (count != 0) {
      candidate.reset(resource_budget_internal::ManagedAllocator<T>{}.allocate(count));
      // Start object lifetimes without value initialization. T is trivial, so
      // this cannot throw and every producer must still overwrite all values.
      std::uninitialized_default_construct_n(candidate.get(), count);
    }
    values_ = std::move(candidate);
    size_ = count;
  }

  void assign(size_t count, T value) {
    ResetForOverwrite(count);
    if (count != 0) std::fill_n(data(), count, value);
  }

  [[nodiscard]] size_t size() const noexcept { return size_; }
  [[nodiscard]] Status ReclassifyResource(resource_budget_internal::ResourceClass owner) {
    return resource_budget_internal::ManagedAllocator<T>::ReclassifyBacking(data(), owner);
  }
  [[nodiscard]] T* data() noexcept { return values_.get(); }
  [[nodiscard]] const T* data() const noexcept { return values_.get(); }
  [[nodiscard]] T* begin() noexcept { return data(); }
  [[nodiscard]] const T* begin() const noexcept { return data(); }
  [[nodiscard]] T* end() noexcept { return size_ == 0 ? data() : data() + size_; }
  [[nodiscard]] const T* end() const noexcept {
    return size_ == 0 ? data() : data() + size_;
  }

private:
  struct Deleter {
    void operator()(T* pointer) const noexcept {
      resource_budget_internal::ManagedAllocator<T>{}.deallocate(pointer, 0);
    }
  };
  using Owner = std::unique_ptr<T[], Deleter>;
  Owner values_;
  size_t size_ = 0;
};

}  // namespace gjxl
