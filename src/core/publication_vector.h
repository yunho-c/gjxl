// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/managed_allocator.h"

namespace gjxl::resource_budget_internal {

// Unlike ManagedVector, this owner can hand its backing to an existing public
// std::vector without copying. Capacity is established only by a fresh count
// or forward-range constructor, never by implicit vector growth. The current
// macOS build uses libc++ in C++20: __init_with_size -> __vallocate ->
// __allocate_at_least returns {allocator.allocate(n), n}. Review that backing
// contract before supporting another standard library/language mode; checking
// capacity only AFTER an unknown allocation would not enforce a hard bound.
#if !defined(_LIBCPP_VERSION) || _LIBCPP_STD_VER != 20
#error "PublicationVector requires the reviewed libc++ C++20 allocation contract"
#endif

template <typename T>
class PublicationVector {
public:
  PublicationVector() = default;
  PublicationVector(const PublicationVector&) = delete;
  PublicationVector& operator=(const PublicationVector&) = delete;
  PublicationVector(PublicationVector&&) noexcept = default;
  PublicationVector& operator=(PublicationVector&& other) noexcept {
    if (this != &other) {
      // Destroy old backing before its ticket, including move-assignment.
      PublicationVector replacement(std::move(other));
      swap(replacement);
    }
    return *this;
  }

  [[nodiscard]] static Status CopyFrom(
    std::span<const T> source, PublicationVector* out,
    ResourceClass owner = ResourceClass::kSerializer) {
    return Construct(source.size(), out, owner, [&] {
      return std::vector<T>(source.begin(), source.end());
    });
  }

  [[nodiscard]] static Status Create(
    size_t count, PublicationVector* out,
    ResourceClass owner = ResourceClass::kRetainedResult) {
    return Construct(count, out, owner, [&] { return std::vector<T>(count); });
  }

  [[nodiscard]] size_t size() const noexcept { return values_.size(); }
  [[nodiscard]] size_t capacity() const noexcept { return values_.capacity(); }
  [[nodiscard]] bool empty() const noexcept { return values_.empty(); }
  [[nodiscard]] const T* data() const noexcept { return values_.data(); }
  [[nodiscard]] std::span<const T> view() const noexcept { return values_; }
  [[nodiscard]] std::span<T> mutable_view() noexcept { return values_; }
  T& operator[](size_t i) noexcept { return values_[i]; }
  const T& operator[](size_t i) const noexcept { return values_[i]; }

  void swap(PublicationVector& other) noexcept {
    values_.swap(other.values_);
    std::swap(allocation_, other.allocation_);
  }
  void Reset() noexcept { PublicationVector{}.swap(*this); }

  /// Caller has finished all fallible work. Publishes the backing without a
  /// copy, then uncharges it. The previous caller-owned result is excluded.
  void PublishTo(std::vector<T>* output) noexcept {
    auto charge = MoveToPublication(output);
  }

  /// Multi-output publication keeps this charge until the containing result
  /// reaches the caller. All subsequent publication steps must be noexcept;
  /// if used with an unpublished candidate, destroy its backing before this
  /// returned charge on any rollback. Never discard the charge prematurely.
  [[nodiscard]] ResourceAllocation MoveToPublication(std::vector<T>* output) noexcept {
    static_assert(std::is_nothrow_move_assignable_v<std::vector<T>>);
    assert(output != nullptr);
    *output = std::move(values_);
    return std::move(allocation_);
  }

  [[nodiscard]] Status TransferTo(ResourceReservation& reservation) {
    return allocation_.valid() ? allocation_.TransferTo(reservation) :
      (empty() ? Status::Ok() : Status::FailedPrecondition("Result backing is not managed"));
  }

  [[nodiscard]] Status Reclassify(ResourceClass owner) {
    return allocation_.valid() ? allocation_.Reclassify(owner) :
      (empty() ? Status::Ok() : Status::FailedPrecondition("Result backing is not managed"));
  }

  friend bool operator==(const PublicationVector& a, const PublicationVector& b) {
    return a.values_ == b.values_;
  }

private:
  template <typename Constructor>
  static Status Construct(size_t count, PublicationVector* out,
                          ResourceClass owner, Constructor&& constructor) {
    if (out == nullptr) return Status::InvalidArgument("Publication vector output is null");
    if (count > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(T))
      return Status::InvalidArgument("Publication vector is too large");
    try {
      PublicationVector candidate;
      const auto context = CurrentResourceContext();
      if (count != 0 && (context.reservation != nullptr || context.track_host_allocations)) {
        const ResourceClassScope resource_class(owner);
        const size_t bytes = count * sizeof(T);
        Status status = PrepareResourceAllocation(bytes, bytes, &candidate.allocation_);
        if (!status.ok()) return status;
      }
      if (count != 0) ManagedHostAllocationCheckpointForTest();
      if (count != 0) candidate.values_ = constructor();
      if (candidate.values_.capacity() != count)
        return Status::Internal("Reviewed publication-vector capacity contract changed");
      if (candidate.allocation_.valid()) {
        Status status = candidate.allocation_.Commit();
        if (!status.ok()) return status;
      }
      *out = std::move(candidate);
      return Status::Ok();
    } catch (const ManagedAllocationFailure& failure) {
      return failure.status();
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("Unable to allocate publication vector backing");
    } catch (const std::length_error&) {
      return Status::InvalidArgument("Publication vector is too large");
    }
  }

  // Member order is intentional: free backing before destroying its charge.
  ResourceAllocation allocation_;
  std::vector<T> values_;
};

}  // namespace gjxl::resource_budget_internal
