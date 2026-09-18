// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/overwrite_array.h"
#include "core/status.h"

namespace gjxl {

/// Logical coefficient sequence backed by masks and ranked nonzero payloads.
/// The owner validates storage before publishing a view. This is not a
/// contiguous range and intentionally has no data() operation.
template <typename T> class SparseCoefficientSpan {
public:
  using value_type = T;

  SparseCoefficientSpan() = default;
  SparseCoefficientSpan(std::span<const uint64_t> masks,
                        std::span<const uint32_t> offsets,
                        std::span<const T> values, size_t begin, size_t length)
      : masks_(masks), offsets_(offsets), values_(values),
        begin_(begin), length_(length) {}

  [[nodiscard]] size_t size() const noexcept { return length_; }
  [[nodiscard]] bool empty() const noexcept { return length_ == 0; }

  [[nodiscard]] T operator[](size_t index) const noexcept {
    return Read(masks_.data(), offsets_.data(), values_.data(), begin_ + index);
  }

  [[nodiscard]] SparseCoefficientSpan subspan(size_t offset,
                                               size_t count) const {
    if (offset > length_ || count > length_ - offset) {
      throw std::out_of_range("Sparse coefficient subspan is out of bounds");
    }
    return {masks_, offsets_, values_, begin_ + offset, count};
  }
  [[nodiscard]] SparseCoefficientSpan subspan(size_t offset) const {
    if (offset > length_) {
      throw std::out_of_range("Sparse coefficient subspan is out of bounds");
    }
    return subspan(offset, length_ - offset);
  }

  [[nodiscard]] size_t NonzeroCount() const noexcept {
    size_t result = 0;
    for (size_t at = begin_; at < begin_ + length_;) {
      const unsigned bit = static_cast<unsigned>(at % 64);
      const size_t count = std::min<size_t>(64 - bit, begin_ + length_ - at);
      const uint64_t bits = count == 64 ? UINT64_MAX : (uint64_t{1} << count) - 1;
      result += std::popcount((masks_[at / 64] >> bit) & bits);
      at += count;
    }
    return result;
  }

  struct Iterator {
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using iterator_concept = std::forward_iterator_tag;
    using reference = T;
    using pointer = void;

    const uint64_t* masks = nullptr;
    const uint32_t* offsets = nullptr;
    const T* values = nullptr;
    size_t index = 0;
    T operator*() const noexcept { return Read(masks, offsets, values, index); }
    Iterator& operator++() noexcept { ++index; return *this; }
    Iterator operator++(int) noexcept { auto old = *this; ++*this; return old; }
    friend bool operator==(const Iterator&, const Iterator&) = default;
  };

  [[nodiscard]] Iterator begin() const noexcept {
    return {masks_.data(), offsets_.data(), values_.data(), begin_};
  }
  [[nodiscard]] Iterator end() const noexcept {
    return {masks_.data(), offsets_.data(), values_.data(), begin_ + length_};
  }

private:
  static T Read(const uint64_t* masks, const uint32_t* offsets,
                 const T* values, size_t index) noexcept {
    const size_t word = index / 64;
    const uint64_t mask = masks[word], flag = uint64_t{1} << (index % 64);
    return mask & flag
      ? values[offsets[word] + std::popcount(mask & (flag - 1))] : T{0};
  }

  std::span<const uint64_t> masks_;
  std::span<const uint32_t> offsets_;
  std::span<const T> values_;
  size_t begin_ = 0;
  size_t length_ = 0;
};

namespace vardct_frame_internal {

/// Mutable assembly input; a completed frame privately owns its moved storage.
/// Copies deep-copy all arrays. Payload intervals can be in any tile order,
/// but each nonzero is represented exactly once and no interval overlaps.
template <typename T> struct SparseAcStorage {
  using value_type = T;
  size_t coefficient_count = 0;
  OverwriteArray<uint64_t> masks;
  OverwriteArray<uint32_t> offsets;
  OverwriteArray<T> values;

  [[nodiscard]] bool ValidShape() const noexcept {
    return coefficient_count != 0 && coefficient_count <= UINT32_MAX &&
      masks.size() == coefficient_count / 64 + (coefficient_count % 64 != 0) &&
      offsets.size() == masks.size() && values.size() <= coefficient_count;
  }

  [[nodiscard]] Status Validate(bool reject_unwritten,
                                 int32_t unwritten) const {
    if (!ValidShape()) return Status::InvalidArgument("Sparse AC shape is invalid");
    if (coefficient_count % 64 &&
        (masks.data()[masks.size() - 1] >> (coefficient_count % 64))) {
      return Status::InvalidArgument("Sparse AC tail mask is invalid");
    }
    std::vector<uint64_t> covered(
        values.size() / 64 + (values.size() % 64 != 0), 0);
    size_t nonzeros = 0;
    for (size_t word = 0; word < masks.size(); ++word) {
      const size_t count = std::popcount(masks.data()[word]);
      size_t at = offsets.data()[word];
      if (at > values.size() || count > values.size() - at) {
        return Status::InvalidArgument("Sparse AC payload range is invalid");
      }
      nonzeros += count;
      for (size_t left = count; left != 0;) {
        const size_t n = std::min<size_t>(64 - at % 64, left);
        const uint64_t bits = (n == 64 ? UINT64_MAX : (uint64_t{1} << n) - 1)
                              << (at % 64);
        if (covered[at / 64] & bits) {
          return Status::InvalidArgument("Sparse AC payload ranges overlap");
        }
        covered[at / 64] |= bits;
        at += n;
        left -= n;
      }
    }
    if (nonzeros != values.size()) {
      return Status::InvalidArgument("Sparse AC payload coverage is incomplete");
    }
    for (const T value : values) {
      if (value == 0 || (reject_unwritten && static_cast<int32_t>(value) == unwritten)) {
        return Status::InvalidArgument("Sparse AC payload contains an invalid value");
      }
    }
    return Status::Ok();
  }

  [[nodiscard]] SparseCoefficientSpan<T> view() const noexcept {
    return {{masks.data(), masks.size()}, {offsets.data(), offsets.size()},
            {values.data(), values.size()}, 0, coefficient_count};
  }
  [[nodiscard]] size_t bytes() const noexcept {
    return masks.size() * sizeof(uint64_t) + offsets.size() * sizeof(uint32_t) +
      values.size() * sizeof(T);
  }
};

}  // namespace vardct_frame_internal
}  // namespace gjxl
