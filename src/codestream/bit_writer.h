// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's encoder bit writer.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

#include "core/managed_allocator.h"
#include "core/status.h"

namespace gjxl {

/// Writes bits to increasing byte addresses, least-significant bit first.
class BitWriter {
public:
  static constexpr size_t kMaxBitsPerWrite = 56;

  BitWriter() = default;
  BitWriter(const BitWriter&) = delete;
  BitWriter& operator=(const BitWriter&) = delete;
  BitWriter(BitWriter&&) noexcept = default;
  BitWriter& operator=(BitWriter&&) noexcept = default;

  [[nodiscard]] size_t bits_written() const noexcept {
    return bits_written_;
  }

  [[nodiscard]] size_t padded_size() const noexcept {
    return bits_written_ / 8 + (bits_written_ % 8 != 0 ? 1 : 0);
  }

  [[nodiscard]] bool byte_aligned() const noexcept {
    return bits_written_ % 8 == 0;
  }

  /// Includes the partially filled final byte, whose unused high bits are zero.
  [[nodiscard]] std::span<const uint8_t> padded_bytes() const noexcept {
    return {storage_.data(), padded_size()};
  }

  /// Writes at most 56 low bits from `bits`.
  [[nodiscard]] Status WriteBits(size_t bit_count, uint64_t bits);

  /// Advances to the next byte boundary using zero bits.
  [[nodiscard]] Status ZeroPadToByte();

  /// Appends an unaligned writer without adding padding between the streams.
  [[nodiscard]] Status Append(const BitWriter& other);

  /// Concatenates padded section bytes. The destination must be byte-aligned.
  [[nodiscard]] Status AppendByteAligned(
    std::span<const BitWriter> sections);

  /// Runs an atomic operation that may append no more than `maximum_bits`.
  ///
  /// Capacity is reserved before the callback. Failure, an exception, or a
  /// bound overrun restores the writer to its exact prior logical state.
  [[nodiscard]] Status WithMaxBits(
    size_t maximum_bits,
    const std::function<Status()>& operation);

private:
  struct AllotmentState {
    size_t limit = 0;
    const AllotmentState* parent = nullptr;
  };

  [[nodiscard]] Status PrepareWrite(size_t bit_count);
  void WriteBitsUnchecked(size_t bit_count, uint64_t bits) noexcept;
  void Rollback(
    size_t previous_bits,
    size_t previous_size,
    uint8_t previous_last_byte) noexcept;

  size_t bits_written_ = 0;
  resource_budget_internal::ManagedVector<
    uint8_t, resource_budget_internal::ResourceClass::kSerializer> storage_;
  const AllotmentState* active_allotment_ = nullptr;
};

}  // namespace gjxl
