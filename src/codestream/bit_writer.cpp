// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted for GJXL from libjxl-tiny's encoder bit writer.

#include "codestream/bit_writer.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>

namespace gjxl {
namespace {

bool TryAdd(size_t left, size_t right, size_t* result) {
  if (result == nullptr ||
      left > std::numeric_limits<size_t>::max() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

bool TryPaddedByteCount(size_t bits, size_t* bytes) {
  if (bytes == nullptr ||
      bits > std::numeric_limits<size_t>::max() - 7) {
    return false;
  }
  *bytes = (bits + 7) / 8;
  return true;
}

Status AllocationFailure() {
  return Status::OutOfMemory("Bit-writer allocation failed");
}

}  // namespace

Status BitWriter::PrepareWrite(size_t bit_count) {
  size_t end_bits = 0;
  if (!TryAdd(bits_written_, bit_count, &end_bits)) {
    return Status::OutOfMemory("Bit-writer size overflow");
  }
  if (active_allotment_ != nullptr && end_bits > active_allotment_->limit) {
    return Status::InvalidArgument("Bit-writer allotment exceeded");
  }

  size_t end_bytes = 0;
  if (!TryPaddedByteCount(end_bits, &end_bytes) ||
      end_bytes > storage_.max_size()) {
    return Status::OutOfMemory("Bit-writer byte size overflow");
  }
  try {
    storage_.resize(end_bytes, 0);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }
  return Status::Ok();
}

void BitWriter::WriteBitsUnchecked(
  size_t bit_count,
  uint64_t bits) noexcept {

  size_t consumed = 0;
  while (consumed < bit_count) {
    const size_t byte_index = bits_written_ / 8;
    const size_t bit_offset = bits_written_ % 8;
    const size_t chunk = std::min<size_t>(
      bit_count - consumed, 8 - bit_offset);
    const uint64_t mask = (uint64_t{1} << chunk) - 1;
    storage_[byte_index] |= static_cast<uint8_t>(
      ((bits >> consumed) & mask) << bit_offset);
    bits_written_ += chunk;
    consumed += chunk;
  }
}

Status BitWriter::WriteBits(size_t bit_count, uint64_t bits) {
  if (bit_count > kMaxBitsPerWrite) {
    return Status::InvalidArgument(
      "A bit-writer call cannot exceed 56 bits");
  }
  if ((bits >> bit_count) != 0) {
    return Status::InvalidArgument(
      "Bit-writer input has set bits outside the requested width");
  }
  if (Status status = PrepareWrite(bit_count); !status.ok()) {
    return status;
  }
  WriteBitsUnchecked(bit_count, bits);
  return Status::Ok();
}

Status BitWriter::ZeroPadToByte() {
  const size_t remainder = bits_written_ % 8;
  if (remainder == 0) {
    return Status::Ok();
  }
  return WriteBits(8 - remainder, 0);
}

Status BitWriter::Append(const BitWriter& other) {
  if (this == &other) {
    return Status::InvalidArgument("A bit writer cannot append itself");
  }
  if (Status status = PrepareWrite(other.bits_written_); !status.ok()) {
    return status;
  }

  const std::span<const uint8_t> bytes = other.padded_bytes();
  const size_t full_bytes = other.bits_written_ / 8;
  for (size_t index = 0; index < full_bytes; ++index) {
    WriteBitsUnchecked(8, bytes[index]);
  }
  const size_t trailing_bits = other.bits_written_ % 8;
  if (trailing_bits != 0) {
    const uint64_t mask = (uint64_t{1} << trailing_bits) - 1;
    WriteBitsUnchecked(trailing_bits, bytes[full_bytes] & mask);
  }
  return Status::Ok();
}

Status BitWriter::AppendByteAligned(std::span<const BitWriter> sections) {
  if (!byte_aligned()) {
    return Status::InvalidArgument(
      "Byte-aligned append requires an aligned destination");
  }

  size_t total_bytes = 0;
  for (const BitWriter& section : sections) {
    if (this == &section) {
      return Status::InvalidArgument(
        "A bit writer cannot concatenate itself");
    }
    if (!TryAdd(total_bytes, section.padded_size(), &total_bytes)) {
      return Status::OutOfMemory("Section byte count overflow");
    }
  }
  if (total_bytes > std::numeric_limits<size_t>::max() / 8) {
    return Status::OutOfMemory("Section bit count overflow");
  }
  const size_t total_bits = total_bytes * 8;
  if (Status status = PrepareWrite(total_bits); !status.ok()) {
    return status;
  }

  size_t destination = bits_written_ / 8;
  for (const BitWriter& section : sections) {
    const std::span<const uint8_t> bytes = section.padded_bytes();
    if (!bytes.empty()) {
      std::memcpy(storage_.data() + destination, bytes.data(), bytes.size());
      destination += bytes.size();
    }
  }
  bits_written_ += total_bits;
  return Status::Ok();
}

void BitWriter::Rollback(
  size_t previous_bits,
  size_t previous_size,
  uint8_t previous_last_byte) noexcept {

  storage_.resize(previous_size);
  if (previous_size != 0) {
    storage_.back() = previous_last_byte;
  }
  bits_written_ = previous_bits;
}

Status BitWriter::WithMaxBits(
  size_t maximum_bits,
  const std::function<Status()>& operation) {

  if (!operation) {
    return Status::InvalidArgument("Bit-writer operation is empty");
  }

  size_t limit = 0;
  if (!TryAdd(bits_written_, maximum_bits, &limit)) {
    return Status::OutOfMemory("Bit-writer allotment overflow");
  }
  if (active_allotment_ != nullptr && limit > active_allotment_->limit) {
    return Status::InvalidArgument(
      "Nested bit-writer allotment exceeds its parent");
  }

  size_t reserved_bytes = 0;
  if (!TryPaddedByteCount(limit, &reserved_bytes) ||
      reserved_bytes > storage_.max_size()) {
    return Status::OutOfMemory("Bit-writer allotment is too large");
  }
  try {
    storage_.reserve(reserved_bytes);
  } catch (const std::bad_alloc&) {
    return AllocationFailure();
  } catch (const std::length_error&) {
    return AllocationFailure();
  }

  const size_t previous_bits = bits_written_;
  const size_t previous_size = storage_.size();
  const uint8_t previous_last_byte =
    previous_size == 0 ? 0 : storage_.back();
  const AllotmentState state{limit, active_allotment_};
  active_allotment_ = &state;

  Status status;
  try {
    status = operation();
  } catch (const std::bad_alloc&) {
    status = AllocationFailure();
  } catch (const std::length_error&) {
    status = AllocationFailure();
  } catch (const std::exception&) {
    status = Status::Internal("Bit-writer operation threw an exception");
  } catch (...) {
    status = Status::Internal("Bit-writer operation failed unexpectedly");
  }
  active_allotment_ = state.parent;

  if (!status.ok()) {
    Rollback(previous_bits, previous_size, previous_last_byte);
  }
  return status;
}

}  // namespace gjxl
