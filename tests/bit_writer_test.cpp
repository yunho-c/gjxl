// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <utility>

#include "codestream/bit_writer.h"

namespace {

template <size_t Size>
bool HasBytes(
  const gjxl::BitWriter& writer,
  const std::array<uint8_t, Size>& expected) {

  return std::ranges::equal(writer.padded_bytes(), expected);
}

bool CheckCrossByteWriteAndAlignment() {
  gjxl::BitWriter writer;
  if (!writer.WriteBits(0, 0).ok() || writer.bits_written() != 0 ||
      !writer.WriteBits(3, 0b101).ok() ||
      !writer.WriteBits(8, 0xD6).ok() ||
      writer.bits_written() != 11 ||
      writer.padded_size() != 2 ||
      !HasBytes(writer, std::array<uint8_t, 2>{0xB5, 0x06}) ||
      !writer.ZeroPadToByte().ok() ||
      writer.bits_written() != 16 ||
      !writer.byte_aligned() ||
      !HasBytes(writer, std::array<uint8_t, 2>{0xB5, 0x06})) {
    std::cerr << "Cross-byte bit writing or alignment is incorrect\n";
    return false;
  }

  const uint64_t maximum_value = (uint64_t{1} << 56) - 1;
  gjxl::BitWriter maximum;
  if (!maximum.WriteBits(56, maximum_value).ok() ||
      maximum.bits_written() != 56 || maximum.padded_size() != 7) {
    std::cerr << "The maximum write width was rejected\n";
    return false;
  }
  return true;
}

bool CheckAppendModes() {
  gjxl::BitWriter first;
  gjxl::BitWriter second;
  if (!first.WriteBits(3, 5).ok() ||
      !second.WriteBits(5, 27).ok() ||
      !first.Append(second).ok() ||
      first.bits_written() != 8 ||
      !HasBytes(first, std::array<uint8_t, 1>{0xDD}) ||
      second.bits_written() != 5 ||
      !HasBytes(second, std::array<uint8_t, 1>{0x1B})) {
    std::cerr << "Unaligned append is incorrect\n";
    return false;
  }

  gjxl::BitWriter section0;
  gjxl::BitWriter section1;
  gjxl::BitWriter output;
  if (!section0.WriteBits(3, 5).ok() ||
      !section1.WriteBits(9, 0x101).ok() ||
      !output.WriteBits(8, 0xAA).ok()) {
    return false;
  }
  std::array<gjxl::BitWriter, 0> empty;
  if (!output.AppendByteAligned(empty).ok()) {
    return false;
  }

  // BitWriter is move-only, so use a vector as the contiguous section owner.
  std::array<gjxl::BitWriter, 2> sections;
  sections[0] = std::move(section0);
  sections[1] = std::move(section1);
  if (!output.AppendByteAligned(sections).ok() ||
      output.bits_written() != 32 ||
      !HasBytes(
        output, std::array<uint8_t, 4>{0xAA, 0x05, 0x01, 0x01}) ||
      sections[0].bits_written() != 3 ||
      sections[1].bits_written() != 9) {
    std::cerr << "Byte-aligned section concatenation is incorrect\n";
    return false;
  }
  return true;
}

bool CheckAllotmentsAndAtomicFailures() {
  gjxl::BitWriter writer;
  if (!writer.WriteBits(3, 5).ok()) {
    return false;
  }
  const gjxl::Status overrun = writer.WithMaxBits(5, [&]() -> gjxl::Status {
    if (gjxl::Status status = writer.WriteBits(4, 0xF); !status.ok()) {
      return status;
    }
    return writer.WriteBits(2, 3);
  });
  if (overrun.code() != gjxl::StatusCode::kInvalidArgument ||
      writer.bits_written() != 3 ||
      !HasBytes(writer, std::array<uint8_t, 1>{0x05})) {
    std::cerr << "Failed allotment did not roll back atomically\n";
    return false;
  }

  bool nested_rejected = false;
  const gjxl::Status nested = writer.WithMaxBits(8, [&]() -> gjxl::Status {
    if (gjxl::Status status = writer.WriteBits(2, 3); !status.ok()) {
      return status;
    }
    nested_rejected =
      writer.WithMaxBits(7, [] { return gjxl::Status::Ok(); }).code() ==
      gjxl::StatusCode::kInvalidArgument;
    return writer.WriteBits(6, 0x2A);
  });
  if (!nested.ok() || !nested_rejected || writer.bits_written() != 11 ||
      !HasBytes(writer, std::array<uint8_t, 2>{0x5D, 0x05})) {
    std::cerr << "Nested allotment bounds are incorrect\n";
    return false;
  }

  const size_t previous_bits = writer.bits_written();
  const std::array<uint8_t, 2> previous_bytes = {0x5D, 0x05};
  if (writer.WithMaxBits(
        std::numeric_limits<size_t>::max(),
        [] { return gjxl::Status::Ok(); }).code() !=
        gjxl::StatusCode::kOutOfMemory ||
      writer.WriteBits(57, 0).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      writer.WriteBits(3, 8).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      writer.WriteBits(0, 1).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      writer.Append(writer).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      writer.bits_written() != previous_bits ||
      !HasBytes(writer, previous_bytes)) {
    std::cerr << "Rejected bit-writer operations changed the output\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckCrossByteWriteAndAlignment() ||
      !CheckAppendModes() ||
      !CheckAllotmentsAndAtomicFailures()) {
    return EXIT_FAILURE;
  }
  std::cout << "All bit-writer tests passed.\n";
  return EXIT_SUCCESS;
}
