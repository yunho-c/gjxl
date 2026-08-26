// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>

#include "codestream/sections.h"

namespace {

template <size_t Size>
bool HasBytes(
  const gjxl::BitWriter& writer,
  const std::array<uint8_t, Size>& expected) {

  return std::ranges::equal(writer.padded_bytes(), expected);
}

template <size_t Size>
bool CheckTocFixture(
  size_t section_size,
  const std::array<uint8_t, Size>& expected) {

  gjxl::BitWriter writer;
  const std::array<size_t, 1> sizes = {section_size};
  if (!gjxl::WriteTocSizes(sizes, &writer).ok() ||
      !writer.byte_aligned() || !HasBytes(writer, expected)) {
    std::cerr << "TOC fixture failed for size " << section_size << '\n';
    return false;
  }
  return true;
}

bool CheckSelectorBoundaries() {
  return
    CheckTocFixture(0, std::array<uint8_t, 3>{0x00, 0x00, 0x00}) &&
    CheckTocFixture(1023, std::array<uint8_t, 3>{0x00, 0xFC, 0x0F}) &&
    CheckTocFixture(1024, std::array<uint8_t, 3>{0x00, 0x01, 0x00}) &&
    CheckTocFixture(1025, std::array<uint8_t, 3>{0x00, 0x05, 0x00}) &&
    CheckTocFixture(17407, std::array<uint8_t, 3>{0x00, 0xFD, 0xFF}) &&
    CheckTocFixture(17408, std::array<uint8_t, 4>{0x00, 0x02, 0x00, 0x00}) &&
    CheckTocFixture(17409, std::array<uint8_t, 4>{0x00, 0x06, 0x00, 0x00}) &&
    CheckTocFixture(
      4211711, std::array<uint8_t, 4>{0x00, 0xFE, 0xFF, 0xFF}) &&
    CheckTocFixture(
      4211712,
      std::array<uint8_t, 5>{0x00, 0x03, 0x00, 0x00, 0x00}) &&
    CheckTocFixture(
      4211713,
      std::array<uint8_t, 5>{0x00, 0x07, 0x00, 0x00, 0x00}) &&
    CheckTocFixture(
      gjxl::kMaximumTocSectionSize,
      std::array<uint8_t, 5>{0x00, 0xFF, 0xFF, 0xFF, 0xFF});
}

bool CheckAlignmentAndConcatenation() {
  gjxl::BitWriter unaligned_toc;
  if (!unaligned_toc.WriteBits(3, 5).ok()) {
    return false;
  }
  const std::array<size_t, 1> zero_size = {0};
  if (!gjxl::WriteTocSizes(zero_size, &unaligned_toc).ok() ||
      !unaligned_toc.byte_aligned() ||
      !HasBytes(
        unaligned_toc, std::array<uint8_t, 3>{0x05, 0x00, 0x00})) {
    std::cerr << "TOC did not align relative to the existing bit offset\n";
    return false;
  }

  std::array<gjxl::BitWriter, 2> sections;
  gjxl::BitWriter output;
  if (!sections[0].WriteBits(3, 5).ok() ||
      !sections[1].WriteBits(9, 0x101).ok() ||
      !gjxl::ConcatenateSections(sections, &output).ok() ||
      output.bits_written() != 24 ||
      !HasBytes(output, std::array<uint8_t, 3>{0x05, 0x01, 0x01}) ||
      sections[0].bits_written() != 3 ||
      sections[1].bits_written() != 9) {
    std::cerr << "Section concatenation or source preservation failed\n";
    return false;
  }

  gjxl::BitWriter assembled;
  if (!gjxl::WriteTocAndSections(sections, &assembled).ok() ||
      !assembled.byte_aligned() || assembled.padded_size() != 7 ||
      !HasBytes(
        assembled,
        std::array<uint8_t, 7>{
          0x00, 0x04, 0x80, 0x00, 0x05, 0x01, 0x01})) {
    std::cerr << "Atomic TOC and section assembly is incorrect\n";
    return false;
  }
  return true;
}

bool CheckAtomicRejections() {
  gjxl::BitWriter writer;
  if (!writer.WriteBits(3, 5).ok()) {
    return false;
  }
  const std::array<size_t, 1> oversized = {
    gjxl::kMaximumTocSectionSize + 1};
  if (gjxl::WriteTocSizes(oversized, &writer).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      writer.bits_written() != 3 ||
      !HasBytes(writer, std::array<uint8_t, 1>{0x05})) {
    std::cerr << "Oversized TOC entry changed its destination\n";
    return false;
  }

  const std::span<const gjxl::BitWriter> self(&writer, 1);
  if (gjxl::WriteTocAndSections(self, &writer).code() !=
        gjxl::StatusCode::kInvalidArgument ||
      writer.bits_written() != 3 ||
      !HasBytes(writer, std::array<uint8_t, 1>{0x05})) {
    std::cerr << "Failed section assembly did not roll back the TOC\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckSelectorBoundaries() ||
      !CheckAlignmentAndConcatenation() ||
      !CheckAtomicRejections()) {
    return EXIT_FAILURE;
  }
  std::cout << "All codestream-section tests passed.\n";
  return EXIT_SUCCESS;
}
