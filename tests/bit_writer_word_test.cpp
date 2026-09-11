// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "codestream/bit_writer.h"

namespace {
using gjxl::BitWriter;
using gjxl::Status;

void Check(bool condition) {
  if (!condition) throw std::runtime_error("Bit-writer word oracle differs");
}
void Check(const Status& status) { Check(status.ok()); }

// Deliberately bit-at-a-time expected bytes, independent of word writes.
struct Oracle {
  size_t bits = 0;
  std::vector<uint8_t> bytes;
  void Write(size_t width, uint64_t value) {
    for (size_t i = 0; i < width; ++i) {
      if (bits % 8 == 0) bytes.push_back(0);
      bytes[bits / 8] |= static_cast<uint8_t>(((value >> i) & 1) << (bits % 8));
      ++bits;
    }
  }
  void Same(const BitWriter& writer) const {
    Check(writer.bits_written() == bits);
    Check(std::ranges::equal(writer.padded_bytes(), bytes));
  }
};

uint64_t Random(uint64_t& state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}
}  // namespace

int main() {
  try {
    uint64_t random = 0x73195CAFBAD;
    size_t cases = 0;
    for (size_t offset = 0; offset < 8; ++offset) {
      for (size_t width = 0; width <= 56; ++width) {
        for (size_t pattern = 0; pattern < 5; ++pattern) {
          BitWriter writer;
          Oracle oracle;
          const uint64_t initial = (uint64_t{1} << offset) - 1;
          Check(writer.WriteBits(offset, initial));
          oracle.Write(offset, initial);
          const uint64_t mask = (uint64_t{1} << width) - 1;
          const uint64_t value = pattern == 0 ? 0 : pattern == 1 ? mask :
            pattern == 2 ? mask & 0xAAAAAAAAAAAAAAAAull : Random(random) & mask;
          Check(writer.WriteBits(width, value));
          oracle.Write(width, value);
          oracle.Same(writer);

          // Dirty both the partial byte and padding, then roll back twice.
          const auto rejected = writer.WithMaxBits(112, [&]() -> Status {
            Check(writer.WriteBits(56, (uint64_t{1} << 56) - 1));
            const auto nested = writer.WithMaxBits(56, [&]() -> Status {
              Check(writer.WriteBits(56, (uint64_t{1} << 56) - 1));
              return Status::InvalidArgument("nested rollback");
            });
            Check(!nested.ok());
            return Status::InvalidArgument("outer rollback");
          });
          Check(!rejected.ok());
          oracle.Same(writer);
          Check(!writer.WriteBits(57, 0).ok());
          Check(!writer.WriteBits(width, uint64_t{1} << width).ok());
          oracle.Same(writer);
          for (size_t i = 0; i < 67; ++i) {
            const size_t next_width = Random(random) % 57;
            const uint64_t next = Random(random) & ((uint64_t{1} << next_width) - 1);
            Check(writer.WriteBits(next_width, next));
            oracle.Write(next_width, next);
            oracle.Same(writer);
          }
          ++cases;
        }
      }
    }
    std::cout << "Bit-writer word oracle PASS cases=" << cases
              << " offsets=8 widths=57 rollback=2\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
