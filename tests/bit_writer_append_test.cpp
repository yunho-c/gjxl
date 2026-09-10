// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "codestream/bit_writer.h"

namespace {
using gjxl::BitWriter;
using gjxl::Status;
using gjxl::StatusCode;
using Bits = std::vector<uint8_t>;

void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void Check(const Status& status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

Bits Pattern(size_t length, unsigned pattern) {
  Bits bits(length);
  uint32_t random = 168;
  for (uint8_t& bit : bits) {
    random = random * 1664525u + 1013904223u;
    bit = pattern < 2 ? static_cast<uint8_t>(pattern)
                      : static_cast<uint8_t>(random >> 31);
  }
  return bits;
}

BitWriter Writer(std::span<const uint8_t> bits) {
  BitWriter writer;
  for (uint8_t bit : bits) Check(writer.WriteBits(1, bit));
  return writer;
}

// Expected bytes are packed directly from individual bits, not via Append
// or the production bit writer. Verify unused high bits as well as length.
void Same(const BitWriter& writer, std::span<const uint8_t> bits) {
  std::vector<uint8_t> expected((bits.size() + 7) / 8, 0);
  for (size_t i = 0; i < bits.size(); ++i)
    expected[i / 8] |= static_cast<uint8_t>(bits[i] << (i % 8));
  Check(writer.bits_written() == bits.size(), "Append logical length differs");
  Check(std::ranges::equal(writer.padded_bytes(), expected), "Append bytes/padding differ");
}

void Join(Bits* destination, std::span<const uint8_t> source) {
  destination->insert(destination->end(), source.begin(), source.end());
}

struct Coverage {
  size_t cases = 0;
  size_t insufficient = 0;
  size_t rollbacks = 0;
  std::array<bool, 8> offsets{};
  std::array<bool, 8> tails{};
};

void Compare(size_t prefix_size, size_t source_size, unsigned pattern,
             Coverage* coverage) {
  const Bits prefix = Pattern(prefix_size, 2);
  const Bits bits = Pattern(source_size, pattern);
  BitWriter source = Writer(bits);
  Same(source, bits);
  Bits expected = prefix;
  Join(&expected, bits);

  BitWriter ordinary = Writer(prefix);
  Check(ordinary.Append(source));
  Same(ordinary, expected);
  BitWriter exact = Writer(prefix);
  Check(exact.WithMaxBits(bits.size(), [&] { return exact.Append(source); }));
  Same(exact, expected);

  BitWriter limited = Writer(prefix);
  if (!bits.empty()) {
    const Status failure = limited.WithMaxBits(bits.size() - 1, [&] {
      return limited.Append(source);
    });
    Check(failure.code() == StatusCode::kInvalidArgument, "Short allotment accepted");
    Same(limited, prefix);
    ++coverage->insufficient;
  } else {
    Check(limited.WithMaxBits(0, [&] { return limited.Append(source); }));
    Same(limited, prefix);
  }

  BitWriter nested = Writer(prefix);
  Check(nested.WithMaxBits(bits.size() + 1, [&]() -> Status {
    if (Status status = nested.WithMaxBits(bits.size(), [&] {
          return nested.Append(source);
        }); !status.ok()) return status;
    return nested.WriteBits(1, 1);
  }));
  Bits with_suffix = expected;
  with_suffix.push_back(1);
  Same(nested, with_suffix);

  BitWriter rollback = Writer(prefix);
  const Status invalid = rollback.WithMaxBits(bits.size() + 1, [&]() -> Status {
    if (Status status = rollback.Append(source); !status.ok()) return status;
    return rollback.WriteBits(1, 2);
  });
  Check(invalid.code() == StatusCode::kInvalidArgument, "Late invalid write accepted");
  Same(rollback, prefix);
  const Status exception = rollback.WithMaxBits(bits.size(), [&]() -> Status {
    Check(rollback.Append(source));
    throw std::runtime_error("intentional append rollback");
  });
  Check(exception.code() == StatusCode::kInternal, "Exception was not converted");
  Same(rollback, prefix);
  coverage->rollbacks += 2;

  // Alternate append and bit writes across every resulting alignment, then
  // append a partial-byte source again. The source must remain unchanged.
  for (size_t i = 0; i < 9; ++i) {
    Check(ordinary.WriteBits(1, i % 2));
    expected.push_back(static_cast<uint8_t>(i % 2));
    Check(ordinary.Append(source));
    Join(&expected, bits);
  }
  Same(ordinary, expected);
  Same(source, bits);

  BitWriter self = Writer(prefix);
  Check(self.Append(self).code() == StatusCode::kInvalidArgument,
        "Self-append accepted");
  Same(self, prefix);
  ++coverage->cases;
  coverage->offsets[prefix_size % 8] = true;
  coverage->tails[source_size % 8] = true;
}
}  // namespace

int main() {
  try {
    Coverage coverage;
    std::vector<size_t> lengths;
    for (size_t length = 0; length <= 257; ++length) lengths.push_back(length);
    for (size_t length : {512u, 1023u, 1024u, 4097u, 65539u}) lengths.push_back(length);
    for (size_t prefix = 0; prefix < 24; ++prefix)
      for (size_t length : lengths)
        for (unsigned pattern = 0; pattern < 3; ++pattern)
          Compare(prefix, length, pattern, &coverage);
    Check(coverage.cases == 18936 && coverage.insufficient == 18864 &&
          coverage.rollbacks == 37872, "Incomplete append matrix");
    Check(std::ranges::all_of(coverage.offsets, [](bool v) { return v; }) &&
          std::ranges::all_of(coverage.tails, [](bool v) { return v; }),
          "Missing alignment/tail coverage");
    std::cout << "Bit-writer append PASS cases=" << coverage.cases
              << " insufficient=" << coverage.insufficient
              << " rollbacks=" << coverage.rollbacks << " offsets=8 tails=8\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
