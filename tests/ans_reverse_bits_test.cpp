// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "codestream/ans_reverse_bits_internal.h"

struct Chunk {
  uint32_t bits;
  unsigned width;
};
struct Coverage {
  std::array<bool, 32> widths{};
  std::array<bool, 56> pending{};
  std::array<bool, 8> tails{};
  size_t cases = 0, failures = 0, chunks = 0, words = 0, payload_bits = 0,
         exact = 0, split = 0, repeated = 0, late = 0, thrown = 0;
};
void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void Check(const gjxl::Status& s) {
  if (!s.ok()) throw std::runtime_error(std::string(s.message()));
}
uint32_t Mask(unsigned width) {
  return static_cast<uint32_t>((uint64_t{1} << width) - 1);
}
uint32_t Next(uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

void Compare(const std::vector<Chunk>& chunks, uint32_t state, unsigned prefix,
             Coverage& coverage) {
  gjxl::codestream_internal::AnsReverseBits packed;
  size_t payload = 0;
  for (const auto c : chunks) {
    Check(c.width <= 31 && (c.bits & ~Mask(c.width)) == 0,
          "Invalid fixture chunk");
    payload += c.width;
  }
  packed.ReserveBits(payload);
  for (const auto c : chunks) {
    coverage.widths[c.width] = true;
    coverage.pending[packed.pending_bits()] = true;
    const unsigned sum = packed.pending_bits() + c.width;
    coverage.exact += sum == 56;
    coverage.split += sum > 56;
    packed.PushValidated(c.bits, c.width);
    Check(packed.pending_bits() < 56 &&
              (packed.pending() >> packed.pending_bits()) == 0,
          "Pending invariant");
  }
  Check(packed.payload_bits() == payload &&
            packed.words().size() == payload / 56 &&
            packed.pending_bits() == payload % 56,
        "Packing length invariant");
  for (const auto word : packed.words())
    Check((word >> 56) == 0, "Packed word overflow");
  const size_t total = prefix + 32 + payload;
  std::vector<uint8_t> expected((total + 7) / 8, 0);
  size_t cursor = 0;
  // Independent bit-at-a-time oracle, not BitWriter or candidate packing.
  const auto put = [&](uint32_t bits, unsigned width) {
    for (unsigned bit = 0; bit < width; ++bit, ++cursor)
      expected[cursor / 8] |=
          static_cast<uint8_t>(((bits >> bit) & 1u) << (cursor % 8));
  };
  put(Mask(prefix), prefix);
  put(state, 32);
  for (auto c = chunks.rbegin(); c != chunks.rend(); ++c)
    put(c->bits, c->width);
  Check(cursor == total, "Oracle length");
  gjxl::BitWriter actual;
  Check(actual.WriteBits(prefix, Mask(prefix)));
  Check(actual.WithMaxBits(payload + 32,
                           [&] { return packed.Append(state, &actual); }));
  Check(actual.bits_written() == total &&
            std::ranges::equal(actual.padded_bytes(), expected),
        "Packed bytes differ");
  expected.resize((total + payload + 32 + 7) / 8, 0);
  put(state, 32);
  for (auto c = chunks.rbegin(); c != chunks.rend(); ++c)
    put(c->bits, c->width);
  Check(packed.Append(state, &actual));
  Check(actual.bits_written() == cursor &&
            std::ranges::equal(actual.padded_bytes(), expected),
        "Repeated direct append differs");
  ++coverage.repeated;
  gjxl::BitWriter limited, sentinel;
  Check(limited.WriteBits(prefix, Mask(prefix)));
  Check(sentinel.WriteBits(prefix, Mask(prefix)));
  const auto failure = limited.WithMaxBits(
      payload + 31, [&] { return packed.Append(state, &limited); });
  Check(failure.code() == gjxl::StatusCode::kInvalidArgument,
        "Short allotment accepted");
  Check(limited.bits_written() == sentinel.bits_written() &&
            std::ranges::equal(limited.padded_bytes(), sentinel.padded_bytes()),
        "Rollback differs");
  gjxl::BitWriter late, thrown;
  Check(late.WriteBits(prefix, Mask(prefix)));
  Check(thrown.WriteBits(prefix, Mask(prefix)));
  const auto late_status =
      late.WithMaxBits(payload + 32, [&]() -> gjxl::Status {
        if (auto s = packed.Append(state, &late); !s.ok()) return s;
        return late.WriteBits(1, 1);
      });
  Check(late_status.code() == gjxl::StatusCode::kInvalidArgument &&
            late.bits_written() == prefix &&
            std::ranges::equal(late.padded_bytes(), sentinel.padded_bytes()),
        "Post-append limit rollback differs");
  ++coverage.late;
  const auto thrown_status =
      thrown.WithMaxBits(payload + 32, [&]() -> gjxl::Status {
        Check(packed.Append(state, &thrown));
        throw std::runtime_error("Expected outer failure");
      });
  Check(thrown_status.code() == gjxl::StatusCode::kInternal &&
            thrown.bits_written() == prefix &&
            std::ranges::equal(thrown.padded_bytes(), sentinel.padded_bytes()),
        "Post-append exception rollback differs");
  ++coverage.thrown;
  Check(packed.Append(state, nullptr).code() ==
            gjxl::StatusCode::kInvalidArgument,
        "Null writer accepted");
  coverage.tails[total % 8] = true;
  ++coverage.cases;
  ++coverage.failures;
  coverage.chunks += chunks.size();
  coverage.words += packed.words().size();
  coverage.payload_bits += payload;
}

int main() {
  try {
    Coverage coverage;
    uint32_t random = 0x1832026u;
    // Every pending position and incoming width, six value patterns and all
    // destination alignments. The initial chunks construct each pending length.
    for (unsigned pending = 0; pending < 56; ++pending)
      for (unsigned width = 0; width <= 31; ++width)
        for (unsigned pattern = 0; pattern < 6; ++pattern)
          for (unsigned prefix = 0; prefix < 8; ++prefix) {
            std::vector<Chunk> chunks;
            unsigned left = pending;
            while (left) {
              const unsigned n = std::min(left, 31u);
              chunks.push_back({Next(random) & Mask(n), n});
              left -= n;
            }
            const uint32_t bits = pattern == 0   ? 0
                                  : pattern == 1 ? Mask(width)
                                  : pattern == 2 ? (0xAAAAAAAAu & Mask(width))
                                  : pattern == 3 ? (0x55555555u & Mask(width))
                                  : pattern == 4
                                      ? (width ? uint32_t{1} << (width - 1) : 0)
                                      : (Next(random) & Mask(width));
            chunks.push_back({bits, width});
            Compare(chunks, Next(random), prefix, coverage);
          }
    // Multiple flushes, arbitrary chunk boundaries, zero-width chunks, and all
    // short stream lengths. One generated stream is checked at every alignment.
    for (size_t length = 0; length <= 257; ++length)
      for (unsigned pattern = 0; pattern < 8; ++pattern) {
        std::vector<Chunk> chunks;
        for (size_t i = 0; i < length; ++i) {
          const unsigned width = pattern == 0   ? 0
                                 : pattern == 1 ? 1
                                 : pattern == 2 ? 16
                                 : pattern == 3 ? 31
                                                : Next(random) % 32;
          chunks.push_back({Next(random) & Mask(width), width});
        }
        for (unsigned prefix = 0; prefix < 8; ++prefix)
          Compare(chunks, Next(random), prefix, coverage);
      }
    // Includes 199,680 tokens with two chunks per token. Worst-case generic
    // chunks use 62 bits/token here, intentionally exceeding ANS's 47-bit
    // bound.
    for (size_t length : {size_t{4096}, size_t{65536}, size_t{399360}})
      for (unsigned pattern = 0; pattern < 4; ++pattern) {
        std::vector<Chunk> chunks;
        chunks.reserve(length);
        for (size_t i = 0; i < length; ++i) {
          const unsigned width = pattern == 0   ? 0
                                 : pattern == 1 ? 31
                                 : pattern == 2 ? (i % 2 ? 16u : 31u)
                                                : Next(random) % 32;
          chunks.push_back({Next(random) & Mask(width), width});
        }
        for (unsigned prefix : {0u, 7u})
          Compare(chunks, Next(random), prefix, coverage);
      }
    Check(std::ranges::all_of(coverage.widths, [](bool b) { return b; }) &&
              std::ranges::all_of(coverage.pending, [](bool b) { return b; }) &&
              std::ranges::all_of(coverage.tails, [](bool b) { return b; }),
          "Incomplete coverage");
    Check(coverage.exact > 0 && coverage.split > 0,
          "Boundary coverage missing");
    std::cout << "ANS reverse bits PASS cases=" << coverage.cases
              << " rollback=" << coverage.failures
              << " chunks=" << coverage.chunks << " words=" << coverage.words
              << " payload_bits=" << coverage.payload_bits
              << " exact=" << coverage.exact << " split=" << coverage.split
              << " repeated=" << coverage.repeated << " late=" << coverage.late
              << " thrown=" << coverage.thrown
              << " widths=32 pending=56 tails=8\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ANS reverse bits FAIL " << e.what() << '\n';
    return 1;
  }
}
