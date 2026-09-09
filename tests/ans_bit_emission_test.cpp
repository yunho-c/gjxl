// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/ans_internal.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void Check(const Status& s) {
  if (!s.ok()) throw std::runtime_error(std::string(s.message()));
}
struct Chunk { uint32_t bits; uint8_t count; };
struct Coverage {
  std::array<bool, 32> widths{};
  std::array<bool, 8> tails{};
  size_t calls = 0, failures = 0, legacy_writes = 0, batched_writes = 0;
  bool exact56 = false;
};
EntropyCode Model(HybridUintConfig config, bool mapped) {
  AnsHistogram histogram;
  histogram.method = 0;
  for (uint32_t symbol = 0; symbol < 256; ++symbol) {
    histogram.frequencies.push_back(16);
    histogram.reciprocal_frequencies.push_back(
      codestream_internal::AnsFrequencyReciprocal(16));
    auto& reverse = histogram.reverse_maps.emplace_back();
    for (uint32_t i = 0; i < 16; ++i)
      reverse.push_back(static_cast<uint16_t>(symbol * 16 + i));
  }
  EntropyCode code;
  code.mode = EntropyCodingMode::kAns;
  code.context_count = 4;
  code.context_map = mapped ? std::vector<uint8_t>{0, 1, 0, 1}
                            : std::vector<uint8_t>{0, 1, 2, 3};
  code.ans_log_alpha_size = 8;
  code.uint_configs.assign(mapped ? 2 : 4, config);
  code.ans_histograms.assign(mapped ? 2 : 4, histogram);
  Check(codestream_internal::ValidateAnsEntropyCode(code));
  return code;
}
void Reference(std::span<const EntropyToken> tokens, const EntropyCode& code,
               BitWriter* writer, Coverage* coverage) {
  std::vector<Chunk> chunks;
  uint32_t state = 0x13u << 16;
  for (size_t i = tokens.size(); i; --i) {
    const auto token = tokens[i - 1];
    const auto cluster = code.context_map[token.context];
    HybridUintToken encoded;
    Check(EncodeHybridUint(token.value, code.uint_configs[cluster], &encoded));
    const auto& histogram = code.ans_histograms[cluster];
    const uint32_t frequency = histogram.frequencies[encoded.symbol];
    coverage->widths[encoded.extra_bit_count] = true;
    if (encoded.extra_bit_count) chunks.push_back({encoded.extra_bits, encoded.extra_bit_count});
    if (uint64_t{state} >= uint64_t{frequency} * (uint64_t{1} << 20)) {
      chunks.push_back({state & 65535u, 16});
      state /= 65536u;
    }
    state = (state / frequency) * 4096u + histogram.reverse_maps[encoded.symbol][state % frequency];
  }
  // Independent ordinary division recurrence and deliberately unbatched writes.
  Check(writer->WriteBits(32, state));
  size_t pending = 32;
  ++coverage->legacy_writes;
  for (auto i = chunks.rbegin(); i != chunks.rend(); ++i) {
    Check(writer->WriteBits(i->count, i->bits));
    ++coverage->legacy_writes;
    if (pending + i->count > 56) {
      ++coverage->batched_writes;
      pending = 0;
    }
    pending += i->count;
    coverage->exact56 |= pending == 56;
  }
  ++coverage->batched_writes;
}
void Same(const BitWriter& a, const BitWriter& b) {
  Check(a.bits_written() == b.bits_written() &&
        std::ranges::equal(a.padded_bytes(), b.padded_bytes()), "Writer bytes differ");
}
void Compare(std::span<const EntropyToken> tokens, const EntropyCode& code,
             size_t prefix, bool split, Coverage* coverage) {
  std::vector<uint32_t> values{0xDEADBEEFu};
  std::vector<uint16_t> contexts{0xBEEFu};
  for (auto t : tokens) { values.push_back(t.value); contexts.push_back(static_cast<uint16_t>(t.context)); }
  values.push_back(0xDEADBEEFu); contexts.push_back(0xBEEFu);
  const auto view = split ? EntropyTokenStreamView::Split(
    std::span(values).subspan(1, tokens.size()), std::span(contexts).subspan(1, tokens.size()))
    : EntropyTokenStreamView::Interleaved(tokens);
  BitWriter expected, actual, internal;
  const uint64_t initial = (uint64_t{1} << prefix) - 1;
  Check(expected.WriteBits(prefix, initial));
  Check(actual.WriteBits(prefix, initial));
  Check(internal.WriteBits(prefix, initial));
  Reference(tokens, code, &expected, coverage);
  uint64_t count = 0;
  Check(codestream_internal::CountAnsTokenStreamBits(view, code, &count));
  Check(count + prefix == expected.bits_written(), "Count-only bits differ");
  // Exact and insufficient allotments include every initial bit alignment.
  Check(actual.WithMaxBits(static_cast<size_t>(count), [&] { return WriteTokenStream(view, code, &actual); }));
  Check(codestream_internal::WriteAnsTokenStream(view, code, &internal));
  Same(actual, expected); Same(internal, expected);
  coverage->tails[count % 8] = true;
  BitWriter limited, sentinel;
  Check(limited.WriteBits(prefix, initial)); Check(sentinel.WriteBits(prefix, initial));
  const auto failure = limited.WithMaxBits(static_cast<size_t>(count - 1), [&] { return WriteTokenStream(view, code, &limited); });
  Check(failure.code() == StatusCode::kInvalidArgument, "Allotment was not rejected");
  Same(limited, sentinel);
  ++coverage->failures;
  ++coverage->calls;
}
}
int main() {
  try {
    Coverage coverage;
    size_t configs = 0;
    for (uint8_t split = 0; split <= 15; ++split)
      for (uint8_t msb = 0; msb <= split; ++msb)
        for (uint8_t lsb = 0; lsb <= split - msb; ++lsb) {
          const HybridUintConfig config{split, msb, lsb};
          ++configs;
          std::vector<EntropyToken> tokens;
          const auto add = [&](uint32_t value) {
            HybridUintToken encoded;
            Check(EncodeHybridUint(value, config, &encoded));
            if (encoded.symbol < kMaximumAnsAlphabetSize)
              tokens.push_back({static_cast<uint32_t>(tokens.size() % 4), value});
          };
          for (uint32_t value = 0; value < 32; ++value) add(value);
          for (uint32_t exponent = 0; exponent < 32; ++exponent) {
            add((uint32_t{1} << exponent) - 1);
            add(uint32_t{1} << exponent);
          }
          add(UINT32_MAX);
          for (bool mapped : {false, true}) {
            const auto code = Model(config, mapped);
            for (bool layout : {false, true})
              for (size_t prefix = 0; prefix < 8; ++prefix)
                Compare(tokens, code, prefix, layout, &coverage);
          }
        }
    const auto code = Model({0, 0, 0}, true);
    std::vector<EntropyToken> tokens;
    uint32_t random = 167;
    for (size_t i = 0; i < 257; ++i) {
      random = random * 1664525u + 1013904223u;
      tokens.push_back({static_cast<uint32_t>(i % 4), random});
    }
    for (size_t length = 0; length <= tokens.size(); ++length)
      for (size_t prefix = 0; prefix < 8; ++prefix)
        for (bool split : {false, true})
          Compare(std::span(tokens).first(length), code, prefix, split, &coverage);
    Check(configs == 816 && coverage.exact56, "Missing config/flush-boundary coverage");
    Check(std::ranges::all_of(coverage.widths, [](bool v) { return v; }), "Missing extra-bit width");
    Check(std::ranges::all_of(coverage.tails, [](bool v) { return v; }), "Missing output tail");
    std::cout << "ANS bit emission PASS configs=" << configs << " comparisons=" << coverage.calls
              << " atomic_failures=" << coverage.failures << " widths=32 tails=8 exact56=1"
              << " reference_chunk_writes=" << coverage.legacy_writes
              << " predicted_batch_writes=" << coverage.batched_writes << '\n';
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
