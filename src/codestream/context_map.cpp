// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Context-map representations and distance-one RLE follow libjxl's
// enc_context_map.cc, enc_lz77.cc and dec_ans.cc. No runtime dependency.

#include "codestream/context_map_internal.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <new>
#include <stdexcept>

#include "codestream/ans_internal.h"
#include "codestream/entropy_internal.h"
#include "codestream/entropy_storage_plan.h"

namespace gjxl::codestream_internal {
namespace {
constexpr HybridUintConfig kMapUint{2, 0, 1};
constexpr HybridUintConfig kLengthUint{0, 0, 0};
constexpr uint32_t kPrefixRepeatSymbol = 32;
constexpr uint32_t kAnsRepeatSymbol = 224;
constexpr size_t kMinimumRepeat = 3;
// Bound length symbols to 20 extra bits, including for unusually large maps.
constexpr size_t kMaximumRepeat = size_t{1} << 20;

Status Overflow() { return Status::OutOfMemory("Context-map storage overflows"); }

Status WriteMapHeader(bool mtf, bool rle, bool ans, BitWriter* writer) {
  // Complex map, MTF flag, then the nested entropy stream's LZ77 flag.
  if (Status s = writer->WriteBits(3, (mtf ? 2u : 0u) | (rle ? 4u : 0u));
      !s.ok()) return s;
  if (!rle) return Status::Ok();
  // LZ77 min_symbol: default 224 for ANS; BitsOffset(15, 8) for Prefix's
  // smaller alphabet. min_length=3 and length uint=(0,0,0) use defaults.
  if (Status s = writer->WriteBits(2, ans ? 0 : 3); !s.ok()) return s;
  if (!ans) {
    if (Status s = writer->WriteBits(15, kPrefixRepeatSymbol - 8); !s.ok())
      return s;
  }
  if (Status s = writer->WriteBits(2 + 4, 0); !s.ok()) return s;
  // Two entropy contexts: literals/lengths -> histogram 0, distance -> 1.
  // A singleton distance-zero histogram encodes distance one without bits.
  return writer->WriteBits(5, 1u | (1u << 1) | (1u << 4));
}

void MoveToFront(std::span<const uint8_t> map, Storage<uint8_t>* transformed) {
  std::array<uint8_t, 256> symbols{};
  for (size_t i = 0; i < symbols.size(); ++i) symbols[i] = static_cast<uint8_t>(i);
  transformed->resize(map.size());
  for (size_t i = 0; i < map.size(); ++i) {
    const uint8_t value = map[i];
    size_t position = 0;
    while (symbols[position] != value) ++position;
    (*transformed)[i] = static_cast<uint8_t>(position);
    std::move_backward(symbols.begin(), symbols.begin() + position,
                       symbols.begin() + position + 1);
    symbols[0] = value;
  }
}

bool TokenizeMap(std::span<const uint8_t> values, bool rle,
                 Storage<HybridUintToken>* tokens) {
  tokens->clear();
  tokens->reserve(values.size());
  std::array<uint64_t, kMaximumAnsAlphabetSize> counts{};
  std::array<HybridUintToken, 256> encoded{};
  for (size_t value = 0; value < encoded.size(); ++value)
    encoded[value] = EncodeHybridUintValidated(static_cast<uint32_t>(value), kMapUint);
  for (uint8_t value : values) ++counts[encoded[value].symbol];
  const size_t populated = std::count_if(counts.begin(), counts.end(),
                                        [](uint64_t n) { return n != 0; });
  std::array<double, 256> cost{};
  for (size_t value = 0; value < cost.size(); ++value) {
    const auto token = encoded[value];
    if (counts[token.symbol] == 0) continue;
    // RLE screening only. Final choices use complete emitted bits. Clamp to
    // the largest representable ANS frequency so very long, nearly constant
    // runs are not screened out by an unrealistically small Shannon cost.
    cost[value] = std::max(
      std::log2(static_cast<double>(values.size()) / counts[token.symbol]),
      std::log2(4096.0 / (4096 - (populated - 1)))) + token.extra_bit_count;
  }
  bool used_rle = false;
  for (size_t i = 0; i < values.size();) {
    const uint8_t value = values[i];
    tokens->push_back(encoded[value]);
    size_t end = ++i;
    while (end < values.size() && values[end] == value) ++end;
    while (i < end) {
      const size_t repeat = std::min(end - i, kMaximumRepeat);
      const size_t length = repeat >= kMinimumRepeat ? repeat - kMinimumRepeat : 0;
      if (rle && repeat >= kMinimumRepeat &&
          repeat * cost[value] > std::bit_width(length) + 1) {
        auto token = EncodeHybridUintValidated(static_cast<uint32_t>(length), kLengthUint);
        token.symbol += kPrefixRepeatSymbol;
        tokens->push_back(token);
        used_rle = true;
      } else {
        for (size_t j = 0; j < repeat; ++j) tokens->push_back(encoded[value]);
      }
      i += repeat;
    }
  }
  return used_rle;
}

Status WriteMapPrefix(std::span<const HybridUintToken> tokens, bool rle,
                       BitWriter* writer) {
  std::array<uint64_t, kPrefixAlphabetSize> counts{};
  for (const auto token : tokens) ++counts[token.symbol];
  std::array<PrefixCode, 2> codes{};
  if (Status s = BuildPrefixCode(counts, &codes[0]); !s.ok()) return s;
  const std::array configs{kMapUint, kLengthUint};
  if (rle) {
    counts.fill(0);
    counts[0] = 1;
    if (Status s = BuildPrefixCode(counts, &codes[1]); !s.ok()) return s;
  }
  const size_t clusters = rle ? 2 : 1;
  if (Status s = WritePrefixCodes(std::span(codes).first(clusters),
                                  std::span(configs).first(clusters), writer);
      !s.ok()) return s;
  const PrefixCode& code = codes[0];
  for (const auto token : tokens) {
    const size_t depth = code.degenerate_symbol == token.symbol ? 0 : code.depths[token.symbol];
    const uint64_t bits = code.bits[token.symbol] |
      (static_cast<uint64_t>(token.extra_bits) << depth);
    if (Status s = writer->WriteBits(depth + token.extra_bit_count, bits); !s.ok())
      return s;
  }
  return Status::Ok();
}
}  // namespace

bool ContextMapEncoding::Matches(std::span<const uint8_t> map) const noexcept {
  return bits_ != 0 && std::ranges::equal(source_, map);
}

Status ContextMapEncoding::AppendTo(BitWriter* writer) const {
  if (writer == nullptr || bits_ == 0)
    return Status::InvalidArgument("Context-map encoding is empty");
  const auto append = [&]() -> Status {
    for (size_t offset = 0; offset < bits_;) {
      const size_t count = std::min(BitWriter::kMaxBitsPerWrite, bits_ - offset);
      uint64_t bits = 0;
      for (size_t i = 0; i < (count + 7) / 8; ++i)
        bits |= uint64_t{bytes_[offset / 8 + i]} << (8 * i);
      bits &= (uint64_t{1} << count) - 1;
      if (Status s = writer->WriteBits(count, bits); !s.ok()) return s;
      offset += count;
    }
    return Status::Ok();
  };
  return writer->WithMaxBits(bits_, std::cref(append));
}

Status EncodeContextMap(std::span<const uint8_t> map, ContextMapEncoding* out) {
  if (out == nullptr || map.empty() || map.size() > UINT32_MAX)
    return Status::InvalidArgument("Context-map search input is invalid");
  try {
    BitWriter best;
    if (Status s = WriteLegacyContextMap(map, &best); !s.ok()) return s;
    const uint8_t maximum = *std::max_element(map.begin(), map.end());
    const auto consider = [&](BitWriter candidate) {
      if (candidate.bits_written() < best.bits_written()) best = std::move(candidate);
    };
    if (maximum != 0) {
      const size_t width = std::bit_width(maximum);
      if (width < 4) {
        BitWriter simple;
        if (Status s = simple.WriteBits(3, 1 | (width << 1)); !s.ok()) return s;
        for (uint8_t value : map)
          if (Status s = simple.WriteBits(width, value); !s.ok()) return s;
        consider(std::move(simple));
      }
      Storage<uint8_t> transformed;
      Storage<HybridUintToken> tokens;
      for (bool mtf : {false, true}) {
        if (mtf) MoveToFront(map, &transformed);
        const std::span<const uint8_t> values = mtf ? std::span<const uint8_t>(transformed) : map;
        for (bool rle : {false, true}) {
          if (!TokenizeMap(values, rle, &tokens) && rle) continue;
          BitWriter prefix;
          if (Status s = WriteMapHeader(mtf, rle, false, &prefix); !s.ok()) return s;
          if (Status s = WriteMapPrefix(tokens, rle, &prefix); !s.ok()) return s;
          consider(std::move(prefix));
          if (rle) {
            for (auto& token : tokens)
              if (token.symbol >= kPrefixRepeatSymbol)
                token.symbol += kAnsRepeatSymbol - kPrefixRepeatSymbol;
          }
          BitWriter ans;
          if (Status s = WriteMapHeader(mtf, rle, true, &ans); !s.ok()) return s;
          if (Status s = WriteContextMapAns(tokens, kMapUint, rle, &ans); !s.ok()) return s;
          consider(std::move(ans));
        }
      }
    }
    ContextMapEncoding result;
    result.source_.assign(map.begin(), map.end());
    const auto bytes = best.padded_bytes();
    result.bytes_.assign(bytes.begin(), bytes.end());
    result.bits_ = best.bits_written();
    *out = std::move(result);
    return Status::Ok();
  } catch (const resource_budget_internal::ManagedAllocationFailure& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return Overflow();
  } catch (const std::length_error&) {
    return Overflow();
  }
}

Status PrepareEntropyContextMap(EntropyCode* code) {
  if (code == nullptr || code->context_map.empty())
    return Status::InvalidArgument("Context-map preparation input is invalid");
  if (code->context_map_encoding.Matches(code->context_map)) return Status::Ok();
  if (*std::max_element(code->context_map.begin(), code->context_map.end()) == 0) {
    code->context_map_encoding = {};
    return Status::Ok();
  }
  return EncodeContextMap(code->context_map, &code->context_map_encoding);
}

Status ComputeContextMapStoragePlan(size_t entries, ContextMapStoragePlan* out) {
  using enum resource_budget_internal::VectorCapacityPolicy;
  if (out == nullptr || entries == 0 || entries > UINT32_MAX)
    return Status::InvalidArgument("Context-map storage input is invalid");
  ContextMapStoragePlan plan;
  // The selected map never exceeds the legacy Prefix incumbent: <=46 bits
  // per entry, plus its one bounded Prefix header. Candidate ANS alphabets
  // can cost more, but those bytes are scratch and are never retained.
  constexpr size_t prefix_header = 3 + 1 + 12 + 20 + 2 + 18 * 4 + 2 * 128 * 8;
  if (entries > (std::numeric_limits<size_t>::max() - prefix_header) / 46)
    return Overflow();
  plan.maximum_bits = prefix_header + entries * 46;
  if (plan.maximum_bits > std::numeric_limits<size_t>::max() - 7 ||
      !plan.owned.AddVector<uint8_t>(entries, kFreshExact) ||
      !plan.owned.AddVector<uint8_t>((plan.maximum_bits + 7) / 8, kFreshExact))
    return Overflow();
  auto& work = plan.working;
  HostStorageBound legacy, candidate, ans;
  if (entries > (std::numeric_limits<size_t>::max() - 10000) / 47)
    return Overflow();
  Status s = ComputeEntropyWriterStorageBound(plan.maximum_bits, &legacy);
  if (!s.ok()) return s;
  s = ComputeEntropyWriterStorageBound(10000 + 47 * entries, &candidate);
  if (!s.ok()) return s;
  s = ComputeContextMapAnsStorageBound(entries, &ans);
  if (!s.ok()) return s;
  // Legacy search: best/current Prefix writers. New search: incumbent and
  // candidate, plus WritePrefixCodes' atomic temporary and validation model.
  if (!work.Add(plan.owned) || !work.Add(legacy, 3) || !work.Add(candidate, 2) ||
      !work.Add(ans) || !work.AddVector<uint8_t>(entries, kFreshExact) ||
      !work.AddVector<HybridUintToken>(entries, kFreshExact) ||
      !work.AddVector<uint8_t>(2 * kPrefixAlphabetSize, kFreshExact, 2) ||
      !work.AddVector<uint8_t>(1, kFreshExact) ||
      !work.AddVector<HybridUintConfig>(2, kFreshExact) ||
      !work.AddVector<PrefixCode>(2, kFreshExact)) return Overflow();
  *out = plan;
  return Status::Ok();
}
}  // namespace gjxl::codestream_internal
