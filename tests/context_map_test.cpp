// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

#include "codestream/context_map_internal.h"
#include "codestream/entropy_internal.h"
#include "codestream/entropy_storage_plan.h"

#if GJXL_CONTEXT_MAP_REFERENCE
#include "lib/jxl/dec_context_map.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/memory_manager_internal.h"
#endif

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;

bool Check(bool ok, const char* message) {
  if (!ok) std::cerr << message << '\n';
  return ok;
}
bool Ok(const Status& status) {
  if (!status.ok()) std::cerr << status.message() << '\n';
  return status.ok();
}
uint32_t Random(uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

bool Decode(std::span<const uint8_t> expected, const BitWriter& writer, size_t prefix) {
#if GJXL_CONTEXT_MAP_REFERENCE
  JxlMemoryManager memory;
  if (!jxl::MemoryManagerInit(&memory, nullptr)) return false;
  jxl::BitReader reader(writer.padded_bytes());
  reader.ReadBits(prefix);
  std::vector<uint8_t> decoded(expected.size());
  size_t histograms = 0;
  const bool success = !!jxl::DecodeContextMap(&memory, &decoded, &histograms, &reader);
  const size_t consumed = reader.TotalBitsConsumed();
  const bool closed = !!reader.Close();
  return Check(success && closed && std::ranges::equal(expected, decoded) &&
                 consumed == writer.bits_written() &&
                 histograms == 1u + *std::max_element(expected.begin(), expected.end()),
               "Pinned libjxl context-map decoding or bit consumption differs");
#else
  (void)expected; (void)writer; (void)prefix;
  return true;
#endif
}

// simple, raw Prefix, MTF Prefix, raw ANS, MTF ANS, RLE Prefix, RLE ANS
std::array<bool, 7> seen{};
bool Case(std::span<const uint8_t> map) {
  ContextMapEncoding encoding, repeat;
  BitWriter legacy;
  if (!Ok(EncodeContextMap(map, &encoding)) ||
      !Ok(EncodeContextMap(map, &repeat)) ||
      !Ok(WriteLegacyContextMap(map, &legacy)) ||
      !Check(encoding == repeat && encoding.Matches(map) &&
               encoding.bits_written() <= legacy.bits_written(),
             "Context-map search is not deterministic or grew the incumbent"))
    return false;
  const auto& bytes = encoding.bytes();
  if (bytes[0] & 1) {
    seen[0] = true;
  } else {
    const bool mtf = bytes[0] & 2;
    const bool rle = bytes[0] & 4;
    const size_t mode_bit = !rle ? 3 : ((bytes[0] >> 3) & 3) == 3 ? 31 : 16;
    const bool ans = !(bytes[mode_bit / 8] & (1u << (mode_bit % 8)));
    seen[rle ? (ans ? 6 : 5) : (ans ? 3 : 1) + mtf] = true;
  }
  for (size_t offset : {0ul, 1ul, 7ul}) {
    BitWriter writer;
    if (!Ok(writer.WriteBits(offset, 0)) || !Ok(encoding.AppendTo(&writer)) ||
        !Check(writer.bits_written() == offset + encoding.bits_written(), "Map append changed its length") ||
        !Decode(map, writer, offset)) return false;
  }
  ContextMapStoragePlan plan;
  if (!Ok(ComputeContextMapStoragePlan(map.size(), &plan))) return false;
  ResourceBudget budget;
  ResourceReservation job;
  if (!Ok(budget.Reserve(plan.working.peak_bytes, &job))) return false;
  {
    ResourceContextScope context({&job, ResourceClass::kSerializer});
    ContextMapEncoding managed;
    if (!Ok(EncodeContextMap(map, &managed)) || !Check(managed == encoding, "Managed map differs"))
      return false;
    const auto snapshot = budget.snapshot();
    const size_t bytes = managed.source().capacity() + managed.bytes().capacity();
    if (!Check(bytes == snapshot.total.live_capacity_bytes && bytes <= plan.owned.retained_bytes &&
                 snapshot.peak_backing_bytes <= plan.working.peak_bytes &&
                 managed.bits_written() <= plan.maximum_bits,
               "Context-map backing or bit bound is incomplete")) return false;
  }
  job.Reset();
  return Check(budget.snapshot().committed_bytes() == 0, "Context-map charge leaked");
}

bool CacheAndFailure() {
  EntropyCode code;
  code.context_count = 64;
  code.context_map.resize(64);
  for (size_t i = 0; i < 64; ++i) code.context_map[i] = (i / 4) % 4;
  code.uint_configs.resize(4);
  code.prefix_codes.resize(4);
  const std::array<uint64_t, 1> counts{1};
  for (auto& prefix : code.prefix_codes)
    if (!Ok(BuildPrefixCode(counts, &prefix))) return false;
  if (!Ok(PrepareEntropyContextMap(&code))) return false;
  const auto cache = code.context_map_encoding;
  ArmManagedHostAllocationFailureAfterForTest(0);
  const Status cached = PrepareEntropyContextMap(&code);
  const bool untouched = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  if (!Ok(cached) || !Check(untouched, "Prepared cache allocated again")) return false;
  code.context_map[0] = 3;
  BitWriter changed, fresh;
  ContextMapEncoding rebuilt;
  if (!Check(!code.context_map_encoding.Matches(code.context_map), "Stale cache matched") ||
      !Ok(WriteContextMap(code, &changed)) ||
      !Ok(EncodeContextMap(code.context_map, &rebuilt)) || !Ok(rebuilt.AppendTo(&fresh)) ||
      !Check(std::ranges::equal(changed.padded_bytes(), fresh.padded_bytes()) &&
               changed.bits_written() == fresh.bits_written() && code.context_map_encoding == cache,
             "Const writer used stale bytes or mutated its cache") ||
      !Decode(code.context_map, changed, 0)) return false;
  ContextMapEncoding sentinel = cache;
  if (!Check(EncodeContextMap({}, &sentinel).code() == StatusCode::kInvalidArgument && sentinel == cache,
             "Invalid map changed its output")) return false;
  ContextMapStoragePlan plan;
  if (!Ok(ComputeContextMapStoragePlan(code.context_map.size(), &plan))) return false;
  for (size_t fail = 0; fail < 16384; ++fail) {
    ResourceBudget budget;
    ResourceReservation job;
    if (!Ok(budget.Reserve(plan.working.peak_bytes, &job))) return false;
    auto output = sentinel;
    bool injected;
    {
      ResourceContextScope scope({&job, ResourceClass::kSerializer});
      ArmManagedHostAllocationFailureAfterForTest(fail);
      const auto status = EncodeContextMap(code.context_map, &output);
      injected = !ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (injected) {
        if (!Check(status.code() == StatusCode::kOutOfMemory && !status.resource_plan_exceeded() &&
                     output == sentinel, "Map allocation failure was not atomic")) return false;
      } else if (!Ok(status) || !Check(output == rebuilt, "Map recovery differs")) {
        return false;
      }
    }
    output = {};
    job.Reset();
    if (!Check(budget.snapshot().committed_bytes() == 0, "Failure sweep leaked a map")) return false;
    if (!injected) return true;
  }
  return Check(false, "Map failure sweep exhausted its limit");
}
}  // namespace

int main() {
  size_t cases = 0;
  for (size_t n : {1ul, 2ul, 3ul, 7ul, 31ul, 99ul, 100ul, 127ul, 256ul, 1024ul, 3465ul, 8192ul}) {
    for (size_t symbols : {1ul, 2ul, 4ul, 8ul, 16ul, 32ul, 64ul, 128ul, 256ul}) {
      if (symbols > n) continue;
      for (size_t pattern = 0; pattern < 5; ++pattern) {
        std::vector<uint8_t> map(n);
        uint32_t state = 0x1234567;
        for (size_t i = 0; i < n; ++i) {
          const uint32_t r = Random(state);
          const size_t value = pattern == 0 ? i % symbols
            : pattern == 1 ? std::min(symbols - 1, i * symbols / n)
            : pattern == 2 ? (r % 16 ? 0 : r % symbols)
            : pattern == 3 ? (i / (1 + (i / 257) % 31)) % symbols
                           : r % symbols;
          map[i] = static_cast<uint8_t>(value);
        }
        // Every histogram ID must occur, as required by the decoder.
        for (size_t i = 0; i < symbols; ++i) map[i] = static_cast<uint8_t>(i);
        if (!Case(map)) {
          std::cerr << "Map case " << n << '/' << symbols << '/' << pattern << '\n';
          return EXIT_FAILURE;
        }
        ++cases;
      }
    }
  }
  // RLE chunk boundaries, including a copy longer than the decoder window.
  std::vector<uint8_t> long_runs((size_t{1} << 21) + 17, 0);
  std::fill(long_runs.begin() + (size_t{1} << 20) + 5, long_runs.end(), 1);
  if (!Case(long_runs) || !CacheAndFailure()) return EXIT_FAILURE;
  if (!Check(std::ranges::all_of(seen, [](bool v) { return v; }),
             "Context-map representation coverage is incomplete")) {
    for (bool v : seen) std::cerr << v;
    std::cerr << '\n';
    return EXIT_FAILURE;
  }
  std::cout << cases + 1 << " context maps passed; reference decoder=" << GJXL_CONTEXT_MAP_REFERENCE << '\n';
  return EXIT_SUCCESS;
}
