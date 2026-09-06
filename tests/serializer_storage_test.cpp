// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <span>
#include <thread>
#include <vector>

#include "codestream/ac_group.h"
#include "codestream/ans_internal.h"
#include "codestream/dc_group.h"
#include "codestream/entropy_internal.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;

bool Check(bool good, const char* message) {
  if (!good) std::cerr << message << '\n';
  return good;
}
bool Ok(const Status& status) {
  if (!status.ok()) std::cerr << status.message() << '\n';
  return status.ok();
}
bool Empty(const ResourceBudget& budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
    s.total.pending_count == 0 && s.open_reservations == 0 &&
    s.waiting_requests == 0, "Serializer storage charge leaked");
}

struct Backings {
  size_t bytes = 0;
  size_t count = 0;
  template <typename T> void Add(const Storage<T>& values) {
    bytes += values.capacity() * sizeof(T);
    count += values.capacity() != 0;
  }
  void Add(const EntropyCode& code, const EntropyCodeCost& cost) {
    Add(code.context_map);
    Add(code.uint_configs);
    Add(code.prefix_codes);
    Add(code.ans_histograms);
    for (const auto& histogram : code.ans_histograms) {
      Add(histogram.frequencies);
      Add(histogram.reverse_maps);
      Add(histogram.reciprocal_frequencies);
      for (const auto& reverse : histogram.reverse_maps) Add(reverse);
    }
    Add(cost.section_token_bits);
  }
  bool Matches(const ResourceBudget& budget) const {
    const auto s = budget.snapshot();
    const auto& serializer = s.classes[static_cast<size_t>(ResourceClass::kSerializer)];
    return Check(s.total.live_capacity_bytes == bytes &&
      s.total.backing_count == count && serializer.live_capacity_bytes == bytes &&
      serializer.backing_count == count && s.total.pending_count == 0,
      "Serializer capacities, nested backings, or owner labels differ");
  }
};

bool CheckNestedTokenOwnership() {
  ResourceBudget budget(1024 * 1024);
  ResourceReservation job;
  if (!Ok(budget.Reserve(1024 * 1024, &job))) return false;
  Storage<SimpleAcGroupTokenTemplate> retained;
  Backings expected;
  {
    // The fixed serializer owner must override the surrounding stage label.
    ResourceContextScope context({&job, ResourceClass::kPreparation});
    Storage<SimpleAcGroupTokenTemplate> groups(2);
    expected.Add(groups);
    for (auto& group : groups) {
      group.values.resize(32, 17);
      group.tokens.resize(32);
      group.block_context_keys.resize(8);
      expected.Add(group.values);
      expected.Add(group.tokens);
      expected.Add(group.block_context_keys);
    }
    const auto* original = groups.data();
    retained = std::move(groups);
    if (!Check(retained.data() == original, "Moving groups copied their backing") ||
        !expected.Matches(budget)) return false;
  }
  job.Reset();
  if (!expected.Matches(budget) ||
      !Check(budget.snapshot().committed_bytes() == expected.bytes,
             "Closed job lost retained token storage")) return false;
  // Destruction after producer scope and reservation, on another thread.
  std::thread consumer([owned = std::move(retained)] {
    if (owned[1].values[0] != 17) std::abort();
  });
  consumer.join();
  return Empty(budget);
}

bool CheckDcTokenBacking() {
  constexpr Extent2D extent{2, 2};
  const std::array<int32_t, 4> plane{1, 3, -4, 9};
  const ConstPlaneI32View view{plane.data(), extent, 2};
  const ConstImage3I32View dc{{view, view, view}};
  std::vector<EntropyToken> oracle;
  if (!Ok(TokenizeSimpleDcGroup(dc, &oracle))) return false;
  ResourceBudget budget(4096);
  ResourceReservation job;
  if (!Ok(budget.Reserve(4096, &job))) return false;
  {
    ResourceContextScope context({&job, ResourceClass::kPreparation});
    Storage<EntropyToken> tokens;
    if (!Ok(TokenizeSimpleDcGroup(dc, &tokens)) ||
        !Check(std::ranges::equal(tokens, oracle), "Managed DC tokens changed"))
      return false;
    Backings expected;
    expected.Add(tokens);
    if (!expected.Matches(budget)) return false;
  }
  job.Reset();
  return Empty(budget);
}

// Exhaust every backing-allocation position of the operation, ending only at
// the first successful run with an unconsumed failure hook. Inputs/oracles are
// caller-owned. Each attempted job has its own ledger, including rollback.
template <typename Output, typename Function>
bool SweepFailures(const Output& sentinel, const Output& oracle,
                   Function&& operation, size_t* allocations = nullptr) {
  for (size_t fail_at = 0; fail_at < 4096; ++fail_at) {
    ResourceBudget budget(16 * 1024 * 1024);
    ResourceReservation job;
    if (!Ok(budget.Reserve(16 * 1024 * 1024, &job))) return false;
    bool injected = false;
    {
      Output output = sentinel;
      ResourceContextScope context({&job, ResourceClass::kSerializer});
      ArmManagedHostAllocationFailureAfterForTest(fail_at);
      const Status status = operation(&output);
      injected = !ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (injected) {
        if (!Check(status.code() == StatusCode::kOutOfMemory && output == sentinel,
                   "Injected serializer failure was not atomic") ||
            !Check(budget.snapshot().total.backing_count == 0 &&
                   budget.snapshot().total.pending_count == 0,
                   "Failed serializer operation retained candidate backing")) {
          std::cerr << "Failure position " << fail_at << '\n';
          return false;
        }
      } else if (!Ok(status) || !Check(output == oracle, "Recovery differs from oracle")) {
        return false;
      }
    }
    job.Reset();
    if (!Empty(budget)) return false;
    if (!injected) {
      if (allocations != nullptr) *allocations = fail_at;
      return Check(fail_at > 0, "Failure sweep covered no allocations");
    }
  }
  return Check(false, "Failure sweep did not reach successful completion");
}

bool CheckSparseAggregationFailures() {
  constexpr std::array<uint32_t, 6> symbols{0, 7, 65535, 65536, 999999, UINT32_MAX};
  std::vector<uint32_t> values(4096);
  for (size_t i = 0; i < values.size(); ++i) values[i] = symbols[i % symbols.size()];
  Storage<WeightedValue> oracle;
  if (!Ok(AggregateEntropyValues(std::span<uint32_t>(values), &oracle))) return false;
  const Storage<WeightedValue> sentinel{{42, 11}};
  size_t allocations = 0;
  if (!SweepFailures(sentinel, oracle, [&](auto* output) {
        return AggregateEntropyValues(std::span<uint32_t>(values), output);
      }, &allocations) ||
      !Check(allocations >= 6, "Sparse hash nodes/buckets escaped the failure sweep"))
    return false;
  // Enough for the dense table, but not a single sparse node: no fallback to
  // the unlimited default domain is permitted after this underplanned job.
  constexpr size_t dense_bytes = 65536 * sizeof(uint64_t);
  ResourceBudget budget(dense_bytes);
  ResourceReservation job;
  if (!Ok(budget.Reserve(dense_bytes, &job))) return false;
  Storage<WeightedValue> output = sentinel;
  {
    ResourceContextScope context({&job, ResourceClass::kSerializer});
    const auto status = AggregateEntropyValues(std::span<uint32_t>(values), &output);
    if (!Check(status.code() == StatusCode::kOutOfMemory && output == sentinel &&
        budget.snapshot().peak_backing_bytes == dense_bytes &&
        budget.snapshot().total.backing_count == 0,
        "Sparse node escaped an exhausted reservation")) return false;
  }
  job.Reset();
  return Empty(budget);
}

struct ModelResult {
  EntropyCode code;
  EntropyCodeCost cost;
  friend bool operator==(const ModelResult&, const ModelResult&) = default;
};

bool CheckEntropyModelFailures() {
  std::vector<EntropyToken> tokens;
  for (uint32_t i = 0; i < 96; ++i) tokens.push_back({i % 2, (i / 2) % 7});
  const std::array sections{EntropyTokenStreamView::Interleaved(tokens)};
  EntropyCode prefix_partition;
  if (!Ok(OptimizeEntropyCode(sections, {.context_count = 2}, &prefix_partition)))
    return false;
  for (size_t mode = 0; mode < 4; ++mode) {
    const auto optimize = [&](ModelResult* result) {
      if (mode == 0) return OptimizeFastPrefixEntropyCode(
        sections, {.context_count = 2}, &result->code, &result->cost);
      if (mode == 3) return OptimizeAnsEntropyCode(
        sections, prefix_partition, &result->code, &result->cost);
      return OptimizeDirectAnsEntropyCode(sections, {.context_count = 2},
        mode == 1 ? DirectAnsEntropyMode::kBalanced : DirectAnsEntropyMode::kHighDensity,
        &result->code, &result->cost);
    };
    ModelResult oracle;
    if (!Ok(optimize(&oracle))) return false;
    ModelResult sentinel;
    sentinel.code.context_count = 77;
    sentinel.cost.model_bits = 19;
    size_t allocations = 0;
    if (!SweepFailures(sentinel, oracle, optimize, &allocations)) return false;
    std::cout << "Model " << mode << ": " << allocations << " allocation failures checked\n";
    ResourceBudget budget(16 * 1024 * 1024);
    ResourceReservation job;
    if (!Ok(budget.Reserve(16 * 1024 * 1024, &job))) return false;
    {
      ResourceContextScope context({&job, ResourceClass::kPreparation});
      ModelResult result;
      if (!Ok(optimize(&result))) return false;
      Backings expected;
      expected.Add(result.code, result.cost);
      if (!expected.Matches(budget) || !Check(result == oracle,
          "Model allocation domain changed coding decisions")) return false;
    }
    job.Reset();
    if (!Empty(budget)) return false;
  }
  return true;
}

template <typename T> struct FailingAllocator : std::allocator<T> {
  using value_type = T;
  template <typename U> struct rebind { using other = FailingAllocator<U>; };
  static inline bool fail = false;
  T* allocate(size_t count) {
    if (fail) throw std::bad_alloc();
    return std::allocator<T>::allocate(count);
  }
};

bool CheckLegacyPublicationAtomicity() {
  AcStrategyGrid strategies;
  if (!Ok(AcStrategyGrid::Create({1, 1}, &strategies))) return false;
  strategies.fill_dct8();
  const int8_t correlation = 0;
  const int32_t quant = 3;
  const uint8_t sharpness = 4;
  const SimpleAcMetadataInput input{
    .y_to_x_map = {&correlation, {1, 1}, 1},
    .y_to_b_map = {&correlation, {1, 1}, 1},
    .strategies = &strategies,
    .raw_quant_field = {&quant, {1, 1}, 1},
    .epf_sharpness = {&sharpness, {1, 1}, 1},
  };
  using Allocator = FailingAllocator<EntropyToken>;
  std::vector<EntropyToken, Allocator> output{{99, 42}};
  size_t anchor_count = 77;
  Allocator::fail = true;
  const Status status = TokenizeSimpleAcMetadata(input, &output, &anchor_count);
  Allocator::fail = false;
  if (!Check(status.code() == StatusCode::kOutOfMemory && anchor_count == 77 &&
      output.size() == 1 && output[0] == EntropyToken{99, 42},
      "Failed legacy publication changed tokens or auxiliary output")) return false;
  if (!Ok(TokenizeSimpleAcMetadata(input, &output, &anchor_count)) ||
      !Check(anchor_count == 1, "Legacy publication did not recover")) return false;
  return Check(TokenizeSimpleAcMetadata(input, nullptr, &anchor_count).code() ==
    StatusCode::kInvalidArgument && TokenizeSimpleAcMetadata(input, &output, nullptr).code() ==
    StatusCode::kInvalidArgument, "Null compatibility outputs were accepted");
}
}  // namespace

int main() {
  if (!Empty(DefaultResourceBudget()) || !CheckNestedTokenOwnership() ||
      !CheckDcTokenBacking() || !CheckSparseAggregationFailures() ||
      !CheckEntropyModelFailures() || !CheckLegacyPublicationAtomicity() ||
      !Empty(DefaultResourceBudget()) ||
      !Check(DefaultResourceBudget().snapshot().peak_backing_bytes == 0,
             "Explicit serializer allocation escaped to the default domain")) return EXIT_FAILURE;
  std::cout << "All serializer storage tests passed.\n";
  return EXIT_SUCCESS;
}
