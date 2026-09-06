// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#ifndef GJXL_REPRESENTATION_ORACLE_ONLY
#include "codestream/representation_storage_plan.h"
#else
#include "codestream/encoder.h"
#endif

#include <cstdlib>
#include <string_view>
#include <thread>

#include "codestream/ac_group.h"
#include "codestream/block_context_map.h"
#include "codestream/coefficient_order.h"
#include "codestream_frame_fixture.h"
#include "core/thread_budget.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;
using codestream_test_internal::FixtureCheck;
using codestream_test_internal::FixtureOk;
using codestream_test_internal::FrameFixture;
using enum VarDctCoefficientOrderBehavior;

constexpr std::array<Extent2D, 8> kExtents{{{1, 1},
                                            {4, 4},
                                            {1, 33},
                                            {33, 1},
                                            {32, 32},
                                            {33, 33},
                                            {128, 64},
                                            {129, 65}}};

bool Empty(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  return FixtureCheck(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
                          s.total.pending_count == 0 &&
                          s.open_reservations == 0,
                      "Representation storage leaked a charge");
}

struct Backings {
  size_t bytes = 0, count = 0;
  template <typename T> void Add(const Storage<T> &v) {
    bytes += v.capacity() * sizeof(T);
    count += v.capacity() != 0;
  }
  void Add(const SimpleCoefficientOrders &orders) {
    for (const auto &family : orders.orders)
      for (const auto &channel : family)
        Add(channel);
  }
  void Add(const SimpleBlockContextMap &map) {
    Add(map.context_map);
    Add(map.qf_thresholds);
  }
  bool Matches(const ResourceBudget &budget, size_t retained) const {
    const auto s = budget.snapshot();
    const auto &serial =
        s.classes[static_cast<size_t>(ResourceClass::kSerializer)];
    return FixtureCheck(
        bytes <= retained && bytes == s.total.live_capacity_bytes &&
            bytes == serial.live_capacity_bytes &&
            count == s.total.backing_count && count == serial.backing_count &&
            s.total.pending_count == 0,
        "Representation retained backing differs from ledger");
  }
};

// Binary oracle mode can be linked against the frozen parent without planner
// symbols. Emit values explicitly, avoiding struct padding, native endian, and
// hash collisions in the comparison. Include capacities (even cleared vectors).
void Emit(size_t value) {
  for (size_t i = 0; i < 8; ++i)
    std::cout.put(
        static_cast<char>((static_cast<uint64_t>(value) >> (8 * i)) & 255));
}
template <typename T> void Emit(const Storage<T> &v) {
  Emit(v.size());
  Emit(v.capacity());
  for (const auto value : v)
    Emit(static_cast<size_t>(value));
}
void Emit(const SimpleCoefficientOrders &orders) {
  Emit(orders.used_order_mask);
  for (const auto &family : orders.orders)
    for (const auto &channel : family)
      Emit(channel);
}
void Emit(const SimpleBlockContextMap &map) {
  Emit(map.num_contexts);
  Emit(map.qf_thresholds);
  Emit(map.context_map);
}
bool EmitOracle() {
  thread_budget_internal::EncodeScope threads(1);
  for (Extent2D extent : kExtents) {
    for (size_t family = 0; family < 8; ++family) {
      FrameFixture f;
      if (!f.Create(extent, family, family % 3))
        return false;
      for (auto behavior : {kFull, kEffort7Dct8Sampled}) {
        SimpleCoefficientOrders orders;
        Storage<EntropyToken> tokens;
        if (!FixtureOk(ComputeSimpleCoefficientOrdersForEncoder(
                f.view(), behavior, &orders)) ||
            !FixtureOk(TokenizeSimpleCoefficientOrders(orders, &tokens)))
          return false;
        Emit(orders);
        Emit(tokens.size());
        Emit(tokens.capacity());
        for (const auto token : tokens) {
          Emit(token.context);
          Emit(token.value);
        }
      }
      SimpleBlockContextMap map;
      Storage<SimpleBlockContextMap> maps;
      if (!FixtureOk(ComputeSimpleBlockContextMapForEncoder(f.view(), &map)) ||
          !FixtureOk(ComputeSimpleBlockContextMapCandidatesForEncoder(f.view(),
                                                                      &maps)))
        return false;
      Emit(map);
      Emit(maps.size());
      Emit(maps.capacity());
      for (const auto &candidate : maps)
        Emit(candidate);
    }
  }
  return std::cout.good() && Empty(DefaultResourceBudget());
}

#ifndef GJXL_REPRESENTATION_ORACLE_ONLY
bool PurePlans() {
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!FixtureOk(budget.Reserve(1, &job)))
    return false;
  {
    ResourceContextScope scope({&job, ResourceClass::kSerializer});
    ArmNextManagedHostAllocationFailureForTest();
    for (size_t w = 1; w <= 137; ++w) {
      for (size_t h = 1; h <= 67; ++h) {
        const size_t b = w * h;
        for (auto behavior : {kFull, kEffort7Dct8Sampled}) {
          for (size_t workers : {1ul, 2ul, 8ul}) {
            CoefficientOrderStoragePlan plan;
            if (!FixtureOk(ComputeCoefficientOrderStoragePlan({w, h}, behavior,
                                                              workers, &plan)))
              return false;
            size_t elements = 0, tokens = 0;
            // Independent five-family recipe, including transpose eligibility.
            for (Extent2D shape :
                 {Extent2D{1, 1}, {2, 2}, {4, 4}, {2, 1}, {4, 2}}) {
              if ((w < 5 && h < 5) ||
                  !((shape.width <= w && shape.height <= h) ||
                    (shape.height <= w && shape.width <= h)))
                continue;
              const size_t blocks = shape.width * shape.height;
              elements += 3 * 64 * blocks;
              tokens += 3 * (1 + 63 * blocks);
            }
            const size_t groups = ((w + 31) / 32) * ((h + 31) / 32);
            const size_t participants = w < 5 && h < 5 ? 0
                                        : b < 1024 ? 1
                                                   : std::min(workers, groups);
            if (!FixtureCheck(plan.maximum_order_elements == elements &&
                                  plan.maximum_tokens == tokens &&
                                  plan.maximum_participants == participants &&
                                  plan.ac_group_count == groups &&
                                  plan.orders.retained_bytes == 4 * elements &&
                                  plan.tokens.retained_bytes ==
                                      2 * sizeof(EntropyToken) * tokens,
                              "Order plan differs from independent recipe"))
              return false;
          }
        }
        for (bool exhaustive : {false, true}) {
          BlockContextMapStoragePlan plan;
          if (!FixtureOk(
                  ComputeBlockContextMapStoragePlan({w, h}, exhaustive, &plan)))
            return false;
          const size_t thresholds = b >= 8192;
          const size_t maps = !exhaustive || b < 1024 ? 1 : 5 + thresholds;
          if (!FixtureCheck(
                  plan.maximum_maps == maps &&
                      plan.maximum_thresholds == thresholds &&
                      plan.maximum_map_entries == 39 * (1 + thresholds) &&
                      plan.maximum_ac_contexts == (b < 1024 ? 1980 : 7920) &&
                      plan.map.retained_bytes == 39 + 43 * thresholds,
                  "Block-map plan differs from independent recipe"))
            return false;
        }
      }
    }
    CoefficientOrderStoragePlan large;
    BlockContextMapStoragePlan map;
    if (!FixtureOk(ComputeCoefficientOrderStoragePlan({1ul << 24, 1}, kFull, 8,
                                                      &large)) ||
        !FixtureOk(
            ComputeBlockContextMapStoragePlan({1ul << 24, 1}, true, &map)) ||
        !FixtureCheck(ManagedHostAllocationFailurePendingForTest() &&
                          budget.snapshot().peak_backing_bytes == 0,
                      "Count-only planning allocated backing"))
      return false;
    DisarmManagedHostAllocationFailureForTest();
  }
  job.Reset();
  CoefficientOrderStoragePlan order;
  BlockContextMapStoragePlan map;
  order.maximum_tokens = 17;
  map.maximum_maps = 19;
  const auto old_order = order;
  const auto old_map = map;
  for (Extent2D invalid : {Extent2D{}, {0, 1}, {SIZE_MAX, 2}}) {
    if (!FixtureCheck(
            !ComputeCoefficientOrderStoragePlan(invalid, kFull, 1, &order)
                    .ok() &&
                order == old_order &&
                !ComputeBlockContextMapStoragePlan(invalid, true, &map).ok() &&
                map == old_map,
            "Invalid representation plan modified output"))
      return false;
  }
  return FixtureCheck(
             !ComputeCoefficientOrderStoragePlan({SIZE_MAX, 1}, kFull, 1,
                                                 &order)
                     .ok() &&
                 !ComputeCoefficientOrderStoragePlan({32, 32}, kFull, 0, &order)
                      .ok() &&
                 !ComputeCoefficientOrderStoragePlan({32, 32}, kFull, 9, &order)
                      .ok() &&
                 !ComputeCoefficientOrderStoragePlan(
                      {32, 32}, static_cast<VarDctCoefficientOrderBehavior>(99),
                      1, &order)
                      .ok() &&
                 !ComputeCoefficientOrderStoragePlan({32, 32}, kFull, 1,
                                                     nullptr)
                      .ok() &&
                 !ComputeBlockContextMapStoragePlan({32, 32}, true, nullptr)
                      .ok() &&
                 order == old_order && map == old_map,
             "Invalid options changed plan") &&
         Empty(budget);
}

bool CheckOrders(const FrameFixture &f, VarDctCoefficientOrderBehavior behavior,
                 size_t workers, const SimpleCoefficientOrders &oracle) {
  CoefficientOrderStoragePlan plan;
  if (!FixtureOk(ComputeCoefficientOrderStoragePlan(f.blocks, behavior, workers,
                                                    &plan)))
    return false;
  const size_t limit = std::max(1ul, plan.working.peak_bytes);
  ResourceBudget budget(limit);
  ResourceReservation job;
  if (!FixtureOk(budget.Reserve(limit, &job)))
    return false;
  {
    SimpleCoefficientOrders orders;
    Storage<EntropyToken> tokens;
    {
      thread_budget_internal::EncodeScope threads(workers);
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      if (!FixtureOk(ComputeSimpleCoefficientOrdersForEncoder(
              f.view(), behavior, &orders)) ||
          !FixtureOk(TokenizeSimpleCoefficientOrders(orders, &tokens)))
        return false;
    }
    Backings backing;
    backing.Add(orders);
    backing.Add(tokens);
    if (!FixtureCheck(
            orders == oracle && tokens.size() <= plan.maximum_tokens &&
                budget.snapshot().peak_backing_bytes <= plan.working.peak_bytes,
            "Order operation exceeded plan or changed with workers") ||
        !backing.Matches(budget, plan.orders.retained_bytes +
                                     plan.tokens.retained_bytes))
      return false;
    job.Reset(); // Returned owners retain their charges after producer exit.
    if (!backing.Matches(budget, plan.orders.retained_bytes +
                                     plan.tokens.retained_bytes))
      return false;
  }
  return Empty(budget);
}

bool CheckMaps(const FrameFixture &f, bool exhaustive) {
  BlockContextMapStoragePlan plan;
  if (!FixtureOk(
          ComputeBlockContextMapStoragePlan(f.blocks, exhaustive, &plan)))
    return false;
  ResourceBudget budget(plan.working.peak_bytes);
  ResourceReservation job;
  if (!FixtureOk(budget.Reserve(plan.working.peak_bytes, &job)))
    return false;
  {
    SimpleBlockContextMap map;
    Storage<SimpleBlockContextMap> maps;
    {
      ResourceContextScope scope({&job, ResourceClass::kPreparation});
      const Status status =
          exhaustive ? ComputeSimpleBlockContextMapCandidatesForEncoder(
                           f.view(), &maps)
                     : ComputeSimpleBlockContextMapForEncoder(f.view(), &map);
      if (!FixtureOk(status))
        return false;
    }
    Backings backing;
    if (exhaustive) {
      if (!FixtureCheck(maps.size() <= plan.maximum_maps, "Too many maps"))
        return false;
      backing.Add(maps);
      for (const auto &candidate : maps)
        backing.Add(candidate);
    } else
      backing.Add(map);
    if (!backing.Matches(budget, plan.output.retained_bytes) ||
        !FixtureCheck(budget.snapshot().peak_backing_bytes <=
                          plan.working.peak_bytes,
                      "Map operation exceeded working plan"))
      return false;
    job.Reset();
    if (!backing.Matches(budget, plan.output.retained_bytes))
      return false;
  }
  return Empty(budget);
}

template <typename Output, typename Function>
bool Failures(size_t bound, const Output &sentinel, Function run) {
  for (size_t fail = 0; fail < 1024; ++fail) {
    ResourceBudget budget(bound);
    ResourceReservation job;
    if (!FixtureOk(budget.Reserve(bound, &job)))
      return false;
    bool injected = false;
    {
      auto output = sentinel; // Previous output is outside this reservation.
      {
        ResourceContextScope scope({&job, ResourceClass::kSerializer});
        ArmManagedHostAllocationFailureAfterForTest(fail);
        const Status status = run(&output);
        injected = !ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        if (injected) {
          if (!FixtureCheck(status.code() == StatusCode::kOutOfMemory &&
                                !status.resource_plan_exceeded() &&
                                output == sentinel &&
                                budget.snapshot().total.backing_count == 0,
                            "Injected failure changed representation output"))
            return false;
        } else if (!FixtureOk(status))
          return false;
      }
    }
    job.Reset();
    if (!Empty(budget))
      return false;
    if (!injected) {
      std::cerr << "Representation allocation failure positions: " << fail
                << '\n';
      return true;
    }
  }
  return FixtureCheck(false, "Representation allocation sweep did not finish");
}

template <typename Output, typename Function>
bool Underplanned(const Output &sentinel, Function run) {
  ResourceBudget budget(1);
  ResourceReservation job;
  if (!FixtureOk(budget.Reserve(1, &job)) || !FixtureOk(job.ReduceCapacity(0)))
    return false;
  auto output = sentinel;
  const size_t default_peak =
      DefaultResourceBudget().snapshot().peak_backing_bytes;
  {
    ResourceContextScope scope({&job, ResourceClass::kSerializer});
    ArmNextManagedHostAllocationFailureForTest();
    const Status status = run(&output);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!FixtureCheck(
            status.resource_plan_exceeded() && pending && output == sentinel &&
                budget.snapshot().peak_backing_bytes == 0 &&
                DefaultResourceBudget().snapshot().peak_backing_bytes ==
                    default_peak,
            "Underplanned representation escaped authorization"))
      return false;
  }
  job.Reset();
  return Empty(budget);
}

bool CheckFailures() {
  FrameFixture f;
  if (!f.Create({129, 65}, 7, 2))
    return false;
  thread_budget_internal::EncodeScope threads(1); // Hook is thread-local.
  CoefficientOrderStoragePlan order_plan;
  BlockContextMapStoragePlan map_plan;
  if (!FixtureOk(ComputeCoefficientOrderStoragePlan(f.blocks, kFull, 1,
                                                    &order_plan)) ||
      !FixtureOk(ComputeBlockContextMapStoragePlan(f.blocks, true, &map_plan)))
    return false;
  SimpleCoefficientOrders orders;
  if (!FixtureOk(
          ComputeSimpleCoefficientOrdersForEncoder(f.view(), kFull, &orders)) ||
      !FixtureCheck(orders.used_order_mask != 0,
                    "Failure fixture has no custom orders"))
    return false;
  SimpleCoefficientOrders order_sentinel;
  order_sentinel.used_order_mask = 123;
  Storage<EntropyToken> token_sentinel{{7, 11}};
  SimpleBlockContextMap map_sentinel = DefaultSimpleBlockContextMap();
  Storage<SimpleBlockContextMap> maps_sentinel{map_sentinel};
  if (!Failures(order_plan.working.peak_bytes, order_sentinel,
                [&](auto *out) {
                  return ComputeSimpleCoefficientOrdersForEncoder(f.view(),
                                                                  kFull, out);
                }) ||
      !Failures(order_plan.working.peak_bytes, token_sentinel,
                [&](auto *out) {
                  return TokenizeSimpleCoefficientOrders(orders, out);
                }) ||
      !Failures(map_plan.working.peak_bytes, maps_sentinel, [&](auto *out) {
        return ComputeSimpleBlockContextMapCandidatesForEncoder(f.view(), out);
      }))
    return false;
  if (!FixtureOk(
          ComputeBlockContextMapStoragePlan(f.blocks, false, &map_plan)) ||
      !Failures(map_plan.working.peak_bytes, map_sentinel, [&](auto *out) {
        return ComputeSimpleBlockContextMapForEncoder(f.view(), out);
      }))
    return false;

  // Hooks are still armed when authorization rejects the very first backing.
  if (!Underplanned(order_sentinel,
                    [&](auto *out) {
                      return ComputeSimpleCoefficientOrdersForEncoder(
                          f.view(), kFull, out);
                    }) ||
      !Underplanned(token_sentinel,
                    [&](auto *out) {
                      return TokenizeSimpleCoefficientOrders(orders, out);
                    }) ||
      !Underplanned(map_sentinel,
                    [&](auto *out) {
                      return ComputeSimpleBlockContextMapForEncoder(f.view(),
                                                                    out);
                    }) ||
      !Underplanned(maps_sentinel, [&](auto *out) {
        return ComputeSimpleBlockContextMapCandidatesForEncoder(f.view(), out);
      }))
    return false;
  return true;
}

bool WorkerUnderplan() {
  FrameFixture f;
  if (!f.Create({64, 32}, 2, 2))
    return false;
  // Enough for the caller's view/dispatch arrays, but not one DCT32 counter
  // vector. A worker's typed failure must reach the caller after every join.
  const size_t limit =
      2 * (sizeof(VarDctAcGroupView) + sizeof(Status) + sizeof(std::thread));
  ResourceBudget budget(limit);
  ResourceReservation job;
  if (!FixtureOk(budget.Reserve(limit, &job)))
    return false;
  SimpleCoefficientOrders output;
  output.used_order_mask = 123;
  const size_t default_peak =
      DefaultResourceBudget().snapshot().peak_backing_bytes;
  {
    thread_budget_internal::EncodeScope threads(2);
    ResourceContextScope scope({&job, ResourceClass::kSerializer});
    const Status status =
        ComputeSimpleCoefficientOrdersForEncoder(f.view(), kFull, &output);
    if (!FixtureCheck(
            status.resource_plan_exceeded() && output.used_order_mask == 123 &&
                budget.snapshot().peak_backing_bytes <= limit &&
                budget.snapshot().total.backing_count == 0 &&
                DefaultResourceBudget().snapshot().peak_backing_bytes ==
                    default_peak,
            "Worker plan failure lost its type, leaked, or changed output"))
      return false;
  }
  job.Reset();
  return Empty(budget);
}

bool SampledAndClearedOrders() {
  for (int pattern : {0, 2}) {
    FrameFixture f;
    if (!f.Create({129, 65}, 0, pattern))
      return false;
    for (auto behavior : {kFull, kEffort7Dct8Sampled}) {
      SimpleCoefficientOrders orders;
      {
        thread_budget_internal::EncodeScope threads(1);
        if (!FixtureOk(ComputeSimpleCoefficientOrdersForEncoder(
                f.view(), behavior, &orders)))
          return false;
      }
      if (!FixtureCheck(
              pattern == 0 ? orders.used_order_mask == 0 &&
                                 orders.orders[0][0].empty() &&
                                 orders.orders[0][0].capacity() == 64
                           : orders.used_order_mask == 1,
              "Sampling/cleared-capacity fixture missed its intended path"))
        return false;
      for (size_t workers : {1ul, 2ul, 8ul})
        if (!CheckOrders(f, behavior, workers, orders))
          return false;
      if (pattern == 2) {
        CoefficientOrderStoragePlan plan;
        if (!FixtureOk(ComputeCoefficientOrderStoragePlan(f.blocks, behavior, 1,
                                                          &plan)) ||
            !Failures(plan.working.peak_bytes, SimpleCoefficientOrders{},
                      [&](auto *out) {
                        thread_budget_internal::EncodeScope threads(1);
                        return ComputeSimpleCoefficientOrdersForEncoder(
                            f.view(), behavior, out);
                      }))
          return false;
      }
    }
  }
  return true;
}

bool RealOperations() {
  for (Extent2D extent : kExtents) {
    for (size_t family = 0; family < 8; ++family) {
      FrameFixture f;
      if (!f.Create(extent, family, family % 3))
        return false;
      for (auto behavior : {kFull, kEffort7Dct8Sampled}) {
        SimpleCoefficientOrders oracle;
        {
          thread_budget_internal::EncodeScope threads(1);
          if (!FixtureOk(ComputeSimpleCoefficientOrdersForEncoder(
                  f.view(), behavior, &oracle)))
            return false;
        }
        for (size_t workers : {1ul, 2ul, 8ul})
          if (!CheckOrders(f, behavior, workers, oracle))
            return false;
      }
      for (bool exhaustive : {false, true})
        if (!CheckMaps(f, exhaustive))
          return false;
    }
  }
  return Empty(DefaultResourceBudget());
}
#endif
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--oracle")
    return EmitOracle() ? EXIT_SUCCESS : EXIT_FAILURE;
#ifndef GJXL_REPRESENTATION_ORACLE_ONLY
  return PurePlans() && RealOperations() && SampledAndClearedOrders() &&
                 CheckFailures() && WorkerUnderplan() &&
                 Empty(DefaultResourceBudget())
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
#else
  return EXIT_FAILURE;
#endif
}
