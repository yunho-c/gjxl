// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include "codestream/token_storage_plan.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include "codec/vardct_frame_view_internal.h"
#include "codestream/ac_group.h"
#include "codestream/block_context_map.h"
#include "codestream/coefficient_order.h"
#include "codestream/dc_group.h"
#include "codestream/entropy_internal.h"
#include "codestream_frame_fixture.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;
using codestream_test_internal::FrameFixture;
using codestream_test_internal::kStrategies;
using enum VectorCapacityPolicy;

bool Check(bool condition, const char *message) {
  if (!condition)
    std::cerr << message << '\n';
  return condition;
}
bool Ok(const Status &status) {
  if (!status.ok())
    std::cerr << status.message() << '\n';
  return status.ok();
}
bool Empty(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  return Check(s.committed_bytes() == 0 && s.total.backing_count == 0 &&
                   s.total.pending_count == 0 && s.open_reservations == 0 &&
                   s.waiting_requests == 0,
               "Token storage test leaked a charge");
}

bool CheckVectorBounds() {
  for (size_t n :
       {size_t{0}, 1ul, 2ul, 3ul, 7ul, 31ul, 32ul, 33ul, 255ul, 1025ul}) {
    for (auto policy : {kFreshExact, kReusedExact, kGrowing}) {
      HostStorageBound bound;
      if (!Check(bound.AddVector<uint32_t>(n, policy),
                 "Vector planning failed"))
        return false;
      ResourceBudget budget;
      ResourceReservation reservation;
      if (!Ok(budget.Reserve(std::max(size_t{1}, bound.peak_bytes),
                             &reservation)))
        return false;
      {
        ResourceContextScope scope({&reservation, ResourceClass::kSerializer});
        ManagedVector<uint32_t> values;
        if (policy == kFreshExact) {
          values.reserve(n);
          for (size_t i = 0; i < n; ++i)
            values.push_back(static_cast<uint32_t>(i));
        } else if (policy == kReusedExact) {
          for (size_t i = 0; i <= n; ++i) {
            values.clear();
            values.reserve(i);
            values.assign(i, 17);
          }
        } else {
          for (size_t i = 0; i < n; ++i)
            values.push_back(static_cast<uint32_t>(i));
          values.resize(n / 2);
          values.shrink_to_fit();
          // Forward-range insertion may replace an existing backing.
          std::vector<uint32_t> input(n - values.size(), 9);
          values.insert(values.begin(), input.begin(), input.end());
          for (size_t step = 0; step < 64 && n != 0; ++step) {
            const size_t size = (step * 127 + 13) % (n + 1);
            if (step % 3 == 0)
              values.assign(size, 7);
            else if (step % 3 == 1)
              values.resize(size);
            else
              values.reserve(size);
          }
        }
        auto snapshot = budget.snapshot();
        if (!Check(snapshot.total.live_capacity_bytes ==
                           values.capacity() * sizeof(uint32_t) &&
                       snapshot.total.live_capacity_bytes <=
                           bound.retained_bytes &&
                       snapshot.peak_backing_bytes <= bound.peak_bytes,
                   "Observed vector backing exceeds its bound"))
          return false;
      }
      reservation.Reset();
      if (!Empty(budget))
        return false;
    }
  }
  HostStorageBound sentinel{7, 11};
  auto bound = sentinel;
  if (!Check(!bound.AddVector<uint64_t>(std::numeric_limits<size_t>::max(),
                                        kGrowing) &&
                 bound == sentinel &&
                 !bound.Add({0, std::numeric_limits<size_t>::max()}) &&
                 bound == sentinel && !bound.Add({2, 1}) && bound == sentinel &&
                 !bound.Add({1, 2}, std::numeric_limits<size_t>::max()) &&
                 bound == sentinel,
             "Overflow changed a storage bound"))
    return false;
  return Empty(DefaultResourceBudget());
}

bool CheckPurePlans() {
  // A one-byte admitted scope plus an armed physical-allocation hook catches
  // managed backing in successful planners; this is not global-new
  // interception.
  ResourceBudget budget(1);
  ResourceReservation reservation;
  if (!Ok(budget.Reserve(1, &reservation)))
    return false;
  {
    ResourceContextScope scope({&reservation, ResourceClass::kSerializer});
    ArmNextManagedHostAllocationFailureForTest();
    for (size_t h = 1; h <= 32; ++h) {
      for (size_t w = 1; w <= 32; ++w) {
        const size_t b = w * h;
        for (size_t anchors : {size_t{1}, (b + 1) / 2, b}) {
          AcGroupTokenCounts counts;
          if (!Ok(ComputeAcGroupTokenCounts({w, h}, anchors, &counts)) ||
              !Check(counts.block_count == b &&
                         counts.coefficient_count == 64 * b &&
                         counts.block_context_keys == 3 * anchors &&
                         counts.token_capacity == 3 * (64 * b + anchors),
                     "AC reservation differs from frozen recipe"))
            return false;
        }
        for (size_t contexts : {495ul, 1980ul, 7425ul, 65534ul}) {
          for (bool fixed : {false, true}) {
            AcGroupTokenStoragePlan plan;
            if (!Ok(ComputeAcGroupTokenStoragePlan({w, h}, b, contexts, fixed,
                                                   &plan)))
              return false;
            const size_t n = 195 * b;
            const size_t populations = std::min(contexts, n);
            const size_t symbols = std::min(n, contexts * 128);
            const size_t expected =
                n * 6 +
                (fixed ? populations * sizeof(SimpleAcContextPopulation) +
                             2 * symbols * sizeof(SimpleAcSymbolPopulation)
                       : 0);
            if (!Check(
                    plan.direct_output.retained_bytes == expected &&
                        plan.direct_output.peak_bytes ==
                            expected +
                                (fixed ? symbols *
                                             sizeof(SimpleAcSymbolPopulation)
                                       : 0) &&
                        plan.template_output.retained_bytes == n * 8 + 12 * b,
                    "AC backing bound differs from independent count recipe"))
              return false;
          }
        }
      }
    }
    for (size_t h = 1; h <= 256; ++h) {
      for (size_t w = 1; w <= 256; ++w) {
        for (size_t anchors : {size_t{1}, w * h}) {
          DcGroupTokenCounts counts;
          if (!Ok(ComputeDcGroupTokenCounts({w, h}, anchors, &counts)) ||
              !Check(counts.dc_tokens == 3 * w * h &&
                         counts.metadata_tokens ==
                             2 * ((w + 7) / 8) * ((h + 7) / 8) + 2 * anchors +
                                 w * h,
                     "DC reservation differs from frozen recipe"))
            return false;
        }
      }
    }
    HostStorageBound natural;
    if (!Ok(ComputeAcNaturalOrderStorageBound(&natural)) ||
        !Check(natural.retained_bytes ==
                       (64 + 256 + 1024 + 128 + 128 + 512 + 512) * 4 &&
                   natural.peak_bytes == natural.retained_bytes + 1024 &&
                   ManagedHostAllocationFailurePendingForTest() &&
                   budget.snapshot().peak_backing_bytes == 0,
               "Pure planning allocated or natural-order bound changed"))
      return false;
    DisarmManagedHostAllocationFailureForTest();
  }
  reservation.Reset();
  AcGroupTokenCounts ac{17, 19, 23, 29};
  const auto old_ac = ac;
  DcGroupTokenCounts dc{17, 19, 23, 29};
  const auto old_dc = dc;
  for (Extent2D invalid : {Extent2D{}, Extent2D{0, 3}, Extent2D{33, 1},
                           Extent2D{1, std::numeric_limits<size_t>::max()}}) {
    if (!Check(!ComputeAcGroupTokenCounts(invalid, 1, &ac).ok() && ac == old_ac,
               "Invalid AC geometry changed output"))
      return false;
  }
  if (!Check(
          !ComputeDcGroupTokenCounts({257, 1}, 1, &dc).ok() && dc == old_dc &&
              !ComputeAcGroupTokenCounts({2, 2}, 0, &ac).ok() && ac == old_ac &&
              !ComputeAcGroupTokenCounts({2, 2}, 5, &ac).ok() && ac == old_ac &&
              !ComputeDcGroupTokenCounts({2, 2}, 5, &dc).ok() && dc == old_dc &&
              !ComputeDcGroupTokenCounts({2, 2}, 0, &dc).ok() && dc == old_dc &&
              !ComputeAcGroupTokenCounts({1, 1}, 1, nullptr).ok() &&
              !ComputeDcGroupTokenCounts({1, 1}, 1, nullptr).ok(),
          "Invalid token count arguments were accepted"))
    return false;
  return Empty(budget) && Empty(DefaultResourceBudget());
}

bool CheckGroupClassSum() {
  for (size_t h : {1ul, 7ul, 31ul, 32ul, 33ul, 65ul, 255ul, 256ul, 257ul}) {
    for (size_t w : {1ul, 9ul, 32ul, 33ul, 63ul, 64ul, 257ul, 513ul}) {
      for (bool exhaustive : {false, true}) {
        TokenizationStorageOptions options{
            .exhaustive = exhaustive,
            .collect_fixed_populations = !exhaustive,
            .context_count = 7425,
            .order_count = exhaustive ? 2ul : 1ul,
            .map_count = exhaustive ? 6ul : 1ul,
            .workers = 8};
        TokenizationStoragePlan plan;
        if (!Ok(ComputeTokenizationStoragePlan({w, h}, options, &plan)))
          return false;
        HostStorageBound expected;
        size_t ac_groups = 0;
        for (size_t y = 0; y < h; y += 32) {
          for (size_t x = 0; x < w; x += 32) {
            Extent2D extent{std::min(32ul, w - x), std::min(32ul, h - y)};
            AcGroupTokenStoragePlan group;
            if (!Ok(ComputeAcGroupTokenStoragePlan(extent,
                                                   extent.width * extent.height,
                                                   7425, !exhaustive, &group)))
              return false;
            if (exhaustive) {
              if (!expected.Add(group.template_output, 2) ||
                  !expected.Add(group.context_output, 12))
                return false;
            } else if (!expected.Add(group.direct_output))
              return false;
            ++ac_groups;
          }
        }
        AcGroupTokenStoragePlan largest;
        const Extent2D extent{std::min(32ul, w), std::min(32ul, h)};
        if (!Ok(ComputeAcGroupTokenStoragePlan(extent,
                                               extent.width * extent.height,
                                               7425, !exhaustive, &largest)))
          return false;
        if (exhaustive) {
          if (!expected.Add(largest.template_scratch,
                            std::min(8ul, 2 * ac_groups)) ||
              !expected.AddVector<SimpleAcGroupTokenTemplate>(ac_groups,
                                                              kFreshExact, 2) ||
              !expected.AddVector<Storage<uint16_t>>(ac_groups, kFreshExact,
                                                     12))
            return false;
        } else {
          HostStorageBound natural;
          if (!Ok(ComputeAcNaturalOrderStorageBound(&natural)) ||
              !expected.Add(natural) ||
              !expected.Add(largest.direct_scratch, std::min(8ul, ac_groups)) ||
              !expected.AddVector<SimpleAcGroupTokenData>(ac_groups,
                                                          kFreshExact) ||
              !expected.AddVector<PreparedFixedAnsCluster>(7425, kFreshExact))
            return false;
        }
        // Frozen private metadata anchor layout, not a sizeof from the planner.
        struct DcAnchor {
          size_t x, y;
          int32_t code;
        };
        const size_t dc_groups = ((w + 255) / 256) * ((h + 255) / 256);
        const size_t dc_expected =
            sizeof(EntropyToken) *
                (6 * w * h + 2 * ((w + 7) / 8) * ((h + 7) / 8)) +
            dc_groups * sizeof(SimpleDcGroupTokenStreams) +
            std::min(w, 256ul) * std::min(h, 256ul) * (sizeof(DcAnchor) + 1);
        if (!Check(plan.ac_group_count == ac_groups && plan.ac == expected &&
                       plan.dc_group_count == dc_groups &&
                       plan.dc.retained_bytes == dc_expected &&
                       plan.dc.peak_bytes == dc_expected,
                   "Constant-time group sum differs from explicit group "
                   "traversal"))
          return false;
      }
    }
  }
  TokenizationStoragePlan sentinel{17, 23, {5, 7}, {11, 13}}, plan = sentinel;
  TokenizationStorageOptions options{.context_count = 1980};
  for (Extent2D extent : {Extent2D{}, Extent2D{0, 1},
                          Extent2D{std::numeric_limits<size_t>::max(), 2},
                          Extent2D{std::numeric_limits<size_t>::max(), 1}}) {
    if (!Check(!ComputeTokenizationStoragePlan(extent, options, &plan).ok() &&
                   plan == sentinel,
               "Aggregate overflow changed output"))
      return false;
  }
  for (int bad = 0; bad < 7; ++bad) {
    auto invalid = options;
    if (bad == 0)
      invalid.context_count = 0;
    if (bad == 1)
      invalid.context_count = 65535;
    if (bad == 2)
      invalid.workers = 0;
    if (bad == 3)
      invalid.map_count = 0;
    if (bad == 4)
      invalid.order_count = 3;
    if (bad == 5)
      invalid.order_count = 2; // Ordinary has one order candidate.
    if (bad == 6)
      invalid.exhaustive = true; // Fixed populations not used here.
    if (!Check(!ComputeTokenizationStoragePlan({32, 32}, invalid, &plan).ok() &&
                   plan == sentinel,
               "Invalid aggregate options changed output"))
      return false;
  }
  // A large valid request is planned without allocating its image or tokens.
  return Ok(ComputeTokenizationStoragePlan({1ul << 24, 1}, options, &plan));
}


template <typename Function>
bool Parallel(size_t tasks, size_t workers, Function &&function) {
  std::atomic<size_t> next{0};
  std::array<std::thread, 8> threads;
  std::array<Status, 8> results;
  const auto context = CurrentResourceContext();
  for (size_t worker = 0; worker < workers; ++worker) {
    threads[worker] = std::thread([&, worker] {
      ResourceContextScope scope(context);
      for (;;) {
        const size_t task = next.fetch_add(1);
        if (task >= tasks)
          break;
        results[worker] = function(task, worker);
        if (!results[worker].ok())
          break;
      }
    });
  }
  for (size_t worker = 0; worker < workers; ++worker)
    threads[worker].join();
  for (size_t worker = 0; worker < workers; ++worker)
    if (!Ok(results[worker]))
      return false;
  return true;
}

bool CheckRealTokenizers(const FrameFixture &fixture, bool exhaustive,
                         bool fixed, size_t workers) {
  auto map = JxlDefaultSimpleBlockContextMap();
  SimpleCoefficientOrders natural, custom;
  custom.used_order_mask = 1;
  for (auto &channel : custom.orders[0]) {
    if (!Ok(ComputeSimpleNaturalCoefficientOrder(AcStrategyType::kDct8,
                                                 &channel)))
      return false;
    std::swap(channel[62], channel[63]);
  }
  TokenizationStorageOptions options{.exhaustive = exhaustive,
                                     .collect_fixed_populations = fixed,
                                     .context_count = map.ac_context_count(),
                                     .order_count = exhaustive ? 2ul : 1ul,
                                     .map_count = exhaustive ? 6ul : 1ul,
                                     .workers = workers};
  TokenizationStoragePlan plan;
  if (!Ok(ComputeTokenizationStoragePlan(fixture.blocks, options, &plan)))
    return false;
  HostStorageBound total = plan.dc;
  if (!total.Add(plan.ac))
    return false;
  ResourceBudget budget(total.peak_bytes);
  ResourceReservation reservation;
  if (!Ok(budget.Reserve(total.peak_bytes, &reservation)))
    return false;
  const auto frame = fixture.view();
  {
    ResourceContextScope scope({&reservation, ResourceClass::kSerializer});
    Storage<SimpleDcGroupTokenStreams> dc;
    if (!Ok(TokenizeSimpleDcGroupsForEncoder(frame, &dc)))
      return false;
    const size_t n = frame.ac_group_count();
    if (!exhaustive) {
      SimpleAcNaturalOrders orders;
      if (!Ok(PrepareSimpleAcNaturalOrders(&orders)))
        return false;
      Storage<SimpleAcGroupTokenData> groups(n);
      std::array<SimpleAcTokenizationScratch, 8> scratch;
      if (!Parallel(n, workers, [&](size_t i, size_t w) {
            return TokenizeSimpleAcGroupForEncoder(
                frame, custom, orders, map, i, fixed, &scratch[w], &groups[i]);
          }))
        return false;
      // The encoder's reduction has exactly one fresh context-count array.
      Storage<PreparedFixedAnsCluster> reduced(fixed ? map.ac_context_count()
                                                     : 0);
      for (size_t i = 0; i < n; ++i) {
        // Oracle temporaries live outside this component's reservation.
        ResourceContextScope untracked({});
        SimpleAcGroupTokenTemplate reference;
        Storage<uint16_t> contexts;
        if (!Ok(BuildSimpleAcGroupTokenTemplateForEncoder(frame, custom, i,
                                                          &reference)) ||
            !Ok(MaterializeSimpleAcGroupContextsForEncoder(reference, map,
                                                           &contexts)) ||
            !Check(groups[i].values == reference.values &&
                       groups[i].contexts == contexts,
                   "Direct tokenization changed relative to template oracle"))
          return false;
      }
    } else {
      std::array<Storage<SimpleAcGroupTokenTemplate>, 2> templates;
      for (auto &groups : templates)
        groups.resize(n);
      if (!Parallel(n * 2, workers, [&](size_t task, size_t) {
            return BuildSimpleAcGroupTokenTemplateForEncoder(
                frame, task < n ? natural : custom, task % n,
                &templates[task / n][task % n]);
          }))
        return false;
      std::array<Storage<Storage<uint16_t>>, 12> contexts;
      for (auto &groups : contexts)
        groups.resize(n);
      if (!Parallel(n * 12, workers, [&](size_t task, size_t) {
            const size_t candidate = task / n;
            return MaterializeSimpleAcGroupContextsForEncoder(
                templates[candidate % 2][task % n], map,
                &contexts[candidate][task % n]);
          }))
        return false;
    }
    if (!Check(budget.snapshot().peak_backing_bytes <= total.peak_bytes,
               "Real tokenizer backing exceeded aggregate plan"))
      return false;
  }
  reservation.Reset();
  return Empty(budget) && Empty(DefaultResourceBudget());
}

bool CheckFailuresAndReuse() {
  FrameFixture fixture;
  if (!fixture.Create({33, 9}, 7, 2))
    return false;
  auto map = DefaultSimpleBlockContextMap();
  SimpleCoefficientOrders orders;
  SimpleAcNaturalOrders natural;
  if (!Ok(PrepareSimpleAcNaturalOrders(&natural)))
    return false;
  ResourceBudget budget(1);
  ResourceReservation reservation;
  if (!Ok(budget.Reserve(1, &reservation)))
    return false;
  SimpleAcGroupTokenData output;
  output.values = {7, 9};
  const auto *original = output.values.data();
  {
    ResourceContextScope scope({&reservation, ResourceClass::kSerializer});
    SimpleAcTokenizationScratch scratch;
    ArmNextManagedHostAllocationFailureForTest();
    auto status = TokenizeSimpleAcGroupForEncoder(
        fixture.view(), orders, natural, map, 0, true, &scratch, &output);
    const bool no_backing = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!Check(
            status.resource_plan_exceeded() && no_backing &&
                output.values.data() == original && output.values.size() == 2 &&
                output.values[0] == 7 && output.values[1] == 9,
            "Underplan failed to reject atomically before backing allocation"))
      return false;
  }
  reservation.Reset();
  if (!Empty(budget))
    return false;

  // One scratch owner sees alternating small/large groups, retaining capacity
  // after smaller requests. Old published outputs are excluded from each call.
  AcGroupTokenStoragePlan group_plan;
  if (!Ok(ComputeAcGroupTokenStoragePlan(
          {32, 9}, 32 * 9, map.ac_context_count(), true, &group_plan)))
    return false;
  HostStorageBound bound = group_plan.direct_output;
  if (!bound.Add(group_plan.direct_scratch))
    return false;
  ResourceBudget reuse_budget(bound.peak_bytes);
  ResourceReservation reuse;
  if (!Ok(reuse_budget.Reserve(bound.peak_bytes, &reuse)))
    return false;
  {
    ResourceContextScope scope({&reuse, ResourceClass::kSerializer});
    SimpleAcTokenizationScratch scratch;
    for (size_t repetition = 0; repetition < 16; ++repetition) {
      SimpleAcGroupTokenData result;
      if (!Ok(TokenizeSimpleAcGroupForEncoder(fixture.view(), orders, natural,
                                              map, repetition % 2, true,
                                              &scratch, &result)))
        return false;
    }
  }
  reuse.Reset();
  return Empty(reuse_budget) && Empty(DefaultResourceBudget());
}

bool CheckInjectedFailures() {
  FrameFixture fixture;
  if (!fixture.Create({9, 7}, 7, 1))
    return false;
  const auto frame = fixture.view();
  auto map = JxlDefaultSimpleBlockContextMap();
  SimpleCoefficientOrders orders;
  SimpleAcNaturalOrders natural;
  SimpleAcGroupTokenTemplate reference;
  if (!Ok(PrepareSimpleAcNaturalOrders(&natural)) ||
      !Ok(BuildSimpleAcGroupTokenTemplateForEncoder(frame, orders, 0,
                                                    &reference)))
    return false;
  TokenizationStoragePlan plan;
  if (!Ok(ComputeTokenizationStoragePlan(
          fixture.blocks, {.context_count = map.ac_context_count()}, &plan)))
    return false;
  HostStorageBound bound = plan.ac;
  if (!bound.Add(plan.dc))
    return false;
  size_t failures = 0;
  for (size_t mode = 0; mode < 4; ++mode) {
    bool reached_success = false;
    for (size_t fail_after = 0; fail_after < 128; ++fail_after) {
      ResourceBudget budget(bound.peak_bytes);
      ResourceReservation reservation;
      if (!Ok(budget.Reserve(bound.peak_bytes, &reservation)))
        return false;
      {
        // Caller-owned sentinels are outside the component reservation.
        SimpleAcGroupTokenData direct;
        direct.values = {7, 9};
        SimpleAcGroupTokenTemplate templated;
        templated.values = {11, 13};
        Storage<uint16_t> contexts{17, 19};
        Storage<SimpleDcGroupTokenStreams> dc(1);
        dc[0].dc_tokens = {{23, 29}};
        const auto *direct_pointer = direct.values.data();
        const auto *template_pointer = templated.values.data();
        const auto *context_pointer = contexts.data();
        const auto *dc_pointer = dc.data();
        Status status;
        bool hook_pending = false;
        {
          ResourceContextScope scope(
              {&reservation, ResourceClass::kSerializer});
          SimpleAcTokenizationScratch scratch;
          ArmManagedHostAllocationFailureAfterForTest(fail_after);
          if (mode == 0)
            status = TokenizeSimpleAcGroupForEncoder(
                frame, orders, natural, map, 0, true, &scratch, &direct);
          if (mode == 1)
            status = BuildSimpleAcGroupTokenTemplateForEncoder(frame, orders, 0,
                                                               &templated);
          if (mode == 2)
            status = MaterializeSimpleAcGroupContextsForEncoder(reference, map,
                                                                &contexts);
          if (mode == 3)
            status = TokenizeSimpleDcGroupsForEncoder(frame, &dc);
          hook_pending = ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
        }
        if (status.ok()) {
          if (!Check(hook_pending, "Injected failure was swallowed"))
            return false;
          reached_success = true;
        } else {
          ++failures;
          if (!Check(
                  !hook_pending && status.code() == StatusCode::kOutOfMemory &&
                      !status.resource_plan_exceeded() &&
                      direct.values.data() == direct_pointer &&
                      direct.values.size() == 2 && direct.values[0] == 7 &&
                      direct.values[1] == 9 &&
                      templated.values.data() == template_pointer &&
                      templated.values.size() == 2 &&
                      templated.values[0] == 11 && templated.values[1] == 13 &&
                      contexts.data() == context_pointer &&
                      contexts.size() == 2 && contexts[0] == 17 &&
                      contexts[1] == 19 && dc.data() == dc_pointer &&
                      dc.size() == 1 && dc[0].dc_tokens.size() == 1 &&
                      dc[0].dc_tokens[0] == EntropyToken{23, 29} &&
                      budget.snapshot().total.pending_count == 0,
                  "Tokenizer allocation failure changed output or escaped its "
                  "bound"))
            return false;
        }
      }
      reservation.Reset();
      if (!Empty(budget) || !Empty(DefaultResourceBudget()))
        return false;
      if (reached_success)
        break;
    }
    if (!Check(reached_success,
               "Failure sweep did not reach successful tokenization"))
      return false;
  }
  std::cout << "Verified " << failures
            << " tokenizer allocation failure positions\n";
  return true;
}
} // namespace

int main() {
  if (!CheckVectorBounds() || !CheckPurePlans() || !CheckGroupClassSum() ||
      !CheckFailuresAndReuse() || !CheckInjectedFailures())
    return EXIT_FAILURE;
  for (size_t family = 0; family < 8; ++family) {
    for (int pattern = 0; pattern < 3; ++pattern) {
      FrameFixture fixture;
      if (!fixture.Create(family == 7 ? Extent2D{65, 33} : Extent2D{9, 7},
                          family, pattern))
        return EXIT_FAILURE;
      for (size_t workers : {1ul, 8ul}) {
        if (!CheckRealTokenizers(fixture, false, false, workers) ||
            !CheckRealTokenizers(fixture, false, true, workers) ||
            !CheckRealTokenizers(fixture, true, false, workers))
          return EXIT_FAILURE;
      }
    }
  }
  std::cout << "Token storage bounds and real allocation checks passed\n";
  return EXIT_SUCCESS;
}
