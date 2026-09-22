// SPDX-License-Identifier: Apache-2.0
#include "codec/vardct_frame_view_internal.h"
#include "codestream/dc_group.h"
#include "codestream/token_storage_plan.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"
#include "quantized_frame_fixture.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::thread_budget_internal;
using namespace gjxl::resource_budget_internal;
using gjxl_test::Check;
static void Require(bool good, const char *message) {
  if (!good)
    throw std::runtime_error(message);
}
int main() {
  try {
    // Four independent DC groups, with narrow right/bottom edges and mixed AC
    // metadata. Deliberately use nonconstant DC values to exercise feedback.
    auto owner = gjxl_test::MakeFrame(gjxl_test::kStrategies.size(), 0, 260);
    const auto blocks = owner.geometry().block_grid().blocks;
    std::array<std::vector<int32_t>, 3> dc;
    ConstImage3I32View dc_view;
    for (size_t c = 0; c < 3; ++c) {
      dc[c].resize(blocks.width * blocks.height);
      for (size_t i = 0; i < dc[c].size(); ++i)
        dc[c][i] =
            int32_t((i * (11 + c * 6) + i / blocks.width * 73) % 1021) - 510;
      dc_view.plane[c] = {dc[c].data(), blocks, blocks.width};
    }
    vardct_frame_internal::VarDctFrameView frame(
        {.input = {.geometry = owner.geometry(),
                   .strategies = &owner.strategies(),
                   .raw_quant_field = owner.raw_quant_field(),
                   .quantizer = &owner.quantizer(),
                   .color_correlation = &owner.color_correlation(),
                   .epf_sharpness = owner.epf_sharpness()},
         .quantized_dc = dc_view});
    size_t cases = 0, failure_cases = 0;
    for (auto prediction :
         {VarDctDcPrediction::kGradient, VarDctDcPrediction::kWeighted}) {
      Storage<SimpleDcGroupTokenStreams> expected;
      {
        EncodeScope scope(1);
        Check(TokenizeSimpleDcGroupsForEncoder(frame, &expected, prediction));
      }
      Require(expected.size() == 4,
              "DC fixture did not cross group boundaries");
      for (size_t workers : {size_t{1}, size_t{2}, size_t{4}, size_t{8}}) {
        std::shared_ptr<const ExecutionDomain> domain;
        Check(ExecutionDomain::Create({0, workers}, &domain));
        TokenizationStoragePlan plan;
        Check(ComputeTokenizationStoragePlan(blocks,
                                             {.context_count = 1485,
                                              .workers = workers,
                                              .dc_prediction = prediction},
                                             &plan));
        ResourceBudget budget(plan.dc.peak_bytes);
        ResourceReservation reservation;
        Check(budget.Reserve(plan.dc.peak_bytes, &reservation));
        {
          ResourceContextScope resources(
              {&reservation, ResourceClass::kSerializer});
          CpuExecutionScope cpu;
          Check(cpu.Start(domain, workers));
          EncodeScope scope(workers);
          Storage<SimpleDcGroupTokenStreams> actual;
          Check(TokenizeSimpleDcGroupsForEncoder(frame, &actual, prediction));
          Require(actual == expected,
                  "Parallel DC changed tokens or group order");
          Require(domain->snapshot().peak_cpu_protected_slots <= workers,
                  "DC exceeded participant cap");
        }
        reservation.Reset();
        Require(budget.snapshot().committed_bytes() == 0, "DC backing leaked");
        Require(domain->snapshot().active_cpu_participants == 0 &&
                    domain->snapshot().reserved_cpu_workers == 0,
                "DC participant leaked");
        ++cases;
      }
      for (auto kind : {WorkerLaunchFailureKind::kSystemError,
                        WorkerLaunchFailureKind::kBadAlloc})
        for (size_t before : {size_t{0}, size_t{1}, size_t{2}}) {
          // The dispatcher caps even an explicit limit at hardware concurrency;
          // the caller occupies one participant slot. Inject only at launches
          // that this host can reach, retaining serial coverage above.
          if (before >=
              std::max(1u, std::thread::hardware_concurrency()) - 1)
            continue;
          CpuParticipantTracker tracker;
          EncodeScope scope(4, &tracker);
          WorkerLaunchFaultForTesting fault{
              WorkerLaunchSite::kSerializerSections, before, kind};
          Storage<SimpleDcGroupTokenStreams> actual(1);
          actual[0].block_x = 999;
          {
            WorkerLaunchFaultScopeForTesting inject(&fault);
            auto status =
                TokenizeSimpleDcGroupsForEncoder(frame, &actual, prediction);
            Require(!status.ok() && fault.triggered &&
                        fault.launched_in_group == before,
                    "DC launch fault did not propagate");
          }
          Require(actual.size() == 1 && actual[0].block_x == 999,
                  "Failed DC published partial output");
          Require(tracker.active() == 0 && tracker.peak() <= 4,
                  "Failed DC left live workers");
          Check(TokenizeSimpleDcGroupsForEncoder(frame, &actual, prediction));
          Require(actual == expected, "DC launch fault poisoned recovery");
          ++cases;
          ++failure_cases;
        }
    }
    std::cout << "Verified " << cases
              << " DC cases (" << failure_cases
              << " launch-failure cases): exact gradient/weighted streams, finite storage "
                 "plans, CPU caps, partial launch failure and recovery.\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
