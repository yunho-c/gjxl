// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/ac_tokenization_provider_internal.h"
#include "codestream/encoder_internal.h"
#include "codestream/entropy_readiness_internal.h"
#include "core/worker_launch_internal.h"
#include "quantized_frame_fixture.h"
#include "environment_test_utils.h"

#include <atomic>
#include <iostream>
#include <optional>
#include <thread>

using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::thread_budget_internal;
using gjxl_test::Check;

namespace {
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

// Begin records the branch ceiling before producing real AC tokens. The old
// schedule calls Begin outside the fork; the earlier schedule must call it in
// the one-participant AC branch, before either entropy result is published.
struct Provider final : AcTokenizationProvider {
  Storage<SimpleAcGroupTokenData> groups;
  Storage<EntropyTokenStreamView> views;
  Storage<PreparedFixedAnsCluster> counts;
  size_t begins = 0, finishes = 0, begin_cpu = 0, finish_cpu = 0;
  int failure = 0;
  Status Begin(const vardct_frame_internal::VarDctFrameView& frame,
      const SimpleCoefficientOrders& orders, const SimpleAcNaturalOrders& natural,
      const SimpleBlockContextMap& map, bool populations) override {
    ++begins;
    begin_cpu = CpuThreadCount();
    if (failure == 1) return Status::DeviceError("Injected Begin failure");
    if (failure == 2) throw std::bad_alloc();
    groups.resize(frame.ac_group_count());
    SimpleAcTokenizationScratch scratch;
    for (size_t i = 0; i < groups.size(); ++i) {
      auto status = TokenizeSimpleAcGroupForEncoder(frame, orders, natural, map,
          i, false, &scratch, &groups[i]);
      if (!status.ok()) return status;
      views.push_back(EntropyTokenStreamView::Split(groups[i].values, groups[i].contexts));
    }
    return populations
        ? CollectDefaultEntropyPopulations(views, map.ac_context_count(), &counts)
        : Status::Ok();
  }
  Status Finish(Storage<EntropyTokenStreamView>* streams,
      Storage<PreparedFixedAnsCluster>* populations) override {
    ++finishes;
    finish_cpu = CpuThreadCount();
    if (failure == 3) return Status::DeviceError("Injected Finish failure");
    if (failure == 4) throw std::bad_alloc();
    *streams = views;
    *populations = counts;
    return Status::Ok();
  }
};

void CheckReleased(const ExecutionDomain& domain, size_t cpu) {
  const auto s = domain.snapshot();
  Require(s.peak_cpu_protected_slots <= cpu && !s.active_cpu_participants &&
      !s.reserved_cpu_workers && !s.suspended_cpu_workers &&
      !s.waiting_cpu_callers && !s.active_reservations,
      "Earlier DC exceeded capacity or leaked participation");
}
}  // namespace

int main() try {
  gjxl_test::SetEnvironment("GJXL_EARLY_ENTROPY", nullptr);
  gjxl_test::SetEnvironment("GJXL_EXPERIMENT_EAGER_ENTROPY", nullptr);
  gjxl_test::SetEnvironment("GJXL_EARLY_DC", nullptr);
  auto owner = gjxl_test::MakeFrame(gjxl_test::kStrategies.size(), 2, 36);
  const auto frame = vardct_frame_internal::BorrowFrame(owner);
  std::vector<uint8_t> expected;
  Check(EncodeVarDctCodestreamFromView(frame, {}, &expected));
  size_t cases = 0;
  for (const char* setting : {static_cast<const char*>(nullptr), "0", "1", "invalid"})
    for (size_t cpu : {1, 2, 4, 8})
      for (int scenario = 0; scenario < 10; ++scenario) {
        if (setting) gjxl_test::SetEnvironment("GJXL_EARLY_DC", setting);
        else gjxl_test::SetEnvironment("GJXL_EARLY_DC", nullptr);
        gjxl_test::SetEnvironment("GJXL_EARLY_ENTROPY", nullptr);
        if (scenario == 9) gjxl_test::SetEnvironment("GJXL_EARLY_ENTROPY", "0");
        std::shared_ptr<const ExecutionDomain> domain;
        Check(ExecutionDomain::Create({0, cpu}, &domain));
        Provider provider;
        provider.failure = scenario >= 1 && scenario <= 4 ? scenario : 0;
        std::vector<uint8_t> bytes{19, 27};
        Status status;
        {
          CpuExecutionScope execution;
          if (scenario != 8) Check(execution.Start(domain, cpu));
          EncodeScope scope(cpu);
          std::optional<CpuWorkerGroup> occupied;
          if (scenario == 5) occupied.emplace(cpu);
          std::optional<EntropyReadinessBatchScope> batch;
          if (scenario == 7) batch.emplace();
          AcTokenizationProviderScope provider_scope(&provider);
          VarDctCodestreamProfile profile;
          status = EncodeVarDctCodestreamFromView(frame, {}, &bytes,
              scenario == 6 ? &profile : nullptr);
        }
        if (provider.failure) {
          Require(!status.ok() && bytes == std::vector<uint8_t>({19, 27}),
                  "Worker failure published partial output");
          Require(provider.finishes == (provider.failure <= 2 ? 0 : 1),
                  "Failed Begin incorrectly reached Finish");
        } else {
          Check(status);
          Require(bytes == expected && provider.begins == 1 && provider.finishes == 1,
                  "Earlier DC changed bytes or provider call count");
          const bool enabled = (!setting || std::strcmp(setting, "0") != 0) &&
              cpu > 1 && std::thread::hardware_concurrency() > 1 && scenario == 0;
          Require(provider.begin_cpu == (enabled ? 1 : cpu),
                  "Earlier DC activation or fallback occurred at the wrong boundary");
        }
        CheckReleased(*domain, cpu);
        ++cases;
      }

  gjxl_test::SetEnvironment("GJXL_EARLY_DC", nullptr);
  gjxl_test::SetEnvironment("GJXL_EARLY_ENTROPY", nullptr);
  for (const char* mode : {"0", "1", "2", "3", "invalid"}) {
    gjxl_test::SetEnvironment("GJXL_EXPERIMENT_EAGER_ENTROPY", mode);
    bool selected = true;
#ifdef GJXL_TOKENIZATION_EXPERIMENT
    selected = std::strcmp(mode, "3") == 0 || std::strcmp(mode, "invalid") == 0;
#endif
    Require(EarlierDcEnabled() == selected, "Experimental entropy precedence changed");
    std::shared_ptr<const ExecutionDomain> domain;
    Check(ExecutionDomain::Create({0, 4}, &domain));
    Provider provider;
    std::vector<uint8_t> bytes;
    {
      CpuExecutionScope execution;
      Check(execution.Start(domain, 4));
      EncodeScope scope(4);
      AcTokenizationProviderScope provider_scope(&provider);
      Check(EncodeVarDctCodestreamFromView(frame, {}, &bytes));
    }
    const bool admitted = selected && std::thread::hardware_concurrency() > 1;
    Require(bytes == expected && provider.begin_cpu == (admitted ? 1 : 4),
            "Diagnostic mode changed the early DC boundary or bytes");
    CheckReleased(*domain, 4);
    ++cases;
  }
  gjxl_test::SetEnvironment("GJXL_EXPERIMENT_EAGER_ENTROPY", nullptr);
  for (auto prediction : {VarDctDcPrediction::kGradient, VarDctDcPrediction::kWeighted})
    for (auto behavior : {VarDctEntropyBehavior::kBalanced,
                         VarDctEntropyBehavior::kHighDensity,
                         VarDctEntropyBehavior::kRateOptimized}) {
      const VarDctCodestreamOptions options{.entropy_behavior = behavior,
                                            .dc_prediction = prediction};
      std::vector<uint8_t> reference;
      Check(EncodeVarDctCodestreamFromView(frame, options, &reference));
      std::shared_ptr<const ExecutionDomain> domain;
      Check(ExecutionDomain::Create({0, 4}, &domain));
      Provider provider;
      std::vector<uint8_t> bytes;
      {
        CpuExecutionScope execution;
        Check(execution.Start(domain, 4));
        EncodeScope scope(4);
        AcTokenizationProviderScope provider_scope(&provider);
        Check(EncodeVarDctCodestreamFromView(frame, options, &bytes));
      }
      const bool admitted = behavior != VarDctEntropyBehavior::kRateOptimized &&
          std::thread::hardware_concurrency() > 1;
      Require(bytes == reference && provider.begin_cpu == (admitted ? 1 : 4),
              "DC predictor or entropy-policy bytes/fallback changed");
      CheckReleased(*domain, 4);
      ++cases;
    }
  // Fault before Begin must fail the earlier launch without publishing output.
  if (std::thread::hardware_concurrency() > 1)
  for (auto failure : {WorkerLaunchFailureKind::kSystemError,
                       WorkerLaunchFailureKind::kBadAlloc}) {
    std::shared_ptr<const ExecutionDomain> domain;
    Check(ExecutionDomain::Create({0, 4}, &domain));
    Provider provider;
    WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kSerializerSections, 0, failure};
    std::vector<uint8_t> bytes{31};
    Status status;
    {
      CpuExecutionScope execution;
      Check(execution.Start(domain, 4));
      EncodeScope scope(4);
      AcTokenizationProviderScope provider_scope(&provider);
      SetWorkerLaunchFaultForTesting(&fault);
      status = EncodeVarDctCodestreamFromView(frame, {}, &bytes);
      SetWorkerLaunchFaultForTesting(nullptr);
    }
    Require(fault.triggered && !status.ok() && bytes == std::vector<uint8_t>({31}) &&
            !provider.begins, "Early launch failure did not precede Begin");
    CheckReleased(*domain, 4);
    ++cases;
  }
  // Explicit nesting must retain the original start boundary.
  {
    std::shared_ptr<const ExecutionDomain> domain;
    Check(ExecutionDomain::Create({0, 4}, &domain));
    Provider provider;
    std::vector<uint8_t> bytes;
    {
      CpuExecutionScope execution;
      Check(execution.Start(domain, 4));
      EncodeScope scope(4);
      ParallelScope nested(4, nullptr, resource_budget_internal::CurrentResourceContext());
      AcTokenizationProviderScope provider_scope(&provider);
      Check(EncodeVarDctCodestreamFromView(frame, {}, &bytes));
    }
    Require(bytes == expected && provider.begin_cpu == 4, "Nested call launched DC early");
    CheckReleased(*domain, 4);
    ++cases;
  }
  for (size_t cpu : {2, 4, 8}) {
    std::shared_ptr<const ExecutionDomain> domain;
    Check(ExecutionDomain::Create({0, cpu}, &domain));
    std::atomic<bool> good{true};
    std::vector<std::jthread> callers;
    for (size_t i = 0; i < 4; ++i) callers.emplace_back([&] {
      for (size_t repeat = 0; repeat < 4; ++repeat) {
        CpuExecutionScope execution;
        if (!execution.Start(domain, 4).ok()) { good = false; return; }
        EncodeScope scope(4);
        Provider provider;
        AcTokenizationProviderScope provider_scope(&provider);
        std::vector<uint8_t> bytes;
        const VarDctCodestreamOptions options;
        if (!EncodeVarDctCodestreamFromView(frame, options, &bytes).ok() ||
            bytes != expected || provider.begins != 1 || provider.finishes != 1) good = false;
      }
    });
    callers.clear();
    Require(good, "Shared calls failed or changed bytes");
    CheckReleased(*domain, cpu);
    cases += 16;
  }
  gjxl_test::SetEnvironment("GJXL_EARLY_DC", nullptr);
  std::cout << "Passed " << cases << " earlier-DC cases: default/opt-out, admission before Begin, "
      "atomic Begin/Finish and launch failures, fallback, and shared-domain calls.\n";
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
