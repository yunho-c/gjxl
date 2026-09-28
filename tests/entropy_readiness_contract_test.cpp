// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Exercise publication, worker failures, budget fallback and shared callers.
#include "codestream/ac_tokenization_provider_internal.h"
#include "codestream/encoder_internal.h"
#include "codestream/entropy_readiness_internal.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"
#include "quantized_frame_fixture.h"
#include <atomic>
#include <iostream>
#include <optional>
#include <thread>
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::thread_budget_internal;
using gjxl_test::Check;
void Require(bool v, const char* message) { if (!v) throw std::runtime_error(message); }
struct Provider final : AcTokenizationProvider {
  Storage<SimpleAcGroupTokenData> groups;
  Storage<EntropyTokenStreamView> views;
  Storage<PreparedFixedAnsCluster> counts;
  WorkerLaunchFaultForTesting* launch_fault = nullptr;
  int failure = 0;
  std::atomic<unsigned> finishes{0};
  size_t finish_cpu_limit = 0;
  Status Begin(const vardct_frame_internal::VarDctFrameView& frame,
      const SimpleCoefficientOrders& orders, const SimpleAcNaturalOrders& natural,
      const SimpleBlockContextMap& map, bool populations) override {
    groups.resize(frame.ac_group_count());
    SimpleAcTokenizationScratch scratch;
    for (size_t i = 0; i < groups.size(); ++i) {
      auto status = TokenizeSimpleAcGroupForEncoder(frame, orders, natural, map,
          i, false, &scratch, &groups[i]);
      if (!status.ok()) return status;
      views.push_back(EntropyTokenStreamView::Split(groups[i].values, groups[i].contexts));
    }
    if (populations) {
      auto status = CollectDefaultEntropyPopulations(views, map.ac_context_count(), &counts);
      if (!status.ok()) return status;
    }
    if (launch_fault) SetWorkerLaunchFaultForTesting(launch_fault);
    return Status::Ok();
  }
  Status Finish(Storage<EntropyTokenStreamView>* streams,
      Storage<PreparedFixedAnsCluster>* populations) override {
    ++finishes;
    finish_cpu_limit = CpuThreadCount();
    if (failure == 1) return Status::DeviceError("Injected completion failure");
    if (failure == 2) throw std::bad_alloc();
    *streams = views;
    *populations = counts;
    return Status::Ok();
  }
};
int main() try {
  // Exercise the original early-entropy boundary independently of earlier DC.
  setenv("GJXL_EARLY_DC", "0", 1);
  auto owner = gjxl_test::MakeFrame(gjxl_test::kStrategies.size(), 2, 36);
  const auto frame = vardct_frame_internal::BorrowFrame(owner);
  std::vector<uint8_t> expected;
  Check(EncodeVarDctCodestreamFromView(frame, {}, &expected));
  size_t cases = 0;
  unsetenv("GJXL_EARLY_ENTROPY");
  unsetenv("GJXL_EXPERIMENT_EAGER_ENTROPY");
  for (const char* mode : {static_cast<const char*>(nullptr), "disabled", "0", "1", "2", "3"})
    for (size_t cpu : {1, 2, 4, 8})
      for (int scenario = 0; scenario < 10; ++scenario) {
        const bool disabled = mode && std::strcmp(mode, "disabled") == 0;
        unsetenv("GJXL_EARLY_ENTROPY");
        unsetenv("GJXL_EXPERIMENT_EAGER_ENTROPY");
        if (disabled) setenv("GJXL_EARLY_ENTROPY", "0", 1);
        else if (mode) setenv("GJXL_EXPERIMENT_EAGER_ENTROPY", mode, 1);
        bool early_requested = !disabled;
#ifdef GJXL_TOKENIZATION_EXPERIMENT
        if (mode && std::strcmp(mode, "0") == 0) early_requested = false;
#endif
        std::shared_ptr<const ExecutionDomain> domain;
        Check(ExecutionDomain::Create({0, cpu}, &domain));
        Provider provider;
        provider.failure = scenario == 1 ? 1 : scenario == 2 ? 2 : 0;
        WorkerLaunchFaultForTesting fault{
            WorkerLaunchSite::kSerializerSections, 0,
            scenario == 3 ? WorkerLaunchFailureKind::kSystemError : WorkerLaunchFailureKind::kBadAlloc};
        const bool inject_launch = (scenario == 3 || scenario == 4) && cpu > 1 &&
            std::thread::hardware_concurrency() > 1;
        if (inject_launch) provider.launch_fault = &fault;
        std::vector<uint8_t> output{9, 7, 5};
        Status status;
        {
          CpuExecutionScope execution;
          if (scenario != 8) Check(execution.Start(domain, cpu));
          EncodeScope scope(cpu);
          std::optional<CpuWorkerGroup> occupied;
          if (scenario == 5) occupied.emplace(cpu);
          std::optional<EntropyReadinessBatchScope> batch_scope;
          if (scenario == 7) batch_scope.emplace();
          std::optional<ParallelScope> nested_scope;
          if (scenario == 9) nested_scope.emplace(cpu, nullptr,
              gjxl::resource_budget_internal::CurrentResourceContext());
          AcTokenizationProviderScope provider_scope(&provider);
          VarDctCodestreamProfile profile;
          status = EncodeVarDctCodestreamFromView(frame, {}, &output,
              scenario == 6 ? &profile : nullptr);
          SetWorkerLaunchFaultForTesting(nullptr);
        }
        if (provider.failure || inject_launch) {
          Require(!status.ok(), "Injected error was not returned");
          Require(output == std::vector<uint8_t>({9,7,5}), "Failed encode published output");
          if (inject_launch) Require(fault.triggered, "Launch fault not reached");
        } else {
          Check(status);
          Require(output == expected, "Scheduling changed encoded bytes");
          Require(provider.finishes == 1, "Finish must execute exactly once");
          if (cpu > 1) {
            const bool early = early_requested && scenario == 0 &&
                std::thread::hardware_concurrency() > 1;
            Require(provider.finish_cpu_limit == (early ? 1 : cpu),
                "Default/override selection or batch/profile/budget/nested fallback changed");
          }
        }
        auto snap = domain->snapshot();
        Require(snap.peak_cpu_protected_slots <= cpu &&
            snap.active_cpu_participants == 0 && snap.reserved_cpu_workers == 0 &&
            snap.active_reservations == 0, "Budget exceeded or workers leaked");
        ++cases;
      }
  unsetenv("GJXL_EARLY_ENTROPY");
  unsetenv("GJXL_EXPERIMENT_EAGER_ENTROPY");
  for (size_t limit : {2, 4, 8}) {
    std::shared_ptr<const ExecutionDomain> domain;
    Check(ExecutionDomain::Create({0, limit}, &domain));
    std::atomic<bool> good{true};
    std::vector<std::jthread> workers;
    for (size_t worker = 0; worker < 4; ++worker) workers.emplace_back([&] {
      for (size_t repeat = 0; repeat < 4; ++repeat) {
        Provider provider;
        std::vector<uint8_t> output;
        CpuExecutionScope execution;
        if (!execution.Start(domain, 4).ok()) { good = false; return; }
        EncodeScope scope(4);
        AcTokenizationProviderScope provider_scope(&provider);
        const auto result = EncodeVarDctCodestreamFromView(frame, {}, &output);
        if (!result.ok() || output != expected || provider.finishes != 1) good = false;
      }
    });
    workers.clear();
    Require(good, "Concurrent shared-domain encoding changed bytes or failed");
    const auto snap = domain->snapshot();
    Require(snap.peak_cpu_protected_slots <= limit && !snap.active_cpu_participants &&
        !snap.reserved_cpu_workers && !snap.suspended_cpu_workers && !snap.active_reservations,
        "Concurrent encoding exceeded capacity or leaked resources");
    cases += 16;
  }
  std::cout << "Passed " << cases << " scheduling contract cases: byte parity, atomic errors, "
      "completion/launch allocation failures, CPU 1/2/4/8, production default and disable override, "
      "batch/profile/nested/unadmitted/budget fallback, and concurrent shared-domain calls.\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
