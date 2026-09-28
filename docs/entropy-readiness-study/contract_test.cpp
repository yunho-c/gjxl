// Isolated experiment: exercise publication, worker failures, and budget fallback.
#include "codestream/ac_tokenization_provider_internal.h"
#include "codestream/encoder_internal.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"
#include "quantized_frame_fixture.h"
#include <atomic>
#include <iostream>
#include <optional>
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
    if (failure == 1) return Status::DeviceError("Injected completion failure");
    if (failure == 2) throw std::bad_alloc();
    *streams = views;
    *populations = counts;
    return Status::Ok();
  }
};
int main() try {
  auto owner = gjxl_test::MakeFrame(gjxl_test::kStrategies.size(), 2, 36);
  const auto frame = vardct_frame_internal::BorrowFrame(owner);
  std::vector<uint8_t> expected;
  Check(EncodeVarDctCodestreamFromView(frame, {}, &expected));
  size_t cases = 0;
  for (const char* mode : {"0", "1", "2"})
    for (size_t cpu : {1, 2, 4, 8})
      for (int scenario = 0; scenario < 7; ++scenario) {
        setenv("GJXL_EXPERIMENT_EAGER_ENTROPY", mode, 1);
        std::shared_ptr<const ExecutionDomain> domain;
        Check(ExecutionDomain::Create({0, cpu}, &domain));
        Provider provider;
        provider.failure = scenario == 1 ? 1 : scenario == 2 ? 2 : 0;
        WorkerLaunchFaultForTesting fault{
            WorkerLaunchSite::kSerializerSections, 0,
            scenario == 3 ? WorkerLaunchFailureKind::kSystemError : WorkerLaunchFailureKind::kBadAlloc};
        const bool inject_launch = (scenario == 3 || scenario == 4) && cpu > 1;
        if (inject_launch) provider.launch_fault = &fault;
        std::vector<uint8_t> output{9, 7, 5};
        Status status;
        {
          CpuExecutionScope execution;
          Check(execution.Start(domain, cpu));
          EncodeScope scope(cpu);
          std::optional<CpuWorkerGroup> occupied;
          if (scenario == 5) occupied.emplace(cpu);
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
        }
        auto snap = domain->snapshot();
        Require(snap.peak_cpu_protected_slots <= cpu &&
            snap.active_cpu_participants == 0 && snap.reserved_cpu_workers == 0 &&
            snap.active_reservations == 0, "Budget exceeded or workers leaked");
        ++cases;
      }
  std::cout << "Passed " << cases << " scheduling contract cases: byte parity, atomic errors, "
      "completion/launch allocation failures, CPU 1/2/4/8, admission fallback, native profile fallback.\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
