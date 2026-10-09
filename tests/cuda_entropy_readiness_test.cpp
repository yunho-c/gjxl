// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/encoder_internal.h"
#include "codestream/entropy_readiness_internal.h"
#include "core/worker_launch_internal.h"
#include "gpu/cuda/cuda_ac_tokenization.h"
#include "gpu/cuda/cuda_backend.h"
#include "gpu/cuda/cuda_backend_internal.h"
#include "environment_test_utils.h"
#include "quantized_frame_fixture.h"

#include <array>
#include <barrier>
#include <iostream>
#include <optional>
#include <thread>

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::cuda_internal;
using namespace gjxl::thread_budget_internal;
using gjxl_test::Check;
using gjxl_test::SetEnvironment;
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

// Observe the actual CUDA provider through the shared serializer. A ceiling of
// one at Begin distinguishes the earlier DC split; one at Finish alone proves
// the later early-entropy split. No sleeps or GPU timing assumptions are used.
struct ObservedProvider final : AcTokenizationProvider {
  GpuBackend& backend;
  std::unique_ptr<AcTokenizationProvider> inner;
  size_t begins = 0, finishes = 0, begin_cpu = 0, finish_cpu = 0;
  int failure = 0;
  ObservedProvider(GpuBackend& gpu, const DeviceBuffer& coefficients)
      : backend(gpu) {
    Check(CreateCudaAcTokenizationProvider(gpu, coefficients, 0, &inner));
  }
  bool SupportsEarlyEntropy() const noexcept override {
    return inner->SupportsEarlyEntropy();
  }
  Status Begin(const vardct_frame_internal::VarDctFrameView& frame,
               const SimpleCoefficientOrders& orders,
               const SimpleAcNaturalOrders& natural,
               const SimpleBlockContextMap& map, bool populations) override {
    ++begins;
    begin_cpu = CpuThreadCount();
    if (failure == 1 || failure == 2)
      Check(ArmNextCudaSubmissionFailureForTest(backend, failure == 1,
                                               failure == 2));
    return inner->Begin(frame, orders, natural, map, populations);
  }
  Status Finish(Storage<EntropyTokenStreamView>* streams,
                Storage<PreparedFixedAnsCluster>* populations) override {
    ++finishes;
    finish_cpu = CpuThreadCount();
    if (failure == 3)
      resource_budget_internal::ArmManagedHostClassAllocationFailureAfterForTest(
          resource_budget_internal::ResourceClass::kSerializer, 0);
    auto status = inner->Finish(streams, populations);
    if (failure == 3) {
      Require(!resource_budget_internal::ManagedHostAllocationFailurePendingForTest(),
              "Finish did not consume injected allocation failure");
      resource_budget_internal::DisarmManagedHostAllocationFailureForTest();
    }
    return status;
  }
};

std::unique_ptr<DeviceBuffer> Upload(GpuBackend& gpu,
                                    const VarDctEncoderFrame& frame) {
  std::vector<int32_t> packed;
  for (size_t g = 0; g < frame.ac_group_count(); ++g) {
    VarDctNativeAcGroupView group;
    Check(frame.GetNativeAcGroup(g, &group));
    std::visit([&](const auto& native) {
      for (const auto channel : native.coefficients)
        for (size_t i = 0; i < native.used_coefficient_count; ++i)
          packed.push_back(channel[i]);
    }, group);
  }
  std::unique_ptr<DeviceBuffer> result;
  Check(gpu.Allocate(packed.size() * sizeof(int32_t), &result));
  Check(gpu.CopyHostToDevice(*result, packed.data(),
                             packed.size() * sizeof(int32_t), 0));
  return result;
}

void Released(const ExecutionDomain& domain, size_t cpu) {
  const auto s = domain.snapshot();
  Require(s.peak_cpu_protected_slots <= cpu && !s.active_cpu_participants &&
              !s.reserved_cpu_workers && !s.suspended_cpu_workers &&
              !s.waiting_cpu_callers && !s.active_reservations,
          "CUDA readiness exceeded CPU capacity or leaked participants");
}

void Matrix(bool smoke) {
  std::unique_ptr<GpuBackend> gpu;
  Check(CreateCudaBackend(&gpu));
  auto owner = gjxl_test::MakeFrame(gjxl_test::kStrategies.size(), 2, 36);
  auto frame = vardct_frame_internal::BorrowFrame(owner);
  auto coefficients = Upload(*gpu, owner);
  size_t cases = 0;
  for (auto behavior : {VarDctEntropyBehavior::kBalanced,
                       VarDctEntropyBehavior::kHighDensity,
                       VarDctEntropyBehavior::kRateOptimized}) {
    if (smoke && behavior != VarDctEntropyBehavior::kBalanced) continue;
    VarDctCodestreamOptions coding{.entropy_behavior = behavior};
    std::vector<uint8_t> reference;
    Check(EncodeVarDctCodestreamFromView(frame, coding, &reference));
    for (int schedule = 0; schedule < 3; ++schedule) {
      SetEnvironment("GJXL_EARLY_ENTROPY", schedule == 0 ? "0" : nullptr);
      SetEnvironment("GJXL_EARLY_DC", schedule == 1 ? "0" : nullptr);
      for (size_t cpu : {1, 2, 4, 8}) {
        if (smoke && cpu != 4) continue;
        // Normal, batch, detailed profile, occupied budget, nested, no CPU
        // participation, submission/completion/allocation failure, launch fault.
        for (int scenario = 0; scenario < 10; ++scenario) {
          std::shared_ptr<const ExecutionDomain> domain;
          Check(ExecutionDomain::Create({0, cpu}, &domain));
          ObservedProvider provider(*gpu, *coefficients);
          provider.failure = scenario >= 6 && scenario <= 8 ? scenario - 5 : 0;
          std::vector<uint8_t> bytes{17, 31};
          Status status;
          WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kSerializerSections};
          const bool early = schedule > 0 && cpu > 1 &&
              behavior != VarDctEntropyBehavior::kRateOptimized &&
              std::thread::hardware_concurrency() > 1;
          const bool inject_launch = scenario == 9 && early && schedule == 2;
          {
            CpuExecutionScope execution;
            if (scenario != 5) Check(execution.Start(domain, cpu));
            EncodeScope scope(cpu);
            std::optional<EntropyReadinessBatchScope> batch;
            if (scenario == 1) batch.emplace();
            std::optional<CpuWorkerGroup> occupied;
            if (scenario == 3) occupied.emplace(cpu);
            std::optional<ParallelScope> nested;
            if (scenario == 4)
              nested.emplace(cpu, nullptr,
                  resource_budget_internal::CurrentResourceContext());
            AcTokenizationProviderScope active(&provider);
            WorkerLaunchFaultScopeForTesting launch(inject_launch ? &fault : nullptr);
            VarDctCodestreamProfile profile;
            status = EncodeVarDctCodestreamFromView(frame, coding, &bytes,
                                                    scenario == 2 ? &profile : nullptr);
          }
          if (provider.failure || inject_launch) {
            Require(!status.ok() && bytes == std::vector<uint8_t>({17, 31}),
                    "CUDA scheduling failure published output or succeeded");
            if (inject_launch)
              Require(fault.triggered && !provider.begins,
                      "Early launch failure did not precede GPU Begin");
          } else {
            Check(status);
            Require(bytes == reference && provider.begins == 1 && provider.finishes == 1,
                    "CUDA schedule changed bytes or provider lifetime");
            const bool admitted = early && (scenario == 0 || scenario == 9);
            Require(provider.begin_cpu == (admitted && schedule == 2 ? 1 : cpu) &&
                        provider.finish_cpu == (admitted ? 1 : cpu),
                    "CUDA readiness activation/fallback used the wrong boundary");
          }
          Released(*domain, cpu);
          ++cases;
        }
      }
    }
  }
  SetEnvironment("GJXL_EARLY_ENTROPY", nullptr);
  SetEnvironment("GJXL_EARLY_DC", nullptr);
  std::cout << "Verified " << cases << " CUDA readiness schedule, fallback, "
      "failure/publication and CPU-budget cases.\n";
}

void SharedCallers() {
  auto owner = gjxl_test::MakeFrame(gjxl_test::kStrategies.size(), 2, 36);
  const auto frame = vardct_frame_internal::BorrowFrame(owner);
  std::vector<uint8_t> reference;
  Check(EncodeVarDctCodestreamFromView(frame, {}, &reference));
  std::shared_ptr<const ExecutionDomain> domain;
  Check(ExecutionDomain::Create({0, 4}, &domain));
  std::barrier gate(2);
  std::array<std::exception_ptr, 2> errors{};
  std::array<std::unique_ptr<GpuBackend>, 2> gpus;
  std::array<std::unique_ptr<DeviceBuffer>, 2> coefficients;
  for (size_t i = 0; i < 2; ++i) {
    Check(CreateCudaBackend(&gpus[i]));
    coefficients[i] = Upload(*gpus[i], owner);
  }
  const auto run = [&](size_t i) {
    try {
      for (size_t repeat = 0; repeat < 6; ++repeat) {
        gate.arrive_and_wait();
        ObservedProvider provider(*gpus[i], *coefficients[i]);
        std::vector<uint8_t> bytes;
        CpuExecutionScope execution;
        Check(execution.Start(domain, 4));
        EncodeScope scope(4);
        AcTokenizationProviderScope active(&provider);
        Check(EncodeVarDctCodestreamFromView(frame, {}, &bytes));
        Require(bytes == reference, "Shared-domain CUDA caller changed bytes");
      }
    } catch (...) {
      errors[i] = std::current_exception();
      gate.arrive_and_drop();
    }
  };
  { std::jthread a(run, 0), b(run, 1); }
  for (auto error : errors) if (error) std::rethrow_exception(error);
  Released(*domain, 4);
  std::cout << "Verified 12 concurrent CUDA calls sharing four CPU participants.\n";
}

void TransferredProvider() {
  std::unique_ptr<GpuBackend> gpu;
  Check(CreateCudaBackend(&gpu));
  auto owner = gjxl_test::MakeFrame(gjxl_test::kStrategies.size(), 2, 36);
  const auto frame = vardct_frame_internal::BorrowFrame(owner);
  auto coefficients = Upload(*gpu, owner);
  ObservedProvider provider(*gpu, *coefficients);
  const auto before = cuda_token_provider_begin_count;
  std::exception_ptr error;
  std::vector<uint8_t> bytes, expected;
  Check(EncodeVarDctCodestreamFromView(frame, {}, &expected));
  // Always execute on another thread, even if the dispatcher normally assigns
  // the AC branch to its caller. Observations must stay with the creating caller.
  {
    std::jthread worker([&] {
      try {
        CpuExecutionScope execution;
        Check(execution.Start({}, 4));
        EncodeScope scope(4);
        AcTokenizationProviderScope active(&provider);
        Check(EncodeVarDctCodestreamFromView(frame, {}, &bytes));
      } catch (...) { error = std::current_exception(); }
    });
  }
  if (error) std::rethrow_exception(error);
  Require(bytes == expected && cuda_token_provider_begin_count == before + 1,
          "Transferred CUDA provider lost output or caller observations");
  std::cout << "Verified transferred-provider lifetime and caller observations.\n";
}
}  // namespace

int main(int argc, char** argv) try {
  const bool smoke = argc == 2 && std::string_view(argv[1]) == "--smoke";
  Require(argc == 1 || smoke, "Unknown arguments");
  SetEnvironment("GJXL_EXPERIMENT_EAGER_ENTROPY", nullptr);
  Matrix(smoke);
  SharedCallers();
  TransferredProvider();
  std::cout << std::flush;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
