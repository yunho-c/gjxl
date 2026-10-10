// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/section_writing_internal.h"
#include "codestream/encoder_internal.h"
#include "codestream/serializer_storage_plan.h"
#include "codestream_frame_fixture.h"
#include "core/worker_launch_internal.h"
#include "environment_test_utils.h"
#include <array>
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::thread_budget_internal;
using namespace gjxl::resource_budget_internal;
using codestream_test_internal::FrameFixture;
using gjxl_test::SetEnvironment;

void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
void Empty(const ExecutionDomain& domain, size_t limit) {
  const auto s = domain.snapshot();
  Require(s.peak_cpu_protected_slots <= limit && !s.active_cpu_participants &&
      !s.reserved_cpu_workers && !s.suspended_cpu_workers &&
      !s.waiting_cpu_callers && !s.active_reservations, "CPU bound or cleanup");
}

void PolicyContracts() {
  SetEnvironment("GJXL_SECTION_WRITE_OVERLAP", nullptr);
  SetEnvironment("GJXL_EXPERIMENT_SECTION_WRITE", nullptr);
  Require(SectionWritingOverlapRequested(), "Production default is disabled");
  Require(!CanOverlapSectionWriting(4), "Unadmitted call enabled overlap");
  SetEnvironment("GJXL_EXPERIMENT_SECTION_WRITE", "0");
#ifdef GJXL_TOKENIZATION_EXPERIMENT
  Require(!SectionWritingOverlapRequested(), "Diagnostic opt-out ignored");
#else
  Require(SectionWritingOverlapRequested(), "Production read diagnostic control");
#endif
  SetEnvironment("GJXL_EXPERIMENT_SECTION_WRITE", "1");
  SetEnvironment("GJXL_SECTION_WRITE_OVERLAP", "0");
  Require(!SectionWritingOverlapRequested(), "Stable opt-out lost precedence");
  SetEnvironment("GJXL_SECTION_WRITE_OVERLAP", nullptr);
  SetEnvironment("GJXL_EXPERIMENT_SECTION_WRITE", nullptr);

  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({0, 4}, &domain));
  {
    CpuExecutionScope execution;
    Ok(execution.Start(domain, 4));
    EncodeScope threads(4);
    const bool parallel = std::thread::hardware_concurrency() > 1;
    Require(CanOverlapSectionWriting(4) == parallel,
            "Default single-image admission differs");
    Require(!CanOverlapSectionWriting(1), "One group enabled overlap");
    {
      EntropyReadinessBatchScope batch;
      Require(!CanOverlapSectionWriting(4), "Batch enabled overlap");
    }
    Require(CanOverlapSectionWriting(4) == parallel, "Batch scope leaked");
    {
      ParallelScope nested(4, nullptr, {});
      Require(!CanOverlapSectionWriting(4), "Nested work enabled overlap");
    }
    {
      EncodeScope one_cpu(1);
      Require(!CanOverlapSectionWriting(4), "CPU=1 enabled overlap");
    }
  }
  Empty(*domain, 4);
}

void DispatchContracts() {
  for (size_t cpu : {1, 2, 4, 8}) {
    std::shared_ptr<const ExecutionDomain> domain;
    Ok(ExecutionDomain::Create({0, cpu}, &domain));
    {
      CpuExecutionScope execution;
      Ok(execution.Start(domain, cpu));
      EncodeScope scope(cpu);
      std::array<std::atomic<unsigned>, 13> counts{};
      Ok(RunSectionWritingTasks(3, 10,
          [&](size_t i) { ++counts[i]; return Status::Ok(); },
          [&](size_t i) { ++counts[3 + i]; return Status::Ok(); }));
      for (const auto& n : counts) Require(n == 1, "Lost or repeated section");
      auto result = RunSectionWritingTasks(3, 10,
          [&](size_t i) { return i == 2 ? Status::InvalidArgument("DC first") : Status::Ok(); },
          [&](size_t) { return Status::Internal("AC second"); });
      Require(result.message() == "DC first", "Failure precedence changed");
      Require(!RunSectionWritingTasks(SIZE_MAX, 1,
          [](size_t) { return Status::Ok(); },
          [](size_t) { return Status::Ok(); }).ok(), "Task count overflow accepted");
      if (cpu >= 4 && std::thread::hardware_concurrency() >= 4) {
        for (auto kind : {WorkerLaunchFailureKind::kSystemError,
                          WorkerLaunchFailureKind::kBadAlloc}) {
          for (size_t before : {0, 1}) {
            WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kSerializerSections, before, kind};
            WorkerLaunchFaultScopeForTesting failure(&fault);
            std::atomic<size_t> active{0};
            const auto job = [&](size_t) { ++active; std::this_thread::yield(); --active; return Status::Ok(); };
            result = RunSectionWritingTasks(3, 10, job, job);
            Require(!result.ok() && fault.triggered && active == 0,
                    "Launch failure did not join workers");
          }
        }
      }
    }
    Empty(*domain, cpu);
  }
}

void EncodeBounded(const FrameFixture& frame, VarDctCodestreamOptions coding,
    size_t cpu, const std::shared_ptr<const ExecutionDomain>& domain,
    const std::vector<uint8_t>& expected, bool profiled = false, bool batch = false) {
  SerializerStoragePlan plan;
  Ok(ComputeSerializerStoragePlan(frame.view().geometry().frame(),
      {coding, cpu, profiled, false}, &plan));
  ResourceBudget budget(plan.working.peak_bytes);
  ResourceReservation reservation;
  Ok(budget.Reserve(plan.working.peak_bytes, &reservation));
  {
    ResourceContextScope resources({&reservation, ResourceClass::kSerializer});
    CpuExecutionScope execution;
    Ok(execution.Start(domain, cpu));
    EncodeScope threads(cpu);
    std::optional<EntropyReadinessBatchScope> batch_scope;
    if (batch) batch_scope.emplace();
    std::vector<uint8_t> actual{9, 7, 5};
    VarDctCodestreamProfile profile;
    Ok(EncodeVarDctCodestreamFromView(frame.view(), coding, &actual,
        profiled ? &profile : nullptr));
    Require(actual == expected, "Section scheduling changed encoded bytes");
    Require(budget.snapshot().peak_backing_bytes <= plan.working.peak_bytes,
            "Mixed writer scratch exceeded plan");
  }
  reservation.Reset();
  Require(budget.snapshot().committed_bytes() == 0, "Serializer backing leaked");
}

int main() try {
  PolicyContracts();
  DispatchContracts();
  size_t cases = 0;
  for (Extent2D blocks : {Extent2D{33, 33}, {257, 3}}) {
    FrameFixture frame;
    Require(frame.Create(blocks, 0, 2), "Fixture creation failed");
    for (auto entropy : {VarDctEntropyBehavior::kBalanced,
                         VarDctEntropyBehavior::kHighDensity,
                         VarDctEntropyBehavior::kRateOptimized,
                         VarDctEntropyBehavior::kMaximumCompression}) {
      VarDctCodestreamOptions coding;
      coding.entropy_behavior = entropy;
      std::vector<uint8_t> expected;
      { EncodeScope serial(1); Ok(EncodeVarDctCodestreamFromView(frame.view(), coding, &expected)); }
      {
        std::shared_ptr<const ExecutionDomain> domain;
        Ok(ExecutionDomain::Create({0, 4}, &domain));
        SerializerStoragePlan enabled, disabled;
        Ok(ComputeSerializerStoragePlan(frame.view().geometry().frame(),
            {coding, 4, false, false}, &enabled));
        SetEnvironment("GJXL_SECTION_WRITE_OVERLAP", "0");
        Ok(ComputeSerializerStoragePlan(frame.view().geometry().frame(),
            {coding, 4, false, false}, &disabled));
        Require(enabled.working.peak_bytes == disabled.working.peak_bytes,
                "Opt-out changed admitted storage plan");
        EncodeBounded(frame, coding, 4, domain, expected);
        SetEnvironment("GJXL_SECTION_WRITE_OVERLAP", nullptr);
        Empty(*domain, 4);
        ++cases;
      }
      for (size_t cpu : {1, 2, 4, 8}) {
        std::shared_ptr<const ExecutionDomain> domain;
        Ok(ExecutionDomain::Create({0, cpu}, &domain));
        EncodeBounded(frame, coding, cpu, domain, expected);
        EncodeBounded(frame, coding, cpu, domain, expected, true);
        Empty(*domain, cpu);
        cases += 2;
      }
      {
        std::shared_ptr<const ExecutionDomain> domain;
        Ok(ExecutionDomain::Create({0, 4}, &domain));
        EncodeBounded(frame, coding, 4, domain, expected, false, true);
        Empty(*domain, 4);
        ++cases;
      }
      if (entropy == VarDctEntropyBehavior::kBalanced) {
        std::shared_ptr<const ExecutionDomain> domain;
        Ok(ExecutionDomain::Create({0, 4}, &domain));
        std::atomic<bool> success{true};
        std::vector<std::jthread> callers;
        for (int i = 0; i < 4; ++i) callers.emplace_back([&] {
          try { EncodeBounded(frame, coding, 4, domain, expected); }
          catch (...) { success = false; }
        });
        callers.clear();
        Require(success, "Concurrent encodes failed");
        Empty(*domain, 4);
        cases += 4;
      }
    }
  }
  std::cout << "Verified section dispatch and " << cases
            << " exact-byte encodes: planned memory, CPU limits, profiles, "
               "shared callers, and launch-failure cleanup.\n";
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
