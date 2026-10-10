// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include "codestream/entropy_readiness_internal.h"
#include "codestream/parallel_sections_internal.h"

namespace {
using namespace gjxl;
using namespace gjxl::thread_budget_internal;
using namespace gjxl::codestream_internal;
void Require(bool condition, const char* message) {
  if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}
template<class Predicate> void Until(Predicate predicate) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!predicate()) {
    Require(std::chrono::steady_clock::now() < end, "Readiness worker timed out");
    std::this_thread::yield();
  }
}
auto Domain(size_t limit) {
  std::shared_ptr<const ExecutionDomain> domain;
  Require(ExecutionDomain::Create({0, limit}, &domain).ok(), "Domain creation failed");
  return domain;
}
void Empty(const ExecutionDomain& domain) {
  const auto s = domain.snapshot();
  Require(s.active_cpu_participants == 0 && s.reserved_cpu_workers == 0 &&
          s.suspended_cpu_workers == 0 && s.waiting_cpu_callers == 0 &&
          s.peak_cpu_protected_slots <= s.effective_cpu_participant_limit,
          "Readiness admission leaked or exceeded capacity");
}
void Selection() {
  for (size_t total : {1, 2, 3, 4, 8})
    for (size_t local : {0, 1, 2, 4, 8})
      for (size_t dc = 1; dc <= 8; ++dc) {
        auto domain = Domain(total);
        {
          CpuExecutionScope root;
          Require(root.Start(domain, local).ok(), "Root admission failed");
          EncodeScope encode(local);
          EntropyReadinessAdmission grant(dc, 8);
          const size_t effective = local ? std::min(local, total) : total;
          Require(grant.admitted() == (dc < 8 && dc + 1 <= effective),
                  "Policy ignored per-image/global capacity or stage ceiling");
          if (grant.admitted()) {
            Require(grant.branches().participants() == 2 && grant.dc().participants() == dc,
                    "DC capacity was not preserved by the split");
          } else {
            Require(domain->snapshot().reserved_cpu_workers == 0,
                    "Declined admission retained a partial reservation");
            CpuWorkerGroup ordinary(effective);
            Require(ordinary.participants() == effective, "Fallback lost ordinary capacity");
          }
        }
        Empty(*domain);
      }
}
void FallbackContexts() {
  EntropyReadinessAdmission unadmitted(1, 8);
  Require(!unadmitted.admitted(), "Legacy unadmitted call enabled readiness");
  auto domain = Domain(4);
  {
    CpuExecutionScope root;
    Require(root.Start(domain, 4).ok(), "Root admission failed");
    EncodeScope encode(4);
    {
      CpuWorkerGroup held(3);
      EntropyReadinessAdmission denied(2, 4);
      Require(!denied.admitted() && domain->snapshot().reserved_cpu_workers == 2,
              "Shared contention leaked a partial grant");
    }
    {
      EntropyReadinessBatchScope batch;
      { EntropyReadinessBatchScope nested; }
      EntropyReadinessAdmission denied(1, 4);
      Require(!denied.admitted(), "Batch scope or nested restoration enabled readiness");
    }
    {
      ParallelScope nested(4, nullptr, {});
      EntropyReadinessAdmission denied(1, 4);
      Require(!denied.admitted(), "Explicit nested work enabled readiness");
    }
    EntropyReadinessAdmission granted(1, 4);
    Require(granted.admitted(), "Fallback context leaked into the next call");
  }
  Empty(*domain);
}
void ProtectedDcCapacity() {
  const size_t dc = std::min<size_t>(3, std::max(1u, std::thread::hardware_concurrency()));
  auto domain = Domain(dc + 1);
  {
    CpuExecutionScope root;
    Require(root.Start(domain, dc + 1).ok(), "Root admission failed");
    EncodeScope encode(dc + 1);
    EntropyReadinessAdmission grant(dc, dc + 1);
    Require(grant.admitted(), "Full reservation failed");
    auto shared = current_cpu_execution.domain;
    std::thread contender([&] {
      Require(shared.TryReserveWorkers(4).count() == 0,
              "Other work stole DC slots between branch and DC launch");
    });
    contender.join();
    std::atomic<size_t> dc_entered{0};
    std::atomic<bool> ac_entered{false};
    constexpr ParallelWorkErrors errors{"alloc", "worker", StatusCode::kOutOfMemory,
      "length", "launch allocation", LaunchFailureAction::kReturnError, "launch"};
    const auto status = RunParallelWork<Storage>(2, grant.branches(), 1,
        WorkerLaunchSite::kSerializerSections, errors, [&](size_t branch, size_t) {
      EncodeScope branch_scope(branch == 0 ? dc : 1);
      if (branch == 1) {
        ac_entered = true;
        Until([&] { return dc_entered == dc; });
        return Status::Ok();
      }
      return RunParallelSections(dc, [&](size_t) {
        ++dc_entered;
        Until([&] { return dc_entered == dc && ac_entered; });
        return Status::Ok();
      }, dc, &grant.dc());
    });
    Require(status.ok() && dc_entered == dc && ac_entered,
            "DC and AC did not execute with all admitted participants");
    Require(domain->snapshot().peak_cpu_protected_slots == dc + 1, "Incorrect protected peak");
  }
  Empty(*domain);
}
void PartialLaunchCleanup() {
  const size_t dc = std::min<size_t>(3, std::max(1u, std::thread::hardware_concurrency()));
  if (dc < 2) return;
  for (auto kind : {WorkerLaunchFailureKind::kSystemError, WorkerLaunchFailureKind::kBadAlloc}) {
    auto domain = Domain(4);
    {
      CpuExecutionScope root;
      Require(root.Start(domain, 4).ok(), "Root admission failed");
      EncodeScope encode(4);
      {
        EntropyReadinessAdmission grant(dc, 4);
        WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kSerializerSections, dc - 2, kind};
        SetWorkerLaunchFaultForTesting(&fault);
        const auto status = RunParallelSections(dc, [](size_t) { return Status::Ok(); }, dc, &grant.dc());
        SetWorkerLaunchFaultForTesting(nullptr);
        Require(!status.ok() && fault.triggered, "Partial DC launch failure was not exercised");
      }
      Require(domain->snapshot().reserved_cpu_workers == 0, "Split launch failure leaked pending slots");
      CpuWorkerGroup retry(4);
      Require(retry.participants() == 4, "Launch failure prevented full capacity reuse");
    }
    Empty(*domain);
  }
}
}  // namespace
int main() {
  Selection();
  FallbackContexts();
  ProtectedDcCapacity();
  PartialLaunchCleanup();
  std::cout << "Readiness admission: 200 capacity cases, fallback contexts, protected DC launch, partial failure cleanup passed\n";
}
