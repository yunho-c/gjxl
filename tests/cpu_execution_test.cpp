// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "core/thread_budget.h"

namespace {
using namespace gjxl;
using namespace gjxl::thread_budget_internal;
void Require(bool good, const char* text) {
  if (!good) { std::cerr << text << '\n'; std::exit(EXIT_FAILURE); }
}
template<class Predicate> void Until(Predicate predicate, const char* text) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!predicate()) {
    Require(std::chrono::steady_clock::now() < deadline, text);
    std::this_thread::yield();
  }
}
std::shared_ptr<const ExecutionDomain> Domain(size_t limit) {
  std::shared_ptr<const ExecutionDomain> domain;
  Require(ExecutionDomain::Create({0, limit}, &domain).ok(), "Domain creation failed");
  return domain;
}
void Empty(const ExecutionDomain& domain) {
  const auto s = domain.snapshot();
  Require(s.active_cpu_participants == 0 && s.reserved_cpu_workers == 0 && s.suspended_cpu_workers == 0 &&
          s.waiting_cpu_callers == 0 && s.peak_cpu_protected_slots <= s.effective_cpu_participant_limit,
          "CPU execution leaked or exceeded capacity");
}

void IdentityAndObservers() {
  auto domain = Domain(4), other = Domain(4);
  auto retained = domain;
  Require(!ExecutionDomain::Create({0, 257}, &domain).ok() && domain == retained,
          "Invalid domain CPU limit changed output");
  CpuParticipantTracker tracker;
  {
    CpuExecutionScope root;
    Require(root.Start(domain, 2).ok() && HasCpuParticipation(), "Root participation failed");
    Require(!root.Start(domain).ok(), "Execution scope accepted a second Start");
    resource_budget_internal::ResourceContext resources;
    resources.domain = domain.get();
    resource_budget_internal::ResourceContextScope resource_scope(resources);
    CpuExecutionScope borrowed, wrong;
    Require(borrowed.Start().ok() && !wrong.Start(other).ok() &&
            domain->snapshot().active_cpu_participants == 1,
            "Nested domain inheritance double counted or changed domain");
    EncodeScope encode(0, &tracker);
    Require(tracker.active() == 1, "Sequential calling-thread work was not tracked");
    {
      EncodeScope same_tracker(0, &tracker);
      ParallelScope first(0, &tracker, resources), second(0, &tracker, resources);
      Require(tracker.active() == 1, "Nested scopes double counted their caller");
    }
    CpuWorkerGroup group(20);
    Require(group.enabled() && group.participants() == 2 &&
            domain->snapshot().reserved_cpu_workers == 1, "Per-image quota was not reserved");
    std::atomic<bool> entered{false}, release{false};
    std::vector<std::thread> workers;
    workers.emplace_back([&] {
      ParallelScope scope(0, &tracker, resources, &group);
      CpuWorkerGroup nested(20);
      Require(nested.participants() == 1, "Automatic nested work exceeded the per-image quota");
      Require(HasCpuParticipation(), "Worker did not inherit a CPU ticket");
      entered = true;
      Until([&] { return release.load(); }, "Worker was not released");
    });
    Until([&] { return entered.load(); }, "Worker failed to activate");
    Require(tracker.active() == 2 && tracker.peak() == 2 &&
            domain->snapshot().active_cpu_participants == 2, "Worker activation accounting failed");
    {
      ParallelScope caller(0, &tracker, resources, &group);
      Require(tracker.active() == 2, "Participating caller consumed an extra worker slot");
    }
    release = true;
    JoinCpuWorkers(workers);
    Require(tracker.active() == 1, "Worker join did not restore caller tracking");
  }
  Require(tracker.active() == 0 && !HasCpuParticipation(), "Execution TLS/observer was not restored");
  Empty(*domain);
  // Legacy component calls do not require a workflow domain or change their
  // existing observer behavior.
  {
    EncodeScope encode(1, &tracker);
    ParallelScope legacy(1, &tracker, {});
    Require(tracker.active() == 1 && !HasCpuParticipation(), "Legacy component scope changed");
  }
  Require(tracker.active() == 0, "Legacy participant tracker leaked");
}

void SuspensionAndFifoResume() {
  auto domain = Domain(1);
  CpuParticipantTracker tracker;
  {
    CpuExecutionScope root;
    Require(root.Start(domain, 1, true).ok(), "Root start failed");
    EncodeScope encode(1, &tracker);
    std::atomic<bool> entered{false};
    std::thread waiter([&] {
      CpuExecutionScope next;
      Require(next.Start(domain).ok(), "Queued caller failed");
      entered = true;
      Until([&] { return domain->snapshot().waiting_cpu_callers == 1; }, "Caller did not queue to resume");
    });
    Until([&] { return domain->snapshot().waiting_cpu_callers == 1; }, "Caller did not queue");
    {
      CpuSuspension suspension;
      Require(!HasCpuParticipation() && tracker.active() == 0, "Blocked caller retained participation");
      CpuSuspension nested;
      CpuExecutionScope forbidden;
      Require(!forbidden.Start(domain).ok(), "Suspended nested execution was accepted");
      Until([&] { return entered.load(); }, "Yielded slot did not admit waiting caller");
    }
    Require(HasCpuParticipation() && tracker.active() == 1, "Caller failed to resume");
    waiter.join();
    const auto timing = root.timing();
    Require(timing.initial_queue_nanoseconds == 0 && timing.resume_queue_nanoseconds > 0 &&
            timing.blocked_nanoseconds > 0, "Queue and blocked timing were not distinguished");
  }
  Empty(*domain);
}

void OnceWaitAtOneSlot() {
  auto domain = Domain(1);
  std::once_flag once;
  std::atomic<bool> first_inside{false}, second_entered{false};
  std::atomic<size_t> completed{0}, invocations{0};
  auto wait = [&] {
    // Mirror MetalSubmission::Wait: suspension must enclose call_once, not
    // merely its body. Both contenders can then finish with one CPU slot.
    CpuSuspension suspension;
    std::call_once(once, [&] {
      ++invocations;
      first_inside = true;
      Until([&] { return second_entered && domain->snapshot().active_cpu_participants == 0; },
            "call_once contender retained its CPU slot");
    });
  };
  std::thread first([&] {
    CpuExecutionScope root;
    Require(root.Start(domain).ok(), "First once caller failed");
    wait(); ++completed;
  });
  Until([&] { return first_inside.load(); }, "First once caller did not enter");
  std::thread second([&] {
    CpuExecutionScope root;
    Require(root.Start(domain).ok(), "Second once caller failed");
    second_entered = true;
    wait(); ++completed;
  });
  first.join(); second.join();
  Require(completed == 2 && invocations == 1, "One-slot call_once failed");
  Empty(*domain);
}

void SuspendedWorkerBound() {
  auto domain = Domain(3);
  {
    CpuExecutionScope root;
    Require(root.Start(domain).ok(), "Nested-join root failed");
    CpuWorkerGroup outer(2);
    std::atomic<bool> child_entered{false}, release_child{false};
    std::vector<std::thread> workers;
    workers.emplace_back([&] {
      CpuWorkerScope worker(&outer);
      CpuWorkerGroup inner(2);
      Require(inner.participants() == 2, "Nested worker was not reserved");
      std::vector<std::thread> descendants;
      descendants.emplace_back([&] {
        CpuWorkerScope child(&inner);
        child_entered = true;
        Until([&] { return release_child.load(); }, "Nested child was not released");
      });
      JoinCpuWorkers(descendants);
      Require(HasCpuParticipation(), "Joined worker failed to resume its reserved slot");
    });
    Until([&] { return child_entered && domain->snapshot().suspended_cpu_workers == 1; },
          "Nested joining worker was not suspended");
    Require(domain->snapshot().active_cpu_participants == 2 &&
            domain->snapshot().peak_cpu_protected_slots == 3,
            "Suspended worker capacity was counted as active or released");
    CpuWorkerGroup extra(3);
    Require(extra.participants() == 1, "Nested joins multiplied dormant worker threads");
    release_child = true;
    JoinCpuWorkers(workers);
    Require(domain->snapshot().suspended_cpu_workers == 0, "Joined worker retained dormant capacity");
  }
  Empty(*domain);
}

void PartialLaunchAndLifetime() {
  auto domain = Domain(4), retained = domain;
  {
    CpuExecutionScope root;
    Require(root.Start(domain).ok(), "Partial launch root failed");
    domain.reset(); // Tickets retain budget state without the original handle.
    {
      CpuWorkerGroup group(4);
      Require(group.participants() == 4, "Partial launch did not reserve slots");
      std::vector<std::thread> workers;
      workers.emplace_back([&] { CpuWorkerScope worker(&group); });
      // Model an exception before the remaining two threads could launch.
      JoinCpuWorkers(workers);
      Require(retained->snapshot().reserved_cpu_workers == 2,
              "Unused slots were activated or reclaimed as live workers");
    }
    Require(retained->snapshot().reserved_cpu_workers == 0, "Unused launch slots leaked");
    CpuWorkerGroup retry(4);
    Require(retry.participants() == 4, "Failed launch capacity could not be reused");
  }
  Empty(*retained);
}
} // namespace

int main() {
  IdentityAndObservers();
  SuspensionAndFifoResume();
  OnceWaitAtOneSlot();
  SuspendedWorkerBound();
  PartialLaunchAndLifetime();
  return EXIT_SUCCESS;
}
