// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/local_learning_queue.h"
#include "modular_histogram_fitting_test_helpers.h"
#include <condition_variable>
#include <iostream>
#include <mutex>

namespace {
using namespace histogram_fitting_test;
using namespace gjxl::thread_budget_internal;
using gjxl::modular_internal::reference_workflow_internal::LocalLearningQueue;
void Released(const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  Check(!s.active_cpu_participants && !s.reserved_cpu_workers &&
        !s.suspended_cpu_workers && !s.live_capacity_bytes && !s.active_reservations,
        "Incremental queue leaked CPU or memory admission");
}
void Run(const std::shared_ptr<const ExecutionDomain> &domain, size_t participants) {
  auto resources = CurrentResourceContext();
  resources.domain = domain.get();
  ResourceContextScope context(resources);
  CpuExecutionScope cpu;
  Ok(cpu.Start(domain, participants));
  EncodeScope participation(participants);
  std::array<size_t, 1024> results{};
  const auto work = [&](size_t job, size_t worker) noexcept { results[job] = job + worker + 1; };
  LocalLearningQueue queue(participants);
  Ok(queue.Start(work));
  size_t next = 0, completed = 0;
  while (completed != results.size()) {
    while (next < results.size() && next - completed < LocalLearningQueue::kCapacity)
      Ok(queue.Submit(next++));
    queue.Progress();
    size_t job;
    while (queue.Take(&job)) {
      Check(results[job] >= job + 1 && results[job] <= job + participants,
            "Completion did not publish the worker payload");
      ++completed;
    }
  }
  queue.Stop();
}
// A held tail from batch A can only finish after a task from batch B runs.
// This would deadlock under a cohort join. No timing/sleep assumptions.
void Overlap(const std::shared_ptr<const ExecutionDomain> &domain) {
  auto resources = CurrentResourceContext();
  resources.domain = domain.get();
  ResourceContextScope context(resources);
  CpuExecutionScope cpu;
  Ok(cpu.Start(domain, 3));
  EncodeScope participation(3);
  std::mutex mutex;
  std::condition_variable changed;
  bool tail_started = false, next_started = false;
  const auto work = [&](size_t job, size_t) noexcept {
    CpuSuspension suspension;
    std::unique_lock lock(mutex);
    if (job == 0) {
      tail_started = true;
      changed.notify_all();
      changed.wait(lock, [&] { return next_started; });
    } else if (job == 2) {
      next_started = true;
      changed.notify_all();
    }
  };
  LocalLearningQueue queue(3);
  Ok(queue.Start(work));
  Ok(queue.Submit(0));
  {
    CpuSuspension suspension;
    std::unique_lock lock(mutex);
    changed.wait(lock, [&] { return tail_started; });
  }
  Ok(queue.Submit(1));
  size_t job;
  while (!queue.Take(&job)) queue.Progress();
  Check(job == 1, "Held tail completed before next-batch admission");
  Ok(queue.Submit(2));
  size_t completed = 1;
  while (completed != 3) {
    queue.Progress();
    while (queue.Take(&job)) ++completed;
  }
}
} // namespace
int main() try {
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({size_t{64} << 20, 4}, &domain));
  for (size_t participants : {size_t{1}, size_t{2}, size_t{4}}) {
    Run(domain, participants);
    Released(*domain);
  }
  Overlap(domain);
  Released(*domain);
  for (auto kind : {WorkerLaunchFailureKind::kSystemError, WorkerLaunchFailureKind::kBadAlloc})
    for (size_t before : {size_t{0}, size_t{1}}) {
      WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kModularTreeLearning, before, kind};
      bool failed = false;
      {
        WorkerLaunchFaultScopeForTesting inject(&fault);
        try { Run(domain, 4); } catch (const std::exception &) { failed = true; }
      }
      Check(failed && fault.triggered && fault.launched_in_group == before,
            "Incremental queue did not join a partial worker launch");
      Released(*domain);
    }
  std::exception_ptr error_a, error_b;
  std::thread a([&] { try { Run(domain, 4); } catch (...) { error_a = std::current_exception(); } });
  std::thread b([&] { try { Run(domain, 4); } catch (...) { error_b = std::current_exception(); } });
  a.join(); b.join();
  if (error_a) std::rethrow_exception(error_a);
  if (error_b) std::rethrow_exception(error_b);
  Check(domain->snapshot().peak_cpu_protected_slots <= 4, "Queues exceeded the shared CPU cap");
  Released(*domain);
  std::cout << "Incremental admission, tail overlap, publication, partial launches and shared domains passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
