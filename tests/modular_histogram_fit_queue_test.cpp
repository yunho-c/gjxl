// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/histogram_fit_queue.h"
#include "codestream/workflow_admission_scope.h"
#include "modular_histogram_fitting_test_helpers.h"
#include <algorithm>
#include <exception>
#include <iostream>
#include <thread>

namespace {
using namespace histogram_fitting_test;
using namespace gjxl::thread_budget_internal;
using gjxl::modular_internal::reference_workflow_internal::HistogramFitQueue;
struct Job {
  HistogramTrainingSamples samples;
  HistogramTreeFitter fit;
  std::array<HistogramPropertyEvaluation, kPropertyCount> result;
  DecisionTree tree;
  Status status;
  bool apply = false;
  size_t evaluations = 0;
};
void Run(const std::shared_ptr<const ExecutionDomain> &domain,
         size_t participants, bool fail = false, size_t ready_target = 32,
         size_t flush_us = 500, bool trace_enabled = false) {
  auto resources = CurrentResourceContext();
  resources.domain = domain.get();
  ResourceContextScope resource_context(resources);
  CpuExecutionScope cpu;
  Ok(cpu.Start(domain, participants));
  EncodeScope participation(participants);
  std::array<Job, 12> jobs;
  for (size_t i = 0; i < jobs.size(); ++i) {
    jobs[i].samples = Fixture(i, 257 + i * 3, i % 2 ? 14 : 2);
    Ok(jobs[i].fit.Initialize(jobs[i].samples));
  }
  const auto work = [&](size_t i) noexcept {
    auto &job = jobs[i];
    if (fail && i == 1 && job.evaluations > 0)
      return HistogramFitQueue::Result::kFailed;
    if (job.apply) {
      job.status = job.fit.Apply(
          std::span(job.result).first(job.samples.property_count));
      job.apply = false;
    }
    if (job.status.ok())
      job.status = job.fit.Advance();
    if (!job.status.ok())
      return HistogramFitQueue::Result::kFailed;
    if (job.fit.complete()) {
      job.status = job.fit.Publish(&job.tree);
      return job.status.ok() ? HistogramFitQueue::Result::kDone
                             : HistogramFitQueue::Result::kFailed;
    }
    return HistogramFitQueue::Result::kGpu;
  };
  HistogramFitQueue queue(participants, ready_target, flush_us);
  Ok(queue.Start(jobs.size(), work));
  std::array<size_t, 256> ready;
  size_t count;
  bool complete;
  HistogramFitQueue::Trace trace;
  while ((complete = queue.Next(ready, &count, trace_enabled ? &trace : nullptr)) && count) {
    if (trace_enabled)
      Check(trace.ready == count && trace.live >= count &&
                trace.cpu_queued + trace.cpu_running + count == trace.live &&
                trace.reason >= 1 && trace.reason <= 3 && trace.next_nanoseconds > 0,
            "Queue trace lost ready/live state");
    Check(std::is_sorted(ready.begin(), ready.begin() + count),
          "Ready nodes lost canonical order");
    for (size_t i : std::span(ready).first(count)) {
      auto &job = jobs[i];
      Check(job.fit.ready() && !job.apply,
            "Queue exposed a non-ready or multiply owned fit");
      // Substitute the exact CPU evaluator for the synchronous device call.
      // This exercises the same cross-worker row/result ownership under TSan.
      Ok(job.fit.EvaluateCpu(
          std::span(job.result).first(job.samples.property_count)));
      ++job.evaluations;
      job.apply = true;
    }
    queue.Resume(std::span(ready).first(count), trace_enabled ? &trace : nullptr);
  }
  queue.Stop();
  Check(complete != fail, "Queue failed to propagate a CPU error or complete");
  if (!fail)
    for (size_t i = 0; i < jobs.size(); ++i) {
      auto expected_samples = Fixture(i, 257 + i * 3, i % 2 ? 14 : 2);
      DecisionTree expected;
      Ok(LearnHistogramTree(expected_samples, &expected));
      Check(expected == jobs[i].tree &&
                SameRows(expected_samples, jobs[i].samples),
            "Migrating fits changed their tree or partition order");
    }
}
void Released(const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  Check(!s.active_cpu_participants && !s.reserved_cpu_workers &&
            !s.suspended_cpu_workers && !s.live_capacity_bytes &&
            !s.active_reservations,
        "Histogram fit queue leaked CPU or memory admission");
}
} // namespace
int main() try {
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({size_t{256} << 20, 4}, &domain));
  for (size_t participants : {size_t{1}, size_t{2}, size_t{4}}) {
    Run(domain, participants);
    Run(domain, participants, true);
    Run(domain, participants, false, 1, 0, true);
    Run(domain, participants, false, 64, 2000, true);
    Released(*domain);
  }
  for (auto kind : {WorkerLaunchFailureKind::kSystemError,
                    WorkerLaunchFailureKind::kBadAlloc})
    for (size_t before : {size_t{0}, size_t{1}}) {
      WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kModularTreeLearning,
                                        before, kind};
      bool failed = false;
      {
        WorkerLaunchFaultScopeForTesting inject(&fault);
        try {
          Run(domain, 4);
        } catch (const std::exception &) {
          failed = true;
        }
      }
      Check(failed && fault.triggered && fault.launched_in_group == before,
            "Histogram queue partial launch failure was not propagated");
      Released(*domain);
    }
  std::exception_ptr error_a, error_b;
  std::thread a([&] {
    try {
      Run(domain, 4);
    } catch (...) {
      error_a = std::current_exception();
    }
  });
  std::thread b([&] {
    try {
      Run(domain, 4);
    } catch (...) {
      error_b = std::current_exception();
    }
  });
  a.join();
  b.join();
  if (error_a)
    std::rethrow_exception(error_a);
  if (error_b)
    std::rethrow_exception(error_b);
  Check(domain->snapshot().peak_cpu_protected_slots <= 4,
        "Concurrent queues exceeded their shared CPU domain");
  Released(*domain);
  std::cout << "Resumable fit migration, ordered ownership, CPU errors and "
               "shared domains passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
