// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/histogram_tree_profile.h"
#include "codec/modular/profile.h"
#include "codestream/modular/reference_learning_profile.h"
#include "codestream/workflow_admission_scope.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"
#include "modular_reference_workflow_fixture.h"
#include <barrier>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::thread_budget_internal;
using codestream_internal::CodestreamBuffer;
using test::modular_workflow_fixture::Fixture;
using test::modular_workflow_fixture::Make;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
auto Domain(size_t memory = size_t{256} << 20, size_t cpus = 4) {
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({memory, cpus}, &domain));
  return domain;
}
const ResourceBudget &Budget(const ExecutionDomain &domain) {
  return codestream_internal::WorkflowAdmission::Budget(domain);
}
Status Prepare(const Fixture &f, ReferenceWorkflowOptions options,
               const std::shared_ptr<const ExecutionDomain> &domain, ReferenceModularFrame *out,
               size_t participants = 4, bool nested = false) {
  auto resources = CurrentResourceContext();
  resources.domain = domain.get();
  ResourceContextScope context(resources);
  CpuExecutionScope cpu;
  if (auto s = cpu.Start(domain, participants); !s.ok())
    return s;
  EncodeScope threads(participants);
  if (nested) {
    ParallelScope scope(participants, nullptr, resources);
    return PrepareReferenceModularFrame(f.view(), options, Budget(*domain), out);
  }
  return PrepareReferenceModularFrame(f.view(), options, Budget(*domain), out);
}
void Released(const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  Check(!s.live_capacity_bytes && !s.reserved_unbacked_bytes && !s.active_reservations &&
            !s.active_cpu_participants && !s.reserved_cpu_workers && !s.suspended_cpu_workers &&
            !s.waiting_cpu_callers && !s.waiting_requests,
        "Parallel learning leaked memory or CPU participation");
}
void Same(const ReferenceModularFrame &a, const ReferenceModularFrame &b) {
  const auto &x = a.stats(), &y = b.stats();
  Check(x.local_trees == y.local_trees && x.selection_calls == y.selection_calls &&
            x.learning_calls == y.learning_calls && x.fits == y.fits &&
            x.empty_learning_calls == y.empty_learning_calls &&
            x.scanned_pixels == y.scanned_pixels && x.selected_samples == y.selected_samples &&
            x.distinct_samples == y.distinct_samples && x.tree_nodes == y.tree_nodes &&
            a.streams().size() == b.streams().size(),
        "Parallel learning changed work");
  for (size_t i = 0; i < a.streams().size(); ++i) {
    const auto &p = a.streams()[i], &q = b.streams()[i];
    Check(p.id == q.id && p.tree == q.tree && p.selection.transforms == q.selection.transforms &&
              p.selection.wp_mode == q.selection.wp_mode &&
              p.learning.first_group == q.learning.first_group &&
              p.learning.stop_group == q.learning.stop_group &&
              p.learning.preparation.population == q.learning.preparation.population &&
              p.learning.preparation.quantization_samples ==
                  q.learning.preparation.quantization_samples,
          "Parallel learning changed a tree, selection or sampling");
  }
}
void SameProfile(const HistogramFittingProfile &a, const HistogramFittingProfile &b) {
  Check(a.calls == b.calls && a.successes == b.successes &&
            a.records_written == b.records_written && !a.records_dropped && !b.records_dropped &&
            a.nodes_by_symbol_span == b.nodes_by_symbol_span &&
            a.properties_by_active_bins == b.properties_by_active_bins &&
            a.properties_by_bin_span == b.properties_by_bin_span,
        "Parallel fitter lost profile records or distributions");
  uint64_t fused = 0, skipped = 0;
  for (size_t i = 0; i < a.by_rows.size(); ++i) {
    Check(a.by_rows[i].work == b.by_rows[i].work, "Parallel fitter changed node work");
    fused += a.by_rows[i].work.fused_property_row_visits;
    skipped += a.by_rows[i].work.skipped_constant_properties;
  }
  Check(fused && skipped, "Parallel profile parity did not exercise fitter work reduction");
  for (size_t i = 0; i < a.by_depth.size(); ++i)
    Check(a.by_depth[i].nodes == b.by_depth[i].nodes && a.by_depth[i].rows == b.by_depth[i].rows,
          "Parallel fitter changed depth work");
  for (size_t i = 0; i < a.records_written; ++i) {
    const auto &x = a.records[i], &y = b.records[i];
    Check(x.rows == y.rows && x.population == y.population && x.predictors == y.predictors &&
              x.properties == y.properties && x.nodes == y.nodes &&
              x.maximum_depth == y.maximum_depth && x.success == y.success,
          "Parallel fit records lost canonical order");
  }
}
void Encode(const ReferenceModularFrame &frame, PreparedModularEntropyOptions entropy,
            const ResourceBudget &budget, CodestreamBuffer *out) {
  ModularWorkflowStoragePlan plan;
  Ok(ComputePreparedModularStreamsStoragePlan(frame.view(), entropy, &plan));
  ResourceReservation reservation;
  Ok(budget.TryReserve(plan.working.peak_bytes, &reservation));
  ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
  Ok(EncodePreparedModularStreams(frame.view(), entropy, out));
}
void CheckTrace(const LearningScheduleProfile &trace, size_t participants, size_t records) {
  Check(trace.jobs_written == records && trace.waves_written && !trace.dropped,
        "Schedule trace lost local jobs");
  size_t traced = 0;
  for (const auto &wave : trace.waves.first(trace.waves_written)) {
    Check(wave.first_job == traced && wave.participants <= participants &&
              wave.start <= wave.dispatch && wave.dispatch <= wave.joined &&
              wave.joined <= wave.finish, "Invalid learning wave trace");
    for (const auto &job : trace.jobs.subspan(wave.first_job, wave.jobs))
      Check(job.worker < wave.participants && job.start >= wave.dispatch &&
                job.finish <= wave.joined && job.start <= job.finish &&
                job.preparation + job.fitting <= job.finish - job.start + 1e-6,
            "Job trace escaped its wave or overlapped preparation/fitting");
    traced += wave.jobs;
  }
  Check(traced == trace.jobs_written, "Wave trace omitted jobs");
}
void Parity(bool encode, bool shared = false) {
  for (bool all : {false, true}) {
    const auto f = Make(shared ? Extent2D{257, 129} : Extent2D{2049, 17},
                        all ? 4 : 3, all ? 16 : 8, all ? 1 : 3);
    auto domain = Domain();
    ReferenceWorkflowOptions options;
    options.predictors = all ? ReferencePredictors::kAll : ReferencePredictors::kBest;
    options.learning_cpu_participants = 1;
    ReferenceModularFrame serial, parallel;
    HistogramFittingProfile expected;
    std::array<HistogramFitRecord, 16> records;
    expected.records = records;
    {
      HistogramFittingProfileSession session(&expected);
      Ok(Prepare(f, options, domain, &serial));
    }
    for (size_t requested : {size_t{2}, size_t{4}, size_t{0}}) {
      options.learning_cpu_participants = requested;
      HistogramFittingProfile actual;
      std::array<HistogramFitRecord, 16> actual_records;
      actual.records = actual_records;
      ModularProfile modular;
      std::array<LearningJobTiming, 32> jobs;
      std::array<LearningWaveTiming, 32> waves;
      LearningScheduleProfile trace;
      trace.jobs = jobs;
      trace.waves = waves;
      {
        ReferenceWorkflowProfileSession scheduling(nullptr, &trace);
        HistogramFittingProfileSession session(&actual);
        ProfileSession stages(&modular);
        Ok(Prepare(f, options, domain, &parallel));
      }
      Same(serial, parallel);
      SameProfile(expected, actual);
      if (shared) {
        Check(!trace.jobs_written && !trace.waves_written &&
                  actual.parallel_property_nodes && actual.peak_property_participants > 1,
              "Shared fit did not retain independent property scheduling");
      } else {
        CheckTrace(trace, requested ? requested : 4, parallel.streams().size());
        Check(trace.waves_written == 1 && waves[0].jobs > waves[0].participants &&
                  parallel.stats().learning_waves == 1 &&
                  !parallel.stats().learning_memory_limited_waves,
              "Ready local fits were split into participant-sized barriers");
        Check(!actual.parallel_property_nodes && actual.peak_property_participants == 1,
              "Local ready queues started nested property teams");
      }
      Check(parallel.stats().learning_peak_participants == (requested ? requested : 4) &&
                modular.training_work.weighted_prediction_samples ==
                    serial.stats().scanned_pixels &&
                !modular.training_work.reused_prediction_samples,
            "Parallel learning did not use its allowance or lost feature counts");
      Check(modular.seconds[size_t(ProfileStage::kTraining)] + 1e-4 >=
                parallel.stats().learning_seconds,
            "Learning wall interval escaped its parent stage");
      for (auto entropy : {options.entropy,
                           PreparedModularEntropyOptions{PreparedModularEntropyPolicy::kPrefix}}) {
        if (!encode)
          break;
        CodestreamBuffer a, b;
        Encode(serial, entropy, Budget(*domain), &a);
        Encode(parallel, entropy, Budget(*domain), &b);
        Check(std::ranges::equal(a.view(), b.view()), "Parallel complete bytes differ");
#ifdef GJXL_REFERENCE_WORKFLOW_ORACLE
        Check(test::modular_reference::DecodeLossless(b.view()) == f.original,
              "Parallel independent decode differs");
#endif
      }
      parallel = {};
    }
    serial = {};
    Released(*domain);
  }
}
void Limits(bool shared = false) {
  const auto f = Make(shared ? Extent2D{257, 129} : Extent2D{2049, 17}, 1, 16, 3);
  ReferenceWorkflowOptions options;
  options.learning_cpu_participants = 8;
  for (size_t domain_cpus : {size_t{1}, size_t{2}, size_t{4}})
    for (size_t image_cpus : {size_t{1}, size_t{2}, size_t{4}}) {
      auto domain = Domain(size_t{64} << 20, domain_cpus);
      ReferenceModularFrame frame;
      Ok(Prepare(f, options, domain, &frame, image_cpus));
      Check(frame.stats().learning_peak_participants <= std::min(domain_cpus, image_cpus) &&
                domain->snapshot().peak_cpu_participants <= std::min(domain_cpus, image_cpus),
            "Learning exceeded a CPU ceiling");
      frame = {};
      Released(*domain);
    }
  auto domain = Domain();
  ReferenceModularFrame frame;
  Ok(PrepareReferenceModularFrame(f.view(), options, Budget(*domain), &frame));
  Check(frame.stats().learning_peak_participants == 1, "Unaccounted private workers launched");
  Ok(Prepare(f, options, domain, &frame, 4, true));
  Check(frame.stats().learning_peak_participants == 1, "Nested learning fanned out");
  options.buffering = ReferenceBuffering::kBuffered;
  Ok(Prepare(f, options, domain, &frame));
  Check(!frame.stats().local_trees && frame.stats().learning_peak_participants > 1 &&
            frame.stats().fits == 1,
        "Global fit did not parallelize its property work");
  frame = {};
  Released(*domain);
}
void MemoryAdmission(Extent2D extent) {
  const auto f = Make(extent, 3, 8, 3);
  auto control = Domain();
  ReferenceWorkflowOptions options;
  options.learning_cpu_participants = 1;
  ReferenceModularFrame serial, parallel;
  Ok(Prepare(f, options, control, &serial));
  const size_t limit = control->snapshot().peak_committed_bytes;
  auto constrained = Domain(limit);
  options.learning_cpu_participants = 4;
  std::array<LearningJobTiming, 32> jobs;
  std::array<LearningWaveTiming, 32> waves;
  LearningScheduleProfile trace;
  trace.jobs = jobs;
  trace.waves = waves;
  {
    ReferenceWorkflowProfileSession session(nullptr, &trace);
    Ok(Prepare(f, options, constrained, &parallel));
  }
  if (parallel.stats().local_trees)
    CheckTrace(trace, 4, parallel.streams().size());
  else
    Check(!trace.jobs_written && !trace.waves_written && !trace.dropped,
          "Shared fitting emitted local scheduling records");
  Same(serial, parallel);
  Check(parallel.stats().learning_memory_limited_waves &&
            parallel.stats().learning_peak_participants < 4 &&
            constrained->snapshot().peak_committed_bytes <= limit,
        "Learning did not reduce concurrency under memory pressure");
  parallel = {};
  serial = {};
  Released(*constrained);
  Released(*control);
}
void Failures(bool shared = false) {
  const auto f = Make(shared ? Extent2D{257, 129} : Extent2D{2049, 17}, 3, 8, 3);
  auto domain = Domain();
  ReferenceModularFrame frame;
  ReferenceWorkflowOptions options;
  Ok(Prepare(Make({3, 3}, 3, 8, 1), options, domain, &frame));
  const auto *identity = frame.streams().data();
  const auto retained = Budget(*domain).snapshot().committed_bytes();
  for (auto kind : {WorkerLaunchFailureKind::kSystemError, WorkerLaunchFailureKind::kBadAlloc})
    for (size_t before : {size_t{0}, size_t{1}}) {
      WorkerLaunchFaultForTesting fault{shared ? WorkerLaunchSite::kModularHistogramProperties
                                                : WorkerLaunchSite::kModularTreeLearning, before, kind};
      WorkerLaunchFaultScopeForTesting inject(&fault);
      const auto status = Prepare(f, options, domain, &frame);
      Check(!status.ok() && fault.triggered && fault.launched_in_group == before &&
                status.code() == (kind == WorkerLaunchFailureKind::kBadAlloc
                                      ? StatusCode::kOutOfMemory
                                      : StatusCode::kInternal) &&
                frame.streams().data() == identity &&
                Budget(*domain).snapshot().committed_bytes() == retained &&
                !domain->snapshot().active_cpu_participants &&
                !domain->snapshot().reserved_cpu_workers,
            "Partial worker launch failed to join, unwind, or preserve output");
    }
  options.learning_cpu_participants = 257;
  Check(Prepare(f, options, domain, &frame).code() == StatusCode::kInvalidArgument &&
            frame.streams().data() == identity,
        "Invalid learning limit changed output");
  frame = {};
  Released(*domain);
}
void SharedDomain(bool shared = false) {
  const auto f = Make(shared ? Extent2D{257, 129} : Extent2D{2049, 33}, 3, 8, 3);
  auto domain = Domain(size_t{128} << 20, 3);
  std::array<ReferenceModularFrame, 2> frames;
  std::array<Status, 2> results;
  std::barrier ready(2);
  auto run = [&](size_t i) {
    ready.arrive_and_wait();
    results[i] = Prepare(f, {}, domain, &frames[i]);
  };
  std::thread a(run, 0), b(run, 1);
  a.join();
  b.join();
  Ok(results[0]);
  Ok(results[1]);
  Same(frames[0], frames[1]);
  Check(domain->snapshot().peak_cpu_participants <= 3 &&
            domain->snapshot().peak_cpu_protected_slots <= 3,
        "Concurrent images exceeded their shared CPU domain");
  frames = {};
  Released(*domain);
}
} // namespace
int main(int argc, char **argv) try {
  const bool preparation_only = argc == 2 && std::string_view(argv[1]) == "--preparation-only";
  Check(argc == 1 || preparation_only, "Unknown test argument");
  Parity(!preparation_only);
  Parity(!preparation_only, true);
  std::cout << "Tree/profile parity passed\n" << std::flush;
  Limits();
  Limits(true);
  MemoryAdmission(preparation_only ? Extent2D{2049, 17} : Extent2D{513, 513});
  std::cout << "CPU and memory limits passed\n" << std::flush;
  MemoryAdmission({257, 129});
  Failures();
  Failures(true);
  SharedDomain();
  SharedDomain(true);
  std::cout << "Parallel local learning: " << (preparation_only ? "preparation" : "exact bytes")
            << ", profiles, CPU/memory limits, "
               "partial launches and shared domains passed\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
