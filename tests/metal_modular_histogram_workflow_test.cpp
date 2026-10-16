// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/reference_workflow_internal.h"
#include "codestream/workflow_admission_scope.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"
#include "gpu/metal/metal_backend.h"
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_internal::reference_workflow_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::thread_budget_internal;
void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
struct Fixture {
  std::array<std::vector<int32_t>, 4> pixels;
  std::array<ModularChannelView, 4> channels;
  std::array<HistogramTrainingImage, 4> images;
  std::array<DecisionTree, 4> expected;
  HistogramSamplingOptions sampling;
  Fixture() {
    sampling.maximum_samples = 8192;
    sampling.split_threshold = 0;
    uint32_t rng = 71;
    for (size_t i = 0; i < images.size(); ++i) {
      constexpr Extent2D extent{129, 33};
      pixels[i].resize(extent.width * extent.height);
      for (size_t y = 0; y < extent.height; ++y)
        for (size_t x = 0; x < extent.width; ++x) {
          rng = 1664525 * rng + 1013904223;
          pixels[i][y * extent.width + x] =
              y < 16 ? int32_t(x + i) : int32_t(rng >> 24);
        }
      channels[i] = {pixels[i], {extent}, extent.width};
      images[i] = {extent, {&channels[i], 1}};
      PreparedHistogramSamples samples;
      Ok(PrepareHistogramSamples({&images[i], 1}, &samples, sampling));
      Ok(LearnPreparedHistogramTree(samples, &expected[i]));
      Check(expected[i].size > 1, "Failure fixture requires a split");
    }
  }
};
auto Domain(size_t memory = size_t{256} << 20) {
  std::shared_ptr<const ExecutionDomain> result;
  Ok(ExecutionDomain::Create({memory, 4}, &result));
  return result;
}
void Released(const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  Check(!s.live_capacity_bytes && !s.active_reservations &&
            !s.reserved_unbacked_bytes && !s.active_cpu_participants &&
            !s.reserved_cpu_workers && !s.suspended_cpu_workers &&
            !s.waiting_cpu_callers && !s.waiting_requests &&
            s.peak_cpu_protected_slots <= 4,
        "Histogram workflow leaked resources or exceeded its CPU domain");
}
Status Run(const Fixture &f, GpuBackend &gpu,
           const std::shared_ptr<const ExecutionDomain> &domain,
           ReferenceWorkflowStats *stats, bool *handled,
           size_t minimum_rows = 64, size_t scratch = 64 * 1024 * 1024,
           size_t leaves = kMaximumTreeLeaves) try {
  auto resources = CurrentResourceContext();
  resources.domain = domain.get();
  ResourceContextScope context(resources);
  CpuExecutionScope cpu;
  if (auto s = cpu.Start(domain, 4); !s.ok())
    return s;
  EncodeScope threads(4);
  std::array<ReferenceModularStream, 4> records;
  std::array<HistogramFitRequest, 4> requests;
  for (size_t i = 0; i < requests.size(); ++i) {
    records[i].id = i * 3 + 1;
    requests[i] = {{&f.images[i], 1}, &records[i], leaves};
  }
  ReferenceWorkflowOptions options;
  options.gpu = &gpu;
  options.fitting_backend = ReferenceFittingBackend::kGpu;
  options.fitting_minimum_rows = minimum_rows;
  options.fitting_execution.maximum_scratch_bytes = scratch;
  stats->local_trees = true;
  const auto status = TryLearnHistogramBatch(
      requests, f.sampling, 4, options,
      codestream_internal::WorkflowAdmission::Budget(*domain), stats, handled);
  if (status.ok() && *handled)
    for (size_t i = 0; i < requests.size(); ++i)
      Check(records[i].tree == f.expected[i],
            "GPU workflow tree differs from native CPU tree");
  return status;
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Test histogram workflow allocation failed");
}
} // namespace
int main() try {
  std::unique_ptr<GpuBackend> gpu;
  Ok(CreateEmbeddedMetalBackend({}, &gpu));
  Fixture fixture;
  auto domain = Domain();
  for (size_t cutoff : {size_t{1}, size_t{64}, size_t{8192}})
    for (size_t scratch : {size_t{1}, size_t{64 * 1024 * 1024}}) {
      ReferenceWorkflowStats stats;
      bool handled = false;
      Ok(Run(fixture, *gpu, domain, &stats, &handled, cutoff, scratch));
      Check(handled && stats.fits == 4, "GPU workflow lost fits");
      Check((stats.histogram_gpu_nodes != 0) == (cutoff < 8192 && scratch > 1),
            "GPU cutoff or scratch admission changed");
      Released(*domain);
    }
  for (size_t memory : {size_t{1}, size_t{256} << 20}) {
    auto limited = Domain(memory);
    ReferenceWorkflowStats stats;
    bool handled = false;
    Ok(Run(fixture, *gpu, limited, &stats, &handled));
    Check(handled == (memory > 1) &&
              (memory > 1 || stats.histogram_admission_fallbacks == 1),
          "Whole-cohort preflight did not return to CPU admission");
    Released(*limited);
  }
  for (size_t kind = 0; kind < 5; ++kind) {
    ReferenceWorkflowStats stats;
    bool handled = false;
    if (kind < 2)
      Ok(ArmNextMetalSubmissionFailureForTest(*gpu, kind == 0, kind == 1));
    if (kind == 2)
      Ok(ArmNextMetalAllocationFailureForTest(*gpu));
    if (kind == 3)
      ArmNextManagedHostAllocationFailureForTest();
    const auto status =
        Run(fixture, *gpu, domain, &stats, &handled, 1, 64 * 1024 * 1024,
            kind == 4 ? 1 : kMaximumTreeLeaves);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(!status.ok() && (kind != 3 || !pending),
          "Fitting failure was retried or ignored");
    if (kind < 2 || kind == 4)
      Check(stats.histogram_gpu_preparations == 1,
            "Failure did not reach GPU fitting");
    if (kind == 4)
      Check(stats.histogram_gpu_nodes > 0,
            "CPU capacity failure did not follow GPU evaluation");
    Released(*domain);
  }
  for (auto kind : {WorkerLaunchFailureKind::kSystemError,
                    WorkerLaunchFailureKind::kBadAlloc}) {
    WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kModularTreeLearning, 1,
                                      kind};
    WorkerLaunchFaultScopeForTesting inject(&fault);
    ReferenceWorkflowStats stats;
    bool handled = false;
    Check(!Run(fixture, *gpu, domain, &stats, &handled).ok() &&
              fault.triggered && fault.launched_in_group == 1,
          "Partial preparation launch did not drain");
    Released(*domain);
  }
  std::exception_ptr errors[2];
  const auto concurrent = [&](size_t i) {
    try {
      ReferenceWorkflowStats stats;
      bool handled = false;
      Ok(Run(fixture, *gpu, domain, &stats, &handled));
      Check(handled && stats.histogram_gpu_nodes, "Concurrent GPU fit skipped");
    } catch (...) {
      errors[i] = std::current_exception();
    }
  };
  std::thread a(concurrent, 0), b(concurrent, 1);
  a.join();
  b.join();
  for (auto e : errors)
    if (e)
      std::rethrow_exception(e);
  Released(*domain);
  std::cout
      << "Batched GPU fitting, CPU cutoffs, admission, device/host/launch "
         "failures, CPU capacity failure and concurrent domains passed\n";
} catch (const std::exception &e) {
  DisarmManagedHostAllocationFailureForTest();
  std::cerr << e.what() << '\n';
  return 1;
}
