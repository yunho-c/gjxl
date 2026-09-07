// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <vector>

#include "codestream/batch_workflow_test.h"
#include "codestream/workflow_admission.h"
#include "codec/frontend_storage_plan.h"
#include "codec/prepared_coefficients_internal.h"
#include "core/thread_budget.h"
#include "core/image_buffer.h"
#include "core/worker_launch_internal.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::thread_budget_internal;
using Site = WorkerLaunchSite;
using Kind = WorkerLaunchFailureKind;
void Require(bool good, const char* message) {
  if (!good) { std::cerr << message << '\n'; std::exit(EXIT_FAILURE); }
}
void Ok(const Status& status) {
  if (!status.ok()) { std::cerr << status.message() << '\n'; std::exit(EXIT_FAILURE); }
}
template<class Predicate> void Until(Predicate predicate, const char* message) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!predicate()) {
    Require(std::chrono::steady_clock::now() < end, message);
    std::this_thread::yield();
  }
}
struct Gate {
  std::mutex mutex;
  std::condition_variable changed;
  std::atomic<bool> entered{false};
  bool released = false;
  void Wait() {
    std::unique_lock lock(mutex);
    entered = true;
    Require(changed.wait_for(lock, std::chrono::seconds(10), [&] { return released; }),
            "Launch-failure gate was not released");
  }
  void Release() {
    std::lock_guard lock(mutex);
    released = true;
    changed.notify_all();
  }
};
void Empty(const ExecutionDomain& domain) {
  Ok(WorkflowAdmission::TrimIdle(domain));
  const auto s = domain.snapshot();
  Require(s.active_cpu_participants == 0 && s.reserved_cpu_workers == 0 &&
          s.suspended_cpu_workers == 0 && s.waiting_cpu_callers == 0 &&
          s.peak_cpu_protected_slots <= s.effective_cpu_participant_limit &&
          s.active_reservations == 0 && s.waiting_requests == 0 &&
          s.live_capacity_bytes == 0 && s.idle_capacity_bytes == 0,
          "Worker launch cleanup leaked or exceeded CPU/memory capacity");
}
bool FallsBack(Site site, Kind kind) {
  return kind == Kind::kSystemError && (site == Site::kColorRows || site == Site::kInitialQuantization ||
                                       site == Site::kForwardTransforms);
}
StatusCode FailureCode(Kind kind) { return kind == Kind::kBadAlloc ? StatusCode::kOutOfMemory : StatusCode::kInternal; }
struct Fixture {
  Image3FBuffer image{{320, 272}}; // Four AC groups and all frontend parallel thresholds.
  std::shared_ptr<const ExecutionDomain> domain;
  VarDctEncodingOptions options;
  std::vector<uint8_t> expected;
  VarDctEncodingSummary summary;
  explicit Fixture(bool metal, bool exact = false) {
    Ok(ExecutionDomain::Create({0, 4}, &domain));
    for (size_t c = 0; c < 3; ++c)
      for (size_t i = 0; i < image.plane(c).size(); ++i)
        image.plane(c)[i] = 0.05f + 0.8f * ((i * (c + 3)) % 127) / 127.0f;
    options.backend = metal ? VarDctBackendPreference::kMetal : VarDctBackendPreference::kCpu;
    if (exact) options.metal_aq_mode = GpuAdaptiveQuantizationMode::kExactCoefficients;
    options.effort = 7; // Includes the coefficient-order path, unlike lower efforts.
    options.execution_domain = domain;
    options.cpu_thread_count = 1;
    Ok(EncodeLinearRgbVarDctCodestream(image.const_view(), options, &expected, &summary));
    Empty(*domain);
  }
};

void SingleFailures(bool metal, bool exact = false, std::optional<Site> selected = {}) {
  Fixture fixture(metal, exact);
  for (Site site : {Site::kColorRows, Site::kInitialQuantization, Site::kForwardTransforms,
                    Site::kCoefficientOrders, Site::kSerializerSections}) {
    if (selected && site != *selected) continue;
    // Prepared forward DCT belongs to exact Metal, not the native CPU pipeline.
    // Resident Metal uses its own frontend kernels; exercise its real CPU tail.
    if (!metal && site == Site::kForwardTransforms) continue;
    if (metal && exact && site != Site::kForwardTransforms) continue;
    if (metal && !exact && site != Site::kCoefficientOrders && site != Site::kSerializerSections) continue;
    for (size_t threads : {0, 4}) {
      fixture.options.cpu_thread_count = threads;
      for (Kind kind : {Kind::kSystemError, Kind::kBadAlloc}) {
        for (size_t before : {0, 1, 2}) {
          WorkerLaunchFaultForTesting fault{site, before, kind};
          fault.context = &fixture;
          fault.before_failure = +[](void* context) noexcept {
            const auto s = static_cast<const Fixture*>(context)->domain->snapshot();
            Require(HasCpuParticipation() && s.active_cpu_participants + s.reserved_cpu_workers +
                    s.suspended_cpu_workers <= s.effective_cpu_participant_limit,
                    "Launch attempted without CPU participation or above reserved capacity");
          };
          std::vector<uint8_t> bytes{3, 1, 4};
          const auto* old_bytes = bytes.data();
          VarDctEncodingSummary summary;
          summary.encoded_bytes = 123;
          VarDctEncodingTiming timing;
          timing.total_nanoseconds = 456;
          timing.attempts.resize(2);
          const auto* old_attempts = timing.attempts.data();
          Status status;
          {
            WorkerLaunchFaultScopeForTesting injection(&fault);
            status = EncodeLinearRgbVarDctCodestreamProfiled(fixture.image.const_view(), fixture.options,
                                                           &bytes, &summary, &timing);
          }
          if (!fault.triggered || fault.launched_in_group != before) {
            std::cerr << "Unreached launch site=" << static_cast<int>(site) << " before=" << before
                      << " threads=" << threads << " metal=" << metal << '\n';
            std::exit(EXIT_FAILURE);
          }
          if (FallsBack(site, kind)) {
            Ok(status);
            Require(bytes == fixture.expected && summary == fixture.summary && timing.total_nanoseconds > 0,
                    "Partial launch serial fallback changed bytes or decisions");
          } else {
            Require(status.code() == FailureCode(kind) && bytes.data() == old_bytes &&
                    bytes == std::vector<uint8_t>({3, 1, 4}) && summary.encoded_bytes == 123 &&
                    timing.total_nanoseconds == 456 && timing.attempts.data() == old_attempts && timing.attempts.size() == 2,
                    "Launch failure changed public output, timing, or its error classification");
          }
          Empty(*fixture.domain);
          Ok(EncodeLinearRgbVarDctCodestream(fixture.image.const_view(), fixture.options, &bytes, &summary));
          Require(bytes == fixture.expected && summary == fixture.summary, "Launch failure poisoned domain reuse");
          Empty(*fixture.domain);
          std::cout << "single site=" << static_cast<int>(site) << " before=" << before
                    << " kind=" << static_cast<int>(kind) << " threads=" << threads
                    << " metal=" << metal << " exact=" << exact << " passed\n" << std::flush;
        }
      }
    }
  }
}

void BatchFailureDuringShutdown(bool metal, Site site, Kind kind) {
  Fixture fixture(metal, site == Site::kForwardTransforms);
  fixture.options.cpu_thread_count = 4;
  std::unique_ptr<VarDctBatchEncoder> driver;
  Ok(VarDctBatchEncoder::Create(1, &driver));
  const std::array<VarDctBatchEncodingRequest, 2> requests{{
    {fixture.image.const_view(), fixture.options}, {fixture.image.const_view(), fixture.options}}};
  Gate fault_gate;
  WorkerLaunchFaultForTesting fault{site, 1, kind};
  fault.context = &fault_gate;
  fault.before_failure = +[](void* context) noexcept { static_cast<Gate*>(context)->Wait(); };
  Status active_status, queued_status;
  std::vector<VarDctBatchEncodingResult> active_results, queued_results(1);
  queued_results[0].codestream = {17};
  const void* queued_address = queued_results.data();
  std::atomic<bool> queued{false}, closing{false}, stopped{false};
  std::thread active([&] {
    batch_execution_observer_for_testing = {&fault, +[](void* context, size_t index, bool entering) noexcept {
      if (index == 0) (void)SetWorkerLaunchFaultForTesting(entering ? static_cast<WorkerLaunchFaultForTesting*>(context) : nullptr);
    }};
    active_status = driver->Encode(requests, &active_results);
  });
  Until([&] { return fault_gate.entered.load(); }, "Batch did not reach partial worker launch");
  std::thread waiting([&] {
    batch_lifecycle_observer_for_testing = {&queued, +[](void* context, BatchLifecycleEventForTesting event) noexcept {
      if (event == BatchLifecycleEventForTesting::kWaitingForDriver) *static_cast<std::atomic<bool>*>(context) = true;
    }};
    queued_status = driver->Encode(requests, &queued_results);
  });
  Until([&] { return queued.load(); }, "Batch caller did not queue behind partial launch");
  std::thread closer([&] {
    batch_lifecycle_observer_for_testing = {&closing, +[](void* context, BatchLifecycleEventForTesting event) noexcept {
      if (event == BatchLifecycleEventForTesting::kClosing) *static_cast<std::atomic<bool>*>(context) = true;
    }};
    driver->Shutdown();
    stopped = true;
  });
  Until([&] { return closing.load(); }, "Shutdown did not close during partial launch");
  Require(!stopped, "Shutdown abandoned partial-launch cleanup");
  fault_gate.Release();
  active.join(); waiting.join(); closer.join();
  Ok(active_status);
  Require(fault.triggered && fault.launched_in_group == 1 && stopped && active_results.size() == 2 &&
          queued_status.code() == StatusCode::kUnavailable && queued_results.data() == queued_address &&
          queued_results.size() == 1 && queued_results[0].codestream == std::vector<uint8_t>{17},
          "Shutdown failed to drain or preserve queued results around launch failure");
  if (FallsBack(site, kind)) {
    Ok(active_results[0].status);
    Require(active_results[0].codestream == fixture.expected, "Drained fallback changed bytes");
  } else {
    Require(active_results[0].status.code() == FailureCode(kind) && active_results[0].codestream.empty(),
            "Batch launch failure was not isolated to its image");
  }
  Require(active_results[0].scheduling.cpu_admitted && active_results[0].scheduling.service_nanoseconds > 0,
          "Launch failure lost its admitted service interval");
  Ok(active_results[1].status);
  Require(active_results[1].codestream == fixture.expected && active_results[1].summary == fixture.summary,
          "Active batch did not complete later images after partial launch failure");
  driver.reset();
  Empty(*fixture.domain);
  std::cout << "batch drain site=" << static_cast<int>(site) << " metal=" << metal << " passed\n" << std::flush;
}

void PreparedForwardFailures() {
  using namespace prepared_coefficients_internal;
  Fixture fixture(false);
  AcStrategyGrid grid;
  Ok(AcStrategyGrid::Create({40, 34}, &grid));
  for (size_t y = 0; y < 34; ++y)
    for (size_t x = 0; x < 40; ++x) Ok(grid.Set(x, y, AcStrategyType::kDct8));
  PreparedForwardDctCoefficients expected;
  {
    EncodeScope serial(1);
    Ok(PrepareForwardDctCoefficients(fixture.image.const_view(), grid, &expected));
  }
  const auto equal = [&](const PreparedForwardDctCoefficients& actual) {
    if (!actual.valid() || actual.pixel_extent != expected.pixel_extent || actual.block_extent != expected.block_extent ||
        actual.color_tile_extent != expected.color_tile_extent || actual.transforms.size() != expected.transforms.size() ||
        actual.color_tile_offsets != expected.color_tile_offsets ||
        actual.color_tile_transform_indices != expected.color_tile_transform_indices) return false;
    for (size_t c = 0; c < 3; ++c)
      if (actual.coefficients[c].size() != expected.coefficients[c].size() ||
          std::memcmp(actual.coefficients[c].data(), expected.coefficients[c].data(),
                      actual.coefficients[c].size() * sizeof(float)) != 0) return false;
    for (size_t i = 0; i < actual.transforms.size(); ++i) {
      const auto &a = actual.transforms[i], &b = expected.transforms[i];
      if (a.block_x != b.block_x || a.block_y != b.block_y || a.strategy != b.strategy ||
          a.coefficient_offset != b.coefficient_offset || a.coefficient_count != b.coefficient_count) return false;
    }
    return true;
  };
  for (size_t threads : {0, 4}) {
    frontend_storage_internal::PreparedForwardStoragePlan plan;
    Ok(frontend_storage_internal::ComputePreparedForwardStoragePlan(fixture.image.extent(), threads, &plan));
    for (Kind kind : {Kind::kSystemError, Kind::kBadAlloc}) {
      for (size_t before : {0, 1, 2}) {
        WorkerLaunchFaultForTesting fault{Site::kForwardTransforms, before, kind};
        PreparedForwardDctCoefficients result;
        result.pixel_extent = {8, 8};
        result.coefficients[0].push_back(-123);
        const auto* sentinel = result.coefficients[0].data();
        for (bool inject : {true, false}) {
          Status status;
          {
            WorkflowAdmission admission;
            Ok(admission.Start(plan.working.peak_bytes, fixture.domain));
            CpuExecutionScope cpu;
            Ok(cpu.Start(fixture.domain, threads));
            EncodeScope scope(threads);
            WorkerLaunchFaultScopeForTesting injection(inject ? &fault : nullptr);
            status = PrepareForwardDctCoefficients(fixture.image.const_view(), grid, &result);
          }
          Require(fault.triggered && fault.launched_in_group == before, "Direct forward launch fault was not reached");
          if (!inject || kind == Kind::kSystemError) {
            Ok(status);
            Require(equal(result), "Forward launch fallback/recovery changed coefficients or metadata");
          } else Require(status.code() == StatusCode::kOutOfMemory && result.pixel_extent == Extent2D{8, 8} &&
                         result.coefficients[0].data() == sentinel && result.coefficients[0].size() == 1 &&
                         result.coefficients[0][0] == -123, "Forward launch failure changed its previous owner");
          result = {};
          Empty(*fixture.domain);
        }
        std::cout << "direct forward before=" << before << " kind=" << static_cast<int>(kind)
                  << " threads=" << threads << " passed\n" << std::flush;
      }
    }
  }
}

void StartupFailures() {
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({0, 1}, &domain));
  for (Kind kind : {Kind::kSystemError, Kind::kBadAlloc}) {
    for (size_t before : {0, 1, 3}) {
      {
        CpuExecutionScope cpu;
        Ok(cpu.Start(domain));
        WorkerLaunchFaultForTesting fault{Site::kBatchDriver, before, kind};
        std::unique_ptr<VarDctBatchEncoder> driver;
        Status status;
        {
          WorkerLaunchFaultScopeForTesting injection(&fault);
          status = VarDctBatchEncoder::Create(4, &driver);
        }
        Require(fault.triggered && fault.launched_in_group == before && status.code() == FailureCode(kind) &&
                !driver && HasCpuParticipation() && domain->snapshot().active_cpu_participants == 1,
                "Batch startup failure leaked workers or failed to resume its caller");
        Ok(VarDctBatchEncoder::Create(4, &driver));
        driver->Shutdown();
        Require(HasCpuParticipation(), "Recovered driver shutdown lost its caller's CPU slot");
      }
      Empty(*domain);
      std::cout << "startup before=" << before << " kind=" << static_cast<int>(kind) << " passed\n" << std::flush;
    }
  }
}
} // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string_view(argv[1]) == "--group") {
    const std::string_view group = argv[2];
    if (group == "startup") StartupFailures();
    else if (group == "forward") PreparedForwardFailures();
    else if (group == "cpu_color") SingleFailures(false, false, Site::kColorRows);
    else if (group == "cpu_aq") SingleFailures(false, false, Site::kInitialQuantization);
    else if (group == "cpu_orders") SingleFailures(false, false, Site::kCoefficientOrders);
    else if (group == "cpu_sections") SingleFailures(false, false, Site::kSerializerSections);
    else if (group == "metal_forward") SingleFailures(true, true, Site::kForwardTransforms);
    else if (group == "metal_orders") SingleFailures(true, false, Site::kCoefficientOrders);
    else if (group == "metal_sections") SingleFailures(true, false, Site::kSerializerSections);
    else if (group == "cpu_drains") {
      BatchFailureDuringShutdown(false, Site::kColorRows, Kind::kSystemError);
      BatchFailureDuringShutdown(false, Site::kInitialQuantization, Kind::kBadAlloc);
      BatchFailureDuringShutdown(false, Site::kCoefficientOrders, Kind::kSystemError);
    } else if (group == "metal_drains") {
      BatchFailureDuringShutdown(true, Site::kSerializerSections, Kind::kSystemError);
      BatchFailureDuringShutdown(true, Site::kForwardTransforms, Kind::kSystemError);
    } else Require(false, "Unknown worker-launch test group");
    std::cout << "Worker launch group " << group << " passed\n";
    return EXIT_SUCCESS;
  }
  const bool cpu_only = argc == 2 && std::string_view(argv[1]) == "--cpu-only";
  StartupFailures();
  PreparedForwardFailures();
  SingleFailures(false);
  BatchFailureDuringShutdown(false, Site::kColorRows, Kind::kSystemError);
  BatchFailureDuringShutdown(false, Site::kInitialQuantization, Kind::kBadAlloc);
  BatchFailureDuringShutdown(false, Site::kCoefficientOrders, Kind::kSystemError);
  if (!cpu_only) {
    SingleFailures(true);
    SingleFailures(true, true);
    BatchFailureDuringShutdown(true, Site::kSerializerSections, Kind::kSystemError);
    BatchFailureDuringShutdown(true, Site::kForwardTransforms, Kind::kSystemError);
  }
  std::cout << "Real worker launch failure checks passed\n";
}
