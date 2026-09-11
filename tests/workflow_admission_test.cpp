// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#include "codestream/batch_workflow_test.h"
#include "codestream/rate_control_internal.h"
#include "codestream/workflow_admission.h"
#include "codestream/workflow_internal.h"
#include "codestream/workflow_lifetime_test.h"
#include "core/image_buffer.h"
#include "gjxl/execution_domain.hpp"
#include "gjxl/gjxl.h"

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;
bool Check(bool good, const char *message) {
  if (!good)
    std::cerr << message << '\n';
  return good;
}
bool Ok(const Status &s) {
  if (!s.ok())
    std::cerr << s.message() << '\n';
  return s.ok();
}
bool Empty(const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  return Check(s.live_capacity_bytes == 0 && s.idle_capacity_bytes == 0 &&
                   s.reserved_unbacked_bytes == 0 && s.active_reservations == 0 &&
                   s.waiting_requests == 0,
               "Execution domain leaked capacity or admission");
}
template <class Predicate> bool Until(Predicate &&predicate) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= end)
      return false;
    std::this_thread::yield();
  }
  return true;
}
Image3FBuffer Image(Extent2D extent) {
  Image3FBuffer image(extent);
  for (size_t c = 0; c < 3; ++c)
    for (size_t i = 0; i < image.plane(c).size(); ++i)
      image.plane(c)[i] = 0.05f + 0.8f * ((i * (c + 3)) % 127) / 127.0f;
  return image;
}
VarDctEncodingOptions Options(size_t mode = 0) {
  VarDctEncodingOptions o;
  o.backend = mode == 0 || mode == 6 || mode == 7 ? VarDctBackendPreference::kCpu
                                                  : VarDctBackendPreference::kMetal;
  o.effort = 1;
  o.cpu_thread_count = 1;
  if (mode == 2 || mode == 8 || mode == 9)
    o.metal_aq_mode = GpuAdaptiveQuantizationMode::kExactCoefficients;
  if (mode == 3)
    o.metal_aq_mode = GpuAdaptiveQuantizationMode::kThroughput;
  if (mode == 4)
    o.metal_aq_mode = GpuAdaptiveQuantizationMode::kMaximumThroughput;
  if (mode == 5 || mode == 6) {
    o.rate_control_mode = VarDctRateControlMode::kMaximumError;
    o.maximum_error = {0.05f, 0.05f, 0.05f};
  }
  if (mode >= 7) {
    o.rate_control_mode = VarDctRateControlMode::kTargetBytes;
    o.target_bytes = 1;
    o.target_size_maximum_attempts = mode == 9 ? 64 : 4;
    o.target_size_tolerance = 0;
  }
  if (mode == 9)
    o.backend = VarDctBackendPreference::kAutomatic;
  return o;
}
bool Plan(const Image3FBuffer &image, const VarDctEncodingOptions &o, WorkflowStoragePlan *p) {
  return Ok(PlanWorkflowAdmission(
      image.extent(),
      {o, WorkflowStorageRoute::kCpu, WorkflowStorageAdapter::kBorrowedLinearRgb, true}, nullptr,
      false, true, p));
}
bool CheckSearchReachability() {
  const auto predicate = +[](float target) { return IsAutomaticMetalTargetEligible(target); };
  for (size_t attempts = 1; attempts <= 64; ++attempts) {
    bool visited = false;
    TargetSizeSearchResult result;
    if (!Ok(SearchTargetSize(
            {.target_bytes = 1, .maximum_attempts = attempts},
            [&](float target, std::vector<uint8_t> *bytes, VarDctEncodingSummary *summary) {
              visited |= predicate(target);
              *bytes = {1, 2, 3, 4};
              summary->encoded_bytes = bytes->size();
              summary->selected_butteraugli_target = target;
              return Status::Ok();
            },
            &result)) ||
        !Check(result.attempt_count == attempts &&
                   visited == TargetSizeSearchMayEvaluate(attempts, predicate),
               "Preflight search reachability disagrees with actual candidate ordering"))
      return false;
  }
  return true;
}
bool CheckSingle(bool cpu_only = false) {
  size_t cases = 0;
  for (size_t mode = 0; mode < 10; ++mode) {
    if (cpu_only && mode != 0 && mode != 6 && mode != 7) continue;
    const auto image = Image(mode == 9 ? Extent2D{128, 96} : Extent2D{33, 25});
    auto options = Options(mode);
    WorkflowStoragePlan plan;
    if (!Plan(image, options, &plan))
      return false;
    std::vector<uint8_t> oracle;
    VarDctEncodingSummary oracle_summary;
    VarDctEncodingTiming timing;
    if (!Ok(EncodeLinearRgbVarDctCodestreamProfiled(image.const_view(), options, &oracle,
                                                    &oracle_summary, &timing)) ||
        !Ok(TrimVarDctPreparationCache()))
      return false;
    std::shared_ptr<const ExecutionDomain> domain;
    if (!Ok(ExecutionDomain::Create({plan.working.peak_bytes}, &domain)))
      return false;
    options.execution_domain = domain;
    const auto default_before = ExecutionDomain::Default()->snapshot().peak_backing_bytes;
    for (size_t repeat = 0; repeat < 3; ++repeat) {
      std::vector<uint8_t> bytes{19};
      VarDctEncodingSummary summary;
      if (!Ok(EncodeLinearRgbVarDctCodestreamProfiled(image.const_view(), options, &bytes, &summary,
                                                      &timing)) ||
          !Check(bytes == oracle && summary == oracle_summary,
                 "Admitted workflow changed decisions or output"))
        return false;
      const auto s = domain->snapshot();
      if (!Check(s.active_reservations == 0 && s.waiting_requests == 0 &&
                     s.live_capacity_bytes == 0 &&
                     s.peak_committed_bytes <= plan.working.peak_bytes &&
                     ExecutionDomain::Default()->snapshot().peak_backing_bytes == default_before,
                 "Public workflow escaped its cap/domain or retained published output"))
        return false;
    }
    const size_t observed_peak = domain->snapshot().peak_backing_bytes;
    if (!Ok(WorkflowAdmission::TrimIdle(*domain)) || !Empty(*domain))
      return false;
    for (size_t underplan : {size_t{1}, observed_peak - 1}) {
      ArmNextWorkflowAdmissionCapacityForTest(underplan);
      std::vector<uint8_t> bytes{19};
      VarDctEncodingSummary summary;
      summary.score_history = {23};
      const auto old_summary = summary;
      timing.total_nanoseconds = 29;
      const auto status = EncodeLinearRgbVarDctCodestreamProfiled(image.const_view(), options,
                                                                  &bytes, &summary, &timing);
      DisarmWorkflowAdmissionCapacityForTest();
      if (!Check(status.resource_plan_exceeded() && bytes == std::vector<uint8_t>{19} &&
                     summary == old_summary && timing.total_nanoseconds == 29,
                 "Public planner violation was retried, escaped or published output") ||
          !Ok(WorkflowAdmission::TrimIdle(*domain)) || !Empty(*domain))
        return false;
      if (!Ok(EncodeLinearRgbVarDctCodestreamProfiled(image.const_view(), options, &bytes, &summary,
                                                      &timing)) ||
          !Check(bytes == oracle && summary == oracle_summary,
                 "Planner violation poisoned domain recovery") ||
          !Ok(WorkflowAdmission::TrimIdle(*domain)) || !Empty(*domain))
        return false;
    }
    // Hard-limit rejection precedes the first managed physical allocation.
    if (!Ok(ExecutionDomain::Create({1}, &domain)))
      return false;
    options.execution_domain = domain;
    std::vector<uint8_t> bytes{19};
    ArmNextManagedHostAllocationFailureForTest();
    const auto status = EncodeLinearRgbVarDctCodestream(image.const_view(), options, &bytes);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (!Check(status.code() == StatusCode::kOutOfMemory && !status.resource_plan_exceeded() &&
                   pending && bytes == std::vector<uint8_t>{19} &&
                   domain->snapshot().peak_committed_bytes == 0,
               "Too-large public request allocated work or changed output"))
      return false;
    options.effort = 0;
    if (!Check(EncodeLinearRgbVarDctCodestream(image.const_view(), options, &bytes).code() ==
                   StatusCode::kInvalidArgument,
               "Admission changed invalid-option precedence"))
      return false;
    ++cases;
  }
  return Check(cases == (cpu_only ? 3 : 10), "Missing public workflow policy coverage");
}

struct Gate {
  std::mutex mutex;
  std::condition_variable_any changed;
  bool released = false;
  std::atomic<bool> entered{false};
  std::stop_token stop;
  static void Observe(bool, bool, void *opaque) noexcept {
    auto &gate = *static_cast<Gate *>(opaque);
    std::unique_lock lock(gate.mutex);
    gate.entered = true;
    gate.changed.wait(lock, gate.stop, [&] { return gate.released; });
  }
  void Release() {
    {
      std::lock_guard lock(mutex);
      released = true;
    }
    changed.notify_all();
  }
};
bool CheckSharedFifo() {
  const auto small = Image({17, 9}), medium = Image({65, 65}), large = Image({257, 193});
  auto options = Options();
  WorkflowStoragePlan small_plan, medium_plan, large_plan;
  if (!Plan(small, options, &small_plan) || !Plan(medium, options, &medium_plan) ||
      !Plan(large, options, &large_plan) ||
      !Check(medium_plan.working.peak_bytes >= small_plan.working.peak_bytes &&
                 large_plan.working.peak_bytes > medium_plan.working.peak_bytes,
             "FIFO fixture has unordered work plans"))
    return false;
  // With medium active, small would fit but large cannot. Once large is
  // admitted, exactly one byte less than small's full envelope remains.
  const size_t limit = large_plan.working.peak_bytes + small_plan.working.peak_bytes - 1;
  std::shared_ptr<const ExecutionDomain> domain;
  if (!Ok(ExecutionDomain::Create({limit}, &domain)))
    return false;
  options.execution_domain = domain;
  Gate blocker, head;
  std::atomic<bool> good{true}, small_done{false};
  const auto encode = [&](const Image3FBuffer &image, Gate *gate, std::stop_token stop) {
    if (gate) {
      gate->stop = stop;
      (void)SetWorkflowLifetimeObserverForTest({Gate::Observe, gate});
    }
    std::vector<uint8_t> bytes;
    VarDctEncodingTiming timing;
    if (!EncodeLinearRgbVarDctCodestreamProfiled(image.const_view(), options, &bytes, nullptr,
                                                 &timing)
             .ok())
      good = false;
    (void)SetWorkflowLifetimeObserverForTest({});
  };
  std::jthread producer([&](std::stop_token stop) { encode(medium, &blocker, stop); });
  if (!Check(Until([&] { return blocker.entered.load(); }),
             "Blocking encode did not reach its gate"))
    return false;
  std::jthread first([&](std::stop_token stop) { encode(large, &head, stop); });
  if (!Check(Until([&] { return domain->snapshot().waiting_requests == 1; }),
             "Large public request did not queue")) {
    blocker.Release();
    return false;
  }
  std::jthread second([&](std::stop_token stop) {
    encode(small, nullptr, stop);
    small_done = true;
  });
  if (!Check(Until([&] { return domain->snapshot().waiting_requests == 2; }) && !small_done,
             "Small request bypassed FIFO admission")) {
    blocker.Release();
    head.Release();
    return false;
  }
  blocker.Release();
  if (!Check(Until([&] { return head.entered.load(); }) && !small_done,
             "Large admitted request did not retain its complete envelope")) {
    head.Release();
    return false;
  }
  head.Release();
  producer.join();
  first.join();
  second.join();
  return Check(good && small_done && domain->snapshot().peak_committed_bytes <= limit,
               "Concurrent calls exceeded their shared allowance") &&
         Empty(*domain);
}

struct BatchObservation {
  const ExecutionDomain *domain = nullptr;
  std::atomic<size_t> active{0}, peak{0}, started{0};
  bool publication = false;
  static void Work(void *opaque, size_t, bool entering) noexcept {
    auto &s = *static_cast<BatchObservation *>(opaque);
    if (!entering) {
      --s.active;
      return;
    }
    ++s.started;
    const auto n = ++s.active;
    auto old = s.peak.load();
    while (old < n && !s.peak.compare_exchange_weak(old, n)) {
    }
  }
  static void Publish(void *opaque, std::span<const VarDctBatchEncodingResult>,
                      std::span<const OwnedEncodingResult> owned) noexcept {
    auto &s = *static_cast<BatchObservation *>(opaque);
    size_t retained = 0;
    for (const auto &result : owned)
      retained += result.codestream.capacity();
    const auto snapshot = s.domain->snapshot();
    s.publication = retained != 0 && snapshot.live_capacity_bytes >= retained &&
                    snapshot.active_reservations == 1 && snapshot.idle_capacity_bytes == 0;
  }
};
bool CheckBatch() {
  std::array<Image3FBuffer, 4> images;
  std::array<VarDctBatchEncodingRequest, 5> requests;
  BatchWorkflowStorageAccumulator accumulator;
  for (size_t i = 0; i < images.size(); ++i) {
    images[i] = Image({33 + 8 * i, 25 + 8 * i});
    requests[i] = {images[i].const_view(), Options(i)};
    WorkflowStoragePlan p;
    if (!Plan(images[i], requests[i].options, &p) || !Ok(accumulator.AddRequest(&p)))
      return false;
  }
  requests.back().options = Options();
  if (!Ok(accumulator.AddRequest(nullptr)))
    return false;
  BatchWorkflowStoragePlan plan;
  if (!Ok(accumulator.Finish(4, 0, &plan)))
    return false;
  const auto minimum = plan.minimum_required_bytes;
  if (!Ok(accumulator.Finish(4, minimum, &plan)) ||
      !Check(plan.in_flight == 1 && plan.trim_after_each_image,
             "Batch fixture does not require tight retirement"))
    return false;
  std::shared_ptr<const ExecutionDomain> domain;
  if (!Ok(ExecutionDomain::Create({minimum}, &domain)))
    return false;
  for (auto &r : requests)
    r.options.execution_domain = domain;
  std::unique_ptr<VarDctBatchEncoder> driver;
  if (!Ok(VarDctBatchEncoder::Create(4, &driver)))
    return false;
  BatchObservation observer;
  observer.domain = domain.get();
  batch_execution_observer_for_testing = {&observer, BatchObservation::Work};
  batch_publication_observer_for_testing = {&observer, BatchObservation::Publish};
  std::vector<VarDctBatchEncodingResult> results(1);
  results[0].codestream = {19};
  const auto status = driver->Encode(requests, &results);
  batch_execution_observer_for_testing = {};
  batch_publication_observer_for_testing = {};
  if (!Ok(status) ||
      !Check(observer.peak == 1 && observer.active == 0 && observer.started == 4 &&
                 observer.publication && results.size() == requests.size() &&
                 results.back().status.code() == StatusCode::kInvalidArgument &&
                 domain->snapshot().peak_committed_bytes <= minimum,
             "Batch failed to enforce work slots, retirement or retained publication"))
    return false;
  if (!Empty(*domain))
    return false;
  // Metadata fits; the first worker's image does not. This is a terminal
  // whole-batch failure, not a published array of partial image results.
  results.resize(1);
  results[0].codestream = {19};
  observer.started = 0;
  observer.publication = false;
  batch_execution_observer_for_testing = {&observer, BatchObservation::Work};
  batch_publication_observer_for_testing = {&observer, BatchObservation::Publish};
  ArmNextWorkflowAdmissionCapacityForTest(plan.result_metadata.peak_bytes);
  const auto violated = driver->Encode(requests, &results);
  DisarmWorkflowAdmissionCapacityForTest();
  batch_execution_observer_for_testing = {};
  batch_publication_observer_for_testing = {};
  if (!Check(violated.resource_plan_exceeded() && observer.started == 1 && observer.active == 0 &&
                 !observer.publication && results.size() == 1 &&
                 results[0].codestream == std::vector<uint8_t>{19},
             "Worker underplan did not stop admission and preserve the whole result") ||
      !Empty(*domain))
    return false;
  if (!Ok(driver->Encode(requests, &results)))
    return false;
  for (size_t i = 0; i < images.size(); ++i) {
    auto o = requests[i].options;
    o.execution_domain.reset();
    std::vector<uint8_t> oracle;
    if (!Ok(EncodeLinearRgbVarDctCodestream(images[i].const_view(), o, &oracle)) ||
        !Ok(results[i].status) ||
        !Check(oracle == results[i].codestream, "Batch pressure changed encoding policy or bytes"))
      return false;
  }
  if (!Ok(TrimVarDctPreparationCache()))
    return false;
  if (!Ok(ExecutionDomain::Create({minimum - 1}, &domain)))
    return false;
  for (auto &r : requests)
    r.options.execution_domain = domain;
  results.resize(1);
  results[0].codestream = {19};
  ArmNextManagedHostAllocationFailureForTest();
  const auto rejected = driver->Encode(requests, &results);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  if (!Check(rejected.code() == StatusCode::kOutOfMemory && pending && results.size() == 1 &&
                 results[0].codestream == std::vector<uint8_t>{19} &&
                 domain->snapshot().peak_committed_bytes == 0,
             "Impossible batch allocated work or published results"))
    return false;
  requests[1].options.execution_domain.reset();
  return Check(driver->Encode(requests, &results).code() == StatusCode::kInvalidArgument,
               "Mixed-domain batch silently escaped an allowance");
}

struct CompletedCacheBatchObservation {
  const ExecutionDomain *domain = nullptr;
  size_t idle_bytes = 0;
  bool published = false;
  static void Publish(void *opaque, std::span<const VarDctBatchEncodingResult>,
                      std::span<const OwnedEncodingResult>) noexcept {
    auto &s = *static_cast<CompletedCacheBatchObservation *>(opaque);
    s.idle_bytes = s.domain->snapshot().idle_capacity_bytes;
    s.published = true;
  }
};

bool CheckBatchCompletedCache() {
  const std::array<Image3FBuffer, 3> images{
      Image({17, 9}), Image({257, 257}), Image({512, 512})};
  size_t batches = 0;
  for (size_t scenario = 0; scenario < 3; ++scenario) {
    std::array<VarDctBatchEncodingRequest, 4> requests;
    std::array<std::vector<uint8_t>, 4> expected;
    std::array<VarDctEncodingSummary, 4> summaries;
    BatchWorkflowStorageAccumulator accumulator;
    for (size_t i = 0; i < requests.size(); ++i) {
      // Same-size resident outputs, changing sizes, and mixed CPU/resident/
      // compatibility routes all share one batch reservation and backend.
      const bool cpu = scenario == 2 && i == 1;
      const bool compatibility = scenario == 2 && i == 2;
      const size_t image = cpu ? 2 : scenario == 1 && i % 2 == 0 ? 0 : 1;
      auto options = Options(cpu ? 0 : compatibility ? 2 : 1);
      options.effort = 7;
      requests[i] = {images[image].const_view(), options};
      WorkflowStoragePlan plan;
      if (!Plan(images[image], options, &plan) ||
          !Ok(accumulator.AddRequest(&plan)) ||
          !Ok(EncodeLinearRgbVarDctCodestream(requests[i].linear_rgb, options,
                                             &expected[i], &summaries[i])))
        return false;
    }
    if (!Ok(TrimVarDctPreparationCache())) return false;
    for (size_t workers : {size_t{1}, size_t{3}}) {
      BatchWorkflowStoragePlan full;
      if (!Ok(accumulator.Finish(workers, 0, &full))) return false;
      // Keep all pools, fall just below the cache-retention threshold, then
      // admit exactly one work slot and the retained results with no idle pool.
      for (size_t limit : {full.working.peak_bytes,
                           full.minimum_required_bytes + full.idle_pools.peak_bytes - 1,
                           full.minimum_required_bytes}) {
        BatchWorkflowStoragePlan plan;
        if (!Ok(accumulator.Finish(workers, limit, &plan))) return false;
        const bool keep = limit == full.working.peak_bytes;
        if (!Check(plan.trim_after_each_image == !keep &&
                       plan.in_flight == (keep ? workers : 1),
                   "Completed-cache fixture chose unexpected retirement"))
          return false;
        std::shared_ptr<const ExecutionDomain> domain;
        if (!Ok(ExecutionDomain::Create({limit, 3}, &domain))) return false;
        for (auto &request : requests) request.options.execution_domain = domain;
        std::unique_ptr<VarDctBatchEncoder> driver;
        if (!Ok(VarDctBatchEncoder::Create(workers, &driver))) return false;
        std::vector<VarDctBatchEncodingResult> results;
        for (size_t repeat = 0; repeat < 2; ++repeat) {
          CompletedCacheBatchObservation observation{domain.get()};
          batch_publication_observer_for_testing = {
              &observation, CompletedCacheBatchObservation::Publish};
          const auto status = driver->Encode(requests, &results);
          batch_publication_observer_for_testing = {};
          const auto snapshot = domain->snapshot();
          if (!Ok(status) ||
              !Check(observation.published && results.size() == requests.size() &&
                         observation.idle_bytes <= plan.idle_pools.peak_bytes &&
                         snapshot.peak_committed_bytes <= limit &&
                         snapshot.active_reservations == 0 && snapshot.waiting_requests == 0,
                     "Completed-frame cache exceeded its batch idle bound or admission")) {
            std::cerr << "scenario=" << scenario << " workers=" << workers
                      << " limit=" << limit << " planned_idle=" << plan.idle_pools.peak_bytes
                      << " actual_idle=" << observation.idle_bytes << '\n';
            return false;
          }
          for (size_t i = 0; i < requests.size(); ++i) {
            if (!Ok(results[i].status) ||
                !Check(results[i].codestream == expected[i] && results[i].summary == summaries[i],
                       "Cached batch changed a single-image result"))
              return false;
          }
          ++batches;
        }
        if (!Ok(WorkflowAdmission::TrimIdle(*domain)) || !Empty(*domain)) return false;
      }
    }
  }
  std::cout << "Completed-frame batch cache: " << batches
            << " batches with mixed sizes/routes, cache retention and tight retirement\n";
  return true;
}

bool CheckCHandles() {
  GJXLExecutionDomainOptions d{};
  if (!Check(gjxl_execution_domain_options_init(&d, sizeof(d)) == GJXL_OK,
             "C domain options initialization failed"))
    return false;
  d.managed_memory_bytes = 1;
  GJXLExecutionDomain *raw_domain = nullptr;
  if (!Check(gjxl_execution_domain_create(&d, &raw_domain) == GJXL_OK, "C domain creation failed"))
    return false;
  std::unique_ptr<GJXLExecutionDomain, decltype(&gjxl_execution_domain_destroy)> domain(
      raw_domain, gjxl_execution_domain_destroy);
  GJXLContextOptions c{};
  GJXLEncoderOptions e{};
  if (!Check(gjxl_context_options_init(&c, sizeof(c)) == GJXL_OK &&
                 gjxl_encoder_options_init(&e, sizeof(e)) == GJXL_OK,
             "C option setup failed"))
    return false;
  c.backend = GJXL_BACKEND_CPU;
  c.num_cpu_threads = 1;
  c.execution_domain = domain.get();
  e.effort = 1;
  GJXLContext *raw = nullptr;
  if (!Check(gjxl_context_create(&c, &raw) == GJXL_OK, "C domain context failed"))
    return false;
  std::unique_ptr<GJXLContext, decltype(&gjxl_context_destroy)> context(raw, gjxl_context_destroy);
  GJXLExecutionDomainSnapshot snapshot{};
  if (!Check(gjxl_execution_domain_snapshot(domain.get(), &snapshot, sizeof(snapshot)) == GJXL_OK &&
                 snapshot.active_reservations == 0,
             "C snapshot failed"))
    return false;
  domain.reset(); // Context must retain the same finite domain, not use default.
  const std::array<uint8_t, 17 * 9 * 3> packed{};
  const GJXLImageView image{
      sizeof(GJXLImageView), 17,    9, GJXL_PIXEL_FORMAT_RGB8_SRGB, packed.data(),
      packed.size(),         17 * 3};
  GJXLBuffer output{};
  ArmNextManagedHostAllocationFailureForTest();
  const auto result = gjxl_encode(context.get(), &image, &e, &output);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  gjxl_buffer_free(&output);
  return Check(result == GJXL_ERROR_OUT_OF_MEMORY && pending,
               "C context lost its domain or admitted conversion above the hard cap");
}

bool CheckMixedApiDomain(bool cpu_only = false) {
  for (size_t mode : {size_t{0}, size_t{1}}) {
    if (cpu_only && mode != 0) continue;
    const auto linear = Image({33, 25});
    auto options = Options(mode);
    WorkflowStoragePlan cpp_plan, c_plan;
    if (!Plan(linear, options, &cpp_plan) ||
        !Ok(PlanWorkflowAdmission(
            linear.extent(),
            {options, WorkflowStorageRoute::kCpu, WorkflowStorageAdapter::kPackedSrgbC}, nullptr,
            false, true, &c_plan)))
      return false;
    const size_t limit = std::max(cpp_plan.working.peak_bytes, c_plan.working.peak_bytes);
    std::shared_ptr<const ExecutionDomain> domain;
    if (!Ok(ExecutionDomain::Create({limit}, &domain)))
      return false;
    options.execution_domain = domain;
    GJXLExecutionDomain *raw_domain = nullptr;
    if (!Ok(CreateCExecutionDomain(domain, &raw_domain)))
      return false;
    std::unique_ptr<GJXLExecutionDomain, decltype(&gjxl_execution_domain_destroy)> c_domain(
        raw_domain, gjxl_execution_domain_destroy);
    if (!Check(RetainExecutionDomain(c_domain.get()) == domain &&
                   RetainExecutionDomain(nullptr) == ExecutionDomain::Default(),
               "C/C++ bridge copied limits instead of sharing the same domain"))
      return false;
    GJXLContextOptions co{};
    GJXLEncoderOptions eo{};
    if (!Check(gjxl_context_options_init(&co, sizeof(co)) == GJXL_OK &&
                   gjxl_encoder_options_init(&eo, sizeof(eo)) == GJXL_OK,
               "Mixed API options failed"))
      return false;
    co.backend = mode == 0 ? GJXL_BACKEND_CPU : GJXL_BACKEND_METAL;
    co.num_cpu_threads = 1;
    co.execution_domain = c_domain.get();
    eo.effort = 1;
    GJXLContext *raw_context = nullptr;
    if (!Check(gjxl_context_create(&co, &raw_context) == GJXL_OK, "Mixed API context failed"))
      return false;
    std::unique_ptr<GJXLContext, decltype(&gjxl_context_destroy)> context(raw_context,
                                                                          gjxl_context_destroy);
    c_domain.reset(); // C context retains the very same domain independently.
    std::array<uint8_t, 33 * 25 * 3> packed{};
    for (size_t i = 0; i < packed.size(); ++i)
      packed[i] = (i * 7) % 256;
    const GJXLImageView image{
        sizeof(GJXLImageView), 33,    25, GJXL_PIXEL_FORMAT_RGB8_SRGB, packed.data(),
        packed.size(),         33 * 3};
    struct COutput {
      GJXLBuffer bytes{};
      ~COutput() { gjxl_buffer_free(&bytes); }
    } c_output;
    Gate gate;
    std::atomic<bool> good{true}, c_done{false};
    const auto fallback_peak = ExecutionDomain::Default()->snapshot().peak_backing_bytes;
    std::jthread cpp([&](std::stop_token stop) {
      gate.stop = stop;
      (void)SetWorkflowLifetimeObserverForTest({Gate::Observe, &gate});
      std::vector<uint8_t> bytes;
      VarDctEncodingTiming timing;
      if (!EncodeLinearRgbVarDctCodestreamProfiled(linear.const_view(), options, &bytes, nullptr,
                                                   &timing)
               .ok())
        good = false;
      (void)SetWorkflowLifetimeObserverForTest({});
    });
    if (!Check(Until([&] { return gate.entered.load(); }), "Mixed API blocker did not enter"))
      return false;
    std::jthread c([&] {
      if (gjxl_encode(context.get(), &image, &eo, &c_output.bytes) != GJXL_OK)
        good = false;
      c_done = true;
    });
    if (!Check(Until([&] { return domain->snapshot().waiting_requests == 1; }) && !c_done &&
                   ExecutionDomain::Default()->snapshot().peak_backing_bytes == fallback_peak,
               "C conversion escaped its shared reservation or bypassed the C++ encode")) {
      gate.Release();
      return false;
    }
    gate.Release();
    cpp.join();
    c.join();
    if (!Check(good && c_done && c_output.bytes.size != 0 &&
                   domain->snapshot().peak_committed_bytes <= limit &&
                   ExecutionDomain::Default()->snapshot().peak_backing_bytes == fallback_peak,
               "Mixed C/C++ calls overspent or escaped their shared domain") ||
        !Ok(WorkflowAdmission::TrimIdle(*domain)) || !Empty(*domain))
      return false;
    // Translate a planner contract failure distinctly at the real C boundary.
    gjxl_buffer_free(&c_output.bytes);
    ArmNextWorkflowAdmissionCapacityForTest(1);
    const auto failed = gjxl_encode(context.get(), &image, &eo, &c_output.bytes);
    DisarmWorkflowAdmissionCapacityForTest();
    if (!Check(failed == GJXL_ERROR_RESOURCE_PLAN_EXCEEDED && c_output.bytes.data == nullptr &&
                   c_output.bytes.size == 0,
               "C API lost the typed planner violation") ||
        !Ok(WorkflowAdmission::TrimIdle(*domain)) || !Empty(*domain))
      return false;
  }
  return true;
}
} // namespace

int main(int argc, char** argv) {
  // Run the shared-domain test before any default-domain oracle. Starting at
  // zero makes the monotonic default peak sensitive even to transient escapes.
  if (!Check(ExecutionDomain::Default()->snapshot().peak_backing_bytes == 0,
      "Default domain was not pristine at test startup")) return EXIT_FAILURE;
  if (argc == 2 && std::string_view(argv[1]) == "--cpu-only")
    return CheckMixedApiDomain(true) && CheckSearchReachability() && CheckSingle(true) &&
      CheckSharedFifo() && CheckCHandles() ? EXIT_SUCCESS : EXIT_FAILURE;
  if (argc != 1) return EXIT_FAILURE;
  return CheckMixedApiDomain() && CheckSearchReachability() && CheckSingle() && CheckSharedFifo() && CheckBatch() &&
                 CheckBatchCompletedCache() && CheckCHandles()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
